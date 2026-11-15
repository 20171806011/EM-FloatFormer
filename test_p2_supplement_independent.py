"""Independent algebraic/metadata/data audit of supplementary receive-only OFDM work.

These tests are separate from the supplementary training/evaluation driver.
Original P2-E files are never written by any of these tests.
"""
import csv, hashlib, json, math, unittest
from collections import defaultdict
from pathlib import Path
import torch
import torch.nn.functional as F
from ofdm_baseline import make_batch, covariance_matrix, estimate_channel, decisions_from_channel
from p2_models import encode_complex, model_factory
import p2_supplement as s

ROOT=Path(__file__).parent
SROOT=ROOT/'p2_supplement'
P2E_HASHES={
 'P2E_REPORT.md':'36228f501e247edc8d97d633c63e5f7ca8a81ef1baaecd59604c4a9d5b37eec1',
 'P2E_FREEZE.json':'935ac73e1cf56a2e9c95f51613a883d4ebf05d15eb672a3ec14dd5b189241342',
 'P2_PROTOCOL.md':'04d54fac6008426487d030208721fb5a9f98c03d44f5b2039cd3e69f101cf087',
 'p2e_results/primary_endpoints.json':'a2f159d44a31a4613c83a1457c5941f705c03fbd10f9b39302f12e8bb0c1473d',
}

def read_csv(path):
    with path.open() as f:return list(csv.DictReader(f))

class SupplementIndependentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(4)
    def test_original_p2e_byte_integrity(self):
        for name,expected in P2E_HASHES.items():
            self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(),expected,name)
    def test_protocol_frozen_code_hash(self):
        self.assertEqual(s.PROTO_SHA,'1f7cc522d6f44a7a1102d68addab514a502fc31717a84a5a3d9d4a1019ae698a')
        self.assertEqual(s.SEEDS['S1'],(34001,34002,34003))
        self.assertEqual(s.SEEDS['S2'],(35001,35002,35003))
        self.assertEqual(s.SEEDS['S3'],(36001,36002,36003))
    def test_rms_ablation_equivalence_and_nontriviality(self):
        torch.manual_seed(420)
        z=torch.complex(torch.randn(16,32),torch.randn(16,32))
        self.assertTrue(torch.equal(s.norm_features(z,'dual_global'),encode_complex(z,'dual')))
        a=s.norm_features(z,'rms_global_only');b=s.norm_features(z,'dual_global')
        self.assertTrue(torch.equal(a[...,:2],b[...,:2]));self.assertEqual(int(torch.count_nonzero(a[...,2])),0)
        aa=s.norm_features(z,'rms_local_only');bb=s.norm_features(z,'dual_local')
        self.assertTrue(torch.equal(aa[...,:2],bb[...,:2]));self.assertEqual(int(torch.count_nonzero(aa[...,2])),0)
        self.assertGreater(float((aa[...,0]-a[...,0]).abs().mean()),0.01)
        self.assertTrue(torch.isfinite(bb).all())
    def test_sparse_no_hidden_pilot_leakage(self):
        torch.manual_seed(110)
        b=make_batch(6,32,10,10,'cpu',s.TRAIN)
        prior=covariance_matrix(32,s.TRAIN,'cpu')
        for count in s.S2_DENSITIES:
            obs,idx=s.sparse_obs(b.yp,count)
            corrupt=b.yp.clone()
            unseen=torch.ones(32,dtype=torch.bool);unseen[idx]=False
            corrupt[:,unseen]+=torch.complex(torch.full_like(corrupt[:,unseen].real,1e5),
                                              torch.full_like(corrupt[:,unseen].imag,-1e5))
            obs2,idx2=s.sparse_obs(corrupt,count)
            self.assertTrue(torch.equal(obs,obs2))
            self.assertTrue(torch.equal(idx,idx2))
            self.assertTrue(torch.equal(s.sparse_lmmse(obs,idx,b.noise_var,prior),
                                        s.sparse_lmmse(obs2,idx2,b.noise_var,prior)))
            interp=s.periodic_ls_interpolation(obs,idx)
            self.assertTrue(torch.allclose(interp[:,idx],obs,atol=1e-7,rtol=0))
    def test_sparse_lmmse_independent_delay_domain_posterior(self):
        torch.manual_seed(909)
        b=make_batch(7,32,3,13,'cpu',s.TRAIN)
        p=torch.tensor(s.TRAIN);p=p/p.sum()
        Fmat=torch.fft.fft(torch.eye(32,len(p),dtype=torch.complex64),dim=0)
        for count in s.S2_DENSITIES:
            yo,idx=s.sparse_obs(b.yp,count)
            c=covariance_matrix(32,s.TRAIN,'cpu')
            a=Fmat[idx]
            inner=(a*p[None,:])@a.conj().T
            noise=b.noise_var[:,:,None]*torch.eye(count,dtype=torch.complex64)[None]
            gain=torch.linalg.solve(inner[None]+noise,yo.unsqueeze(-1)).squeeze(-1)
            posterior_t=(gain@a.conj())*p
            expected=posterior_t@Fmat.T
            obtained=s.sparse_lmmse(yo,idx,b.noise_var,c)
            self.assertLess(float((obtained-expected).abs().max()),3e-6)
    def test_full_pilot_lmmse_matches_existing_implementation(self):
        torch.manual_seed(1209)
        b=make_batch(12,32,0,20,'cpu',s.TRAIN)
        idx=s.pilot_indices(32);c=covariance_matrix(32,s.TRAIN,'cpu')
        sparse=s.sparse_lmmse(b.yp,idx,b.noise_var,c)
        classic=estimate_channel(b.yp,b.noise_var,'lmmse',covariance=c)
        self.assertLess(float((sparse-classic).abs().max()),3e-6)
    def test_dft_ridge_normal_equation(self):
        torch.manual_seed(912)
        b=make_batch(4,32,5,10,'cpu',s.TRAIN)
        for count in s.S2_DENSITIES:
            obs,idx=s.sparse_obs(b.yp,count)
            h=s.ridge_dft8(obs,idx,b.noise_var)
            taps=torch.fft.ifft(h,dim=-1)[:,:8]
            A=s.dft_matrix(8)[idx]
            lhs=A.conj().T@A
            eq=taps@lhs.T+(b.noise_var+1e-5)*taps
            rhs=obs@A.conj()
            self.assertLess(float((eq-rhs).abs().max()),5e-5)
    def test_training_mixed_shares_bits_and_noise(self):
        x=s.make_mixed_batch(128,0,20,94007,'train3')
        y=s.make_mixed_batch(128,0,20,94007,'mixture')
        self.assertTrue(torch.equal(x.bits,y.bits))
        self.assertTrue(torch.equal(x.noise_var,y.noise_var))
        self.assertTrue(torch.allclose(x.yp-x.h,y.yp-y.h,atol=5e-7))
        self.assertTrue(torch.allclose(x.yd-x.h*torch.complex(
            2*x.bits[...,0].float()-1,2*x.bits[...,1].float()-1)/math.sqrt(2),
            y.yd-y.h*torch.complex(2*y.bits[...,0].float()-1,2*y.bits[...,1].float()-1)/math.sqrt(2),atol=5e-7))
    def test_mixture_empirical_prior_validation(self):
        for mode in s.S3_MODES:
            for seed in s.SEEDS['S3']:
                z=torch.load(SROOT/f's3_checkpoints/prior_{mode}_seed{seed}.pt',weights_only=True)
                self.assertEqual(z['samples'],s.STEPS*s.BATCH)
                self.assertEqual(z['protocol_sha256'],s.PROTO_SHA)
                c=z['covariance']
                self.assertLess(float((c-c.conj().T).abs().max()),1e-5)
                self.assertGreaterEqual(float(torch.linalg.eigvalsh(c).min()),-2e-5)
                if mode=='train3':expected=covariance_matrix(32,s.TRAIN,'cpu')
                else:expected=sum((covariance_matrix(32,pp,'cpu') for pp in s.MIX_PROFILES.values()))/4
                rel=float(torch.linalg.norm(c-expected)/torch.linalg.norm(expected))
                self.assertLess(rel,.10,f'{mode} {seed}: relative covariance error {rel}')
    def test_csv_integer_counts_and_summary_recompute(self):
        expected_per_seed={'s1':(3*3*6),'s2':(3*3*4*5),'s3':(len(s.TEST_PROFILES)*3*9)}
        for tag, per_seed in expected_per_seed.items():
            rows=read_csv(SROOT/f'{tag}_results/ber_by_seed.csv')
            groups=defaultdict(list)
            self.assertEqual(len(rows),3*per_seed)
            for r in rows:
                self.assertEqual(int(r['bits']),s.FRAMES*64)
                e=int(r['errors']);n=int(r['bits']);self.assertTrue(0<=e<=n)
                self.assertAlmostEqual(float(r['ber']),e/n,places=12)
                self.assertIn(int(r['seed']),s.SEEDS[tag.upper()])
                key=tuple((k,r[k]) for k in r if k not in ('seed','errors','bits','ber'))
                groups[key].append(r)
            sm=read_csv(SROOT/f'{tag}_results/ber_summary.csv')
            self.assertEqual(len(sm),len(groups))
            for z in sm:
                k=tuple((key,z[key]) for key in z if key not in ('seeds','errors','bits','ber','seed_sd'))
                self.assertIn(k,groups)
                rs=groups[k];err=sum(int(r['errors']) for r in rs);nb=sum(int(r['bits']) for r in rs)
                self.assertEqual(int(z['errors']),err)
                self.assertEqual(int(z['bits']),nb)
                self.assertAlmostEqual(float(z['ber']),err/nb,places=12)
    def test_saved_checkpoint_replay_s1(self):
        seed=34001;pi=1;snr=10
        recorded={(int(r['seed']),r['profile'],int(r['snr_db']),r['method']):int(r['errors'])
                  for r in read_csv(SROOT/'s1_results/ber_by_seed.csv')}
        models={}
        for name in s.ENCODINGS:
            obj=torch.load(SROOT/f's1_checkpoints/{name}_seed{seed}.pt',weights_only=True)
            m=s.LocalResidualNew(name);m.load_state_dict(obj['state_dict']);m.eval();models[name]=m
        err=dict.fromkeys(models,0)
        torch.manual_seed(140000000+seed*100+pi*10+snr)
        with torch.inference_mode():
            for off in range(0,s.FRAMES,128):
                b=make_batch(min(128,s.FRAMES-off),32,snr,snr,'cpu',s.MIX_PROFILES['equal4'])
                for n,m in models.items():err[n]+=s.bit_err(m(b.yp,b.yd,b.noise_var),b)
        for n in models:self.assertEqual(err[n],recorded[seed,'equal4',snr,n])
    def test_saved_checkpoint_replay_s2(self):
        seed=35001;pi=2;snr=10;count=8
        r={ (int(q['seed']),q['profile'],int(q['snr_db']),int(q['pilots']),q['method']):int(q['errors'])
           for q in read_csv(SROOT/'s2_results/ber_by_seed.csv') }
        obj=torch.load(SROOT/f's2_checkpoints/density{count}_seed{seed}.pt',weights_only=True)
        m=s.EquivComplexLocal();m.load_state_dict(obj['state_dict']);m.eval()
        prior=covariance_matrix(32,s.TRAIN,'cpu');e=0
        torch.manual_seed(150000000+seed*100+pi*10+snr)
        with torch.inference_mode():
            for off in range(0,s.FRAMES,128):
                b=make_batch(min(128,s.FRAMES-off),32,snr,snr,'cpu',s.MIX_PROFILES['long6'])
                obs,idx=s.sparse_obs(b.yp,count)
                base=s.sparse_lmmse(obs,idx,b.noise_var,prior)
                e+=s.bit_err(m(base,b.yd,b.noise_var),b)
        self.assertEqual(e,r[seed,'long6',snr,count,'res_equiv_lmmse_residual'])
    def test_saved_checkpoint_replay_s3(self):
        seed=36001;pi=5;snr=10
        r={(int(z['seed']),z['profile'],int(z['snr_db']),z['method']):int(z['errors'])
           for z in read_csv(SROOT/'s3_results/ber_by_seed.csv')}
        models={}
        for mode in s.S3_MODES:
            for n in s.S3_MODELS:
                obj=torch.load(SROOT/f's3_checkpoints/{n}_{mode}_seed{seed}.pt',weights_only=True)
                m=s.init_s3(n,seed);m.load_state_dict(obj['state_dict']);m.eval();models[mode+'__'+n]=m
        err=dict.fromkeys(models,0)
        torch.manual_seed(160000000+seed*100+pi*10+snr)
        with torch.inference_mode():
            for off in range(0,s.FRAMES,128):
                b=make_batch(min(128,s.FRAMES-off),32,snr,snr,'cpu',s.TEST_PROFILES['eight_tap'])
                for n,m in models.items():err[n]+=s.bit_err(m(b.yp,b.yd,b.noise_var),b)
        for n in models:self.assertEqual(err[n],r[seed,'eight_tap',snr,n])

if __name__=='__main__':unittest.main(verbosity=2)
