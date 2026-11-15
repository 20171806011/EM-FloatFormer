#!/usr/bin/env python3
"""Frozen P2 plan-alignment supplements, benign receive-only synthetic QPSK/OFDM.

Run only after reading the pre-execution P2_SUPPLEMENT_PROTOCOL.md. Nothing here
controls emitters, real RF hardware, surveillance systems or transmit channels.
Original P2E_FREEZE.json, P2E_REPORT.md and original checkpoints are immutable.
"""
from __future__ import annotations
import argparse, csv, copy, hashlib, json, math, os, platform, statistics, time
from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
import torch
import torch.nn.functional as F
from ofdm_baseline import make_batch, Batch, covariance_matrix, estimate_channel, decisions_from_channel
from p1c_architecture import TRAIN_PROFILE
from p2_models import LocalResidual, EquivComplexLocal, model_factory, encode_complex

N=32
TRAIN=tuple(TRAIN_PROFILE)
MIX_PROFILES={'train3': TRAIN, 'equal4':(.25,)*4, 'long6':(.40,.22,.15,.10,.08,.05),'two_tap':(.7,.3)}
TEST_PROFILES={**MIX_PROFILES,'late3':(.1,.2,.7),'eight_tap':(.34,.20,.13,.10,.08,.06,.05,.04)}
SEEDS={'S1':(34001,34002,34003),'S2':(35001,35002,35003),'S3':(36001,36002,36003)}
STEPS=400
BATCH=64
FRAMES=512
SNRS=(0,10,20)
ENCODINGS=('raw','log','dual_global','rms_global_only','rms_local_only','dual_local')
S2_DENSITIES=(32,16,8,4)
S3_MODES=('train3','mixture')
S3_MODELS=('res_equiv_local','p1_cnn_log')
PROTO_SHA=hashlib.sha256(Path(__file__).with_name('P2_SUPPLEMENT_PROTOCOL.md').read_bytes()).hexdigest()


def write_csv(path,rows):
    if not rows:return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w',newline='',encoding='utf8') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)


def record_config(out,branch,mode,extra=None):
    out.mkdir(parents=True,exist_ok=True)
    metadata=dict(phase=branch,task=mode,protocol_sha256=PROTO_SHA,train_seeds=SEEDS[branch],
      training_steps=STEPS,batch_size=BATCH,eval_frames=FRAMES,snrs=SNRS,
      python=platform.python_version(),torch=torch.__version__,cpu_threads=torch.get_num_threads(),
      notes='Only new exploratory research. Frozen P2-E unchanged. No transmission or military applications.')
    if extra:metadata.update(extra)
    (out/f'{mode}_config.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False))


def bit_err(pred,b):return int(torch.count_nonzero((pred>0)!=b.bits.bool()).item())


def standard_result(rows,branch):
    groups=defaultdict(list)
    for r in rows:groups[tuple((k,r[k]) for k in r if k not in ('seed','errors','bits','ber'))].append(r)
    summary=[]
    for k,rs in sorted(groups.items()):
        base={a:v for a,v in k};err=sum(int(x['errors']) for x in rs);bits=sum(int(x['bits']) for x in rs)
        summary.append(dict(**base,seeds=len(rs),errors=err,bits=bits,ber=err/bits,
                            seed_sd=statistics.stdev(x['ber'] for x in rs) if len(rs)>1 else 0.0))
    return summary


def norm_features(z,kind):
    if kind in ('raw','log'):
        return encode_complex(z,kind)
    if kind=='dual_global':
        return encode_complex(z,'dual')
    p=z.abs().square()
    if kind in ('rms_global_only',):
        s=(p.mean(dim=1,keepdim=True)+1.e-6).sqrt()
        third=torch.zeros_like(z.real)
    elif kind in ('rms_local_only','dual_local'):
        local=F.avg_pool1d(F.pad(p.unsqueeze(1),(2,2),mode='circular'),kernel_size=5,stride=1).squeeze(1)
        s=(local+1.e-6).sqrt()
        third=torch.zeros_like(z.real) if kind=='rms_local_only' else torch.log(s)
    else:raise ValueError(kind)
    return torch.stack((z.real/s,z.imag/s,third.expand_as(z.real)),dim=-1)


def extra_features(yp,yd,nv,kind):
    az=yd*yp.conj()/(yp.abs().square()+nv+1.e-2)
    angles=2*math.pi*torch.arange(yp.shape[-1],device=yp.device)/yp.shape[-1]
    aux=torch.stack((az.real,az.imag,angles.cos()[None,:].expand_as(yp.real),
                     angles.sin()[None,:].expand_as(yp.real),
                     nv.log10().expand_as(yp.real)),dim=-1)
    return torch.cat((norm_features(yp,kind),norm_features(yd,kind),aux),dim=-1)


class LocalResidualNew(LocalResidual):
    """Parameter-identical 11-channel P2B residual CNN; only input transform differs."""
    def __init__(self,kind):super().__init__(kind='raw');self.supp_kind=kind
    def channel(self,yp,yd,nv):
        x=self.stem(extra_features(yp,yd,nv,self.supp_kind).transpose(1,2))
        for conv,norm in zip(self.layers,self.norms):
            x=x+0.5*F.gelu(norm(conv(x).transpose(1,2)).transpose(1,2))
        d=self.proj(x).transpose(1,2)
        scale=(yp.abs().square().mean(dim=1,keepdim=True)+1.e-6).sqrt()
        return yp+torch.sigmoid(self.logit_gate)*scale*torch.complex(d[...,0],d[...,1])


def init_s1(seed):
    torch.manual_seed(21341000+seed)
    first=LocalResidualNew('raw');state=copy.deepcopy(first.state_dict())
    out={}
    for n in ENCODINGS:
        m=LocalResidualNew(n);m.load_state_dict(state,strict=True)
        out[n]=m
    return out


def train_s1(out):
    rows=[];out.mkdir(parents=True,exist_ok=True)
    for seed in SEEDS['S1']:
        models=init_s1(seed);opt={n:torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=.0001) for n,m in models.items()}
        torch.manual_seed(41000000+seed)
        start=time.perf_counter()
        for step in range(1,STEPS+1):
            b=make_batch(BATCH,N,0,20,'cpu',TRAIN)
            for n,m in models.items():
                m.train();opt[n].zero_grad(set_to_none=True)
                logits=m(b.yp,b.yd,b.noise_var)
                loss=F.binary_cross_entropy_with_logits(logits,b.bits.float())
                loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.)
                opt[n].step()
            if step in (1,STEPS):print('S1 train',seed,step,'loss',round(loss.item(),4),flush=True)
        for n,m in models.items():
            torch.save(dict(seed=seed,model=n,steps=STEPS,protocol_sha256=PROTO_SHA,
                            state_dict=m.state_dict()),out/f'{n}_seed{seed}.pt')
            rows.append(dict(seed=seed,model=n,steps=STEPS,params=sum(p.numel() for p in m.parameters()),
                             seconds=round(time.perf_counter()-start,2)))
    write_csv(out/'training.csv',rows);record_config(out,'S1','train',{'encodings':ENCODINGS})


def eval_s1(ckpt,out):
    rows=[]
    for seed in SEEDS['S1']:
        models={}
        for n in ENCODINGS:
            obj=torch.load(ckpt/f'{n}_seed{seed}.pt',map_location='cpu',weights_only=True)
            assert obj['protocol_sha256']==PROTO_SHA and obj['seed']==seed and obj['model']==n
            m=LocalResidualNew(n);m.load_state_dict(obj['state_dict']);m.eval();models[n]=m
        for pi,(p,prof) in enumerate([('train3',TRAIN),('equal4',MIX_PROFILES['equal4']),('long6',MIX_PROFILES['long6'])]):
            for snr in SNRS:
                torch.manual_seed(140000000+seed*100+pi*10+snr)
                err={n:0 for n in ENCODINGS};bits=0
                with torch.inference_mode():
                    for off in range(0,FRAMES,128):
                        b=make_batch(min(128,FRAMES-off),N,snr,snr,'cpu',prof)
                        for n,m in models.items():err[n]+=bit_err(m(b.yp,b.yd,b.noise_var),b)
                        bits+=b.bits.numel()
                for n in ENCODINGS:
                    rows.append(dict(seed=seed,profile=p,snr_db=snr,method=n,
                                     errors=err[n],bits=bits,ber=err[n]/bits))
    write_csv(out/'ber_by_seed.csv',rows)
    write_csv(out/'ber_summary.csv',standard_result(rows,'S1'))
    record_config(out,'S1','eval',{'test_rng_base':140000000})
    return rows


def pilot_indices(count):
    assert N%count==0
    return torch.arange(0,N,N//count,dtype=torch.long)


def sparse_obs(yp,count):
    """Return ONLY observed tones; no potentially hidden full-pilot tensor passed on."""
    idx=pilot_indices(count)
    return yp.index_select(-1,idx),idx


def periodic_ls_interpolation(yobs,idx):
    count=idx.numel();stride=N//count
    j=torch.arange(N,device=yobs.device)
    left=(j//stride)%count;right=(left+1)%count
    alpha=(j%stride).to(yobs.real.dtype)/stride
    return yobs[:,left]*(1-alpha[None,:])+yobs[:,right]*alpha[None,:]


def dft_matrix(taps):
    return torch.fft.fft(torch.eye(N,taps,dtype=torch.complex64),dim=0)


def ridge_dft8(yobs,idx,nv):
    A=dft_matrix(8)[idx,:]
    lhs=A.conj().T@A
    eye=torch.eye(8,dtype=torch.complex64)
    # A fixed physically motivated regularizer, not tuned on held-out outcomes.
    mat=lhs[None,:,:]+(nv.view(-1,1,1)+1e-5)*eye[None,:,:]
    rhs=yobs@A.conj()
    taps=torch.linalg.solve(mat,rhs.unsqueeze(-1)).squeeze(-1)
    return taps@dft_matrix(8).T


def sparse_lmmse(yobs,idx,nv,cov):
    cobs=cov.index_select(0,idx).index_select(1,idx)
    co=cov.index_select(1,idx)
    mat=cobs[None,:,:]+nv.view(-1,1,1)*torch.eye(len(idx),dtype=cov.dtype)
    w=torch.linalg.solve(mat,yobs.unsqueeze(-1)).squeeze(-1)
    return w@co.T


def init_s2(seed,count):
    torch.manual_seed(52300000+seed)  # same neural initial parameters for densities
    return EquivComplexLocal()


def train_s2(out):
    out.mkdir(parents=True,exist_ok=True)
    rows=[];prior=covariance_matrix(N,TRAIN,'cpu')
    for seed in SEEDS['S2']:
        models={count:init_s2(seed,count) for count in S2_DENSITIES}
        opts={count:torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=1e-4) for count,m in models.items()}
        torch.manual_seed(52000000+seed)
        t0=time.perf_counter()
        for step in range(1,STEPS+1):
            b=make_batch(BATCH,N,0,20,'cpu',TRAIN)
            for count,m in models.items():
                obs,idx=sparse_obs(b.yp,count)
                with torch.no_grad():h0=sparse_lmmse(obs,idx,b.noise_var,prior)
                m.train();opts[count].zero_grad(set_to_none=True)
                l=F.binary_cross_entropy_with_logits(m(h0,b.yd,b.noise_var),b.bits.float())
                l.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opts[count].step()
            if step in (1,STEPS):print('S2 train',seed,step,'loss',round(l.item(),4),flush=True)
        for count,m in models.items():
            torch.save(dict(seed=seed,density=count,steps=STEPS,protocol_sha256=PROTO_SHA,
                      state_dict=m.state_dict()),out/f'density{count}_seed{seed}.pt')
            rows.append(dict(seed=seed,pilots=count,steps=STEPS,
                        params=sum(p.numel() for p in m.parameters()),seconds=round(time.perf_counter()-t0,2)))
    write_csv(out/'training.csv',rows);record_config(out,'S2','train',{'pilot_counts':S2_DENSITIES})


def eval_s2(ckpt,out):
    rows=[];prior=covariance_matrix(N,TRAIN,'cpu')
    for seed in SEEDS['S2']:
        models={}
        for count in S2_DENSITIES:
            obj=torch.load(ckpt/f'density{count}_seed{seed}.pt',map_location='cpu',weights_only=True)
            assert obj['seed']==seed and obj['density']==count and obj['protocol_sha256']==PROTO_SHA
            m=init_s2(seed,count);m.load_state_dict(obj['state_dict']);m.eval();models[count]=m
        for pi,(p,prof) in enumerate([('train3',TRAIN),('equal4',MIX_PROFILES['equal4']),('long6',MIX_PROFILES['long6'])]):
            matched=covariance_matrix(N,prof,'cpu')
            for snr in SNRS:
                torch.manual_seed(150000000+seed*100+pi*10+snr)
                methods=('sparse_ls_interp','dft8_ridge','lmmse_train3','res_equiv_lmmse_residual','lmmse_matched_privileged')
                err={(count,n):0 for count in S2_DENSITIES for n in methods}
                bits=0
                with torch.inference_mode():
                    for off in range(0,FRAMES,128):
                        b=make_batch(min(128,FRAMES-off),N,snr,snr,'cpu',prof)
                        for count in S2_DENSITIES:
                            obs,idx=sparse_obs(b.yp,count)
                            base=sparse_lmmse(obs,idx,b.noise_var,prior)
                            hs={
                              'sparse_ls_interp':periodic_ls_interpolation(obs,idx),
                              'dft8_ridge':ridge_dft8(obs,idx,b.noise_var),
                              'lmmse_train3':base,
                              'lmmse_matched_privileged':sparse_lmmse(obs,idx,b.noise_var,matched)}
                            for n,h in hs.items():
                                pr=decisions_from_channel(b.yd,h,b.noise_var)
                                err[(count,n)]+=int((pr!=b.bits).sum())
                            err[(count,'res_equiv_lmmse_residual')]+=bit_err(models[count](base,b.yd,b.noise_var),b)
                        bits+=b.bits.numel()
                for count in S2_DENSITIES:
                    for n in methods:
                        e=err[count,n]
                        rows.append(dict(seed=seed,profile=p,snr_db=snr,pilots=count,method=n,
                                         errors=e,bits=bits,ber=e/bits))
        print('S2 eval',seed,'complete',flush=True)
    write_csv(out/'ber_by_seed.csv',rows);write_csv(out/'ber_summary.csv',standard_result(rows,'S2'))
    record_config(out,'S2','eval',{'test_rng_base':150000000,'observation_type':'masked pilot-only, no nonpilot leakage'})
    return rows


def make_mixed_batch(bs,snr_lo,snr_hi,seed,regime):
    """Paired bit/noise/random-tap draws; regime chooses channel tap power law only.
    A separate RNG generates the mixture index so train3 draws are comparable.
    """
    g=torch.Generator(device='cpu');g.manual_seed(seed)
    hr=torch.randn(bs,8,generator=g);hi=torch.randn(bs,8,generator=g)
    bits=torch.randint(0,2,(bs,N,2),generator=g)
    snr_db=torch.empty(bs,1).uniform_(snr_lo,snr_hi,generator=g)
    nv=10.0**(-snr_db/10)
    npilot=torch.complex(torch.randn(bs,N,generator=g),torch.randn(bs,N,generator=g))*(nv/2).sqrt()
    ndata=torch.complex(torch.randn(bs,N,generator=g),torch.randn(bs,N,generator=g))*(nv/2).sqrt()
    if regime=='train3':choice=torch.zeros(bs,dtype=torch.long)
    elif regime=='mixture':
        gprof=torch.Generator(device='cpu');gprof.manual_seed(seed+87463911)
        choice=torch.randint(0,4,(bs,),generator=gprof)
    else:raise ValueError(regime)
    profiles=list(MIX_PROFILES.values()); tap_weights=torch.zeros(bs,8)
    for j,p in enumerate(profiles):
        power=torch.tensor(p,dtype=torch.float32);power=power/power.sum()
        tap_weights[choice==j,:len(p)]=power[None,:]
    h_time=torch.complex(hr,hi)*(tap_weights/2).sqrt()
    h=torch.fft.fft(h_time,n=N,dim=-1)
    symbol=torch.complex(2*bits[...,0].float()-1,2*bits[...,1].float()-1)/math.sqrt(2)
    return Batch(h+npilot,h*symbol+ndata,h,bits,nv)


def empirical_covar(cross,total_noise,total_frames):
    c=cross/total_frames-(total_noise/total_frames)*torch.eye(N,dtype=torch.complex64)
    c=(c+c.conj().T)/2
    evals,evecs=torch.linalg.eigh(c)
    cc=(evecs*torch.clamp(evals,min=0)[None,:])@evecs.conj().T
    return (cc+cc.conj().T)/2


def init_s3(name,seed):
    # Same initialization for model+seed in train3 and mixture regimes.
    torch.manual_seed(61000000+seed+(0 if name=='res_equiv_local' else 91731))
    return model_factory(name)


def train_s3(out):
    out.mkdir(parents=True,exist_ok=True);rows=[]
    for seed in SEEDS['S3']:
        for mode in S3_MODES:
            models={n:init_s3(n,seed) for n in S3_MODELS}
            opts={n:torch.optim.AdamW(m.parameters(),lr=.002,weight_decay=1e-4) for n,m in models.items()}
            cross=torch.zeros((N,N),dtype=torch.complex64);nvsum=0.;num=0
            t0=time.perf_counter()
            for step in range(1,STEPS+1):
                b=make_mixed_batch(BATCH,0,20,70000000+seed*1000+step,mode)
                cross += b.yp.T @ b.yp.conj()
                nvsum+=float(b.noise_var.sum());num+=b.yp.shape[0]
                for n,m in models.items():
                    m.train();opts[n].zero_grad(set_to_none=True)
                    loss=F.binary_cross_entropy_with_logits(m(b.yp,b.yd,b.noise_var),b.bits.float())
                    loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.);opts[n].step()
                if step in (1,STEPS):print('S3 train',seed,mode,step,'loss',round(loss.item(),4),flush=True)
            cov=empirical_covar(cross,nvsum,num)
            assert torch.isfinite(cov).all()
            torch.save(dict(seed=seed,mode=mode,steps=STEPS,samples=num,
                            protocol_sha256=PROTO_SHA,covariance=cov),out/f'prior_{mode}_seed{seed}.pt')
            for n,m in models.items():
                torch.save(dict(seed=seed,mode=mode,model=n,steps=STEPS,
                     protocol_sha256=PROTO_SHA,state_dict=m.state_dict()),out/f'{n}_{mode}_seed{seed}.pt')
                rows.append(dict(seed=seed,regime=mode,model=n,steps=STEPS,
                                 samples=num,params=sum(p.numel() for p in m.parameters()),
                                 seconds=round(time.perf_counter()-t0,2)))
    write_csv(out/'training.csv',rows);record_config(out,'S3','train',{'modes':S3_MODES,'training_profiles':MIX_PROFILES})


def eval_s3(ckpt,out):
    rows=[]
    for seed in SEEDS['S3']:
        models={};priors={}
        for mode in S3_MODES:
            prior=torch.load(ckpt/f'prior_{mode}_seed{seed}.pt',map_location='cpu',weights_only=True)
            assert prior['protocol_sha256']==PROTO_SHA and prior['samples']==STEPS*BATCH
            priors[mode]=prior['covariance']
            for n in S3_MODELS:
                obj=torch.load(ckpt/f'{n}_{mode}_seed{seed}.pt',map_location='cpu',weights_only=True)
                assert obj['seed']==seed and obj['mode']==mode and obj['model']==n
                m=init_s3(n,seed);m.load_state_dict(obj['state_dict']);m.eval();models[mode,n]=m
        for pi,(p,prof) in enumerate(TEST_PROFILES.items()):
            matched=covariance_matrix(N,prof,'cpu')
            for snr in SNRS:
                torch.manual_seed(160000000+seed*100+pi*10+snr)
                err={f'{mode}__{n}':0 for mode in S3_MODES for n in S3_MODELS}
                err.update({'ls':0,'dft8':0,'train3_empirical_lmmse':0,
                            'mixture_empirical_lmmse':0,'matched_test_privileged':0})
                bits=0
                with torch.inference_mode():
                    for off in range(0,FRAMES,128):
                        b=make_batch(min(128,FRAMES-off),N,snr,snr,'cpu',prof)
                        for (mode,n),m in models.items():err[f'{mode}__{n}']+=bit_err(m(b.yp,b.yd,b.noise_var),b)
                        hs={'ls':b.yp,'dft8':estimate_channel(b.yp,b.noise_var,'dft',taps=8),
                            'train3_empirical_lmmse':estimate_channel(b.yp,b.noise_var,'lmmse',covariance=priors['train3']),
                            'mixture_empirical_lmmse':estimate_channel(b.yp,b.noise_var,'lmmse',covariance=priors['mixture']),
                            'matched_test_privileged':estimate_channel(b.yp,b.noise_var,'lmmse',covariance=matched)}
                        for n,h in hs.items():err[n]+=int((decisions_from_channel(b.yd,h,b.noise_var)!=b.bits).sum())
                        bits+=b.bits.numel()
                for n,e in err.items():rows.append(dict(seed=seed,profile=p,snr_db=snr,method=n,
                                                       errors=e,bits=bits,ber=e/bits))
        print('S3 eval',seed,'complete',flush=True)
    write_csv(out/'ber_by_seed.csv',rows);write_csv(out/'ber_summary.csv',standard_result(rows,'S3'))
    record_config(out,'S3','eval',{'test_rng_base':160000000,'training_profiles':MIX_PROFILES,'heldout':['late3','eight_tap']})
    return rows


def smoke():
    torch.manual_seed(438)
    b=make_batch(6,N,10,10,'cpu')
    old=encode_complex(b.yp,'dual')
    assert torch.allclose(old,norm_features(b.yp,'dual_global'),atol=0,rtol=0)
    assert torch.isfinite(norm_features(b.yp,'rms_local_only')).all()
    assert set(ENCODINGS)==set(init_s1(34001))
    for c in S2_DENSITIES:
        obs,idx=sparse_obs(b.yp,c)
        interp=periodic_ls_interpolation(obs,idx)
        assert torch.allclose(interp[:,idx],obs,atol=1e-7)
        ctrain=covariance_matrix(N,TRAIN,'cpu')
        h=sparse_lmmse(obs,idx,b.noise_var,ctrain)
        assert h.shape==b.h.shape and torch.isfinite(h).all()
        other=ridge_dft8(obs,idx,b.noise_var)
        assert other.shape==h.shape and torch.isfinite(other).all()
    for mode in S3_MODES:
        mix=make_mixed_batch(8,10,10,4455,mode)
        assert mix.yp.shape==(8,N) and torch.isfinite(mix.yp).all()
    print('S1/S2/S3 smoke passes',PROTO_SHA,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--phase',required=True,choices=['S1','S2','S3','smoke'])
    p.add_argument('--mode',choices=['train','eval','all'],default='all')
    p.add_argument('--root',default='p2_supplement')
    p.add_argument('--only-seed',type=int,default=None,help='run one frozen seed for reliable checkpointing')
    args=p.parse_args();torch.set_num_threads(min(4,os.cpu_count() or 4))
    if args.phase=='smoke':return smoke()
    if args.only_seed is not None:
        assert args.only_seed in SEEDS[args.phase], 'seed must be in frozen protocol'
        SEEDS[args.phase]=(args.only_seed,)
    root=Path(args.root);ck=root/f'{args.phase.lower()}_checkpoints';out=root/f'{args.phase.lower()}_results'
    train_func={'S1':train_s1,'S2':train_s2,'S3':train_s3}[args.phase]
    eval_func={'S1':eval_s1,'S2':eval_s2,'S3':eval_s3}[args.phase]
    if args.mode in ('train','all'):train_func(ck)
    if args.mode in ('eval','all'):eval_func(ck,out)
    print('COMPLETE',args.phase,args.mode,flush=True)

if __name__=='__main__':main()
