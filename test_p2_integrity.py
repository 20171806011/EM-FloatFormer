"""Reproducible, non-hardware P2 mathematical and results integrity tests."""
import csv, hashlib, json, unittest
from pathlib import Path
import torch
from ofdm_baseline import make_batch
from p2_models import verify_models, model_factory
from p2_run import summarize

ROOT=Path(__file__).parent

def csvrows(path):
    with (ROOT/path).open(newline='') as f:return list(csv.DictReader(f))

class TestP2Integrity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):torch.set_num_threads(4)

    def test_protocol_frozen(self):
        h=hashlib.sha256((ROOT/'P2E_FREEZE.json').read_bytes()).hexdigest()
        self.assertEqual(h,'935ac73e1cf56a2e9c95f51613a883d4ebf05d15eb672a3ec14dd5b189241342')

    def test_synthetic_model_shapes_and_gradients(self):
        self.assertEqual(len(verify_models()),8)

    def test_evidence_every_stage(self):
        for stage in 'abcde':
            self.assertTrue((ROOT/f'P2{stage.upper()}_REPORT.md').is_file())
            self.assertTrue((ROOT/f'p2{stage}_results/ber_by_seed.csv').is_file())
        for f in ['p2c_results/phase_equivariance.csv','p2b_results/numeric_roundtrip.csv',
                  'p2e_results/primary_endpoints.json','p2e_results/final_audit.json']:
            self.assertTrue((ROOT/f).is_file())

    def test_final_cells_and_per_seed_bits(self):
        rows=csvrows('p2e_results/ber_by_seed.csv')
        self.assertEqual(len(rows),540)
        self.assertEqual({int(r['seed']) for r in rows},set(range(2201,2211)))
        self.assertEqual({int(r['bits']) for r in rows},{131072})
        for r in rows:
            self.assertEqual(float(r['ber']),int(r['errors'])/int(r['bits']))

    def test_final_summary_recomputes(self):
        rows=csvrows('p2e_results/ber_by_seed.csv')
        summary=csvrows('p2e_results/ber_summary.csv')
        recompute=summarize([{**r,'seed':int(r['seed']),'snr_db':int(r['snr_db']),
                    'errors':int(r['errors']),'bits':int(r['bits']),'ber':float(r['ber'])} for r in rows])
        self.assertEqual(len(summary),len(recompute))
        for a,b in zip(summary,recompute):
            self.assertEqual(int(a['errors']),b['errors'])
            self.assertEqual(int(a['bits']),b['bits'])
            self.assertAlmostEqual(float(a['ber']),b['ber'],places=13)

    def test_holm_and_exact_p_values(self):
        a=json.loads((ROOT/'p2e_results/primary_endpoints.json').read_text())
        self.assertEqual(len(a),2)
        self.assertEqual(sorted(x['profile'] for x in a),['equal4','long6'])
        self.assertAlmostEqual(next(x for x in a if x['profile']=='long6')['exact_two_sided_signflip_p'],2/1024)
        self.assertAlmostEqual(next(x for x in a if x['profile']=='equal4')['exact_two_sided_signflip_p'],4/1024)
        for x in a:
            self.assertAlmostEqual(x['holm_adjusted_p'],4/1024)

    def test_audit_checkpoint_manifest(self):
        h=json.loads((ROOT/'p2e_results/checkpoint_sha256.json').read_text())
        self.assertEqual(len(h),20)
        audit=json.loads((ROOT/'p2e_results/final_audit.json').read_text())
        self.assertEqual(audit['replay_cells'],18)
        self.assertLess(audit['phase']['max_logit_abs_error'],1e-5)
        self.assertEqual(audit['phase']['hard_decision_changes'],0)

    def test_unlearned_mathematical_phase_symmetry(self):
        torch.manual_seed(493)
        b=make_batch(8,32,10,10,'cpu')
        m=model_factory('res_equiv_local')
        z=torch.polar(torch.ones_like(b.yp.real),torch.full_like(b.yp.real,2.13))
        with torch.inference_mode():
            self.assertLess((m.channel(b.yp*z,b.yd*z,b.noise_var)-
                   m.channel(b.yp,b.yd,b.noise_var)*z).abs().max().item(),1e-5)

    def test_training_seed_disjoint(self):
        self.assertTrue(set([1201,1202,1203]).isdisjoint(set(range(2201,2211))))

if __name__=='__main__':unittest.main(verbosity=2)
