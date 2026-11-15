#!/usr/bin/env python3
"""Pure-data P1-E audit checks; requires completed p1e_results CSV/JSON."""
import csv
import json
import math
import unittest
from collections import defaultdict
from pathlib import Path

from p1e_replication import SEEDS, primary_results, exact_two_sided_signflip

ROOT=Path(__file__).parent/'p1e_results'

def rows(name):
    with (ROOT/name).open(newline='') as f:return list(csv.DictReader(f))

class IntegrityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.ber=rows('ber_by_seed.csv')
        cls.sum=rows('ber_summary.csv')
        cls.paired=rows('paired_frame_deltas.csv')

    def test_exact_primary_discrete_p(self):
        self.assertAlmostEqual(exact_two_sided_signflip([1.,2.,3.,4.,5.]),0.0625)
        self.assertAlmostEqual(exact_two_sided_signflip([1.,2.,3.,-4.,-2.]),1.)

    def test_frame_count_per_seed_scenario(self):
        self.assertEqual(len(self.ber),len(SEEDS)*3*3*9)
        self.assertEqual({int(r['bits']) for r in self.ber},{2048*64})
        self.assertEqual(len({r['seed'] for r in self.ber}),5)

    def test_ber_exact(self):
        for r in self.ber:
            self.assertEqual(int(r['errors'])<=int(r['bits']),True)
            self.assertAlmostEqual(float(r['ber']),int(r['errors'])/int(r['bits']),places=14)

    def test_pooled_counts(self):
        for r in self.sum:
            subset=[x for x in self.ber if x['profile']==r['profile'] and x['snr_db']==r['snr_db'] and x['method']==r['method']]
            self.assertEqual(len(subset),5)
            self.assertEqual(int(r['errors']),sum(int(x['errors']) for x in subset))
            self.assertEqual(int(r['bits']),sum(int(x['bits']) for x in subset))

    def test_primary_matches_bit_counts(self):
        p=json.loads((ROOT/'primary_endpoints.json').read_text())
        recalculated=primary_results([dict(seed=int(r['seed']),profile=r['profile'],snr_db=int(r['snr_db']),method=r['method'],ber=float(r['ber'])) for r in self.ber])
        for a,b in zip(p,recalculated):
            self.assertEqual(a['profile'],b['profile'])
            self.assertAlmostEqual(a['mean_delta_ber_pp'],b['mean_delta_ber_pp'],places=10)
            self.assertEqual(a['signflip_p_two_sided'],b['signflip_p_two_sided'])
            self.assertEqual(a['holm_adjusted_p'],b['holm_adjusted_p'])

    def test_frame_deltas_match_bit_counts(self):
        pairs=defaultdict(int)
        counts=defaultdict(int)
        for r in self.paired:
            key=(int(r['seed']),r['profile'],int(r['snr_db']))
            pairs[key]+=int(r['transformer_minus_cnn_bit_errors'])
            counts[key]+=1
        self.assertEqual(len(pairs),10)
        for (seed,profile,snr),diff in pairs.items():
            self.assertEqual(counts[seed,profile,snr],2048)
            lookup=lambda name: next(x for x in self.ber if int(x['seed'])==seed and x['profile']==profile and int(x['snr_db'])==snr and x['method']==name)
            trans=lookup('transformer__logamp_phase'); cnn=lookup('circular_cnn__logamp_phase')
            self.assertEqual(diff,int(trans['errors'])-int(cnn['errors']))

    def test_replay_was_exact(self):
        v=json.loads((ROOT/'replay_verification.json').read_text())
        self.assertEqual(v['status'],'PASS')
        self.assertEqual(v['exact_matching_cells'],81)

if __name__=='__main__':unittest.main()
