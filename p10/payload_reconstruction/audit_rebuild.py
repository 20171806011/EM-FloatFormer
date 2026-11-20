"""Recount all paired outcomes in a held-out P9 waveform reconstruction experiment."""
import argparse
import json
from pathlib import Path
import numpy as np

SEED_START = 981120030000
SEEDS = list(range(SEED_START, SEED_START + 128))
REGIMES = ('full_reset', 'partial_shift', 'continuous')
ARMS = ('p9', 'v1', 'clear', 'pilot_only', 'rebuild', 'default', 'gate', 'code_gate')
AUX = ((2, 5, 8, 13, 16), (1, 4, 7, 12, 15))
CORE = tuple(r for r in range(20) if r not in (0, 9, 10, 19))
TARGETS = (tuple(r for r in CORE if r not in AUX[0]), AUX[0])
BOOT = np.random.default_rng(981120).integers(0, len(SEEDS), size=(10000, len(SEEDS)))


def paired_interval(numerator, denominator, scale=1.):
    """Clustered ratio-of-sums. Input is one count/energy per physical seed."""
    n = np.asarray(numerator, dtype=float)
    d = np.asarray(denominator, dtype=float)
    assert n.shape == d.shape == (len(SEEDS),)
    ns = n[BOOT].sum(axis=1)
    ds = d[BOOT].sum(axis=1)
    positive = ds > 0
    quantiles = np.quantile(scale * ns[positive] / ds[positive], [.025, .975]) if positive.any() else None
    return dict(value=float(scale * n.sum()/d.sum()) if d.sum() > 0 else None,
                ci95=quantiles.tolist() if quantiles is not None else None,
                valid_replicates=int(positive.sum()))


def check(raw):
    records = raw['records']
    by_key = {(r['regime'], r['seed']): r for r in records}
    assert len(by_key) == len(records) == len(SEEDS) * len(REGIMES)
    assert set(by_key) == {(reg, seed) for reg in REGIMES for seed in SEEDS}
    for (_, _), r in by_key.items():
        assert r['probe_energy'] > 0
        seen = []
        for f, fold in enumerate(r['folds']):
            assert f == fold['fold']
            assert tuple(fold['aux']) == AUX[f] and tuple(fold['targets']) == TARGETS[f]
            assert not set(fold['targets']) & set((*fold['aux'], 0, 9, 10, 19))
            seen.extend(fold['targets'])
            a, d = fold['action'], fold['incumbent']
            assert d in ('p9','v1') and a in ('rebuild',d)
            assert a == ('rebuild' if fold['scores']['rebuild'] > fold['scores']['default'] else d)
            assert 0 < fold['attention_pvalue'] <= 1
            assert d == ('v1' if fold['v1_used'] and fold['attention_pvalue'] <= .05 else 'p9')
            for k in ('rebuild','default'):
                assert len(fold['val_per_row'][k]) == 2
                np.testing.assert_allclose(sum(fold['val_per_row'][k]),fold['scores'][k])
            oracle_expected = r['regime']=='full_reset' and r['seed'] < SEED_START+32
            assert ('oracle_common_phase' in fold) == oracle_expected
            assert bool(fold['code_used']) == (fold['code_scores'][1] > fold['code_scores'][0])
            code_source = 'rebuild' if fold['code_used'] else d
            for key in ARMS:
                e = fold['errors'][key]
                assert len(e)==len(fold['targets']) and all(x in (0,1) for x in e)
                sse = fold['probe_sse'][key]
                assert np.isfinite(sse) and sse >= 0
            for output, source in (('gate',a), ('default',d), ('code_gate',code_source)):
                assert fold['errors'][output] == fold['errors'][source]
                np.testing.assert_allclose(fold['probe_sse'][output], fold['probe_sse'][source])
        assert len(seen)==len(set(seen))==16 and set(seen)==set(CORE)
    return by_key


def summarize_scope(by_key, regimes):
    # Each statistical group contains the same physical seed used for all selected mechanisms.
    errors = {arm: [] for arm in ARMS}
    energy = []
    sse = {arm: [] for arm in ARMS}
    gate_folds, gate_links, code_folds, code_links = 0, 0, 0, 0
    oracle2 = []
    oracle2plus = []
    for seed in SEEDS:
        seed_errors = {a: [] for a in ARMS}
        probe_sse = {a: 0. for a in ARMS}
        probe_energy = 0.
        oracle2_seed, oracle3_seed = 0, 0
        for reg in regimes:
            r = by_key[reg,seed]
            probe_energy += r['probe_energy']
            gate_links += int(any(f['action']=='rebuild' for f in r['folds']))
            code_links += int(any(f['code_used'] for f in r['folds']))
            for f in r['folds']:
                code_folds += int(f['code_used'])
                gate_folds += int(f['action']=='rebuild')
                # For the single hidden row on a link, use a target-count-weighted mean of fold SSE.
                weight = len(f['targets'])/16
                for a in ARMS:
                    seed_errors[a].extend(f['errors'][a])
                    probe_sse[a] += weight * f['probe_sse'][a]
                v = min(sum(f['errors'][a]) for a in ('p9','v1'))
                oracle2_seed += v
                oracle3_seed += min(v, sum(f['errors']['rebuild']))
        for a in ARMS:
            errors[a].append(seed_errors[a]); sse[a].append(probe_sse[a])
        energy.append(probe_energy)
        oracle2.append(oracle2_seed); oracle2plus.append(oracle3_seed)
    e = {a:np.asarray(v, dtype=bool) for a,v in errors.items()}
    common = e['p9'] & e['v1']
    all_right = ~e['p9'] & ~e['v1']
    triple = common & e['clear']
    n = np.full(len(SEEDS),16*len(regimes))
    common_n, triple_n = common.sum(1), triple.sum(1)
    def arm_metrics(a):
        rescued = (common & ~e[a]).sum(1)
        triple_rescued = (triple & ~e[a]).sum(1)
        return dict(wrong=int(e[a].sum()), bler=paired_interval(e[a].sum(1),n),
                    rescued_common=int(rescued.sum()), rate_rescued_common=paired_interval(rescued,common_n),
                    rescued_triple=int(triple_rescued.sum()),rate_rescued_triple=paired_interval(triple_rescued,triple_n),
                    wrong_if_both_old_right=int((all_right & e[a]).sum()),
                    wrong_if_either_old_right=int((~common & e[a]).sum()),
                    probe_nmse=paired_interval(sse[a],energy))
    def compare(a,b,subset):
        recovered=(subset & e[b] & ~e[a]).sum(1)
        lost=(subset & e[a] & ~e[b]).sum(1)
        return dict(rescued=int(recovered.sum()),lost=int(lost.sum()),
                    net_rescued=int((recovered-lost).sum()),
                    gain_pp=paired_interval(recovered-lost,subset.sum(1),100))
    contrasts = {}
    for a,b in [('rebuild','pilot_only'),('rebuild','clear'),('rebuild','p9'),
                ('rebuild','v1'),('rebuild','default'),('gate','default'),('gate','p9'),('code_gate','default'),('code_gate','gate')]:
        contrasts[a+'_over_'+b]=dict(all_blocks=compare(a,b,np.ones_like(common)),
                                    common_old_failures=compare(a,b,common))
    # On these particular outputs, the two-action block oracle can only err in the common-failure subset.
    return dict(physical_seed_groups=len(SEEDS), links=len(SEEDS)*len(regimes), blocks=int(n.sum()),
                common_old_failures=int(common_n.sum()),triple_failures=int(triple_n.sum()),
                both_old_right=int(all_right.sum()),eligible_any_old_right=int((~common).sum()),
                rebuild_used_folds=int(gate_folds), rebuild_used_links=int(gate_links),
                code_rebuild_used_folds=int(code_folds),code_rebuild_used_links=int(code_links),
                old_fold_oracle_wrong=sum(oracle2), with_rebuild_fold_oracle_wrong=sum(oracle2plus),
                arm_metrics={a:arm_metrics(a) for a in ARMS}, contrasts=contrasts)


def summarize_phase(by_key, regimes):
    results = {}
    for arm in ('clear','pilot_only','rebuild'):
        scores={}
        for m in ('rel_phase','rel_freq','abs_phase','abs_freq'):
            error,weight=[],[]
            for seed in SEEDS:
                folds=[f for reg in regimes for f in by_key[reg,seed]['folds']]
                error.append(sum(f['phase'][arm][m+'_sse'] for f in folds))
                weight.append(sum(f['phase'][arm][m+'_weight'] for f in folds))
            estimate=paired_interval(error,weight)
            scores[m+'_rmse']=dict(value=estimate['value']**.5,
                                   ci95=np.sqrt(estimate['ci95']).tolist())
        results[arm]=scores
    return results


def oracle_phase_subset(by_key):
    # Deliberately privileged common phase on the same prescribed 32 reset channels.
    metrics = {}
    for name in ('pilot_only','rebuild'):
        wrong = corrected = rescued = lost = common_old = common_rescued = 0
        for seed in SEEDS[:32]:
            for fold in by_key['full_reset',seed]['folds']:
                ini=np.asarray(fold['errors'][name],bool)
                final=np.asarray(fold['oracle_common_phase'][name],bool)
                assert final.shape==ini.shape
                common=np.array(fold['errors']['v1'],bool) & np.array(fold['errors']['p9'],bool)
                wrong+=int(ini.sum()); corrected+=int(final.sum())
                rescued+=int((ini & ~final).sum()); lost+=int((~ini & final).sum())
                common_old+=int(common.sum()); common_rescued+=int((common & ~final).sum())
        metrics[name] = dict(wrong=wrong, oracle_wrong=corrected,rescued=rescued,lost=lost,
                             common_old=common_old,oracle_rescued_common=common_rescued)
    return dict(regime='full_reset', nlinks=32,blocks=512,metrics=metrics)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',type=Path,default=Path('rebuild_raw.json'))
    p.add_argument('--out',type=Path,default=Path('rebuild_audit.json'))
    opts=p.parse_args()
    rec=check(json.loads(opts.input.read_text()))
    data={reg:summarize_scope(rec,[reg]) for reg in REGIMES}
    data['equal_weight']=summarize_scope(rec,REGIMES)
    for reg in REGIMES:
        data[reg]['phase']=summarize_phase(rec,[reg])
    data['equal_weight']['phase']=summarize_phase(rec,REGIMES)
    oracle_phase=oracle_phase_subset(rec)
    opts.out.write_text(json.dumps(dict(data,oracle_phase_subset=oracle_phase),ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    for reg,vals in data.items():
        print(reg, 'both-wrong',vals['common_old_failures'],'rebuild rescues',
              vals['arm_metrics']['rebuild']['rescued_common'],'selected rescues',
              vals['arm_metrics']['gate']['rescued_common'],'P9/old/clear/pilot/rebuild/frozen/gated',
              [vals['arm_metrics'][a]['wrong'] for a in ARMS])

if __name__=='__main__':
    main()
