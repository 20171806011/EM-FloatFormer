"""Independent recounts of link IDs, fold membership, rank vetoes and attainable action bound."""
import hashlib
import json
from collections import Counter
from itertools import combinations
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def lines(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()]


def ids(rows):
    keys = [(r['regime'],r['seed']) for r in rows]
    assert len(keys)==len(set(keys))
    return {s for _,s in keys}


def verify_records(rows, start):
    regimes = {'continuous','partial_shift','full_reset'}
    assert ids(rows) == set(range(start,start+128))
    assert len(rows)==3*128 and {r['regime'] for r in rows}==regimes
    for r in rows:
        assert tuple(f['fold'] for f in r['folds']) == (0,1)
        targets = [tuple(f['targets']) for f in r['folds']]
        assert len(targets[0])==11 and len(targets[1])==5
        assert len(set(sum(targets,()))) == 16
        assert set(sum(targets,())) == set(range(20))-{0,9,10,19}
        for f,aux in zip(r['folds'],[(2,5,8,13,16),(1,4,7,12,15)]):
            assert not set(f['targets']).intersection((*aux,0,9,10,19))
            for k in ['p9_errors','v1_errors']:
                assert len(f[k])==len(f['targets']) and set(f[k])<={0,1}
            assert f['used'] or f['p9_errors']==f['v1_errors']
        assert r['folds'][0]['score']==r['folds'][1]['score']


def audit():
    training = lines('train.jsonl')
    cal1 = lines('cal.jsonl')
    first = read('strong16.json')
    arm_records = read('learn_strong_results.json')
    second = read('strong_fresh.json')
    cal2 = read('cal_active.json')['records']
    split_ids = [ids(x) for x in (training, cal1, first['records'], cal2, second['records'])]
    assert all(len(s)==128 for s in split_ids)
    assert all(not (a & b) for a,b in combinations(split_ids,2))
    assert all(rec.get('regime','full_reset')=='full_reset' for rec in cal1+cal2)
    assert len(cal1)==len(cal2)==128
    weights = read('learn_weights.json')
    assert weights['train_sha256']==hashlib.sha256(Path('train.jsonl').read_bytes()).hexdigest()
    assert set(weights['train_seeds']) == split_ids[0]
    verify_records(first['records'],971120030000)
    verify_records(second['records'],971120050000)

    learned={(r['regime'],r['seed']):r for r in arm_records['links']}
    assert len(learned)==len(first['records'])
    scores=[r['score'] for r in cal1]
    counters=[Counter(),Counter()]
    oracle=Counter()
    for r in first['records']:
        for j,(f,g) in enumerate(zip(r['folds'],learned[r['regime'],r['seed']]['folds'])):
            old,fix=f['p9_errors'],f['v1_errors']
            accept = f['used'] and (1+sum(s>=f['score'] for s in scores))/(1+len(scores)) <= .05
            assert f['v2_used']==accept and f['v2_errors']==(fix if accept else old)
            for k,x in g['losses'].items():
                assert x == (fix if g['accepted'][k] else old)
                counters[j][k] += sum(x)
            # Choose fully informed action before applying it uniformly to this whole fold.
            m=min(sum(old),sum(fix))
            assert m==sum(g['losses']['utility_oracle'])
            oracle[r['regime']]+=m
    for j,key in enumerate(['fold_11','fold_5']):
        reported=arm_records['by_fold'][key]['equal_weight']['arms']
        assert all(k in reported and v==reported[k]['wrong'] for k,v in counters[j].items())
    sums=counters[0]+counters[1]
    rep=arm_records['by_fold']['all_16']['equal_weight']['arms']
    assert all(k in rep and v==rep[k]['wrong'] for k,v in sums.items())
    print(json.dumps(dict(seed_ranges=[(min(s),max(s)) for s in split_ids],
          equal_counts=dict(sums),oracle_by_regime=dict(oracle),oracle_by_fold=[c['utility_oracle'] for c in counters],
          oracle_advantage_over_v1_by_fold=[c['v1']-c['utility_oracle'] for c in counters]),indent=2))


if __name__=='__main__':
    audit()
