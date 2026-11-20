"""Primary 11-row diagnostic, before the subsequent active-fold calibration experiment."""
import json
from pathlib import Path
import numpy as np

def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]

def main():
    cal = np.array([r['score'] for r in read('cal.jsonl')])
    arms = ('p9','v1','positive_pilot_odds','rank_gate')
    records = read('eval.jsonl')
    summary = {}
    for reg in ('partial_shift','continuous','full_reset'):
        rows = [r for r in records if r['regime'] == reg]
        stats = {}
        for arm in arms:
            switches, count, probe_loss = [],0,0.
            for r in rows:
                p_rank = (1 + (cal >= r['score']).sum()) / (len(cal)+1)
                allow = {'p9':False, 'v1':True, 'positive_pilot_odds':r['score']>0.,
                         'rank_gate':p_rank<=.05}[arm]
                switches.append(allow and r['used'])
                key = 'v1' if switches[-1] else 'p9'
                count += sum(r[key+'_errors'])
                probe_loss += r[key+'_sse']
            stats[arm] = dict(wrong=count,blocks=11*len(rows),links_used=sum(map(int,switches)),
                              nmse=probe_loss/sum(r['energy'] for r in rows))
        summary[reg] = stats
    Path('native_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary))

if __name__=='__main__':
    main()
