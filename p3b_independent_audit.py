#!/usr/bin/env python3
"""Independent CSV/checkpoint-count audit and descriptive P3-B summaries.
No model fitting or test-time model selection. Merely checks frozen output and reports ALL cells.
Purely simulated passive civilian QPSK/OFDM.
"""
from pathlib import Path
import csv,json,hashlib,statistics,collections,math,platform
import torch
from p3b_time_link import ALL_CONDITIONS,PROFILES,N
from p3b_train_eval import SEEDS,SNRS,NEURAL,CLASSICAL,FRAMES,STEPS,BATCH
from p3b_receivers import DataResidualHead

BASE=Path(__file__).parent
R=BASE/'p3b_results'

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with open(p,newline='',encoding='utf-8') as f:return list(csv.DictReader(f))
def write(name,rows):
    with (R/name).open('w',newline='',encoding='utf-8') as f:
        if not rows:return
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def audit():
    raw=read(R/'ber_by_seed.csv'); summary=read(R/'ber_summary.csv')
    methods=set(NEURAL+CLASSICAL)
    assert len(raw)==5*5*3*9*len(methods)==9450,len(raw)
    assert len(summary)==5*3*9*len(methods)==1890,len(summary)
    a={}; prim={(r['profile'],int(r['snr_db']),r['condition'],r['method']):r for r in summary}
    for r in raw:
        key=(int(r['seed']),r['profile'],int(r['snr_db']),r['condition'],r['method'])
        assert key not in a,key
        assert key[0] in SEEDS and key[1] in PROFILES and key[2] in SNRS and key[3] in ALL_CONDITIONS and key[4] in methods
        assert (int(r['frames']),int(r['bits']))==(FRAMES,FRAMES*60)
        assert 0<=int(r['errors'])<=int(r['bits'])
        assert abs(float(r['ber'])-int(r['errors'])/int(r['bits']))<1e-14
        assert math.isfinite(float(r['scale_aligned_symbol_mse']))
        a[key]=r
    assert len(a)==len(raw)
    # Compare per-seed original CSV to concatenated results, not the same parser output.
    for seed in SEEDS:
        separate=read(R/f'eval_seed{seed}.csv')
        assert len(separate)==5*3*9*len(methods)
        for r in separate:
            k=(int(r['seed']),r['profile'],int(r['snr_db']),r['condition'],r['method'])
            assert k in a and r==a[k],k
    for k,r in prim.items():
        rs=[a[(s,*k)] for s in SEEDS]
        assert int(r['seeds'])==5 and int(r['bits'])==5*FRAMES*60
        assert int(r['errors'])==sum(int(v['errors']) for v in rs)
        assert abs(float(r['ber'])-int(r['errors'])/int(r['bits']))<1e-14
        assert abs(float(r['mean_seed_ber'])-statistics.mean(float(v['ber']) for v in rs))<1e-14
        for field in ('tx_power_ratio','burst_rate','adc_saturation','cpe_abs_mean'):
            if field in r and r[field]!='':assert math.isfinite(float(r[field]))
    # Check checkpoint references, state/metadata and exact SHA256.
    cks=[]
    prot=digest(BASE/'P3B_PROTOCOL.md')
    for seed in SEEDS:
        regs=read(R/f'training_seed{seed}.csv')
        assert len(regs)==6
        for z in regs:
            name=z['model'];assert name in NEURAL
            path=BASE/'p3b_checkpoints'/z['checkpoint']
            assert path.is_file() and z['sha256']==digest(path) and z['protocol_sha256']==prot
            ck=torch.load(path,map_location='cpu',weights_only=True)
            assert ck['key']==name and ck['seed']==seed and ck['steps']==STEPS and ck['batch']==BATCH
            assert ck['protocol_sha256']==prot
            net=DataResidualHead(name.split('__')[1]); net.load_state_dict(ck['state_dict'],strict=True)
            params=sum(x.numel() for x in net.parameters())
            assert params<5000 and params==int(z['params'])
            cks.append(dict(seed=seed,method=name,params=params,path=path.name,sha256=digest(path)))
    assert len(cks)==30
    write('checkpoints_sha256.csv',cks)
    # Strict paired delta by training seed, show no p values.
    paired=[]
    for p in PROFILES:
        for snr in SNRS:
            for cond in ALL_CONDITIONS:
                for seed in SEEDS:
                    frozen=float(a[(seed,p,snr,cond,'frozen_d')]['ber'])*100
                    for name in NEURAL:
                        ber=float(a[(seed,p,snr,cond,name)]['ber'])*100
                        paired.append(dict(seed=seed,profile=p,snr_db=snr,condition=cond,model=name,ber_pp=ber,
                            frozen_d_ber_pp=frozen,model_minus_frozen_pp=ber-frozen))
    write('paired_neural_vs_frozen.csv',paired)
    # Predeclared four mild conditions in equal4/long6 at 10 dB, fixed methods in full.
    primary=[]
    comparators=('ls','dft8','dft12','threshold','threshold_no_align','train_lmmse','dd_threshold')
    for p in ('equal4','long6'):
        for axis in ('pa','impulse','cpe','adc'):
            condition=axis+'_mild'
            vals={m:100*float(prim[(p,10,condition,m)]['ber']) for m in methods}
            classical=min(((vals[k],k) for k in comparators),key=lambda q:q[0])
            # List all six arms, no preselection after seeing outcome.
            for m in NEURAL:
                diffs=[100*(float(a[(s,p,10,condition,m)]['ber'])-float(a[(s,p,10,condition,'frozen_d')]['ber'])) for s in SEEDS]
                baseline_deltas=[100*(float(a[(s,p,10,condition,m)]['ber'])-float(a[(s,p,10,condition,classical[1])]['ber'])) for s in SEEDS]
                clean_frozen=100*float(prim[(p,10,'clean','frozen_d')]['ber'])
                primary.append(dict(profile=p,condition=condition,model=m,ber_pp=vals[m],
                    model_minus_frozen_pp=vals[m]-vals['frozen_d'],
                    best_classical_method=classical[1],best_classical_ber_pp=classical[0],
                    model_minus_classical_pp=vals[m]-classical[0],
                    frozen_d_clean_ber_pp=clean_frozen,
                    mean_seed_model_minus_frozen_pp=statistics.mean(diffs),
                    per_seed_model_minus_frozen_pp=';'.join(f'{x:.6f}' for x in diffs),
                    per_seed_model_minus_classical_pp=';'.join(f'{x:.6f}' for x in baseline_deltas)))
    write('primary_exploratory_10db.csv',primary)
    # Calibration checks from raw results; repeated per method, use fixed classical anchor.
    calibration=[]
    for snr in SNRS:
        for c in ALL_CONDITIONS:
            ref=prim[('train3',snr,c,'ls')]
            vals={f:float(ref[f]) for f in ['tx_power_ratio','adc_saturation','burst_rate','cpe_abs_mean']}
            calibration.append(dict(snr_db=snr,condition=c,**vals))
    write('impairment_calibration.csv',calibration)
    # Important qualitative checks, 10 dB train3.
    stats={(v['condition'],int(v['snr_db'])):v for v in calibration}
    assert stats[('pa_strong',10)]['tx_power_ratio']<stats[('pa_mild',10)]['tx_power_ratio']<1.0
    assert stats[('impulse_strong',10)]['burst_rate']>stats[('impulse_mild',10)]['burst_rate']
    assert stats[('cpe_strong',10)]['cpe_abs_mean']>stats[('cpe_mild',10)]['cpe_abs_mean']
    assert stats[('adc_strong',10)]['adc_saturation']>stats[('adc_mild',10)]['adc_saturation']
    assert stats[('clean',10)]['burst_rate']==0
    # Ensure positive-scale-aligned symbol error is not mislabeled as measured EVM.
    audit=dict(protocol_sha256=prot,raw_rows=len(raw),summary_rows=len(summary),raw_aggregate_counts_verified=len(summary),
        separate_per_seed_records_verified=len(raw),checkpoints_verified=len(cks),paired_records=len(paired),
        primary_exploratory_records=len(primary),privileged_diagnostic_rows=len(read(R/'privileged_reference_by_seed.csv')),
        conditions=list(ALL_CONDITIONS),methods=sorted(methods),snrs=list(SNRS),seeds=list(SEEDS),
        bits_per_row=FRAMES*60,bits_per_method_condition_profile_snr=5*FRAMES*60,
        test_regime='purely synthetic passive civil QPSK/OFDM',
        frozen_notice='P2-E/P3-A/P3-A2 untouched',
        scalar_mse_note='scale_aligned_symbol_mse is an uncalibrated logit decision-proxy, NOT a measured RF-EVM',
        limitations=['not independent laboratory replication','true-channel LMMSE privileged diagnostics added after primary evaluation; not used for training or baseline win','head-only timing excludes frozen backbone; full path timed independently'])
    (R/'independent_audit.json').write_text(json.dumps(audit,indent=2,ensure_ascii=False)+'\n')
    print(json.dumps({k:audit[k] for k in ('raw_rows','summary_rows','raw_aggregate_counts_verified','separate_per_seed_records_verified','checkpoints_verified','paired_records','primary_exploratory_records','privileged_diagnostic_rows')},indent=2))

if __name__=='__main__':audit()
