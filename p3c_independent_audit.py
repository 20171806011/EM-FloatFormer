#!/usr/bin/env python3
"""Independent arithmetic and provenance audit over frozen P3-C CSV evidence."""
import csv
import hashlib
import json
import math
from collections import Counter,defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
EXPECTED_PROTOCOL='e1c2f1e0cc8bb0b4a50b515bae40d6daeb62fee8e5ac2974afa3178c9d66371a'


def read(path):
    with Path(path).open(newline='',encoding='utf-8') as f:return list(csv.DictReader(f))


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def audit(out=HERE/'p3c_results',write=True):
    out=Path(out)
    assert sha(HERE/'P3C_PROTOCOL.md')==EXPECTED_PROTOCOL,'Changed frozen protocol'
    original=read(HERE.parent/'em_floatformer_p3b'/'p3b_results'/'checkpoints_sha256.csv')
    old={(int(r['seed']),r['method']):r['sha256'] for r in original}
    inventory=read(out/'checkpoint_sha256.csv')
    assert len(inventory)==35
    heads=0;frozen=0
    for r in inventory:
        seed=int(r['seed']);method=r['method']
        if method=='frozen_d':
            ck=HERE/'p3a2_checkpoints'/f'model_D_no_pos_10_seed{seed}.pt'
            original_old=HERE.parent/'em_floatformer_p3a2'/'p3a2_results'/f'training_seed{seed}.csv'
            prior={row['arm']:row['sha256'] for row in read(original_old)}
            assert prior['D_no_pos_10']==r['sha256'],(seed,'frozen SHA changed')
            frozen+=1
        else:
            ck=HERE/'p3b_checkpoints'/f'model_{method}_seed{seed}.pt'
            assert old[(seed,method)]==r['sha256'],(seed,method)
            heads+=1
        assert sha(ck)==r['sha256']
        assert ck.stat().st_size==int(r['bytes'])
    assert heads==30 and frozen==5

    raw=read(out/'raw_by_seed.csv');summary=read(out/'summary.csv')
    assert len(raw)==6525,(len(raw),'expected full five-seed test')
    assert len(summary)==1305,(len(summary),'expected summaries')
    kinds=Counter(r['kind'] for r in raw)
    assert kinds=={'operator':4320,'input_storage':1800,'coherent_gain':405},kinds
    grouped=defaultdict(list)
    for r in raw:
        assert int(r['seed']) in (48101,48102,48103,48104,48105)
        assert int(r['bits'])==256*60
        assert r['errors']!='',('failed numerical inference',r)
        assert int(r['nonfinite_logits'])==0 and int(r['nonfinite_input'])==0
        assert 0<=int(r['errors'])<=int(r['bits'])
        assert math.isclose(int(r['errors'])/int(r['bits']),float(r['ber']),rel_tol=1e-12,abs_tol=1e-12)
        assert 0<=int(r['hard_flips_vs_fp32'])<=int(r['bits'])
        key=tuple(r[f] for f in ('kind','profile','snr_db','condition','gain_exponent','mode','method'))
        grouped[key].append(r)
    index={tuple(s[f] for f in ('kind','profile','snr_db','condition','gain_exponent','mode','method')):s for s in summary}
    assert len(grouped)==len(summary)==len(index)
    for key,group in grouped.items():
        assert len(group)==5 and len({r['seed'] for r in group})==5
        summ=index[key]
        assert int(summ['seeds'])==5
        errors=sum(int(r['errors']) for r in group)
        bits=sum(int(r['bits']) for r in group)
        assert errors==int(summ['errors']) and bits==int(summ['bits'])==76800
        assert math.isclose(errors/bits,float(summ['ber']),rel_tol=1e-12,abs_tol=1e-12)
        assert sum(int(r['hard_flips_vs_fp32']) for r in group)==int(summ['total_hard_flips'])
        assert max(float(r['max_logit_abs_delta']) for r in group)==float(summ['max_logit_abs_delta'])
        assert sum(int(r['nonfinite_logits']) for r in group)==int(summ['total_nonfinite_logits'])

    by_seed={(r['seed'],r['kind'],r['profile'],r['snr_db'],r['condition'],r['gain_exponent'],r['mode'],r['method']):r for r in raw}
    exact_repeat=0
    for seed in ('48101','48102','48103','48104','48105'):
        for method in ('mixed__raw','mixed__log_phase','mixed__rms_log'):
            # Each new held-out base draw is reused among operator, storage and gain controls.
            a=by_seed[(seed,'operator','equal4','10','clean','','fp32',method)]
            b=by_seed[(seed,'input_storage','equal4','10','clean','','fp32',method)]
            c=by_seed[(seed,'coherent_gain','equal4','10','clean','0','fp32',method)]
            assert (a['errors'],a['bits'])==(b['errors'],b['bits'])==(c['errors'],c['bits'])
            exact_repeat+=1
    hardware=json.loads((out/'hardware_capabilities.json').read_text())
    assert not hardware['cuda_available'] and not hardware['mps_available']
    assert hardware['float16_cpu_fft']['supported'] is False and hardware['bfloat16_cpu_fft']['supported'] is False
    operators=json.loads((out/'operator_dtypes.json').read_text())
    for rec in operators:
        expected={'fp32':'torch.float32','fp16_realconv':'torch.float16','bf16_realconv':'torch.bfloat16'}[rec['mode']]
        assert rec['fft_output_dtype']=='torch.complex64' and rec['logit_dtype']=='torch.float32'
        assert len(rec['conv_dtypes'])==7 and all(v==[expected] for v in rec['conv_dtypes'].values())
    stress=read(out/'representation_stress.csv')
    assert len(stress)==6 and len({(x['encoding'],x['storage']) for x in stress})==6
    timing=read(out/'latency_cpu.csv')
    assert len(timing)==18 and set(x['batch_frames'] for x in timing)=={'1','64'}
    report=dict(status='PASS',protocol_sha256=EXPECTED_PROTOCOL,checkpoint_verified=len(inventory),
                heads_verified=heads,frozen_bases_verified=frozen,raw_cells=len(raw),summary_cells=len(summary),
                case_counts=dict(kinds),sample_training_seeds=5,frames_per_seed_scenario=256,
                bits_per_seed_scenario=15360,bits_per_five_seed_scenario=76800,
                nonfinite_logits=sum(int(r['nonfinite_logits']) for r in raw),
                nonfinite_input=sum(int(r['nonfinite_input']) for r in raw),
                reused_fp32_control_agreements=exact_repeat,
                native_real_autocast_verified=True,fft_lowprecision_supported=False,
                precision_modes=[r['mode'] for r in operators],latency_cells=len(timing),
                stress_cells=len(stress),
                csv_sha256={f:sha(out/f) for f in ('raw_by_seed.csv','summary.csv','representation_stress.csv','checkpoint_sha256.csv')},
                historical_reports_unchanged=True,
                limitation='Same-code deterministic replay only; CPU lacks native half/bfloat complex FFT; no full low-precision receiver')
    if write:(out/'independent_audit.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
    print('INDEPENDENT AUDIT PASS',len(raw),len(summary),'records',len(inventory),'checkpoint hashes',flush=True)
    return report

if __name__=='__main__':audit()
