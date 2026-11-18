"""Replay archived one-dimensional development fits and independently audit signs and counts."""
import itertools
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from p4.classical_baselines import bussgang_lmmse, iterative_dd_pa_cancellation
from p4.em_floatformer import JointPhysicalLoss, create_floatformer
from p4.simulator import generate_frame


ROOT = Path(__file__).resolve().parent
PLAN = json.loads((ROOT/"p4cd_factorial_protocol.json").read_text(encoding="utf-8"))


def independent_contrasts(rows, column="ber"):
    """Use direct cell means, with no Yates code or matrix from the study implementation."""
    cube = {tuple(r[k] for k in ("gauge", "hermitian", "joint")):r for r in rows}
    assert set(cube) == set(itertools.product([False, True], repeat=3))
    terms = {}
    for degree in range(1,4):
        for indexes in itertools.combinations(range(3), degree):
            means = {}
            for level in (-1,1):
                values = [r[column] for flags, r in cube.items()
                          if (-1)**sum(not flags[i] for i in indexes) == level]
                assert len(values) == 4
                means[level] = sum(values)/4
            terms["".join("GHJ"[i] for i in indexes)] = means[-1] - means[1]
    terms["grand_mean_ber"] = sum(r[column] for r in rows)/8
    return terms


def verify_export(blob):
    lookup = {x["seed"]:x for x in blob["per_seed"]}
    measured = {}
    for seed in PLAN["dev_seeds"]:
        rows = [r for r in blob["rows"] if r["seed"]==seed]
        for r in rows:
            assert len(r["gains"])==3
            assert 0 <= r["bit_errors"] <= r["total_bits"]
            assert r["bit_errors"] == sum(c["bit_errors"] for c in r["gains"])
            assert r["total_bits"] == sum(c["total_bits"] for c in r["gains"])
            assert r["ber"] == r["bit_errors"]/r["total_bits"]
            for c in r["gains"]:
                assert type(c["bit_errors"]) is int and c["total_bits"]==96*32*4
        vals = independent_contrasts(rows)
        stats = lookup[seed]
        for term, name in zip("GHJ", ("F1_gauge", "F2_hermitian", "F3_joint")):
            assert abs(stats["main_effects"][name]-vals[term]) < 1e-12
        for pair, name in zip(("GH","GJ","HJ"), ("G:H","G:J","H:J")):
            assert abs(stats["interactions"][name]-vals[pair]) < 1e-12
        assert abs(stats["three_way"]-vals["GHJ"]) < 1e-12
        # Reconstruct all eight raw BERs from independently computed coefficients.
        for r in rows:
            levels = dict(zip("GHJ", [1 if r[k] else -1 for k in ("gauge","hermitian","joint")]))
            base = vals["grand_mean_ber"]
            for term, effect in vals.items():
                if term=="grand_mean_ber":
                    continue
                sign = 1
                for letter in term:
                    sign *= levels[letter]
                base -= .5*sign*effect
            assert abs(base-r["ber"]) < 1e-12
        measured[seed] = vals
    return measured


def build(seed, spec):
    gauge, hermitian, joint, width = spec
    def make(g,h,j):
        torch.manual_seed(seed)
        return create_floatformer(g,h,j,d_model=width,depth=1,n_heads=2)
    candidate = make(gauge,hermitian,joint)
    with torch.no_grad():
        for target in (make(True,hermitian,True),make(True,True,True)):
            lookup = target.state_dict()
            for name, p in candidate.named_parameters():
                if name in lookup and p.shape==lookup[name].shape:
                    p.copy_(lookup[name])
    return candidate


def fit_again(seed, spec):
    model = build(seed, spec)
    loss_fn = JointPhysicalLoss(lambda_nmse=1.,lambda_pa=.25)
    gen = torch.Generator().manual_seed(seed+1000)
    delta_names = [n for n,_ in model.named_parameters() if "gamma_eff" not in n]
    params = dict(model.named_parameters())
    opt = torch.optim.Adam([dict(params=[params[n] for n in delta_names],lr=.005),
                            dict(params=list(model.gamma_eff.parameters()),lr=.015)])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=32,eta_min=.0005)
    touched = set()
    for step in range(32):
        pa, amp = (("strong",1.25),("mild",2.25),("clean",float("inf")))[step%3]
        frame = generate_frame(12,n=32,profile="equal4",snr_db=(10.,30.),pa_condition=pa,generator=gen)
        opt.zero_grad()
        loss = loss_fn(model(frame.yp,frame.yd,frame.noise_var),frame,A=amp)
        assert torch.isfinite(loss)
        loss.backward()
        for name, p in model.named_parameters():
            assert p.grad is not None and torch.isfinite(p.grad).all(), (seed,spec,name)
            if torch.count_nonzero(p.grad):
                touched.add(name)
        torch.nn.utils.clip_grad_norm_(model.parameters(),2.)
        opt.step()
        scheduler.step()
    assert set(params)==touched, (seed,spec,set(params)-touched)
    return model.eval(), len(touched)


def measure(llr, truth):
    return {"bce_nats":float(F.binary_cross_entropy_with_logits(llr,truth.float())),
            "bit_errors":int(((llr>0)!=truth).sum()),"total_bits":truth.numel()}


def run():
    matched = json.loads((ROOT/"results/p4c_factorial.json").read_text())
    fixed = json.loads((ROOT/"results/p4c_fixed_width.json").read_text())
    result = {"development_seeds":PLAN["dev_seeds"],"matched":verify_export(matched),
              "fixed_width":verify_export(fixed),"replay":[],"unit_gain_classical":[]}
    for seed in PLAN["dev_seeds"]:
        sample = generate_frame(96,n=32,profile="equal4",snr_db=10.,pa_condition="strong",
                                generator=torch.Generator().manual_seed(seed+2000000))
        inputs = (sample.yp,sample.yd,sample.noise_var)
        with torch.no_grad():
            bus = bussgang_lmmse(*inputs,A=1.25).llrs
            _,dd,_,_ = iterative_dd_pa_cancellation(*inputs,A=1.25,iterations=2)
        result["unit_gain_classical"].extend(dict(seed=seed,model=name,**measure(llr,sample.bits))
                                             for name,llr in (("Bussgang-LMMSE",bus),("DD-PA",dd)))
        for a, spec in enumerate(PLAN["arms"],1):
            old = next(r for r in matched["rows"] if (r["seed"],r["arm_id"])==(seed,a))
            alt = next(r for r in fixed["rows"] if (r["seed"],r["arm_id"])==(seed,a))
            fixed_spec = list(spec[:3])+[8]
            assert sum(p.numel() for p in build(seed,fixed_spec).parameters())==alt["parameters"]
            model, touched = fit_again(seed,spec)
            assert sum(p.numel() for p in model.parameters())==old["parameters"]
            gauges = []
            with torch.no_grad():
                for r in old["gains"]:
                    gain = r["scale"]
                    metric = measure(model(sample.yp*gain,sample.yd*gain,sample.noise_var*(gain*gain)).llrs,sample.bits)
                    assert metric["bit_errors"]==r["bit_errors"] and metric["total_bits"]==r["total_bits"]
                    gauges.append(dict(scale=gain,**metric))
            result["replay"].append(dict(seed=seed,arm_id=a,finite_nonzero_gradient_parameter_tensors=touched,gains=gauges))
        print(f"Independent replay OK for eight arms: {seed}",flush=True)
    return result


if __name__=="__main__":
    torch.set_num_threads(1)
    result = run()
    out = ROOT/"results/p4c_independent_replay.json"
    out.with_suffix(".tmp").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n",encoding="utf-8")
    out.with_suffix(".tmp").replace(out)

