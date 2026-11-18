"""A paired S05 training study and layer-by-layer audit of SparseUnrolledEM."""
import argparse
from contextlib import nullcontext
import hashlib
import inspect
import json
import math
from pathlib import Path

import torch
from p4 import slot_2d
from p4.classical_baselines import calibrated_16qam_llrs
from p4.next_pilots import joint_soft_channel, sparse_slot
from p4.next_sparse_neural import cross_disabled, summarize
from p4.stage3_sparse import RESULTS, SparseUnrolledEM, FIRST_FIT
from p4.slot_2d import JointPhysicalLoss2D, SlotSimulator2D

SEEDS = [FIRST_FIT,73652,73653]
DRAWS = list(range(887201,887217))
STEPS = 120
FREEZE = RESULTS/"stage3_sparse_protocol.json"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol():
    folder = Path(__file__).resolve().parent
    # S05 uses these receiver/simulator primitives. Workload dispatchers are separate callers.
    primitives = ("SlotData","SlotSimulator2D","_draw_slot_taps","_apply_continuous_slot_channel",
                  "_resolve_pa_params","get_dmrs_mask","get_slot_masks","_gold_sequence_bits",
                  "generate_dmrs","BidirectionalBlock2D","ComplexRoPE2D","GreenAttention2D",
                  "JointPhysicalLoss2D","lmmse_2d_interpolate","compute_2d_correlation","_solve_2d_lmmse_core")
    return {"stage":"S05 development, physical channel iteration with bidirectional pilot/grid attention",
            "training_seeds":SEEDS,"evaluation_seeds":DRAWS,"original_gate_seed":42,"comparable_old_test_seeds":"not used for training",
            "slot":"14x32, CP16, one eight-tone pilot comb at symbol 2, clean PA, CFO=0, fd=.01, train3, SNR20 dB; iid uncoded Gray 16-QAM",
            "known_receiver_information":"Observed y, known pilot values and mask, receiver noise variance, design covariance from the fixed three-tap profile and nominal Doppler. Sample taps, symbol labels, actual tap rotations and transmitted payload withheld during inference.",
            "training":{"steps":STEPS,"batch_size":8,"train_rng_offset":3000000,"lr":.002,"eta_min":.0002,"clip":2.,
                "optimizer":"Adam cosine, final weights only","objective":"Signed JointPhysicalLoss2D, BCE + NMSE + 0.25 forward PA consistency (clean PA); target channel/data only in training loss."},
            "architecture":"pilot-RMS normalized pilot MAP followed by six exact soft-symbol/weighted least-squares updates; two width-12 two-head canonical bidirectional EM blocks read the channel, y, posterior innovation and symbol variance after updates 3 and 6. Their bounded complex corrections are projected onto delay bins 0..2. Recurrent pilot/grid states carry between the two blocks. Shared posterior logits use the P4-A .035 floor and 20*tanh bound.",
            "arms":"full and cross-disconnected models retrained with paired initialization, optimizer and training frames; full weights evaluated with both cross-attention outputs suppressed; matched zero-correction six-update solver; twelve updates for a computational-depth control; legacy unnormalized MAP6; exact nearest LS.",
            "gate":"At least 4 dB improvement in pooled mean per-slot NMSE vs nearest LS; report each training seed on 16x8 fresh slots and the untouched historical 8-slot seed-42 draw. Default user-facing checkpoint fixed as first seed. Evaluate any gains over physics explicitly; crossing the 4 dB gate does not assign gains to attention.",
            "primary_contrasts":"Full vs six-update zero correction, frozen cross-off, retrained cross-off and twelve-update physics. Mean over fits sharing the same slots, descriptive paired intervals conditional on fits. Layer NMSE plus normalized data reconstruction residual. Hard bit counts also reported.",
            "source_sha256":{p:sha(folder/p) for p in ("stage3_sparse.py","stage3_sparse_study.py","next_pilots.py",
                "next_sparse_neural.py","simulator.py","classical_baselines.py","em_floatformer.py")},
            "slot_primitives_sha256":{n:hashlib.sha256(inspect.getsource(getattr(slot_2d,n)).encode()).hexdigest() for n in primitives},
            "train3_pdp":list(slot_2d.SLOT_PROFILES["train3"]), "torch":str(torch.__version__)}


def train(seed, arm):
    ckpt = RESULTS/f"stage3_sparse_{arm}_{seed}.pt"
    model = SparseUnrolledEM()
    if ckpt.exists():
        saved = torch.load(ckpt,weights_only=True)
        if (saved["freeze"],saved["seed"],saved["arm"])!=(sha(FREEZE),seed,arm):
            raise RuntimeError("Checkpoint identity mismatch.")
        model.load_state_dict(saved["weights"])
        return model.eval(), saved["history"]
    # Reset before constructing the paired initial state, independently of caller RNG.
    torch.manual_seed(seed)
    model = SparseUnrolledEM().train()
    g = torch.Generator().manual_seed(3000000+seed)
    sim = SlotSimulator2D(n_symb=14,n_sc=32,cp=16,profile="train3",fd=.01,pilot_symbols=(2,),comb=4)
    optimizer = torch.optim.Adam(model.parameters(),lr=.002)
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max=STEPS,eta_min=.0002)
    objective = JointPhysicalLoss2D()
    active = set()
    losses = []
    with (cross_disabled(model) if arm=="cut" else nullcontext()):
        for step in range(STEPS):
            slot = sim.generate_slot(batch_size=8,snr_db=20.,pa_condition="clean",generator=g)
            optimizer.zero_grad()
            out = model(slot.y_grid,slot.pilot_tx,slot.pilot_mask,slot.noise_var)
            loss = objective(out,slot)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss")
            loss.backward()
            for name,p in model.named_parameters():
                if name not in active and p.grad is not None and bool(p.grad.abs().max()>0):
                    active.add(name)
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.,error_if_nonfinite=True)
            optimizer.step()
            cosine.step()
            losses.append(float(loss.detach()))
    names = dict(model.named_parameters())
    if arm=="full" and len(active)!=len(names):
        raise RuntimeError(f"Disconnected trainable parameters: {set(names)-active}")
    history = {"steps":len(losses),"loss_first_10":sum(losses[:10])/10,"loss_last_10":sum(losses[-10:])/10,
        "active_tensors":len(active),"parameter_tensors":len(names),
        "active_parameters":sum(names[n].numel() for n in active),"allocated_parameters":sum(p.numel() for p in names.values())}
    tmp = ckpt.with_suffix(".tmp")
    torch.save({"weights":model.state_dict(),"seed":seed,"arm":arm,"history":history,"freeze":sha(FREEZE)},tmp)
    tmp.replace(ckpt)
    print(f"trained {seed}/{arm}: {history}",flush=True)
    return model.eval(),history


def score(h,slot,llr=None):
    if not torch.isfinite(h).all():
        raise FloatingPointError("Nonfinite channel estimate")
    h = h.expand_as(slot.y_grid)
    nmse = (h-slot.h_true).abs().square().sum((1,2))/slot.h_true.abs().square().sum((1,2))
    if llr is None:
        hd = h[:,slot.data_mask].to(torch.complex128)
        power = hd.abs().square().clamp_min(1e-20)
        z = slot.y_grid[:,slot.data_mask].to(torch.complex128)*hd.conj()/power
        llr = calibrated_16qam_llrs(z,slot.noise_var/power+.035)
    else:
        llr = llr[:,slot.data_mask]
    if not torch.isfinite(llr).all():
        raise FloatingPointError("Nonfinite soft demapping")
    return {"nmse_per_slot":nmse.tolist(),
            "bit_errors_per_slot":((llr>0)!=slot.bits).sum((1,2)).tolist()}


def stage_errors(model,stages,slot):
    data = slot.y_grid[:,slot.data_mask].to(torch.complex128)
    symbols = model.symbols
    out = {}
    for key,h in stages.items():
        hd = h[:,slot.data_mask].to(torch.complex128)
        d = data.unsqueeze(-1)-hd.unsqueeze(-1)*symbols
        prob = torch.softmax(-d.abs().square()/slot.noise_var.unsqueeze(-1),dim=-1)
        innovation = data-hd*(prob*symbols).sum(-1)
        residual = innovation.abs().square().sum(-1)/data.abs().square().sum(-1)
        nmse = (h-slot.h_true).abs().square().sum((1,2))/slot.h_true.abs().square().sum((1,2))
        out[key] = {"sum_slot_nmse":float(nmse.sum()),"sum_slot_normalized_payload_innovation":float(residual.sum())}
    return out


@torch.inference_mode()
def evaluate(models,draw):
    slot = sparse_slot(draw,4,2,20.)
    tones = slot.pilot_mask[2].nonzero().flatten()
    idx = (torch.arange(32)[:,None]-tones).abs().argmin(-1)
    nearest = slot.yp_sparse[:,idx][:,None]
    methods = {"nearest_ls":score(nearest,slot)}
    args = (slot.y_grid,slot.pilot_tx,slot.pilot_mask,slot.noise_var)
    methods["legacy_map6"] = score(joint_soft_channel(*args,"train3",.01,16,iterations=6),slot)
    traces = {}
    reference = next(iter(models.values()))
    for n in (6,12):
        out,stages = reference(*args,learned=False,iterations=n,trace=True)
        methods[f"physics{n}"] = score(out.h_hat,slot,out.llrs)
        traces[f"physics{n}"] = stage_errors(reference, stages,slot)
    for (seed,arm),model in models.items():
        modes = ("full","frozen_cut") if arm=="full" else ("retrained_cut",)
        for mode in modes:
            key = f"{seed}/{mode}"
            with (nullcontext() if mode=="full" else cross_disabled(model)):
                out,stages = model(*args,trace=True)
            methods[key] = score(out.h_hat,slot,out.llrs)
            traces[key] = stage_errors(model,stages,slot)
    return {"seed":draw,"slots":len(slot.y_grid),"bits_per_slot":slot.bits[0].numel(),"methods":methods,"stages":traces}


def run():
    if json.loads(FREEZE.read_bytes())!=protocol():
        raise RuntimeError("S05 source or settings differ from the sealed development recipe.")
    torch.set_num_threads(2)
    models,training={},{}
    for seed in SEEDS:
        for arm in ("full","cut"):
            models[seed,arm],training[f"{seed}/{arm}"] = train(seed,arm)
    rows = [evaluate(models,d) for d in [*DRAWS,42]]
    report = {"protocol_sha256":sha(FREEZE),"training":training,
              "fresh":summarize(rows[:-1]),"legacy_gate":summarize(rows[-1:]),"rows":rows}
    (RESULTS/"stage3_sparse_evaluation.json").write_text(json.dumps(report,indent=2,allow_nan=False),encoding="utf-8")
    print(json.dumps({"fresh":report["fresh"],"legacy":report["legacy_gate"]}),flush=True)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze",action="store_true")
    args = parser.parse_args()
    if args.freeze:
        with FREEZE.open("x",encoding="utf-8") as handle:
            json.dump(protocol(),handle,indent=2,allow_nan=False)
    else:
        run()
