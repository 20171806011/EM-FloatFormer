"""S05 task-adapted EM receiver with a pilot-only three-tap Fourier initialization."""
import argparse
from contextlib import ExitStack
import hashlib
import json
import math
from pathlib import Path

import torch

from p4.next_pilots import estimate_from_pilots, joint_soft_channel, sparse_slot
from p4.slot_2d import EMFloatFormer2D, JointPhysicalLoss2D, SlotSimulator2D


FOLDER = Path(__file__).resolve().parent / "results"
FREEZE = FOLDER / "next_sparse_neural_protocol.json"
TRAIN_SEEDS = [73511, 73512, 73513]
FRESH = list(range(85821, 85837))


class SparseFourierEM(EMFloatFormer2D):
    """The declared support is 0,1,2 samples; no estimated tap is given to the model."""

    @staticmethod
    def snapshot(yp, n_sc=32):
        taps = torch.fft.ifft(yp, dim=-1)[..., :3]
        return torch.fft.fft(taps, n=n_sc, dim=-1)

    def _compute_rbf_prior(self, yp_bar, t_p, f_p, t_g, f_g, n_symb, n_sc):
        comb = torch.arange(0, n_sc, 4, device=f_p.device, dtype=f_p.dtype)
        if not torch.equal(f_p, comb) or not bool((t_p == 2).all()):
            raise ValueError("This S05 study uses the symbol-2 comb-4 grid")
        return self.snapshot(yp_bar, n_sc)[:, None].expand(-1, n_symb, -1)


def recipe():
    root = Path(__file__).resolve().parent
    return {
        "title": "Task-adapted S05 2D EM with a pilot-only causal Fourier snapshot",
        "training": {"seeds": TRAIN_SEEDS, "data_rng_offset": 1000000, "steps": 120, "batch": 8,
                     "n_symb":14, "n_sc":32, "cp":16, "profile":"train3", "pilot_symbol":2,
                     "comb":4, "snr_db":20, "normalized_doppler":0.01, "cfo":0, "pa":"clean",
                     "loss":"BCE + normalized channel MSE + 0.25 PA residual, unweighted means",
                     "optimizer":"Adam backbone 0.008, gamma 0.02, clip 2, cosine eta_min 0.0008",
                     "model":"EMFloatFormer2D width 16, two heads, two blocks, 3-tap causal DFT pilot prior constant across time.",
                     "selection":"Final update, all three prespecified seeds. No checkpoint selection."},
        "test": {"fresh_seeds":FRESH, "original_gate_seed":42, "slots_per_draw":8,
                 "primary":"Per-slot channel NMSE, averaged before 10log10(nearest LS NMSE / model NMSE). Original >=4 dB requirement; report each replicate and each draw.",
                 "secondary":"Uncoded payload hard bit errors; each full slot has 440*4 decisions.",
                 "controls":"Unwrapped nearest LS, pilot LMMSE, identical static three-tap DFT prior, six-step joint soft MAP, and two-row-trained zero-shot EM.",
                 "interventions":"Zero both directional cross-attention outputs (axial grid/pilot stream still active), or blank received payload grid (distribution-shift diagnostic). Frozen weights."},
        "source_sha256":{f:hashlib.sha256((root/f).read_bytes()).hexdigest() for f in
                         ("next_sparse_neural.py", "next_pilots.py", "slot_2d.py", "simulator.py",
                          "em_floatformer.py", "classical_baselines.py", "results/p4b_seed2026_state.pt")},
    }


def train(seed):
    torch.manual_seed(seed)
    model = SparseFourierEM()
    gen = torch.Generator().manual_seed(1000000 + seed)
    simulator = SlotSimulator2D(n_symb=14, n_sc=32, cp=16, profile="train3", fd=.01,
                                pilot_symbols=(2,), comb=4)
    criterion = JointPhysicalLoss2D()
    optimizer = torch.optim.Adam([
        {"params":[p for n,p in model.named_parameters() if "gamma_eff" not in n], "lr":.008},
        {"params":model.gamma_eff.parameters(), "lr":.02},
    ])
    decay = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=120, eta_min=.0008)
    losses = []
    active = {name:False for name,_ in model.named_parameters()}
    for step in range(120):
        slot = simulator.generate_slot(batch_size=8, snr_db=20., pa_condition="clean", generator=gen)
        out = model(slot.yp_sparse, slot.y_grid, slot.pilot_mask, slot.noise_var)
        loss = criterion(out, slot)
        optimizer.zero_grad()
        loss.backward()
        for name,param in model.named_parameters():
            if param.grad is None:
                raise RuntimeError(f"Disconnected parameter {name}")
            if not active[name]:
                active[name] = bool(torch.any(param.grad != 0))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0, error_if_nonfinite=True)
        optimizer.step()
        decay.step()
        losses.append(float(loss.detach()))
    if not all(active.values()) or not all(map(math.isfinite, losses)):
        raise FloatingPointError("Inactive training parameter or nonfinite loss")
    print(f"trained {seed}, active parameter tensors {sum(active.values())}", flush=True)
    return model.eval(), {"seed":seed,"loss_first_10":sum(losses[:10])/10,"loss_last_10":sum(losses[-10:])/10,
                          "all_parameter_tensors_had_nonzero_gradient":len(active),"steps":len(losses)}


def cross_disabled(model):
    stack = ExitStack()
    for block in model.blocks:
        for layer in (block.cross_g2p, block.cross_p2g):
            stack.enter_context(layer.register_forward_hook(lambda module, args, out: torch.zeros_like(out)))
    return stack


def metrics(h, slot, hard=None):
    nmse = (h - slot.h_true).abs().square().sum((-2,-1))/slot.h_true.abs().square().sum((-2,-1))
    if hard is None:
        # Exact 16-QAM bitwise MAP on the fitted diagonal channel (no PA, known AWGN).
        from p4.classical_baselines import calibrated_16qam_llrs
        hdata = h.expand_as(slot.y_grid)[:, slot.data_mask]
        power = hdata.abs().square().clamp_min(1e-12)
        z = slot.y_grid[:,slot.data_mask]*hdata.conj()/power
        hard = calibrated_16qam_llrs(z,slot.noise_var/power)>0
    return {"nmse_per_slot":nmse.tolist(), "bit_errors_per_slot":(hard!=slot.bits).sum((1,2)).tolist()}


@torch.inference_mode()
def evaluate(models, zero_shot, seed):
    slot = sparse_slot(seed, 4, 2, 20.)
    cols = slot.pilot_mask[2].nonzero().flatten()
    idx = (torch.arange(32)[:,None]-cols).abs().argmin(-1)
    nearest = slot.yp_sparse[:,idx][:,None].expand_as(slot.y_grid)
    args = (slot.y_grid,slot.pilot_tx,slot.pilot_mask,slot.noise_var,"train3",.01,16)
    methods = {
        "nearest_ls":metrics(nearest,slot),
        "static_fourier":metrics(SparseFourierEM.snapshot(slot.yp_sparse)[:,None].expand_as(slot.y_grid),slot),
        "pilot_lmmse":metrics(estimate_from_pilots(*args),slot),
        "joint_map":metrics(joint_soft_channel(*args),slot),
    }
    forward = (slot.yp_sparse,slot.y_grid,slot.pilot_mask,slot.noise_var)
    original = zero_shot(*forward)
    methods["zero_shot_em"] = metrics(original.h_hat,slot,original.llrs[:,slot.data_mask,:]>0)
    y_blank = slot.y_grid.clone()
    y_blank[:,slot.data_mask] = 0
    for sk,model in models.items():
        full = model(*forward)
        with cross_disabled(model):
            cut = model(*forward)
        blank = model(slot.yp_sparse,y_blank,slot.pilot_mask,slot.noise_var)
        for label, out in (("full",full),("cross_off",cut),("payload_blank",blank)):
            methods[f"{sk}/{label}"] = metrics(out.h_hat,slot,out.llrs[:,slot.data_mask,:]>0)
    return {"seed":seed,"slots":8,"bits_per_slot":slot.bits[0].numel(),"methods":methods}


def summarize(rows):
    out = {}
    for name in rows[0]["methods"]:
        sums = [sum(r["methods"][name]["nmse_per_slot"]) for r in rows]
        nearest = [sum(r["methods"]["nearest_ls"]["nmse_per_slot"]) for r in rows]
        gains = [10*math.log10(n/s) for n,s in zip(nearest,sums)]
        out[name] = {"mean_nmse":sum(sums)/sum(r["slots"] for r in rows),
                     "gain_db":10*math.log10(sum(nearest)/sum(sums)),
                     "draws_at_least_4db":sum(g >=4 for g in gains),
                     "bit_errors":sum(sum(r["methods"][name]["bit_errors_per_slot"]) for r in rows)}
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seal",action="store_true")
    args = parser.parse_args()
    config = recipe()
    if args.seal:
        if FREEZE.exists():
            assert json.loads(FREEZE.read_text()) == config
        else:
            FREEZE.write_text(json.dumps(config,indent=2,allow_nan=False),encoding="utf-8")
        return
    if json.loads(FREEZE.read_text()) != config:
        raise RuntimeError("Frozen S05 design differs from live source")
    torch.set_num_threads(2)
    models, training = {}, []
    for seed in TRAIN_SEEDS:
        models[seed], identity = train(seed)
        training.append(identity)
    zero_shot = EMFloatFormer2D().eval()
    zero_shot.load_state_dict(torch.load(FOLDER/"p4b_seed2026_state.pt",weights_only=True))
    rows = [evaluate(models,zero_shot,seed) for seed in [*FRESH,42]]
    result = {"protocol_sha256":hashlib.sha256(FREEZE.read_bytes()).hexdigest(),"training":training,
              "rows":rows,"fresh":summarize(rows[:-1]),"original_gate":summarize(rows[-1:])}
    (FOLDER/"next_sparse_neural_evaluation.json").write_text(json.dumps(result,indent=2,allow_nan=False),
                                                             encoding="utf-8")


if __name__ == "__main__":
    main()
