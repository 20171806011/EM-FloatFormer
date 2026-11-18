"""Pilot/data Transformer with unrolled soft 16-QAM channel estimation for S05."""
import math
from pathlib import Path

import torch
from torch import nn

from p4.classical_baselines import calibrated_16qam_llrs
from p4.em_floatformer import ComplexLinear, ComplexRMSNorm, FloatFormerOutput
from p4.next_pilots import sparse_slot
from p4.slot_2d import BidirectionalBlock2D, SLOT_PROFILES


FIRST_FIT = 73651
RESULTS = Path(__file__).resolve().parent / "results"


class SparseUnrolledEM(nn.Module):
    """Six physical updates and two bidirectional attention corrections.

    The fixed covariance models the declared train3 ensemble. The receiver
    consumes received samples, known pilot symbols/positions and noise variance.
    """

    def __init__(self, width=12, heads=2):
        super().__init__()
        time = torch.arange(14, dtype=torch.float64)
        freq = torch.arange(32, dtype=torch.float64)
        pdp = torch.tensor(SLOT_PROFILES["train3"], dtype=torch.float64)
        pdp /= pdp.sum()
        temporal = torch.special.bessel_j0(2*math.pi*.01*1.5*(time[:, None]-time).abs())
        values, vectors = torch.linalg.eigh(temporal)
        time_basis = vectors * values.clamp_min(0).sqrt()
        phase = -2*math.pi*freq[:, None]*torch.arange(len(pdp), dtype=time.dtype)/32
        freq_basis = torch.polar(torch.ones_like(phase), phase) * pdp.sqrt()
        design = torch.kron(time_basis.contiguous().to(torch.complex128), freq_basis.contiguous())
        self.register_buffer("design", design)
        self.register_buffer("identity", torch.eye(design.size(1), dtype=design.dtype))
        axis = torch.tensor([-3., -1., 1., 3.], dtype=time.dtype) / math.sqrt(10)
        self.register_buffer("symbols", torch.complex(axis[:, None], axis[None, :]).flatten())
        t, f = torch.meshgrid(time, freq, indexing="ij")
        self.register_buffer("times", t.flatten())
        self.register_buffer("tones", f.flatten())
        self.embed_p = ComplexLinear(1, width)
        self.embed_grid = ComplexLinear(4, width)
        self.blocks = nn.ModuleList([BidirectionalBlock2D(width, heads) for _ in range(2)])
        self.norms = nn.ModuleList([ComplexRMSNorm(width) for _ in range(2)])
        self.readouts = nn.ModuleList([ComplexLinear(width, 1) for _ in range(2)])
        for readout in self.readouts:
            nn.init.normal_(readout.weight_r, std=.01/math.sqrt(width))
            nn.init.normal_(readout.weight_i, std=.01/math.sqrt(width))

    def _weighted_solve(self, y, symbols, power, nv, design):
        """Gaussian prior and soft-symbol sufficient statistics for complex least squares."""
        adj = design.mH
        weight = power.reshape(len(y), -1) / nv
        normal = self.identity + adj @ (weight.unsqueeze(-1) * design)
        rhs = (symbols.conj()*y).reshape(len(y), -1) / nv
        coeff = torch.linalg.solve(normal, adj @ rhs.unsqueeze(-1))
        return (self.design @ coeff).reshape(len(y), 14, 32)

    def _beliefs(self, y, h, pilot_grid, mask, nv):
        score = -(y.unsqueeze(-1) - h.unsqueeze(-1)*self.symbols).abs().square() / nv[:, :, None, None]
        probability = torch.softmax(score, dim=-1)
        mean = (probability * self.symbols).sum(-1)
        power = (probability * self.symbols.abs().square()).sum(-1)
        return (torch.where(mask, pilot_grid, mean), torch.where(mask, pilot_grid.abs().square(), power))

    def forward(self, y_grid, pilot_tx, pilot_mask, noise_var, *, learned=True, trace=False,
                iterations=6):
        if y_grid.ndim != 3 or tuple(y_grid.shape[1:]) != (14, 32) or pilot_mask.shape != (14,32):
            raise ValueError("The S05 receiver is defined on a 14 by 32 slot.")
        if pilot_mask.dtype != torch.bool or pilot_tx.shape != (len(y_grid), int(pilot_mask.sum())) or pilot_tx.size(1)==0:
            raise ValueError("Expected the known pilot vector in row-major mask order.")
        if noise_var.numel() != len(y_grid) or bool((noise_var <= 0).any()):
            raise ValueError("The soft 16-QAM estimator expects a positive noise variance per slot.")
        if iterations < 1 or (learned and iterations != 6):
            raise ValueError("The learned receiver uses six physical iterations.")
        y64, pilots = y_grid.to(torch.complex128), pilot_tx.to(torch.complex128)
        yp = y64[:, pilot_mask]*pilots.conj()
        scale = (torch.linalg.vector_norm(yp, dim=1, keepdim=True)/math.sqrt(yp.size(1))).clamp_min(1e-38)
        y = y64 / scale.unsqueeze(-1)
        nv = noise_var.reshape(len(y), 1).double() / scale.square()
        pilot_grid = torch.zeros_like(y)
        pilot_grid[:, pilot_mask] = pilots
        h = self._weighted_solve(y[:, pilot_mask], pilots, pilots.abs().square(), nv,
                                 self.design[pilot_mask.flatten()])
        stages = {}
        def remember(name):
            if trace:
                stages[name] = (h * scale.unsqueeze(-1)).to(y_grid.dtype)
        remember("pilot")
        p_state = g_state = None
        coords = pilot_mask.nonzero().T.to(torch.float64)
        for step in range(1, iterations + 1):
            mean, power = self._beliefs(y, h, pilot_grid, pilot_mask, nv)
            h = self._weighted_solve(y, mean, power, nv, self.design)
            remember(f"map_{step}")
            if learned and step in (3, 6):
                idx = step//3 - 1
                mean, power = self._beliefs(y, h, pilot_grid, pilot_mask, nv)
                innovation = y - h*mean
                uncertainty = (power - mean.abs().square()).clamp_min(0)
                grid = self.embed_grid(torch.stack((y, h, innovation, h*uncertainty), dim=-1))
                pilot = self.embed_p((yp/scale).unsqueeze(-1))
                if p_state is not None:
                    pilot, grid = pilot + p_state, grid + g_state
                p_state, g_state = self.blocks[idx](pilot, grid, *coords, self.times, self.tones)
                correction = self.readouts[idx](self.norms[idx](g_state)).squeeze(-1)
                correction = .25 * correction / (1 + correction.abs().square()).sqrt()
                # Project the bounded local correction into the declared three-delay support.
                impulse = torch.fft.ifft(correction, dim=-1)[..., :len(SLOT_PROFILES["train3"])]
                h = h + torch.fft.fft(impulse, n=32, dim=-1)
                remember(f"learned_{step}")
        power = h.abs().square().clamp_min(1e-12)
        z = y*h.conj()/power
        variance = nv.unsqueeze(-1)/power + .035
        llr = 20 * torch.tanh(calibrated_16qam_llrs(z, variance) / 20)
        out = FloatFormerOutput(h_hat=(h*scale.unsqueeze(-1)).to(y_grid.dtype),
                                llrs=llr.to(y_grid.real.dtype), dist_est=torch.zeros_like(y_grid))
        return (out, stages) if trace else out


def load_receiver(training_seed=FIRST_FIT):
    import hashlib
    ckpt = torch.load(RESULTS/f"stage3_sparse_full_{int(training_seed)}.pt", weights_only=True)
    freeze = hashlib.sha256((RESULTS/"stage3_sparse_protocol.json").read_bytes()).hexdigest()
    if ckpt["seed"] != training_seed or ckpt["arm"] != "full" or ckpt["freeze"] != freeze:
        raise ValueError("Checkpoint does not match the frozen S05 fit.")
    model = SparseUnrolledEM().eval()
    model.load_state_dict(ckpt["weights"])
    return model


@torch.inference_mode()
def run_ultra_sparse_workload(comb=4, pilot_symbol=2, snr_db=20., seed=42, *, training_seed=FIRST_FIT):
    slot = sparse_slot(seed, comb, pilot_symbol, snr_db)
    model = load_receiver(training_seed)
    out = model(slot.y_grid, slot.pilot_tx, slot.pilot_mask, slot.noise_var)
    tones = slot.pilot_mask[pilot_symbol].nonzero().flatten()
    idx = (torch.arange(32, device=tones.device)[:, None]-tones).abs().argmin(-1)
    nearest = slot.yp_sparse[:,idx][:,None].expand_as(slot.y_grid)
    truth = slot.h_true.abs().square().sum((1,2))
    base = float(((nearest-slot.h_true).abs().square().sum((1,2))/truth).mean())
    nmse = float(((out.h_hat-slot.h_true).abs().square().sum((1,2))/truth).mean())
    return {"stable":bool(torch.isfinite(out.llrs).all() and torch.isfinite(out.h_hat).all()),
            "gain_over_ls_db":10*math.log10(base/max(nmse,1e-300)),
            "overhead":float(slot.pilot_mask.float().mean()),"nmse_ls":base, "nmse_2d":nmse,
            "receiver":"SparseUnrolledEM", "training_seed":training_seed}
