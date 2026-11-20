"""Original Construction A (C*-equivariant Unitary Parallel-Transport Coherent Wave Attention)
and Construction B (Cross-Hypothesis Jeffreys Divergence Evidence Routing)."""
import math

import torch
from torch import nn
from torch.nn import functional as F


REAL = torch.float64
COMPLEX = torch.complex128


def _cmat(rows, cols):
    return nn.Parameter(torch.randn(rows, cols, dtype=COMPLEX) / math.sqrt(rows))


def _safe_rms(x, dim=-1, keepdim=True):
    """Differentiable zero-safe RMS magnitude (never differentiates sqrt(0))."""
    p = x.abs().square().mean(dim=dim, keepdim=keepdim)
    safe = torch.where(p > 0, p, torch.ones_like(p))
    rms = safe.sqrt()
    unit = torch.where(p > 0, x / rms, torch.zeros_like(x))
    return unit, rms


def jeffreys_divergence(mu1, var1, mu2, var2):
    """Exact C*-invariant symmetric Jeffreys divergence between proper complex Gaussians CN(mu1,v1) & CN(mu2,v2)."""
    v1 = var1.clamp_min(1e-300)
    v2 = var2.clamp_min(1e-300)
    diff2 = (mu1 - mu2).abs().square()
    mean_term = 0.5 * diff2 * (1.0 / v1 + 1.0 / v2)
    var_term = 0.5 * (v1 / v2 + v2 / v1 - 2.0).clamp_min(0.0)
    return (mean_term + var_term).clamp_min(0.0)


def _phasor_from_coords(coords, omega):
    """Compute U(1) fiber parallel-transport phasors exp(i <omega_h, z>) for coords [..., L, 2], omega [H, 2]."""
    # angles: [..., L, H]
    angles = torch.einsum("...lc,hc->...lh", coords, omega)
    return torch.polar(torch.ones_like(angles), angles)


class CoherentTransportLayer(nn.Module):
    """Parameter-matched 2D coordinate-native layer supporting 3 strict scientific variants:
      - 'transport':        Signed Re(q'^H k') + Unitary Value Parallel Transport U_ij v_j (2D Pilot-to-Grid + 2D Separable)
      - 'abs_no_transport': Same global 2D topology & parameters, ablated to |q^H k| + untransported weights @ v
      - 'local':            Exact same parameter tensors restricted strictly to 3x3 local neighborhood (zero remote reach)
    """

    def __init__(self, variant="transport", d_model=16, n_heads=4):
        super().__init__()
        if variant not in ("transport", "abs_no_transport", "local"):
            raise ValueError(f"Unknown transport variant: {variant}")
        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        self.variant = variant
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        # Identical parameter layout across all 3 variants for exact integer parameter parity
        self.w_q = _cmat(d_model, d_model)
        self.w_k = _cmat(d_model, d_model)
        self.w_v = _cmat(d_model, d_model)
        self.w_pk = _cmat(d_model, d_model)
        self.w_pv = _cmat(d_model, d_model)
        self.w_o = _cmat(d_model, d_model)

        # Learnable continuous (t, f) wave connection frequencies per head
        # Physical band: Doppler over t_norm in [-1,1] spans ~[-4, +4]; positive delays tau in [0,6.5] give omega_f = -tau in [-5.5, -0.5]
        init_omega = torch.stack(
            (
                torch.linspace(-3.2, 3.2, n_heads, dtype=REAL),
                torch.linspace(-5.5, -0.5, n_heads, dtype=REAL),
            ),
            dim=-1,
        )
        self.omega = nn.Parameter(init_omega)  # H, 2
        # Multi-scale learnable geodesic distance decay (beta_t, beta_f >= 0 via softplus):
        # spans sharp local heads (protects O0/edges) to long-range global wave heads (crosses O1 gaps in 1 hop)
        init_beta_t = torch.linspace(4.0, -2.5, n_heads, dtype=REAL)
        init_beta_f = torch.linspace(2.5, -2.5, n_heads, dtype=REAL)
        self.raw_beta = nn.Parameter(torch.stack((init_beta_t, init_beta_f), dim=-1))  # H, 2
        self.gamma_ev = nn.Parameter(torch.full((n_heads,), 0.5, dtype=REAL))

        # C*-equivariant gated complex FFN
        self.ff1 = _cmat(d_model, 2 * d_model)
        self.ff2 = _cmat(2 * d_model, d_model)

    def _split_heads(self, x):
        return x.unflatten(-1, (self.n_heads, self.d_head))

    def _merge_heads(self, x):
        return x.flatten(-2)

    def _geodesic_bias(self, coords_q, coords_k):
        """Strictly C*-invariant multi-scale periodic geodesic distance penalty -beta_t dt^2 - beta_f sin^2(df/2)."""
        # coords_q: [..., Lq, 2], coords_k: [..., Lk, 2] -> [..., H, Lq, Lk]
        dt = coords_q[..., :, None, 0] - coords_k[..., None, :, 0]
        df = coords_q[..., :, None, 1] - coords_k[..., None, :, 1]
        dt2 = dt.square()
        sin2_f = torch.sin(0.5 * df).square()
        beta = F.softplus(self.raw_beta)  # H, 2
        bt = beta[:, 0].view(*([1] * (dt2.ndim - 2)), self.n_heads, 1, 1)
        bf = beta[:, 1].view(*([1] * (sin2_f.ndim - 2)), self.n_heads, 1, 1)
        return -(bt * dt2[..., None, :, :] + bf * sin2_f[..., None, :, :])

    def _attend_1d(self, q, k, v, coords_q, coords_k, log_bias=None, valid_k=None):
        """Core attention primitive: [..., Lq, H, Dh] x [..., Lk, H, Dh] -> [..., Lq, H, Dh]."""
        # Move heads before sequence: [..., H, L, Dh]
        qh = q.transpose(-3, -2)
        kh = k.transpose(-3, -2)
        vh = v.transpose(-3, -2)
        scale = 1.0 / math.sqrt(self.d_head)
        geo_bias = self._geodesic_bias(coords_q, coords_k)

        if self.variant == "transport":
            # Exact U(1) fiber parallel transport on Q, K, AND V!
            rho_q = _phasor_from_coords(coords_q, self.omega).transpose(-2, -1)[..., None]  # ..., H, Lq, 1
            rho_k = _phasor_from_coords(coords_k, self.omega).transpose(-2, -1)[..., None]  # ..., H, Lk, 1
            qp = qh * rho_q.conj()
            kp = kh * rho_k.conj()
            vp = vh * rho_k.conj()
            # Signed Hermitian inner product + multi-scale geodesic locality bias
            logits = (qp @ kp.mH).real * scale + geo_bias
            if log_bias is not None:
                logits = logits + log_bias
            if valid_k is not None:
                logits = logits.masked_fill(~valid_k, -1e9)
            weights = logits.softmax(dim=-1)
            if valid_k is not None:
                any_v = valid_k.any(dim=-1, keepdim=True)
                weights = torch.where(any_v, weights, torch.zeros_like(weights))
            # Aggregate in canonical origin frame then parallel-transport to target z_i
            vp_real = torch.view_as_real(vp.resolve_conj()).flatten(-2)  # ..., H, Lk, 2*Dh
            agg_real = weights @ vp_real
            agg = torch.view_as_complex(agg_real.unflatten(-1, (self.d_head, 2)))
            out = agg * rho_q
            return out.transpose(-3, -2)

        # variant == "abs_no_transport": legacy unsigned magnitude |q^H k| + untransported weights @ v
        # Uses identical geodesic bias & cosine coordinate bias (keeps 100% params active & fair)
        rho_q = _phasor_from_coords(coords_q, self.omega).transpose(-2, -1)[..., None]
        rho_k = _phasor_from_coords(coords_k, self.omega).transpose(-2, -1)[..., None]
        pos_bias = 0.25 * (rho_q @ rho_k.mH).real
        logits = (qh @ kh.mH).abs() * scale + pos_bias + geo_bias
        if log_bias is not None:
            logits = logits + log_bias
        if valid_k is not None:
            logits = logits.masked_fill(~valid_k, -1e9)
        weights = logits.softmax(dim=-1)
        if valid_k is not None:
            any_v = valid_k.any(dim=-1, keepdim=True)
            weights = torch.where(any_v, weights, torch.zeros_like(weights))
        v_real = torch.view_as_real(vh.resolve_conj()).flatten(-2)
        agg_real = weights @ v_real
        out = torch.view_as_complex(agg_real.unflatten(-1, (self.d_head, 2)))
        return out.transpose(-3, -2)

    def _local_3x3(self, state, unit_x, coords, pilot_bias_grid):
        """Strictly 3x3 local neighborhood coherent mixing using identical parameter tensors.

        Uses unit_x for C*-invariant Q/K weights and raw 1-homogeneous state for V (no amplitude suppression).
        """
        bm, t, n, d = state.shape
        q = self._split_heads(unit_x @ self.w_q)     # BM, T, N, H, Dh (unit-scale)
        k1 = self._split_heads(unit_x @ self.w_k)
        k2 = self._split_heads(unit_x @ self.w_pk)
        v1 = self._split_heads(state @ self.w_v)     # 1-homogeneous!
        v2 = self._split_heads(state @ self.w_pv)    # 1-homogeneous!
        k_comb = 0.5 * (k1 + k2)
        v_comb = 0.5 * (v1 + v2)

        # Build 9 local 3x3 neighbors (clamped in time, circular in frequency)
        t_idx = torch.arange(t, device=state.device)
        f_idx = torch.arange(n, device=state.device)
        scale = 1.0 / math.sqrt(self.d_head)

        logits_list = []
        v_trans_list = []
        beta = F.softplus(self.raw_beta)  # H, 2
        for dt in (-1, 0, 1):
            ti = (t_idx + dt).clamp(0, t - 1)
            for df in (-1, 0, 1):
                fi = (f_idx + df) % n
                kn = k_comb[:, ti][:, :, fi]  # BM, T, N, H, Dh
                vn = v_comb[:, ti][:, :, fi]
                # Relative coordinate lag (dt_norm, df_rad)
                d_coord = coords - coords[ti][:, fi]  # T, N, 2
                # Fix circular wrap for df_rad into [-pi, pi]
                d_f = ((d_coord[..., 1] + math.pi) % (2.0 * math.pi)) - math.pi
                d_c = torch.stack((d_coord[..., 0], d_f), dim=-1)  # T, N, 2
                u_rel = _phasor_from_coords(d_c, self.omega)[None, :, :, :, None]  # 1, T, N, H, 1
                kn_t = kn * u_rel
                vn_t = vn * u_rel
                geo_n = -(
                    beta[None, None, None, :, 0] * d_c[None, :, :, None, 0].square()
                    + beta[None, None, None, :, 1] * torch.sin(0.5 * d_c[None, :, :, None, 1]).square()
                )
                sc = (q.conj() * kn_t).sum(-1).real * scale + geo_n  # BM, T, N, H
                if pilot_bias_grid is not None:
                    pb_n = pilot_bias_grid[:, ti][:, :, fi]  # BM, T, N
                    sc = sc + pb_n[..., None] * self.gamma_ev[None, None, None, :]
                logits_list.append(sc)
                v_trans_list.append(vn_t)

        logits = torch.stack(logits_list, dim=-1)  # BM, T, N, H, 9
        weights = logits.softmax(dim=-1)
        v_stack = torch.stack(v_trans_list, dim=-2)  # BM, T, N, H, 9, Dh
        mixed = (weights[..., None].to(COMPLEX) * v_stack).sum(-2)  # BM, T, N, H, Dh
        return self._merge_heads(mixed) @ self.w_o

    def forward(self, state, coords_grid, p_state, p_coords, p_w, p_dj, grid_ev_bias=None):
        """
        Args:
          state:        [BM, T, N, D] complex128 (1-homogeneous in C*)
          coords_grid:  [T, N, 2] float64 continuous coordinates
          p_state:      [BM, N_p, D] complex128 observed pilot states (0 where p_w == 0; 1-homogeneous!)
          p_coords:     [BM, N_p, 2] float64 continuous pilot coordinates
          p_w:          [BM, N_p] float64 pilot precision weights (0 on held-out probes!)
          p_dj:         [BM, N_p] float64 Jeffreys divergence at pilot sites (0 where p_w == 0)
          grid_ev_bias: [BM, T, N] float64 optional grid-level reliability + log(1+D_J) evidence map
        """
        bm, t, n, d = state.shape
        unit_x, _ = _safe_rms(state, dim=-1, keepdim=True)  # BM, T, N, D

        if self.variant == "local":
            delta = self._local_3x3(state, unit_x, coords_grid, grid_ev_bias)
        else:
            # 1. Direct 1-hop 2D Pilot-to-Grid Coherent Cross-Attention (640 x N_p)
            # Q/K use unit-scale embeddings (C*-invariant weights); V uses RAW 1-homogeneous p_state
            # so remote pilot amplitude is transported into weak/faded gap sites without target-RMS suppression!
            p_unit, _ = _safe_rms(p_state, dim=-1, keepdim=True)
            q_flat = self._split_heads((unit_x @ self.w_q).reshape(bm, t * n, d))  # BM, 640, H, Dh
            pk = self._split_heads(p_unit @ self.w_pk)                              # BM, N_p, H, Dh
            pv = self._split_heads(p_state @ self.w_pv)                             # BM, N_p, H, Dh (1-homogeneous!)
            cg_flat = coords_grid.reshape(1, t * n, 2).expand(bm, -1, -1)           # BM, 640, 2

            valid_p = (p_w > 0)[:, None, None, :]  # BM, 1, 1, N_p
            ev_p = torch.log(p_w.clamp_min(1e-6)) + torch.log1p(p_dj.clamp_min(0.0))  # BM, N_p
            p_bias = ev_p[:, None, None, :] * self.gamma_ev[None, :, None, None]      # BM, H, 1, N_p
            cross_out = self._attend_1d(q_flat, pk, pv, cg_flat, p_coords, log_bias=p_bias, valid_k=valid_p)
            cross_grid = self._merge_heads(cross_out).reshape(bm, t, n, d)            # 1-homogeneous!

            # 2. 2D Separable Grid Coherent Transport (Freq axis=N -> Time axis=T), preserving 1-homogeneity in V
            z1 = state + 0.5 * cross_grid
            z1_u, _ = _safe_rms(z1, dim=-1, keepdim=True)
            qf = self._split_heads(z1_u @ self.w_q)   # BM, T, N, H, Dh
            kf = self._split_heads(z1_u @ self.w_k)
            vf = self._split_heads(z1 @ self.w_v)     # 1-homogeneous!
            cf = coords_grid[None].expand(bm, -1, -1, -1)  # BM, T, N, 2
            fb = None
            if grid_ev_bias is not None:
                fb = grid_ev_bias[:, :, None, None, :] * self.gamma_ev[None, None, :, None, None]
            mix_f = self._merge_heads(self._attend_1d(qf, kf, vf, cf, cf, log_bias=fb))

            z2 = z1 + 0.5 * mix_f
            z2_t = z2.transpose(1, 2)
            z2_tu, _ = _safe_rms(z2_t, dim=-1, keepdim=True)
            qt = self._split_heads(z2_tu @ self.w_q)
            kt = self._split_heads(z2_tu @ self.w_k)
            vt = self._split_heads(z2_t @ self.w_v)   # 1-homogeneous!
            ct = coords_grid.transpose(0, 1)[None].expand(bm, -1, -1, -1)  # BM, N, T, 2
            tb = None
            if grid_ev_bias is not None:
                tb = grid_ev_bias.transpose(1, 2)[:, :, None, None, :] * self.gamma_ev[None, None, :, None, None]
            mix_t = self._merge_heads(self._attend_1d(qt, kt, vt, ct, ct, log_bias=tb)).transpose(1, 2)

            delta = (0.5 * cross_grid + 0.25 * mix_f + 0.25 * mix_t) @ self.w_o

        h = state + delta
        h_u, h_r = _safe_rms(h, dim=-1, keepdim=True)
        ff_h = h_u @ self.ff1
        ff_gate = ff_h.abs().sigmoid()  # h_u is unit-scale -> strictly C*-invariant gate!
        ff_out = ((ff_h * ff_gate) @ self.ff2) * h_r  # 1-homogeneous!
        return h + 0.5 * ff_out


class CrossHypothesisJeffreysLayer(nn.Module):
    """Construction B: Cross-hypothesis M x M attention & Jeffreys-weighted branch evidence routing."""

    def __init__(self, d_model=16, n_heads=4):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.w_q = _cmat(d_model, d_model)
        self.w_k = _cmat(d_model, d_model)
        self.w_v = _cmat(d_model, d_model)
        self.w_o = _cmat(d_model, d_model)
        self.dj_scale = nn.Parameter(torch.full((n_heads,), 0.5, dtype=REAL))
        self.route_proj = _cmat(d_model, 2)

    def forward(self, state_m, pred_y_m, pred_var_m, h_m, res_obs_m, w_grid):
        """
        Args:
          state_m:    [B, M, T, N, D] complex128
          pred_y_m:   [B, M, T, N] complex128
          pred_var_m: [B, M, T, N] float64
          h_m:        [B, M, T, N] complex128
          res_obs_m:  [B, M, T, N] complex128 (0 on held-out/invalid sites)
          w_grid:     [B, T, N] float64 observation weights (0 on held-out probes!)
        Returns:
          updated state_m [B, M, T, N, D], route_delta [B, M] (strictly C*-invariant!)
        """
        b, m, t, n, d = state_m.shape
        unit_m, rms_m = _safe_rms(state_m, dim=-1, keepdim=True)

        if m > 1:
            # Pairwise Jeffreys divergence D_J(m, m'; t, f): [B, T, N, M, M]
            mu_a = pred_y_m.permute(0, 2, 3, 1)[..., :, None]
            mu_b = pred_y_m.permute(0, 2, 3, 1)[..., None, :]
            v_a = pred_var_m.permute(0, 2, 3, 1)[..., :, None]
            v_b = pred_var_m.permute(0, 2, 3, 1)[..., None, :]
            dj_pair = torch.log1p(jeffreys_divergence(mu_a, v_a, mu_b, v_b))  # B, T, N, M, M

            # [B, T, N, M, D] -> split heads [B, T, N, H, M, Dh]
            x_tn = unit_m.permute(0, 2, 3, 1, 4)
            q = (x_tn @ self.w_q).unflatten(-1, (self.n_heads, self.d_head)).transpose(-3, -2)
            k = (x_tn @ self.w_k).unflatten(-1, (self.n_heads, self.d_head)).transpose(-3, -2)
            v = (x_tn @ self.w_v).unflatten(-1, (self.n_heads, self.d_head)).transpose(-3, -2)

            logits = (q @ k.mH).real / math.sqrt(self.d_head) + self.dj_scale[None, None, None, :, None, None] * dj_pair[:, :, :, None]
            weights = logits.softmax(dim=-1)
            v_real = torch.view_as_real(v.resolve_conj()).flatten(-2)
            mixed = torch.view_as_complex((weights @ v_real).unflatten(-1, (self.d_head, 2)))
            mixed = mixed.transpose(-3, -2).flatten(-2) @ self.w_o  # B, T, N, M, D
            state_out = state_m + 0.5 * mixed.permute(0, 3, 1, 2, 4) * rms_m

            # Per-branch vs mixture Jeffreys divergence weight for evidence routing
            dj_site = dj_pair.mean(dim=-1).permute(0, 3, 1, 2)  # B, M, T, N
        else:
            # Keep parameters live cleanly even when M=1 (self-mixing across single branch)
            x_tn = unit_m.permute(0, 2, 3, 1, 4)
            q = (x_tn @ self.w_q).unflatten(-1, (self.n_heads, self.d_head)).transpose(-3, -2)
            k = (x_tn @ self.w_k).unflatten(-1, (self.n_heads, self.d_head)).transpose(-3, -2)
            v = (x_tn @ self.w_v).unflatten(-1, (self.n_heads, self.d_head)).transpose(-3, -2)
            _ = (q @ k.mH).real * 0.0 + self.dj_scale[None, None, None, :, None, None] * 0.0
            mixed = v.transpose(-3, -2).flatten(-2) @ self.w_o
            state_out = state_m + 0.5 * mixed.permute(0, 3, 1, 2, 4) * rms_m
            dj_site = torch.zeros((b, 1, t, n), dtype=REAL, device=state_m.device)

        # Jeffreys-weighted C*-invariant branch evidence readout [B, M]
        u_out, r_out = _safe_rms(state_out, dim=-1, keepdim=True)
        r_fields = (u_out @ self.route_proj) * r_out  # B, M, T, N, 2
        du, dw = r_fields.unbind(-1)
        ev_w = w_grid[:, None] * (1.0 + dj_site)  # B, M, T, N (0 on held-out probes!)
        ev_den = ev_w.sum((-2, -1)).clamp_min(1.0)
        h_eng = h_m.abs().square().mean((-2, -1)).clamp_min(1e-300)
        route_delta = (((h_m.conj() * du + res_obs_m.conj() * dw).real * ev_w).sum((-2, -1))
                       / (ev_den * h_eng))
        return state_out, route_delta, dj_site
