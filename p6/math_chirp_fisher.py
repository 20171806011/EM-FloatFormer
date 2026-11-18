"""Pilot-aperture information for the public two-frequency chirp family."""
import torch


def reliability(y, noise, mask):
    """Least supported normalized Fisher direction after profiling constant phase."""
    mask = mask.to(y.real.dtype)
    count = mask.sum(-1)
    nominal_total = count.sum()
    t = torch.linspace(-1., 1., mask.shape[0], dtype=y.real.dtype, device=y.device)
    design = torch.stack((t, t.square()), -1)
    centroid = (count @ design) / nominal_total.clamp_min(1.)
    centered = design - centroid
    full = centered.mT @ (count[:, None] * centered)
    if torch.linalg.matrix_rank(full).item() < 2:
        return y.real.new_zeros(y.shape[0])
    factor = torch.linalg.cholesky(full)
    rows = torch.linalg.solve_triangular(factor, centered.mT, upper=False).mT
    pilot_power = (y.abs().square().mean(1) * mask).sum(-1) / count.clamp_min(1.)
    nv = noise[:, None]
    relative = pilot_power / torch.where(nv > 0, nv, 1.)
    # Subtract the expected noise power and saturate each row at unit signal SNR.
    usable = torch.where(nv > 0, (relative - 1.).clamp(0., 1.), (pilot_power > 0).double())
    w = count * usable
    total = w.sum(-1, keepdim=True)
    mean = (w @ rows) / torch.where(total > 0, total, 1.)
    local = rows - mean[:, None]
    actual = local.mT @ (w[..., None] * local)
    supported = torch.linalg.eigvalsh(actual)[:, 0].clamp(0., 1.)
    return torch.where((w == count).all(-1), 1., supported)
