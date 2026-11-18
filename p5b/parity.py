"""One differentiable normalized min-sum update on the public P4 (128,64) code."""

import math

import torch

from p4.slot_2d import _get_default_ldpc
from p5.unrolled_em_transformer import _llrs


def min_sum_extrinsic(llr):
    """Sum incoming check messages only. Positive LLR favors one.

    Each check sends the product of its other signs times their minimum
    magnitude, scaled by the P4 normalization of 0.75. The kink at magnitude
    ties has the ordinary top-k subgradient. Codeword and burst axes are
    independent. The adjacency tables encode the public H alone.
    """
    code = _get_default_ldpc(128)
    if llr.shape[-1] != code.n:
        raise ValueError("Expected complete 128-bit words on the last axis")
    # Conventional check products use positive-zero messages. A padding
    # entry with infinite magnitude and positive sign is neutral.
    words = -llr.reshape(-1, code.n)
    padding = words.new_full((len(words), 1), float("inf"))
    grid = torch.cat((words, padding), dim=-1)[:, code._cn_pad_vn]
    signs = torch.where(grid < 0, -1., 1.).to(words.dtype)
    smallest, indices = grid.abs().topk(2, dim=-1, largest=False, sorted=True)
    positions = torch.arange(code._dc_max, device=llr.device)
    other_min = torch.where(positions == indices[..., :1], smallest[..., 1:2], smallest[..., :1])
    # Convert the outgoing check messages back to the positive-one convention.
    outgoing = -0.75 * signs.prod(dim=-1, keepdim=True) * signs * other_min
    edges = outgoing.flatten(1)[:, code._edge_to_cn_flat]
    result = torch.zeros_like(words).scatter_add(1, code._edge_vn.expand_as(edges), edges)
    return result.reshape_as(llr)


def soft_parity_symbols(z, variance, informed, pilot_mask):
    """Posterior Gray 16-QAM means from the receiver field and one check round.

    The channel enters the final symbol likelihood exactly once. Only check
    messages supply the bit prior. Pilots are excluded from codewords and
    have a zero placeholder so the caller can reinsert known pilot symbols.
    """
    data = ~pilot_mask
    z_data = z[:, data]
    sigma = variance[:, data]
    observed = informed[:, data]
    channel = torch.where(observed[..., None], _llrs(z_data, sigma), 0.)
    external = min_sum_extrinsic(channel.reshape(z.shape[0], -1, 128)).reshape_as(channel)
    external = external.reshape(*z_data.shape, 2, 2)
    pam = z.real.new_tensor([-3., -1., 1., 3.]) / math.sqrt(10)
    squared_dist = (torch.view_as_real(z_data)[..., None] - pam).square()
    likelihood = torch.where(observed[..., None, None], -squared_dist / sigma[..., None, None], 0.)
    # Bits on sorted PAM axes (-3,-1,+1,+3) are (00,01,11,10).
    sign, inner = external.unbind(-1)
    prior = torch.stack((torch.zeros_like(sign), inner, sign + inner, sign), dim=-1)
    posterior = (likelihood + prior).softmax(-1)
    # Antisymmetric expression gives exact zero on a uniform posterior.
    mean = (3 * (posterior[..., 3] - posterior[..., 0])
            + posterior[..., 2] - posterior[..., 1]) / math.sqrt(10)
    soft = torch.zeros_like(z)
    soft[:, data] = torch.complex(mean[..., 0], mean[..., 1])
    return soft
