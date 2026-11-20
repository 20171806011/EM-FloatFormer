"""Score selected B rows with the frozen P9 strong_large receiver configuration."""
from __future__ import annotations

import numpy as np

from p8c.link import extra_rows
from p9.channel import build_sc_channel_matrix, effective_row_noise
from p9.contract import CORE_ROWS


def strong_errors(obs, truth, mu, var, llrs, rows):
    """Return one 0/1 block error per requested data row, preserving caller order."""
    from p9.decoder import decode_strong_sc

    public_data_rows = (*CORE_ROWS, *extra_rows(obs.training_rows))
    truth_index = {r: idx for idx, r in enumerate(public_data_rows)}
    rows = tuple(rows)
    if not all(r in truth_index for r in rows):
        raise ValueError("Expected payload rows only")
    errors = []
    for row in rows:
        channel = build_sc_channel_matrix(mu[:, row])
        noise = effective_row_noise(var, obs.n0, rows=(row,))[0]
        estimate = decode_strong_sc(
            obs.y_b[row], channel, noise, list_size="large", bp_iters=20,
            turbo_pic=True, pic_scale=0.75, initial_llrs=llrs[row].reshape(128)
        )
        errors.append(int(not np.array_equal(estimate.info_bits, truth.b_info[truth_index[row]])))
    return errors
