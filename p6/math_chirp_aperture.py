"""R2 checks for effective pilot geometry on intact, partial and scattered dev data."""
from dataclasses import replace

import torch

from p6 import experiment as ex, iter5_dev as old
from p6.data import PAIRS, dev_frame
from p6.math_chirp_dev import OUT, checkpoint
from p6.math_chirp_fisher import reliability
from p6.axial_em_transformer import _gauge


@torch.no_grad()
def evaluate():
    rows = []
    for pair in PAIRS:
        frame = dev_frame(2, 6, pair, 32)
        model = old.model_for(2, pair, "parity", .05, False)
        record = checkpoint(model, old.location(2, pair, "parity", 48, .05, False))
        mask = frame.obs.pilot_mask
        pilots = mask.nonzero()
        for label, multiplier, selected in (
            ("intact", 1., mask),
            *((f"first_times_{s}", s, mask & (torch.arange(mask.shape[0])[:,None] == 0))
              for s in (.8, .6, .4, .2, 0.)),
            ("inner_row_erased", 0., mask & (torch.arange(mask.shape[0])[:,None] == 9)),
            ("interleaved_half_erased", 0., mask & ((torch.arange(mask.shape[0])[:,None]
                         + torch.arange(mask.shape[1])[None]//2) % 2 == 0)),
            ("four_staggered_pilots_per_row", 0., mask & ((torch.arange(mask.shape[0])[:,None]
                         + torch.arange(mask.shape[1])[None]//2) % 4 == 0))):
            obs = frame.obs
            field = obs.y.clone()
            field[:, :, selected] *= multiplier
            obs = replace(obs, y=field)
            y, nv, _ = _gauge(obs, True)
            support = reliability(y, nv, mask)
            scores = {}
            for mode in ("baseline", "fisher"):
                model.chirp_mode = mode
                scores[mode] = ex.score([frame], lambda _: model(obs))
            rows.append(dict(pair=pair, condition=label, affected=int(selected.sum()),
                             pilot_sites=len(pilots), scale=multiplier, stream=6, batch=32,
                             checkpoint=record["checkpoint_sha256"], lambda_by_burst=support.tolist(),
                             scores=scores))
    ex.save(OUT / "aperture.json", dict(rows=rows))


if __name__ == "__main__":
    ex.setup()
    evaluate()
