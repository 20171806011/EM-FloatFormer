"""R3 Green probe using uniform weights on the three parallel axial lines."""

from pathlib import Path
import statistics

import torch

from p5.study import save, setup
from p5.wavefield_sim import green_kernel
from p6.axial_operator import _probe_coordinates
from p6.data import PAIRS, dev_frame, dev_seed
from p6.experiment import correlation


@torch.no_grad()
def run():
    shape = (4, 5, 32)
    coords = _probe_coordinates(*shape)
    off = ~torch.eye(len(coords), dtype=torch.bool)
    equal = coords[:, None] == coords[None, :]
    masks = [equal[..., [j for j in range(3) if j != axis]].all(-1) for axis in range(3)]
    weights = [1 / size for size in shape]
    parallel = sum(mask.double() * weight for mask, weight in zip(masks, weights)) / 3
    per_pair = {}
    for pair in PAIRS:
        frame = dev_frame(3, 8, pair, 8, batch=0)
        truth = green_kernel(frame, coords)
        values = correlation(parallel[off], truth[:, off]).tolist()
        per_pair[str(pair)] = dict(seed=dev_seed(3, 8, pair, 0), per_burst=values,
                                   mean=statistics.mean(values))
    return dict(regime=3, stream=8, batch_index=0, batch_size=8, total_bursts=24,
                torch_threads=torch.get_num_threads(), torch_interop_threads=torch.get_num_interop_threads(),
                coordinates=coords.tolist(), off_diagonal_entries=int(off.sum()),
                axes=["spatial", "time", "frequency"], sizes=shape, full_axis_weights=weights,
                off_diagonal_entries_on_each_axis=[int((mask & off).sum()) for mask in masks],
                off_diagonal_entries_off_all_axes=int((parallel[off] == 0).sum()),
                aggregation="Each burst uses 2256 off-diagonal entries; pair and pooled scores average burst Pearson coefficients.",
                parallel_joint=dict(by_pair=per_pair, mean=statistics.mean(p["mean"] for p in per_pair.values())),
                historical_product=dict(value=1/(shape[0]*shape[1]*shape[2]), pearson=None,
                                        reason="Uniform axis weights yield a constant product."))


if __name__ == "__main__":
    torch.set_num_interop_threads(1)
    setup()
    result = run()
    save(Path(__file__).parent / "results" / "iter2_checks" / "green_support_control.json", result)
    print(result["parallel_joint"])
