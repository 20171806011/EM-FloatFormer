"""Check every fixed reference tensor against the paired initialization."""

import torch

from p5.study import rng, save, setup
from p5b.diagnostics import ARMS, RESULTS
from p5b.operator_em_transformer import OperatorEMTransformer


def verify():
    rows = []
    for regime, pairs in ((1, 10), (2, 5), (3, 5)):
        for pair in range(1, pairs + 1):
            directory = RESULTS / f"r{regime}" / f"pair{pair:02d}"
            for arm in ARMS:
                torch.manual_seed(rng(regime, 4, pair))
                initial = OperatorEMTransformer(variant=arm)
                state = torch.load(directory / f"{arm}.pt", weights_only=True, map_location="cpu")
                live, frozen = [], []
                for key, tensor in initial.named_parameters():
                    if not torch.equal(tensor, state[key]):
                        (live if tensor.requires_grad else frozen).append(key)
                trainable_count = sum(p.requires_grad for p in initial.parameters())
                rows.append(dict(regime=regime, pair=pair, arm=arm, changed_trainables=live,
                                 changed_frozen=frozen, passed=len(live) == trainable_count and not frozen))
    return dict(checkpoints=len(rows), cases=rows, passed=all(row["passed"] for row in rows))


if __name__ == "__main__":
    setup()
    result = verify()
    save(RESULTS / "frozen_reference.json", result)
    print(f"{result['checkpoints']} checkpoints, passed={result['passed']}")
    raise SystemExit(0 if result["passed"] else 1)
