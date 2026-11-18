"""Recreate coded receiver counts in batch-first order and audit tests directly from the CSV."""
import csv
from collections import Counter
from dataclasses import replace

import torch

from .slot_2d import SlotSimulator2D, _get_default_ldpc, _synchronize_slot, decode_min_sum, idd_pa_2d


@torch.inference_mode()
def count_again(model, seed, budget):
    code = _get_default_ldpc(128)
    rows = []
    for index, (condition, profile, fd, cfo) in enumerate((("id", "equal4", .01, 0.),
                 ("ood_doppler", "equal4", .05, .02), ("ood_delay", "long12", .01, 0.))):
        simulator = SlotSimulator2D(n_symb=14, n_sc=32, cp=16, profile=profile, fd=fd, cfo=cfo)
        counts = {}
        for batch in range(budget.eval_batches):
            generator = torch.Generator().manual_seed(10_000_000 + seed*1000 + index*100 + batch)
            slot = _synchronize_slot(simulator.generate_slot(batch_size=budget.eval_batch, snr_db=10.,
                    pa_condition="strong", coded=True, ldpc=code, generator=generator))
            fixed = replace(slot, profile="equal4", fd=.01, A=2.25, alpha=.5)
            output = model(slot.yp_sparse, slot.y_grid, slot.pilot_mask, slot.noise_var)
            predictions = {"emfloatformer2d": output.llrs[:, slot.data_mask]}
            for name, observed in (("_fixed", fixed), ("", slot)):
                for label, iterations in (("lmmse_2d", 0), ("idd_pa_2d", 2)):
                    predictions[label+name] = idd_pa_2d(slot_data=observed, iterations=iterations)["llrs"]
            for method, logits in predictions.items():
                if not torch.isfinite(logits).all():
                    raise FloatingPointError(f"Independent replay produced nonfinite LLRs for {method}.")
                batch_counts = counts.setdefault(method, Counter())
                batch_counts["total_uncoded_bits"] += slot.bits.numel()
                batch_counts["uncoded_bit_errors"] += int(torch.count_nonzero((logits > 0) != slot.bits))
                for scores, truth in zip(logits, slot.info_bits):
                    words = scores.flatten()[:len(truth)*code.n].reshape(len(truth), code.n)
                    mismatch = decode_min_sum(words, code.h_matrix, max_iter=25) != truth
                    batch_counts["block_errors"] += int(torch.count_nonzero(mismatch.any(dim=1)))
                    batch_counts["total_blocks"] += len(truth)
                    batch_counts["bit_errors"] += int(torch.count_nonzero(mismatch))
                    batch_counts["total_bits"] += truth.numel()
        rows.extend({"condition": condition, "method": k, **v} for k, v in counts.items())
    return rows


def audit_csv(path, records, primary):
    expected = {(r["seed"], row["condition"], row["method"]): row for r in records for row in r["rows"]}
    counts = {}
    with path.open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            key = int(row.pop("seed")), row["condition"], row["method"]
            if key in counts:
                raise ValueError("Duplicate row in confirmatory CSV.")
            converted = {k: v if k in ("condition", "method") else int(v) for k, v in row.items()}
            if converted != expected.get(key):
                raise ValueError(f"Confirmatory CSV cell mismatch at {key}.")
            counts[key] = converted["block_errors"]
    if counts.keys() != expected.keys():
        raise ValueError("Confirmatory CSV is missing expected rows.")
    raw = []
    seeds = [r["seed"] for r in records]
    for test in primary:
        cond, comparator = test["condition"], test["baseline"]
        diff = [counts[s, cond, comparator] - counts[s, cond, "emfloatformer2d"] for s in seeds]
        histogram = {0: 1}
        for d in diff:
            h = Counter()
            for total, multiplicity in histogram.items():
                h[total-d] += multiplicity
                h[total+d] += multiplicity
            histogram = h
        p = sum(n for v, n in histogram.items() if abs(v) >= abs(sum(diff)))/sum(histogram.values())
        if diff != test["baseline_minus_em_block_errors"] or p != test["exact_p_value"]:
            raise ValueError("Independent sign-flip audit disagrees.")
        raw.append(p)
    order = sorted(range(len(raw)), key=raw.__getitem__)
    for rank, i in enumerate(order):
        adjusted = min(1., max((len(raw)-j)*raw[order[j]] for j in range(rank+1)))
        if adjusted != primary[i]["holm_p_value"]:
            raise ValueError("Independent Holm audit disagrees.")
    return {"csv_rows": len(counts), "contrasts_reproduced": len(raw)}
