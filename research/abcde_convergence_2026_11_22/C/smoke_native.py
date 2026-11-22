"""Single synthetic grid. Matches JS against the fixed repository's actual torch demapper."""
import json
from pathlib import Path
import subprocess

import numpy as np
import torch
import native_bridge as bridge
from ofdm_data import DATA_COORDS_101, USED_SCS, PILOT_SYM, PILOT_SCS_14, PILOT_SCS_7


def complex_pairs(p):
    a = np.asarray(p, dtype=np.float64)
    return a[..., 0] + 1j * a[..., 1]


def main():
    proc = subprocess.run(["node", str(Path(__file__).with_name("synthetic.cjs")), "--native-fixture"],
                          check=True, text=True, capture_output=True)
    raw = json.loads(proc.stdout)
    y = complex_pairs(raw["y"])
    pilots = np.asarray(raw["pilots"])
    contexts = [{**c, "hLS": complex_pairs(c["hLS"]), "hCA": complex_pairs(c["hCA"]),
                 "nv": np.asarray(c["nv"]), "logw": np.asarray(c["logw"])}
                for c in raw["contexts"]]
    s, k = DATA_COORDS_101.T
    metrics = {}
    for method in ("discrete", "continuous"):
        got = bridge.receive(y, pilots, contexts, method)
        indexed = {(e["s"], e["k"]): e["llr"] for e in raw["expected"][method]}
        assert len(indexed) == len(s) == 1400
        ref = np.asarray([indexed[ss, kk] for ss, kk in zip(s, USED_SCS[k])])
        assert got.shape == (1400, 4) and np.isfinite(got).all()
        np.testing.assert_allclose(got, ref, rtol=2e-5, atol=2e-4)
        metrics[method] = {"max_abs_error_to_js": float(np.max(np.abs(got-ref))),
                           "hard_decision_mismatches": int(np.sum((got > 0) != (ref > 0)))}
        # Float ties near zero may disagree; require equality away from the tolerance.
        assert np.all(((got > 0) == (ref > 0)) | (np.abs(ref) <= 2e-4))
        for fold, ctx in enumerate(contexts):
            target = k % 2 == fold
            changed = y.copy()
            changed[s[target], USED_SCS[k[target]]] = 11-23j
            view1, valid1 = bridge.support_view(y, pilots, fold)
            view2, valid2 = bridge.support_view(changed, pilots, fold)
            np.testing.assert_array_equal(view1, view2)
            assert not valid1[s[target], USED_SCS[k[target]]].any()
            assert valid1[PILOT_SYM, pilots].all()
            np.testing.assert_array_equal(valid1, valid2)
            hf = bridge.classical_head(y, pilots, fold, ctx, method)
            altered = bridge.classical_head(changed, pilots, fold, ctx, method)
            for original, mutated in zip(hf, altered):
                torch.testing.assert_close(original, mutated, rtol=0, atol=0)
            torch.testing.assert_close(hf[0][0], torch.tensor(ctx["logw"], dtype=torch.float32),
                                      rtol=0, atol=0)
            assert torch.count_nonzero(hf[2]).item() == 0
    # Exercise actual mask-aware frontend and both native CFO methods on a fresh synthetic reference bank.
    from fold_frontend import make_builder
    refs = np.stack([np.ones(101, dtype=complex), np.exp(1j*np.linspace(-.3,.3,101))])
    builder = make_builder(refs)
    native_ctx = bridge.build_contexts(y, pilots, builder)
    for fold in (0, 1):
        target = k % 2 == fold
        changed = y.copy()
        changed[s[target], USED_SCS[k[target]]] = -8+3j
        other = builder(*bridge.support_view(changed, pilots, fold), pilots, fold)
        for key in native_ctx[fold]:
            np.testing.assert_array_equal(native_ctx[fold][key], other[key])
    for method in ("discrete", "continuous"):
        pred = bridge.receive(y, pilots, native_ctx, method)
        assert pred.shape == (1400, 4) and np.isfinite(pred).all()
    # Hidden legacy pilot slots must not be visible in the sparse-configuration adapter.
    sparse = PILOT_SCS_7
    _, valid = bridge.support_view(y, sparse, 0)
    assert not valid[PILOT_SYM, np.setdiff1d(PILOT_SCS_14, sparse)].any()
    print(json.dumps({"status":"passed","dataset":"synthetic_only","frames":1,
                      "native":metrics,"torch":torch.__version__}, indent=2))


if __name__ == "__main__":
    main()
