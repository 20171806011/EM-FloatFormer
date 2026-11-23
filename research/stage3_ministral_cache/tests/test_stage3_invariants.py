"""Fast local CPU invariants & regression test suite for Stage 3."""
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch

from research.stage3_ministral_cache.analytic import (
    readout_from_table,
    solve_full_iq_known_param,
)
from research.stage3_ministral_cache.data import generate_mothers, verify_cross_split_isolation
from research.stage3_ministral_cache.eval_direct import load_frozen_models, run_regression_replay_check
from research.stage3_ministral_cache.ministral_connector import (
    WaveformCacheConnector,
    build_ref_feature,
)
from research.stage3_ministral_cache.schema import (
    CONDITION_SPECS,
    ROOTS,
    Reference,
    build_query_prompt,
    dedup_references,
    sector_of_complex,
)


def test_regression_504f3a44():
    models = load_frozen_models()
    reg = run_regression_replay_check(models)
    assert reg["regression_pass"] is True
    assert reg["max_abs_diff_vs_504f3a44"] < 1e-12


def test_sector_equivariance():
    # Exact canonical boundary and interior coordinates (avoid transcendental exp rounding on 45 deg boundaries)
    pts = np.array([1.0 + 0.0j, 0.2 + 0.9j, -1.0 + 0.2j, -0.3 - 1.4j, 1.0 + 1.0j, -1.0 + 1.0j, -1.0 - 1.0j, 1.0 - 1.0j])
    expected = np.array([0, 1, 2, 3, 1, 2, 3, 0], dtype=np.int64)
    np.testing.assert_array_equal(sector_of_complex(pts), expected)
    for k in range(4):
        np.testing.assert_array_equal(sector_of_complex(pts * ROOTS[k]), (expected + k) % 4)


def test_cross_split_orbit_isolation():
    audit = verify_cross_split_isolation(max_blocks_per_cond=1)
    assert audit["cross_split_orbit_collisions"] == 0, f"Found orbit collisions: {audit}"


@pytest.mark.parametrize("cname", ["C3_mid_pilot_cont", "C4_cfo_ramp", "C6_decoherent_order"])
def test_quadrature_128_vs_256(cname):
    batch = generate_mothers("s3_dev", cname, max_blocks=1)
    cond = batch.condition
    refs = batch.references_for_arm("valid")
    models = load_frozen_models()
    eq = models["equivariant_b8_s17"]
    a1 = eq.table[np.asarray(eq.encode(batch.y1), dtype=np.int64)]
    a2 = eq.table[np.asarray(eq.encode(batch.y2), dtype=np.int64)]

    out128 = readout_from_table(a1, a2, refs, cond, grid=128)
    out256 = readout_from_table(a1, a2, refs, cond, grid=256)
    for key in ("q1", "q2", "q3", "q3_after_pilot"):
        assert np.max(np.abs(out128[key] - out256[key])) < 1e-9

    iq128 = solve_full_iq_known_param(batch.y1, batch.y2, refs, cond, grid=128)
    iq256 = solve_full_iq_known_param(batch.y1, batch.y2, refs, cond, grid=256)
    for key in ("q1", "q2", "q3", "q3_after_pilot"):
        # Branch cut at phi_2 = +/- pi under fractional bridge (alpha=0.5) leaves O(5e-11) tail diff at sigma^2*h=0.25
        assert np.max(np.abs(iq128[key] - iq256[key])) < 1e-9


@pytest.mark.parametrize("cname", ["C2_discrete_walk", "C3_mid_pilot_cont"])
@pytest.mark.parametrize("comp_name", ["full_iq_known_param", "hand_b8", "equivariant_b8_s17"])
def test_rotations_01_hold_and_11_roll(cname, comp_name):
    batch = generate_mothers("s3_dev", cname, max_blocks=1)
    cond = batch.condition
    models = load_frozen_models()
    m = models.get(comp_name)

    def run_arm(s1, s2, arm, coord_rot=0):
        v1, v2 = batch.rotated_view(s1, s2)
        refs = batch.references_for_arm(arm, coord_rot=coord_rot)
        if comp_name == "full_iq_known_param":
            return {
                "rep": solve_full_iq_known_param(v1, v2, refs, cond, protocol="repeat_x1"),
                "ind": solve_full_iq_known_param(v1, v2, refs, cond, protocol="independent_x3"),
            }
        a1 = m.table[np.asarray(m.encode(v1), dtype=np.int64)]
        a2 = m.table[np.asarray(m.encode(v2), dtype=np.int64)]
        return {
            "rep": readout_from_table(a1, a2, refs, cond, protocol="repeat_x1"),
            "ind": readout_from_table(a1, a2, refs, cond, protocol="independent_x3"),
        }

    r00 = run_arm(0, 0, "valid")
    r01 = run_arm(0, 1, "valid")
    r11 = run_arm(1, 1, "valid")
    rc = run_arm(1, 1, "valid", coord_rot=1)

    tol = 3e-13
    assert np.max(np.abs(r01["rep"]["q1"] - np.roll(r00["rep"]["q1"], 1, axis=1))) < tol
    assert np.max(np.abs(r11["rep"]["q1"] - r00["rep"]["q1"])) < tol
    assert np.max(np.abs(r01["rep"]["q3"] - r00["rep"]["q3"])) < tol
    assert np.max(np.abs(r11["rep"]["q3"] - np.roll(r00["rep"]["q3"], 1, axis=1))) < tol
    assert np.max(np.abs(r00["ind"]["q3"] - 0.25)) < tol
    assert np.max(np.abs(r01["ind"]["q3"] - 0.25)) < tol
    assert np.max(np.abs(r11["ind"]["q3"] - 0.25)) < tol
    assert np.max(np.abs(r01["rep"]["q2"] - r00["rep"]["q2"])) < tol
    assert np.max(np.abs(r11["rep"]["q2"] - np.roll(r00["rep"]["q2"], 1, axis=1))) < tol
    assert np.max(np.abs(rc["rep"]["q1"] - r00["rep"]["q1"])) < tol
    assert np.max(np.abs(rc["rep"]["q2"] - r00["rep"]["q2"])) < tol
    assert np.max(np.abs(rc["rep"]["q3"] - np.roll(r00["rep"]["q3"], 1, axis=1))) < tol


def test_dedup_and_negative_control():
    batch = generate_mothers("s3_dev", "C1_coherent_2db", max_blocks=1)
    cond = batch.condition
    eq = load_frozen_models()["equivariant_b8_s17"]
    a1 = eq.table[np.asarray(eq.encode(batch.y1), dtype=np.int64)]
    a2 = eq.table[np.asarray(eq.encode(batch.y2), dtype=np.int64)]

    out_none = readout_from_table(a1, a2, batch.references_for_arm("none"), cond)["q2"]
    out_ind = readout_from_table(a1, a2, batch.references_for_arm("independent"), cond)["q2"]
    out_val = readout_from_table(a1, a2, batch.references_for_arm("valid"), cond)["q2"]
    out_dup = readout_from_table(a1, a2, batch.references_for_arm("duplicate_copy"), cond)["q2"]
    out_unsafe = readout_from_table(a1, a2, batch.references_for_arm("duplicate_copy"), cond, allow_double_count_unsafe=True)["q2"]

    assert np.max(np.abs(out_ind - out_none)) < 3e-13
    assert np.max(np.abs(out_dup - out_val)) < 3e-13
    assert np.mean(0.5 * np.abs(out_unsafe - out_val).sum(axis=1)) > 0.01

    # Conflict integrity checks in dedup_references
    r_ok = Reference(7, True, batch.p_valid1[0], 0.0, 2.0)
    r_bad_meta = Reference(7, False, batch.p_valid1[0], 0.0, 2.0)
    with pytest.raises(ValueError):
        dedup_references((r_ok, r_bad_meta))
    r_bad_samp = Reference(7, True, batch.p_valid2[0], 0.0, 2.0)
    with pytest.raises(ValueError):
        dedup_references((r_ok, r_bad_samp))

    # Prompt deduplication consistency: deduped duplicate_copy emits strictly 1 <REF> and matches valid verbatim
    p_valid, _ = build_query_prompt("Q2", cond, ref_arm="valid", references=batch.references_for_arm("valid"), deduplicate=True)
    p_dedup, _ = build_query_prompt("Q2", cond, ref_arm="duplicate_copy", references=batch.references_for_arm("duplicate_copy"), deduplicate=True)
    p_raw, _ = build_query_prompt("Q2", cond, ref_arm="duplicate_copy", references=batch.references_for_arm("duplicate_copy"), deduplicate=False)
    assert p_dedup.count("<REF>") == 1
    assert p_dedup == p_valid
    assert p_raw.count("<REF>") == 2


def test_connector_zero_leakage_and_gradients():
    conn = WaveformCacheConnector(num_states=256, hidden_size=64, target_embed_std=0.02)
    z1 = torch.tensor([0, 17, 255, 42], dtype=torch.long)
    z2 = torch.tensor([5, 19, 128, 3], dtype=torch.long)
    w1, w2 = conn.encode_windows(z1, z2)
    assert w1.shape == (4, 4, 64) and w2.shape == (4, 4, 64)
    assert not torch.allclose(w1, w2)  # distinct heads + inputs guarantee w1 != w2

    batch = generate_mothers("s3_dev", "C3_mid_pilot_cont", max_blocks=1)
    ref_feat = torch.tensor(build_ref_feature(batch.references_for_arm("valid")[0])[:4], dtype=torch.float32)
    assert ref_feat.shape == (4, 12)
    er = conn.encode_refs(ref_feat)
    assert er.shape == (4, 4, 64)
    hp = conn.encode_prefix(4)
    assert hp.shape == (4, 8, 64)

    loss = w1.pow(2).sum() + w2.pow(2).sum() + er.pow(2).sum() + hp.pow(2).sum()
    loss.backward()
    for name, p in conn.named_parameters():
        assert p.grad is not None and float(p.grad.abs().sum()) > 0.0, f"Zero grad in {name}"

    if importlib.util.find_spec("transformers") is not None:
        from research.stage3_ministral_cache.ministral_connector import MinistralCacheHost

        host = MinistralCacheHost(mock_tiny_for_test=True)
        pref = torch.randn((2, 12, host.hidden_size), device=host.device)
        # Cannot use tokenizer if mock has None, verify freeze invariant cleanly
        assert all(not p.requires_grad for p in host.lm.parameters())
