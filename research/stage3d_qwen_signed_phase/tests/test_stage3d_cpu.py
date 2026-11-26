"""CPU unit & invariant test suite for Stage 3D."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import torch

from research.stage3_ministral_cache.schema import ROOTS, ROTATIONS, matched_sum, sector_of_complex
from research.stage3d_qwen_signed_phase.data_stage3d import (
    ARM_NAMES,
    CONNECTOR_SEEDS,
    EQUIVARIANT_B8_S17_SHA256,
    NS_BLIND_G1,
    Q1_CHOICES,
    Q2_CHOICES,
    Q3_CHOICES,
    build_q2_arm_references,
    build_stage3d_prompt,
    condition_spec_from_params,
    generate_paired_cfo_blocks,
    load_s17_encoder,
    public_phase_slots,
    verify_stage3d_orbit_isolation,
)
from research.stage3d_qwen_signed_phase.qwen_connector import QwenCacheHost, SignedPhaseCacheConnector
from research.stage3d_qwen_signed_phase.recompute_all import (
    CELL_ORDER,
    Q2_ARMS,
    compute_16way_group_avg,
    compute_representative_readout,
    recompute_all,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


def test_encoder_sha_lock():
    enc, sha = load_s17_encoder(REPO_ROOT)
    assert sha == EQUIVARIANT_B8_S17_SHA256
    assert len(enc.table) == 16


def test_paired_cfo_generator_invariants():
    enc, _ = load_s17_encoder(REPO_ROOT)
    bmap, seeds = generate_paired_cfo_blocks(NS_BLIND_G1, 0, 2, 2, 2, 4.0, (-0.40, 0.40), split_prefix="G1_test")
    assert len(seeds) == 2
    neg, pos = bmap[-0.40], bmap[0.40]
    np.testing.assert_array_equal(neg.x1, pos.x1)
    np.testing.assert_array_equal(neg.x2, pos.x2)
    np.testing.assert_array_equal(neg.g, pos.g)
    np.testing.assert_array_equal(neg.y1, pos.y1)

    for b_idx in range(2):
        sl = slice(b_idx * 64, (b_idx + 1) * 64)
        np.testing.assert_array_equal(np.sort(16 * neg.x1[sl] + 4 * neg.x2[sl] + neg.g[sl]), np.arange(64))

    for t, fld, sym_attr, win_attr in ((2.0, "phi_2", "x2", "y2"), (4.0, "phi_3", "x1", "y3_repeat")):
        np.testing.assert_allclose(getattr(pos, fld) - getattr(neg, fld), 2.0 * 0.40 * t, atol=1e-12)
        base_n = ROOTS[(getattr(neg, sym_attr) + neg.g) % 4, None] / np.sqrt(3.0)
        n_neg = getattr(neg, win_attr) - base_n * np.exp(1j * getattr(neg, fld)[:, None])
        n_pos = getattr(pos, win_attr) - base_n * np.exp(1j * getattr(pos, fld)[:, None])
        np.testing.assert_allclose(n_neg, n_pos, rtol=0, atol=2e-14)

    # Check (0,1) and (1,1) rotation laws on s17 and k3_repeat
    z1_0 = enc.encode(pos.y1)
    z2_0 = enc.encode(pos.y2)
    v1_01, v2_01 = pos.rotated_view(0, 1)
    np.testing.assert_array_equal(enc.encode(v1_01), z1_0)
    np.testing.assert_array_equal(enc.encode(v2_01), (z2_0 + 4) % 16)
    v1_11, v2_11 = pos.rotated_view(1, 1)
    np.testing.assert_array_equal(enc.encode(v1_11), (z1_0 + 4) % 16)
    np.testing.assert_array_equal(enc.encode(v2_11), (z2_0 + 4) % 16)
    np.testing.assert_array_equal(sector_of_complex(matched_sum(pos.y3_repeat * ROOTS[1])), (pos.k3_repeat + 1) % 4)


def test_zero_leakage_prompts_and_slots():
    cond_pos = condition_spec_from_params(+0.40, 2, 2, 4.0, tag="G1_secret_leak_tag")
    cond_neg = condition_spec_from_params(-0.40, 2, 2, 4.0, tag="G1_secret_leak_tag")
    p_pos, ch_pos = build_stage3d_prompt("Q3", cond_pos, protocol="repeat_x1")
    p_neg, ch_neg = build_stage3d_prompt("Q3", cond_neg, protocol="repeat_x1")
    assert ch_pos == Q3_CHOICES == ch_neg
    assert "G1_secret_leak_tag" not in p_pos
    assert "C4" not in p_pos
    assert "+0.40" in p_pos and "-0.40" in p_neg
    for c in Q3_CHOICES:
        assert c in p_pos
    assert "<W1><W2><PHASE>" in p_pos

    s_pos = public_phase_slots(+0.40, 2, 2, 4.0, zero_arm=False)
    s_neg = public_phase_slots(-0.40, 2, 2, 4.0, zero_arm=False)
    s_zero = public_phase_slots(+0.40, 2, 2, 4.0, zero_arm=True)
    assert s_pos.shape == (6,)
    np.testing.assert_allclose(s_pos[0::2], s_neg[0::2], atol=1e-7)
    np.testing.assert_allclose(s_pos[1::2], -s_neg[1::2], atol=1e-7)
    np.testing.assert_array_equal(s_zero, np.zeros(6, dtype=np.float32))

    # Q2 dedup vs raw prompt check
    bmap, _ = generate_paired_cfo_blocks(950405, 0, 1, 2, 2, 4.0, (0.35,))
    batch = bmap[0.35]
    refs_v, dv = build_q2_arm_references(batch, "valid")
    refs_dd, ddd = build_q2_arm_references(batch, "duplicate_copy_dedup")
    refs_dr, ddr = build_q2_arm_references(batch, "duplicate_copy_raw")
    pv, _ = build_stage3d_prompt("Q2", batch.condition, references=refs_v, deduplicate=dv)
    pdd, _ = build_stage3d_prompt("Q2", batch.condition, references=refs_dd, deduplicate=ddd)
    pdr, _ = build_stage3d_prompt("Q2", batch.condition, references=refs_dr, deduplicate=ddr)
    assert pv == pdd and pdd.count("<REF>") == 1
    assert pdr.count("<REF>") == 2


def test_orbit_isolation():
    res = verify_stage3d_orbit_isolation(max_blocks=1)
    assert res["cross_split_orbit_collisions"] == 0


def test_paired_connector_init_and_mock_host():
    torch.manual_seed(CONNECTOR_SEEDS[0])
    c_phase = SignedPhaseCacheConnector(num_states=16, hidden_size=64, target_embed_std=0.02)
    torch.manual_seed(CONNECTOR_SEEDS[0])
    c_zero = SignedPhaseCacheConnector(num_states=16, hidden_size=64, target_embed_std=0.02)
    assert c_phase.param_count() == c_zero.param_count()
    for (n1, p1), (n2, p2) in zip(c_phase.named_parameters(), c_zero.named_parameters()):
        assert n1 == n2
        assert torch.equal(p1, p2)

    z1 = torch.tensor([0, 3, 7, 15], dtype=torch.long)
    z2 = torch.tensor([1, 4, 8, 12], dtype=torch.long)
    ps = torch.tensor(public_phase_slots(0.40, 2, 2, 4.0)).unsqueeze(0).expand(4, -1)
    rf = torch.randn(4, 12)
    w1, w2 = c_phase.encode_windows(z1, z2)
    ep = c_phase.encode_phase(ps)
    er = c_phase.encode_refs(rf)
    pr = c_phase.encode_prefix(4)
    loss = w1.pow(2).sum() + w2.pow(2).sum() + ep.pow(2).sum() + er.pow(2).sum() + pr.pow(2).sum()
    loss.backward()
    for name, p in c_phase.named_parameters():
        assert p.grad is not None and float(p.grad.abs().sum()) > 0.0, name

    if importlib.util.find_spec("transformers") is not None:
        mock_host = QwenCacheHost(mock_tiny_for_test=True)
        assert all(not p.requires_grad for p in mock_host.lm.parameters())


def test_recompute_all_clean_directory_roundtrip(tmp_path: Path):
    data_dir = tmp_path / "results"
    data_dir.mkdir(parents=True)
    rng = np.random.default_rng(42)

    temps_q3 = np.ones((2, 3), dtype=np.float64)
    temps_q2 = np.ones((2, 3), dtype=np.float64)
    q3_bundle = {"temps_q3": temps_q3, "temps_q2": temps_q2, "seeds": np.asarray(CONNECTOR_SEEDS, dtype=np.int64)}

    for cid, n_m in (("G1_main_extrap_both", 128), ("G2_interp_both", 64), ("G3_extrap_angle", 64), ("G4_extrap_freq", 64), ("H_c4_035", 64)):
        cal_ref = rng.dirichlet(np.ones(4), size=(2, 256))
        cal_mass = np.full((2, 256), 1.0 / 256.0)
        # Make phase arm match cal_ref closely and zero arm nearly uniform -> strong positive gain
        raw_ll = np.zeros((2, 3, 2, 256, 4), dtype=np.float64)
        raw_ll[0] = np.log(np.clip(cal_ref, 1e-4, 1.0))[None, ...] + 0.01 * rng.normal(size=(3, 2, 256, 4))
        raw_ll[1] = 0.01 * rng.normal(size=(3, 2, 256, 4))
        codes = rng.integers(0, 256, size=(2, 3, n_m), dtype=np.uint8)
        targets = rng.integers(0, 4, size=(2, 3, n_m), dtype=np.uint8)
        for s_i in range(2):
            for r_i in range(3):
                for m_i in range(n_m):
                    targets[s_i, r_i, m_i] = int(np.argmax(cal_ref[s_i, codes[s_i, r_i, m_i]]))
        q3_bundle[f"{cid}__raw_ll"] = raw_ll
        q3_bundle[f"{cid}__codes"] = codes
        q3_bundle[f"{cid}__targets"] = targets
        q3_bundle[f"{cid}__cal_ref"] = cal_ref
        q3_bundle[f"{cid}__cal_mass"] = cal_mass
        q3_bundle[f"{cid}__direct_table_probs"] = np.full((2, 3, n_m, 4), 0.25)
        q3_bundle[f"{cid}__full_iq_probs"] = np.full((2, 3, n_m, 4), 0.25)
        if cid == "G1_main_extrap_both":
            q3_bundle["G1_main_extrap_both__indep_raw_ll"] = np.zeros((2, 3, 2, 256, 4), dtype=np.float64)
            q3_bundle["G1_main_extrap_both__indep_targets"] = rng.integers(0, 4, size=(2, 3, n_m), dtype=np.uint8)
            q3_bundle["G1_main_extrap_both__rot10_codes"] = codes[:, 0, :]
            q3_bundle["G1_main_extrap_both__rot10_targets"] = targets[:, 0, :]
            q3_bundle["G1_main_extrap_both__shuffled_perm"] = np.broadcast_to(np.arange(n_m)[None, None, :], (2, 3, n_m)).astype(np.int32)

    np.savez_compressed(data_dir / "blind_q3_scores.npz", **q3_bundle)
    np.savez_compressed(
        data_dir / "blind_q2_scores.npz",
        q2__raw_ll=rng.normal(size=(2, 3, len(Q2_ARMS), 64, 4)),
        temps_q2=temps_q2,
        q2__labels=rng.integers(0, 4, size=64, dtype=np.uint8),
        q2__z1=rng.integers(0, 16, size=64, dtype=np.uint8),
        q2__z2=rng.integers(0, 16, size=64, dtype=np.uint8),
        q2__work_a_ref_probs=np.full((6, 64, 4), 0.25),
    )

    # Check Work G group avg equivariance algebraic identity
    tbl = rng.dirichlet(np.ones(4), size=(256,))
    pf = compute_16way_group_avg(tbl)
    f0 = compute_representative_readout(tbl)
    for z in range(256):
        z1, z2 = divmod(z, 16)
        z_rot = 16 * ((z1 + 4) % 16) + z2
        np.testing.assert_allclose(pf[z_rot], np.roll(pf[z], 1), atol=1e-12)
        np.testing.assert_allclose(f0[z_rot], np.roll(f0[z], 1), atol=1e-12)

    summ = recompute_all(data_dir)
    out_json = data_dir / "summary_master.json"
    out_json.write_text(json.dumps(summ, indent=2) + "\n", encoding="utf-8")

    # Copy ONLY recompute_all.py and results/ into a fresh clean directory outside repo!
    clean_dir = tmp_path / "clean_verify"
    clean_dir.mkdir()
    shutil.copy2(Path(recompute_all.__globals__["__file__"]), clean_dir / "recompute_all.py")
    shutil.copytree(data_dir, clean_dir / "results")
    subprocess.run(
        [sys.executable, "-B", "recompute_all.py", "--data-dir", "results", "--verify-against", "results/summary_master.json"],
        cwd=clean_dir,
        check=True,
    )
