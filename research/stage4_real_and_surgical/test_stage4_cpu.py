"""Mandatory $0 CPU verification test suite for research/stage4_real_and_surgical."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from research.stage3_ministral_cache import analytic as s3_analytic
from research.stage3_ministral_cache.data import generate_mothers
from research.stage3_ministral_cache.ministral_connector import _compute_host_fingerprint
from research.stage3_ministral_cache.schema import CONDITION_SPECS, ROOTS, Reference
from research.stage3d_qwen_signed_phase.data_stage3d import load_s17_encoder
from research.stage3d_qwen_signed_phase.qwen_connector import QwenCacheHost
from research.stage4_real_and_surgical.equi_connector import (
    ARM_NAMES_STAGE4,
    PhysicalKernelS17,
    Stage4MatchedConnector,
    extract_canonical_ref_summary,
    lift_scores,
    unlift_probs,
)
from research.stage4_real_and_surgical.packet_c16 import (
    decode_c16,
    encode_c16,
    rotate_c16,
)
from research.stage4_real_and_surgical.phaseB_surgical import (
    _build_blind_suites,
    _predict_model_on_suites,
    audit_pure_and_mix_kl,
    run_phaseB_surgical_and_lemmo,
    run_stage4_synthetic_blind,
)
from research.stage4_real_and_surgical.real_sdr_and_stage6_gpu import (
    Stage6ActiveHostAdapter,
    merge_surgical_lie_schur_soup,
    run_stage4_real_sdr_gpu,
    run_stage6_active_eig_gpu,
)
from research.stage4_real_and_surgical.real_sdr_causal import (
    export_dual_cache_c16,
    validate_branch_literal_parity,
    validate_cttc_splits,
    validate_pluto_splits,
)
from research.stage4_real_and_surgical.sdr_real_pipeline import (
    load_bundle_npz,
    run_real_sdr_cpu_pipeline,
)
from research.stage4_real_and_surgical.surgical_adapter import (
    QWEN4B_DN_LAYERS,
    QWEN4B_FA_LAYERS,
    QWEN4B_STOP_LAYER,
    TIER_P1_MAX_PARAMS,
    FP32ComplexNullspaceCayley,
    SurgicalHostAdapter,
    UnconstrainedLoRAAdapter,
    _get_decoder_layers,
    apply_complex_nullspace_fast,
    apply_j_d,
    calibrate_b_align_from_host,
    compute_complex_nullspace_projector,
    param_budget_audit,
)
from research.stage4_real_and_surgical.train_eval_stage4 import (
    collect_kfac_complex_projectors,
    run_gate1_pilot_69step,
    run_stage4_smoke_and_probe,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_packet_c16_65536_sweep():
    words = np.arange(65536, dtype="<u2")
    dec = decode_c16(words)
    assert np.array_equal(dec["s17"], words >> 12)
    assert np.array_equal(dec["turn"], words >> 14)

    # Known M5 fixture vector (-1 + 0j with s17=10 -> bytes 0ca8)
    w_fix = encode_c16(10, -1.0 + 0.0j, 0.0, 1.0, True)
    assert w_fix.tobytes() == bytes.fromhex("0ca8")

    # C4 rotation invariance on lower 14 bits
    for r in range(4):
        rot = rotate_c16(words, r)
        assert np.array_equal(rot & 0x3FFF, words & 0x3FFF)
        assert np.array_equal(rot >> 14, ((words >> 14) + r) % 4)


def test_matched_4arm_param_parity_and_bitwise_c4():
    enc, _ = load_s17_encoder(REPO_ROOT)
    counts = {}
    models = {}
    for arm in ARM_NAMES_STAGE4:
        m = Stage4MatchedConnector(arm, enc.table, hidden_size=2560)
        counts[arm] = m.param_count()
        models[arm] = m
        assert counts[arm] == 488_400, f"{arm} param count={counts[arm]} != 488400"

    # PhysicalKernelS17 vs analytic.readout_from_table parity check (< 1e-3)
    kern = PhysicalKernelS17(enc.table)
    for cname in ("C3_mid_pilot_cont", "C2_discrete_walk"):
        cond = CONDITION_SPECS[cname]
        batch = generate_mothers("s3_dev", cname, max_blocks=1)
        z1 = np.asarray(enc.encode(batch.y1), dtype=np.int64)
        z2 = np.asarray(enc.encode(batch.y2), dtype=np.int64)
        t1, t2 = enc.table[z1], enc.table[z2]
        codes_t = torch.from_numpy(np.stack([z1, z2], axis=-1)).to(torch.long)
        meta_t = torch.tensor(
            [[cond.h, cond.ell, cond.t_p, cond.omega, cond.sigma_phi, cond.snr_db, 0.0]] * len(z1),
            dtype=torch.float32,
        )
        dd_t = torch.full((len(z1),), float(cond.d), dtype=torch.float32)

        res_pre = s3_analytic.readout_from_table(t1, t2, (), cond, protocol="repeat_x1")
        p_q1 = kern.direct_physical_probs(codes_t, meta_t, "Q1", discrete_d=dd_t).numpy()
        p_q3 = kern.direct_physical_probs(codes_t, meta_t, "Q3_repeat_x1", discrete_d=dd_t).numpy()
        assert np.max(np.abs(p_q1 - res_pre["q1"])) < 1e-3, f"Q1 physics mismatch on {cname}"
        assert np.max(np.abs(p_q3 - res_pre["q3"])) < 1e-3, f"Q3 physics mismatch on {cname}"

        for q2_arm in ("valid", "fresh_resample"):
            refs = batch.references_for_arm(q2_arm, coord_rot=0)
            rs_t, _ = extract_canonical_ref_summary(
                refs,
                len(z1),
                torch.device("cpu"),
                omega=float(cond.omega),
                sigma_phi=float(cond.sigma_phi),
                drift_type=str(cond.drift_type),
                discrete_d=float(cond.d),
            )
            res_q2 = s3_analytic.readout_from_table(t1, t2, refs, cond, protocol="repeat_x1")
            p_q2 = kern.direct_physical_probs(codes_t, meta_t, "Q2", m3_q2=rs_t[:, 0:4], discrete_d=dd_t).numpy()
            assert np.max(np.abs(p_q2 - res_q2["q2"])) < 1e-3, f"Q2 ({q2_arm}) physics mismatch on {cname}"

    # Bitwise C4 orbit invariance check on mock QwenCacheHost + Q3_indep autograd check
    host = QwenCacheHost(mock_tiny_for_test=True)
    small_pure = Stage4MatchedConnector("equivariant_host_pure", enc.table, hidden_size=host.hidden_size)
    small_res = Stage4MatchedConnector("equivariant_physical_residual", enc.table, hidden_size=host.hidden_size)

    base_codes = torch.tensor([[2, 3], [0, 1], [1, 2]], dtype=torch.long)
    ps = torch.tensor([[0.8, 0.6, 0.5, -0.5, 0.9, 0.1]] * 3, dtype=torch.float32)
    meta = torch.tensor([[2.0, 2.0, 4.0, 0.35, 0.10, 2.0, 0.0]] * 3, dtype=torch.float32)
    rs_pilot = torch.zeros((3, 29), dtype=torch.float32)
    rs_pilot[:, 0:4] = torch.tensor([0.6, 0.2, 0.1, 0.1])
    rs_pilot[:, 4:8] = rs_pilot[:, 0:4]

    for mod in (small_pure, small_res):
        mod.eval()
        with torch.no_grad():
            # Verify unlift_probs(lift_scores(p, codes, task), codes, task) == p identically
            p_rand = torch.tensor([[0.5, 0.2, 0.2, 0.1], [0.1, 0.6, 0.2, 0.1], [0.25, 0.35, 0.15, 0.25]])
            codes_rand = torch.tensor([[2, 15], [7, 1], [13, 9]], dtype=torch.long)
            for tname in ("Q1", "Q2", "Q3_repeat_x1", "Q3_after_pilot"):
                rt = unlift_probs(lift_scores(p_rand, codes_rand, tname), codes_rand, tname)
                assert float((rt - p_rand).abs().max().item()) == 0.0, f"unlift/lift mismatch on {tname}"

            # Verify zero and naive_additive zero out coupled harmonic slots fused[:, -12:]
            tid_zero = torch.zeros((3,), dtype=torch.long)
            for uncoupled_arm in ("zero", "naive_additive"):
                uc_mod = Stage4MatchedConnector(uncoupled_arm, enc.table, hidden_size=host.hidden_size)
                f_uc, _, _ = uc_mod.build_fused_features(base_codes, 4, tid_zero, ps, meta, ref_summary=rs_pilot)
                assert float(f_uc[:, -12:].abs().max().item()) == 0.0
            f_cp, _, _ = mod.build_fused_features(base_codes, 4, tid_zero, ps, meta, ref_summary=rs_pilot)
            assert float(f_cp[:, -12:].abs().max().item()) > 0.0

            _, _, p0_q1, _ = mod.score_batch(host, "test", ("a", "b", "c", "d"), "Q1", base_codes, 4, ps, meta, ref_summary=rs_pilot)
            _, _, p0_q2, _ = mod.score_batch(host, "test", ("a", "b", "c", "d"), "Q2", base_codes, 4, ps, meta, ref_summary=rs_pilot)
            _, _, p0_q3, _ = mod.score_batch(host, "test", ("a", "b", "c", "d"), "Q3_repeat_x1", base_codes, 4, ps, meta, ref_summary=rs_pilot)
            for r1 in range(4):
                for r2 in range(4):
                    rot_codes = torch.stack([(base_codes[:, 0] + 4 * r1) % 16, (base_codes[:, 1] + 4 * r2) % 16], dim=-1)
                    _, _, pr_q1, _ = mod.score_batch(host, "test", ("a", "b", "c", "d"), "Q1", rot_codes, 4, ps, meta, ref_summary=rs_pilot)
                    assert float((pr_q1 - p0_q1.roll(r2 - r1, dims=-1)).abs().max().item()) == 0.0

                    _, _, pr_q2, _ = mod.score_batch(host, "test", ("a", "b", "c", "d"), "Q2", rot_codes, 4, ps, meta, ref_summary=rs_pilot)
                    assert float((pr_q2 - p0_q2.roll(r1, dims=-1)).abs().max().item()) == 0.0

                    _, _, pr_q3, _ = mod.score_batch(host, "test", ("a", "b", "c", "d"), "Q3_repeat_x1", rot_codes, 4, ps, meta, ref_summary=rs_pilot)
                    assert float((pr_q3 - p0_q3.roll(r1, dims=-1)).abs().max().item()) == 0.0

        # Ensure Q3_independent_x3 backward() preserves grad_fn without raising RuntimeError
        mod.train()
        mod.zero_grad(set_to_none=True)
        lp_ind, _, _, _ = mod.score_batch(
            host, "test", ("a", "b", "c", "d"), "Q3_independent_x3", base_codes, 4, ps, meta, ref_summary=rs_pilot
        )
        F.nll_loss(lp_ind, torch.tensor([0, 1, 2])).backward()
        mod.zero_grad(set_to_none=True)


def test_ref_pool_duplicate_distractor_and_order_invariance():
    rng = np.random.default_rng(77)
    s1 = np.ones((4, 3), dtype=np.complex128)
    s2 = 1j * np.ones((4, 3), dtype=np.complex128)
    dist = -np.ones((4, 3), dtype=np.complex128) + 0.2 * rng.standard_normal((4, 3))
    r0 = Reference(10, True, s1, 4.0, 2.0)
    r1 = Reference(25, True, s2, 4.0, 2.0)
    rd = Reference(99, False, dist, 4.0, 2.0)

    base_sum, _ = extract_canonical_ref_summary([r0, r1], 4, torch.device("cpu"))
    dup_sum, _ = extract_canonical_ref_summary([r0, r0, r1, r0], 4, torch.device("cpu"))
    dist_sum, _ = extract_canonical_ref_summary([rd, r1, None, r0, r1, rd], 4, torch.device("cpu"))
    rev_sum, _ = extract_canonical_ref_summary([r1, r0], 4, torch.device("cpu"))

    assert float((base_sum - dup_sum).abs().max().item()) == 0.0
    assert float((base_sum - dist_sum).abs().max().item()) == 0.0
    assert float((base_sum - rev_sum).abs().max().item()) == 0.0


def test_complex_nullspace_cayley_and_mode1_stopgrad_mock_host():
    d = 64
    k_c = 12
    rng = torch.Generator().manual_seed(123)
    x_text = torch.randn(256, d, generator=rng)
    # Dominated by k_c=12 complex modes
    c_gen = x_text.T @ x_text
    proj = compute_complex_nullspace_projector(c_gen, k_c=k_c)
    p_real = proj["P_perp_real"]
    u_c = proj["U_c"]

    # Check [P_perp^C, J_d] == 0
    v_rand = torch.randn(16, d, generator=rng)
    assert float((apply_j_d(v_rand @ p_real.T) - (apply_j_d(v_rand) @ p_real.T)).abs().max().item()) < 1e-5
    # Check fast projector matches dense matrix
    assert float((apply_complex_nullspace_fast(v_rand, u_c) - (v_rand @ p_real.T)).abs().max().item()) < 1e-5

    # Check FP32 Cayley unitarity & text-subspace identity
    cay = FP32ComplexNullspaceCayley(d, r_c=4, u_c=u_c)
    with torch.no_grad():
        cay.v_re.normal_(std=0.25)
        cay.v_im.normal_(std=0.25)
    q_mat = cay.materialize_unitary()
    eye = torch.eye(d // 2, dtype=torch.complex64)
    assert float((q_mat.conj().T @ q_mat - eye).abs().max().item()) < 1e-5

    # Forward matches materialized matrix and acts as identity on top 2k_c text eigenvectors
    y_fast = cay(v_rand)
    z_rand = torch.complex(v_rand[:, 0::2], v_rand[:, 1::2]) @ q_mat.T
    y_mat = torch.empty_like(v_rand)
    y_mat[:, 0::2] = z_rand.real
    y_mat[:, 1::2] = z_rand.imag
    assert float((y_fast - y_mat).abs().max().item()) < 1e-5

    top_vec_z = u_c[:, :4].T  # (4, m)
    top_vec_r = torch.zeros(4, d)
    top_vec_r[:, 0::2] = top_vec_z.real
    top_vec_r[:, 1::2] = top_vec_z.imag
    assert float((cay(top_vec_r) - top_vec_r).abs().max().item()) < 1e-6

    # Attach SurgicalHostAdapter to mock QwenCacheHost and verify Mode I stop-grad + PureKL=0 + immutable host
    host = QwenCacheHost(mock_tiny_for_test=True)
    enc, _ = load_s17_encoder(REPO_ROOT)
    conn = Stage4MatchedConnector("equivariant_host_pure", enc.table, hidden_size=host.hidden_size)
    sha_before = host.host_sha256

    c_mock = torch.eye(host.hidden_size) * 0.1
    u_c_mock = compute_complex_nullspace_projector(c_mock, k_c=4)["U_c"]
    adapter = SurgicalHostAdapter(
        host.lm,
        fa_layers=(1,),
        dn_layers=(),
        stop_layer=0,
        mode="mode1_stopgrad_l10",
        u_c_by_layer={1: u_c_mock},
    )
    try:
        # PureKL == 0 when g_EM == 0 (even with non-zero adapter weights!)
        with torch.no_grad():
            for cay_mod in adapter.fa_cayleys.values():
                cay_mod.v_re.normal_(std=0.2)
        emb = torch.randn(2, 6, host.hidden_size, generator=rng)
        attn = torch.ones(2, 6, dtype=torch.long)
        with torch.no_grad():
            base = host.lm(inputs_embeds=emb, attention_mask=attn, use_cache=False).logits
            with adapter.activate(g_em=0):
                gated = host.lm(inputs_embeds=emb, attention_mask=attn, use_cache=False).logits
        assert float((base - gated).abs().max().item()) == 0.0

        # Mode I stop_layer=0 must block gradient flow to connector and layer 0
        words = torch.tensor([[1, 2], [3, 0]], dtype=torch.long)
        ps = torch.zeros(2, 6)
        meta = torch.tensor([[2.0, 2.0, 4.0, 0.2, 0.1, 2.0, 0.0]] * 2)
        rs = torch.zeros(2, 29)
        rs[:, 0:8] = 0.25
        lp, _, _, _ = conn.score_batch(host, "t", ("a", "b", "c", "d"), "Q1", words, 4, ps, meta, ref_summary=rs, surgical_adapter=adapter)
        F.nll_loss(lp, torch.tensor([0, 1])).backward()
        assert all(p.grad is None for p in conn.parameters()), "Mode I stop_grad failed to block connector grad!"
        assert all(p.grad is None for p in host.lm.parameters()), "Frozen host params must have no grad!"
        assert _compute_host_fingerprint(host.lm) == sha_before

        # Verify _masked_write_tensor on prefix (B, pref_l), KV continuation (B*4, step_len < pref_l), and uncached (B*4, pref_l + max_len)
        b_sz, pref_l, step_l, max_l = 2, 10, 4, 5
        mask = torch.tensor([[1, 1, 0, 0, 0, 1, 0, 0, 0, 0], [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]], dtype=torch.bool)
        with adapter.activate(g_em=1, m_acc_mask=mask):
            x_pref = torch.ones((b_sz, pref_l, 8))
            assert torch.equal(adapter._masked_write_tensor(x_pref), mask.unsqueeze(-1).expand(-1, -1, 8).float())
            x_cont = torch.ones((b_sz * 4, step_l, 8))
            assert float(adapter._masked_write_tensor(x_cont).abs().max().item()) == 0.0
            x_uncached = torch.ones((b_sz * 4, pref_l + max_l, 8))
            got_uncached = adapter._masked_write_tensor(x_uncached)
            exp_prefix = mask.repeat_interleave(4, dim=0).unsqueeze(-1).expand(-1, -1, 8).float()
            assert torch.equal(got_uncached[:, :pref_l, :], exp_prefix)
            assert float(got_uncached[:, pref_l:, :].abs().max().item()) == 0.0
    finally:
        adapter.remove()

    # Verify interleaved_half_head_dim only modifies query halves [Q_h, Gate_h] per head
    from research.stage4_real_and_surgical.surgical_adapter import NullspaceLoRADelta

    head_dim = 8
    n_heads = 2
    d_in = 16
    inter_lora = NullspaceLoRADelta(d_in, n_heads * head_dim * 2, r=4, interleaved_half_head_dim=head_dim)
    with torch.no_grad():
        inter_lora.lora_b.fill_(1.0)
        x_t = torch.randn(2, 5, d_in)
        base_out = torch.zeros(2, 5, n_heads * head_dim * 2)
        delta = inter_lora(x_t)
        res = base_out.clone().view(2, 5, n_heads, 2 * head_dim)
        res[..., :head_dim] = res[..., :head_dim] + delta.view(2, 5, n_heads, head_dim)
        res_flat = res.reshape_as(base_out).view(2, 5, n_heads, 2 * head_dim)
        assert float(res_flat[..., head_dim:].abs().max().item()) == 0.0
        assert float(res_flat[..., :head_dim].abs().max().item()) > 0.0


def test_4b_param_budget_strictly_under_tier_p1_cap():
    # Verify Qwen3.5-4B cumulative param count exact bound <= 2,184,964
    from research.stage4_real_and_surgical.surgical_adapter import (
        FP32ComplexNullspaceCayley,
        NullspaceLoRADelta,
        SchurIrrepReadout,
    )

    enc, _ = load_s17_encoder(REPO_ROOT)
    conn_4b = Stage4MatchedConnector("equivariant_host_pure", enc.table, hidden_size=2560)
    c_params = conn_4b.param_count()
    assert c_params == 488_400

    # 5 FA layers + 4 DN layers + Schur readout on d=2560 (using interleaved gated q_proj)
    fa_params = 5 * (
        sum(p.numel() for p in FP32ComplexNullspaceCayley(4096, r_c=4).parameters())
        + sum(p.numel() for p in NullspaceLoRADelta(2560, 8192, r=8, interleaved_half_head_dim=256).parameters())
        + 2 * sum(p.numel() for p in NullspaceLoRADelta(2560, 1024, r=8).parameters())
    )
    dn_params = 4 * (
        sum(p.numel() for p in FP32ComplexNullspaceCayley(4096, r_c=4).parameters())
        + sum(p.numel() for p in NullspaceLoRADelta(2560, 8192, r=8, out_slice=(2048, 8192)).parameters())
        + 2 * sum(p.numel() for p in NullspaceLoRADelta(2560, 32, r=4).parameters())
    )
    readout_params = sum(p.numel() for p in SchurIrrepReadout(2560, r=11).parameters())
    total_4b = c_params + fa_params + dn_params + readout_params
    assert total_4b <= TIER_P1_MAX_PARAMS, f"4B total {total_4b} exceeds Tier P1 cap {TIER_P1_MAX_PARAMS}"


def test_causal_sdr_splits_and_mock_pilot_e2e(tmp_path: Path):
    p_res = validate_pluto_splits()
    c_res = validate_cttc_splits()
    b_res = validate_branch_literal_parity(REPO_ROOT)
    assert p_res["ok"] and c_res["ok"] and b_res["cttc_branch_checked"] and b_res["pluto_branch_checked"]

    enc, _ = load_s17_encoder(REPO_ROOT)
    iq = (np.ones((6, 3)) + 0.1j * np.ones((6, 3))) / math.sqrt(3.0)
    dc = export_dual_cache_c16(iq, enc, noise_variance=0.5, available=True)
    assert dc["words_c16"].dtype == np.dtype("<u2")

    smoke_dir = tmp_path / "smoke"
    s_rep = run_stage4_smoke_and_probe(REPO_ROOT, smoke_dir, mock_tiny_for_test=True)
    assert s_rep["pure_kl_g_em_0"] == 0.0
    assert s_rep["max_logit_diff_g_em_0"] == 0.0
    assert s_rep["mode1_surgical_live_ok"] is True

    pilot_dir = tmp_path / "pilot"
    p_rep = run_gate1_pilot_69step(
        REPO_ROOT,
        pilot_dir,
        seed=202611,
        steps=2,
        batch_size=8,
        micro_batch_size=4,
        train_blocks=1,
        dev_blocks=1,
        mock_tiny_for_test=True,
    )
    assert set(p_rep["pilot"]["arms"].keys()) == set(ARM_NAMES_STAGE4)
    pure_dev = p_rep["pilot"]["arms"]["equivariant_host_pure"]["dev_metrics"]
    assert "opt_temp_dev" in pure_dev and "temp_cal_dev_nll_informative" in pure_dev
    assert "per_task" in p_rep["voi"]["probe_a"] and len(p_rep["voi"]["probe_a"]["per_task"]) > 0
    assert p_rep["voi"]["gate2_recommendation"]["decision"] in (
        "ABORT_FIX_CPU_CONNECTOR",
        "STAY_FROZEN_BRANCH0",
        "TRIGGER_MODE1_SURGICAL_96",
    )


def test_phaseB_surgical_lemmo_and_blind_mock_e2e(tmp_path: Path):
    enc, _ = load_s17_encoder(REPO_ROOT)
    host = QwenCacheHost(mock_tiny_for_test=True)
    b_align = calibrate_b_align_from_host(host)
    assert b_align.shape == (host.hidden_size, 3)
    assert np.allclose(b_align.norm(dim=0).numpy(), np.ones(3), atol=1e-5)

    # Verify UnconstrainedLoRAAdapter cumulative budget on 4B geometry strictly <= TIER_P1_MAX_PARAMS
    from transformers import Qwen2Config, Qwen2ForCausalLM

    # Approximate 4B-sized mock linear dimensions cleanly via Stage4 formula verified against UnconstrainedLoRAAdapter on mock host
    lemmo_mock = UnconstrainedLoRAAdapter(host.lm, fa_layers=(1,), dn_layers=(), stop_layer=0, r_lora=8)
    try:
        conn_4b = Stage4MatchedConnector("equivariant_host_pure", enc.table, hidden_size=2560)
        lemmo_4b_expected = (
            5
            * (
                4 * (4096 + 2560)
                + 8 * (2560 + 4096)
                + 2 * 8 * (2560 + 1024)
            )
            + 4
            * (
                4 * (4096 + 2560)
                + 8 * (2560 + 6144)
                + 2 * 4 * (2560 + 32)
            )
        )
        assert conn_4b.param_count() + lemmo_4b_expected <= TIER_P1_MAX_PARAMS
        assert lemmo_mock.param_count() > 0
    finally:
        lemmo_mock.remove()

    out_dir = tmp_path / "phaseB_out"
    ckpt_dir = tmp_path / "ckpts"
    kfac_p = tmp_path / "kfac_missing.pt"
    pb = run_phaseB_surgical_and_lemmo(
        REPO_ROOT,
        out_dir,
        ckpt_dir,
        kfac_p,
        seed=202611,
        steps=2,
        batch_size=8,
        micro_batch_size=4,
        train_blocks=1,
        dev_blocks=1,
        mock_tiny_for_test=True,
    )
    assert pb["surgical_96"]["audit"]["pure_kl_max"] == 0.0
    assert pb["surgical_96"]["audit"]["pure_max_logit_diff"] == 0.0
    assert pb["surgical_96"]["host_sha256"] == pb["lemmo_tier_p1"]["host_sha256"]
    assert pb["surgical_96"]["under_tier_p1_cap"] is True
    assert pb["lemmo_tier_p1"]["under_tier_p1_cap"] is True

    cache_dir = tmp_path / "blind_cache"
    eval_host = QwenCacheHost(mock_tiny_for_test=True)
    done_models: list[str] = []
    blind = run_stage4_synthetic_blind(
        REPO_ROOT,
        out_dir,
        ckpt_dir,
        kfac_p,
        seed=202611,
        g1_blocks=1,
        eval_bs=16,
        mock_tiny_for_test=True,
        host=eval_host,
        cache_dir=cache_dir,
        on_model_done=done_models.append,
    )
    expected_models = {
        "direct_physical",
        "zero",
        "naive_additive",
        "equivariant_host_pure",
        "equivariant_physical_residual",
        "mode1_surgical_96",
        "lemmo_tier_p1",
    }
    assert set(blind["models"].keys()) == expected_models
    assert set(done_models) == expected_models
    for mname, md in blind["models"].items():
        assert math.isfinite(md["G1_main_extrap_both"]["nll"])
        assert len(md["Q2_C4_blind"]) >= 7
        assert math.isfinite(md["G1_counterfactuals"]["mean_sym_kl_factual_vs_slot_flipped"])
        assert (cache_dir / f"{mname}.pt").exists()
    assert "holm_bonferroni_family" in blind["m2_paired_stats"]

    # Cache hit check: re-running with same host & cache_dir must trigger zero recomputations and match identically
    reused_callbacks: list[str] = []
    blind_cached = run_stage4_synthetic_blind(
        REPO_ROOT,
        out_dir,
        ckpt_dir,
        kfac_p,
        seed=202611,
        g1_blocks=1,
        eval_bs=16,
        mock_tiny_for_test=True,
        host=eval_host,
        cache_dir=cache_dir,
        on_model_done=reused_callbacks.append,
    )
    assert reused_callbacks == []
    assert blind_cached["models"] == blind["models"]


def test_blind_dedup_vs_bruteforce_parity() -> None:
    """Verify deduplicated `_predict_model_on_suites` matches brute-force `score_batch` across all 4 arms."""
    host = QwenCacheHost(mock_tiny_for_test=True)
    encoder, _ = load_s17_encoder(REPO_ROOT)
    full_suites = _build_blind_suites(encoder, g1_blocks=1, other_blocks=1)
    # Build compact multi-task suite with explicit gauge-equivalent duplicates & discrete_d variation
    g_q3 = dict(full_suites["G1_main_extrap_both"][0])
    g_q3_disc = dict(full_suites["G1_main_extrap_both"][0])
    g_q3_disc["discrete_d"] = np.where(np.arange(len(g_q3["discrete_d"])) % 2 == 0, 0.0, 0.15).astype(np.float32)
    sample_suites = {
        "G1_main_extrap_both": [g_q3, g_q3_disc],
        "Q2_C4_blind": [dict(full_suites["Q2_C4_blind"][0]), dict(full_suites["Q2_C4_blind"][2])],
        "G1_counterfactuals": [dict(full_suites["G1_counterfactuals"][1])],
    }

    for arm in ARM_NAMES_STAGE4:
        torch.manual_seed(17)
        conn = Stage4MatchedConnector(arm, encoder.table, hidden_size=host.hidden_size).to(host.device)
        conn.eval()
        dedup_preds = _predict_model_on_suites(host, conn, sample_suites, adapter=None, eval_bs=7, scorer_mode="kv_cached")

        with torch.no_grad():
            for sname, grps in sample_suites.items():
                for g, p_dedup in zip(grps, dedup_preds[sname]):
                    n = len(g["words"])
                    bf_chunks = []
                    for st in range(0, n, 11):
                        ed = min(n, st + 11)
                        words = torch.from_numpy(g["words"][st:ed]).to(host.device)
                        ps = torch.from_numpy(g["phase_slots"][st:ed]).to(host.device)
                        meta = torch.from_numpy(g["meta_scalars"][st:ed]).to(host.device)
                        rs = torch.from_numpy(g["ref_summary"][st:ed]).to(host.device)
                        dd = torch.from_numpy(g["discrete_d"][st:ed]).to(host.device)
                        _, _, probs_bf, _ = conn.score_batch(
                            host,
                            g["prompt"],
                            g["choices"],
                            g["qtype"],
                            words,
                            4,
                            ps,
                            meta,
                            ref_summary=rs,
                            scorer_mode="kv_cached",
                            discrete_d=dd,
                        )
                        bf_chunks.append(probs_bf.detach().cpu().numpy().astype(np.float64))
                    p_bf = np.concatenate(bf_chunks, axis=0)
                    max_diff = float(np.max(np.abs(p_dedup - p_bf)))
                    assert max_diff < 1e-6, f"arm={arm} suite={sname} max_diff={max_diff}"


def test_real_sdr_causal_pipeline_sentinel_and_dual_cache(tmp_path: Path):
    rep = run_real_sdr_cpu_pipeline(
        REPO_ROOT,
        tmp_path / "raw",
        tmp_path / "cache",
        mock_synthetic_for_test=True,
        seed=202611,
    )
    sent = rep["sentinel_zero_leakage"]
    assert sent["ok"] is True
    assert sent["pluto_s17_max_diff"] == 0
    assert sent["pluto_w16_max_diff"] == 0
    assert sent["pluto_meta_max_diff"] == 0.0
    assert sent["pluto_ref_none_max_diff"] == 0.0
    assert sent["cttc_causal_fit_max_diff"] == 0.0

    pluto_bundle = load_bundle_npz(tmp_path / "cache" / "pluto_dual_cache_c16.npz")
    cttc_bundle = load_bundle_npz(tmp_path / "cache" / "cttc_dual_cache_c16.npz")
    for sh in (*pluto_bundle["splits"].values(), pluto_bundle["stress"]):
        assert np.array_equal(sh["words_c16"] >> 12, sh["s17_codes"])
    for sh in (*cttc_bundle["splits"].values(), cttc_bundle["stress"], cttc_bundle["q3_pairs"]):
        assert np.array_equal(sh["words_c16"] >> 12, sh["s17_codes"])
    assert len(np.unique(pluto_bundle["stress"]["cluster_ids"])) >= 2

    phys = rep["cpu_physical_results"]
    for dom in ("pluto_test", "cttc_in_domain_test", "cttc_unseen_holdout_f708_snr10"):
        for dec in ("direct_s17_4bit", "dual_cache_c16_16bit", "continuous_causal_bayes"):
            q2_d = phys[dom][dec]["q2"]
            assert q2_d["max_prob_diff_dup_vs_two"] == 0.0
            assert q2_d["max_prob_diff_indep_vs_two"] == 0.0


def test_w2_lie_schur_soup_unitarity_and_pure_kl_zero():
    enc, _ = load_s17_encoder(REPO_ROOT)
    host = QwenCacheHost(mock_tiny_for_test=True)
    layers = _get_decoder_layers(host.lm)
    n_l = len(layers)
    fa_l = [i for i in QWEN4B_FA_LAYERS if i < n_l] or ([n_l - 1] if n_l >= 1 else [])
    dn_l = [i for i in QWEN4B_DN_LAYERS if i < n_l]
    stop_l = min(QWEN4B_STOP_LAYER, max(0, n_l - 2))
    proj_map = collect_kfac_complex_projectors(host, sorted(set(fa_l + dn_l)), k_c=4)
    b_align = calibrate_b_align_from_host(host)

    s_a = SurgicalHostAdapter(
        host.lm, fa_layers=fa_l, dn_layers=dn_l, stop_layer=stop_l, mode="mode1_stopgrad_l10", u_c_by_layer=proj_map, b_align=b_align
    )
    s_b = SurgicalHostAdapter(
        host.lm, fa_layers=fa_l, dn_layers=dn_l, stop_layer=stop_l, mode="mode1_stopgrad_l10", u_c_by_layer=proj_map, b_align=b_align
    )
    s_soup = SurgicalHostAdapter(
        host.lm, fa_layers=fa_l, dn_layers=dn_l, stop_layer=stop_l, mode="mode1_stopgrad_l10", u_c_by_layer=proj_map, b_align=b_align
    )
    try:
        s_a.remove()
        s_b.remove()
        with torch.no_grad():
            for tag_seed, ad in ((101, s_a), (202, s_b)):
                g = torch.Generator().manual_seed(tag_seed)
                for mdict in (ad.fa_cayleys, ad.dn_cayleys, ad.fa_loras, ad.dn_loras, ad.dn_gate_loras):
                    for mod in mdict.values():
                        for p in mod.parameters():
                            p.copy_(0.05 * torch.randn(p.shape, generator=g, dtype=p.dtype))
                for p in (ad.schur_readout.proj_coord, ad.schur_readout.proj_in):
                    p.copy_(0.05 * torch.randn(p.shape, generator=g, dtype=p.dtype))

        diag = merge_surgical_lie_schur_soup(s_a, s_b, s_soup, w_a=0.4)
        assert diag["max_cayley_unitary_norm_defect"] < 1e-5
        assert diag["schur_readout_reconstruction_err"] < 1e-5

        torch.manual_seed(303)
        for k_str, cay in (*s_soup.fa_cayleys.items(), *s_soup.dn_cayleys.items()):
            m = cay.u_re.shape[0]
            probe = torch.randn((4, 2 * m), dtype=torch.float32)
            q_probe = cay(probe)
            norm_err = float(
                torch.max(torch.abs(q_probe.norm(dim=-1) - probe.norm(dim=-1)) / probe.norm(dim=-1).clamp_min(1e-6)).item()
            )
            assert norm_err < 1e-5
            u_c = proj_map.get(int(k_str))
            if u_c is not None and u_c.shape[0] == m:
                u, v = cay._projected_uv()
                u_c128 = u_c.to(torch.complex128)
                assert float(torch.max(torch.abs(u_c128.conj().T @ u.to(torch.complex128))).item()) < 1e-5
                assert float(torch.max(torch.abs(u_c128.conj().T @ v.to(torch.complex128))).item()) < 1e-5

        test_x = torch.randn((2, 4, host.hidden_size))
        u_c0 = proj_map[fa_l[0]]
        if u_c0.shape[0] * 2 == host.hidden_size:
            res_null = apply_complex_nullspace_fast(test_x, u_c0)
            res_c = torch.complex(res_null[..., 0::2], res_null[..., 1::2]).to(torch.complex64)
            uc64 = u_c0.to(torch.complex64)
            assert float(torch.max(torch.abs(res_c @ uc64.conj())).item()) < 1e-5

        pure_conn = Stage4MatchedConnector("equivariant_host_pure", enc.table, hidden_size=host.hidden_size)
        aud = audit_pure_and_mix_kl(host, pure_conn, s_soup, is_gated_surgical=True)
        assert aud["passes_pure_kl_zero"] is True
        assert aud["pure_kl_max"] == 0.0
        assert aud["pure_max_logit_diff"] == 0.0
    finally:
        s_soup.remove()


def test_real_sdr_gpu_and_stage6_active_eig_mock_e2e(tmp_path: Path):
    run_real_sdr_cpu_pipeline(
        REPO_ROOT,
        tmp_path / "raw",
        tmp_path / "cache",
        mock_synthetic_for_test=True,
        seed=202611,
    )
    ckpt_dir = tmp_path / "ckpts"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    kfac_path = tmp_path / "kfac_mock.pt"
    host = QwenCacheHost(mock_tiny_for_test=True)

    sdr_rep = run_stage4_real_sdr_gpu(
        REPO_ROOT,
        tmp_path / "cache",
        tmp_path / "out",
        ckpt_dir,
        kfac_path,
        seed=202611,
        steps_adapt=2,
        steps_soup=2,
        batch_size=4,
        micro_batch_size=2,
        eval_bs=16,
        mock_tiny_for_test=True,
        host=host,
        cache_dir=tmp_path / "real_cache",
    )
    assert (tmp_path / "real_cache" / "stage4_real_pure_c16_s202611.pt").exists()
    sub2 = sdr_rep["sub2_adapted_c16_vs_4bit"]
    sub3 = sdr_rep["sub3_cross_domain_lie_schur_soup"]
    assert sub3["surgical_lie_schur_soup"]["audit"]["passes_pure_kl_zero"] is True

    bw_models = (
        sub2["equivariant_host_pure_c16"],
        sub2["equivariant_host_pure_s17_4bit"],
        sub2["equivariant_physical_residual_c16"],
        sub3["surgical_lie_schur_soup"],
    )
    for model_entry in bw_models:
        bw = model_entry["blackwell_staircase"]
        for dom_k in ("pluto_test", "cttc_in_domain_test", "cttc_unseen_holdout_f708_snr10"):
            for tier_k in ("q2_none", "q2_one", "q2_two"):
                assert tier_k in bw[dom_k]

    assert Stage6ActiveHostAdapter(hidden_size=2560).param_count() <= 620_000

    layers = _get_decoder_layers(host.lm)
    n_l = len(layers)
    fa_l = [i for i in QWEN4B_FA_LAYERS if i < n_l] or ([n_l - 1] if n_l >= 1 else [])
    dn_l = [i for i in QWEN4B_DN_LAYERS if i < n_l]
    proj_map = collect_kfac_complex_projectors(host, sorted(set(fa_l + dn_l)), k_c=4)
    surg_frozen = SurgicalHostAdapter(
        host.lm,
        fa_layers=fa_l,
        dn_layers=dn_l,
        stop_layer=-1,
        mode="mode1_stopgrad_l10",
        u_c_by_layer=proj_map,
        b_align=calibrate_b_align_from_host(host),
    ).to(host.device)
    surg_frozen.requires_grad_(False)
    surg_frozen.eval()
    try:
        head_probe = Stage6ActiveHostAdapter(hidden_size=host.hidden_size).to(host.device)
        head_probe.train()
        logits_probe = head_probe.forward_states(
            host,
            torch.zeros((2, 5), dtype=torch.long),
            torch.zeros((2,), dtype=torch.long),
            torch.tensor([1, 2, 3, 4], dtype=torch.long),
            bits=16,
            surgical_adapter=surg_frozen,
        )
        logits_probe.square().mean().backward()
        assert head_probe.feat_proj.weight.grad is not None
        assert float(head_probe.feat_proj.weight.grad.abs().max().item()) > 0.0
    finally:
        surg_frozen.remove()

    s6_rep = run_stage6_active_eig_gpu(
        REPO_ROOT,
        tmp_path / "out",
        ckpt_dir,
        kfac_path,
        seed=202611,
        steps=2,
        batch_size=4,
        eval_bs=16,
        mock_tiny_for_test=True,
        host=host,
        cache_dir=tmp_path / "s6_cache",
    )
    expected_models = {
        "physical_c16_reference",
        "host_frozen_c16",
        "host_frozen_s17_4bit",
        "host_surgical_c16",
        "host_lemmo_c16",
    }
    assert set(s6_rep["models"].keys()) == expected_models
    for mname in expected_models:
        for fam_res in s6_rep["models"][mname]["per_family"]:
            assert fam_res["m3_reveal_max_abs_err"] < 2e-12
    assert s6_rep["models"]["host_surgical_c16"]["pure_kl_audit"]["passes_pure_kl_zero"] is True
    assert s6_rep["models"]["host_surgical_c16"]["under_tier_p1_cap"] is True
