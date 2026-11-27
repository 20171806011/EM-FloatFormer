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


def test_cn_sft_gamma_f_and_3way_bus():
    from research.stage4_real_and_surgical.cn_sft_adapter import (
        build_three_way_orthogonal_bus,
        calibrate_b_align_gamma_f,
        verify_rmsnorm_gamma,
    )

    class Qwen3_5RMSNorm(torch.nn.Module):
        def __init__(self, d: int, w: torch.Tensor):
            super().__init__()
            self.weight = torch.nn.Parameter(w.clone())
            self.eps = 1e-6

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            xf = x.float()
            rms = torch.sqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.eps)
            return ((xf / rms) * (1.0 + self.weight.float())).to(x.dtype)

    class Qwen2RMSNorm(torch.nn.Module):
        def __init__(self, d: int, w: torch.Tensor):
            super().__init__()
            self.weight = torch.nn.Parameter(w.clone())
            self.variance_epsilon = 1e-6

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            xf = x.float()
            rms = torch.sqrt(xf.pow(2).mean(dim=-1, keepdim=True) + self.variance_epsilon)
            return ((xf / rms) * self.weight.float()).to(x.dtype)

    torch.manual_seed(202611)
    host = QwenCacheHost(mock_tiny_for_test=True)
    d = host.hidden_size
    n_zero = Qwen3_5RMSNorm(d, 0.25 * torch.randn(d))
    n_one = Qwen2RMSNorm(d, 1.0 + 0.25 * torch.randn(d))
    assert verify_rmsnorm_gamma(n_zero) <= 1e-4
    assert verify_rmsnorm_gamma(n_one) <= 1e-4

    # Replace mock final norm with strongly non-uniform zero-centered Qwen3_5RMSNorm
    host.lm.model.norm = Qwen3_5RMSNorm(d, 0.35 * torch.randn(d))
    ba = calibrate_b_align_gamma_f(host)
    assert ba["fixed_err"] < 1e-4
    assert ba["old_tilt_err"] > ba["fixed_err"]

    x_prose = torch.randn(128, d)
    u_c_d = compute_complex_nullspace_projector(x_prose.T @ x_prose, k_c=8)["U_c"]
    e_bus, b_stats = build_three_way_orthogonal_bus(u_c_d, ba["m_eff"], bus_dim=22, seed=202611)
    assert e_bus.shape == (d, 22)
    assert b_stats["cand_bleed_max"] < 1e-4
    assert b_stats["prose_bleed_max"] < 1e-4
    assert float((e_bus.T @ e_bus - torch.eye(22)).abs().max().item()) < 1e-5


def test_cn_sft_speccayley_headspace_and_nope_lora():
    from research.stage4_real_and_surgical.cn_sft_adapter import FoldedNoPELoRADelta, SpecCayley

    torch.manual_seed(7)
    d_o = 32
    x_head = torch.randn(128, d_o)
    u_c_o = compute_complex_nullspace_projector(x_head.T @ x_head, k_c=4)["U_c"]

    cay = SpecCayley(dim=d_o, r_c=3, r_s=2, u_c=u_c_o, sigma_max=2.0)
    m_init = cay.materialize_matrix()
    eye_m = torch.eye(d_o // 2, dtype=torch.complex64)
    assert float((m_init - eye_m).abs().max().item()) < 1e-6

    with torch.no_grad():
        cay.v_re.normal_(std=0.4)
        cay.v_im.normal_(std=0.4)
        cay.sigma_raw.normal_(std=1.5)
    m_mat = cay.materialize_matrix()
    # Exact identity on u_c_o
    assert float((m_mat @ u_c_o.to(torch.complex64) - u_c_o.to(torch.complex64)).abs().max().item()) < 1e-5
    # Singular values bounded strictly inside [1/sigma_max, sigma_max] = [0.5, 2.0]
    svs = torch.linalg.svdvals(m_mat)
    assert float(svs.min().item()) >= 0.5 - 1e-4
    assert float(svs.max().item()) <= 2.0 + 1e-4

    # Forward matches materialized matrix and commutes with J_d
    probe = torch.randn(6, d_o)
    y_fast = cay(probe)
    z_p = torch.complex(probe[:, 0::2], probe[:, 1::2]) @ m_mat.T
    y_exp = torch.empty_like(probe)
    y_exp[:, 0::2] = z_p.real
    y_exp[:, 1::2] = z_p.imag
    assert float((y_fast - y_exp).abs().max().item()) < 1e-5
    assert float((apply_j_d(cay(probe)) - cay(apply_j_d(probe))).abs().max().item()) < 1e-5

    # Strict NoPE + Gate masking on interleaved q_proj + P_perp on u_c_d
    x_d = torch.randn(128, d_o)
    u_c_d = compute_complex_nullspace_projector(x_d.T @ x_d, k_c=4)["U_c"]
    lora = FoldedNoPELoRADelta(
        in_features=d_o,
        out_features=64,
        r=4,
        u_c=u_c_d,
        head_dim=16,
        rot_dim=4,
        interleaved_gate=True,
    )
    with torch.no_grad():
        lora.lora_b.fill_(1.0)
        x_in = torch.randn(2, 5, d_o)
        base = torch.zeros(2, 5, 64)
        full = lora.add_into_output(base, lora(x_in)).view(2, 5, 2, 32)
        assert float(full[..., :4].abs().max().item()) == 0.0      # RoPE [0:rot_dim] strictly untouched
        assert float(full[..., 16:].abs().max().item()) == 0.0     # Gate [head_dim:2*head_dim] strictly untouched
        assert float(full[..., 4:16].abs().max().item()) > 0.0     # NoPE [rot_dim:head_dim] active
        # Input lying purely inside prose u_c_d yields exact zero output
        uc_vec = u_c_d[:, :2].T
        prose_in = torch.zeros(2, d_o)
        prose_in[:, 0::2] = uc_vec.real
        prose_in[:, 1::2] = uc_vec.imag
        assert float(lora(prose_in).abs().max().item()) < 1e-5


def test_cn_sft_circuit_a_norm_invariance_and_bf16_safety():
    from research.stage4_real_and_surgical.cn_sft_adapter import (
        CompiledN1CircuitAOverlay,
        F_REAL_4X3,
        ShadowGDNBallastBridge,
        build_regime2_separate_carrier,
        build_three_way_orthogonal_bus,
    )

    torch.manual_seed(99)
    d = 512
    cand_w = torch.linalg.qr(torch.randn(d, 4))[0].T.contiguous()  # (4, 512) orthonormal rows
    gamma_f = 0.8 + 0.6 * torch.rand(d)
    m_cand = cand_w - cand_w.mean(dim=0, keepdim=True)
    m_eff = m_cand * gamma_f.unsqueeze(0)
    b_align = (torch.linalg.pinv(m_eff.to(torch.float64)) @ F_REAL_4X3.to(torch.float64)).float()
    e_bus, _ = build_three_way_orthogonal_bus(None, m_eff, bus_dim=22, seed=99)

    bridge = ShadowGDNBallastBridge(e_bus, s_med_0=1.0, s_med_1=1.0, s_med_2=1.0)
    g0 = 0.7 + 0.7 * torch.rand(d)
    g1 = 0.7 + 0.7 * torch.rand(d)
    g2 = 0.7 + 0.7 * torch.rand(d)
    ca = CompiledN1CircuitAOverlay(
        e_bus=e_bus, b_align=b_align, m_eff=m_eff, gamma_0=g0, gamma_1=g1, gamma_2=g2, s_med_0=1.0, s_med_1=1.0, s_med_2=1.0
    )
    assert bool((ca.ws0 >= 0.0).all() and (ca.ws1 >= 0.0).all() and (ca.ws2 >= 0.0).all())

    # Test ShadowGDN KV continuity + zero output without <PHASE>
    m_mask = torch.zeros((2, 8), dtype=torch.bool)
    m_mask[:, :6] = True
    no_phase_h = torch.zeros((2, 8, d), dtype=torch.float32)
    no_phase_h[:, 0:2, :] = e_bus[:, 0:1].T
    out_np = bridge.forward_pass(no_phase_h, m_mask)
    assert float((out_np @ e_bus[:, 6:14]).abs().max().item()) < 1e-6

    # Inject real carrier + verify exact Fourier logits invariant across arbitrary residual scales
    enc, _ = load_s17_encoder(REPO_ROOT)
    kern = PhysicalKernelS17(enc.table)
    words = torch.tensor([[1, 2], [3, 0]], dtype=torch.long)
    omega = 0.35
    h_step, ell_step, t_p = 2.0, 2.0, 4.0
    c_w = math.cos(omega * (h_step + ell_step))
    s_w = math.sin(omega * (h_step + ell_step))
    ps = torch.tensor([[math.cos(omega * h_step), math.sin(omega * h_step), c_w, s_w, c_w, s_w]] * 2, dtype=torch.float32)
    meta = torch.tensor([[h_step, ell_step, t_p, omega, 0.10, 2.0, 0.0]] * 2, dtype=torch.float32)
    carrier = build_regime2_separate_carrier(kern.prior_canon, words, "Q3_repeat_x1", ps, meta)

    entry = torch.zeros((2, 8, d), dtype=torch.float32)
    entry[:, 0:2, :] = carrier["w1_vec"] @ e_bus[:, :16].T
    entry[:, 2:4, :] = carrier["w2_vec"] @ e_bus[:, :16].T
    entry[:, 4:6, :] = carrier["ph_vec"] @ e_bus[:, :16].T
    br_p1 = bridge.forward_pass(entry, m_mask)
    br_p2 = bridge.forward_pass(torch.zeros((8, 3, d), dtype=torch.float32), m_mask)
    assert float((br_p2[0, -1] - br_p1[0, -1]).abs().max().item()) < 1e-4

    def _rms_norm(x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(dim=-1, keepdim=True) + 1e-6)
        return (x / rms) * g

    # Build pure orthogonal background unit directions (outside m_eff AND e_bus)
    blk = torch.cat((e_bus, m_eff.T), dim=1)
    q_blk, _ = torch.linalg.qr(blk, mode="reduced")
    bg_raw = torch.randn(d, 4)
    bg_raw = bg_raw - q_blk @ (q_blk.T @ bg_raw)
    bg_q = torch.linalg.qr(bg_raw, mode="reduced")[0] * math.sqrt(float(d))
    bg0_u, bg1_u, bg2_u = bg_q[:, 0], bg_q[:, 1], bg_q[:, 2]

    def _set_orth_rms(x_bus: torch.Tensor, bg_dir: torch.Tensor, target_rms: float) -> torch.Tensor:
        cur_e = x_bus.pow(2).mean(dim=-1, keepdim=True)
        need = (target_rms ** 2 - cur_e).clamp_min(1e-6).sqrt()
        return x_bus + bg_dir.view(1, 1, d) * need

    gate = torch.zeros((2, 8), dtype=torch.float32)
    gate[:, -1] = 1.0
    logits_by_scale = []
    for s0, s1, s2 in ((0.7, 1.4, 0.8), (1.8, 0.6, 1.5)):
        h0 = _set_orth_rms(br_p1, bg0_u, s0)
        d0 = ca.l0_delta(_rms_norm(h0, g0), gate)
        h1_no_bg = h0 - ((h0 @ bg0_u.view(d, 1)) / float(d)) * bg0_u.view(1, 1, d) + d0
        h1 = _set_orth_rms(h1_no_bg, bg1_u, s1)
        d1 = ca.l1_delta(_rms_norm(h1, g1), gate)
        # Inject massive random cand-aligned host bias into both x_2 and mlp_out!
        bias_vec = (3.0 * m_eff[0] - 2.0 * m_eff[2]).view(1, 1, d)
        mlp_bias = (2.5 * m_eff[1] - 1.5 * m_eff[3]).view(1, 1, d).expand(2, 8, d)
        h2_no_bg = h1 - ((h1 @ bg1_u.view(d, 1)) / float(d)) * bg1_u.view(1, 1, d) + d1 + bias_vec
        h2 = _set_orth_rms(h2_no_bg, bg2_u, s2)
        d2 = ca.l2_delta(_rms_norm(h2, g2), gate, mlp_out=mlp_bias)
        h_fin = h2 + mlp_bias + d2
        # Normalize h_fin by exact final RMS & apply gamma_f -> cand_w logits
        log_out = (_rms_norm(h_fin[:, -1, :], gamma_f) @ cand_w.T)
        log_out = log_out - log_out.mean(dim=-1, keepdim=True)
        logits_by_scale.append(log_out)

    rel_diff = float((logits_by_scale[0] - logits_by_scale[1]).norm().item() / logits_by_scale[0].norm().clamp_min(1e-6).item())
    assert rel_diff < 0.03, f"Non-uniform RMSNorm scale invariance failed: rel_diff={rel_diff}"


def test_cn_sft_exact_4b_param_budget():
    from types import SimpleNamespace
    from research.stage4_real_and_surgical.cn_sft_adapter import CNSFTHostAdapter, QWEN4B_MLP_TUNE_LAYERS, F_REAL_4X3

    class TinyZeroNorm(torch.nn.Module):
        def __init__(self, d: int):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.zeros(d))
            self.eps = 1e-6

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return x

    class DummyProj(torch.nn.Module):
        def __init__(self, in_f: int, out_f: int):
            super().__init__()
            self.in_features = in_f
            self.out_features = out_f

    # Lightweight 32-layer Qwen3.5-4B geometry skeleton (< 100 KB RAM) for exact parameter audit
    d = 2560
    fa_set = set(QWEN4B_FA_LAYERS)
    dn_set = set(QWEN4B_DN_LAYERS)
    layers = torch.nn.ModuleList()
    for idx in range(32):
        blk = torch.nn.Module()
        blk.post_attention_layernorm = TinyZeroNorm(d)
        blk.mlp = torch.nn.Identity()
        if idx in fa_set:
            attn_m = torch.nn.Module()
            attn_m.head_dim = 256
            attn_m.num_heads = 16
            attn_m.num_key_value_groups = 4
            attn_m.o_proj = DummyProj(4096, d)
            attn_m.q_proj = DummyProj(d, 8192)
            attn_m.k_proj = DummyProj(d, 1024)
            attn_m.v_proj = DummyProj(d, 1024)
            blk.self_attn = attn_m
        if idx in dn_set:
            dn_m = torch.nn.Module()
            dn_m.key_dim = 2048
            dn_m.value_dim = 4096
            dn_m.out_proj = DummyProj(4096, d)
            dn_m.in_proj_qkv = DummyProj(d, 8192)
            dn_m.in_proj_a = DummyProj(d, 32)
            dn_m.in_proj_b = DummyProj(d, 32)
            blk.linear_attn = dn_m
        layers.append(blk)

    fake_lm = torch.nn.Module()
    fake_lm.model = torch.nn.Module()
    fake_lm.model.layers = layers
    fake_lm.config = SimpleNamespace(hidden_size=d, head_dim=256, num_attention_heads=16, partial_rotary_factor=0.25)

    u_c_d = {i: torch.zeros((d // 2, 16), dtype=torch.complex64) for i in (*QWEN4B_FA_LAYERS, *QWEN4B_DN_LAYERS)}
    u_c_o = {i: torch.zeros((2048, 16), dtype=torch.complex64) for i in (*QWEN4B_FA_LAYERS, *QWEN4B_DN_LAYERS)}
    u_c_mlp = {i: torch.zeros((d // 2, 16), dtype=torch.complex64) for i in QWEN4B_MLP_TUNE_LAYERS}
    adapter = CNSFTHostAdapter(
        fake_lm,
        u_c_d=u_c_d,
        u_c_o=u_c_o,
        u_c_mlp=u_c_mlp,
        u_c_bus=None,
        b_align_info={"b_align": torch.zeros((d, 3)), "m_eff": torch.zeros((4, d))},
        e_bus=torch.zeros((d, 22)),
        tail_rms_medians={29: 1.0, 30: 1.0, 31: 1.0, "final": 1.0},
        rot_dim_override=64,
    )
    try:
        enc, _ = load_s17_encoder(REPO_ROOT)
        conn_4b = Stage4MatchedConnector("equivariant_host_pure", enc.table, hidden_size=2560)
        cum_trainable = conn_4b.param_count() + adapter.param_count()
        cum_with_n1_dense = cum_trainable + adapter.circuit_a.n1_parameter_count_dense_equiv(2560)
        assert cum_trainable == 1_666_839
        assert cum_trainable <= 1_850_000
        assert cum_with_n1_dense == 2_089_239
        assert cum_with_n1_dense <= TIER_P1_MAX_PARAMS
    finally:
        adapter.remove()


def test_cn_sft_mock_probe_e2e(tmp_path: Path):
    from research.stage4_real_and_surgical.cn_sft_probe import run_cn_sft_4b_probe

    out_dir = tmp_path / "cn_out"
    ckpt_dir = tmp_path / "cn_ckpts"
    rep = run_cn_sft_4b_probe(
        REPO_ROOT,
        out_dir,
        ckpt_dir,
        seed=202611,
        steps=2,
        batch_size=8,
        micro_batch_size=4,
        g1_blocks=1,
        eval_bs=16,
        mock_tiny_for_test=True,
    )
    assert rep["early_aborted"] is False
    assert rep["U0"]["audit"]["passes_pure_kl_zero"] is True
    assert rep["U0"]["audit"]["pure_kl_max"] == 0.0
    assert rep["U0"]["audit"]["pure_max_logit_diff"] == 0.0
    assert rep["U1"]["audit"]["passes_pure_kl_zero"] is True
    assert rep["U1"]["audit"]["pure_kl_max"] == 0.0
    assert rep["U1"]["host_sha256"] == rep["host_sha256"]
    assert "U3_gate_off_audit" in rep
    assert rep["U2"]["skipped"] is False
    assert (ckpt_dir / "stage4_cn_sft_u1_s202611.pt").exists()
    assert (out_dir / "cn_sft_4b_probe_results.json").exists()


def test_cn_sft_4layer_end_to_end_residual_and_active_mix_kl():
    from types import SimpleNamespace
    from transformers import Qwen2Config, Qwen2ForCausalLM
    from research.stage4_real_and_surgical.cn_sft_adapter import (
        CNSFTHostAdapter,
        F_REAL_4X3,
        build_three_way_orthogonal_bus,
        calibrate_b_align_gamma_f,
        collect_dual_space_kfac_projectors,
    )

    torch.manual_seed(202611)
    cfg = Qwen2Config(
        vocab_size=256,
        hidden_size=256,
        intermediate_size=512,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=256,
    )
    lm = Qwen2ForCausalLM(cfg).eval()
    # Embed scale helper using vocab tokens so K-FAC & eval share identical RMS distribution
    prompts = [f"calibration prompt block number {k} end" for k in range(12)]
    ids_list = [[(ord(ch) % 240) + 8 for ch in pr] for pr in prompts]

    class _MiniHost:
        def __init__(self, model):
            self.lm = model
            self.device = torch.device("cpu")
            self.dtype = torch.float32
            self.hidden_size = 256
            self.tokenizer = SimpleNamespace(
                encode=lambda s, add_special_tokens=False: [(ord(c) % 240) + 8 for c in s][:16]
            )
            self.text_config = cfg

        def _lm_forward(self, **kwargs):
            return self.lm(**kwargs)

    host = _MiniHost(lm)
    kfac = collect_dual_space_kfac_projectors(
        host, fa_layers=[1, 3], dn_layers=[], mlp_layers=[1], prompts=prompts, stop_layer=0
    )
    ba = calibrate_b_align_gamma_f(host)
    eb, _ = build_three_way_orthogonal_bus(kfac["u_c_bus"], ba["m_eff"], bus_dim=22, seed=202611)
    ad = CNSFTHostAdapter(
        lm,
        kfac["u_c_d"],
        kfac["u_c_o"],
        kfac["u_c_mlp"],
        kfac["u_c_bus"],
        ba,
        eb,
        kfac["tail_rms_medians"],
        fa_layers=[1, 3],
        dn_layers=[],
        mlp_tune_layers=[1],
        stop_layer=0,
        arm_mode="u0_zeroshot",
        regime=2,
    )
    try:
        emb = lm.get_input_embeddings()
        tok_seq = torch.tensor([ids_list[0][:10]], dtype=torch.long)
        x = emb(tok_seq)
        attn = torch.ones((1, x.shape[1]), dtype=torch.long)
        m_mask = torch.zeros((1, x.shape[1]), dtype=torch.bool)
        m_mask[:, 1:7] = True

        with torch.no_grad():
            base_logits = lm(inputs_embeds=x, attention_mask=attn, use_cache=False).logits[:, -1, ba["digit_ids"]]

        # 1. With g_em == 0: exact bitwise zero diff
        q1_w1 = torch.zeros((1, 16))
        q1_w1[:, 0], q1_w1[:, 1], q1_w1[:, 2], q1_w1[:, 13] = 0.65, -0.35, 0.20, 1.0
        q1_w2 = torch.zeros((1, 16))
        q1_w2[:, 3], q1_w2[:, 4], q1_w2[:, 5], q1_w2[:, 14] = 0.50, 0.40, -0.15, 1.0
        q1_ph = torch.zeros((1, 16))
        q1_ph[:, 6], q1_ph[:, 7], q1_ph[:, 8], q1_ph[:, 9], q1_ph[:, 15] = 0.70, -0.30, 0.40, 1.0, 1.0
        q1_car = {"w1_vec": q1_w1, "w2_vec": q1_w2, "ph_vec": q1_ph, "shift": torch.zeros((1,), dtype=torch.long)}
        with torch.no_grad(), ad.activate(g_em=0, m_acc_mask=m_mask, carrier=q1_car, t_div=0, task_id=0):
            pure_logits = lm(inputs_embeds=x, attention_mask=attn, use_cache=False).logits[:, -1, ba["digit_ids"]]
        assert float((pure_logits - base_logits).abs().max().item()) == 0.0

        # 2. With qgate = 0.0 AND active Q1 carrier attached: exact zero active_carrier_mix_kl
        with torch.no_grad():
            ad.schur_readout.qgate_w.zero_()
            ad.schur_readout.qgate_b.fill_(-12.0)
            with ad.activate(g_em=1, m_acc_mask=m_mask, carrier=q1_car, t_div=0, task_id=0):
                mix_act_logits = lm(inputs_embeds=x, attention_mask=attn, use_cache=False).logits[:, -1, ba["digit_ids"]]
        assert float((mix_act_logits - base_logits).abs().max().item()) == 0.0

        # 3. With qgate = 1.0 on Q1, Q2, and Q3 carriers: bus survives 3 tail layers and matches exact F_real theory
        with torch.no_grad():
            ad.schur_readout.qgate_b.fill_(12.0)

        cases = []
        # Q1 case
        t2r = q1_w2[:, 3] * q1_ph[:, 6] - q1_w2[:, 4] * q1_ph[:, 7]
        t2i = q1_w2[:, 3] * q1_ph[:, 7] + q1_w2[:, 4] * q1_ph[:, 6]
        t22 = q1_w2[:, 5] * q1_ph[:, 8]
        p0 = (q1_w1[:, 0] * t2r - q1_w1[:, 1] * t2i) / math.sqrt(2.0)
        p1 = (q1_w1[:, 0] * t2i + q1_w1[:, 1] * t2r) / math.sqrt(2.0)
        p2 = 0.5 * (q1_w1[:, 2] * t22)
        cases.append((0, q1_car, torch.stack((p0, p1, p2), dim=-1)))

        # Q2 case
        q2_w1 = torch.zeros((1, 16))
        q2_w1[:, 0], q2_w1[:, 1], q2_w1[:, 2], q2_w1[:, 13] = 0.72, 0.25, -0.18, 1.0
        q2_w2 = torch.zeros((1, 16))
        q2_w2[:, 14] = 1.0
        q2_ph = torch.zeros((1, 16))
        q2_ph[:, 9], q2_ph[:, 10], q2_ph[:, 11], q2_ph[:, 12], q2_ph[:, 15] = 1.0, 0.55, -0.45, 0.30, 1.0
        q2_car = {"w1_vec": q2_w1, "w2_vec": q2_w2, "ph_vec": q2_ph, "shift": torch.zeros((1,), dtype=torch.long)}
        cases.append((
            1,
            q2_car,
            torch.stack((
                (q2_w1[:, 0] * q2_ph[:, 10] - q2_w1[:, 1] * q2_ph[:, 11]) / math.sqrt(2.0),
                (q2_w1[:, 0] * q2_ph[:, 11] + q2_w1[:, 1] * q2_ph[:, 10]) / math.sqrt(2.0),
                0.5 * (q2_w1[:, 2] * q2_ph[:, 12]),
            ), dim=-1),
        ))

        # Q3 case
        q3_w1 = torch.zeros((1, 16))
        q3_w1[:, 0], q3_w1[:, 1], q3_w1[:, 2], q3_w1[:, 13] = -0.60, 0.42, 0.30, 1.0
        q3_w2 = torch.zeros((1, 16))
        q3_w2[:, 14] = 1.0
        q3_ph = torch.zeros((1, 16))
        q3_ph[:, 9], q3_ph[:, 10], q3_ph[:, 11], q3_ph[:, 12], q3_ph[:, 15] = 1.0, -0.52, -0.48, 0.25, 1.0
        q3_car = {"w1_vec": q3_w1, "w2_vec": q3_w2, "ph_vec": q3_ph, "shift": torch.zeros((1,), dtype=torch.long)}
        cases.append((
            2,
            q3_car,
            torch.stack((
                (q3_w1[:, 0] * q3_ph[:, 10] - q3_w1[:, 1] * q3_ph[:, 11]) / math.sqrt(2.0),
                (q3_w1[:, 0] * q3_ph[:, 11] + q3_w1[:, 1] * q3_ph[:, 10]) / math.sqrt(2.0),
                0.5 * (q3_w1[:, 2] * q3_ph[:, 12]),
            ), dim=-1),
        ))

        for tid, car, expected_canon in cases:
            with torch.no_grad(), ad.activate(g_em=1, m_acc_mask=m_mask, carrier=car, t_div=0, task_id=tid):
                out_logits = lm(inputs_embeds=x, attention_mask=attn, use_cache=False).logits[:, -1, ba["digit_ids"]]
            diag = ad.circuit_a._last_diag
            assert 0.4 <= diag["rho_2"] <= 2.5, f"tid={tid} rho_2 out of range: {diag}"
            assert 0.2 <= diag["eta_2"] <= 3.0, f"tid={tid} eta_2 out of range: {diag}"
            centered_out = out_logits - out_logits.mean(dim=-1, keepdim=True)
            expected_centered = 3.55 * (expected_canon @ F_REAL_4X3.T)
            err = float((centered_out - expected_centered).abs().max().item())
            assert err < 0.25, f"tid={tid} end-to-end logit err={err:.4f} >= 0.25"
            assert int(centered_out.argmax(dim=-1).item()) == int(expected_centered.argmax(dim=-1).item())
    finally:
        ad.remove()

