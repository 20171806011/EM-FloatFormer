"""Unit & invariant tests for the CTTC real-data QPSK empirical pipeline."""
from __future__ import annotations

import math
import numpy as np

from research.realdata_cttc_qpsk.adapter import (
    HOLDOUT_IDX,
    KNOWN_IDX,
    PAD_SYMS,
    PAYLOAD_IDX,
    PILOT_IDX,
    PKT_LEN,
    SPLIT_SRC_RANGES,
    SPS_NOM,
    SYNC_IDX,
    PacketSample,
    make_rrc_taps,
    verify_zero_leakage,
)
from research.realdata_cttc_qpsk.baselines import ClassicalConfig, run_classical_receiver
from research.realdata_cttc_qpsk.evaluator import (
    QPSK_CONSTELLATION,
    PacketEvalRecord,
    bits_to_qpsk,
    compute_ece,
    compute_holdout_channel_metrics,
    compute_segment_metrics,
    mixture_qpsk_posterior,
    paired_cluster_inference,
    qpsk_bit_llrs_gaussian,
    qpsk_to_bits,
)
from research.realdata_cttc_qpsk.experiments import (
    CandidateBank,
    decode_from_candidate_bank,
)


def test_qpsk_gray_roundtrip_and_orthogonality():
    rng = np.random.default_rng(12345)
    bits = rng.integers(0, 2, size=512, dtype=np.uint8)
    syms = bits_to_qpsk(bits)
    assert np.allclose(np.abs(syms), 1.0, atol=1e-12)
    bits_back = qpsk_to_bits(syms)
    assert np.array_equal(bits, bits_back)


def test_packet_partition_disjoint_and_complete():
    assert len(SYNC_IDX) == 6
    assert len(PILOT_IDX) == 4
    assert len(KNOWN_IDX) == 10
    assert len(HOLDOUT_IDX) == 16
    assert len(PAYLOAD_IDX) == 230
    s_known = set(KNOWN_IDX.tolist())
    s_hold = set(HOLDOUT_IDX.tolist())
    s_pay = set(PAYLOAD_IDX.tolist())
    assert len(s_known & s_hold) == 0
    assert len(s_known & s_pay) == 0
    assert len(s_hold & s_pay) == 0
    assert (s_known | s_hold | s_pay) == set(range(PKT_LEN))


def test_split_src_ranges_strictly_disjoint_with_guard_band():
    intervals = []
    for sp, roles in SPLIT_SRC_RANGES.items():
        for role, (lo, hi) in roles.items():
            assert lo < hi
            intervals.append((lo, hi, f"{sp}_{role}"))
    intervals.sort()
    for i in range(len(intervals) - 1):
        lo1, hi1, n1 = intervals[i]
        lo2, hi2, n2 = intervals[i + 1]
        assert lo2 - hi1 >= 4096, f"Guard band < 4096 between {n1} and {n2}: {lo2 - hi1}"
    # Circular wrap-around guard band
    assert 500_000 - intervals[-1][1] + intervals[0][0] >= 4096


def test_mixture_qpsk_posterior_exact_single_component_and_multimodal():
    rng = np.random.default_rng(7)
    y = rng.standard_normal(32) + 1j * rng.standard_normal(32)
    h = np.full(32, 0.7 - 0.5j, dtype=np.complex128)
    nv = np.full(32, 0.25, dtype=np.float64)

    # 1) Single-component mixture must match single-Gaussian LLR to machine precision
    llrs_g = qpsk_bit_llrs_gaussian(y, h, nv)
    _, llrs_m, _ = mixture_qpsk_posterior(y, h[None, :], nv[None, :], np.array([1.0]))
    assert np.allclose(llrs_g, llrs_m, atol=1e-11)

    # 2) Two-component phase-ambiguous mixture: h0=+1, h1=-2, weights=(0.6, 0.4) (posterior_fusion §5.1 analogue)
    # At y = +2+2j (scaled by 1/sqrt2), amplitude=2 uniquely identifies branch 1 (h=-2), so x = -y/2 = (-1-1j)/sqrt2 -> bits (1,1)!
    # Whereas linear posterior mean h_mix = 0.6*(1) + 0.4*(-2) = -0.2 would STILL flip sign or lose amplitude cue!
    y_test = np.array([(-2.0) * QPSK_CONSTELLATION[3]], dtype=np.complex128)  # true x = (-1-j)/s2 (bits 1,1), h=-2 -> y = +sqrt2*(1+j)
    h_two = np.array([[1.0 + 0j], [-2.0 + 0j]], dtype=np.complex128)
    nv_two = np.array([[0.02], [0.02]], dtype=np.float64)
    w_two = np.array([0.6, 0.4], dtype=np.float64)
    sym_mix, llr_mix, _ = mixture_qpsk_posterior(y_test, h_two, nv_two, w_two)
    assert np.allclose(sym_mix[0], QPSK_CONSTELLATION[3])
    assert llr_mix[0] > 5.0 and llr_mix[1] > 5.0  # confidently predicts bits (1, 1)

    # Compare hard MAP branch (selects h0=+1, predicts x = (+1+j)/s2 -> bits 0,0 -> WRONG!)
    bank = CandidateBank(
        y_sym=np.repeat(y_test, PKT_LEN),
        timing_rx=0.0,
        h_cands=np.repeat(h_two, PKT_LEN, axis=1),
        nv_cands=np.repeat(nv_two, PKT_LEN, axis=1),
        log_ev=np.log(w_two),
        cand_cfos=np.array([0.0, 0.0]),
        cand_feats=np.zeros((2, 12)),
        cand_names=("b0", "b1"),
        build_ms=0.0,
    )
    out_hard = decode_from_candidate_bank(bank, w_two, mode="hard_select")
    out_sym_mix = decode_from_candidate_bank(bank, w_two, mode="sym_mixture")
    out_joint_mix = decode_from_candidate_bank(bank, w_two, mode="mixture_likelihood", block_temp=0.5)
    assert np.allclose(out_hard.hard_syms[0], QPSK_CONSTELLATION[0])        # wrong!
    assert np.allclose(out_sym_mix.hard_syms[0], QPSK_CONSTELLATION[3])     # exact!
    assert np.allclose(out_joint_mix.hard_syms[0], QPSK_CONSTELLATION[3])   # exact!
    # Check total variance identity: v_total = v_within + v_between = 0.02 + (0.6*1.2^2 + 0.4*1.8^2) = 0.02 + 2.16 = 2.18
    assert abs(out_joint_mix.nv_hat[0] - 2.18) < 1e-12


def test_rrc_unit_energy_and_synthetic_classical_receiver():
    h_rrc = make_rrc_taps()
    assert abs(float(np.sum(h_rrc ** 2)) - 1.0) < 1e-12

    # Synthesize a clean 4x oversampled QPSK packet with known CFO = -650 Hz and complex gain h = 0.8*exp(j*0.7)
    rng = np.random.default_rng(999)
    bits = rng.integers(0, 2, size=2 * (PKT_LEN + 2 * PAD_SYMS), dtype=np.uint8)
    syms_pad = bits_to_qpsk(bits)
    # Upsample by 4 and convolve with raised cosine (self-convolved RRC)
    up = np.zeros(len(syms_pad) * SPS_NOM, dtype=np.complex128)
    up[::SPS_NOM] = syms_pad
    rc = np.convolve(h_rrc, h_rrc, mode="full")
    rc /= rc[len(rc) // 2]  # peak = 1 at zero lag
    wf = np.convolve(up, rc, mode="same")

    cfo = -650.0
    h0 = 0.8 * np.exp(1j * 0.7)
    t_rx = (np.arange(len(wf), dtype=np.float64) - PAD_SYMS * SPS_NOM) / 2_000_000.0
    nv = 1e-3  # ~28 dB SNR
    noise = math.sqrt(nv / 2.0) * (rng.standard_normal(len(wf)) + 1j * rng.standard_normal(len(wf)))
    y_iq = h0 * wf * np.exp(2j * math.pi * cfo * t_rx) + noise

    x_true = syms_pad[PAD_SYMS : PAD_SYMS + PKT_LEN]
    pkt = PacketSample(
        unit_id="synth_0",
        capture_id="synth",
        freq_mhz=2450,
        nom_snr_db=20,
        cluster_id="synth__L00",
        loop_id=0,
        split="test",
        role="target",
        rx_start_sample=0,
        frac_rx_offset=0.0,
        src_sym_start=0,
        time_sec=2.0,
        bg_noise_var=nv,
        y_iq_4x=y_iq.astype(np.complex64),
        x_known=x_true[KNOWN_IDX],
        x_all_true=x_true,
        h_offline_ref=h0 * np.exp(2j * math.pi * cfo * np.arange(PKT_LEN) / 500_000.0),
        offline_cfo_hz=cfo,
        offline_sig_pow=abs(h0) ** 2,
    )
    out = run_classical_receiver(pkt, ClassicalConfig(), mode="mmse")
    assert abs(out.est_cfo_hz - cfo) < 60.0
    m = compute_segment_metrics(out.x_eq[PAYLOAD_IDX], x_true[PAYLOAD_IDX], bit_llrs=None, hard_syms_override=out.hard_syms[PAYLOAD_IDX])
    assert m["ser"] == 0.0
    assert m["evm_unbiased_db"] < -20.0


def test_paired_cluster_exact_signflip():
    recs_a = []
    recs_b = []
    for g in range(8):
        recs_a.append(
            PacketEvalRecord(
                unit_id=f"u{g}", capture_id="c0", freq_mhz=2450, nom_snr_db=0,
                cluster_id=f"cl{g}", split="test", method="A", success=True, payload_ser=0.20,
            )
        )
        recs_b.append(
            PacketEvalRecord(
                unit_id=f"u{g}", capture_id="c0", freq_mhz=2450, nom_snr_db=0,
                cluster_id=f"cl{g}", split="test", method="B", success=True, payload_ser=0.10,
            )
        )
    res = paired_cluster_inference(recs_a, recs_b, "payload_ser", group_by="cluster_id")
    assert res["n_clusters"] == 8
    assert res["wins"] == 8 and res["losses"] == 0
    assert abs(res["mean_improvement"] - 0.10) < 1e-12
    assert res["signflip_mode"] == "exact_256"
    assert abs(res["signflip_p_one_sided"] - (1.0 / 256.0)) < 1e-12
    assert abs(res["signflip_p_two_sided"] - (2.0 / 256.0)) < 1e-12
    assert compute_ece(np.array([0.9, 0.9, 0.1, 0.1]), np.array([1, 1, 0, 0])) < 0.15
