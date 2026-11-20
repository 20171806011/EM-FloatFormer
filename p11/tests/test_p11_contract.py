"""Unit tests for `p11.contract` and `p11` package invariants."""
from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
from decimal import Decimal
from fractions import Fraction
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest
import warnings

import numpy as np

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
import torch

import p9.contract as p9_contract
from p9.independent_ref import check_no_hardcoded_n0
import p11.contract as p11_contract
from p11.contract import (
    ALL_REGIMES,
    ANCHOR,
    COMPLEX_DTYPE,
    CORE_ROWS,
    CP,
    DELAYS,
    FRAME_DURATION,
    HISTORICAL_SEED_RANGES,
    LINK_DURATION,
    N,
    OFDM_PILOT_ROWS,
    OFFGRID_ALL_REGIMES,
    OFFGRID_ANCHOR_ROWS,
    OFFGRID_DELAY_MAX,
    OFFGRID_DELAY_MIN,
    OFFGRID_RAY_COUNTS,
    OFFGRID_REGIMES_ID,
    OFFGRID_REGIMES_OOD,
    P10_BASELINE_COMMIT,
    P11_NAMESPACE_SPAN,
    P11_SEED_RANGES,
    P8C_BASELINE_COMMIT,
    P8_MANIFOLD_COMMIT,
    P9_BASELINE_COMMIT,
    POWERS,
    PRIMARY_SNR_DB,
    PROBE_ROW,
    REAL_DTYPE,
    REGIMES_ID,
    REGIMES_OOD,
    REGIMES_SANITY,
    SC_LAYOUTS,
    SEED_P11_DEV,
    SEED_P11_E2E,
    SEED_P11_FORMAL_ID,
    SEED_P11_FORMAL_OOD,
    SEED_P11_TRAIN,
    SEED_P9_DEV,
    SEED_P9_FORMAL_ID,
    SEED_P9_FORMAL_OOD,
    SEED_P9_TRAIN,
    SNR_GRID_DB,
    T,
    AssimilationResult,
    CandidateSummary,
    GateOutput,
    HiddenTruth,
    OffGridObservation,
    OffGridTruth,
    PerRayGateOutput,
    PublicObservation,
    apply_fractional_sc_operator,
    build_fractional_sc_matrix,
    build_offgrid_sc_channel_matrix,
    enforce_determinism,
    fractional_delay_freq_response,
    fractional_delay_kernel,
    fractional_delay_matrix,
    is_historical_seed,
    jeffreys_complex_gaussian,
    mix_bit_llrs,
    n0_from_snr,
    signed_subcarrier_indices,
    unitary_dft_matrix,
)

P11_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = P11_DIR.parent


class TestP11ReexportsAndProvenance(unittest.TestCase):
    def test_p9_reexports_and_geometry(self) -> None:
        self.assertIs(PublicObservation, p9_contract.PublicObservation)
        self.assertIs(HiddenTruth, p9_contract.HiddenTruth)
        self.assertIs(CandidateSummary, p9_contract.CandidateSummary)
        self.assertIs(GateOutput, p9_contract.GateOutput)
        self.assertIs(jeffreys_complex_gaussian, p9_contract.jeffreys_complex_gaussian)
        self.assertIs(mix_bit_llrs, p9_contract.mix_bit_llrs)
        self.assertIs(n0_from_snr, p9_contract.n0_from_snr)

        self.assertEqual(N, 32)
        self.assertEqual(CP, 16)
        self.assertEqual(T, 20)
        self.assertEqual(PROBE_ROW, 19)
        self.assertEqual(ANCHOR, 10)
        self.assertEqual(ANCHOR, SC_LAYOUTS["sparse"][0])
        self.assertEqual(len(CORE_ROWS), 16)
        self.assertTrue({0, 9, 10, 19}.isdisjoint(CORE_ROWS))
        self.assertEqual(OFDM_PILOT_ROWS, (0, 9, 10, 19))
        self.assertTrue(np.array_equal(DELAYS, p9_contract.DELAYS))
        self.assertTrue(np.array_equal(POWERS, p9_contract.POWERS))
        self.assertEqual(FRAME_DURATION, p9_contract.FRAME_DURATION)
        self.assertEqual(LINK_DURATION, p9_contract.LINK_DURATION)
        self.assertEqual(SNR_GRID_DB, p9_contract.SNR_GRID_DB)
        self.assertEqual(PRIMARY_SNR_DB, p9_contract.PRIMARY_SNR_DB)
        self.assertEqual(REGIMES_ID, p9_contract.REGIMES_ID)
        self.assertEqual(REGIMES_OOD, p9_contract.REGIMES_OOD)
        self.assertEqual(REGIMES_SANITY, p9_contract.REGIMES_SANITY)
        self.assertEqual(ALL_REGIMES, p9_contract.ALL_REGIMES)
        self.assertEqual(SEED_P9_DEV, p9_contract.SEED_P9_DEV)
        self.assertEqual(SEED_P9_TRAIN, p9_contract.SEED_P9_TRAIN)
        self.assertEqual(SEED_P9_FORMAL_ID, p9_contract.SEED_P9_FORMAL_ID)
        self.assertEqual(SEED_P9_FORMAL_OOD, p9_contract.SEED_P9_FORMAL_OOD)

    def test_provenance_dtypes_and_offgrid_constants(self) -> None:
        self.assertEqual(P8C_BASELINE_COMMIT, p9_contract.P8C_BASELINE_COMMIT)
        self.assertEqual(P8_MANIFOLD_COMMIT, p9_contract.P8_MANIFOLD_COMMIT)
        self.assertEqual(P9_BASELINE_COMMIT, "faeea639727aab72740cdd0a719106f83824226e")
        self.assertEqual(P10_BASELINE_COMMIT, "9e9d93bcbf6345b8645314b223cd954b2f31e63a")
        self.assertIs(REAL_DTYPE, np.float64)
        self.assertIs(COMPLEX_DTYPE, np.complex128)
        self.assertEqual(OFFGRID_DELAY_MIN, 0.0)
        self.assertEqual(OFFGRID_DELAY_MAX, 6.5)
        self.assertEqual(OFFGRID_RAY_COUNTS, (2, 3, 4, 5))
        self.assertEqual(OFFGRID_ANCHOR_ROWS, (3, 6, 10, 13, 16))
        self.assertEqual(OFFGRID_ALL_REGIMES, OFFGRID_REGIMES_ID + OFFGRID_REGIMES_OOD)
        self.assertEqual(len(OFFGRID_REGIMES_ID), 4)
        self.assertEqual(len(OFFGRID_REGIMES_OOD), 2)


class TestP11SeedDisjointness(unittest.TestCase):
    def test_p11_ranges_pairwise_disjoint_and_disjoint_from_historical(self) -> None:
        self.assertEqual(SEED_P11_DEV, 1_110_000_000_000)
        self.assertEqual(SEED_P11_TRAIN, 1_120_000_000_000)
        self.assertEqual(SEED_P11_FORMAL_ID, 1_130_000_000_000)
        self.assertEqual(SEED_P11_FORMAL_OOD, 1_140_000_000_000)
        self.assertEqual(SEED_P11_E2E, 1_150_000_000_000)
        self.assertEqual(P11_NAMESPACE_SPAN, 10_000_000_000)

        p11_items = list(P11_SEED_RANGES.items())
        self.assertEqual(len(p11_items), 5)
        for idx, (name_a, (lo_a, hi_a)) in enumerate(p11_items):
            self.assertEqual(hi_a - lo_a, P11_NAMESPACE_SPAN)
            for name_b, (lo_b, hi_b) in p11_items[idx + 1 :]:
                self.assertFalse(
                    max(lo_a, lo_b) < min(hi_a, hi_b),
                    f"P11 intervals overlap: {name_a} and {name_b}",
                )
            for hist_name, (h_lo, h_hi) in HISTORICAL_SEED_RANGES.items():
                self.assertFalse(
                    max(lo_a, h_lo) < min(hi_a, h_hi),
                    f"P11 interval {name_a} overlaps historical {hist_name}",
                )
            for offset in (*range(64), 999_999_999, P11_NAMESPACE_SPAN - 1):
                self.assertIsNone(is_historical_seed(lo_a + offset))

    def test_known_historical_seeds_detected(self) -> None:
        cases = [
            (42, "p1_to_p4_legacy"),
            (74001, "p1_to_p4_legacy"),
            (912512, "p1_to_p4_legacy"),
            (10_000_000, "p1_to_p4_extended"),
            (10_102_026, "p1_to_p4_extended"),
            (10_104_040, "p1_to_p4_extended"),
            (17_261_119, "p1_to_p4_extended"),
            (20_261_120, "p1_to_p4_extended"),
            (28_161_119, "p1_to_p4_extended"),
            (78_321_000, "p1_to_p4_extended"),
            (84_001_000, "p1_to_p4_extended"),
            (84_300_000, "p1_to_p4_extended"),
            (87_999_999, "p1_to_p4_extended"),
            (88_000_000, "p8_manifold_dev_ladder"),
            (105_001_000, "p5_p6"),
            (400_000_000, "p3_and_exploratory"),
            (438_100_000, "p3_and_exploratory"),
            (467_000_000, "p3_and_exploratory"),
            (603_000_000, "p3_and_exploratory"),
            (700_400_000, "p3_and_exploratory"),
            (709_100_000, "p3_and_exploratory"),
            (731_000_000, "p3_and_exploratory"),
            (911_000_000, "p3_and_exploratory"),
            (970_000_000, "p3_and_exploratory"),
            (990_000_000, "p3_and_exploratory"),
            (1_190_000_000, "p3_and_exploratory"),
            (1_299_999_999, "p3_and_exploratory"),
            (2_147_483_647, "p3_and_exploratory"),
            (5_220_100_000, "p3b_data_seed"),
            (5_220_599_999, "p3b_data_seed"),
            (602_611_190_000, "p6_followup"),
            (602_611_193_170, "p6_followup"),
            (602_612_090_001, "p6_followup"),
            (702_611_190_000, "p7"),
            (713_611_190_000, "p6_softinfo"),
            (713_611_200_003, "p6_softinfo"),
            (802_611_190_000, "p8_manifold"),
            (802_611_790_303, "p8_manifold"),
            (812_611_190_000, "p8_prefix"),
            (813_611_190_071, "p6_consistency"),
            (814_611_190_000, "p6_consistency"),
            (814_611_200_003, "p6_consistency"),
            (822_611_190_000, "p8c"),
            (910_000_000_000, "p9_dev"),
            (920_000_000_000, "p9_train"),
            (930_000_000_000, "p9_formal_id"),
            (940_000_000_000, "p9_formal_ood"),
            (960_261_120_000, "p10_9e9d93b"),
            (971_120_050_000, "p10_9e9d93b"),
            (981_120_040_000, "p10_9e9d93b"),
            (982_000_000_000, "p10_9e9d93b"),
            (1_010_000_000_000, "p10_external_dev"),
            (1_020_000_000_000, "p10_external_train"),
            (1_030_000_000_000, "p10_external_formal_id"),
            (1_040_000_000_000, "p10_external_formal_ood"),
            (1_050_000_000_000, "p10_external_reserved_1050"),
            (9_102_379_953_802, "p4_stage3_link_genie_literal"),
        ]
        for seed, expected_key in cases:
            self.assertEqual(is_historical_seed(seed), expected_key)

    def test_historical_ranges_pairwise_disjoint_and_endpoints(self) -> None:
        items = list(HISTORICAL_SEED_RANGES.items())
        self.assertEqual(len(items), 24)
        for idx, (name_a, (lo_a, hi_a)) in enumerate(items):
            self.assertGreaterEqual(lo_a, 0)
            self.assertLess(lo_a, hi_a)
            self.assertEqual(is_historical_seed(lo_a), name_a)
            self.assertEqual(is_historical_seed(hi_a - 1), name_a)
            for name_b, (lo_b, hi_b) in items[idx + 1 :]:
                self.assertFalse(
                    max(lo_a, lo_b) < min(hi_a, hi_b),
                    f"Historical intervals overlap: {name_a} and {name_b}",
                )
        self.assertIsNone(is_historical_seed(-1))
        self.assertIsNone(is_historical_seed(2_147_483_648))
        self.assertEqual(is_historical_seed(np.int64(602_611_190_000)), "p6_followup")
        with self.assertRaises(TypeError):
            is_historical_seed(np.bool_(True))  # type: ignore[arg-type]

    def test_is_historical_seed_rejects_bool_and_float(self) -> None:
        for bad in (True, False, np.bool_(True), np.bool_(False), 1.5, "42"):
            with self.assertRaises(TypeError):
                is_historical_seed(bad)  # type: ignore[arg-type]


class TestP11Determinism(unittest.TestCase):
    _THREAD_VARS = ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS")

    def setUp(self) -> None:
        self._orig_threads = torch.get_num_threads()
        self._orig_det = torch.are_deterministic_algorithms_enabled()

    def tearDown(self) -> None:
        torch.set_num_threads(self._orig_threads)
        torch.use_deterministic_algorithms(self._orig_det)

    def _patched_thread_env(self, overrides: dict[str, str]):
        saved = {k: os.environ.get(k) for k in self._THREAD_VARS}

        class _Ctx:
            def __enter__(self_ctx) -> None:
                for k in self._THREAD_VARS:
                    os.environ.pop(k, None)
                os.environ.update(overrides)

            def __exit__(self_ctx, *args: object) -> None:
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v

        return _Ctx()

    def test_require_env_positive_with_unset_or_one_mkl(self) -> None:
        with self._patched_thread_env({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}):
            status = enforce_determinism(require_env=True)
            self.assertEqual(status["torch_num_threads"], 1)
            self.assertTrue(status["deterministic_algorithms"])
            self.assertEqual(os.environ["MKL_NUM_THREADS"], "1")
            self.assertEqual(torch.get_num_threads(), 1)
            self.assertTrue(torch.are_deterministic_algorithms_enabled())

        with self._patched_thread_env(
            {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
        ):
            status = enforce_determinism(require_env=True)
            self.assertEqual(status["mkl_num_threads"], "1")

    def test_require_env_negative_cases(self) -> None:
        invalid_envs = [
            {},
            {"OPENBLAS_NUM_THREADS": "1"},
            {"OMP_NUM_THREADS": "1"},
            {"OPENBLAS_NUM_THREADS": "4", "OMP_NUM_THREADS": "1"},
            {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "2"},
            {"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "4"},
        ]
        for env in invalid_envs:
            with self._patched_thread_env(env):
                before = dict(os.environ)
                with self.assertRaises(RuntimeError):
                    enforce_determinism(require_env=True)
                self.assertEqual(dict(os.environ), before)
                self.assertEqual(os.environ.get("MKL_NUM_THREADS"), env.get("MKL_NUM_THREADS"))

        clean_env = {k: v for k, v in os.environ.items() if k not in self._THREAD_VARS}
        clean_env["OMP_NUM_THREADS"] = "1"
        proc = subprocess.run(
            [
                sys.executable,
                "-B",
                "-c",
                (
                    "import os, sys, p11.contract as c; "
                    "raised = False\n"
                    "try:\n"
                    "    c.enforce_determinism(require_env=True)\n"
                    "except RuntimeError:\n"
                    "    raised = True\n"
                    "assert raised\n"
                    "assert 'MKL_NUM_THREADS' not in os.environ\n"
                    "assert 'torch' not in sys.modules\n"
                ),
            ],
            cwd=str(REPO_ROOT),
            env=clean_env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=f"Subprocess check failed: {proc.stderr}")

    def test_permissive_mode_sets_env_and_torch(self) -> None:
        with self._patched_thread_env({"OPENBLAS_NUM_THREADS": "8", "MKL_NUM_THREADS": "4"}):
            status = enforce_determinism(require_env=False)
            self.assertFalse(status["require_env"])
            self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "1")
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "1")
            self.assertEqual(os.environ["MKL_NUM_THREADS"], "1")
            self.assertEqual(torch.get_num_threads(), 1)
            self.assertTrue(torch.are_deterministic_algorithms_enabled())

    def test_require_env_rejects_non_bool(self) -> None:
        for bad_req in ("false", "true", 1, 0, None, [True], np.array([True])):
            with self.assertRaises(TypeError):
                enforce_determinism(require_env=bad_req)  # type: ignore[arg-type]
        with self._patched_thread_env({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1"}):
            status = enforce_determinism(require_env=np.bool_(True))  # type: ignore[arg-type]
            self.assertIs(status["require_env"], True)


class TestFractionalDelayOperators(unittest.TestCase):
    def test_unitarity_adjointness_group_duality_and_parseval(self) -> None:
        rng = np.random.default_rng(12345)
        delays = (0.0, 0.25, 0.5, 1.37, 2.0, 3.14159265, 4.0, 5.83, 6.5)
        for n in (16, 17, 32):
            eye = np.eye(n, dtype=COMPLEX_DTYPE)
            f_mat = unitary_dft_matrix(n)
            x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(COMPLEX_DTYPE)
            x_energy = float(np.sum(np.abs(x) ** 2))

            for tau in delays:
                d_mat = fractional_delay_matrix(tau, n=n)
                self.assertEqual(d_mat.dtype, COMPLEX_DTYPE)
                self.assertTrue(np.array_equal(d_mat, fractional_delay_kernel(tau, n=n)))
                self.assertLess(np.max(np.abs(d_mat.conj().T @ d_mat - eye)), 1e-14)
                self.assertLess(np.max(np.abs(d_mat @ d_mat.conj().T - eye)), 1e-14)

                d_neg = fractional_delay_matrix(-tau, n=n)
                self.assertLess(np.max(np.abs(d_neg - d_mat.conj().T)), 1e-14)

                freq_resp = fractional_delay_freq_response(tau, n=n)
                dual = f_mat @ d_mat @ f_mat.conj().T
                self.assertLess(np.max(np.abs(dual - np.diag(freq_resp))), 1e-14)

                y = d_mat @ x
                y_energy = float(np.sum(np.abs(y) ** 2))
                self.assertLess(abs(y_energy - x_energy) / max(x_energy, 1.0), 1e-13)

            for a, b in ((0.25, 1.37), (2.0, 3.14159265), (5.83, -2.5)):
                d_comp = fractional_delay_matrix(a, n=n) @ fractional_delay_matrix(b, n=n)
                d_sum = fractional_delay_matrix(a + b, n=n)
                self.assertLess(np.max(np.abs(d_comp - d_sum)), 1e-14)

            x_inf = max(float(np.max(np.abs(x))), 1.0)
            for d_int in (0, 1, 2, 4, 7):
                y_int = fractional_delay_matrix(float(d_int), n=n) @ x
                self.assertLess(np.max(np.abs(y_int - np.roll(x, d_int))) / x_inf, 1e-14)

    def test_sc_matrix_and_operator_equivalence_and_parseval(self) -> None:
        rng = np.random.default_rng(67890)
        n = 32
        p_rays = 4
        ray_gains = (rng.standard_normal((p_rays, n)) + 1j * rng.standard_normal((p_rays, n))).astype(COMPLEX_DTYPE)
        delays = np.array([0.0, 1.35, 2.75, 5.12], dtype=REAL_DTYPE)

        h_mat = build_offgrid_sc_channel_matrix(ray_gains, delays, n=n)
        self.assertTrue(np.array_equal(h_mat, build_fractional_sc_matrix(ray_gains, delays, n=n)))

        x1 = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(COMPLEX_DTYPE)
        y1_op = apply_fractional_sc_operator(x1, ray_gains, delays, n=n)
        y1_mat = h_mat @ x1
        self.assertLess(np.max(np.abs(y1_op - y1_mat)), 1e-14)

        x2 = (rng.standard_normal((5, n)) + 1j * rng.standard_normal((5, n))).astype(COMPLEX_DTYPE)
        y2_op = apply_fractional_sc_operator(x2, ray_gains, delays, n=n)
        y2_mat = (h_mat @ x2.T).T
        self.assertLess(np.max(np.abs(y2_op - y2_mat)), 1e-14)

        # Single unit-modulus ray preserves exact Parseval energy
        phases = rng.uniform(-np.pi, np.pi, size=(1, n))
        g_unit = np.exp(1j * phases).astype(COMPLEX_DTYPE)
        y_unit = apply_fractional_sc_operator(x2, g_unit, np.array([2.43], dtype=REAL_DTYPE), n=n)
        e_in = np.sum(np.abs(x2) ** 2, axis=-1)
        e_out = np.sum(np.abs(y_unit) ** 2, axis=-1)
        self.assertLess(np.max(np.abs(e_out - e_in) / np.maximum(e_in, 1.0)), 1e-13)

    def test_invalid_inputs_raise_value_error(self) -> None:
        for bad_n in (0, -3, True):
            with self.assertRaises(ValueError):
                signed_subcarrier_indices(bad_n)  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                fractional_delay_matrix(1.0, n=bad_n)  # type: ignore[arg-type]
        for bad_delay in (
            math.nan,
            math.inf,
            -math.inf,
            True,
            False,
            np.bool_(True),
            np.bool_(False),
            1.0 + 2.0j,
            1.0 + 0.0j,
            np.complex128(0.5 + 0j),
        ):
            with self.assertRaises(ValueError):
                fractional_delay_matrix(bad_delay)  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                fractional_delay_freq_response(bad_delay)  # type: ignore[arg-type]
        for bad_scalar in ([1.0], np.array([1.0, 2.0])):
            with self.assertRaises(ValueError):
                fractional_delay_matrix(bad_scalar)  # type: ignore[arg-type]
        for bad_delay_arr in (
            [True],
            [False, True],
            [1.0, True],
            np.array([True, False], dtype=np.bool_),
            [1.0 + 2.0j],
            np.array([1.0 + 0j]),
            np.array([1.0, True], dtype=object),
            np.array([1.0 + 0j], dtype=object),
            np.array([0.0, math.nan]),
        ):
            with self.assertRaises(ValueError):
                fractional_delay_freq_response(bad_delay_arr)  # type: ignore[arg-type]
        for bad_delays in (
            np.array([True, False], dtype=np.bool_),
            np.array([1.0 + 0j, 2.0 + 0j]),
        ):
            with self.assertRaises(ValueError):
                build_offgrid_sc_channel_matrix(
                    np.zeros((2, N), dtype=COMPLEX_DTYPE),
                    bad_delays,  # type: ignore[arg-type]
                )
            with self.assertRaises(ValueError):
                apply_fractional_sc_operator(
                    np.zeros(N, dtype=COMPLEX_DTYPE),
                    np.zeros((2, N), dtype=COMPLEX_DTYPE),
                    bad_delays,  # type: ignore[arg-type]
                )
        with self.assertRaises(ValueError):
            build_offgrid_sc_channel_matrix(np.zeros((2, N), dtype=COMPLEX_DTYPE), np.zeros(3, dtype=REAL_DTYPE))
        with self.assertRaises(ValueError):
            apply_fractional_sc_operator(
                np.zeros(N - 1, dtype=COMPLEX_DTYPE),
                np.zeros((2, N), dtype=COMPLEX_DTYPE),
                np.zeros(2, dtype=REAL_DTYPE),
            )

    def test_delay_validation_rejects_all_bypass_categories_across_all_operators(self) -> None:
        class _CustomFloat:
            def __float__(self) -> float:
                return 1.5

        class _CustomArray:
            def __array__(self, dtype: object = None) -> np.ndarray:
                return np.array([0.5, 1.5], dtype=REAL_DTYPE if dtype is None else dtype)

        cyclic_list: list[object] = []
        cyclic_list.append(cyclic_list)
        cyclic_obj = np.empty(1, dtype=object)
        cyclic_obj[0] = cyclic_obj

        n_test = 4
        x1 = np.ones(n_test, dtype=COMPLEX_DTYPE)
        x2 = np.ones((2, n_test), dtype=COMPLEX_DTYPE)
        gains_by_p = {p: np.ones((p, n_test), dtype=COMPLEX_DTYPE) for p in (0, 1, 2)}

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            longdouble_inf = np.longdouble("1e500")

        # Scalar / 0-D malformed cases across all 11 categories + custom/buffer/overflow cases
        scalar_0d_cases: list[tuple[str, object]] = [
            ("0d_bool_true", np.array(True, dtype=np.bool_)),
            ("0d_bool_false", np.array(False, dtype=np.bool_)),
            ("0d_obj_0d_bool", np.array(np.array(True, dtype=np.bool_), dtype=object)),
            ("py_bool_true", True),
            ("py_bool_false", False),
            ("np_bool_true", np.bool_(True)),
            ("np_bool_false", np.bool_(False)),
            ("str_1_5", "1.5"),
            ("str_0_0", "0.0"),
            ("str_nan", "nan"),
            ("str_inf", "inf"),
            ("np_str_0d", np.array("1.5")),
            ("obj_str_0d", np.array("1.5", dtype=object)),
            ("bytes_1_5", b"1.5"),
            ("bytes_0_0", b"0.0"),
            ("np_bytes_0d", np.array(b"1.5")),
            ("obj_bytes_0d", np.array(b"1.5", dtype=object)),
            ("bytearray", bytearray(b"1.5")),
            ("memoryview", memoryview(b"123")),
            ("timedelta_1s", np.timedelta64(1, "s")),
            ("timedelta_5ms", np.timedelta64(5, "ms")),
            ("timedelta_nat", np.timedelta64("NaT", "s")),
            ("obj_timedelta_0d", np.array(np.timedelta64(1, "s"), dtype=object)),
            ("datetime_day", np.datetime64("2026-01-01")),
            ("datetime_nat", np.datetime64("NaT")),
            ("obj_datetime_0d", np.array(np.datetime64("2026-01-01"), dtype=object)),
            ("void_scalar", np.array([(1.0,)], dtype=[("x", "f8")])[0]),
            ("obj_void_0d", np.array(np.array([(1.0,)], dtype=[("x", "f8")])[0], dtype=object)),
            ("masked_0d_true", np.ma.masked_array(1.5, mask=True)),
            ("masked_0d_false", np.ma.masked_array(1.5, mask=False)),
            ("ma_masked_const", np.ma.masked),
            ("none", None),
            ("obj_none_0d", np.array(None, dtype=object)),
            ("dict", {"a": 1.0}),
            ("set", {1.0, 2.0}),
            ("complex_real", 1.0 + 0.0j),
            ("complex_imag", 1.0 + 2.0j),
            ("np_complex64", np.complex64(0.5 + 0j)),
            ("np_complex128", np.complex128(0.5 + 1j)),
            ("0d_complex_arr", np.array(1.0 + 0j, dtype=np.complex128)),
            ("obj_complex_0d", np.array(1.0 + 0j, dtype=object)),
            ("py_nan", math.nan),
            ("py_inf", math.inf),
            ("py_neg_inf", -math.inf),
            ("f16_nan", np.float16("nan")),
            ("f32_ninf", np.float32("-inf")),
            ("f64_nan", np.float64("nan")),
            ("f64_inf", np.float64("inf")),
            ("0d_nan_arr", np.array(math.nan)),
            ("int_overflow_10e400", 10**400),
            ("longdouble_inf", longdouble_inf),
            ("decimal_1_5", Decimal("1.5")),
            ("fraction_3_2", Fraction(3, 2)),
            ("custom_float", _CustomFloat()),
            ("custom_array", _CustomArray()),
            ("torch_0d_bool", torch.tensor(True)),
        ]

        # 1-D malformed cases paired with exact length P in {0, 1, 2}
        vec_1d_cases: list[tuple[str, int, object]] = [
            # Cat 1: 0-D bool ndarray inside lists/tuples/object arrays
            ("mixed_list_0d_bool_p2", 2, [1.0, np.array(True, dtype=np.bool_)]),
            ("mixed_tuple_0d_bool_p2", 2, (1.0, np.array(False, dtype=np.bool_))),
            ("list_0d_bool_p1", 1, [np.array(True, dtype=np.bool_)]),
            ("tuple_0d_bool_p1", 1, (np.array(False, dtype=np.bool_),)),
            ("obj_0d_bool_p1", 1, np.array([np.array(True, dtype=np.bool_)], dtype=object)),
            ("obj_mixed_0d_bool_p2", 2, np.array([1.0, np.array(True, dtype=np.bool_)], dtype=object)),
            ("obj_nested_tuple_0d_bool_p1", 1, np.array([(1.0, np.array(False, dtype=np.bool_))], dtype=object)),
            # Cat 2: bool / np.bool_ / 1-D bool ndarray
            ("list_bool_p1", 1, [True]),
            ("list_mixed_bool_p2", 2, [1.0, True]),
            ("tuple_mixed_np_bool_p2", 2, (1.0, np.bool_(False))),
            ("bool_arr_p1", 1, np.array([True], dtype=np.bool_)),
            ("bool_arr_p2", 2, np.array([True, False], dtype=np.bool_)),
            ("bool_arr_p0", 0, np.array([], dtype=np.bool_)),
            ("obj_bool_p2", 2, np.array([1.0, True], dtype=object)),
            ("obj_np_bool_p2", 2, np.array([1.0, np.bool_(False)], dtype=object)),
            ("obj_nested_bool_p1", 1, np.array([(1.0, True)], dtype=object)),
            ("obj_1d_bool_p1", 1, np.array([np.array([True], dtype=np.bool_)], dtype=object)),
            # Cat 3: str
            ("list_str_p1", 1, ["1.5"]),
            ("list_str_p2", 2, ["1.5", "2.5"]),
            ("tuple_str_p2", 2, ("1.0", "2.0")),
            ("list_mixed_str_p2", 2, [1.0, "1.5"]),
            ("str_arr_p1", 1, np.array(["1.5"])),
            ("str_arr_p2", 2, np.array(["1.5", "2.5"])),
            ("str_arr_p0", 0, np.array([], dtype="<U1")),
            ("obj_str_p1", 1, np.array(["1.0"], dtype=object)),
            ("obj_mixed_str_p2", 2, np.array([1.0, "1.5"], dtype=object)),
            # Cat 4: bytes / bytearray / memoryview
            ("list_bytes_p1", 1, [b"1.5"]),
            ("list_bytes_p2", 2, [b"1.5", b"2.5"]),
            ("list_mixed_bytes_p2", 2, [1.0, b"1.5"]),
            ("bytes_arr_p1", 1, np.array([b"1.5"])),
            ("bytes_arr_p2", 2, np.array([b"1.5", b"2.5"])),
            ("bytes_arr_p0", 0, np.array([], dtype="|S1")),
            ("obj_bytes_p1", 1, np.array([b"1.5"], dtype=object)),
            ("list_bytearray_p1", 1, [bytearray(b"1.5")]),
            ("list_memoryview_p1", 1, [memoryview(b"12")]),
            # Cat 5: timedelta64
            ("list_timedelta_p1", 1, [np.timedelta64(1, "s")]),
            ("timedelta_arr_p1", 1, np.array([np.timedelta64(1, "s")])),
            ("timedelta_arr_p2", 2, np.array([np.timedelta64(1, "s"), np.timedelta64(2, "s")])),
            ("obj_timedelta_p1", 1, np.array([np.timedelta64(1, "s")], dtype=object)),
            # Cat 6: datetime64
            ("list_datetime_p1", 1, [np.datetime64("2026-01-01")]),
            ("datetime_arr_p1", 1, np.array([np.datetime64("2026-01-01")])),
            ("datetime_arr_p2", 2, np.array([np.datetime64("2026-01-01"), np.datetime64("2026-01-02")])),
            ("obj_datetime_p1", 1, np.array([np.datetime64("2026-01-01")], dtype=object)),
            # Cat 7: structured void
            ("void_arr_p1", 1, np.array([(1.0,)], dtype=[("x", "f8")])),
            ("void_arr_p2", 2, np.array([(1.0,), (2.0,)], dtype=[("x", "f8")])),
            ("void_arr_two_fields_p1", 1, np.array([(1.0, 2.0)], dtype=[("x", "f8"), ("y", "f8")])),
            ("obj_void_p1", 1, np.array([np.array([(1.0,)], dtype=[("x", "f8")])[0]], dtype=object)),
            # Cat 8: MaskedArray
            ("masked_arr_p1_true", 1, np.ma.masked_array([1.5], mask=[True])),
            ("masked_arr_p1_false", 1, np.ma.masked_array([1.5], mask=[False])),
            ("masked_arr_p2_partial", 2, np.ma.masked_array([1.0, 2.0], mask=[False, True])),
            ("masked_arr_p2_unmasked", 2, np.ma.masked_array([1.0, 2.0], mask=[False, False])),
            ("masked_arr_p0", 0, np.ma.masked_array([], mask=[], dtype=REAL_DTYPE)),
            ("obj_masked_p1", 1, np.array([np.ma.masked], dtype=object)),
            ("list_masked_p1", 1, [np.ma.masked_array(1.5, mask=True)]),
            # Cat 9: None & empty object array
            ("list_none_p1", 1, [None]),
            ("list_mixed_none_p2", 2, [1.0, None]),
            ("tuple_mixed_none_p2", 2, (1.0, None)),
            ("obj_none_p1", 1, np.array([None], dtype=object)),
            ("obj_mixed_none_p2", 2, np.array([1.0, None], dtype=object)),
            ("obj_empty_p0", 0, np.array([], dtype=object)),
            # Cat 10: complex
            ("list_complex_p1", 1, [1.0 + 2.0j]),
            ("list_mixed_complex_p2", 2, [1.0, 2.0 + 0j]),
            ("list_mixed_0d_complex_p2", 2, [1.0, np.array(1.0 + 0j, dtype=COMPLEX_DTYPE)]),
            ("complex64_arr_p1", 1, np.array([0.5 + 0j], dtype=np.complex64)),
            ("complex128_arr_p2", 2, np.array([0.5 + 0j, 1.0 - 2j], dtype=COMPLEX_DTYPE)),
            ("complex128_arr_p0", 0, np.array([], dtype=COMPLEX_DTYPE)),
            ("obj_complex_p2", 2, np.array([1.0, 2.0 + 0j], dtype=object)),
            ("obj_0d_complex_p1", 1, np.array([np.array(1.0 + 0j, dtype=COMPLEX_DTYPE)], dtype=object)),
            # Cat 11: nan / inf
            ("list_nan_p1", 1, [math.nan]),
            ("list_nan_p2", 2, [0.0, math.nan]),
            ("list_inf_p2", 2, [0.0, math.inf]),
            ("arr_nan_p2", 2, np.array([0.0, np.nan])),
            ("arr_inf_p2", 2, np.array([0.0, np.inf])),
            ("arr_ninf_p2", 2, np.array([0.0, -np.inf])),
            ("obj_nan_p2", 2, np.array([1.0, float("nan")], dtype=object)),
            # Extra overflow / Decimal / Fraction / torch / cyclic
            ("list_overflow_p1", 1, [10**400]),
            ("list_decimal_p1", 1, [Decimal("1.5")]),
            ("list_fraction_p1", 1, [Fraction(3, 2)]),
            ("list_torch_bool_p2", 2, [1.0, torch.tensor(True)]),
            ("obj_torch_bool_p1", 1, np.array([torch.tensor(True)], dtype=object)),
            ("torch_req_grad_p2", 2, torch.tensor([0.5, 1.5], requires_grad=True)),
            ("cyclic_obj_p1", 1, cyclic_obj),
        ]

        # Nested / 2-D / non-rectangular cases
        nested_cases: list[tuple[str, object]] = [
            ("nested_tuple_0d_bool_1", ((1.0, (2.0, np.array(False, dtype=np.bool_))),)),
            ("nested_list_0d_bool_2", [(1.0, [np.array(True, dtype=np.bool_)])]),
            ("2d_obj_0d_bool", np.array([[1.0], [np.array(True, dtype=np.bool_)]], dtype=object)),
            ("nested_tuple_bool_1", ((True,),)),
            ("nested_tuple_bool_2", ((1.0, (2.0, False)),)),
            ("nested_list_np_bool", [(1.0, [np.bool_(True)])]),
            ("nested_tuple_complex", ((1.0, (0.0 + 0.0j,)),)),
            ("obj_empty_bool_child", np.array([np.empty(0, dtype=np.bool_)], dtype=object)),
            ("ragged_list", [[1.0, 2.0], [3.0]]),
            ("mixed_scalar_and_1d_arr", [1.0, np.array([2.0])]),
            ("cyclic_list", cyclic_list),
        ]

        # 1. fractional_delay_freq_response on ALL cases
        for label, bad in (
            scalar_0d_cases
            + [(lbl, val) for lbl, _, val in vec_1d_cases]
            + nested_cases
        ):
            with self.subTest(op="fractional_delay_freq_response", case=label):
                with self.assertRaises(ValueError):
                    fractional_delay_freq_response(bad, n=n_test)  # type: ignore[arg-type]

        # 2. fractional_delay_matrix and fractional_delay_kernel on ALL 0-D cases + representative 1-D/nested
        for op_name, op_fn in (
            ("fractional_delay_matrix", fractional_delay_matrix),
            ("fractional_delay_kernel", fractional_delay_kernel),
        ):
            for label, bad in scalar_0d_cases:
                with self.subTest(op=op_name, case=label):
                    with self.assertRaises(ValueError) as ctx:
                        op_fn(bad, n=n_test)  # type: ignore[arg-type]
                    self.assertNotIn("delay must be a scalar", str(ctx.exception))
            for label, bad in (
                ("mixed_list_0d_bool", [1.0, np.array(True, dtype=np.bool_)]),
                ("obj_0d_bool", np.array([np.array(True, dtype=np.bool_)], dtype=object)),
                ("list_str", ["1.5"]),
                ("masked_1d", np.ma.masked_array([1.5], mask=[True])),
                ("cyclic_list", cyclic_list),
            ):
                with self.subTest(op=op_name, case=label):
                    with self.assertRaises(ValueError):
                        op_fn(bad, n=n_test)  # type: ignore[arg-type]

        # 3. All 3 SC operators with shape-matched ray_gains_row
        for label, p_len, bad in vec_1d_cases:
            gains_p = gains_by_p[p_len]
            for op_name, call_fn in (
                ("build_offgrid_sc_channel_matrix", lambda g, d: build_offgrid_sc_channel_matrix(g, d, n=n_test)),
                ("build_fractional_sc_matrix", lambda g, d: build_fractional_sc_matrix(g, d, n=n_test)),
                ("apply_fractional_sc_operator_1d", lambda g, d: apply_fractional_sc_operator(x1, g, d, n=n_test)),
            ):
                with self.subTest(op=op_name, case=label):
                    with self.assertRaises(ValueError) as ctx:
                        call_fn(gains_p, bad)
                    self.assertNotIn("Expected ray_gains_row", str(ctx.exception))

        # 0-D and nested cases on SC operators also fail delay validation before shape check
        gains_1 = gains_by_p[1]
        for label, bad in scalar_0d_cases + nested_cases:
            for op_name, call_fn in (
                ("build_offgrid_sc_channel_matrix", lambda g, d: build_offgrid_sc_channel_matrix(g, d, n=n_test)),
                ("build_fractional_sc_matrix", lambda g, d: build_fractional_sc_matrix(g, d, n=n_test)),
                ("apply_fractional_sc_operator_1d", lambda g, d: apply_fractional_sc_operator(x1, g, d, n=n_test)),
            ):
                with self.subTest(op=op_name, case=label):
                    with self.assertRaises(ValueError) as ctx:
                        call_fn(gains_1, bad)
                    self.assertNotIn("Expected ray_gains_row", str(ctx.exception))

        # Representative 2-D batch x2 on apply_fractional_sc_operator
        for label, p_len, bad in (
            ("mixed_list_0d_bool_p2", 2, [1.0, np.array(True, dtype=np.bool_)]),
            ("datetime_arr_p1", 1, np.array([np.datetime64("2026-01-01")])),
            ("bytes_p1", 1, [b"1.5"]),
            ("masked_p2", 2, np.ma.masked_array([1.0, 2.0], mask=[False, True])),
        ):
            with self.subTest(op="apply_fractional_sc_operator_2d", case=label):
                with self.assertRaises(ValueError) as ctx:
                    apply_fractional_sc_operator(x2, gains_by_p[p_len], bad, n=n_test)  # type: ignore[arg-type]
                self.assertNotIn("Expected ray_gains_row", str(ctx.exception))

    def test_ray_gains_and_x_reject_malformed_inputs(self) -> None:
        n_test = 4
        x_ok = np.ones(n_test, dtype=COMPLEX_DTYPE)
        gains_ok = np.ones((1, n_test), dtype=COMPLEX_DTYPE)
        delays_1 = np.array([0.5], dtype=REAL_DTYPE)

        bad_gains_cases: list[tuple[str, object]] = [
            ("bool_arr_2d", np.ones((1, n_test), dtype=np.bool_)),
            ("nested_bool_list", [[True, False, True, False]]),
            ("mixed_0d_bool_list", [[1.0, np.array(True, dtype=np.bool_), 0.0, 0.0]]),
            ("obj_0d_bool_2d", np.array([[np.array(True, dtype=np.bool_), 1.0, 2.0, 3.0]], dtype=object)),
            ("str_2d", [["1+2j", "2", "3", "4"]]),
            ("bytes_2d", [[b"1.0", b"2.0", b"3.0", b"4.0"]]),
            ("bytearray", bytearray(b"1234")),
            ("memoryview", memoryview(b"1234")),
            ("masked_2d", np.ma.masked_array(np.ones((1, n_test)), mask=[[False, True, False, False]])),
            ("masked_2d_false", np.ma.masked_array(np.ones((1, n_test)), mask=False)),
            ("datetime_2d", np.array([[np.datetime64("2026-01-01")] * n_test])),
            ("timedelta_2d", np.array([[np.timedelta64(1, "s")] * n_test])),
            ("void_2d", np.array([[(1.0,)] * n_test], dtype=[("x", "f8")])),
            ("dict", {"a": 1.0}),
            ("torch_bool", torch.ones((1, n_test), dtype=torch.bool)),
            ("torch_req_grad", torch.ones((1, n_test), requires_grad=True)),
            ("nan_2d", [[1.0, 2.0, 3.0, math.nan]]),
            ("complex_inf_2d", [[1.0 + math.inf * 1j, 0.0, 0.0, 0.0]]),
        ]
        for label, bad_g in bad_gains_cases:
            for op_name, call_fn in (
                ("build_offgrid_sc_channel_matrix", lambda g: build_offgrid_sc_channel_matrix(g, delays_1, n=n_test)),
                ("build_fractional_sc_matrix", lambda g: build_fractional_sc_matrix(g, delays_1, n=n_test)),
                ("apply_fractional_sc_operator", lambda g: apply_fractional_sc_operator(x_ok, g, delays_1, n=n_test)),
            ):
                with self.subTest(op=op_name, bad_gains=label):
                    with self.assertRaises(ValueError) as ctx:
                        call_fn(bad_g)
                    self.assertNotIn("Expected ray_gains_row", str(ctx.exception))

        bad_x_cases: list[tuple[str, object]] = [
            ("bool_arr_1d", np.ones(n_test, dtype=np.bool_)),
            ("bool_list", [True, False, True, False]),
            ("mixed_0d_bool_list", [1.0, np.array(True, dtype=np.bool_), 0.0, 0.0]),
            ("obj_0d_bool_1d", np.array([np.array(True, dtype=np.bool_), 1.0, 2.0, 3.0], dtype=object)),
            ("str_1d", ["1+2j", "2", "3", "4"]),
            ("bytes_1d", [b"1.0", b"2.0", b"3.0", b"4.0"]),
            ("bytearray", bytearray(b"1234")),
            ("memoryview", memoryview(b"1234")),
            ("masked_1d", np.ma.masked_array(np.ones(n_test), mask=[False, True, False, False])),
            ("datetime_1d", np.array([np.datetime64("2026-01-01")] * n_test)),
            ("timedelta_1d", np.array([np.timedelta64(1, "s")] * n_test)),
            ("void_1d", np.array([(1.0,)] * n_test, dtype=[("x", "f8")])),
            ("dict", {"a": 1.0}),
            ("torch_bool", torch.ones(n_test, dtype=torch.bool)),
            ("torch_req_grad", torch.ones(n_test, requires_grad=True)),
            ("nan_1d", [1.0, 2.0, 3.0, math.nan]),
        ]
        for label, bad_x in bad_x_cases:
            with self.subTest(bad_x=label):
                with self.assertRaises(ValueError) as ctx:
                    apply_fractional_sc_operator(bad_x, gains_ok, delays_1, n=n_test)  # type: ignore[arg-type]
                self.assertNotIn("Expected x with trailing dimension", str(ctx.exception))

    def test_delay_validation_accepts_valid_real_numeric_inputs(self) -> None:
        n_test = 4
        scalar_valid = (
            0,
            -2,
            1.5,
            np.int32(2),
            np.int64(-1),
            np.uint8(3),
            np.float16(0.5),
            np.float32(1.25),
            np.float64(-0.75),
            np.array(1.5, dtype=REAL_DTYPE),
            np.array(2, dtype=np.int64),
        )
        for val in scalar_valid:
            fr = fractional_delay_freq_response(val, n=n_test)
            dm = fractional_delay_matrix(val, n=n_test)
            dk = fractional_delay_kernel(val, n=n_test)
            self.assertEqual(fr.shape, (n_test,))
            self.assertEqual(fr.dtype, COMPLEX_DTYPE)
            self.assertEqual(dm.shape, (n_test, n_test))
            self.assertEqual(dm.dtype, COMPLEX_DTYPE)
            self.assertTrue(np.array_equal(dm, dk))

        x_valid = np.ones(n_test, dtype=COMPLEX_DTYPE)
        vec_valid: list[tuple[int, object]] = [
            (3, [0, 1.5, np.float32(2.5)]),
            (3, (0.0, 1, np.int64(2))),
            (2, [np.array(0.5, dtype=REAL_DTYPE), np.array(1, dtype=np.int64)]),
            (2, np.array([0.0, 1.5], dtype=np.float32)),
            (2, np.array([1, 2], dtype=np.int64)),
            (0, []),
            (0, ()),
            (0, np.empty(0, dtype=REAL_DTYPE)),
        ]
        for p_len, delays_val in vec_valid:
            gains_p = np.ones((p_len, n_test), dtype=COMPLEX_DTYPE)
            fr = fractional_delay_freq_response(delays_val, n=n_test)  # type: ignore[arg-type]
            h_mat = build_offgrid_sc_channel_matrix(gains_p, delays_val, n=n_test)  # type: ignore[arg-type]
            h_alias = build_fractional_sc_matrix(gains_p, delays_val, n=n_test)  # type: ignore[arg-type]
            y_op = apply_fractional_sc_operator(x_valid, gains_p, delays_val, n=n_test)  # type: ignore[arg-type]
            self.assertEqual(fr.shape, (p_len, n_test))
            self.assertEqual(fr.dtype, COMPLEX_DTYPE)
            self.assertEqual(h_mat.shape, (n_test, n_test))
            self.assertEqual(h_mat.dtype, COMPLEX_DTYPE)
            self.assertTrue(np.array_equal(h_mat, h_alias))
            self.assertEqual(y_op.shape, (n_test,))
            self.assertEqual(y_op.dtype, COMPLEX_DTYPE)

        # Empty-batch x and real/float32/complex64 ray_gains_row & x compatibility
        y_empty_batch = apply_fractional_sc_operator(
            np.ones((0, n_test), dtype=COMPLEX_DTYPE),
            np.ones((1, n_test), dtype=np.complex64),
            np.array([0.5], dtype=np.float32),
            n=n_test,
        )
        self.assertEqual(y_empty_batch.shape, (0, n_test))
        self.assertEqual(y_empty_batch.dtype, COMPLEX_DTYPE)

        y_real_inputs = apply_fractional_sc_operator(
            [1, 2, 3, 4],
            [[0.5, 1.0, 1.5, 2.0]],
            [0.5],
            n=n_test,
        )
        self.assertEqual(y_real_inputs.shape, (n_test,))
        self.assertEqual(y_real_inputs.dtype, COMPLEX_DTYPE)


class TestP11DataclassesAndStaticAST(unittest.TestCase):
    def test_dataclasses_are_frozen(self) -> None:
        dummy_gate = GateOutput(
            history_weight=np.zeros(T, dtype=REAL_DTYPE),
            probe_weight=0.5,
            mix_llrs=np.zeros((T, N, 4), dtype=REAL_DTYPE),
            mix_probe_pred=np.zeros(N, dtype=COMPLEX_DTYPE),
            mix_probe_var=np.ones(N, dtype=REAL_DTYPE),
            mix_mean_ray=np.zeros((3, T, N), dtype=COMPLEX_DTYPE),
            mix_var_ray=np.ones((3, T, N), dtype=REAL_DTYPE),
        )
        per_ray = PerRayGateOutput(
            gate_out=dummy_gate,
            ray_weights=np.zeros((3, T), dtype=REAL_DTYPE),
            subset_posteriors=None,
            subset_logliks=None,
        )
        self.assertIsNone(per_ray.ray_diag)
        with self.assertRaises(FrozenInstanceError):
            per_ray.ray_diag = {}  # type: ignore[misc]

        assim = AssimilationResult(
            gate_out=dummy_gate,
            decoded_info={0: np.zeros(64, dtype=np.uint8)},
            assimilated_rows=(9, 11),
            row_weights={9: 0.7, 11: 0.7},
            row_margins={9: 6.0, 11: 6.0},
            n_rounds=2,
            elapsed_ms=1.0,
        )
        with self.assertRaises(FrozenInstanceError):
            assim.n_rounds = 3  # type: ignore[misc]

        obs = OffGridObservation(
            y_a=np.zeros((T, N), dtype=COMPLEX_DTYPE),
            y_b=np.zeros((T, N), dtype=COMPLEX_DTYPE),
            training_rows=(10,),
            probe_row=PROBE_ROW,
            n0=0.25,
            pilot_tx_b=np.zeros((1, N), dtype=COMPLEX_DTYPE),
            pilot_tx_a=np.zeros((4, N // 2), dtype=COMPLEX_DTYPE),
        )
        with self.assertRaises(FrozenInstanceError):
            obs.n0 = 0.5  # type: ignore[misc]

        truth = OffGridTruth(
            regime="offgrid_continuous",
            n_rays=3,
            delays=np.array([0.0, 1.5, 3.2], dtype=REAL_DTYPE),
            powers=np.array([0.6, 0.25, 0.15], dtype=REAL_DTYPE),
            freqs_a=np.zeros(3, dtype=REAL_DTYPE),
            freqs_b=np.zeros(3, dtype=REAL_DTYPE),
            chirps=np.zeros(3, dtype=REAL_DTYPE),
            changed_rays=(),
            shift_row=None,
            corrupted_rows=(),
            data_rows=CORE_ROWS,
            b_info=np.zeros((len(CORE_ROWS), 64), dtype=np.uint8),
            b_code=np.zeros((len(CORE_ROWS), 128), dtype=np.uint8),
            y_b_probe_noisy=np.zeros(N, dtype=COMPLEX_DTYPE),
            y_b_probe_clean=np.zeros(N, dtype=COMPLEX_DTYPE),
            h_b_true_mat=np.zeros((T, N, N), dtype=COMPLEX_DTYPE),
        )
        with self.assertRaises(FrozenInstanceError):
            truth.n_rays = 4  # type: ignore[misc]

    def test_ast_no_hardcoded_n0_and_zero_inference_mode(self) -> None:
        targets = [
            P11_DIR / "__init__.py",
            P11_DIR / "contract.py",
            P11_DIR / "tests" / "__init__.py",
            P11_DIR / "tests" / "test_p11_contract.py",
        ]
        report = check_no_hardcoded_n0(targets, strict=True)
        self.assertTrue(report["clean"], msg=f"AST check failed: {report}")

        for path in targets:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Attribute) and node.attr == "inference_mode":
                    self.fail(f"Forbidden attribute inference_mode in {path}:{node.lineno}")
                if isinstance(node, ast.Name) and node.id == "inference_mode":
                    self.fail(f"Forbidden name inference_mode in {path}:{node.lineno}")

    def test_contract_has_no_top_level_torch_import(self) -> None:
        contract_path = Path(p11_contract.__file__).resolve()
        tree = ast.parse(contract_path.read_text(encoding="utf-8"), filename=str(contract_path))
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    self.assertNotEqual(alias.name.split(".")[0], "torch")
            elif isinstance(node, ast.ImportFrom) and node.module:
                self.assertNotEqual(node.module.split(".")[0], "torch")

        proc = subprocess.run(
            [sys.executable, "-B", "-c", "import sys, p11.contract; assert 'torch' not in sys.modules"],
            cwd=str(REPO_ROOT),
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=f"Fresh import loaded torch: {proc.stderr}")


if __name__ == "__main__":
    unittest.main()
