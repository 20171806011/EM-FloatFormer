"""Unit tests for `p11.contract` and `p11` package invariants."""
from __future__ import annotations

import ast
from dataclasses import FrozenInstanceError
import math
import os
from pathlib import Path
import subprocess
import sys
import unittest

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
            (88_000_000, "p8_manifold_dev_ladder"),
            (105_001_000, "p5_p6"),
            (702_611_190_000, "p7"),
            (802_611_190_000, "p8_manifold"),
            (802_611_790_303, "p8_manifold"),
            (812_611_190_000, "p8_prefix"),
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

    def test_is_historical_seed_rejects_bool_and_float(self) -> None:
        for bad in (True, False, 1.5, "42"):
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
                with self.assertRaises(RuntimeError):
                    enforce_determinism(require_env=True)

    def test_permissive_mode_sets_env_and_torch(self) -> None:
        with self._patched_thread_env({"OPENBLAS_NUM_THREADS": "8", "MKL_NUM_THREADS": "4"}):
            status = enforce_determinism(require_env=False)
            self.assertFalse(status["require_env"])
            self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "1")
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "1")
            self.assertEqual(os.environ["MKL_NUM_THREADS"], "1")
            self.assertEqual(torch.get_num_threads(), 1)
            self.assertTrue(torch.are_deterministic_algorithms_enabled())


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
        for bad_delay in (math.nan, math.inf, -math.inf, True):
            with self.assertRaises(ValueError):
                fractional_delay_matrix(bad_delay)  # type: ignore[arg-type]
            with self.assertRaises(ValueError):
                fractional_delay_freq_response(bad_delay)  # type: ignore[arg-type]
        with self.assertRaises(ValueError):
            build_offgrid_sc_channel_matrix(np.zeros((2, N), dtype=COMPLEX_DTYPE), np.zeros(3, dtype=REAL_DTYPE))
        with self.assertRaises(ValueError):
            apply_fractional_sc_operator(
                np.zeros(N - 1, dtype=COMPLEX_DTYPE),
                np.zeros((2, N), dtype=COMPLEX_DTYPE),
                np.zeros(2, dtype=REAL_DTYPE),
            )


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
