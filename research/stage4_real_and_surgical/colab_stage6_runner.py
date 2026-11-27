"""CLI entrypoint for Google Colab Stage 5 & Stage 6 (4B -> 9B -> 27B Mixed-NF4)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Optional

_FILE_P = Path(__file__).resolve()
REPO_ROOT = _FILE_P.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from research.stage4_real_and_surgical.colab_stage6_scaling import (  # noqa: E402
    MODEL_SPECS,
    RELEASE_TAG,
    detect_colab_gpu_profile,
    download_stage4_release_artifacts,
    run_scaled_model_stage6,
    run_stage6_4b_full,
)


def _parse_tri_bool(raw: str) -> Optional[bool]:
    val = str(raw).strip().lower()
    if val in ("auto", "none", ""):
        return None
    if val in ("1", "true", "yes", "on"):
        return True
    if val in ("0", "false", "no", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected auto/true/false, got {raw!r}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Google Colab Stage 5 & Stage 6 runner (zero Modal GPU spend)."
    )
    p.add_argument(
        "--stage",
        required=True,
        choices=("download_artifacts", "stage6_4b", "stage6_9b", "stage6_27b", "all"),
        help="Execution stage to run.",
    )
    p.add_argument(
        "--work-dir",
        type=Path,
        default=REPO_ROOT / ".colab_work",
        help="Working directory for cached checkpoints, K-FAC, and Real-SDR .npz bundles.",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "research" / "stage4_real_and_surgical" / "results",
        help="Directory for output JSON reports.",
    )
    p.add_argument(
        "--release-tag",
        type=str,
        default=RELEASE_TAG,
        help="GitHub Release tag hosting verified Stage 4 artifacts.",
    )
    p.add_argument("--seed", type=int, default=202611, help="Deterministic RNG seed.")
    p.add_argument("--steps-stage6", type=int, default=240, help="Stage 6 adapter steps per arm.")
    p.add_argument(
        "--g1-blocks-scaled",
        type=int,
        default=32,
        help="G1 blind blocks for scaled 9B/27B synthetic blind checks.",
    )
    p.add_argument(
        "--run-u1-scaled",
        nargs="?",
        const="true",
        default="auto",
        type=str,
        help="Run U1 training on 9B/27B: auto (profile default), true, or false.",
    )
    p.add_argument(
        "--force-quant",
        type=str,
        default=None,
        choices=("bf16", "mixed_nf4"),
        help="Override auto-detected quantization mode (bf16 or mixed_nf4).",
    )
    p.add_argument("--eval-bs", type=int, default=None, help="Override auto eval batch size.")
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect hardware profile, resolved paths, and execution plan without loading models.",
    )
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    run_u1_opt = _parse_tri_bool(args.run_u1_scaled)

    profile = detect_colab_gpu_profile()
    header = {
        "stage": args.stage,
        "repo_root": str(REPO_ROOT),
        "work_dir": str(args.work_dir),
        "results_dir": str(args.results_dir),
        "release_tag": args.release_tag,
        "gpu_profile": profile,
        "force_quant": args.force_quant,
        "run_u1_scaled": run_u1_opt,
        "dry_run": bool(args.dry_run),
    }
    print(json.dumps(header, ensure_ascii=False, indent=2), flush=True)

    if args.dry_run:
        return 0

    args.work_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    if args.stage in ("download_artifacts", "all"):
        dl_rep = download_stage4_release_artifacts(
            args.work_dir, release_tag=args.release_tag, repo_root=REPO_ROOT
        )
        print(json.dumps(dl_rep, ensure_ascii=False, indent=2), flush=True)
        if args.stage == "download_artifacts":
            return 0

    if args.stage in ("stage6_4b", "all"):
        q4 = args.force_quant or profile["models"]["4b"].get("quant", MODEL_SPECS["4b"]["default_quant"])
        rep4 = run_stage6_4b_full(
            REPO_ROOT,
            args.work_dir,
            args.results_dir,
            seed=args.seed,
            quant=q4,
            steps_stage6=args.steps_stage6,
            eval_bs=args.eval_bs,
        )
        print(
            json.dumps(
                {
                    "stage6_4b_done": True,
                    "real_sdr_holm": rep4["real_sdr_4b"]["holm_bonferroni_real_sdr"],
                    "cnsft_real_sdr_audit": rep4["real_sdr_4b_cnsft"]["audit"],
                    "stage6_4b_16v4": rep4["stage6_4b"]["paired_16bit_vs_4bit_frozen_host"],
                    "stage6_4b_cnsft": rep4["stage6_4b_cnsft"]["pooled"],
                },
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )

    if args.stage in ("stage6_9b", "all"):
        if args.stage == "all" and not profile["models"]["9b"].get("supported", False) and args.force_quant is None:
            print("[Colab Runner] Skipping stage6_9b on unsupported VRAM tier (pass --stage stage6_9b --force-quant mixed_nf4 to force).", flush=True)
        else:
            rep9 = run_scaled_model_stage6(
                "9b",
                REPO_ROOT,
                args.work_dir,
                args.results_dir,
                seed=args.seed,
                quant_override=args.force_quant,
                steps_stage6=args.steps_stage6,
                g1_blocks=args.g1_blocks_scaled,
                eval_bs=args.eval_bs,
                run_u1=run_u1_opt,
            )
            print(
                json.dumps(
                    {
                        "stage6_9b_done": True,
                        "quant_mode": rep9["quant_mode"],
                        "u0_blind_g1": rep9["cn_sft_u0"]["blind"]["G1_main_extrap_both"],
                        "u0_audit": rep9["cn_sft_u0"]["audit"],
                        "stage6_16v4": rep9["stage6_active_eig"]["paired_16bit_vs_4bit_frozen_host"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )

    if args.stage in ("stage6_27b", "all"):
        if args.stage == "all" and not profile["models"]["27b"].get("supported", False) and args.force_quant is None:
            print("[Colab Runner] Skipping stage6_27b on unsupported VRAM tier (<35 GiB).", flush=True)
        else:
            rep27 = run_scaled_model_stage6(
                "27b",
                REPO_ROOT,
                args.work_dir,
                args.results_dir,
                seed=args.seed,
                quant_override=args.force_quant,
                steps_stage6=args.steps_stage6,
                g1_blocks=args.g1_blocks_scaled,
                eval_bs=args.eval_bs,
                run_u1=run_u1_opt,
            )
            print(
                json.dumps(
                    {
                        "stage6_27b_done": True,
                        "quant_mode": rep27["quant_mode"],
                        "quant_audit": rep27["quant_audit"],
                        "u0_blind_g1": rep27["cn_sft_u0"]["blind"]["G1_main_extrap_both"],
                        "u0_audit": rep27["cn_sft_u0"]["audit"],
                        "stage6_16v4": rep27["stage6_active_eig"]["paired_16bit_vs_4bit_frozen_host"],
                    },
                    ensure_ascii=False,
                    indent=2,
                ),
                flush=True,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
