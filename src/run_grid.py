"""Hyperparameter grid runner over LoRA rank, DoRA on/off, data formats, and random seeds.

Safely appends one row per run to results/results.csv to prevent data loss on disconnects,
captures full environmental metadata, and catches per-run failures gracefully.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import logging
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.env import Timer, build_run_metadata, compute_config_hash
from src.eval import evaluate
from src.train import train

logger = logging.getLogger(__name__)

# Complete schema for results/results.csv
CSV_COLUMNS = [
    "timestamp",
    "git_commit",
    "config_name",
    "config_hash",
    "seed",
    "device",
    "device_name",
    "device_memory_gb",
    "torch_version",
    "transformers_version",
    "peft_version",
    "model_name",
    "is_synthetic",
    "lora_r",
    "use_dora",
    "data_format",
    "num_train_samples",
    "num_eval_samples",
    "wall_clock_seconds",
    "final_train_loss",
    "seen_accuracy",
    "seen_ci_95",
    "unseen_accuracy",
    "unseen_ci_95",
    "generalization_gap",
    "majority_baseline",
    "majority_label",
    "mmstar_accuracy",
    "run_status",
    "predictions_file",
    "error_message",
]


def append_result_row(csv_path: Path, row_dict: Dict[str, Any]) -> None:
    """Safely append a single experiment result row to CSV with flushing.

    Args:
        csv_path: Path to results/results.csv.
        row_dict: Dictionary with values matching CSV_COLUMNS.
    """
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.is_file()

    with csv_path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)
        f.flush()
        os.fsync(f.fileno())

    logger.info("Appended run result to %s", csv_path)


def run_grid(
    config_path: str | Path,
    ranks: Optional[List[int]] = None,
    dora_options: Optional[List[bool]] = None,
    formats: Optional[List[str]] = None,
    seeds: Optional[List[int]] = None,
    max_runs: Optional[int] = None,
) -> None:
    """Execute hyperparameter sweep and log every trial.

    Args:
        config_path: Path to base YAML configuration.
        ranks: List of LoRA ranks to test (default: [16, 32, 64]).
        dora_options: List of DoRA boolean settings (default: [False, True]).
        formats: List of prompt formats (default: ['MCQA', 'QA']).
        seeds: List of seeds to evaluate (default: [42, 123, 456]).
        max_runs: Optional maximum number of runs to execute.
    """
    cfg_file = Path(config_path)
    base_dir = cfg_file.parent.parent
    results_csv = base_dir / "results" / "results.csv"

    with cfg_file.open("r", encoding="utf-8") as f:
        base_config = yaml.safe_load(f)

    # Defaults from research specification
    ranks = ranks or [16, 32, 64]
    dora_options = dora_options or [False, True]
    formats = formats or ["MCQA", "QA"]
    seeds = seeds or [42, 123, 456]

    grid_combinations = list(itertools.product(ranks, dora_options, formats, seeds))
    if max_runs:
        grid_combinations = grid_combinations[:max_runs]

    logger.info("Starting hyperparameter grid: %d total configurations", len(grid_combinations))

    for run_idx, (r, dora, fmt, seed) in enumerate(grid_combinations, start=1):
        run_name = f"grid_r{r}_dora{int(dora)}_{fmt}_s{seed}"
        logger.info(
            "=== Grid Run %d/%d: rank=%d, DoRA=%s, format=%s, seed=%d ===",
            run_idx,
            len(grid_combinations),
            r,
            dora,
            fmt,
            seed,
        )

        overrides: Dict[str, Any] = {
            "experiment_name": run_name,
            "model": {
                "lora_r": r,
                "lora_alpha": r * 2,
                "use_dora": dora,
            },
            "data": {
                "format": fmt,
                "seed": seed,
            },
            "training": {
                "seed": seed,
            },
        }

        timer = Timer()
        with timer:
            try:
                # 1. Train
                train_out = train(config_path, override_args=overrides)
                model = train_out["model"]
                processor = train_out["processor"]
                train_meta = train_out["metadata"]

                # 2. Evaluate
                eval_out = evaluate(
                    config_path,
                    model=model,
                    processor=processor,
                    run_id=run_name,
                )

                seen = eval_out["seen"]
                unseen = eval_out["unseen"]
                mmstar = eval_out.get("mmstar")

                row = {
                    "timestamp": train_meta["timestamp"],
                    "git_commit": train_meta["git_commit"],
                    "config_name": cfg_file.name,
                    "config_hash": train_meta["config_hash"],
                    "seed": seed,
                    "device": train_meta["device"],
                    "device_name": train_meta["device_name"],
                    "device_memory_gb": train_meta["device_memory_gb"],
                    "torch_version": train_meta["torch_version"],
                    "transformers_version": train_meta["transformers_version"],
                    "peft_version": train_meta["peft_version"],
                    "model_name": train_meta["model_name"],
                    "is_synthetic": base_config.get("is_synthetic", False),
                    "lora_r": r,
                    "use_dora": dora,
                    "data_format": fmt,
                    "num_train_samples": train_meta["num_train_samples"],
                    "num_eval_samples": seen["total"] + unseen["total"],
                    "wall_clock_seconds": round(timer.elapsed, 2),
                    "final_train_loss": train_meta.get("final_train_loss"),
                    "seen_accuracy": seen["accuracy_pct"],
                    "seen_ci_95": seen["ci_95"],
                    "unseen_accuracy": unseen["accuracy_pct"],
                    "unseen_ci_95": unseen["ci_95"],
                    "generalization_gap": round(seen["accuracy_pct"] - unseen["accuracy_pct"], 2),
                    "majority_baseline": seen["majority_baseline_pct"],
                    "majority_label": seen["majority_label"],
                    "mmstar_accuracy": mmstar["accuracy_pct"] if mmstar else None,
                    "run_status": "SUCCESS",
                    "predictions_file": eval_out["predictions_file"],
                    "error_message": "",
                }
                append_result_row(results_csv, row)

            except Exception as e:
                err_msg = f"{type(e).__name__}: {str(e)}"
                stack = traceback.format_exc()
                logger.error("Run %s failed: %s\n%s", run_name, err_msg, stack)

                # Still record failed run for complete scientific audit
                meta = build_run_metadata(
                    config_name=cfg_file.name,
                    config_hash=compute_config_hash(cfg_file),
                    seed=seed,
                    model_name=base_config.get("model", {}).get("name", "VLM"),
                    wall_clock_seconds=timer.elapsed,
                )
                row = {
                    "timestamp": meta["timestamp"],
                    "git_commit": meta["git_commit"],
                    "config_name": cfg_file.name,
                    "config_hash": meta["config_hash"],
                    "seed": seed,
                    "device": meta["device"],
                    "device_name": meta["device_name"],
                    "device_memory_gb": meta["device_memory_gb"],
                    "torch_version": meta["torch_version"],
                    "transformers_version": meta["transformers_version"],
                    "peft_version": meta["peft_version"],
                    "model_name": meta["model_name"],
                    "is_synthetic": base_config.get("is_synthetic", False),
                    "lora_r": r,
                    "use_dora": dora,
                    "data_format": fmt,
                    "num_train_samples": 0,
                    "num_eval_samples": 0,
                    "wall_clock_seconds": round(timer.elapsed, 2),
                    "final_train_loss": None,
                    "seen_accuracy": None,
                    "seen_ci_95": None,
                    "unseen_accuracy": None,
                    "unseen_ci_95": None,
                    "generalization_gap": None,
                    "majority_baseline": None,
                    "majority_label": None,
                    "mmstar_accuracy": None,
                    "run_status": "FAILED",
                    "predictions_file": "",
                    "error_message": err_msg,
                }
                append_result_row(results_csv, row)


def main() -> None:
    """CLI runner for hyperparameter grid search."""
    parser = argparse.ArgumentParser(description="LiveVQA Hyperparameter Grid Runner")
    parser.add_argument("--config", type=str, required=True, help="Base config YAML")
    parser.add_argument("--ranks", type=int, nargs="+", default=None, help="LoRA ranks to test")
    parser.add_argument("--dora", type=str, nargs="+", default=None, help="DoRA settings (true/false)")
    parser.add_argument("--formats", type=str, nargs="+", default=None, help="Data formats (MCQA QA)")
    parser.add_argument("--seeds", type=int, nargs="+", default=None, help="Random seeds")
    parser.add_argument("--max-runs", type=int, default=None, help="Maximum number of grid runs")
    parser.add_argument("--quick", action="store_true", help="Run 1 trial only for sanity test")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    dora_opts = None
    if args.dora:
        dora_opts = [d.lower() in ("true", "1", "yes") for d in args.dora]

    max_runs = 1 if args.quick else args.max_runs

    run_grid(
        config_path=args.config,
        ranks=args.ranks,
        dora_options=dora_opts,
        formats=args.formats,
        seeds=args.seeds,
        max_runs=max_runs,
    )


if __name__ == "__main__":
    main()
