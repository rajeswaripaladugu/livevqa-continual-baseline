"""Laptop-friendly replication runner for the LiveVQA continual baseline.

Executes within CPU and memory constraints (~8GB RAM, no GPU required):
1. Visual Knowledge Updating across 3 formats (MCQA vs QA vs QAR) [Paper Section 4.1]
2. LoRA / DoRA adapter rank comparison (r=8, 16) [Paper Section 4.2]
3. Continual learning sequential updates (K=3 chunks, BWT, Forgetting Matrix)
4. Diagnostic plot generation and README results updating.
"""

from __future__ import annotations

import argparse
import copy
import logging
from pathlib import Path
from typing import Any, Dict, List

import yaml

from src.analyze import main as analyze_main
from src.continual import run_continual_learning
from src.eval import evaluate
from src.train import train

logger = logging.getLogger(__name__)


def run_laptop_replication(
    base_config_path: str = "configs/laptop_cpu.yaml",
    run_formats: bool = True,
    run_adapters: bool = True,
    run_continual: bool = True,
) -> None:
    """Run laptop-scale replication of LiveVQA knowledge updating baseline."""
    cfg_path = Path(base_config_path)
    if not cfg_path.is_file():
        raise FileNotFoundError(f"Config not found: {cfg_path}")

    with cfg_path.open("r", encoding="utf-8") as f:
        base_cfg = yaml.safe_load(f)

    temp_cfg_dir = cfg_path.parent / "temp"
    temp_cfg_dir.mkdir(parents=True, exist_ok=True)

    # 1. Format Ablation (MCQA, QA, QAR) - Replicating Figure 6
    if run_formats:
        logger.info("==================================================================")
        logger.info("PHASE 1: Format Ablation (MCQA vs. QA vs. QAR) [Paper Section 4.1]")
        logger.info("==================================================================")
        for fmt in ["MCQA", "QA", "QAR"]:
            cfg = copy.deepcopy(base_cfg)
            cfg["experiment_name"] = f"laptop_format_{fmt.lower()}"
            cfg["data"]["format"] = fmt
            # Train for 1 epoch
            temp_file = temp_cfg_dir / f"{cfg['experiment_name']}.yaml"
            with temp_file.open("w", encoding="utf-8") as f:
                yaml.dump(cfg, f)

            logger.info(">>> Running Training & Evaluation for format: %s", fmt)
            train_res = train(temp_file)
            evaluate(temp_file, model=train_res["model"], processor=train_res["processor"])

    # 2. Adapter Capacity & DoRA Ablation (r=8, 16, DoRA) - Replicating Figure 7
    if run_adapters:
        logger.info("==================================================================")
        logger.info("PHASE 2: Adapter Scaling & LoRA vs. DoRA [Paper Section 4.2]")
        logger.info("==================================================================")
        for r in [8, 16]:
            for use_dora in [False, True]:
                dora_label = "dora" if use_dora else "lora"
                exp_name = f"laptop_adapter_{dora_label}_r{r}"
                cfg = copy.deepcopy(base_cfg)
                cfg["experiment_name"] = exp_name
                cfg["model"]["lora_r"] = r
                cfg["model"]["lora_alpha"] = 2 * r
                cfg["model"]["use_dora"] = use_dora
                temp_file = temp_cfg_dir / f"{exp_name}.yaml"
                with temp_file.open("w", encoding="utf-8") as f:
                    yaml.dump(cfg, f)

                logger.info(">>> Running Adapter: %s (rank r=%d)", dora_label.upper(), r)
                train_res = train(temp_file)
                evaluate(temp_file, model=train_res["model"], processor=train_res["processor"])

    # 3. Continual Learning Sequential Updating (K=3 chunks)
    if run_continual:
        logger.info("==================================================================")
        logger.info("PHASE 3: Continual Learning Sequential Updating [Roadmap Phase 2]")
        logger.info("==================================================================")
        continual_cfg = copy.deepcopy(base_cfg)
        continual_cfg["experiment_name"] = "laptop_continual_k3"
        temp_file = temp_cfg_dir / "laptop_continual_k3.yaml"
        with temp_file.open("w", encoding="utf-8") as f:
            yaml.dump(continual_cfg, f)

        res = run_continual_learning(
            temp_file,
            num_chunks=3,
            replay_capacity=4,
            epochs_per_stage=1,
        )
        logger.info("Continual Learning Complete: BWT = %.2f%%, Avg Forgetting = %.2f%%", res["bwt_pct"], res["average_forgetting_pct"])

    # 4. Generate all summary plots and inject into README
    logger.info("==================================================================")
    logger.info("PHASE 4: Compiling Results and Generating Diagnostic Visualizations")
    logger.info("==================================================================")
    analyze_main()
    logger.info("Laptop replication baseline pipeline completed successfully!")


def main() -> None:
    """CLI parser for laptop runner."""
    parser = argparse.ArgumentParser(description="LiveVQA Laptop Replication Runner")
    parser.add_argument("--config", type=str, default="configs/laptop_cpu.yaml", help="Path to base laptop config")
    parser.add_argument("--quick", action="store_true", help="Run a quick single-trial verification")
    parser.add_argument("--formats-only", action="store_true", help="Run only format ablation")
    parser.add_argument("--adapters-only", action="store_true", help="Run only adapter ablation")
    parser.add_argument("--continual-only", action="store_true", help="Run only continual learning")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.quick:
        # Just run single format and continual
        run_laptop_replication(args.config, run_formats=True, run_adapters=False, run_continual=True)
    elif args.formats_only:
        run_laptop_replication(args.config, run_formats=True, run_adapters=False, run_continual=False)
    elif args.adapters_only:
        run_laptop_replication(args.config, run_formats=False, run_adapters=True, run_continual=False)
    elif args.continual_only:
        run_laptop_replication(args.config, run_formats=False, run_adapters=False, run_continual=True)
    else:
        # Full laptop suite
        run_laptop_replication(args.config, run_formats=True, run_adapters=True, run_continual=True)


if __name__ == "__main__":
    main()
