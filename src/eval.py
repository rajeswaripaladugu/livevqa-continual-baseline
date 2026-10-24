"""Evaluation module with strict letter matching, Wilson 95% confidence interval, and raw predictions logging.

Audits every prediction, reports majority-class baselines alongside seen/unseen/MMStar scores,
and rejects ambiguous or verbose non-answers.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml
from PIL import Image

from src.data import (
    VQASample,
    compute_answer_distribution,
    format_mcqa_prompt,
    load_dataset,
    split_by_image,
)
from src.env import Timer, build_run_metadata, compute_config_hash
from src.models import load_model_and_processor

logger = logging.getLogger(__name__)


def extract_answer_letter(raw_text: str) -> Optional[str]:
    """Strictly extract a multiple-choice letter (A, B, C, D) from raw model output.

    Accepts:
        - "C", "(c)", "C.", "(C)", "C)"
        - "Answer: C", "Answer: (C)", "Answer is C"
        - "Option C", "Choice: C"
        - "The correct answer is (B)"

    Rejects:
        - Empty or whitespace strings
        - Verbose paragraphs without an explicit option indicator
        - Multiple conflicting option mentions (e.g., "Either A or B")

    Args:
        raw_text: Raw generation string from model.

    Returns:
        Uppercase letter ('A', 'B', 'C', 'D') or None if extraction fails/is rejected.
    """
    if not raw_text or not raw_text.strip():
        return None

    text = raw_text.strip()

    # Pattern 1: Exact standalone letter with optional surrounding brackets/punctuation
    match_single = re.fullmatch(r"[\s\(\[\{]*([A-Za-z])[\s\)\]\}\.\:\,]*", text)
    if match_single:
        letter = match_single.group(1).upper()
        if letter in "ABCD":
            return letter

    # Pattern 2: Starts with letter followed by punctuation or space: "A. The object...", "B) ..."
    match_leading = re.match(r"^[\s\(\[]*([A-Za-z])[\s\)\]\.\:\,]+", text)
    if match_leading:
        letter = match_leading.group(1).upper()
        if letter in "ABCD":
            # Guard against starting words like "A dog is..." by checking length or following tokens
            tokens = text.split()
            if len(tokens) <= 5 or text.startswith(("A.", "B.", "C.", "D.", "A)", "B)", "C)", "D)")):
                return letter

    # Pattern 3: Explicit answer prefix anywhere in short output
    # e.g., "Answer: C", "The correct answer is (D)", "Option: B"
    prefix_patterns = [
        r"(?:the\s+)?(?:correct\s+)?(?:answer|choice|option)\s*(?:is)?\s*[:\-\)]?\s*\(?([A-Za-z])\)?",
        r"(?:select|choose)\s*\(?([A-Za-z])\)?",
    ]

    for pat in prefix_patterns:
        matches = re.findall(pat, text, flags=re.IGNORECASE)
        valid_letters = [m.upper() for m in matches if m.upper() in "ABCD"]
        if len(set(valid_letters)) == 1:
            return valid_letters[0]
        elif len(set(valid_letters)) > 1:
            # Conflicting letters in text -> reject
            return None

    return None


def compute_wilson_confidence_interval(
    num_correct: int,
    total: int,
    confidence: float = 0.95,
) -> Tuple[float, float]:
    """Compute the Wilson score 95% confidence interval for a binomial proportion.

    Formula:
        center = (p + z^2 / (2n)) / (1 + z^2 / n)
        margin = (z / (1 + z^2 / n)) * sqrt(p(1 - p)/n + z^2 / (4n^2))

    Args:
        num_correct: Count of successes.
        total: Total number of trials.
        confidence: Confidence level (default: 0.95 -> z = 1.95996).

    Returns:
        Tuple of (ci_lower, ci_upper) clamped between 0.0 and 1.0.
    """
    if total <= 0:
        return 0.0, 0.0

    p_hat = num_correct / total
    z = 1.959963984540054  # Normal quantile for 95% two-sided CI

    denominator = 1.0 + (z**2) / total
    center = (p_hat + (z**2) / (2.0 * total)) / denominator
    spread = (z / denominator) * math.sqrt(
        (p_hat * (1.0 - p_hat) / total) + ((z**2) / (4.0 * (total**2)))
    )

    ci_lower = max(0.0, center - spread)
    ci_upper = min(1.0, center + spread)
    return round(ci_lower, 4), round(ci_upper, 4)


def evaluate_split(
    model: torch.nn.Module,
    processor: Any,
    samples: List[VQASample],
    split_name: str,
    device: torch.device,
    base_dir: Optional[Path] = None,
    max_new_tokens: int = 16,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Evaluate model on a list of samples and collect raw audit predictions.

    Args:
        model: VLM model.
        processor: Multimodal processor.
        samples: List of VQASample items.
        split_name: Name of split (e.g. 'train_seen', 'eval_unseen', 'mmstar').
        device: PyTorch device.
        base_dir: Root directory for relative images.
        max_new_tokens: Tokens to generate.

    Returns:
        Tuple of (metrics_dict, raw_predictions_list).
    """
    if not samples:
        return {
            "split": split_name,
            "total": 0,
            "correct": 0,
            "accuracy": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "majority_baseline": 0.0,
            "majority_label": None,
        }, []

    model.eval()
    base_dir = base_dir or Path(".")
    dist_stats = compute_answer_distribution(samples)
    majority_baseline = dist_stats["majority_baseline"]
    majority_label = dist_stats["majority_label"]

    raw_predictions: List[Dict[str, Any]] = []
    correct_count = 0

    with torch.no_grad():
        for s in samples:
            img_path = Path(s.image)
            if not img_path.is_file():
                img_path = base_dir / s.image
            if not img_path.is_file():
                img_path = base_dir / "data" / s.image

            if img_path.is_file():
                img = Image.open(img_path).convert("RGB")
            else:
                img = Image.new("RGB", (128, 128), color=(220, 220, 220))

            prompt = format_mcqa_prompt(s.question, options=s.options)
            inputs = processor(
                text=prompt,
                images=img,
                return_tensors="pt",
            )

            for k in ["input_ids", "attention_mask", "pixel_values"]:
                if k in inputs and inputs[k] is not None:
                    inputs[k] = inputs[k].to(device)

            if hasattr(model, "generate"):
                out_ids = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                )
                # Decode only new tokens
                input_len = inputs["input_ids"].shape[1]
                gen_ids = out_ids[0][input_len:] if out_ids.shape[1] > input_len else out_ids[0]
                gen_text = processor.decode(gen_ids, skip_special_tokens=True)
            else:
                gen_text = ""

            parsed_answer = extract_answer_letter(gen_text)
            ground_truth = s.answer.upper().strip()
            is_correct = bool(parsed_answer is not None and parsed_answer == ground_truth)

            if is_correct:
                correct_count += 1

            raw_predictions.append({
                "id": s.id,
                "split": split_name,
                "ground_truth": ground_truth,
                "raw_output": gen_text.strip(),
                "parsed_answer": parsed_answer,
                "is_correct": is_correct,
                "is_synthetic": s.is_synthetic,
            })

    total = len(samples)
    acc = round(correct_count / total, 4) if total > 0 else 0.0
    ci_low, ci_high = compute_wilson_confidence_interval(correct_count, total)

    metrics = {
        "split": split_name,
        "total": total,
        "correct": correct_count,
        "accuracy": acc,
        "accuracy_pct": round(acc * 100, 2),
        "ci_lower": ci_low,
        "ci_upper": ci_high,
        "ci_95": f"[{ci_low*100:.1f}%, {ci_high*100:.1f}%]",
        "majority_baseline": majority_baseline,
        "majority_baseline_pct": round(majority_baseline * 100, 2),
        "majority_label": majority_label,
    }

    logger.info(
        "Split: %s | Acc: %.2f%% (95%% CI: %s) | Majority Baseline: %.2f%% (Label: %s)",
        split_name,
        metrics["accuracy_pct"],
        metrics["ci_95"],
        metrics["majority_baseline_pct"],
        majority_label,
    )
    return metrics, raw_predictions


def evaluate(
    config_path: str | Path,
    model: Optional[torch.nn.Module] = None,
    processor: Optional[Any] = None,
    run_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Full evaluation orchestrator over seen, unseen, and optional MMStar splits.

    Saves every raw prediction to results/predictions_<run_id>.jsonl for auditing.

    Args:
        config_path: Path to configuration YAML.
        model: Optional pre-loaded or trained model instance.
        processor: Optional pre-loaded processor.
        run_id: Unique identifier for audit logging.

    Returns:
        Summary evaluation results dictionary.
    """
    cfg_file = Path(config_path)
    with cfg_file.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    base_dir = cfg_file.parent.parent

    if model is None or processor is None:
        model, processor = load_model_and_processor(config)
        model.to(device)

    data_cfg = config.get("data", {})
    seed = config.get("training", {}).get("seed", 42)

    # Load and split dataset
    train_path = data_cfg.get("train_path", "data/synthetic_smoke.jsonl")
    resolved_path = train_path if Path(train_path).is_file() else (base_dir / train_path)
    all_samples = load_dataset(resolved_path, base_dir=base_dir)

    if data_cfg.get("split_by_image", True):
        seen_samples, unseen_samples = split_by_image(
            all_samples,
            train_ratio=data_cfg.get("train_split_ratio", 0.8),
            seed=seed,
        )
    else:
        seen_samples = all_samples
        unseen_samples = []

    # Optional MMStar dataset
    mmstar_samples: List[VQASample] = []
    mmstar_path = data_cfg.get("mmstar_path")
    if mmstar_path:
        res_mmstar = mmstar_path if Path(mmstar_path).is_file() else (base_dir / mmstar_path)
        if res_mmstar.is_file():
            mmstar_samples = load_dataset(res_mmstar, base_dir=base_dir)

    all_raw_predictions: List[Dict[str, Any]] = []

    # Evaluate seen
    seen_metrics, seen_preds = evaluate_split(
        model, processor, seen_samples, "seen", device, base_dir=base_dir
    )
    all_raw_predictions.extend(seen_preds)

    # Evaluate unseen
    unseen_metrics, unseen_preds = evaluate_split(
        model, processor, unseen_samples, "unseen", device, base_dir=base_dir
    )
    all_raw_predictions.extend(unseen_preds)

    # Evaluate MMStar if present
    mmstar_metrics = None
    if mmstar_samples:
        mmstar_metrics, mmstar_preds = evaluate_split(
            model, processor, mmstar_samples, "mmstar", device, base_dir=base_dir
        )
        all_raw_predictions.extend(mmstar_preds)

    # Save raw predictions for complete research auditability
    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    if not run_id:
        run_id = f"{config.get('experiment_name', 'run')}_{int(datetime.now(timezone.utc).timestamp())}"

    pred_file = results_dir / f"predictions_{run_id}.jsonl"
    with pred_file.open("w", encoding="utf-8") as f:
        for p in all_raw_predictions:
            f.write(json.dumps(p) + "\n")
    logger.info("Saved %d audited raw predictions to %s", len(all_raw_predictions), pred_file)

    # Append evaluation record to results.csv for research auditing
    results_csv = results_dir / "results.csv"
    from src.run_grid import append_result_row
    meta = build_run_metadata(
        config_name=cfg_file.name,
        config_hash=compute_config_hash(cfg_file),
        seed=seed,
        model_name=config.get("model", {}).get("name", "VLM"),
        num_train_samples=len(seen_samples),
        num_eval_samples=len(seen_samples) + len(unseen_samples),
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
        "is_synthetic": config.get("is_synthetic", False),
        "lora_r": config.get("model", {}).get("lora_r", 16),
        "use_dora": config.get("model", {}).get("use_dora", False),
        "data_format": data_cfg.get("format", "MCQA"),
        "num_train_samples": len(seen_samples),
        "num_eval_samples": len(seen_samples) + len(unseen_samples),
        "wall_clock_seconds": round(meta.get("wall_clock_seconds", 0.0), 2),
        "final_train_loss": None,
        "seen_accuracy": seen_metrics["accuracy_pct"],
        "seen_ci_95": seen_metrics["ci_95"],
        "unseen_accuracy": unseen_metrics["accuracy_pct"],
        "unseen_ci_95": unseen_metrics["ci_95"],
        "generalization_gap": round(seen_metrics["accuracy_pct"] - unseen_metrics["accuracy_pct"], 2),
        "majority_baseline": seen_metrics["majority_baseline_pct"],
        "majority_label": seen_metrics["majority_label"],
        "mmstar_accuracy": mmstar_metrics["accuracy_pct"] if mmstar_metrics else None,
        "run_status": "SUCCESS",
        "predictions_file": str(pred_file),
        "error_message": "",
    }
    append_result_row(results_csv, row)

    eval_summary = {
        "run_id": run_id,
        "config_name": cfg_file.name,
        "seen": seen_metrics,
        "unseen": unseen_metrics,
        "generalization_gap": round(seen_metrics["accuracy"] - unseen_metrics["accuracy"], 4),
        "mmstar": mmstar_metrics,
        "predictions_file": str(pred_file),
    }

    return eval_summary


def main() -> None:
    """CLI runner for evaluation."""
    parser = argparse.ArgumentParser(description="LiveVQA Continual Baseline Evaluator")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    parser.add_argument("--run-id", type=str, default=None, help="Identifier for predictions file")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    evaluate(args.config, run_id=args.run_id)


if __name__ == "__main__":
    main()
