"""Phase 2 Continual Learning: sequential updates, forgetting matrix, and experience replay.

Splits visual knowledge updates into K chunks (chronological if date field exists, else ordered),
trains sequentially, evaluates all chunks after each stage, and computes the K x K accuracy matrix,
Backward Transfer (BWT), and Average Forgetting.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml

from src.data import VQASample, compute_answer_distribution, load_dataset
from src.env import Timer, build_run_metadata, compute_config_hash
from src.eval import evaluate_split
from src.models import load_model_and_processor
from src.train import VQADataset, build_masked_batch, compute_steps

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """Experience-replay buffer for rehearsal in continual learning."""

    def __init__(self, capacity: int = 16, seed: int = 42) -> None:
        self.capacity = capacity
        self.buffer: List[VQASample] = []
        self.rng = random.Random(seed)

    def add(self, samples: List[VQASample]) -> None:
        """Add samples using reservoir sampling to maintain equal representation."""
        if self.capacity <= 0:
            return

        for s in samples:
            if len(self.buffer) < self.capacity:
                self.buffer.append(s)
            else:
                idx = self.rng.randint(0, len(self.buffer))
                if idx < self.capacity:
                    self.buffer[idx] = s

    def sample(self, count: int) -> List[VQASample]:
        """Draw samples from buffer."""
        if not self.buffer or count <= 0:
            return []
        if count >= len(self.buffer):
            return list(self.buffer)
        return self.rng.sample(self.buffer, count)

    def __len__(self) -> int:
        return len(self.buffer)


def partition_into_chunks(
    samples: List[VQASample],
    num_chunks: int = 3,
) -> List[List[VQASample]]:
    """Partition dataset into K chunks.

    Sorts chronologically if date/timestamp exists on samples, otherwise maintains list order.

    Args:
        samples: List of VQASample objects.
        num_chunks: Number of sequential learning stages K.

    Returns:
        List of K sample lists.
    """
    if num_chunks <= 1:
        return [samples]

    # Check if dates exist
    has_dates = any(s.date is not None for s in samples)
    if has_dates:
        sorted_samples = sorted(samples, key=lambda s: s.date or "")
        logger.info("Partitioning %d samples chronologically into %d chunks", len(samples), num_chunks)
    else:
        sorted_samples = list(samples)
        logger.info("Partitioning %d samples in sequential order into %d chunks", len(samples), num_chunks)

    chunk_size = math.ceil(len(sorted_samples) / num_chunks)
    chunks = [
        sorted_samples[i * chunk_size : (i + 1) * chunk_size]
        for i in range(num_chunks)
    ]
    # Filter out empty chunks if any
    return [c for c in chunks if c]


def run_continual_learning(
    config_path: str | Path,
    num_chunks: int = 3,
    replay_capacity: int = 0,
    epochs_per_stage: int = 1,
) -> Dict[str, Any]:
    """Execute continual learning evaluation across K chunks.

    Produces K x K accuracy matrix, Backward Transfer (BWT), and Average Forgetting.

    Args:
        config_path: Path to YAML configuration.
        num_chunks: Number of sequential stages K (default: 3).
        replay_capacity: Replay buffer size (0 to disable).
        epochs_per_stage: Epochs per chunk.

    Returns:
        Dictionary with accuracy matrix, forgetting scores, and metadata.
    """
    cfg_file = Path(config_path)
    base_dir = cfg_file.parent.parent

    with cfg_file.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = config.get("training", {}).get("seed", 42)

    # Load dataset
    data_cfg = config.get("data", {})
    train_path = data_cfg.get("train_path", "data/synthetic_smoke.jsonl")
    res_path = train_path if Path(train_path).is_file() else (base_dir / train_path)
    all_samples = load_dataset(res_path, base_dir=base_dir)

    chunks = partition_into_chunks(all_samples, num_chunks=num_chunks)
    K = len(chunks)
    logger.info("Loaded %d chunks for continual learning", K)

    # Initialize model & processor
    model, processor = load_model_and_processor(config)
    model.to(device)

    replay_buffer = ReplayBuffer(capacity=replay_capacity, seed=seed)

    # Accuracy Matrix: R[i][j] = accuracy on chunk j after training stage i (0-indexed)
    R = np.zeros((K, K), dtype=float)

    timer = Timer()
    with timer:
        for stage_idx in range(K):
            current_chunk = chunks[stage_idx]
            logger.info("=== Continual Stage %d/%d: Training on Chunk %d (%d samples) ===", stage_idx + 1, K, stage_idx + 1, len(current_chunk))

            # Mix in replay samples if enabled
            train_items = list(current_chunk)
            if replay_capacity > 0 and len(replay_buffer) > 0:
                replay_samples = replay_buffer.sample(min(len(replay_buffer), len(current_chunk) // 2))
                train_items.extend(replay_samples)
                logger.info("Added %d replay samples to training stage %d", len(replay_samples), stage_idx + 1)

            # Train on stage_idx data
            data_format = config.get("data", {}).get("format", "MCQA")
            dataset = VQADataset(
                train_items,
                processor,
                format_type=data_format,
                max_length=config.get("training", {}).get("max_length", 256),
                base_dir=base_dir,
            )
            batch_size = max(1, config.get("training", {}).get("batch_size", 2))
            dataloader = torch.utils.data.DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=True,
                collate_fn=lambda items: build_masked_batch(items, processor, max_length=config.get("training", {}).get("max_length", 256)),
            )

            trainable_params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.AdamW(trainable_params, lr=float(config.get("training", {}).get("learning_rate", 1e-4)))

            model.train()
            for ep in range(epochs_per_stage):
                for batch in dataloader:
                    for k in ["input_ids", "attention_mask", "pixel_values", "labels"]:
                        if k in batch and batch[k] is not None:
                            batch[k] = batch[k].to(device)

                    optimizer.zero_grad()
                    out = model(**batch)
                    if out.loss is not None:
                        out.loss.backward()
                        optimizer.step()

            # Add current chunk to replay buffer for future stages
            if replay_capacity > 0:
                replay_buffer.add(current_chunk)

            # Evaluate on ALL K chunks after stage_idx
            model.eval()
            for eval_idx in range(K):
                eval_samples = chunks[eval_idx]
                metrics, _ = evaluate_split(
                    model,
                    processor,
                    eval_samples,
                    f"chunk_{eval_idx+1}",
                    device,
                    base_dir=base_dir,
                    format_type=config.get("data", {}).get("format", "MCQA"),
                )
                R[stage_idx, eval_idx] = metrics["accuracy"]
                logger.info("Stage %d -> Eval on Chunk %d: Accuracy = %.2f%%", stage_idx + 1, eval_idx + 1, metrics["accuracy"] * 100)

    # Compute Continual Metrics
    # Backward Transfer: BWT = (1 / (K - 1)) * sum_{j=1}^{K-1} (R[K-1, j] - R[j, j])
    bwt = 0.0
    if K > 1:
        bwt = float(np.mean([R[K - 1, j] - R[j, j] for j in range(K - 1)]))

    # Forgetting: F_j = max_{l in 0..K-2} (R[l, j] - R[K-1, j])
    # Average Forgetting = (1 / (K - 1)) * sum_{j=0}^{K-2} max(0, F_j)
    forgetting_per_chunk: List[float] = []
    if K > 1:
        for j in range(K - 1):
            peak_prior = float(np.max(R[: K - 1, j]))
            final_acc = float(R[K - 1, j])
            f_j = max(0.0, peak_prior - final_acc)
            forgetting_per_chunk.append(round(f_j, 4))
        avg_forgetting = float(np.mean(forgetting_per_chunk))
    else:
        avg_forgetting = 0.0

    results_dir = base_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    matrix_file = results_dir / f"continual_matrix_{config.get('experiment_name', 'smoke')}.json"

    continual_summary = {
        "num_chunks": K,
        "replay_capacity": replay_capacity,
        "accuracy_matrix": R.tolist(),
        "bwt": round(bwt, 4),
        "bwt_pct": round(bwt * 100, 2),
        "average_forgetting": round(avg_forgetting, 4),
        "average_forgetting_pct": round(avg_forgetting * 100, 2),
        "forgetting_per_chunk": forgetting_per_chunk,
        "wall_clock_seconds": round(timer.elapsed, 2),
    }

    with matrix_file.open("w", encoding="utf-8") as f:
        json.dump(continual_summary, f, indent=2)

    logger.info("Saved Continual Matrix to %s", matrix_file)
    logger.info("Backward Transfer: %.2f%% | Average Forgetting: %.2f%%", bwt * 100, avg_forgetting * 100)
    return continual_summary


def main() -> None:
    """CLI runner for continual learning benchmark."""
    parser = argparse.ArgumentParser(description="LiveVQA Continual Learning Benchmark")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--num-chunks", type=int, default=3, help="Number of chunks K")
    parser.add_argument("--replay-size", type=int, default=0, help="Replay buffer capacity")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per stage")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    run_continual_learning(
        args.config,
        num_chunks=args.num_chunks,
        replay_capacity=args.replay_size,
        epochs_per_stage=args.epochs,
    )


if __name__ == "__main__":
    main()
