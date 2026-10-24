"""Training pipeline with label masking, gradient accumulation, cosine warmup, and checkpointing.

Ensures only answer tokens contribute to loss, fp16 GradScaler is used when on CUDA,
gradient accumulation steps are counted with math.ceil, and runs are fully seeded.
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

from src.data import VQASample, format_mcqa_prompt, load_dataset, split_by_image
from src.env import Timer, build_run_metadata, compute_config_hash
from src.models import load_model_and_processor

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set seeds for reproducibility across random, numpy, and torch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class VQADataset(Dataset):
    """PyTorch Dataset for visual question answering with prompt/label masking."""

    def __init__(
        self,
        samples: List[VQASample],
        processor: Any,
        max_length: int = 512,
        base_dir: Optional[Path] = None,
    ) -> None:
        self.samples = samples
        self.processor = processor
        self.max_length = max_length
        self.base_dir = base_dir or Path(".")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        sample = self.samples[idx]

        # Resolve image
        img_path = Path(sample.image)
        if not img_path.is_file():
            img_path = self.base_dir / sample.image
        if not img_path.is_file():
            img_path = self.base_dir / "data" / sample.image

        if img_path.is_file():
            image = Image.open(img_path).convert("RGB")
        else:
            # Fallback for synthetic/missing image during tests
            image = Image.new("RGB", (128, 128), color=(200, 200, 200))

        prompt = format_mcqa_prompt(sample.question, options=sample.options)
        target = f" {sample.answer}"
        full_text = f"{prompt}{target}"

        return {
            "id": sample.id,
            "image": image,
            "prompt": prompt,
            "target": target,
            "full_text": full_text,
            "answer": sample.answer,
        }


def build_masked_batch(
    batch_items: List[Dict[str, Any]],
    processor: Any,
    max_length: int = 512,
) -> Dict[str, torch.Tensor]:
    """Tokenize and mask labels so that ONLY answer tokens contribute to loss.

    Prompt tokens are assigned -100 so CrossEntropy ignores them.

    Args:
        batch_items: List of dictionary items from VQADataset.
        processor: Multimodal processor or tokenizer.
        max_length: Maximum sequence length.

    Returns:
        Batch dictionary with input_ids, attention_mask, pixel_values, and labels.
    """
    images = [item["image"] for item in batch_items]
    full_texts = [item["full_text"] for item in batch_items]
    prompts = [item["prompt"] for item in batch_items]

    # Process full inputs
    batch = processor(
        text=full_texts,
        images=images,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
    )

    # Process prompt alone to determine prompt token lengths
    prompt_batch = processor(
        text=prompts,
        images=None,
        padding=False,
        max_length=max_length,
        return_tensors="pt",
    )

    input_ids = batch["input_ids"]
    labels = input_ids.clone()

    # Mask padding and prompt tokens
    pad_token_id = getattr(processor, "pad_token_id", 0)

    for i in range(len(batch_items)):
        prompt_len = len(prompt_batch["input_ids"][i])
        # Mask everything up to prompt_len with -100
        labels[i, :prompt_len] = -100
        # Also mask pad tokens
        if pad_token_id is not None:
            labels[i, input_ids[i] == pad_token_id] = -100

    batch["labels"] = labels
    return batch


def compute_steps(
    dataset_len: int,
    batch_size: int,
    grad_accum_steps: int,
    epochs: int,
) -> Tuple[int, int]:
    """Calculate the exact number of optimization update steps using ceil.

    Args:
        dataset_len: Number of samples in training dataset.
        batch_size: Per-device batch size.
        grad_accum_steps: Gradient accumulation steps.
        epochs: Number of training epochs.

    Returns:
        Tuple of (steps_per_epoch, total_steps).
    """
    num_batches = math.ceil(dataset_len / batch_size) if dataset_len > 0 else 0
    steps_per_epoch = math.ceil(num_batches / grad_accum_steps) if num_batches > 0 else 0
    total_steps = steps_per_epoch * epochs
    return steps_per_epoch, total_steps


def train(
    config_path: str | Path,
    override_args: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute training pipeline according to configuration.

    Args:
        config_path: Path to YAML configuration.
        override_args: Optional dictionary to override config settings.

    Returns:
        Dictionary of training metadata and final loss.
    """
    cfg_file = Path(config_path)
    with cfg_file.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if override_args:
        for k, v in override_args.items():
            if k in config:
                if isinstance(v, dict) and isinstance(config[k], dict):
                    config[k].update(v)
                else:
                    config[k] = v
            else:
                config[k] = v

    train_cfg = config.get("training", {})
    seed = train_cfg.get("seed", 42)
    set_seed(seed)

    # Device selection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using compute device: %s", device)

    # Load Model and Processor
    model, processor = load_model_and_processor(config)
    model.to(device)

    # Load Data
    data_cfg = config.get("data", {})
    train_path = data_cfg.get("train_path", "data/synthetic_smoke.jsonl")
    base_dir = cfg_file.parent.parent

    # Auto-generate synthetic data if missing and synthetic is expected
    if not Path(train_path).is_file() and not (base_dir / train_path).is_file():
        if config.get("is_synthetic", False) or "smoke" in config.get("experiment_name", "").lower():
            from src.data import generate_synthetic_dataset
            logger.info("Generating synthetic smoke dataset at %s", train_path)
            generate_synthetic_dataset(base_dir / train_path, num_samples=train_cfg.get("max_samples", 16), seed=seed)

    resolved_path = train_path if Path(train_path).is_file() else (base_dir / train_path)
    all_samples = load_dataset(resolved_path, base_dir=base_dir)

    # Split by image if required
    if data_cfg.get("split_by_image", True):
        train_samples, eval_samples = split_by_image(
            all_samples,
            train_ratio=data_cfg.get("train_split_ratio", 0.8),
            seed=seed,
        )
    else:
        train_samples = all_samples

    max_samples = train_cfg.get("max_samples")
    if max_samples and max_samples < len(train_samples):
        train_samples = train_samples[:max_samples]

    dataset = VQADataset(train_samples, processor, max_length=train_cfg.get("max_length", 256), base_dir=base_dir)
    batch_size = max(1, train_cfg.get("batch_size", 2))
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=lambda items: build_masked_batch(items, processor, max_length=train_cfg.get("max_length", 256)),
    )

    epochs = train_cfg.get("epochs", 1)
    grad_accum_steps = max(1, train_cfg.get("gradient_accumulation_steps", 1))
    steps_per_epoch, total_steps = compute_steps(len(dataset), batch_size, grad_accum_steps, epochs)

    logger.info(
        "Dataset size: %d | Batch size: %d | Grad Accum: %d | Steps/epoch: %d | Total steps: %d",
        len(dataset),
        batch_size,
        grad_accum_steps,
        steps_per_epoch,
        total_steps,
    )

    # Optimizer & Scheduler
    lr = float(train_cfg.get("learning_rate", 1e-4))
    weight_decay = float(train_cfg.get("weight_decay", 0.01))
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    if not trainable_params:
        trainable_params = list(model.parameters())

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=weight_decay)
    warmup_steps = int(total_steps * train_cfg.get("warmup_ratio", 0.1))
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=max(1, total_steps),
    )

    use_fp16 = train_cfg.get("fp16", False) and torch.cuda.is_available()
    scaler = torch.amp.GradScaler("cuda", enabled=use_fp16)

    # Output checkpoint directory
    exp_name = config.get("experiment_name", "run")
    ckpt_dir = base_dir / "checkpoints" / exp_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    losses: List[float] = []
    timer = Timer()

    model.train()
    with timer:
        optimizer.zero_grad()
        accum_loss = 0.0

        for epoch in range(1, epochs + 1):
            logger.info("--- Starting Epoch %d/%d ---", epoch, epochs)
            for batch_idx, batch in enumerate(dataloader):
                # Move tensors to device
                for k in ["input_ids", "attention_mask", "pixel_values", "labels"]:
                    if k in batch and batch[k] is not None:
                        batch[k] = batch[k].to(device)

                with torch.amp.autocast("cuda", enabled=use_fp16):
                    outputs = model(**batch)
                    loss = outputs.loss
                    if loss is not None:
                        # Scale loss for gradient accumulation
                        scaled_loss = loss / grad_accum_steps
                    else:
                        scaled_loss = torch.tensor(0.0, device=device, requires_grad=True)

                if use_fp16:
                    scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                loss_val = loss.item() if loss is not None else 0.0
                accum_loss += loss_val

                # Optimization step when accumulation boundary is met or at final batch
                if (batch_idx + 1) % grad_accum_steps == 0 or (batch_idx + 1) == len(dataloader):
                    if use_fp16:
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

                    window_avg_loss = accum_loss / min(grad_accum_steps, (batch_idx % grad_accum_steps) + 1)
                    losses.append(window_avg_loss)
                    logger.info("Epoch %d | Batch %d/%d | Window Loss: %.4f", epoch, batch_idx + 1, len(dataloader), window_avg_loss)
                    accum_loss = 0.0

            # Save epoch checkpoint
            ckpt_file = ckpt_dir / f"checkpoint_epoch_{epoch}.pt"
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": config,
            }, ckpt_file)
            logger.info("Saved epoch checkpoint: %s", ckpt_file)

    final_loss = losses[-1] if losses else 0.0
    config_hash = compute_config_hash(cfg_file)
    metadata = build_run_metadata(
        config_name=cfg_file.name,
        config_hash=config_hash,
        seed=seed,
        model_name=config.get("model", {}).get("name", "VLM"),
        num_train_samples=len(dataset),
        num_eval_samples=0,
        wall_clock_seconds=timer.elapsed,
        extra={
            "final_train_loss": round(final_loss, 4),
            "epochs": epochs,
            "total_steps": total_steps,
        },
    )

    logger.info("Training completed in %.2fs. Final Loss: %.4f", timer.elapsed, final_loss)
    return {
        "metadata": metadata,
        "checkpoint_dir": str(ckpt_dir),
        "final_loss": final_loss,
        "model": model,
        "processor": processor,
    }


def main() -> None:
    """CLI runner for training."""
    parser = argparse.ArgumentParser(description="LiveVQA Continual Baseline Training Runner")
    parser.add_argument("--config", type=str, required=True, help="Path to config YAML")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    train(args.config)


if __name__ == "__main__":
    main()
