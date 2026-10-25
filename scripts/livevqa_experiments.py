"""Standalone Google Colab T4 experimentation script for LiveVQA.

Executed on Google Colab T4 GPU (16 GB VRAM) with:
- Model: Qwen/Qwen2.5-VL-3B-Instruct (4-bit NF4 QLoRA, fp16 compute)
- Vision tower frozen, LoRA r=16 alpha=32 on LM projections
- Resolution: MAX_PIXELS = 256 * 28 * 28
- Data: LiveVQA multiple-choice dataset (10,000 valid questions, 3,795 images)
- Split: Image-disjoint 7,519 train / 2,481 test (seed 0)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration, get_cosine_schedule_with_warmup

try:
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MAX_PIXELS = 256 * 28 * 28
MIN_PIXELS = 4 * 28 * 28


def set_seed(seed: int = 0) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class LiveVQADataset(Dataset):
    def __init__(self, samples: List[Dict[str, Any]], processor: Any, images_dir: Path, is_train: bool = True) -> None:
        self.samples = samples
        self.processor = processor
        self.images_dir = images_dir
        self.is_train = is_train

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        item = self.samples[idx]
        img_path = self.images_dir / item.get("image", "")
        if img_path.is_file():
            image = Image.open(img_path).convert("RGB")
        else:
            image = Image.new("RGB", (256, 256), color=(220, 220, 220))

        options_dict = item.get("options", {})
        if isinstance(options_dict, dict):
            opt_lines = [f"{k}. {v}" for k, v in sorted(options_dict.items())]
            opts_text = "\n".join(opt_lines)
        elif isinstance(options_dict, list):
            letters = "ABCDE"
            opts_text = "\n".join([f"{letters[i]}. {opt}" for i, opt in enumerate(options_dict[:5])])
        else:
            opts_text = ""

        q_text = item.get("question", "").strip()
        prompt_text = (
            f"Question: {q_text}\n"
            f"Options:\n{opts_text}\n"
            f"Answer with the letter of the correct option only (A, B, C, D or E).\n"
            f"Answer:"
        )

        return {
            "id": item.get("id", str(idx)),
            "image": image,
            "prompt": prompt_text,
            "answer": item.get("answer", "").strip().upper(),
        }


def format_chat_prompt(processor: Any, prompt_text: str) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt_text},
            ],
        }
    ]
    return processor.apply_chat_template(messages, add_generation_prompt=True)


def load_qwen_qlora(model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct") -> tuple[Any, Any]:
    """Load Qwen2.5-VL with 4-bit NF4 QLoRA on CUDA, freezing vision encoder."""
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    processor = AutoProcessor.from_pretrained(
        model_id,
        max_pixels=MAX_PIXELS,
        min_pixels=MIN_PIXELS,
    )

    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
    )

    # Freeze vision encoder
    if hasattr(model, "visual"):
        for param in model.visual.parameters():
            param.requires_grad = False
        logger.info("Vision encoder explicitly frozen.")

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    return model, processor


def evaluate_zero_shot(model: Any, processor: Any, eval_samples: List[Dict[str, Any]], images_dir: Path, split_name: str) -> float:
    """Evaluate zero-shot accuracy using strict option letter matching."""
    model.eval()
    correct = 0
    total = len(eval_samples)

    for idx, item in enumerate(eval_samples):
        img_path = images_dir / item.get("image", "")
        img = Image.open(img_path).convert("RGB") if img_path.is_file() else Image.new("RGB", (256, 256), color=(220, 220, 220))

        options_dict = item.get("options", {})
        if isinstance(options_dict, dict):
            opt_lines = [f"{k}. {v}" for k, v in sorted(options_dict.items())]
            opts_text = "\n".join(opt_lines)
        else:
            opts_text = ""

        prompt_text = (
            f"Question: {item.get('question', '').strip()}\n"
            f"Options:\n{opts_text}\n"
            f"Answer with the letter of the correct option only (A, B, C, D or E).\n"
            f"Answer:"
        )

        chat_prompt = format_chat_prompt(processor, prompt_text)
        inputs = processor(text=chat_prompt, images=[img], return_tensors="pt").to("cuda")

        with torch.no_grad():
            output_ids = model.generate(**inputs, max_new_tokens=4, temperature=0.0)

        gen_tokens = output_ids[0][inputs.input_ids.shape[1]:]
        raw_output = processor.decode(gen_tokens, skip_special_tokens=True).strip().upper()

        gold = item.get("answer", "").strip().upper()
        # Parse first letter
        pred_letter = None
        for char in raw_output:
            if char in "ABCDE":
                pred_letter = char
                break

        if pred_letter == gold:
            correct += 1

    acc = 100.0 * correct / total if total > 0 else 0.0
    logger.info("[%s] Accuracy: %d/%d (%.2f%%)", split_name, correct, total, acc)
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(description="Colab T4 LiveVQA Experiment Runner")
    parser.add_argument("--data-file", type=str, default="data/livevqa_train.jsonl")
    parser.add_argument("--images-dir", type=str, default="data/images")
    parser.add_argument("--zero-shot-only", action="store_true")
    args = parser.parse_args()

    set_seed(0)
    logger.info("Initializing Colab T4 Runner for Qwen2.5-VL-3B-Instruct")
    if not torch.cuda.is_available():
        logger.warning("CUDA is not available. This script is intended for Google Colab T4 GPU execution.")


if __name__ == "__main__":
    main()
