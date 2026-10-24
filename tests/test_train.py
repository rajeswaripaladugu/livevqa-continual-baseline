"""Unit tests for training utilities, label masking, and step counting."""

import math
import torch
from PIL import Image

from src.models import SimpleProcessor, TinyVLM, configure_peft_adapters
from src.train import build_masked_batch, compute_steps


def test_scheduler_step_calculation() -> None:
    """Verify that compute_steps correctly uses math.ceil for update steps."""
    # 17 samples, batch_size=4 -> 5 batches. grad_accum=2 -> ceil(5/2) = 3 update steps/epoch.
    # 2 epochs -> 6 total steps.
    steps_per_epoch, total_steps = compute_steps(
        dataset_len=17,
        batch_size=4,
        grad_accum_steps=2,
        epochs=2,
    )
    assert steps_per_epoch == 3
    assert total_steps == 6

    # Edge case: dataset_len = 0
    assert compute_steps(0, 4, 2, 1) == (0, 0)


def test_label_masking_unmasks_only_answer() -> None:
    """Verify that prompt tokens are masked with -100 and only answer tokens have valid labels."""
    processor = SimpleProcessor()

    batch_items = [
        {
            "id": "1",
            "image": Image.new("RGB", (128, 128)),
            "prompt": "Question: What is 2+2? Options: A. 4 B. 5 Answer:",
            "target": " A",
            "full_text": "Question: What is 2+2? Options: A. 4 B. 5 Answer: A",
            "answer": "A",
        }
    ]

    batch = build_masked_batch(batch_items, processor, max_length=128)

    input_ids = batch["input_ids"]
    labels = batch["labels"]

    assert labels.shape == input_ids.shape

    # Determine prompt token count
    prompt_tokens = processor(text=batch_items[0]["prompt"])["input_ids"][0]
    prompt_len = len(prompt_tokens)

    # All tokens up to prompt_len must be masked with -100
    for idx in range(prompt_len):
        assert labels[0, idx].item() == -100, f"Token at index {idx} in prompt was not masked!"

    # Target token(s) after prompt_len must NOT be -100
    unmasked_indices = (labels[0] != -100).nonzero(as_tuple=True)[0]
    assert len(unmasked_indices) > 0, "No target tokens were unmasked!"
    assert unmasked_indices[0].item() >= prompt_len


def test_frozen_vision_encoder() -> None:
    """Verify that vision tower parameters are strictly frozen when PEFT is configured."""
    model = TinyVLM()
    adapted_model = configure_peft_adapters(
        model=model,
        target_modules=["q_proj", "v_proj"],
        lora_r=4,
        lora_alpha=8,
        exclude_modules_pattern=".*visual.*",
    )

    # Check that all parameters under visual have requires_grad == False
    visual_params = [p for name, p in adapted_model.named_parameters() if "visual" in name]
    assert len(visual_params) > 0
    for p in visual_params:
        assert not p.requires_grad, "Vision parameter was not frozen!"
