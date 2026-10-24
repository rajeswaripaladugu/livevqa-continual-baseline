"""Data processing, schema validation, split-by-image, and synthetic smoke data generation.

Implements strict validation of visual QA samples, options formatting for dict/list/in-text,
image-disjoint data splitting, and answer distribution / majority-class baseline calculation.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from PIL import Image, ImageDraw

logger = logging.getLogger(__name__)


@dataclass
class VQASample:
    """Standardized multimodal question-answering sample."""

    id: str
    image: str
    question: str
    answer: str
    options: Optional[Dict[str, str] | List[str]] = None
    date: Optional[str] = None
    is_synthetic: bool = False
    metadata: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert sample to dictionary."""
        return asdict(self)


def format_mcqa_prompt(
    question: str,
    options: Optional[Dict[str, str] | List[str]] = None,
    options_key: str = "options",
) -> str:
    """Format question and multiple-choice options into an explicit prompt.

    Handles three representations:
    1. Dictionary: {'A': 'text', 'B': 'text'}
    2. List: ['text1', 'text2', 'text3', 'text4']
    3. Already formatted inside the question string.

    Args:
        question: Raw question text.
        options: Optional dictionary or list of choice texts.
        options_key: Key name where options were stored.

    Returns:
        Structured prompt with question, lettered options, and answer cue.
    """
    clean_q = question.strip()
    formatted_options = ""

    # Check if options are provided externally
    if isinstance(options, dict) and options:
        opt_lines = [f"{k}. {v}" for k, v in sorted(options.items())]
        formatted_options = "\n".join(opt_lines)
    elif isinstance(options, (list, tuple)) and options:
        letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        opt_lines = [f"{letters[i]}. {opt}" for i, opt in enumerate(options)]
        formatted_options = "\n".join(opt_lines)
    else:
        # Options might already be embedded in the question
        pass

    if formatted_options:
        prompt = (
            f"Question: {clean_q}\n"
            f"Options:\n{formatted_options}\n"
            f"Select the single best option. Output only the capital letter (A, B, C, or D).\n"
            f"Answer:"
        )
    else:
        prompt = (
            f"Question: {clean_q}\n"
            f"Select the single best option. Output only the capital letter (A, B, C, or D).\n"
            f"Answer:"
        )

    return prompt


def validate_sample(sample_dict: Dict[str, Any], base_dir: Optional[Path] = None) -> VQASample:
    """Validate sample dictionary schema and return typed VQASample.

    Args:
        sample_dict: Input row dictionary.
        base_dir: Optional root directory for resolving image paths.

    Raises:
        ValueError: If mandatory fields are missing or invalid.
    """
    if "id" not in sample_dict:
        raise ValueError("Missing mandatory field 'id' in sample")
    if "image" not in sample_dict:
        raise ValueError(f"Missing mandatory field 'image' in sample {sample_dict.get('id')}")
    if "question" not in sample_dict:
        raise ValueError(f"Missing mandatory field 'question' in sample {sample_dict.get('id')}")
    if "answer" not in sample_dict:
        raise ValueError(f"Missing mandatory field 'answer' in sample {sample_dict.get('id')}")

    sample_id = str(sample_dict["id"])
    image_ref = str(sample_dict["image"])
    question = str(sample_dict["question"]).strip()
    raw_answer = str(sample_dict["answer"]).strip().upper()

    # Extract uppercase letter if formatted e.g. "(C)" or "C."
    clean_answer = raw_answer.strip("()[]{} .")
    if not clean_answer:
        clean_answer = raw_answer

    options = sample_dict.get("options")
    date_val = sample_dict.get("date") or sample_dict.get("timestamp")
    is_synth = bool(sample_dict.get("is_synthetic", False) or "synthetic" in image_ref.lower())

    return VQASample(
        id=sample_id,
        image=image_ref,
        question=question,
        answer=clean_answer,
        options=options,
        date=str(date_val) if date_val else None,
        is_synthetic=is_synth,
        metadata=sample_dict.get("metadata"),
    )


def load_dataset(file_path: str | Path, base_dir: Optional[Path] = None) -> List[VQASample]:
    """Load JSON or JSONL file into a list of validated VQASamples.

    Args:
        file_path: Path to dataset file.
        base_dir: Root path for resolving relative image paths.

    Returns:
        List of validated VQASample objects.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    samples: List[VQASample] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    samples.append(validate_sample(data, base_dir=base_dir))
                except Exception as e:
                    logger.warning("Error parsing line %d in %s: %s", line_idx, path, e)
    else:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                for item in data:
                    samples.append(validate_sample(item, base_dir=base_dir))
            elif isinstance(data, dict):
                # Handle dictionary of samples
                for item in data.values():
                    if isinstance(item, dict):
                        samples.append(validate_sample(item, base_dir=base_dir))

    logger.info("Loaded %d validated samples from %s", len(samples), path)
    return samples


def split_by_image(
    samples: List[VQASample],
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Tuple[List[VQASample], List[VQASample]]:
    """Split samples into train and evaluation sets disjoint by image.

    Guarantees that questions regarding the same image never cross train/eval splits.

    Args:
        samples: List of VQASample items.
        train_ratio: Fraction of images allocated to training (default: 0.8).
        seed: Random seed for deterministic shuffling.

    Returns:
        Tuple of (train_samples, eval_samples).
    """
    if not samples:
        return [], []

    # Map image paths to their associated samples
    image_to_samples: Dict[str, List[VQASample]] = {}
    for s in samples:
        image_to_samples.setdefault(s.image, []).append(s)

    unique_images = sorted(image_to_samples.keys())
    rng = random.Random(seed)
    rng.shuffle(unique_images)

    split_idx = int(round(len(unique_images) * train_ratio))
    # Ensure at least 1 image in each split if there are at least 2 images
    if 0 < split_idx < len(unique_images):
        pass
    elif len(unique_images) > 1:
        split_idx = max(1, min(split_idx, len(unique_images) - 1))

    train_images: Set[str] = set(unique_images[:split_idx])
    eval_images: Set[str] = set(unique_images[split_idx:])

    train_samples: List[VQASample] = []
    eval_samples: List[VQASample] = []

    for img, s_list in image_to_samples.items():
        if img in train_images:
            train_samples.extend(s_list)
        else:
            eval_samples.extend(s_list)

    # Verification of zero image leakage
    train_img_set = {s.image for s in train_samples}
    eval_img_set = {s.image for s in eval_samples}
    leakage = train_img_set.intersection(eval_img_set)
    if leakage:
        raise RuntimeError(f"CRITICAL: Image leakage detected between splits! Leaked images: {leakage}")

    logger.info(
        "Split-by-image complete: %d train samples (%d images), %d eval samples (%d images). Zero overlap.",
        len(train_samples),
        len(train_images),
        len(eval_samples),
        len(eval_images),
    )
    return train_samples, eval_samples


def compute_answer_distribution(samples: List[VQASample]) -> Dict[str, Any]:
    """Compute label distribution and majority-class baseline.

    Args:
        samples: List of VQASample items.

    Returns:
        Dictionary with class counts, proportions, majority letter, and majority baseline accuracy.
    """
    if not samples:
        return {
            "total": 0,
            "counts": {},
            "proportions": {},
            "majority_label": None,
            "majority_baseline": 0.0,
        }

    counts: Dict[str, int] = {}
    for s in samples:
        letter = s.answer.upper()
        counts[letter] = counts.get(letter, 0) + 1

    total = len(samples)
    proportions = {k: round(v / total, 4) for k, v in counts.items()}
    majority_label = max(counts.keys(), key=lambda k: counts[k])
    majority_baseline = counts[majority_label] / total

    return {
        "total": total,
        "counts": counts,
        "proportions": proportions,
        "majority_label": majority_label,
        "majority_baseline": round(majority_baseline, 4),
    }


def generate_synthetic_dataset(
    output_path: str | Path,
    image_dir: Optional[str | Path] = None,
    num_samples: int = 16,
    seed: int = 42,
) -> Path:
    """Generate a synthetic visual question answering dataset for pipeline smoke testing.

    Creates rendered geometric shapes (circles, rectangles, triangles) with known colors,
    multiple-choice questions, and saves to JSONL with 'synthetic' markings.

    Args:
        output_path: Filepath for the generated JSONL.
        image_dir: Directory to save rendered synthetic PNGs.
        num_samples: Total number of samples to generate (default: 16).
        seed: Random seed for reproducibility.

    Returns:
        Path to generated JSONL dataset.
    """
    out_file = Path(output_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)

    if image_dir is None:
        img_dir = out_file.parent / "synthetic_images"
    else:
        img_dir = Path(image_dir)
    img_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    shapes = ["circle", "square", "triangle"]
    colors = [
        ("red", (220, 50, 50)),
        ("blue", (50, 100, 220)),
        ("green", (50, 180, 50)),
        ("yellow", (230, 200, 30)),
    ]

    records: List[Dict[str, Any]] = []

    for i in range(num_samples):
        shape = shapes[i % len(shapes)]
        color_name, rgb = colors[(i // len(shapes)) % len(colors)]

        # Render 128x128 synthetic image
        img = Image.new("RGB", (128, 128), color=(240, 240, 240))
        draw = ImageDraw.Draw(img)

        if shape == "circle":
            draw.ellipse([24, 24, 104, 104], fill=rgb, outline=(0, 0, 0))
        elif shape == "square":
            draw.rectangle([24, 24, 104, 104], fill=rgb, outline=(0, 0, 0))
        elif shape == "triangle":
            draw.polygon([(64, 20), (20, 108), (108, 108)], fill=rgb, outline=(0, 0, 0))

        img_rel_path = f"synthetic_images/synth_{i:03d}_{shape}_{color_name}.png"
        img_abs_path = out_file.parent / img_rel_path
        img_abs_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(img_abs_path)

        # Question template: question about shape or color
        correct_letter = ["A", "B", "C", "D"][i % 4]
        other_letters = [ltr for ltr in ["A", "B", "C", "D"] if ltr != correct_letter]

        if i % 2 == 0:
            question = "What geometric shape is shown in the image?"
            correct_text = shape.capitalize()
            distractors = [s.capitalize() for s in shapes if s != shape] + ["Hexagon"]
            options_dict = {correct_letter: correct_text}
            for ltr, dist in zip(other_letters, distractors[:3]):
                options_dict[ltr] = dist
        else:
            question = "What primary color is the shape in the image?"
            correct_text = color_name.capitalize()
            distractors = [c[0].capitalize() for c in colors if c[0] != color_name]
            options_dict = {correct_letter: correct_text}
            for ltr, dist in zip(other_letters, distractors[:3]):
                options_dict[ltr] = dist

        # Add optional chronological date
        date_str = f"2024-{(i % 12) + 1:02d}-01"

        records.append({
            "id": f"synthetic_{i:04d}",
            "image": str(img_rel_path).replace("\\", "/"),
            "question": question,
            "options": options_dict,
            "answer": correct_letter,
            "date": date_str,
            "is_synthetic": True,
            "metadata": {
                "source": "synthetic_smoke_generator",
                "shape": shape,
                "color": color_name,
            },
        })

    with out_file.open("w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    logger.info("Generated %d synthetic smoke samples at %s", len(records), out_file)
    return out_file


def main() -> None:
    """CLI for dataset verification, statistics, and synthetic generation."""
    parser = argparse.ArgumentParser(description="LiveVQA Data Processor & Synthetic Generator")
    parser.add_argument("--generate-synthetic", action="store_true", help="Generate synthetic smoke dataset")
    parser.add_argument("--output", type=str, default="data/synthetic_smoke.jsonl", help="Output file path")
    parser.add_argument("--num-samples", type=int, default=16, help="Number of synthetic samples")
    parser.add_argument("--stats", type=str, default=None, help="Path to dataset to compute statistics for")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    if args.generate_synthetic:
        gen_path = generate_synthetic_dataset(args.output, num_samples=args.num_samples)
        samples = load_dataset(gen_path, base_dir=Path(args.output).parent)
        dist = compute_answer_distribution(samples)
        logger.info("Synthetic Dataset Created: %s", gen_path)
        logger.info("Answer Distribution: %s", dist["counts"])
        logger.info("Majority-Class Baseline: %.2f%% (Letter: %s)", dist["majority_baseline"] * 100, dist["majority_label"])

    if args.stats:
        samples = load_dataset(args.stats)
        dist = compute_answer_distribution(samples)
        logger.info("Dataset: %s | Total: %d", args.stats, dist["total"])
        logger.info("Counts: %s", dist["counts"])
        logger.info("Majority Baseline: %.2f%%", dist["majority_baseline"] * 100)


if __name__ == "__main__":
    main()
