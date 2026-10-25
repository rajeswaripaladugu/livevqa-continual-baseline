"""Unit tests for dataset loading, split-by-image, options formatting, and baselines."""

import pytest
from src.data import (
    VQASample,
    compute_answer_distribution,
    format_mcqa_prompt,
    format_prompt,
    generate_synthetic_dataset,
    load_dataset,
    split_by_image,
    validate_sample,
)


def test_options_formatting_dict() -> None:
    """Verify options formatting when options is a dictionary."""
    question = "Which planet is closest to the Sun?"
    options = {"A": "Venus", "B": "Mercury", "C": "Mars", "D": "Earth"}
    prompt = format_mcqa_prompt(question, options=options)

    assert "Question: Which planet is closest to the Sun?" in prompt
    assert "A. Venus" in prompt
    assert "B. Mercury" in prompt
    assert "C. Mars" in prompt
    assert "D. Earth" in prompt
    assert "Select the single best option." in prompt


def test_options_formatting_list() -> None:
    """Verify options formatting when options is a list of strings."""
    question = "Identify the color of the apple."
    options = ["Red", "Blue", "Green", "Purple"]
    prompt = format_mcqa_prompt(question, options=options)

    assert "A. Red" in prompt
    assert "B. Blue" in prompt
    assert "C. Green" in prompt
    assert "D. Purple" in prompt


def test_options_formatting_none_or_in_text() -> None:
    """Verify options formatting when options are already inside the question string."""
    question = "What is the answer?\nA. First choice\nB. Second choice"
    prompt = format_mcqa_prompt(question, options=None)

    assert "Question: What is the answer?\nA. First choice\nB. Second choice" in prompt
    assert "Select the single best option." in prompt


def test_split_by_image_has_zero_overlap() -> None:
    """Ensure that questions referencing the same image NEVER cross the train/eval split."""
    samples = [
        VQASample(id="1", image="img_A.jpg", question="Q1", answer="A"),
        VQASample(id="2", image="img_A.jpg", question="Q2", answer="B"),
        VQASample(id="3", image="img_B.jpg", question="Q3", answer="C"),
        VQASample(id="4", image="img_B.jpg", question="Q4", answer="D"),
        VQASample(id="5", image="img_C.jpg", question="Q5", answer="A"),
        VQASample(id="6", image="img_D.jpg", question="Q6", answer="B"),
        VQASample(id="7", image="img_E.jpg", question="Q7", answer="C"),
        VQASample(id="8", image="img_F.jpg", question="Q8", answer="D"),
    ]

    train_split, eval_split = split_by_image(samples, train_ratio=0.75, seed=42)

    assert len(train_split) > 0
    assert len(eval_split) > 0
    assert len(train_split) + len(eval_split) == len(samples)

    train_images = {s.image for s in train_split}
    eval_images = {s.image for s in eval_split}

    overlap = train_images.intersection(eval_images)
    assert len(overlap) == 0, f"Detected image leakage in split-by-image: {overlap}"


def test_majority_baseline_computation() -> None:
    """Verify computation of label distribution and majority-class baseline."""
    # 5 samples: A (3), B (1), C (1) -> Majority is A with 3/5 = 60.0%
    samples = [
        VQASample(id="1", image="img1.png", question="Q1", answer="A"),
        VQASample(id="2", image="img2.png", question="Q2", answer="A"),
        VQASample(id="3", image="img3.png", question="Q3", answer="A"),
        VQASample(id="4", image="img4.png", question="Q4", answer="B"),
        VQASample(id="5", image="img5.png", question="Q5", answer="C"),
    ]

    dist = compute_answer_distribution(samples)
    assert dist["total"] == 5
    assert dist["counts"]["A"] == 3
    assert dist["majority_label"] == "A"
    assert dist["majority_baseline"] == 0.60


def test_schema_validation() -> None:
    """Verify validation of mandatory fields."""
    valid_dict = {
        "id": "item_01",
        "image": "img.png",
        "question": "What is this?",
        "answer": "B",
    }
    sample = validate_sample(valid_dict)
    assert sample.id == "item_01"
    assert sample.answer == "B"

    # Missing mandatory field 'question'
    with pytest.raises(ValueError, match="Missing mandatory field 'question'"):
        validate_sample({"id": "item_02", "image": "img.png", "answer": "B"})


def test_synthetic_data_generation(tmp_path) -> None:
    """Verify that synthetic generator creates correctly marked synthetic items."""
    out_file = tmp_path / "synthetic_smoke.jsonl"
    gen_path = generate_synthetic_dataset(out_file, num_samples=8, seed=42)

    assert gen_path.is_file()
    samples = load_dataset(gen_path)
    assert len(samples) == 8
    for s in samples:
        assert s.is_synthetic is True
        assert "synthetic" in s.id.lower()


def test_format_prompt_mcqa_qa_qar() -> None:
    """Verify format_prompt output across MCQA, QA, and QAR formats from the paper."""
    sample = VQASample(
        id="sample_01",
        image="test.png",
        question="What is the location shown?",
        answer="A",
        options={"A": "Paris", "B": "London", "C": "Tokyo", "D": "Rome", "E": "Berlin"},
        ground_truth_text="Paris",
        reasoning="The Eiffel Tower is located in Paris.",
        level=1,
        domain="news",
    )

    # 1. MCQA
    p_mcqa, t_mcqa = format_prompt(sample, format_type="MCQA")
    assert "A. Paris" in p_mcqa
    assert "E. Berlin" in p_mcqa
    assert t_mcqa == " A"

    # 2. QA
    p_qa, t_qa = format_prompt(sample, format_type="QA")
    assert "Question: What is the location shown?" in p_qa
    assert "Answer:" in p_qa
    assert "A. Paris" not in p_qa
    assert t_qa == " Paris"

    # 3. QAR
    p_qar, t_qar = format_prompt(sample, format_type="QAR")
    assert "Question: What is the location shown?" in p_qar
    assert "Paris" in t_qar
    assert "Reasoning: The Eiffel Tower is located in Paris." in t_qar


def test_livevqa_5_choice_validation() -> None:
    """Verify that 5-choice options (A-E) and LiveVQA fields are parsed correctly."""
    sample_dict = {
        "id": "livevqa_test_01",
        "image": "livevqa_images/test.png",
        "question": "Which facility is shown?",
        "options": {"A": "Opt A", "B": "Opt B", "C": "Opt C", "D": "Opt D", "E": "Opt E"},
        "answer": "E",
        "ground_truth_text": "Opt E",
        "ground_truth_list": ["Opt E", "Option E"],
        "reasoning": "Reasoning explanation text.",
        "level": 2,
        "domain": "news",
    }
    sample = validate_sample(sample_dict)
    assert sample.answer == "E"
    assert len(sample.options) == 5
    assert sample.level == 2
    assert sample.domain == "news"
    assert sample.ground_truth_text == "Opt E"

