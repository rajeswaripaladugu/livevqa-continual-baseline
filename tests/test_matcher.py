"""Unit tests for the strict multiple-choice letter answer matcher and Wilson CI."""

import pytest
from src.eval import compute_wilson_confidence_interval, extract_answer_letter


@pytest.mark.parametrize(
    "raw_text, expected_letter",
    [
        ("C", "C"),
        ("(c)", "C"),
        ("C.", "C"),
        ("(C)", "C"),
        ("Answer: C", "C"),
        ("Answer: (C)", "C"),
        ("Answer is B", "B"),
        ("Option C", "C"),
        ("Choice: A", "A"),
        ("The correct option is D", "D"),
        ("The correct answer is (B)", "B"),
        ("A. Red circle", "A"),
        ("B) Blue square", "B"),
        ("  C  ", "C"),
        ("\nAnswer: D\n", "D"),
    ],
)
def test_matcher_accepted_cases(raw_text: str, expected_letter: str) -> None:
    """Verify that valid letter variants and answer formats are correctly parsed."""
    parsed = extract_answer_letter(raw_text)
    assert parsed == expected_letter, f"Failed for input: {raw_text!r}, got: {parsed!r}"


@pytest.mark.parametrize(
    "raw_text",
    [
        "",
        "   ",
        "\n\t",
        "I am an AI assistant and I cannot see images clearly.",
        "This is an interesting visual query about modern events.",
        "Either A or B might be correct depending on context.",
        "Could be Option C, but also Option D.",
        "None of the above options look right.",
        "The image contains a cat, a dog, and a tree.",
        "XYZ",
        "E",
        "123",
    ],
)
def test_matcher_rejected_cases(raw_text: str) -> None:
    """Verify that verbose, ambiguous, or empty responses are strictly rejected."""
    parsed = extract_answer_letter(raw_text)
    assert parsed is None, f"Expected rejection for {raw_text!r}, but extracted {parsed!r}"


def test_wilson_confidence_interval() -> None:
    """Verify Wilson 95% confidence interval computation across boundary cases."""
    # Zero trials
    assert compute_wilson_confidence_interval(0, 0) == (0.0, 0.0)

    # 50% on 100 trials: center should be 0.50, bounds within [0.4, 0.6]
    low, high = compute_wilson_confidence_interval(50, 100)
    assert 0.0 <= low < 0.5 < high <= 1.0
    assert abs((low + high) / 2 - 0.5) < 0.05

    # 100% on 10 trials: lower bound should be > 0.65, upper bound clamped to 1.0
    low_100, high_100 = compute_wilson_confidence_interval(10, 10)
    assert low_100 > 0.65
    assert high_100 == 1.0

    # 0% on 10 trials: lower bound 0.0, upper bound < 0.35
    low_0, high_0 = compute_wilson_confidence_interval(0, 10)
    assert low_0 == 0.0
    assert high_0 < 0.35
