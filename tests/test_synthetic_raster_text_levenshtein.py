from __future__ import annotations

import random

import pytest

from customs_ai_agent.evaluation.synthetic_raster_text import _levenshtein


def _reference_levenshtein(reference: str, candidate: str) -> int:
    """Small, direct DP oracle kept independent from the production algorithm."""

    if len(reference) < len(candidate):
        reference, candidate = candidate, reference
    previous = list(range(len(candidate) + 1))
    for reference_index, reference_character in enumerate(reference, start=1):
        current = [reference_index]
        for candidate_index, candidate_character in enumerate(candidate, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[candidate_index] + 1,
                    previous[candidate_index - 1]
                    + (reference_character != candidate_character),
                )
            )
        previous = current
    return previous[-1]


@pytest.mark.parametrize(
    ("reference", "candidate"),
    [
        ("", ""),
        ("", "asymmetric"),
        ("asymmetric", ""),
        ("a", "a"),
        ("a", "b"),
        ("kitten", "sitting"),
        ("short", "a much longer string"),
        ("Caf\u00e9", "Cafe\u0301"),
        ("\ud55c\uae00 \u6587\u5b57 \U0001f642", "\ud55c\uae00\u6587\u5b57\U0001f643"),
        ("\U0001f469\u200d\U0001f4bb", "\U0001f469\U0001f4bb"),
        ("line one\nline two", "line one line two"),
    ],
)
def test_bit_parallel_levenshtein_matches_reference_for_edge_cases(
    reference: str, candidate: str
) -> None:
    assert _levenshtein(reference, candidate) == _reference_levenshtein(
        reference, candidate
    )


def test_bit_parallel_levenshtein_matches_deterministic_randomized_unicode() -> None:
    randomizer = random.Random(20260826)
    alphabet = tuple("abCZ09 _-/\n\u00e9\u0301\ud55c\u754c\U0001f642\U0001f469\u200d\U0001f4bb")

    for _ in range(500):
        reference = "".join(
            randomizer.choice(alphabet) for _ in range(randomizer.randrange(65))
        )
        candidate = "".join(
            randomizer.choice(alphabet) for _ in range(randomizer.randrange(65))
        )
        expected = _reference_levenshtein(reference, candidate)
        assert _levenshtein(reference, candidate) == expected
        assert _levenshtein(candidate, reference) == expected


def test_bit_parallel_levenshtein_accepts_exact_product_guard_boundary() -> None:
    reference = "a" * 4_000
    candidate = "a" * 2_500

    assert len(reference) * len(candidate) == 10_000_000
    assert _levenshtein(reference, candidate) == 1_500


def test_bit_parallel_levenshtein_preserves_input_guard_before_calculation() -> None:
    value = "a" * 3_163

    with pytest.raises(ValueError, match="^raster text CER input is too large$"):
        _levenshtein(value, value)
