"""Answer extraction copes with the shapes agents actually produce."""

import pytest

from s2rmerge.answers import extract_choice, extract_numeric, normalize_latex


@pytest.mark.parametrize(
    "response, expected",
    [
        ("Working... The answer is 140", "140"),
        ("so the answer is 72.", "72"),
        ("We get \\boxed{18} at the end.", "18"),
        ("No marker here, just 5 then 11", "11"),
    ],
)
def test_numeric_extraction(response, expected):
    assert extract_numeric(response) == expected


@pytest.mark.parametrize(
    "response, expected",
    [
        ("The answer is D", "D"),
        ("after eliminating B and C, the answer is A.", "A"),
        ("\\boxed{E}", "E"),
    ],
)
def test_choice_extraction(response, expected):
    assert extract_choice(response) == expected


def test_non_text_responses_are_not_guessed_at():
    # A failed agent call yields None; that must not become a plausible answer.
    assert extract_numeric(None) is None
    assert extract_choice(None) is None


def test_latex_normalisation_canonicalises_equivalents():
    assert normalize_latex("\\frac12") == "\\frac{1}{2}"
    assert normalize_latex("0.5") == "\\frac{1}{2}"
    assert normalize_latex("3/4") == "\\frac{3}{4}"
    assert normalize_latex("\\sqrt3") == "\\sqrt{3}"
    assert normalize_latex("50\\%") == "50"
