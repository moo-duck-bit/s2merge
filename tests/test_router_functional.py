"""The routing rules behave the way the two stages rely on."""

import numpy as np
import pytest

from s2rmerge.router.functional import (
    coverage_threshold,
    normalized_entropy,
    select_by_coverage,
    softmax,
)


def test_entropy_is_zero_when_certain_and_one_when_uniform():
    assert normalized_entropy(np.array([1.0, 0.0, 0.0])) == pytest.approx(0.0, abs=1e-6)
    assert normalized_entropy(np.array([1 / 3, 1 / 3, 1 / 3])) == pytest.approx(1.0)


def test_entropy_is_comparable_across_candidate_counts():
    # Normalising by log(n) is what lets Stage-1 and Stage-2 share one
    # threshold curve despite scoring different numbers of candidates.
    assert normalized_entropy(np.full(4, 0.25)) == pytest.approx(
        normalized_entropy(np.full(16, 1 / 16))
    )


def test_lower_temperature_sharpens_the_distribution():
    scores = np.array([1.0, 0.5, 0.2])
    assert softmax(scores, 0.1).max() > softmax(scores, 1.0).max()


def test_coverage_grows_with_uncertainty():
    bounds = dict(rho_min=0.3, rho_max=0.8, tau=0.5, kappa=8.0)

    certain = coverage_threshold(0.0, **bounds)
    ambiguous = coverage_threshold(1.0, **bounds)

    assert 0.3 <= certain < ambiguous <= 0.8
    # tau centres the transition, so it sits halfway between the bounds.
    assert coverage_threshold(0.5, **bounds) == pytest.approx(0.55)


def test_selection_stops_once_the_mass_is_covered():
    probabilities = np.array([0.5, 0.3, 0.15, 0.05])
    ids = ["a", "b", "c", "d"]

    selected, _, covered = select_by_coverage(probabilities, ids, 0.75)

    assert selected == ["a", "b"]
    assert covered == pytest.approx(0.8)


def test_selection_size_follows_the_threshold_not_a_fixed_k():
    probabilities = np.array([0.5, 0.3, 0.15, 0.05])
    ids = ["a", "b", "c", "d"]

    narrow, _, _ = select_by_coverage(probabilities, ids, 0.4)
    wide, _, _ = select_by_coverage(probabilities, ids, 0.99)

    assert len(narrow) == 1
    assert len(wide) == 4


def test_selection_never_returns_empty():
    probabilities = np.array([0.6, 0.4])
    selected, _, _ = select_by_coverage(probabilities, ["a", "b"], 0.0)
    assert selected == ["a"]
