r"""The scoring rules the two routing stages share.

Both stages do the same four things at different granularities: score
candidates against the query representation, turn the scores into a
distribution, measure how uncertain that distribution is, and take as many
candidates as the uncertainty says are needed.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import numpy as np

DEFAULT_EPSILON = 1e-10


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    norms = np.linalg.norm(a) * np.linalg.norm(b)
    if norms == 0:
        return 0.0
    return float(np.dot(a, b) / norms)


def softmax(scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Temperature-scaled softmax; lower temperature gives a sharper choice."""
    scaled = scores / temperature
    scaled = scaled - np.max(scaled)
    exponentiated = np.exp(scaled)
    return exponentiated / np.sum(exponentiated)


def normalized_entropy(probabilities: np.ndarray, epsilon: float = DEFAULT_EPSILON) -> float:
    """Entropy divided by its maximum, so uncertainty lands in ``[0, 1]``.

    Dividing by ``log(n)`` is what makes the two stages comparable: block-level
    and role-level distributions have different numbers of candidates.
    """
    n = len(probabilities)
    if n <= 1:
        return 0.0

    clipped = np.clip(probabilities, float(epsilon), 1.0)
    entropy = -np.sum(clipped * np.log(clipped))
    return float(np.clip(entropy / np.log(n), 0.0, 1.0))


def sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


def coverage_threshold(
    uncertainty: float,
    rho_min: float,
    rho_max: float,
    tau: float,
    kappa: float,
) -> float:
    r"""How much probability mass this query's selection has to cover.

    .. math:: \rho(\tilde u) = \rho_{min} + (\rho_{max} - \rho_{min})\,
              \sigma(\kappa(\tilde u - \tau))

    A confident query stays near ``rho_min`` and recruits few candidates; an
    ambiguous one is pushed toward ``rho_max`` and recruits more. ``tau`` is the
    uncertainty at which the transition is centred and ``kappa`` how sharp it is.
    """
    return rho_min + (rho_max - rho_min) * sigmoid(kappa * (uncertainty - tau))


def select_by_coverage(
    probabilities: np.ndarray,
    ids: Sequence[str],
    threshold: float,
) -> Tuple[List[str], List[float], float]:
    """Take candidates in descending probability until ``threshold`` is covered.

    This is what replaces a fixed Top-K: the number of candidates is decided by
    the shape of the distribution rather than set in advance.
    """
    order = np.argsort(probabilities)[::-1]

    selected_ids: List[str] = []
    selected_probabilities: List[float] = []
    covered = 0.0

    for index in order:
        covered += float(probabilities[index])
        selected_ids.append(ids[index])
        selected_probabilities.append(float(probabilities[index]))
        if covered >= threshold:
            break

    return selected_ids, selected_probabilities, covered
