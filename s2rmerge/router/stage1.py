r"""Stage-1: how wide a slice of the agent pool the query needs.

Each block is a high-level domain with a prototype embedding. A block scores as

.. math:: s_B = \cos(v, p_B) + \lambda\,\pi(h, B)

where :math:`\pi(h, B)` is a fixed prior keyed on the domain hint Stage-0 read
out of the query. The embedding term carries the continuous similarity; the
prior breaks ties the embedding alone cannot resolve.

The scores become a distribution, its normalised entropy is the routing
uncertainty, and that uncertainty sets how much probability mass the selection
has to cover. Confident queries end up with one block, ambiguous ones with
several.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np

from s2rmerge.router.config import RouterConfig
from s2rmerge.router.functional import (
    cosine_similarity,
    coverage_threshold,
    normalized_entropy,
    select_by_coverage,
    softmax,
)
from s2rmerge.router.prototypes import Prototypes


@dataclass
class StageDecision:
    """What one routing stage chose, and the numbers behind it."""

    selected: List[str]
    probabilities: Dict[str, float]
    uncertainty: float
    threshold: float
    coverage: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "selected": list(self.selected),
            "probabilities": {k: float(v) for k, v in self.probabilities.items()},
            "uncertainty": self.uncertainty,
            "threshold": self.threshold,
            "coverage": self.coverage,
        }


class BlockRouter:
    """Stage-1: pick the domain blocks a query falls in."""

    def __init__(self, config: RouterConfig, block_prototypes: Prototypes):
        self.config = config
        self.prototypes = block_prototypes
        self.block_ids = list(block_prototypes)
        self.settings = config.stage1

    def block_prior(self, domain_hint: str) -> Dict[str, float]:
        """Prior boost per block for a domain hint, zero where none is defined."""
        priors = {block_id: 0.0 for block_id in self.block_ids}
        if not domain_hint or self.config.lambda_prior == 0:
            return priors

        mapping = self.config.domain_priors.get(domain_hint.lower().strip())
        if not mapping:
            return priors

        for block_id, boost in mapping.items():
            if block_id in priors:
                priors[block_id] = float(boost)
        return priors

    def score(self, vector: np.ndarray, domain_hint: str = "") -> Dict[str, float]:
        priors = self.block_prior(domain_hint)
        return {
            block_id: cosine_similarity(vector, self.prototypes[block_id])
            + self.config.lambda_prior * priors[block_id]
            for block_id in self.block_ids
        }

    def route(self, vector: np.ndarray, domain_hint: str = "") -> StageDecision:
        scores = self.score(vector, domain_hint)
        distribution = softmax(
            np.array([scores[block_id] for block_id in self.block_ids]),
            self.settings.temperature,
        )
        uncertainty = normalized_entropy(distribution, self.settings.epsilon)
        threshold = coverage_threshold(
            uncertainty,
            self.settings.coverage.rho_min,
            self.settings.coverage.rho_max,
            self.settings.coverage.tau,
            self.settings.coverage.kappa,
        )
        selected, probabilities, covered = select_by_coverage(
            distribution, self.block_ids, threshold
        )

        return StageDecision(
            selected=selected,
            probabilities=dict(zip(selected, probabilities)),
            uncertainty=uncertainty,
            threshold=threshold,
            coverage=covered,
        )
