"""Stage-2: which agents inside the selected blocks actually take part.

The mechanics repeat Stage-1 — score, softmax, entropy, cover — but the
candidate set is restricted to roles belonging to the blocks Stage-1 kept.

That restriction is the point of splitting the stages. Scoring every role in
the pool at once puts semantically unrelated roles in the same competition, and
the resulting entropy then measures domain ambiguity rather than role ambiguity,
which inflates the agent count for queries that need only a couple of
specialists. Conditioning on the blocks makes role-level uncertainty mean what
its name says.
"""

from __future__ import annotations

import logging
from typing import List, Sequence

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
from s2rmerge.router.stage1 import StageDecision

logger = logging.getLogger(__name__)


class RoleRouter:
    """Stage-2: pick the participating agents within the chosen blocks."""

    def __init__(self, config: RouterConfig, role_prototypes: Prototypes):
        self.config = config
        self.prototypes = role_prototypes
        self.settings = config.stage2

    def candidates(self, blocks: Sequence[str]) -> List[str]:
        """Roles of the selected blocks, de-duplicated, in block order."""
        candidates: List[str] = []
        for block_id in blocks:
            for role in self.config.roles_of(block_id):
                if role in self.prototypes and role not in candidates:
                    candidates.append(role)
                elif role not in self.prototypes:
                    logger.warning("no prototype for role %r; skipping", role)
        return candidates

    def route(self, vector: np.ndarray, blocks: Sequence[str]) -> StageDecision:
        candidates = self.candidates(blocks)

        if not candidates:
            raise ValueError(f"blocks {list(blocks)} contain no routable roles")

        if len(candidates) == 1:
            only = candidates[0]
            return StageDecision(
                selected=[only],
                probabilities={only: 1.0},
                uncertainty=0.0,
                threshold=self.settings.coverage.rho_min,
                coverage=1.0,
            )

        scores = np.array(
            [cosine_similarity(vector, self.prototypes[role]) for role in candidates]
        )
        distribution = softmax(scores, self.settings.temperature)
        uncertainty = normalized_entropy(distribution, self.settings.epsilon)
        threshold = coverage_threshold(
            uncertainty,
            self.settings.coverage.rho_min,
            self.settings.coverage.rho_max,
            self.settings.coverage.tau,
            self.settings.coverage.kappa,
        )
        selected, probabilities, covered = select_by_coverage(
            distribution, candidates, threshold
        )

        return StageDecision(
            selected=selected,
            probabilities=dict(zip(selected, probabilities)),
            uncertainty=uncertainty,
            threshold=threshold,
            coverage=covered,
        )
