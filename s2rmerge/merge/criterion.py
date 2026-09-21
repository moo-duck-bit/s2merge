r"""Choosing which agent to merge, and which agent absorbs it.

The transfer efficiency index of node :math:`c` is

.. math::

    \Delta W(c) = \frac{\sum w_{in}}{\mathrm{Var}(w_{in}) + \epsilon}
                - \frac{\sum w_{out}}{\mathrm{Var}(w_{out}) + \epsilon}

where :math:`w_{in}` and :math:`w_{out}` are the learned weights of the edges
entering and leaving :math:`c`. Each term is total signal strength divided by
how erratic that signal is, so it reads as *stable* signal strength. A node
whose stable incoming signal outweighs its stable outgoing signal takes
information in without passing it on, which is the structural bottleneck the
merge step removes: the target is :math:`c^{*} = \arg\max_c \Delta W(c)`.

Exactly one node is selected per merge step, so the criterion needs no ratio or
threshold to tune.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np

EPSILON = 1e-8


@dataclass(frozen=True)
class MergeTarget:
    """The node to absorb and the score that selected it."""

    node_id: str
    delta_w: float
    scores: Dict[str, float]


@dataclass(frozen=True)
class MergePartner:
    """The node that absorbs the target, and their strongest shared weight."""

    node_id: str
    score: float


def _stable_strength(weights: np.ndarray) -> float:
    """Total weight divided by its variance: strength discounted by instability."""
    if weights.size == 0:
        return 0.0
    return float(weights.sum() / (weights.var() + EPSILON))


def transfer_efficiency(adjacency: np.ndarray, node_ids: Sequence[str]) -> Dict[str, float]:
    r"""Compute :math:`\Delta W(c)` for every node."""
    scores: Dict[str, float] = {}
    for index, node_id in enumerate(node_ids):
        incoming = np.delete(adjacency[:, index], index)
        outgoing = np.delete(adjacency[index, :], index)
        scores[node_id] = _stable_strength(incoming) - _stable_strength(outgoing)
    return scores


def select_merge_target(adjacency: np.ndarray, node_ids: Sequence[str]) -> MergeTarget:
    r"""The node with the largest :math:`\Delta W`, i.e. the worst bottleneck."""
    if len(node_ids) < 2:
        raise ValueError("a merge needs at least two nodes")

    scores = transfer_efficiency(adjacency, node_ids)
    target_id = max(scores, key=scores.__getitem__)
    return MergeTarget(node_id=target_id, delta_w=scores[target_id], scores=scores)


def select_merge_partner(
    adjacency: np.ndarray,
    node_ids: Sequence[str],
    target_id: str,
    execution_order: Optional[Sequence[str]] = None,
) -> Optional[MergePartner]:
    r"""The neighbour with the strongest learned interaction with the target.

    Candidates are every node connected to the target in either direction, and
    the score is :math:`S(c^{*}, j) = \max(\tilde A_{j,c^{*}}, \tilde A_{c^{*},j})`.
    Ties go first to a node that sends *to* the target, then to whichever of
    those sent most recently in the round's execution order, since that is the
    agent whose output most directly shaped the target's reasoning.
    """
    target = list(node_ids).index(target_id)
    order = list(execution_order) if execution_order is not None else list(node_ids)

    candidates: List[Dict[str, float]] = []
    for index, node_id in enumerate(node_ids):
        if index == target:
            continue
        incoming = float(adjacency[index, target])
        outgoing = float(adjacency[target, index])
        if incoming <= 0 and outgoing <= 0:
            continue
        candidates.append(
            {
                "node_id": node_id,
                "score": max(incoming, outgoing),
                "sends_to_target": incoming >= outgoing,
                "position": order.index(node_id) if node_id in order else -1,
            }
        )

    if not candidates:
        return None

    best_score = max(candidate["score"] for candidate in candidates)
    tied = [c for c in candidates if c["score"] == best_score]
    if len(tied) > 1:
        senders = [c for c in tied if c["sends_to_target"]]
        tied = senders or tied
        tied.sort(key=lambda c: c["position"], reverse=True)

    winner = tied[0]
    return MergePartner(node_id=str(winner["node_id"]), score=float(winner["score"]))
