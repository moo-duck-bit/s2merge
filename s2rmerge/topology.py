"""Initial communication topologies.

A topology is the fixed mask over candidate edges: an entry of 0 means the edge
can never exist, 1 means its presence is left to the learned logit. The paper's
robustness study runs the same method over three of these.
"""

from __future__ import annotations

import random
from typing import Dict, List

Mask = List[List[int]]

TOPOLOGIES = ("FullConnected", "Layered", "Random", "Chain", "Star", "Debate")


def _full_connected(n: int) -> Mask:
    return [[int(i != j) for j in range(n)] for i in range(n)]


def _layered(n: int, layers: int = 2) -> Mask:
    """Agents split into layers; each layer only feeds the next one."""
    base, remainder = divmod(n, layers)
    layer_of: List[int] = []
    for layer in range(layers):
        layer_of.extend([layer] * (base + (1 if layer < remainder else 0)))
    return [[int(layer_of[j] == layer_of[i] + 1) for j in range(n)] for i in range(n)]


def _random(n: int, rng: random.Random) -> Mask:
    return [[rng.randint(0, 1) if i != j else 0 for j in range(n)] for i in range(n)]


def _chain(n: int) -> Mask:
    return [[int(j == i + 1) for j in range(n)] for i in range(n)]


def _star(n: int) -> Mask:
    """Upper-triangular: every agent feeds all later agents."""
    return [[int(j > i) for j in range(n)] for i in range(n)]


def _debate(n: int) -> Mask:
    """No intra-round edges; agents only see each other across rounds."""
    return [[0] * n for _ in range(n)]


def build_masks(topology: str, n: int, seed: int = 0) -> Dict[str, Mask]:
    """Spatial and temporal fixed masks for ``n`` agents."""
    if topology not in TOPOLOGIES:
        raise ValueError(f"unknown topology {topology!r}; choose from {list(TOPOLOGIES)}")

    all_to_all = [[1] * n for _ in range(n)]

    if topology == "FullConnected":
        spatial = _full_connected(n)
    elif topology == "Layered":
        spatial = _layered(n)
    elif topology == "Random":
        spatial = _random(n, random.Random(seed))
    elif topology == "Chain":
        spatial = _chain(n)
    elif topology == "Star":
        spatial = _star(n)
    else:
        spatial = _debate(n)

    return {"fixed_spatial_masks": spatial, "fixed_temporal_masks": all_to_all}
