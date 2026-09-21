"""Block and role prototype embeddings.

A prototype is the embedding of a block's or role's written description. They
are cached on disk so a run does not re-embed them, but the cache is keyed by
the exact set of ids in the config: edit the config and the stale cache is
rebuilt rather than silently reused.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Mapping

import numpy as np

from s2rmerge.router.config import RouterConfig

logger = logging.getLogger(__name__)

Prototypes = Dict[str, np.ndarray]


def _load_cache(path: Path, expected_ids: Mapping[str, str]) -> Prototypes | None:
    if not path.exists():
        return None

    cached = np.load(path, allow_pickle=True).item()
    if set(cached) != set(expected_ids):
        logger.info("%s does not match the config; regenerating", path.name)
        return None
    return cached


def _build(descriptions: Mapping[str, str], embedding_model) -> Prototypes:
    return {key: embedding_model.encode(text) for key, text in descriptions.items()}


def _save(path: Path, prototypes: Prototypes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, prototypes)


def load_prototypes(config: RouterConfig, embedding_model) -> tuple[Prototypes, Prototypes]:
    """Return the block and role prototypes, embedding them if need be."""
    block_descriptions = {
        block_id: block.description for block_id, block in config.blocks.items()
    }
    role_descriptions = {
        role_id: config.role_descriptions[role_id] for role_id in config.role_ids
    }

    blocks = _load_cache(config.block_prototype_path, block_descriptions)
    if blocks is None:
        blocks = _build(block_descriptions, embedding_model)
        _save(config.block_prototype_path, blocks)

    roles = _load_cache(config.role_prototype_path, role_descriptions)
    if roles is None:
        roles = _build(role_descriptions, embedding_model)
        _save(config.role_prototype_path, roles)

    return blocks, roles
