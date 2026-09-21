"""Merge-based structural optimisation of the communication graph."""

from s2rmerge.merge.criterion import (
    MergePartner,
    MergeTarget,
    select_merge_partner,
    select_merge_target,
    transfer_efficiency,
)
from s2rmerge.merge.fusion import concatenate, fuse_prompts
from s2rmerge.merge.operator import MergeOutcome, merge_once

__all__ = [
    "MergeOutcome",
    "MergePartner",
    "MergeTarget",
    "concatenate",
    "fuse_prompts",
    "merge_once",
    "select_merge_partner",
    "select_merge_target",
    "transfer_efficiency",
]
