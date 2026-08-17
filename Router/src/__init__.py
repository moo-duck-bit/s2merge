"""
2-Stage Soft-Gated, Uncertainty-Aware Multi-Agent Router
"""

from .router import Router
from .stage0 import Stage0RepresentationGenerator
from .stage1 import Stage1BlockRouter
from .stage2 import Stage2RoleRouter
from .utils import (
    load_config,
    cosine_similarity,
    softmax_with_temperature,
    normalized_entropy,
    adaptive_coverage_threshold,
    cumulative_selection
)

__version__ = "1.0.0"

__all__ = [
    "Router",
    "Stage0RepresentationGenerator",
    "Stage1BlockRouter",
    "Stage2RoleRouter",
    "load_config",
    "cosine_similarity",
    "softmax_with_temperature",
    "normalized_entropy",
    "adaptive_coverage_threshold",
    "cumulative_selection"
]
