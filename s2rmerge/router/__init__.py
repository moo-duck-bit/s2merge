"""Two-stage soft-gated, uncertainty-aware agent router."""

from s2rmerge.router.config import RouterConfig, load_router_config
from s2rmerge.router.router import Router, RoutingResult
from s2rmerge.router.stage1 import StageDecision

__all__ = [
    "Router",
    "RouterConfig",
    "RoutingResult",
    "StageDecision",
    "load_router_config",
]
