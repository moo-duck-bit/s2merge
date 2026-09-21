"""The two-stage soft-gated, uncertainty-aware router.

Given a query it returns the agents that should collaborate on it: Stage-0
builds one semantic representation, Stage-1 fixes the domain scope, Stage-2
picks the agents inside that scope. Neither stage uses a fixed Top-K; both take
as many candidates as the routing uncertainty calls for.

Routing is per query. Every query gets its own representation, its own
uncertainty and its own agent set, which is what makes the selection adaptive
rather than a dataset-level configuration.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from s2rmerge import accounting
from s2rmerge.accounting import LEDGER
from s2rmerge.router.config import RouterConfig, load_router_config
from s2rmerge.router.prototypes import load_prototypes
from s2rmerge.router.stage0 import RepresentationGenerator
from s2rmerge.router.stage1 import BlockRouter, StageDecision
from s2rmerge.router.stage2 import RoleRouter

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RoutingResult:
    """The agent set chosen for one query."""

    agents: List[str]
    agent_probabilities: Dict[str, float]
    blocks: StageDecision
    roles: StageDecision
    domain_hint: str

    @property
    def num_agents(self) -> int:
        return len(self.agents)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "agents": list(self.agents),
            "agent_probabilities": dict(self.agent_probabilities),
            "blocks": self.blocks.as_dict(),
            "roles": self.roles.as_dict(),
            "domain_hint": self.domain_hint,
        }

    def __str__(self) -> str:
        agents = ", ".join(
            f"{agent} ({self.agent_probabilities[agent]:.3f})" for agent in self.agents
        )
        return (
            f"{self.num_agents} agents from blocks {self.blocks.selected} "
            f"(block u={self.blocks.uncertainty:.3f}, role u={self.roles.uncertainty:.3f}): "
            f"{agents}"
        )


@lru_cache(maxsize=4)
def _load_embedding_model(name: str):
    from sentence_transformers import SentenceTransformer

    # Kept on CPU so it does not compete with a local vLLM server for GPU memory.
    return SentenceTransformer(name, device="cpu")


class Router:
    """Selects the agents that answer a query."""

    def __init__(
        self,
        config_path: Path,
        embedding_model=None,
        llm_client=None,
    ):
        self.config: RouterConfig = load_router_config(Path(config_path))

        self.embedding_model = embedding_model or _load_embedding_model(
            self.config.embedding_model
        )
        if llm_client is None:
            from s2rmerge.router.llm_client import create_llm_client

            llm_client = create_llm_client(self.config)
        self.llm_client = llm_client

        block_prototypes, role_prototypes = load_prototypes(self.config, self.embedding_model)
        self.stage0 = RepresentationGenerator(self.config, self.embedding_model, self.llm_client)
        self.stage1 = BlockRouter(self.config, block_prototypes)
        self.stage2 = RoleRouter(self.config, role_prototypes)

        logger.info(
            "router ready: %d blocks, %d roles, embeddings from %s",
            len(block_prototypes),
            len(role_prototypes),
            self.config.embedding_model,
        )

    def route(self, question: str, use_summary: bool = True) -> RoutingResult:
        """Choose the agent set for one query."""
        with LEDGER.stage(accounting.ROUTER):
            representation = self.stage0.build(question, use_summary=use_summary)

        blocks = self.stage1.route(representation.vector, representation.domain_hint)
        roles = self.stage2.route(representation.vector, blocks.selected)

        agents = sorted(
            roles.selected, key=lambda role: roles.probabilities[role], reverse=True
        )
        return RoutingResult(
            agents=agents,
            agent_probabilities=roles.probabilities,
            blocks=blocks,
            roles=roles,
            domain_hint=representation.domain_hint,
        )

    def route_all(self, questions: List[str], use_summary: bool = True) -> List[RoutingResult]:
        return [self.route(question, use_summary=use_summary) for question in questions]
