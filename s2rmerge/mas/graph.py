"""The communication graph agents collaborate over.

Nodes are agents; a directed edge carries one message. Every possible edge has a
learnable logit, and sigmoid(logit) is the probability that the edge is present
in a sampled graph. Sampling a graph and scoring the answer it produces gives
the policy gradient that trains those logits, which is what the paper's
optimisation step does:

    A <- A + eta * (1/M) * sum_m  mu(G_m) * grad log p(G_m)

Edges come in two kinds. *Spatial* edges connect agents inside one communication
round; they are constrained to stay acyclic so a round has a valid execution
order. *Temporal* edges carry an agent's previous-round output into the next
round. Each round owns its own parameter vector, so a graph with ``T`` rounds
holds ``T`` spatial vectors and ``T - 1`` temporal ones.

Adapted from AgentDropout (ACL 2025); the node-dropout sampling of the original
is not used here, because S2R-Merge consolidates nodes instead of dropping them.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import shortuuid
import torch

from s2rmerge.mas.agents.registry import AgentRegistry
from s2rmerge.mas.logs import log_agent_turn
from s2rmerge.mas.node import Node

DEFAULT_EDGE_PROBABILITY = 0.5
NODE_TIMEOUT_SECONDS = 6000
MAX_NODE_RETRIES = 3


def _probability_to_logit(probability: float) -> torch.Tensor:
    return torch.log(torch.tensor(probability / (1 - probability)))


class Graph:
    """A pool of agents plus the learnable topology they talk over."""

    def __init__(
        self,
        domain: str,
        llm_name: Optional[str],
        agent_names: Sequence[str],
        decision_method: str,
        fixed_spatial_masks: Optional[Sequence[Sequence[int]]] = None,
        fixed_temporal_masks: Optional[Sequence[Sequence[int]]] = None,
        node_kwargs: Optional[Sequence[Dict[str, Any]]] = None,
        rounds: int = 1,
        optimized: bool = True,
        initial_edge_probability: float = DEFAULT_EDGE_PROBABILITY,
    ):
        n = len(agent_names)
        if fixed_spatial_masks is None:
            fixed_spatial_masks = [[int(i != j) for j in range(n)] for i in range(n)]
        if fixed_temporal_masks is None:
            fixed_temporal_masks = [[1] * n for _ in range(n)]

        self.id = shortuuid.ShortUUID().random(length=4)
        self.domain = domain
        self.llm_name = llm_name
        self.agent_names = list(agent_names)
        self.rounds = rounds
        self.optimized = optimized
        self.node_kwargs = list(node_kwargs) if node_kwargs is not None else [{} for _ in agent_names]
        self.conversation_log_path = None

        self.nodes: Dict[str, Node] = {}
        self.potential_spatial_edges: List[Tuple[str, str]] = []
        self.potential_temporal_edges: List[Tuple[str, str]] = []

        self.decision_node: Node = AgentRegistry.get(
            decision_method, domain=self.domain, llm_name=self.llm_name
        )

        self._init_nodes()
        self._init_potential_edges()

        self.fixed_spatial_masks = torch.tensor(fixed_spatial_masks, dtype=torch.float32)
        self.fixed_temporal_masks = torch.tensor(fixed_temporal_masks, dtype=torch.float32)
        self._assert_mask_shapes()

        self.reset_parameters(initial_edge_probability)

    # ------------------------------------------------------------------ setup

    def _init_nodes(self) -> None:
        for agent_name, kwargs in zip(self.agent_names, self.node_kwargs):
            node = AgentRegistry.get(
                agent_name, domain=self.domain, llm_name=self.llm_name, **kwargs
            )
            self.add_node(node)

    def _init_potential_edges(self) -> None:
        self.potential_spatial_edges = [
            (source, target) for source in self.nodes for target in self.nodes
        ]
        self.potential_temporal_edges = list(self.potential_spatial_edges)

    def _assert_mask_shapes(self) -> None:
        n = self.num_nodes
        for name, mask in (
            ("fixed_spatial_masks", self.fixed_spatial_masks),
            ("fixed_temporal_masks", self.fixed_temporal_masks),
        ):
            if tuple(mask.shape) != (n, n):
                raise ValueError(
                    f"{name} has shape {tuple(mask.shape)} but the graph has {n} agents"
                )

    def add_node(self, node: Node) -> Node:
        node_id = node.id or shortuuid.ShortUUID().random(length=4)
        while node_id in self.nodes:
            node_id = shortuuid.ShortUUID().random(length=4)
        node.id = node_id
        self.nodes[node_id] = node
        return node

    def reset_parameters(self, initial_edge_probability: float = DEFAULT_EDGE_PROBABILITY) -> None:
        """Re-initialise every edge logit and mask for the current node set.

        Called once at construction and again after a merge, because merging
        changes the node set and invalidates the weights learned for the old
        topology.
        """
        self._init_potential_edges()
        self._assert_mask_shapes()

        n_spatial = len(self.potential_spatial_edges)
        n_temporal = len(self.potential_temporal_edges)
        init_logit = _probability_to_logit(initial_edge_probability)

        flat_spatial = self.fixed_spatial_masks.reshape(-1)
        flat_temporal = self.fixed_temporal_masks.reshape(-1)

        self.spatial_logits = torch.nn.ParameterList(
            torch.nn.Parameter(torch.full((n_spatial,), init_logit), requires_grad=self.optimized)
            for _ in range(self.rounds)
        )
        self.spatial_masks = [flat_spatial.clone() for _ in range(self.rounds)]

        self.temporal_logits = torch.nn.ParameterList(
            torch.nn.Parameter(torch.full((n_temporal,), init_logit), requires_grad=self.optimized)
            for _ in range(max(self.rounds - 1, 0))
        )
        self.temporal_masks = [flat_temporal.clone() for _ in range(max(self.rounds - 1, 0))]

    def parameters(self) -> List[torch.nn.Parameter]:
        return list(self.spatial_logits) + list(self.temporal_logits)

    # ------------------------------------------------------------- properties

    @property
    def num_nodes(self) -> int:
        return len(self.nodes)

    @property
    def node_ids(self) -> List[str]:
        return list(self.nodes)

    def find_node(self, node_id: str) -> Node:
        try:
            return self.nodes[node_id]
        except KeyError:
            raise KeyError(f"no node {node_id!r} among {self.node_ids}") from None

    def spatial_adjacency(self) -> torch.Tensor:
        """Learned edge weights, averaged over rounds, as an ``n x n`` matrix.

        Entry ``(i, j)`` is the probability that agent ``i`` sends to agent
        ``j``, and is exactly zero for an edge the topology or a pruning step
        has switched off. This is the matrix the merge criterion reads.
        """
        n = self.num_nodes
        with torch.no_grad():
            per_round = torch.stack(
                [
                    (torch.sigmoid(logits) * masks).reshape(n, n)
                    for logits, masks in zip(self.spatial_logits, self.spatial_masks)
                ]
            )
        return per_round.mean(dim=0)

    # --------------------------------------------------------- graph sampling

    def _clear_spatial_connections(self) -> None:
        for node in self.nodes.values():
            node.spatial_predecessors = []
            node.spatial_successors = []
        self.decision_node.spatial_predecessors = []
        self.decision_node.spatial_successors = []

    def _clear_temporal_connections(self) -> None:
        for node in self.nodes.values():
            node.temporal_predecessors = []
            node.temporal_successors = []

    def _creates_cycle(self, candidate: Node, targets: set) -> bool:
        if candidate in targets:
            return True
        return any(self._creates_cycle(successor, targets) for successor in candidate.spatial_successors)

    def sample_spatial_edges(self, round_index: int, temperature: float = 1.0) -> torch.Tensor:
        """Draw this round's intra-round edges; returns log p of the draw."""
        self._clear_spatial_connections()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized)]

        logits = self.spatial_logits[round_index]
        masks = self.spatial_masks[round_index]

        for (source_id, target_id), logit, mask in zip(self.potential_spatial_edges, logits, masks):
            if mask == 0.0:
                continue
            source, target = self.find_node(source_id), self.find_node(target_id)
            if self._creates_cycle(target, {source}):
                continue

            if not self.optimized:
                source.add_successor(target, "spatial")
                continue

            edge_probability = torch.sigmoid(logit / temperature)
            if torch.rand(1) < edge_probability:
                source.add_successor(target, "spatial")
                log_probs.append(torch.log(edge_probability))
            else:
                log_probs.append(torch.log(1 - edge_probability))

        return torch.sum(torch.stack(log_probs))

    def sample_temporal_edges(self, round_index: int, temperature: float = 1.0) -> torch.Tensor:
        """Draw the edges carrying the previous round's outputs into this one."""
        self._clear_temporal_connections()
        log_probs = [torch.tensor(0.0, requires_grad=self.optimized)]
        if round_index == 0:
            return torch.sum(torch.stack(log_probs))

        logits = self.temporal_logits[round_index - 1]
        masks = self.temporal_masks[round_index - 1]

        for (source_id, target_id), logit, mask in zip(self.potential_temporal_edges, logits, masks):
            if mask == 0.0:
                continue
            source, target = self.find_node(source_id), self.find_node(target_id)

            if not self.optimized:
                source.add_successor(target, "temporal")
                continue

            edge_probability = torch.sigmoid(logit / temperature)
            if torch.rand(1) < edge_probability:
                source.add_successor(target, "temporal")
                log_probs.append(torch.log(edge_probability))
            else:
                log_probs.append(torch.log(1 - edge_probability))

        return torch.sum(torch.stack(log_probs))

    # --------------------------------------------------------------- forward

    async def arun(self, task: Dict[str, str], num_rounds: Optional[int] = None) -> Tuple[List[Any], torch.Tensor]:
        """Run one query through the graph.

        Returns the decision agent's answer and the log-probability of the
        sampled topology, which is what the policy gradient differentiates.
        """
        num_rounds = num_rounds or self.rounds
        log_prob = torch.tensor(0.0, requires_grad=self.optimized)

        for round_index in range(num_rounds):
            log_prob = log_prob + self.sample_spatial_edges(round_index)
            log_prob = log_prob + self.sample_temporal_edges(round_index)
            await self._execute_round(task, round_index)
            self._update_memory()

        final_answers = await self._decide(task, num_rounds)
        return final_answers, log_prob

    async def _execute_round(self, task: Dict[str, str], round_index: int) -> None:
        """Run every agent once, in an order consistent with the sampled edges."""
        in_degree = {
            node_id: len(node.spatial_predecessors) for node_id, node in self.nodes.items()
        }
        ready = [node_id for node_id, degree in in_degree.items() if degree == 0]

        while ready:
            node_id = ready.pop(0)
            node = self.nodes[node_id]
            await self._execute_node(node, task, round_index)

            for successor in node.spatial_successors:
                if successor.id not in self.nodes:
                    continue
                in_degree[successor.id] -= 1
                if in_degree[successor.id] == 0:
                    ready.append(successor.id)

    async def _execute_node(self, node: Node, task: Dict[str, str], round_index: int) -> None:
        last_error: Optional[Exception] = None
        for _ in range(MAX_NODE_RETRIES):
            try:
                await asyncio.wait_for(node.async_execute(task), timeout=NODE_TIMEOUT_SECONDS)
                last_error = None
                break
            except Exception as error:  # noqa: BLE001 - one agent must not abort the query
                last_error = error

        if last_error is not None:
            print(f"[graph {self.id}] node {node.id} ({node.role}) failed: {last_error}")
            node.outputs = ["None."]

        log_agent_turn(
            self.conversation_log_path,
            agent=f"{node.role} ({node.id})",
            round_index=round_index + 1,
            task=task.get("task", ""),
            output=node.outputs,
        )

    async def _decide(self, task: Dict[str, str], num_rounds: int) -> List[Any]:
        if not self.potential_spatial_edges:
            return list(self.nodes.values())[0].outputs

        for node in self.nodes.values():
            node.add_successor(self.decision_node)

        await self.decision_node.async_execute(task)
        final_answers = self.decision_node.outputs or ["No answer from the decision node"]

        log_agent_turn(
            self.conversation_log_path,
            agent=f"{self.decision_node.role} (decision)",
            round_index=num_rounds + 1,
            task=task.get("task", ""),
            output=final_answers,
        )
        return final_answers

    def _update_memory(self) -> None:
        for node in self.nodes.values():
            node.update_memory()

    # ------------------------------------------------------ structural edits

    def remove_node(self, node_id: str) -> Node:
        """Drop a node and every edge touching it, then shrink the fixed masks.

        The caller is expected to follow up with :meth:`reset_parameters`, since
        the learned logits are sized for the old node set.
        """
        index = self.node_ids.index(node_id)
        node = self.nodes.pop(node_id)

        node.clear_connections()
        for other in self.nodes.values():
            for attribute in (
                "spatial_predecessors",
                "spatial_successors",
                "temporal_predecessors",
                "temporal_successors",
            ):
                neighbours = getattr(other, attribute)
                setattr(other, attribute, [n for n in neighbours if n.id != node_id])

        keep = torch.tensor(
            [i for i in range(self.fixed_spatial_masks.shape[0]) if i != index], dtype=torch.long
        )
        self.fixed_spatial_masks = self.fixed_spatial_masks[keep][:, keep]
        self.fixed_temporal_masks = self.fixed_temporal_masks[keep][:, keep]
        self.agent_names.pop(index)
        self.node_kwargs.pop(index)
        return node

    def prune_edges(self, dropout_rate: float) -> None:
        """Keep the top ``ceil((1 - beta) * |E|)`` edges of each round by weight.

        This is the post-merge edge pruning step: it runs on the retrained
        weights of the merged topology, not on the pre-merge graph.
        """
        for logits, masks in zip(self.spatial_logits, self.spatial_masks):
            _prune_in_place(logits, masks, dropout_rate)
        for logits, masks in zip(self.temporal_logits, self.temporal_masks):
            _prune_in_place(logits, masks, dropout_rate)

    def sparsity(self) -> float:
        """Fraction of candidate edges still active, averaged over rounds."""
        masks = list(self.spatial_masks) + list(self.temporal_masks)
        if not masks:
            return 0.0
        return float(np.mean([mask.mean().item() for mask in masks]))

    def __repr__(self) -> str:
        return f"<Graph {self.id} nodes={self.num_nodes} rounds={self.rounds}>"


def _prune_in_place(logits: torch.Tensor, masks: torch.Tensor, dropout_rate: float) -> None:
    """Zero the lowest-weight active entries of ``masks``."""
    active = masks > 0
    num_active = int(active.sum().item())
    if num_active == 0:
        return

    num_keep = int(np.ceil((1 - dropout_rate) * num_active))
    num_drop = max(num_active - num_keep, 1 if dropout_rate > 0 else 0)
    if num_drop == 0:
        return

    with torch.no_grad():
        scores = logits.detach().clone()
        scores[~active] = float("inf")  # never re-drop an already inactive edge
        drop_idx = torch.argsort(scores)[:num_drop]
        masks[drop_idx] = 0.0
