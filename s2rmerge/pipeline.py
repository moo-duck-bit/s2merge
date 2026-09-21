"""The S2R-Merge pipeline: route, optimise, merge, prune, evaluate.

For each query the router picks an agent set. Queries that were routed to the
same set share one communication graph, which is then optimised by policy
gradient, compressed by absorption-based node merges, and finally edge-pruned.
Running with ``router_config=None`` skips routing and puts the whole agent pool
in a single graph, which is the "w/o Router" ablation.

The full sequence for one graph is:

1. policy-gradient optimisation of the edge logits on the training split;
2. ``merge_iterations`` absorptions, each followed by re-optimisation, because
   a merge changes the topology and invalidates the learned weights;
3. edge pruning on the merged topology;
4. evaluation on the test split.
"""

from __future__ import annotations

import asyncio
import copy
import logging
import random
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from s2rmerge import accounting
from s2rmerge.accounting import LEDGER
from s2rmerge.benchmarks import Accuracy, Benchmark, Record, get_benchmark
from s2rmerge.mas.graph import Graph
from s2rmerge.mas.logs import log_note, open_transcript
from s2rmerge.mas.prompts.registry import PromptSetRegistry
from s2rmerge.merge import merge_once
from s2rmerge.paths import RESULT_DIR
from s2rmerge.router import Router
from s2rmerge.topology import build_masks

logger = logging.getLogger(__name__)


@dataclass
class Settings:
    """Everything that varies between runs."""

    benchmark: str
    llm_name: str
    router_config: Optional[Path] = None
    topology: str = "FullConnected"
    merge_iterations: Optional[int] = None
    optimization_steps: int = 5
    batch_size: int = 4
    learning_rate: float = 0.1
    edge_dropout: float = 0.1
    rounds: Optional[int] = None
    train_samples: Optional[int] = None
    test_samples: Optional[int] = None
    max_agent_groups: Optional[int] = None
    use_fusion: bool = True
    seed: int = 0
    transcript: bool = True

    @property
    def use_router(self) -> bool:
        return self.router_config is not None


@dataclass
class GroupReport:
    """Result for one routed agent set."""

    agents: List[str]
    num_test: int
    accuracy: float
    merges: List[str] = field(default_factory=list)
    final_nodes: int = 0
    final_sparsity: float = 0.0


@dataclass
class RunReport:
    """Everything a run produced."""

    benchmark: str
    metric_name: str
    score: float
    num_evaluated: int
    groups: List[GroupReport]
    usage: Dict[str, Dict[str, float]]
    seconds: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "benchmark": self.benchmark,
            "metric": self.metric_name,
            "score": self.score,
            "num_evaluated": self.num_evaluated,
            "seconds": self.seconds,
            "usage": self.usage,
            "groups": [
                {
                    "agents": group.agents,
                    "num_test": group.num_test,
                    "accuracy": group.accuracy,
                    "merges": group.merges,
                    "final_nodes": group.final_nodes,
                    "final_sparsity": group.final_sparsity,
                }
                for group in self.groups
            ],
        }


AgentSet = Tuple[str, ...]


class Pipeline:
    """Runs one benchmark end to end."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.benchmark: Benchmark = get_benchmark(settings.benchmark)
        self.rounds = settings.rounds or self.benchmark.rounds
        self.merge_iterations = (
            settings.merge_iterations
            if settings.merge_iterations is not None
            else self.benchmark.merge_iterations
        )
        self.router: Optional[Router] = (
            Router(settings.router_config) if settings.use_router else None
        )
        if self.router is not None:
            self._check_roles_exist(self.router.config.role_ids)

        random.seed(settings.seed)
        torch.manual_seed(settings.seed)

        self.run_name = f"{settings.benchmark}_{time.strftime('%Y%m%d-%H%M%S')}"
        self.output_dir = RESULT_DIR / self.run_name
        self.transcript_path = (
            open_transcript(self.output_dir / "transcript.txt", self.run_name)
            if settings.transcript
            else None
        )

    # ------------------------------------------------------------------- run

    async def run(self) -> RunReport:
        started = time.time()
        LEDGER.reset()

        train_records = self._load("train", self.settings.train_samples)
        test_records = self._load("test", self.settings.test_samples)
        logger.info(
            "%s: %d training and %d test records",
            self.benchmark.name,
            len(train_records),
            len(test_records),
        )

        groups = self._group_by_agent_set(train_records, test_records)
        logger.info("%d distinct agent set(s) after routing", len(groups))

        reports: List[GroupReport] = []
        correct = 0
        evaluated = 0

        for agents, (group_train, group_test) in groups.items():
            report = await self._run_group(list(agents), group_train, group_test)
            reports.append(report)
            correct += round(report.accuracy * report.num_test)
            evaluated += report.num_test

        score = correct / evaluated if evaluated else 0.0
        return RunReport(
            benchmark=self.benchmark.name,
            metric_name=self.benchmark.metric_name,
            score=score,
            num_evaluated=evaluated,
            groups=reports,
            usage=LEDGER.as_dict(),
            seconds=time.time() - started,
        )

    def _check_roles_exist(self, role_ids: Sequence[str]) -> None:
        """Check that every role the router can select exists in the prompt set.

        Run at startup so a config and its prompt set cannot drift apart
        unnoticed over the course of an experiment.
        """
        known = set(self._full_pool())
        unknown = [role for role in role_ids if role not in known]
        if unknown:
            raise ValueError(
                f"router config lists roles the {self.benchmark.domain!r} prompt set "
                f"does not define: {unknown}"
            )

    def _load(self, split: str, limit: Optional[int]) -> List[Record]:
        if split == "train" and not getattr(self.benchmark, "has_train_split", True):
            logger.warning(
                "%s has no separate training split; the graph is optimised on the "
                "evaluation set",
                self.benchmark.name,
            )
        records = self.benchmark.load(split)
        return records[:limit] if limit else records

    # --------------------------------------------------------------- routing

    def _full_pool(self) -> List[str]:
        return PromptSetRegistry.get(self.benchmark.domain).roles()

    def _group_by_agent_set(
        self, train_records: Sequence[Record], test_records: Sequence[Record]
    ) -> Dict[AgentSet, Tuple[List[Record], List[Record]]]:
        """Bucket queries by the agent set the router chose for them."""
        if self.router is None:
            pool = tuple(self._full_pool())
            return {pool: (list(train_records), list(test_records))}

        with LEDGER.stage(accounting.ROUTER):
            train_sets = [self._route(record) for record in train_records]
            test_sets = [self._route(record) for record in test_records]

        test_counts: Dict[AgentSet, int] = defaultdict(int)
        for agents in test_sets:
            test_counts[agents] += 1

        kept = self._limit_groups(test_counts)
        groups: Dict[AgentSet, Tuple[List[Record], List[Record]]] = {
            agents: ([], []) for agents in kept
        }

        for record, agents in zip(train_records, train_sets):
            groups[_nearest(agents, kept)][0].append(record)
        for record, agents in zip(test_records, test_sets):
            groups[_nearest(agents, kept)][1].append(record)

        return {agents: split for agents, split in groups.items() if split[1]}

    def _route(self, record: Record) -> AgentSet:
        result = self.router.route(record.task)
        log_note(self.transcript_path, f"router: {result}")
        return tuple(result.agents)

    def _limit_groups(self, counts: Dict[AgentSet, int]) -> List[AgentSet]:
        ordered = sorted(counts, key=counts.__getitem__, reverse=True)
        limit = self.settings.max_agent_groups
        if limit is None or len(ordered) <= limit:
            return ordered
        logger.info(
            "keeping the %d most common agent sets; %d rarer ones are folded into "
            "the closest kept set",
            limit,
            len(ordered) - limit,
        )
        return ordered[:limit]

    # ----------------------------------------------------------------- group

    async def _run_group(
        self, agents: List[str], train_records: List[Record], test_records: List[Record]
    ) -> GroupReport:
        logger.info("agent set %s: %d test queries", agents, len(test_records))
        graph = self._build_graph(agents)

        if not train_records:
            logger.warning("no training queries routed here; using the test split to optimise")
            train_records = test_records

        await self._optimize(graph, train_records)

        merges: List[str] = []
        for iteration in range(self.merge_iterations):
            outcome = await merge_once(
                graph, llm_name=self.settings.llm_name, use_fusion=self.settings.use_fusion
            )
            if outcome is None:
                logger.info("no further merge is possible after %d iteration(s)", iteration)
                break
            logger.info("merge %d: %s", iteration + 1, outcome)
            merges.append(str(outcome))
            # The merge reset the edge logits, so relearn them on the new topology.
            await self._optimize(graph, train_records)

        graph.prune_edges(self.settings.edge_dropout)

        accuracy = await self._evaluate(graph, test_records)
        return GroupReport(
            agents=agents,
            num_test=len(test_records),
            accuracy=accuracy.value,
            merges=merges,
            final_nodes=graph.num_nodes,
            final_sparsity=graph.sparsity(),
        )

    def _build_graph(self, agents: List[str]) -> Graph:
        masks = build_masks(self.settings.topology, len(agents), seed=self.settings.seed)
        graph = Graph(
            domain=self.benchmark.domain,
            llm_name=self.settings.llm_name,
            agent_names=[self.benchmark.agent_type] * len(agents),
            decision_method=self.benchmark.decision_method,
            node_kwargs=[{"role": role} for role in agents],
            rounds=self.rounds,
            **masks,
        )
        graph.conversation_log_path = self.transcript_path
        return graph

    # ---------------------------------------------------------- optimisation

    async def _optimize(self, graph: Graph, records: Sequence[Record]) -> None:
        r"""Policy-gradient update of the edge logits.

        Each step samples one communication graph per query in the batch, scores
        the answers, and ascends :math:`\mu(G)\nabla \log p(G)` — the estimator
        used because an LLM's output is not differentiable.
        """
        optimizer = torch.optim.Adam(graph.parameters(), lr=self.settings.learning_rate)

        with LEDGER.stage(accounting.OPTIMIZATION):
            for step in range(self.settings.optimization_steps):
                batch = _batch(records, self.settings.batch_size, step)
                if not batch:
                    break

                started = time.time()
                answers, log_probs = await self._run_batch(graph, batch)

                losses = []
                utilities = []
                for answer, log_prob, record in zip(answers, log_probs, batch):
                    utility = float(self.benchmark.is_correct(answer, record))
                    utilities.append(utility)
                    losses.append(-log_prob * utility)

                loss = torch.mean(torch.stack(losses))
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                logger.info(
                    "  step %d/%d: utility %.2f, loss %.4f (%.1fs)",
                    step + 1,
                    self.settings.optimization_steps,
                    sum(utilities) / len(utilities),
                    loss.item(),
                    time.time() - started,
                )

    async def _run_batch(self, graph: Graph, batch: Sequence[Record]):
        """Run a batch concurrently, each query on its own copy of the graph.

        The copies share the logit tensors, so gradients from every query in the
        batch accumulate into the same parameters, but each query gets its own
        node outputs and sampled edges.
        """
        tasks = []
        for record in batch:
            realized = _clone_sharing_parameters(graph)
            tasks.append(asyncio.create_task(realized.arun(record.as_input(), self.rounds)))

        results = await asyncio.gather(*tasks)
        answers = [result[0][0] for result in results]
        log_probs = [result[1] for result in results]
        return answers, log_probs

    # ------------------------------------------------------------ evaluation

    async def _evaluate(self, graph: Graph, records: Sequence[Record]) -> Accuracy:
        accuracy = Accuracy()
        graph.optimized = False  # evaluate on the learned topology, do not resample

        with LEDGER.stage(accounting.INFERENCE):
            for start in range(0, len(records), self.settings.batch_size):
                batch = records[start : start + self.settings.batch_size]
                answers, _ = await self._run_batch(graph, batch)
                for answer, record in zip(answers, batch):
                    accuracy.update(self.benchmark.is_correct(answer, record))
                logger.info(
                    "  evaluated %d/%d: %s", accuracy.total, len(records), accuracy
                )

        return accuracy


def _batch(records: Sequence[Record], size: int, step: int) -> List[Record]:
    start = (step * size) % max(len(records), 1)
    return list(records[start : start + size])


def _clone_sharing_parameters(graph: Graph) -> Graph:
    clone = copy.deepcopy(graph)
    clone.spatial_logits = graph.spatial_logits
    clone.temporal_logits = graph.temporal_logits
    clone.spatial_masks = graph.spatial_masks
    clone.temporal_masks = graph.temporal_masks
    return clone


def _nearest(agents: AgentSet, kept: Sequence[AgentSet]) -> AgentSet:
    """The kept agent set sharing the most agents with ``agents``."""
    if agents in kept:
        return agents
    return max(kept, key=lambda candidate: len(set(candidate) & set(agents)))
