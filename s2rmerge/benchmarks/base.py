"""Benchmark interface.

A benchmark supplies the train/test records, the kind of agent that answers
them, and the rule that decides whether a response is correct. Everything the
pipeline needs to know about a dataset lives behind this interface, so the
route/merge/prune machinery is written once rather than once per dataset.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass(frozen=True)
class Record:
    """One benchmark item."""

    task: str
    """The query presented to the agents."""

    answer: str
    """The reference answer, or the test harness for code benchmarks."""

    meta: Dict[str, Any] = field(default_factory=dict)

    def as_input(self) -> Dict[str, str]:
        """The payload a communication graph is run on."""
        return {"task": self.task}


class Benchmark(ABC):
    """A dataset plus the agent configuration used to answer it."""

    name: str
    domain: str
    """Key the prompt set is registered under."""

    agent_type: str
    """Key the answering agent is registered under."""

    decision_method: str = "FinalRefer"
    """Agent that aggregates the final answer."""

    rounds: int = 2
    """Communication rounds per query (T in the paper)."""

    merge_iterations: int = 1
    """Merge depth chosen for this benchmark by the accuracy-token trade-off."""

    metric_name: str = "accuracy"

    @abstractmethod
    def load(self, split: str) -> List[Record]:
        """Records for ``split``, one of ``"train"`` or ``"test"``."""

    @abstractmethod
    def is_correct(self, response: Optional[str], record: Record) -> bool:
        """Whether ``response`` answers ``record`` correctly."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r} rounds={self.rounds}>"
