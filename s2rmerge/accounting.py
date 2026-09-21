"""Token and cost accounting, attributed to the pipeline stage that spent them.

Every LLM call made anywhere in the project ends up in :data:`LEDGER` through
:meth:`Ledger.record`. The stage a call belongs to is taken from a context
variable rather than passed around, so the attribution survives the
``asyncio.gather`` fan-out used when a communication graph runs, and no caller
has to snapshot counters before and after a phase.

Reported prompt/completion token totals are the sum over all stages.
"""

from __future__ import annotations

import contextvars
from contextlib import contextmanager
from dataclasses import dataclass, replace
from typing import Dict, Iterator, Tuple

ROUTER = "router"
OPTIMIZATION = "optimization"
FUSION = "fusion"
INFERENCE = "inference"

STAGES: Tuple[str, ...] = (ROUTER, OPTIMIZATION, FUSION, INFERENCE)

_active_stage: contextvars.ContextVar[str] = contextvars.ContextVar(
    "s2rmerge_stage", default=INFERENCE
)


@dataclass(frozen=True)
class Usage:
    """Tokens and dollar cost spent by one stage."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens

    def __add__(self, other: "Usage") -> "Usage":
        return Usage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.cost + other.cost,
        )

    def as_dict(self) -> Dict[str, float]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost": self.cost,
        }

    def __str__(self) -> str:
        return (
            f"{self.prompt_tokens:,} prompt / {self.completion_tokens:,} completion "
            f"tokens, ${self.cost:.4f}"
        )


class Ledger:
    """Per-stage token and cost totals for a single experiment run."""

    def __init__(self) -> None:
        self._usage: Dict[str, Usage] = {stage: Usage() for stage in STAGES}

    def reset(self) -> None:
        self._usage = {stage: Usage() for stage in STAGES}

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Attribute every call made inside this block to ``name``."""
        if name not in self._usage:
            raise ValueError(f"unknown stage {name!r}, expected one of {STAGES}")
        token = _active_stage.set(name)
        try:
            yield
        finally:
            _active_stage.reset(token)

    def record(self, prompt_tokens: int, completion_tokens: int, cost: float = 0.0) -> None:
        stage = _active_stage.get()
        current = self._usage[stage]
        self._usage[stage] = replace(
            current,
            prompt_tokens=current.prompt_tokens + int(prompt_tokens),
            completion_tokens=current.completion_tokens + int(completion_tokens),
            cost=current.cost + float(cost),
        )

    def __getitem__(self, stage: str) -> Usage:
        return self._usage[stage]

    @property
    def total(self) -> Usage:
        return sum(self._usage.values(), Usage())

    def as_dict(self) -> Dict[str, Dict[str, float]]:
        report = {stage: usage.as_dict() for stage, usage in self._usage.items()}
        report["total"] = self.total.as_dict()
        return report

    def report(self) -> str:
        lines = [f"{'stage':<14}{'prompt':>14}{'completion':>14}{'cost':>12}"]
        for stage in STAGES:
            usage = self._usage[stage]
            lines.append(
                f"{stage:<14}{usage.prompt_tokens:>14,}{usage.completion_tokens:>14,}"
                f"{usage.cost:>12.4f}"
            )
        total = self.total
        lines.append(
            f"{'total':<14}{total.prompt_tokens:>14,}{total.completion_tokens:>14,}"
            f"{total.cost:>12.4f}"
        )
        return "\n".join(lines)


LEDGER = Ledger()
