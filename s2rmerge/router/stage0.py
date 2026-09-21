r"""Stage-0: one semantic representation, shared by both routing stages.

The surface form of a query often under-describes the expertise it needs, so an
LLM is asked for a structured summary of it — never for an answer. The summary
is a weak prior, not a replacement: the representation mixes it with the query
itself,

.. math:: v = \alpha\,\mathrm{Embed}(q) + (1 - \alpha)\,\mathrm{Embed}(t)

and this single vector is what both Stage-1 and Stage-2 score against. Keeping
one representation is deliberate — it means every difference between the stages
comes from their selection policy, not from a different view of the query.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np

from s2rmerge.router.config import RouterConfig

logger = logging.getLogger(__name__)

SUMMARY_FIELDS = ("task_format", "domain_hint", "key_concepts", "routing_summary")


@dataclass
class QuerySummary:
    """The routing-relevant signals an LLM read out of a query."""

    task_format: str = ""
    domain_hint: str = ""
    key_concepts: List[str] = field(default_factory=list)
    routing_summary: str = ""

    @classmethod
    def from_json(cls, payload: str) -> "QuerySummary":
        data = json.loads(payload)
        missing = [name for name in SUMMARY_FIELDS if name not in data]
        if missing:
            logger.warning("summary is missing %s; treating them as empty", ", ".join(missing))
        return cls(
            task_format=data.get("task_format", "") or "",
            domain_hint=data.get("domain_hint", "") or "",
            key_concepts=list(data.get("key_concepts", []) or []),
            routing_summary=data.get("routing_summary", "") or "",
        )

    def as_text(self) -> str:
        """Flatten the summary into the string that gets embedded."""
        return " | ".join(
            [
                f"Task format: {self.task_format}",
                f"Domain: {self.domain_hint}",
                f"Key concepts: {', '.join(self.key_concepts)}",
                f"Summary: {self.routing_summary}",
            ]
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "task_format": self.task_format,
            "domain_hint": self.domain_hint,
            "key_concepts": list(self.key_concepts),
            "routing_summary": self.routing_summary,
        }


@dataclass
class Representation:
    """The query vector and the summary it was built from."""

    vector: np.ndarray
    summary: Optional[QuerySummary]

    @property
    def domain_hint(self) -> str:
        return self.summary.domain_hint if self.summary else ""


class RepresentationGenerator:
    """Builds the mixed query representation of Stage-0."""

    def __init__(self, config: RouterConfig, embedding_model, llm_client):
        self.config = config
        self.embedding_model = embedding_model
        self.llm_client = llm_client

    def summarize(self, question: str) -> Optional[QuerySummary]:
        prompt = self.config.summary_prompt.format(question=question)
        try:
            return QuerySummary.from_json(self.llm_client.generate(prompt))
        except json.JSONDecodeError:
            logger.warning("summary was not valid JSON; routing on the query alone")
        except Exception as error:  # noqa: BLE001 - routing must survive a flaky call
            logger.warning("summary generation failed (%s); routing on the query alone", error)
        return None

    def build(self, question: str, use_summary: bool = True) -> Representation:
        question_embedding = self.embedding_model.encode(question)
        if not use_summary:
            return Representation(vector=question_embedding, summary=None)

        summary = self.summarize(question)
        if summary is None:
            return Representation(vector=question_embedding, summary=None)

        summary_embedding = self.embedding_model.encode(summary.as_text())
        alpha = self.config.alpha
        vector = alpha * question_embedding + (1 - alpha) * summary_embedding
        return Representation(vector=vector, summary=summary)
