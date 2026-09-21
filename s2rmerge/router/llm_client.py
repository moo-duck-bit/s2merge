"""The LLM the router asks for a structured summary of a query.

Stage-0 makes one short, synchronous call per query. The backend is chosen by
``stage0.llm_model`` in the router config: a local model goes to the vLLM
server, anything else to the OpenAI API. Tokens land in the ledger under the
router stage, because in the paper's accounting they are the sunk cost of
routing rather than part of inference.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from s2rmerge.mas.llm.registry import LLMRegistry
from s2rmerge.router.config import RouterConfig

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are a helpful assistant that analyzes questions and provides structured "
    "summaries in JSON format. Provide ONLY the JSON output, no additional text."
)


class SummaryClient:
    """Synchronous wrapper around the async chat backends."""

    def __init__(self, model: str):
        self.model = model
        self.llm = LLMRegistry.get(model)

    def generate(self, prompt: str) -> str:
        """Return the model's reply, which Stage-0 parses as JSON."""
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]
        return _run_sync(self.llm.agen(messages))


class StubSummaryClient:
    """Fixed summary for tests and offline smoke runs.

    Selected only by asking for it, never as a silent fallback: a run that
    quietly routed on a canned summary would not be reproducing anything.
    """

    def __init__(self, summary: Optional[dict] = None):
        self.summary = summary or {
            "task_format": "question answering",
            "domain_hint": "general",
            "key_concepts": ["reasoning"],
            "routing_summary": "requires analytical reasoning",
        }
        logger.warning("using the stub summary client; routing will not reflect the query")

    def generate(self, prompt: str) -> str:
        return json.dumps(self.summary)


def _run_sync(coroutine):
    """Run ``coroutine`` to completion from synchronous code."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)

    # Already inside a loop (the pipeline is async), so nest this one call.
    import nest_asyncio

    nest_asyncio.apply()
    return asyncio.get_event_loop().run_until_complete(coroutine)


def create_llm_client(config: RouterConfig):
    """Build the Stage-0 client named by the router config."""
    if config.llm_model == "stub":
        return StubSummaryClient()
    return SummaryClient(config.llm_model)
