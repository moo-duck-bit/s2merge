"""Shared plumbing for agents that answer by calling an LLM."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from s2rmerge.mas.llm.registry import LLMRegistry
from s2rmerge.mas.node import Node
from s2rmerge.mas.prompts.registry import PromptSetRegistry

MAX_PEER_OUTPUT_CHARS = 2000
"""Peer outputs are truncated so a crowded round still fits the context window."""


class LLMNode(Node):
    """An agent whose answer is one chat completion.

    Subclasses supply :meth:`build_prompt`; everything else — resolving the
    role, its system prompt, and the call itself — is handled here.
    """

    agent_name = "LLMNode"

    def __init__(
        self,
        id: Optional[str] = None,
        role: Optional[str] = None,
        domain: str = "",
        llm_name: str = "",
    ):
        super().__init__(id, self.agent_name, domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = role if role is not None else self.prompt_set.get_role()
        self.constraint = self.prompt_set.get_constraint(self.role)

    def build_prompt(
        self,
        task: Dict[str, str],
        spatial_info: Dict[str, Dict[str, Any]],
        temporal_info: Dict[str, Dict[str, Any]],
    ) -> Tuple[str, str]:
        """Return the ``(system, user)`` prompt pair for this turn."""
        raise NotImplementedError

    async def _async_execute(self, task, spatial_info, temporal_info) -> Any:
        system_prompt, user_prompt = self.build_prompt(task, spatial_info, temporal_info)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return await self.llm.agen(messages)


def truncate(text: Any, limit: int = MAX_PEER_OUTPUT_CHARS) -> str:
    text = str(text)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}... (truncated)"


def peer_reports(info: Dict[str, Dict[str, Any]], tense: str) -> str:
    """Render other agents' outputs as a block for the user prompt."""
    lines: List[str] = []
    for agent_id, entry in info.items():
        output = entry.get("output")
        if not output or output == "None.":
            continue
        lines.append(
            f"Agent {agent_id} as a {entry['role']} {tense}:\n\n{truncate(output)}\n"
        )
    return "\n".join(lines)
