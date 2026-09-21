"""Agent for the math benchmarks (GSM8K, MultiArith, SVAMP, AQuA)."""

from __future__ import annotations

from typing import Any, Dict, Tuple

from s2rmerge.answers import extract_choice, extract_numeric
from s2rmerge.mas.agents.base import LLMNode, peer_reports
from s2rmerge.mas.agents.registry import AgentRegistry
from s2rmerge.mas.executor.python_executor import execute_code_get_return

_ANSWER_EXTRACTORS = {"aqua": extract_choice}
"""Domains whose answers are option letters rather than numbers."""


@AgentRegistry.register("MathSolver")
class MathSolver(LLMNode):
    """Solves a math problem, optionally informed by what other agents said.

    The ``Math Solver`` role is given peers' final answers as a numeric hint,
    following the progressive-hint prompting the GSM8K few-shot block assumes.
    Every other role sees the peers' full reasoning instead.
    """

    agent_name = "MathSolver"

    def _extract(self, text: Any) -> str:
        extractor = _ANSWER_EXTRACTORS.get(self.domain, extract_numeric)
        return extractor(text) or ""

    def build_prompt(self, task, spatial_info, temporal_info) -> Tuple[str, str]:
        user_prompt = self.prompt_set.get_answer_prompt(question=task["task"], role=self.role)

        if self.role == "Math Solver":
            hints = [
                self._extract(entry["output"])
                for entry in list(spatial_info.values()) + list(temporal_info.values())
            ]
            hints = [hint for hint in hints if hint]
            if hints:
                user_prompt += f"(Hint: The answer is near to {' '.join(hints)})."
            return self.constraint, user_prompt

        this_round = peer_reports(spatial_info, "answered")
        last_round = peer_reports(temporal_info, "answered last round")
        if this_round:
            user_prompt += (
                "\n\nOther agents answered the same question as follows:\n\n"
                f"{this_round}\n"
            )
        if last_round:
            user_prompt += (
                "\n\nIn the previous round, the answers were:\n\n"
                f"{last_round}\n"
            )
        return self.constraint, user_prompt

    async def _async_execute(self, task: Dict[str, str], spatial_info, temporal_info) -> Any:
        response = await super()._async_execute(task, spatial_info, temporal_info)
        if self.role != "Programming Expert" or not isinstance(response, str):
            return response

        # This role answers with code; run it so downstream agents see a value.
        code = response.removeprefix("```python").strip().removesuffix("```").strip()
        try:
            value = execute_code_get_return(code)
        except Exception as error:  # noqa: BLE001 - agent-authored code may do anything
            return f"{response}\nError executing code: {error}"
        return f"{response}\nthe answer is {value}"
