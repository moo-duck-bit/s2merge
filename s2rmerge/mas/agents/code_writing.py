"""Agent for code synthesis (HumanEval)."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

from s2rmerge.mas.agents.base import LLMNode, truncate
from s2rmerge.mas.agents.registry import AgentRegistry
from s2rmerge.mas.executor.python_executor import PyExecutor

_INTERNAL_TEST_TIMEOUT = 10


def doctest_assertions(prompt: str) -> List[str]:
    """Turn the ``>>>`` examples in a docstring into assert statements.

    These are the only tests available before submission, so agents use them to
    check each other's code; the graded HumanEval tests are never shown.
    """
    lines = (line.strip() for line in prompt.split("\n") if line.strip())
    assertions = []
    iterator = iter(lines)
    for line in iterator:
        if not line.startswith(">>>"):
            continue
        call = line[4:]
        expected = next(iterator, None)
        if expected:
            assertions.append(f"assert {call} == {expected}")
    return assertions


def _is_code_block(text: Any) -> bool:
    return isinstance(text, str) and text.startswith("```python") and text.endswith("```")


def _strip_fence(text: str) -> str:
    return text.removeprefix("```python").strip().removesuffix("```").strip()


@AgentRegistry.register("CodeWriting")
class CodeWriting(LLMNode):
    """Writes an implementation, informed by peers' code and its test results."""

    agent_name = "CodeWriting"

    def __init__(self, id=None, role=None, domain="", llm_name=""):
        super().__init__(id, role, domain, llm_name)
        self.internal_tests: List[str] = []
        self._accepted_peer_solution: Any = None

    def _describe_peers(self, info: Dict[str, Dict[str, Any]]) -> str:
        blocks = []
        for agent_id, entry in info.items():
            output = entry.get("output")
            if not _is_code_block(output):
                blocks.append(
                    f"Agent {agent_id} as a {entry['role']} provides: {truncate(output)}\n"
                )
                continue

            passed, feedback, _ = PyExecutor().execute(
                _strip_fence(output), self.internal_tests, timeout=_INTERNAL_TEST_TIMEOUT
            )
            if passed and self.internal_tests:
                # A peer already passes every visible test; adopt it verbatim.
                self._accepted_peer_solution = output
                return ""
            blocks.append(
                f"Agent {agent_id} as a {entry['role']} wrote:\n\n{output}\n\n"
                f"Passes the visible tests: {passed}.\nFeedback:\n\n{feedback}\n"
            )
        return "\n".join(blocks)

    def build_prompt(self, task, spatial_info, temporal_info) -> Tuple[str, str]:
        this_round = self._describe_peers(spatial_info)
        last_round = "" if self._accepted_peer_solution else self._describe_peers(temporal_info)

        user_prompt = f"The task is:\n\n{task['task']}\n"
        if this_round:
            user_prompt += f"\nThe outputs and feedback of other agents are:\n\n{this_round}\n"
        if last_round:
            user_prompt += f"\nIn the previous round they were:\n\n{last_round}\n"
        return self.constraint, user_prompt

    async def _async_execute(self, task, spatial_info, temporal_info) -> Any:
        self.internal_tests = doctest_assertions(task["task"])
        self._accepted_peer_solution = None

        system_prompt, user_prompt = self.build_prompt(task, spatial_info, temporal_info)
        if self._accepted_peer_solution is not None:
            return self._accepted_peer_solution

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return await self.llm.agen(messages)
