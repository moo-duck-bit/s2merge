"""Decision agents: they read the round's answers and emit the final one."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, Optional, Tuple

from s2rmerge.mas.agents.base import truncate
from s2rmerge.mas.agents.code_writing import _is_code_block, _strip_fence, doctest_assertions
from s2rmerge.mas.agents.registry import AgentRegistry
from s2rmerge.mas.executor.python_executor import PyExecutor
from s2rmerge.mas.llm.registry import LLMRegistry
from s2rmerge.mas.node import Node
from s2rmerge.mas.prompts.registry import PromptSetRegistry

_INTERNAL_TEST_TIMEOUT = 10


class _DecisionNode(Node):
    """Base for decision agents; they have no role of their own to route."""

    agent_name = "Decision"

    def __init__(self, id: Optional[str] = None, domain: str = "", llm_name: str = ""):
        super().__init__(id, self.agent_name, domain, llm_name)
        self.prompt_set = PromptSetRegistry.get(domain)
        self.role = self.prompt_set.get_decision_role()


class _LLMDecisionNode(_DecisionNode):
    """Decision agent that asks the model to pick or synthesise an answer."""

    def __init__(self, id: Optional[str] = None, domain: str = "", llm_name: str = ""):
        super().__init__(id, domain, llm_name)
        self.llm = LLMRegistry.get(llm_name)
        self.constraint = self.prompt_set.get_decision_constraint()

    def build_prompt(self, task, spatial_info) -> Tuple[str, str]:
        raise NotImplementedError

    async def _async_execute(self, task, spatial_info, temporal_info) -> Any:
        system_prompt, user_prompt = self.build_prompt(task, spatial_info)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        return await self.llm.agen(messages)


@AgentRegistry.register("FinalRefer")
class FinalRefer(_LLMDecisionNode):
    """Picks the most reliable answer and restates it."""

    agent_name = "FinalRefer"

    def build_prompt(self, task, spatial_info) -> Tuple[str, str]:
        system_prompt = f"{self.role}.\n {self.constraint}"
        answers = "\n\n".join(
            f"{agent_id}: {entry['output']}" for agent_id, entry in spatial_info.items()
        )
        few_shot = self.prompt_set.get_decision_few_shot()
        user_prompt = (
            f"{few_shot} The task is:\n\n {task['task']}.\n"
            f"The outputs of the other agents are:\n\n{answers}"
        )
        return system_prompt, user_prompt


@AgentRegistry.register("FinalWriteCode")
class FinalWriteCode(_LLMDecisionNode):
    """Writes the submitted implementation, told which peers passed the tests."""

    agent_name = "FinalWriteCode"

    def build_prompt(self, task, spatial_info) -> Tuple[str, str]:
        system_prompt = f"{self.role}.\n {self.constraint}"
        internal_tests = doctest_assertions(task["task"])

        blocks = []
        for agent_id, entry in spatial_info.items():
            output = entry.get("output")
            if _is_code_block(output):
                passed, feedback, _ = PyExecutor().execute(
                    _strip_fence(output), internal_tests, timeout=_INTERNAL_TEST_TIMEOUT
                )
                blocks.append(
                    f"Agent {agent_id} as a {entry['role']} wrote:\n\n{output}\n\n"
                    f"Passes the visible tests: {passed}.\nFeedback:\n\n{feedback}\n"
                )
            else:
                blocks.append(
                    f"Agent {agent_id} as a {entry['role']} provides: {truncate(output)}\n"
                )

        user_prompt = (
            f"The task is:\n\n{task['task']}.\n"
            f"The outputs and feedback of the other agents are:\n\n" + "\n".join(blocks)
        )
        return system_prompt, user_prompt


@AgentRegistry.register("FinalMajorVote")
class FinalMajorVote(_DecisionNode):
    """Returns the most common answer; makes no model call."""

    agent_name = "FinalMajorVote"

    async def _async_execute(self, task, spatial_info: Dict[str, Dict[str, Any]], temporal_info) -> Any:
        votes = Counter(
            self.prompt_set.postprocess_answer(entry["output"]) for entry in spatial_info.values()
        )
        if not votes:
            return ""
        return votes.most_common(1)[0][0]
