"""Agent for knowledge-recall multiple choice (MMLU)."""

from __future__ import annotations

from typing import Tuple

from s2rmerge.mas.agents.base import LLMNode, peer_reports
from s2rmerge.mas.agents.registry import AgentRegistry


@AgentRegistry.register("AnalyzeAgent")
class AnalyzeAgent(LLMNode):
    """Answers a multiple-choice question from its own domain expertise."""

    agent_name = "AnalyzeAgent"

    def build_prompt(self, task, spatial_info, temporal_info) -> Tuple[str, str]:
        user_prompt = f"The task is: {task['task']}\n"

        this_round = peer_reports(spatial_info, "answered")
        last_round = peer_reports(temporal_info, "answered last round")
        if this_round:
            user_prompt += f"\nThe outputs of other agents are:\n\n{this_round}\n"
        if last_round:
            user_prompt += f"\nIn the previous round, the outputs were:\n\n{last_round}\n"

        return self.constraint, user_prompt
