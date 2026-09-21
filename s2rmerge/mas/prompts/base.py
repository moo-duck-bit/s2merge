"""Prompt-set interface.

A prompt set holds everything domain-specific about how agents are addressed:
the role catalogue, each role's system prompt, the user prompt wrapper, and how
the final-decision agent is instructed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Union


class PromptSet(ABC):
    """Role descriptions and prompt templates for one benchmark domain."""

    @staticmethod
    @abstractmethod
    def roles() -> List[str]:
        """Every role this domain defines, in catalogue order."""

    @staticmethod
    @abstractmethod
    def get_role() -> str:
        """Next role from the catalogue, cycling."""

    @staticmethod
    @abstractmethod
    def get_constraint(role: str) -> str:
        """System prompt for ``role``."""

    @staticmethod
    @abstractmethod
    def get_answer_prompt(question: str, role: str) -> str:
        """User prompt wrapping ``question`` for ``role``."""

    @staticmethod
    @abstractmethod
    def get_decision_role() -> str:
        """System prompt persona for the final-decision agent."""

    @staticmethod
    @abstractmethod
    def get_decision_constraint() -> str:
        """Instructions for the final-decision agent."""

    @staticmethod
    def get_decision_few_shot() -> str:
        """Few-shot block prepended to the final-decision prompt."""
        return ""

    @staticmethod
    def postprocess_answer(answer: Union[str, List[str]]) -> str:
        """Normalise one agent output before it is shown to the decision agent."""
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        return answer if isinstance(answer, str) else str(answer)
