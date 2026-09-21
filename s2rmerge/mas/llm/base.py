"""Chat model interface shared by every backend."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List

ChatMessage = Dict[str, str]
"""One ``{"role": ..., "content": ...}`` entry of a chat prompt."""


class LLM(ABC):
    """A chat model the communication graph can call."""

    def __init__(self, model_name: str):
        self.model_name = model_name

    @abstractmethod
    async def agen(self, messages: List[ChatMessage]) -> str:
        """Return the assistant reply to ``messages``."""
