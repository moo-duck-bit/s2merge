"""Prompt-set lookup by benchmark domain."""

from __future__ import annotations

from typing import Dict, Type

from s2rmerge.mas.prompts.base import PromptSet


class PromptSetRegistry:
    """Maps a benchmark domain onto its prompt set."""

    _prompt_sets: Dict[str, Type[PromptSet]] = {}

    @classmethod
    def register(cls, domain: str):
        def decorator(prompt_set: Type[PromptSet]) -> Type[PromptSet]:
            if domain in cls._prompt_sets:
                raise ValueError(f"domain {domain!r} is already registered")
            cls._prompt_sets[domain] = prompt_set
            return prompt_set

        return decorator

    @classmethod
    def keys(cls):
        return cls._prompt_sets.keys()

    @classmethod
    def get(cls, domain: str) -> Type[PromptSet]:
        try:
            return cls._prompt_sets[domain]
        except KeyError:
            raise KeyError(
                f"no prompt set registered for domain {domain!r}; "
                f"known domains: {sorted(cls._prompt_sets)}"
            ) from None
