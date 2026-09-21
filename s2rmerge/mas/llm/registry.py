"""Backend lookup by model name."""

from __future__ import annotations

from typing import Optional, Type

from s2rmerge.mas.llm.base import LLM

DEFAULT_MODEL = "gpt-4o"

_LOCAL_MODEL_MARKERS = ("llama", "qwen")


class LLMRegistry:
    """Maps a model name onto the backend that can serve it."""

    _backends: dict = {}

    @classmethod
    def register(cls, name: str):
        def decorator(backend: Type[LLM]) -> Type[LLM]:
            cls._backends[name] = backend
            return backend

        return decorator

    @classmethod
    def keys(cls):
        return cls._backends.keys()

    @classmethod
    def get(cls, model_name: Optional[str] = None) -> LLM:
        model_name = model_name or DEFAULT_MODEL
        lowered = model_name.lower()

        if any(marker in lowered for marker in _LOCAL_MODEL_MARKERS):
            backend = "llama"
        else:
            backend = "GPTChat"

        return cls._backends[backend](model_name)
