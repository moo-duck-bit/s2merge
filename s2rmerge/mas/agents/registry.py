"""Agent lookup by name."""

from __future__ import annotations

from typing import Dict, Type

from s2rmerge.mas.node import Node


class AgentRegistry:
    """Maps an agent name onto the node class that implements it."""

    _agents: Dict[str, Type[Node]] = {}

    @classmethod
    def register(cls, name: str):
        def decorator(agent: Type[Node]) -> Type[Node]:
            if name in cls._agents:
                raise ValueError(f"agent {name!r} is already registered")
            cls._agents[name] = agent
            return agent

        return decorator

    @classmethod
    def keys(cls):
        return cls._agents.keys()

    @classmethod
    def get(cls, name: str, **kwargs) -> Node:
        try:
            agent = cls._agents[name]
        except KeyError:
            raise KeyError(
                f"unknown agent {name!r}; known agents: {sorted(cls._agents)}"
            ) from None
        return agent(**kwargs)
