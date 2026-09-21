"""A single agent in the communication graph."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import shortuuid


class Node(ABC):
    """One agent: a role, a prompt, and the messages it exchanges.

    Predecessors are the agents whose output this one reads. *Spatial* links are
    within a round, *temporal* links reach back to the previous round, which is
    why the previous round's output is kept in :attr:`last_memory`.
    """

    def __init__(
        self,
        id: Optional[str] = None,
        agent_name: str = "",
        domain: str = "",
        llm_name: str = "",
    ):
        self.id = id or shortuuid.ShortUUID().random(length=4)
        self.agent_name = agent_name
        self.domain = domain
        self.llm_name = llm_name
        self.role = ""

        self.spatial_predecessors: List["Node"] = []
        self.spatial_successors: List["Node"] = []
        self.temporal_predecessors: List["Node"] = []
        self.temporal_successors: List["Node"] = []

        self.inputs: List[Any] = []
        self.outputs: List[Any] = []
        self.raw_inputs: List[Any] = []
        self.last_memory: Dict[str, List[Any]] = {"inputs": [], "outputs": [], "raw_inputs": []}

    def add_predecessor(self, other: "Node", edge: str = "spatial") -> None:
        predecessors = getattr(self, f"{edge}_predecessors")
        if other not in predecessors:
            predecessors.append(other)
            getattr(other, f"{edge}_successors").append(self)

    def add_successor(self, other: "Node", edge: str = "spatial") -> None:
        successors = getattr(self, f"{edge}_successors")
        if other not in successors:
            successors.append(other)
            getattr(other, f"{edge}_predecessors").append(self)

    def clear_connections(self) -> None:
        self.spatial_predecessors = []
        self.spatial_successors = []
        self.temporal_predecessors = []
        self.temporal_successors = []

    def update_memory(self) -> None:
        self.last_memory = {
            "inputs": self.inputs,
            "outputs": self.outputs,
            "raw_inputs": self.raw_inputs,
        }

    @staticmethod
    def _latest(outputs: Any) -> Optional[Any]:
        if isinstance(outputs, list):
            return outputs[-1] if outputs else None
        return outputs

    def spatial_info(self) -> Dict[str, Dict[str, Any]]:
        """This round's output of each agent feeding into this one."""
        info = {}
        for predecessor in self.spatial_predecessors:
            output = self._latest(predecessor.outputs)
            if output is None:
                continue
            info[predecessor.id] = {"role": predecessor.role, "output": output}
        return info

    def temporal_info(self) -> Dict[str, Dict[str, Any]]:
        """Previous round's output of each agent feeding into this one."""
        info = {}
        for predecessor in self.temporal_predecessors:
            output = self._latest(predecessor.last_memory["outputs"])
            if output is None:
                continue
            info[predecessor.id] = {"role": predecessor.role, "output": output}
        return info

    async def async_execute(self, task: Dict[str, str]) -> List[Any]:
        result = await self._async_execute(task, self.spatial_info(), self.temporal_info())
        self.outputs = result if isinstance(result, list) else [result]
        return self.outputs

    @abstractmethod
    async def _async_execute(
        self,
        task: Dict[str, str],
        spatial_info: Dict[str, Dict[str, Any]],
        temporal_info: Dict[str, Dict[str, Any]],
    ) -> Any:
        """Produce this agent's answer to ``task``."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.id} role={self.role!r}>"
