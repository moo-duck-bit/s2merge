"""Communication-graph multi-agent system.

Adapted from AgentDropout (ACL 2025) and, before it, AgentPrune. Importing this
package registers the agents and prompt sets that :class:`Graph` looks up by
name.
"""

from s2rmerge.mas import agents, prompts  # noqa: F401  (registration side effect)
from s2rmerge.mas.graph import Graph
from s2rmerge.mas.node import Node

__all__ = ["Graph", "Node"]
