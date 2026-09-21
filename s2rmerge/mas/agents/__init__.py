"""Agent implementations.

Importing this package registers every agent with :class:`AgentRegistry`.
"""

from s2rmerge.mas.agents.analyze_agent import AnalyzeAgent
from s2rmerge.mas.agents.code_writing import CodeWriting
from s2rmerge.mas.agents.final_decision import FinalMajorVote, FinalRefer, FinalWriteCode
from s2rmerge.mas.agents.math_solver import MathSolver
from s2rmerge.mas.agents.registry import AgentRegistry

__all__ = [
    "AgentRegistry",
    "AnalyzeAgent",
    "CodeWriting",
    "FinalMajorVote",
    "FinalRefer",
    "FinalWriteCode",
    "MathSolver",
]
