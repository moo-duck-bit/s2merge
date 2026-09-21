"""Domain prompt sets.

Importing this package registers every prompt set with :class:`PromptSetRegistry`.
"""

from s2rmerge.mas.prompts.aqua import AQUAPromptSet
from s2rmerge.mas.prompts.base import PromptSet
from s2rmerge.mas.prompts.gsm8k import GSM8KPromptSet
from s2rmerge.mas.prompts.humaneval import HumanEvalPromptSet
from s2rmerge.mas.prompts.mmlu import MMLUPromptSet
from s2rmerge.mas.prompts.registry import PromptSetRegistry

__all__ = [
    "AQUAPromptSet",
    "GSM8KPromptSet",
    "HumanEvalPromptSet",
    "MMLUPromptSet",
    "PromptSet",
    "PromptSetRegistry",
]
