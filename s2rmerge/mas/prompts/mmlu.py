"""Role catalogue and prompts for MMLU four-option multiple choice.

MMLU spans 57 subjects, so the pool is organised as fifteen domain specialists —
three for each of the five domain blocks the router scores in Stage-1. Every
role carries its own description, so a routed agent answers from a genuinely
distinct perspective rather than from a shared generic instruction.
"""

from __future__ import annotations

import itertools
from typing import List

from s2rmerge.mas.prompts.base import PromptSet
from s2rmerge.mas.prompts.registry import PromptSetRegistry

ROLES: List[str] = [
    # MathLogic
    "Mathematician",
    "Statistician",
    "Formal Logician",
    # CS_Eng_Physics
    "Computer Scientist",
    "Engineer",
    "Physicist",
    # Bio_Med
    "Biologist",
    "Clinician",
    "Chemist",
    # Econ_Law_Social
    "Economist",
    "Legal Scholar",
    "Political Scientist",
    # Humanities
    "Historian",
    "Philosopher",
    "Psychologist",
]

_role_cycle = itertools.cycle(ROLES)

_ANSWER_FORMAT = (
    "Your reply must be under 100 words and contain both your answer and a brief "
    "step-by-step analysis. The first line of your reply must contain only one "
    "letter: A, B, C or D."
)

ROLE_DESCRIPTION = {
    "Mathematician": (
        "You are a mathematician. Identify the mathematical structure behind the question "
        "and reason from definitions, theorems and exact calculation rather than intuition. "
        "State the rule you are applying before you apply it. "
    ),
    "Statistician": (
        "You are a statistician. Reason about distributions, sampling, uncertainty and "
        "inference. Watch for base-rate errors, confounding and misread conditional "
        "probabilities. "
    ),
    "Formal Logician": (
        "You are a formal logician. Restate the question as premises and a conclusion, then "
        "check which option follows validly. Name any fallacy an incorrect option relies on. "
    ),
    "Computer Scientist": (
        "You are a computer scientist. Reason about algorithms, data structures, "
        "computability and complexity. Trace execution or check invariants when the question "
        "concerns program behaviour. "
    ),
    "Engineer": (
        "You are an engineer. Reason from physical constraints, tolerances and standard "
        "design practice. Check that quantities are dimensionally consistent and of a "
        "plausible magnitude. "
    ),
    "Physicist": (
        "You are a physicist. Identify the governing law or conservation principle, check "
        "limiting cases, and verify units before committing to an option. "
    ),
    "Biologist": (
        "You are a biologist. Reason across molecular, cellular and organism scales, and "
        "distinguish mechanism from correlation when evaluating each option. "
    ),
    "Clinician": (
        "You are a clinician. Work from presentation to differential diagnosis, weigh the "
        "options by how well each explains the findings, and note contraindications. "
    ),
    "Chemist": (
        "You are a chemist. Reason about structure, reactivity, stoichiometry and "
        "thermodynamics, and balance any equation the question depends on. "
    ),
    "Economist": (
        "You are an economist. Reason about incentives, marginal effects, equilibrium and "
        "trade-offs, and separate positive claims from normative ones. "
    ),
    "Legal Scholar": (
        "You are a legal scholar. Identify the governing rule or doctrine, apply it to the "
        "facts element by element, and note where jurisdictions differ. "
    ),
    "Political Scientist": (
        "You are a political scientist. Reason about institutions, incentives of actors and "
        "the empirical evidence for each option, avoiding partisan framing. "
    ),
    "Historian": (
        "You are a historian. Place the question in its period, check chronology and "
        "causation, and reject options that are anachronistic. "
    ),
    "Philosopher": (
        "You are a philosopher. Clarify the concepts at stake, reconstruct the strongest "
        "argument for each option, and identify the assumption that decides between them. "
    ),
    "Psychologist": (
        "You are a psychologist. Reason from established findings in cognition, development "
        "and behaviour, and distinguish a theory's claims from popular misreadings of it. "
    ),
}

_GENERIC_CONSTRAINT = (
    "You will be given a question and four answers enumerated as A, B, C and D. "
    "Exactly one is correct. Use the reasoning of other agents as advice, with critical "
    "thinking, and give your own answer. Do not imitate another agent's analysis. "
)


@PromptSetRegistry.register("mmlu")
class MMLUPromptSet(PromptSet):
    """Prompts for MMLU."""

    @staticmethod
    def roles():
        return list(ROLES)

    @staticmethod
    def get_role():
        return next(_role_cycle)

    @staticmethod
    def get_constraint(role):
        description = ROLE_DESCRIPTION.get(role, _GENERIC_CONSTRAINT)
        return f"{description}{_ANSWER_FORMAT}"

    @staticmethod
    def get_answer_prompt(question, role=None):
        return question

    @staticmethod
    def get_decision_role():
        return (
            "You are the top decision-maker. You are good at analysing and summarising "
            "other people's opinions, finding errors and giving final answers."
        )

    @staticmethod
    def get_decision_constraint():
        return (
            "You will be given a question and four answers enumerated as A, B, C and D. "
            "Exactly one is correct. You will also be given other people's answers and "
            "analyses. Your reply must contain only one letter and no other characters. "
            "For example, your reply can be A."
        )

    @staticmethod
    def postprocess_answer(answer):
        if isinstance(answer, list):
            answer = answer[0] if answer else ""
        if not isinstance(answer, str):
            raise TypeError(f"expected a string answer, got {type(answer).__name__}")
        # Agents are told to open with the option letter.
        return answer[0] if answer else answer
