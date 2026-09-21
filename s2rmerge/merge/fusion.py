"""Prompt-level fusion: keeping an absorbed agent's role alive in the prompt.

Removing a node removes its perspective too, unless that perspective is written
down somewhere. Before the target is dropped, an LLM folds its role description
into the partner's system prompt: the partner's own role stays primary, and only
the non-redundant part of the target's role is carried across.

If the fusion call fails the two prompts are concatenated instead, which is the
ablation baseline the paper compares against, so a failure degrades quality
rather than the run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from s2rmerge import accounting
from s2rmerge.accounting import LEDGER

if TYPE_CHECKING:  # pragma: no cover - avoids importing the LLM stack eagerly
    from s2rmerge.mas.node import Node

FUSION_SYSTEM_PROMPT = """You are an expert at merging agent instructions while preserving role clarity.
Your task is to combine two agent prompts intelligently:
1. Maintain the PRIMARY agent's core role and responsibilities
2. Extract only the key, non-redundant insights from the SECONDARY agent
3. Integrate secondary insights naturally without compromising the primary role
4. Keep the result concise and actionable
5. Output ONLY the merged prompt text, no explanations."""

FUSION_USER_PROMPT = """Merge these two agent prompts:

PRIMARY AGENT (role: {primary_role}):
{primary_constraint}

SECONDARY AGENT (role: {secondary_role}):
{secondary_constraint}

Create a merged prompt that:
- Keeps the PRIMARY agent's role as the main focus
- Adds valuable context from the SECONDARY agent if relevant
- Removes redundant information
- Maintains clarity and conciseness

Output the merged prompt:"""


def concatenate(primary: "Node", secondary: "Node") -> str:
    """The ablation baseline: append the absorbed prompt unchanged."""
    return (
        f"{primary.constraint}\n\n"
        f"Additional context from the merged {secondary.role} agent: {secondary.constraint}"
    )


async def fuse_prompts(
    primary: "Node", secondary: "Node", llm_name: Optional[str] = None
) -> str:
    """Return the system prompt ``primary`` should carry after absorbing ``secondary``."""
    from s2rmerge.mas.llm.registry import LLMRegistry

    llm = LLMRegistry.get(llm_name or primary.llm_name)
    messages = [
        {"role": "system", "content": FUSION_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": FUSION_USER_PROMPT.format(
                primary_role=primary.role,
                primary_constraint=primary.constraint,
                secondary_role=secondary.role,
                secondary_constraint=secondary.constraint,
            ),
        },
    ]

    with LEDGER.stage(accounting.FUSION):
        try:
            merged = await llm.agen(messages)
        except Exception as error:  # noqa: BLE001 - fall back rather than lose the run
            print(f"prompt-level fusion failed ({error}); concatenating instead")
            return concatenate(primary, secondary)

    return merged.strip() if isinstance(merged, str) and merged.strip() else concatenate(primary, secondary)
