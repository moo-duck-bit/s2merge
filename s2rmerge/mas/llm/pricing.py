"""Token counting and USD pricing for the backends used in the experiments.

Prices are per 1K tokens, as published by OpenAI. Models served locally through
vLLM are priced at zero; only their token counts matter, which is what the paper
reports.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Dict, Tuple

import tiktoken

from s2rmerge.accounting import LEDGER

# model prefix -> (input price, output price) per 1K tokens
OPENAI_PRICING: Dict[str, Tuple[float, float]] = {
    "gpt-4o-mini": (0.00015, 0.0006),
    "gpt-4o": (0.005, 0.015),
    "gpt-4-turbo": (0.01, 0.03),
    "gpt-4-32k": (0.06, 0.12),
    "gpt-4": (0.03, 0.06),
    "gpt-3.5-turbo": (0.0015, 0.002),
}

_FALLBACK_ENCODING = "cl100k_base"


def is_openai_model(model: str) -> bool:
    return model.startswith("gpt-")


def price_per_1k(model: str) -> Tuple[float, float]:
    """Input/output price for ``model``, or ``(0.0, 0.0)`` if it is not billed."""
    for prefix, prices in OPENAI_PRICING.items():
        if model.startswith(prefix):
            return prices
    return (0.0, 0.0)


@lru_cache(maxsize=8)
def _openai_encoding(model: str):
    try:
        return tiktoken.encoding_for_model(model)
    except KeyError:
        return tiktoken.get_encoding(_FALLBACK_ENCODING)


@lru_cache(maxsize=4)
def _hf_tokenizer(model: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model, use_fast=True)


def count_tokens(model: str, text: str) -> int:
    """Number of tokens ``model`` sees in ``text``.

    OpenAI models are counted with tiktoken. Local models are counted with their
    own Hugging Face tokenizer, falling back to tiktoken if the tokenizer cannot
    be loaded (for instance when the weights are not on disk).
    """
    if is_openai_model(model):
        return len(_openai_encoding(model).encode(text))
    try:
        return len(_hf_tokenizer(model)(text)["input_ids"])
    except Exception:
        return len(_openai_encoding(_FALLBACK_ENCODING).encode(text))


def record_usage(prompt: str, response: str, model: str) -> Tuple[int, int, float]:
    """Charge one completion to the ledger and return its usage."""
    prompt_tokens = count_tokens(model, prompt)
    completion_tokens = count_tokens(model, response)

    input_price, output_price = price_per_1k(model)
    cost = (prompt_tokens * input_price + completion_tokens * output_price) / 1000

    LEDGER.record(prompt_tokens, completion_tokens, cost)
    return prompt_tokens, completion_tokens, cost
