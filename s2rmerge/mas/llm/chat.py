"""Chat backends: the OpenAI API and a local vLLM server.

Both are reached through the OpenAI client. A vLLM server identifies a model by
the path it was launched with, so ``LLAMA_MODEL_ID`` has to match the ``--model``
argument given to ``vllm serve``.
"""

from __future__ import annotations

import os
from typing import List

import async_timeout
from dotenv import load_dotenv
from openai import AsyncOpenAI
from tenacity import retry, stop_after_attempt, wait_fixed, wait_random_exponential

from s2rmerge.mas.llm.base import LLM, ChatMessage
from s2rmerge.mas.llm.pricing import record_usage
from s2rmerge.mas.llm.registry import LLMRegistry

load_dotenv()

OPENAI_BASE_URL = os.getenv("BASE_URL") or None
OPENAI_API_KEY = os.getenv("API_KEY", "")

VLLM_BASE_URL = os.getenv("VLLM_BASE_URL", "http://localhost:6789/v1")
VLLM_API_KEY = os.getenv("VLLM_API_KEY", "EMPTY")
LLAMA_MODEL_ID = os.getenv("LLAMA_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")

OPENAI_TIMEOUT = 1000
VLLM_TIMEOUT = 3000


def _joined_prompt(messages: List[ChatMessage]) -> str:
    return "".join(message["content"] for message in messages)


async def _complete(client: AsyncOpenAI, model: str, messages: List[ChatMessage], timeout: int) -> str:
    async with async_timeout.timeout(timeout):
        completion = await client.chat.completions.create(model=model, messages=messages)

    response = completion.choices[0].message.content
    if not isinstance(response, str):
        raise RuntimeError(f"{model} returned a non-text completion")

    record_usage(_joined_prompt(messages), response, model)
    return response


@retry(wait=wait_random_exponential(max=100), stop=stop_after_attempt(3))
async def achat_openai(model: str, messages: List[ChatMessage]) -> str:
    client = AsyncOpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)
    return await _complete(client, model, messages, OPENAI_TIMEOUT)


@retry(wait=wait_fixed(2), stop=stop_after_attempt(5))
async def achat_vllm(messages: List[ChatMessage]) -> str:
    # The served model id is fixed by how vLLM was launched; the name carried in
    # the config only selects this backend.
    client = AsyncOpenAI(api_key=VLLM_API_KEY, base_url=VLLM_BASE_URL)
    return await _complete(client, LLAMA_MODEL_ID, messages, VLLM_TIMEOUT)


@LLMRegistry.register("GPTChat")
class GPTChat(LLM):
    """Hosted OpenAI model."""

    async def agen(self, messages: List[ChatMessage]) -> str:
        return await achat_openai(self.model_name, messages)


@LLMRegistry.register("llama")
class LlamaChat(LLM):
    """Open-weight model served locally by vLLM."""

    async def agen(self, messages: List[ChatMessage]) -> str:
        return await achat_vllm(messages)
