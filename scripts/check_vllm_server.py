"""Check that the local vLLM server is up before starting a long run."""

from __future__ import annotations

import os
import sys

import requests

VLLM_HOST = os.getenv("VLLM_HOST", "http://localhost:6789")
LLAMA_MODEL_ID = os.getenv("LLAMA_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")


def main() -> int:
    try:
        response = requests.get(f"{VLLM_HOST}/health", timeout=5)
    except requests.exceptions.RequestException:
        print(f"no vLLM server at {VLLM_HOST}. Start one with:")
        print(f"  vllm serve {LLAMA_MODEL_ID} --port 6789 --max-model-len 8192")
        return 1

    if response.status_code != 200:
        print(f"vLLM server at {VLLM_HOST} replied {response.status_code}")
        return 1

    print(f"vLLM server is up at {VLLM_HOST}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
