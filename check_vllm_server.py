#!/usr/bin/env python3
"""Check if vLLM server is running"""

import os
import sys

import requests

VLLM_HOST = os.getenv("VLLM_HOST", "http://localhost:6789")
LLAMA_MODEL_ID = os.getenv("LLAMA_MODEL_ID", "meta-llama/Llama-3.1-8B-Instruct")

try:
    response = requests.get(f"{VLLM_HOST}/health", timeout=5)
    if response.status_code == 200:
        print("vLLM server is running")
        print(f"Status: {response.json()}")
        sys.exit(0)
    else:
        print(f"vLLM server returned status code: {response.status_code}")
        sys.exit(1)
except requests.exceptions.ConnectionError:
    print("vLLM server is not running")
    print("\nPlease start the server with:")
    print(f"  vllm serve {LLAMA_MODEL_ID} \\")
    print("    --host 0.0.0.0 \\")
    print("    --port 6789 \\")
    print("    --max-model-len 8192")
    sys.exit(1)
except Exception as e:
    print(f"Error checking vLLM server: {e}")
    sys.exit(1)
