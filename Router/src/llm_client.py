"""
LLM Client for Router
Supports OpenAI API (GPT models)
"""

import os
import json
import logging
from typing import Dict, Any, Optional

# Import AgentDropout's global cost tracking
try:
    from AgentDropout.utils.globals import (
        Cost, PromptTokens, CompletionTokens,
        SunkPromptTokens, SunkCompletionTokens, SunkCost
    )
    from AgentDropout.llm.price import get_model_price
    COST_TRACKING_AVAILABLE = True
except ImportError:
    COST_TRACKING_AVAILABLE = False
    logging.warning("AgentDropout cost tracking not available")

logger = logging.getLogger(__name__)


class OpenAIClient:
    """
    OpenAI API client for structured summary generation
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        temperature: float = 0.0,
        max_tokens: int = 500
    ):
        """
        Args:
            api_key: OpenAI API key
            model: Model name (default: gpt-4o-mini)
            temperature: Sampling temperature (0 = deterministic)
            max_tokens: Maximum tokens in response
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "OpenAI library not installed. "
                "Install it with: pip install openai"
            )

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        logger.info(f"OpenAI client initialized (model={model})")

    def generate(self, prompt: str) -> str:
        """
        Generate structured summary using OpenAI API

        Args:
            prompt: Input prompt

        Returns:
            JSON string with structured summary
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "system",
                        "content": "You are a helpful assistant that analyzes questions and provides structured summaries in JSON format. Provide ONLY the JSON output, no additional text."
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens
            )

            # Try to update global cost tracking at runtime (import dynamically)
            try:
                from AgentDropout.utils.globals import (
                    Cost, PromptTokens, CompletionTokens,
                    SunkPromptTokens, SunkCompletionTokens, SunkCost
                )
                from AgentDropout.llm.price import get_model_price

                # Attempt to extract usage information in multiple possible shapes
                usage = None
                if hasattr(response, 'usage'):
                    usage = response.usage
                elif isinstance(response, dict) and 'usage' in response:
                    usage = response['usage']

                if usage is not None:
                    # prompt/completion tokens may be attributes or dict keys
                    try:
                        prompt_tokens = int(getattr(usage, 'prompt_tokens', None) or usage.get('prompt_tokens'))
                    except Exception:
                        prompt_tokens = 0
                    try:
                        completion_tokens = int(getattr(usage, 'completion_tokens', None) or usage.get('completion_tokens'))
                    except Exception:
                        completion_tokens = 0
                    logger.info(f"Router LLM (T_sunk): prompt={prompt_tokens}, completion={completion_tokens}")

                    # Update token counters (both total and sunk)
                    try:
                        PromptTokens.instance().add(prompt_tokens)
                        CompletionTokens.instance().add(completion_tokens)
                        # Router tokens are sunk cost (T_sunk)
                        SunkPromptTokens.instance().add(prompt_tokens)
                        SunkCompletionTokens.instance().add(completion_tokens)
                    except Exception:
                        logger.debug("Failed to update token counters")

                    # Calculate and update cost
                    try:
                        cost = get_model_price(self.model, prompt_tokens, completion_tokens)
                        try:
                            Cost.instance().add(cost)
                            SunkCost.instance().add(cost)
                        except Exception:
                            logger.debug("Failed to add cost to Cost singleton")
                        logger.debug(f"Router API call: {prompt_tokens} prompt + {completion_tokens} completion tokens, ${cost:.6f}")
                    except Exception as e:
                        logger.warning(f"Could not calculate cost for model {self.model}: {e}")
            except Exception as e:
                logger.debug(f"Cost tracking unavailable at runtime: {e}")

            # Extract response
            content = response.choices[0].message.content

            # Try to count cost/tokens using AgentDropout.llm.price.cost_count if available
            try:
                from AgentDropout.llm.price import cost_count
                try:
                    price, p_len, c_len = cost_count(prompt, content, self.model)
                    logger.debug(f"Router price_count: ${price:.6f} (p={p_len}, c={c_len})")
                except Exception as e:
                    logger.debug(f"cost_count failed: {e}")
            except Exception:
                logger.debug("cost_count not available; skipping cost accumulation via cost_count")

            # Validate JSON
            try:
                json.loads(content)  # Just to validate
            except json.JSONDecodeError:
                logger.warning("Response is not valid JSON, wrapping in default structure")
                content = json.dumps({
                    "task_format": "unknown",
                    "domain_hint": "general",
                    "key_concepts": [],
                    "routing_summary": content
                })

            return content

        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            # Return default structure on error
            return json.dumps({
                "task_format": "unknown",
                "domain_hint": "general",
                "key_concepts": [],
                "routing_summary": "Error generating summary"
            })


class LlamaClient:
    """
    Llama local vLLM client for Router
    """

    def __init__(self, model: str = "Meta-Llama-3.1-8B-Instruct"):
        self.model = model
        logger.info(f"Llama client initialized (model={model})")

    def generate(self, prompt: str) -> str:
        """
        Generate structured summary using local Llama model
        """
        import asyncio
        from AgentDropout.llm.gpt_chat import achat_llama

        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that analyzes questions and provides structured summaries in JSON format. Provide ONLY the JSON output, no additional text."
            },
            {
                "role": "user",
                "content": prompt
            }
        ]

        # Handle event loop properly
        try:
            # Try to get existing event loop
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # If loop is running, use nest_asyncio or create task
                import nest_asyncio
                nest_asyncio.apply()
                content = loop.run_until_complete(achat_llama(self.model, messages))
            else:
                # Loop exists but not running
                content = loop.run_until_complete(achat_llama(self.model, messages))
        except RuntimeError:
            # No event loop exists, create new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                content = loop.run_until_complete(achat_llama(self.model, messages))
            finally:
                loop.close()

        # Validate JSON
        try:
            json.loads(content)
        except:
            logger.warning("Response is not valid JSON, wrapping")
            content = json.dumps({
                "task_format": "code generation",
                "domain_hint": "algorithm",
                "key_concepts": ["programming"],
                "routing_summary": content[:100]
            })

        return content


class MockLLMClient:
    """
    Mock LLM client for testing (no API calls)
    """

    def __init__(self):
        logger.warning("Using MockLLMClient - no real LLM calls will be made")

    def generate(self, prompt: str) -> str:
        """
        Return mock structured summary

        Args:
            prompt: Input prompt (ignored)

        Returns:
            JSON string with mock summary
        """
        return json.dumps({
            "task_format": "multiple choice question",
            "domain_hint": "general knowledge",
            "key_concepts": ["reasoning", "analysis"],
            "routing_summary": "requires analytical reasoning and domain knowledge"
        })


def create_llm_client(config: Dict[str, Any], api_keys: Optional[Dict[str, Any]] = None) -> Any:
    """
    Factory function to create LLM client

    Args:
        config: Router configuration
        api_keys: API keys configuration (optional)

    Returns:
        LLM client instance (Llama, OpenAI, or Mock)
    """
    # Check if using Llama model (local vLLM)
    llm_model = config.get('stage0', {}).get('llm_model', 'gpt-4o-mini')

    if 'llama' in llm_model.lower() or 'meta-llama' in llm_model.lower():
        logger.info(f"Using Llama local vLLM client: {llm_model}")
        return LlamaClient(model=llm_model)

    # Try to load API keys from config file
    if api_keys is None:
        try:
            import yaml
            api_keys_path = "config/api_keys.yaml"
            if os.path.exists(api_keys_path):
                with open(api_keys_path, 'r') as f:
                    api_keys = yaml.safe_load(f)
                logger.info("API keys loaded from config/api_keys.yaml")
        except Exception as e:
            logger.debug(f"Could not load API keys from file: {e}")

    # Fallback: try environment variable OPENAI_API_KEY
    if not api_keys:
        env_key = os.environ.get('OPENAI_API_KEY')
        if env_key:
            logger.info("OPENAI_API_KEY found in environment; using it for OpenAI client")
            api_keys = {
                'openai': {
                    'api_key': env_key,
                    'model': os.environ.get('OPENAI_MODEL', 'gpt-4o-mini'),
                    'temperature': float(os.environ.get('OPENAI_TEMPERATURE', '0.0')),
                    'max_tokens': int(os.environ.get('OPENAI_MAX_TOKENS', '500'))
                }
            }

    # Try to create OpenAI client
    if api_keys and 'openai' in api_keys:
        try:
            openai_config = api_keys['openai']
            client = OpenAIClient(
                api_key=openai_config['api_key'],
                model=openai_config.get('model', 'gpt-4o-mini'),
                temperature=openai_config.get('temperature', 0.0),
                max_tokens=openai_config.get('max_tokens', 500)
            )
            logger.info("Using real OpenAI API client")
            return client
        except Exception as e:
            logger.error(f"Failed to create OpenAI client: {e}")

    # Fallback to mock client
    logger.warning("Using MockLLMClient - install openai and configure API keys for real LLM")
    return MockLLMClient()
