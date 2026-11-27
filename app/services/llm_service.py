import asyncio
import logging
import time
from typing import Optional, Tuple

import anthropic
import groq
import openai

from app.config import (
    ANTHROPIC_API_KEY,
    GROQ_API_KEY,
    HARDCODED_MODEL,
    OPENAI_API_KEY,
    REQUEST_TIMEOUT_SECONDS,
)
from app.exceptions import (
    AuthenticationException,
    LLMTimeoutException,
    ProviderAPIException,
    UnsupportedModelException,
)
from app.schemas import ChatResponse, TokenUsage

logger = logging.getLogger("app.llm")


class LLMService:
    """Unified service router for OpenAI, Anthropic, and Groq."""

    @staticmethod
    def resolve_provider(model: str) -> str:
        """Identify which provider serves the given model."""
        normalized = model.lower().strip()
        if normalized.startswith(("gpt-", "o1", "o3", "chatgpt")):
            return "openai"
        elif normalized.startswith("claude-"):
            return "anthropic"
        elif normalized.startswith(("llama-", "mixtral-", "gemma", "whisper")):
            return "groq"
        else:
            raise UnsupportedModelException(model)

    @classmethod
    async def generate(
        cls,
        message: str,
        model: Optional[str] = None,
        system_prompt: Optional[str] = None,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> ChatResponse:
        """Call the appropriate LLM provider, enforcing timeouts and logging metrics."""
        selected_model = model.strip() if model else HARDCODED_MODEL
        provider = cls.resolve_provider(selected_model)

        logger.info(
            f"[LLM Request Initiated] Provider: {provider.capitalize()} | Model: {selected_model} | MaxTokens: {max_tokens}"
        )
        start_time = time.perf_counter()

        try:
            if provider == "openai":
                response_text, prompt_tokens, completion_tokens = await asyncio.wait_for(
                    cls._call_openai(selected_model, message, system_prompt, max_tokens, temperature),
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
            elif provider == "anthropic":
                response_text, prompt_tokens, completion_tokens = await asyncio.wait_for(
                    cls._call_anthropic(selected_model, message, system_prompt, max_tokens, temperature),
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
            elif provider == "groq":
                response_text, prompt_tokens, completion_tokens = await asyncio.wait_for(
                    cls._call_groq(selected_model, message, system_prompt, max_tokens, temperature),
                    timeout=REQUEST_TIMEOUT_SECONDS,
                )
            else:
                raise UnsupportedModelException(selected_model)

        except (asyncio.TimeoutError, TimeoutError):
            latency = round(time.perf_counter() - start_time, 3)
            logger.error(
                f"[LLM Timeout] Provider: {provider.capitalize()} | Model: {selected_model} | "
                f"Latency: {latency:.3f}s ({latency * 1000:.1f}ms) | Timeout limit: {REQUEST_TIMEOUT_SECONDS}s"
            )
            raise LLMTimeoutException(
                f"Call to {provider} (model: '{selected_model}') exceeded timeout of {REQUEST_TIMEOUT_SECONDS}s."
            )
        except Exception as e:
            latency = round(time.perf_counter() - start_time, 3)
            logger.error(
                f"[LLM Failure] Provider: {provider.capitalize()} | Model: {selected_model} | "
                f"Latency: {latency:.3f}s ({latency * 1000:.1f}ms) | Error: {str(e)}"
            )
            raise

        latency = round(time.perf_counter() - start_time, 3)
        total_tokens = prompt_tokens + completion_tokens

        logger.info(
            f"[LLM Metrics] Provider: {provider.capitalize()} | Model: {selected_model} | "
            f"Latency: {latency:.3f}s ({latency * 1000:.1f}ms) | "
            f"Tokens: [Prompt: {prompt_tokens}, Completion: {completion_tokens}, Total: {total_tokens}]"
        )

        return ChatResponse(
            response=response_text,
            model_used=selected_model,
            provider=provider.capitalize(),
            usage=TokenUsage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            ),
            latency_seconds=latency,
        )

    @staticmethod
    async def _call_openai(
        model: str,
        message: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> Tuple[str, int, int]:
        if not OPENAI_API_KEY:
            raise AuthenticationException(
                "OpenAI API key not configured. Set OPENAI_API_KEY in your .env file."
            )

        client = openai.AsyncOpenAI(api_key=OPENAI_API_KEY, timeout=REQUEST_TIMEOUT_SECONDS)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

        try:
            kwargs = {"model": model, "messages": messages}
            if not model.startswith(("o1", "o3", "gpt-6")):
                kwargs["temperature"] = temperature
            kwargs["max_completion_tokens"] = max_tokens

            response = await client.chat.completions.create(**kwargs)
            content = response.choices[0].message.content or ""
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            return content, prompt_tokens, completion_tokens

        except openai.AuthenticationError as e:
            raise AuthenticationException(f"OpenAI Authentication Failed: {e}")
        except openai.RateLimitError as e:
            raise ProviderAPIException("OpenAI", f"Rate limit reached: {e}", status_code=429)
        except openai.APITimeoutError as e:
            raise LLMTimeoutException(f"OpenAI timeout: {e}")
        except openai.APIError as e:
            raise ProviderAPIException("OpenAI", str(e), status_code=502)

    @staticmethod
    async def _call_anthropic(
        model: str,
        message: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> Tuple[str, int, int]:
        if not ANTHROPIC_API_KEY:
            raise AuthenticationException(
                "Anthropic API key not configured. Set ANTHROPIC_API_KEY in your .env file."
            )

        client = anthropic.AsyncAnthropic(api_key=ANTHROPIC_API_KEY, timeout=REQUEST_TIMEOUT_SECONDS)
        kwargs = {
            "model": model,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": message}],
        }
        if system_prompt:
            kwargs["system"] = system_prompt

        try:
            response = await client.messages.create(**kwargs)
            text_blocks = [b.text for b in response.content if hasattr(b, "text")]
            content = "".join(text_blocks)
            prompt_tokens = response.usage.input_tokens if response.usage else 0
            completion_tokens = response.usage.output_tokens if response.usage else 0
            return content, prompt_tokens, completion_tokens

        except anthropic.AuthenticationError as e:
            raise AuthenticationException(f"Anthropic Authentication Failed: {e}")
        except anthropic.RateLimitError as e:
            raise ProviderAPIException("Anthropic", f"Rate limit reached: {e}", status_code=429)
        except anthropic.APITimeoutError as e:
            raise LLMTimeoutException(f"Anthropic timeout: {e}")
        except anthropic.APIError as e:
            raise ProviderAPIException("Anthropic", str(e), status_code=502)

    @staticmethod
    async def _call_groq(
        model: str,
        message: str,
        system_prompt: Optional[str],
        max_tokens: int,
        temperature: float,
    ) -> Tuple[str, int, int]:
        if not GROQ_API_KEY:
            raise AuthenticationException(
                "Groq API key not configured. Set GROQ_API_KEY in your .env file."
            )

        client = groq.AsyncGroq(api_key=GROQ_API_KEY, timeout=REQUEST_TIMEOUT_SECONDS)
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": message})

        try:
            response = await client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            content = response.choices[0].message.content or ""
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            return content, prompt_tokens, completion_tokens

        except groq.AuthenticationError as e:
            raise AuthenticationException(f"Groq Authentication Failed: {e}")
        except groq.RateLimitError as e:
            raise ProviderAPIException("Groq", f"Rate limit reached: {e}", status_code=429)
        except groq.APITimeoutError as e:
            raise LLMTimeoutException(f"Groq timeout: {e}")
        except groq.APIError as e:
            raise ProviderAPIException("Groq", str(e), status_code=502)
