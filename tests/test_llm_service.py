import pytest
from unittest.mock import patch, AsyncMock
from app.services.llm_service import LLMService
from app.exceptions import (
    UnsupportedModelException,
    LLMTimeoutException,
)


def test_resolve_provider_openai():
    assert LLMService.resolve_provider("gpt-4o") == "openai"
    assert LLMService.resolve_provider("gpt-4o-mini") == "openai"
    assert LLMService.resolve_provider("gpt-6-luna") == "openai"
    assert LLMService.resolve_provider("o1") == "openai"
    assert LLMService.resolve_provider("o3-mini") == "openai"


def test_resolve_provider_anthropic():
    assert LLMService.resolve_provider("claude-3-5-sonnet-20241022") == "anthropic"
    assert LLMService.resolve_provider("claude-3-5-haiku-20241022") == "anthropic"
    assert LLMService.resolve_provider("claude-3-opus-20240229") == "anthropic"


def test_resolve_provider_groq():
    assert LLMService.resolve_provider("llama-3.3-70b-versatile") == "groq"
    assert LLMService.resolve_provider("llama-3.1-8b-instant") == "groq"
    assert LLMService.resolve_provider("mixtral-8x7b-32768") == "groq"
    assert LLMService.resolve_provider("gemma2-9b-it") == "groq"


def test_resolve_provider_unsupported():
    with pytest.raises(UnsupportedModelException):
        LLMService.resolve_provider("bard-ultra-v1")

    with pytest.raises(UnsupportedModelException):
        LLMService.resolve_provider("custom-random-model")


@pytest.mark.asyncio
async def test_generate_success_mocked_openai():
    with patch.object(
        LLMService, "_call_openai", new=AsyncMock(return_value=("Mocked OpenAI response", 10, 20))
    ):
        res = await LLMService.generate(message="Hello", model="gpt-4o-mini")
        assert res.response == "Mocked OpenAI response"
        assert res.model_used == "gpt-4o-mini"
        assert res.provider == "Openai"
        assert res.usage.prompt_tokens == 10
        assert res.usage.completion_tokens == 20
        assert res.usage.total_tokens == 30


@pytest.mark.asyncio
async def test_generate_success_mocked_groq():
    with patch.object(
        LLMService, "_call_groq", new=AsyncMock(return_value=("Mocked Groq response", 15, 30))
    ):
        res = await LLMService.generate(message="Hello", model="llama-3.3-70b-versatile")
        assert res.response == "Mocked Groq response"
        assert res.provider == "Groq"
        assert res.usage.total_tokens == 45


@pytest.mark.asyncio
async def test_generate_success_mocked_anthropic():
    with patch.object(
        LLMService, "_call_anthropic", new=AsyncMock(return_value=("Mocked Claude response", 8, 16))
    ):
        res = await LLMService.generate(message="Hello", model="claude-3-5-sonnet-20241022")
        assert res.response == "Mocked Claude response"
        assert res.provider == "Anthropic"
        assert res.usage.total_tokens == 24


@pytest.mark.asyncio
async def test_generate_timeout_handling():
    with patch.object(LLMService, "_call_openai", side_effect=TimeoutError("Request timed out")):
        with pytest.raises(LLMTimeoutException):
            await LLMService.generate(message="Test", model="gpt-4o")
