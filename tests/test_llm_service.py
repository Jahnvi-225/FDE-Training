import pytest
from unittest.mock import patch, AsyncMock
from fde_training.services.llm_service import LLMService
from fde_training.budget import spending_tracker
from fde_training.exceptions import (
    UnsupportedModelException,
    SpendingLimitExceededException,
    LLMTimeoutException,
    AuthenticationException,
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
    # Mock _call_openai to return mock text, prompt_tokens, completion_tokens
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
        assert res.cost_usd > 0
        assert res.total_spent_usd > 0


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
async def test_generate_budget_exhaustion_blocks_call():
    # Exhaust budget
    spending_tracker.limit_usd = 0.05
    await spending_tracker.record_spending(0.06)

    with pytest.raises(SpendingLimitExceededException):
        await LLMService.generate(message="Test", model="gpt-4o")


@pytest.mark.asyncio
async def test_generate_timeout_handling():
    # Simulate provider timeout
    with patch.object(LLMService, "_call_openai", side_effect=TimeoutError("Request timed out")):
        with pytest.raises(LLMTimeoutException):
            await LLMService.generate(message="Test", model="gpt-4o")
