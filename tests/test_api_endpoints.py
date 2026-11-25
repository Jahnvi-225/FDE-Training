import pytest
from unittest.mock import patch, AsyncMock
from fde_training.budget import spending_tracker
from fde_training.exceptions import (
    SpendingLimitExceededException,
    LLMTimeoutException,
    AuthenticationException,
    ProviderAPIException,
)
from fde_training.services.llm_service import LLMService


@pytest.mark.asyncio
async def test_health_endpoints(async_client):
    r1 = await async_client.get("/health")
    assert r1.status_code == 200
    data1 = r1.json()
    assert data1["status"] == "healthy"
    assert data1["service"] == "fde-training"

    r2 = await async_client.get("/healthz")
    assert r2.status_code == 200
    assert r2.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_root_endpoint(async_client):
    r = await async_client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert "endpoints" in data


@pytest.mark.asyncio
async def test_config_endpoint(async_client):
    r = await async_client.get("/config")
    assert r.status_code == 200
    data = r.json()
    assert "hardcoded_model" in data
    assert "spending_limit_usd" in data
    assert "request_timeout_seconds" in data
    assert data["supported_providers"] == ["OpenAI", "Anthropic", "Groq"]


@pytest.mark.asyncio
async def test_budget_endpoints(async_client):
    # Check budget
    r = await async_client.get("/budget")
    assert r.status_code == 200
    data = r.json()
    assert "spending_limit_usd" in data
    assert data["total_spent_usd"] == 0.0

    # Record some spending and test reset
    await spending_tracker.record_spending(1.50)
    r2 = await async_client.get("/budget")
    assert r2.json()["total_spent_usd"] == 1.50

    r_reset = await async_client.post("/budget/reset")
    assert r_reset.status_code == 200
    assert r_reset.json()["total_spent_usd"] == 0.0


@pytest.mark.asyncio
async def test_chat_endpoint_success(async_client):
    with patch.object(
        LLMService, "_call_openai", new=AsyncMock(return_value=("API endpoint working", 15, 25))
    ):
        payload = {"message": "Hello test", "model": "gpt-4o-mini"}
        r = await async_client.post("/chat", json=payload)
        assert r.status_code == 200
        data = r.json()
        assert data["response"] == "API endpoint working"
        assert data["model_used"] == "gpt-4o-mini"
        assert data["provider"] == "Openai"
        assert data["usage"]["total_tokens"] == 40


# ==============================================================================
# GLOBAL EXCEPTION HANDLER TESTS
# ==============================================================================

@pytest.mark.asyncio
async def test_global_error_unsupported_model(async_client):
    r = await async_client.post("/chat", json={"message": "Hi", "model": "invalid-model-prefix"})
    assert r.status_code == 400
    data = r.json()
    assert data["status_code"] == 400
    assert data["error_type"] == "unsupported_model"
    assert "Unsupported or unrecognized model" in data["error"]


@pytest.mark.asyncio
async def test_global_error_spending_limit_exceeded(async_client):
    with patch.object(
        LLMService, "generate", side_effect=SpendingLimitExceededException("Budget cap hit!")
    ):
        r = await async_client.post("/chat", json={"message": "Hi"})
        assert r.status_code == 429
        data = r.json()
        assert data["status_code"] == 429
        assert data["error_type"] == "spending_limit_exceeded"
        assert "Budget cap hit!" in data["error"]


@pytest.mark.asyncio
async def test_global_error_llm_timeout(async_client):
    with patch.object(
        LLMService, "generate", side_effect=LLMTimeoutException("Timed out waiting for Groq")
    ):
        r = await async_client.post("/chat", json={"message": "Hi"})
        assert r.status_code == 504
        data = r.json()
        assert data["status_code"] == 504
        assert data["error_type"] == "llm_timeout"
        assert "Timed out waiting for Groq" in data["error"]


@pytest.mark.asyncio
async def test_global_error_authentication(async_client):
    with patch.object(
        LLMService, "generate", side_effect=AuthenticationException("Missing API key")
    ):
        r = await async_client.post("/chat", json={"message": "Hi"})
        assert r.status_code == 401
        data = r.json()
        assert data["status_code"] == 401
        assert data["error_type"] == "authentication_error"
        assert "Missing API key" in data["error"]


@pytest.mark.asyncio
async def test_global_error_provider_api(async_client):
    with patch.object(
        LLMService, "generate", side_effect=ProviderAPIException("Anthropic", "Capacity overloaded", 529)
    ):
        r = await async_client.post("/chat", json={"message": "Hi"})
        assert r.status_code == 529
        data = r.json()
        assert data["status_code"] == 529
        assert data["error_type"] == "provider_api_error"
        assert "Capacity overloaded" in data["error"]


@pytest.mark.asyncio
async def test_global_error_unhandled_exception(async_client):
    with patch.object(
        LLMService, "generate", side_effect=RuntimeError("Unexpected system failure")
    ):
        r = await async_client.post("/chat", json={"message": "Hi"})
        assert r.status_code == 500
        data = r.json()
        assert data["status_code"] == 500
        assert data["error_type"] == "internal_server_error"
        assert "An unexpected internal server error occurred" in data["error"]
