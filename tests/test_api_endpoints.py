import pytest
from unittest.mock import patch, AsyncMock
from app.exceptions import (
    LLMTimeoutException,
    AuthenticationException,
    ProviderAPIException,
)
from app.services.llm_service import LLMService


@pytest.mark.asyncio
async def test_health_endpoint(async_client):
    r = await async_client.get("/health")
    assert r.status_code == 200
    data = r.json()
    assert data["status"] == "healthy"
    assert data["service"] == "fde-training"


@pytest.mark.asyncio
async def test_root_endpoint(async_client):
    r = await async_client.get("/")
    assert r.status_code == 200
    data = r.json()
    assert "endpoints" in data


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
