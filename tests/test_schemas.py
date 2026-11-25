import pytest
from pydantic import ValidationError
from fde_training.schemas import ChatRequest, TokenUsage, ChatResponse, BudgetStatus, ErrorResponse
from datetime import datetime, timezone


def test_valid_chat_request_defaults():
    req = ChatRequest(message="What is Python?")
    assert req.message == "What is Python?"
    assert req.model is None
    assert req.max_tokens == 1024
    assert req.temperature == 0.7


def test_chat_request_validation_limits():
    # max_tokens must be >= 1 and <= 8192
    with pytest.raises(ValidationError):
        ChatRequest(message="Test", max_tokens=0)

    with pytest.raises(ValidationError):
        ChatRequest(message="Test", max_tokens=10000)

    # temperature must be >= 0.0 and <= 2.0
    with pytest.raises(ValidationError):
        ChatRequest(message="Test", temperature=-0.1)

    with pytest.raises(ValidationError):
        ChatRequest(message="Test", temperature=2.5)


def test_token_usage_schema():
    usage = TokenUsage(prompt_tokens=10, completion_tokens=25, total_tokens=35)
    assert usage.prompt_tokens == 10
    assert usage.completion_tokens == 25
    assert usage.total_tokens == 35


def test_chat_response_schema():
    resp = ChatResponse(
        response="Hello world",
        model_used="gpt-6-luna",
        provider="Openai",
        usage=TokenUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
        cost_usd=0.00005,
        total_spent_usd=0.00005,
        remaining_budget_usd=4.99995,
        latency_seconds=1.23,
    )
    assert resp.response == "Hello world"
    assert resp.latency_seconds == 1.23


def test_budget_status_schema():
    status = BudgetStatus(
        spending_limit_usd=5.0,
        total_spent_usd=1.0,
        remaining_budget_usd=4.0,
        budget_exhausted=False,
    )
    assert status.remaining_budget_usd == 4.0


def test_error_response_schema():
    err = ErrorResponse(
        error="Test error",
        error_type="test_type",
        status_code=400,
        timestamp=datetime.now(timezone.utc),
    )
    assert err.status_code == 400
