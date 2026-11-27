from app.exceptions import (
    LLMServiceException,
    LLMTimeoutException,
    AuthenticationException,
    UnsupportedModelException,
    ProviderAPIException,
)


def test_base_exception_defaults():
    exc = LLMServiceException("General error")
    assert exc.message == "General error"
    assert exc.status_code == 500
    assert exc.error_type == "llm_error"


def test_timeout_exception():
    exc = LLMTimeoutException("Provider timed out after 30s")
    assert exc.status_code == 504
    assert exc.error_type == "llm_timeout"


def test_authentication_exception():
    exc = AuthenticationException("Missing Groq API Key")
    assert exc.status_code == 401
    assert exc.error_type == "authentication_error"


def test_unsupported_model_exception():
    exc = UnsupportedModelException("unknown-llm-model")
    assert exc.status_code == 400
    assert exc.error_type == "unsupported_model"
    assert "unknown-llm-model" in exc.message


def test_provider_api_exception():
    exc = ProviderAPIException("Anthropic", "Overloaded", status_code=529)
    assert exc.status_code == 529
    assert exc.error_type == "provider_api_error"
    assert "Provider 'Anthropic' error: Overloaded" in exc.message
