class LLMServiceException(Exception):
    """Base exception for all LLM service-related errors."""
    def __init__(self, message: str, status_code: int = 500, error_type: str = "llm_error"):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_type = error_type


class SpendingLimitExceededException(LLMServiceException):
    """Raised when request or cumulative cost exceeds the configured spending limit."""
    def __init__(self, message: str = "Spending limit exceeded."):
        super().__init__(message, status_code=429, error_type="spending_limit_exceeded")


class LLMTimeoutException(LLMServiceException):
    """Raised when an LLM provider call exceeds the configured timeout."""
    def __init__(self, message: str = "Request to LLM provider timed out."):
        super().__init__(message, status_code=504, error_type="llm_timeout")


class AuthenticationException(LLMServiceException):
    """Raised when an API key is missing or rejected by the provider."""
    def __init__(self, message: str = "Missing or invalid API key."):
        super().__init__(message, status_code=401, error_type="authentication_error")


class UnsupportedModelException(LLMServiceException):
    """Raised when a requested model cannot be routed to any known provider."""
    def __init__(self, model: str):
        super().__init__(
            f"Unsupported or unrecognized model '{model}'. Supported providers: OpenAI, Anthropic, Groq.",
            status_code=400,
            error_type="unsupported_model",
        )


class ProviderAPIException(LLMServiceException):
    """Raised when a downstream LLM provider returns an API or rate-limit error."""
    def __init__(self, provider: str, message: str, status_code: int = 502):
        super().__init__(
            f"Provider '{provider}' error: {message}",
            status_code=status_code,
            error_type="provider_api_error",
        )
