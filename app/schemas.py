from datetime import datetime
from typing import Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    message: str = Field(..., examples=["Explain quantum computing in two sentences."])
    model: Optional[str] = Field(
        None,
        description="Optional model override. If omitted, uses the hardcoded server model.",
        examples=["gpt-4o-mini"],
    )
    system_prompt: Optional[str] = Field(
        None,
        description="Optional system prompt to steer the assistant's behavior.",
        examples=["You are a helpful and concise AI assistant."],
    )
    max_tokens: Optional[int] = Field(
        1024,
        description="Maximum number of tokens to generate.",
        ge=1,
        le=8192,
    )
    temperature: Optional[float] = Field(
        0.7,
        description="Sampling temperature between 0.0 and 2.0.",
        ge=0.0,
        le=2.0,
    )


class TokenUsage(BaseModel):
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt")
    completion_tokens: int = Field(..., description="Number of tokens in the response")
    total_tokens: int = Field(..., description="Total tokens consumed")


class ChatResponse(BaseModel):
    response: str = Field(..., description="Generated text answer from the LLM")
    model_used: str = Field(..., description="Exact model name used for completion")
    provider: str = Field(..., description="Provider that serviced the request (OpenAI, Anthropic, or Groq)")
    usage: TokenUsage = Field(..., description="Token breakdown for the request")
    latency_seconds: float = Field(..., description="Duration of the LLM call in seconds")


class ErrorResponse(BaseModel):
    error: str = Field(..., description="Summary error message")
    error_type: str = Field(..., description="Error classification tag")
    status_code: int = Field(..., description="HTTP status code")
    timestamp: datetime = Field(..., description="UTC timestamp of the error")
