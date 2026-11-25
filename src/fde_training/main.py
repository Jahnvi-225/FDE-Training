from datetime import datetime, timezone
import logging
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
import uvicorn

from fde_training.budget import spending_tracker
from fde_training.config import (
    HARDCODED_MODEL,
    REQUEST_TIMEOUT_SECONDS,
    SPENDING_LIMIT_USD,
)
from fde_training.exceptions import LLMServiceException
from fde_training.schemas import (
    BudgetStatus,
    ChatRequest,
    ChatResponse,
    ErrorResponse,
)
from pydantic import BaseModel, Field

from pathlib import Path

# Setup logging: log both to console (stdout) and persistent logs/app.log file
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / "app.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(log_file, encoding="utf-8"),
    ],
)
logger = logging.getLogger("fde_training")


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["healthy"])
    service: str = Field(..., examples=["fde-training"])
    version: str = Field(..., examples=["0.1.0"])
    timestamp: datetime = Field(...)


app = FastAPI(
    title="FDE Training Multi-Provider LLM API",
    version="0.1.0",
    description="Production-ready FastAPI service integrating OpenAI, Anthropic, and Groq with spending limits, timeouts, and global error handling.",
)


# ==============================================================================
# GLOBAL ERROR HANDLING
# ==============================================================================

@app.exception_handler(LLMServiceException)
async def handle_llm_service_exception(request: Request, exc: LLMServiceException):
    """Handles all domain-specific LLM exceptions (spending limit, timeout, auth, provider errors)."""
    logger.warning(f"Handled LLM exception on {request.url.path}: [{exc.error_type}] {exc.message}")
    error_payload = ErrorResponse(
        error=exc.message,
        error_type=exc.error_type,
        status_code=exc.status_code,
        timestamp=datetime.now(timezone.utc),
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_payload.model_dump(mode="json"),
    )


@app.exception_handler(Exception)
async def handle_unhandled_exception(request: Request, exc: Exception):
    """Fallback catch-all handler for any unexpected system exceptions."""
    logger.error(f"Unhandled exception on {request.url.path}: {str(exc)}", exc_info=True)
    error_payload = ErrorResponse(
        error="An unexpected internal server error occurred. Please check server logs.",
        error_type="internal_server_error",
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        timestamp=datetime.now(timezone.utc),
    )
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_payload.model_dump(mode="json"),
    )


# ==============================================================================
# CHAT ENDPOINT
# ==============================================================================

@app.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    tags=["Chat"],
    summary="Multi-provider LLM chat completion",
    description="Dispatches prompt to the configured LLM (OpenAI, Anthropic, or Groq) with spending limits and timeouts.",
    responses={
        400: {"model": ErrorResponse, "description": "Unsupported model"},
        401: {"model": ErrorResponse, "description": "Authentication / missing API key"},
        429: {"model": ErrorResponse, "description": "Spending limit exceeded or rate limit"},
        502: {"model": ErrorResponse, "description": "Downstream provider API error"},
        504: {"model": ErrorResponse, "description": "Provider call timed out"},
    },
)
async def chat(request: ChatRequest) -> ChatResponse:
    from fde_training.services.llm_service import LLMService

    return await LLMService.generate(
        message=request.message,
        model=request.model,
        system_prompt=request.system_prompt,
        max_tokens=request.max_tokens or 1024,
        temperature=request.temperature if request.temperature is not None else 0.7,
    )


# ==============================================================================
# BUDGET & CONFIGURATION ENDPOINTS
# ==============================================================================

@app.get(
    "/budget",
    response_model=BudgetStatus,
    tags=["Budget & Controls"],
    summary="Inspect current spending and remaining budget",
)
async def get_budget() -> BudgetStatus:
    status_dict = await spending_tracker.get_status()
    return BudgetStatus(**status_dict)


@app.post(
    "/budget/reset",
    tags=["Budget & Controls"],
    summary="Reset accumulated spending tracker to zero",
)
async def reset_budget():
    await spending_tracker.reset()
    return {"message": "Spending tracker reset successfully.", "total_spent_usd": 0.0}


@app.get(
    "/config",
    tags=["General"],
    summary="Inspect active server configuration",
)
async def get_server_config():
    return {
        "hardcoded_model": HARDCODED_MODEL,
        "spending_limit_usd": SPENDING_LIMIT_USD,
        "request_timeout_seconds": REQUEST_TIMEOUT_SECONDS,
        "supported_providers": ["OpenAI", "Anthropic", "Groq"],
    }


# ==============================================================================
# HEALTH & SYSTEM ENDPOINTS
# ==============================================================================

@app.get(
    "/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check endpoint",
)
async def health_check() -> HealthResponse:
    return HealthResponse(
        status="healthy",
        service="fde-training",
        version="0.1.0",
        timestamp=datetime.now(timezone.utc),
    )


@app.get("/healthz", include_in_schema=False)
async def healthz():
    return await health_check()


@app.get("/", tags=["General"])
async def root():
    return {
        "message": "FDE Training Multi-Provider LLM API is operational.",
        "endpoints": {
            "chat": "POST /chat",
            "health": "GET /health",
            "budget": "GET /budget",
            "config": "GET /config",
            "docs": "GET /docs",
        },
    }


def start():
    uvicorn.run("fde_training.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    start()
