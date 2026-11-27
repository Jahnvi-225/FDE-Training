from datetime import datetime, timezone
import logging
from pathlib import Path
from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

from app.exceptions import LLMServiceException
from app.schemas import ChatRequest, ChatResponse, ErrorResponse
from app.services.llm_service import LLMService

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
logger = logging.getLogger("app")


class HealthResponse(BaseModel):
    status: str = Field(..., examples=["healthy"])
    service: str = Field(..., examples=["fde-training"])
    version: str = Field(..., examples=["0.1.0"])
    timestamp: datetime = Field(...)


app = FastAPI(
    title="FDE Training Multi-Provider LLM API",
    version="0.1.0",
    description="FastAPI service integrating OpenAI, Anthropic, and Groq with timeouts and global error handling.",
)


# ==============================================================================
# GLOBAL ERROR HANDLING
# ==============================================================================

@app.exception_handler(LLMServiceException)
async def handle_llm_service_exception(request: Request, exc: LLMServiceException):
    """Handles all domain-specific LLM exceptions (timeout, auth, unsupported model, provider errors)."""
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
# ENDPOINTS
# ==============================================================================

@app.post(
    "/chat",
    response_model=ChatResponse,
    status_code=status.HTTP_200_OK,
    tags=["Chat"],
    summary="Multi-provider LLM chat completion",
    description="Dispatches prompt to the configured LLM (OpenAI, Anthropic, or Groq) with timeouts and logging.",
    responses={
        400: {"model": ErrorResponse, "description": "Unsupported model"},
        401: {"model": ErrorResponse, "description": "Authentication / missing API key"},
        429: {"model": ErrorResponse, "description": "Provider rate limit"},
        502: {"model": ErrorResponse, "description": "Downstream provider API error"},
        504: {"model": ErrorResponse, "description": "Provider call timed out"},
    },
)
async def chat(request: ChatRequest) -> ChatResponse:
    return await LLMService.generate(
        message=request.message,
        model=request.model,
        system_prompt=request.system_prompt,
        max_tokens=request.max_tokens or 1024,
        temperature=request.temperature if request.temperature is not None else 0.7,
    )


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


@app.get("/", tags=["General"])
async def root():
    return {
        "message": "FDE Training Multi-Provider LLM API is operational.",
        "endpoints": {
            "chat": "POST /chat",
            "health": "GET /health",
            "docs": "GET /docs",
        },
    }


def start():
    uvicorn.run("app.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    start()
