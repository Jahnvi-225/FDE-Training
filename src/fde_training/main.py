from datetime import datetime, timezone
from fastapi import FastAPI, status
from pydantic import BaseModel, Field
import uvicorn


class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy", description="Current health status of the service")
    service: str = Field(..., example="fde-training", description="Name of the service")
    version: str = Field(..., example="0.1.0", description="API version")
    timestamp: datetime = Field(..., description="UTC timestamp of the health check")


app = FastAPI(
    title="FDE Training API",
    version="0.1.0",
    description="FastAPI service with health monitoring endpoint",
)


@app.get(
    "/health",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["Health"],
    summary="Health check endpoint",
    description="Returns the current operational health status of the API service.",
)
async def health_check() -> HealthResponse:
    """Check health status of the API."""
    return HealthResponse(
        status="healthy",
        service="fde-training",
        version="0.1.0",
        timestamp=datetime.now(timezone.utc),
    )


@app.get(
    "/healthz",
    response_model=HealthResponse,
    status_code=status.HTTP_200_OK,
    tags=["Health"],
    include_in_schema=False,
)
async def healthz() -> HealthResponse:
    """Kubernetes-style liveness probe alias."""
    return await health_check()


@app.get("/", tags=["General"])
async def root():
    """Root endpoint welcoming users and directing to docs and health."""
    return {
        "message": "Welcome to FDE Training API",
        "endpoints": {
            "health": "/health",
            "docs": "/docs",
        },
    }


def start():
    """Entrypoint to run the server directly."""
    uvicorn.run("fde_training.main:app", host="127.0.0.1", port=8000, reload=True)


if __name__ == "__main__":
    start()
