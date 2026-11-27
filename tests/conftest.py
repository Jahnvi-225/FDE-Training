import pytest_asyncio
import httpx
from app.main import app


@pytest_asyncio.fixture
async def async_client():
    """Asynchronous test client using ASGI transport."""
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
