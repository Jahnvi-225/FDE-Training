import pytest
import pytest_asyncio
import httpx
from fde_training.budget import spending_tracker
from fde_training.main import app


@pytest_asyncio.fixture(autouse=True)
async def reset_spending_tracker():
    """Ensure every test begins with a clean, reset spending tracker."""
    original_limit = spending_tracker.limit_usd
    await spending_tracker.reset()
    yield
    await spending_tracker.reset()
    spending_tracker.limit_usd = original_limit


@pytest_asyncio.fixture
async def async_client():
    """Asynchronous test client using ASGI transport."""
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        yield client
