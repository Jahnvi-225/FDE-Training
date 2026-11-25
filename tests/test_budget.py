import pytest
from fde_training.budget import SpendingTracker
from fde_training.exceptions import SpendingLimitExceededException


@pytest.mark.asyncio
async def test_initial_state():
    tracker = SpendingTracker(limit_usd=10.0)
    status = await tracker.get_status()
    assert status["spending_limit_usd"] == 10.0
    assert status["total_spent_usd"] == 0.0
    assert status["remaining_budget_usd"] == 10.0
    assert not status["budget_exhausted"]


@pytest.mark.asyncio
async def test_record_spending_accumulates():
    tracker = SpendingTracker(limit_usd=5.0)
    await tracker.record_spending(0.50)
    await tracker.record_spending(1.25)
    status = await tracker.get_status()
    assert status["total_spent_usd"] == 1.75
    assert status["remaining_budget_usd"] == 3.25
    assert not status["budget_exhausted"]


@pytest.mark.asyncio
async def test_budget_exhaustion_flag():
    tracker = SpendingTracker(limit_usd=2.0)
    await tracker.record_spending(2.0)
    status = await tracker.get_status()
    assert status["total_spent_usd"] == 2.0
    assert status["remaining_budget_usd"] == 0.0
    assert status["budget_exhausted"] is True


@pytest.mark.asyncio
async def test_check_budget_raises_when_exceeded():
    tracker = SpendingTracker(limit_usd=1.0)
    await tracker.record_spending(0.90)

    # Next request of 0.20 exceeds 1.00
    with pytest.raises(SpendingLimitExceededException) as exc_info:
        await tracker.check_budget_available(estimated_cost=0.20)

    assert "Spending limit reached!" in str(exc_info.value)
    assert exc_info.value.status_code == 429
    assert exc_info.value.error_type == "spending_limit_exceeded"


@pytest.mark.asyncio
async def test_reset_functionality():
    tracker = SpendingTracker(limit_usd=5.0)
    await tracker.record_spending(3.50)
    await tracker.reset()
    status = await tracker.get_status()
    assert status["total_spent_usd"] == 0.0
    assert status["remaining_budget_usd"] == 5.0
