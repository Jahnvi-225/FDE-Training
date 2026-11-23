import asyncio
from fde_training.config import SPENDING_LIMIT_USD
from fde_training.exceptions import SpendingLimitExceededException


class SpendingTracker:
    """Thread-safe spending and budget tracking manager."""

    def __init__(self, limit_usd: float = SPENDING_LIMIT_USD):
        self.limit_usd = limit_usd
        self.total_spent_usd: float = 0.0
        self._lock = asyncio.Lock()

    async def check_budget_available(self, estimated_cost: float = 0.0) -> None:
        """Validate whether the budget can accommodate the expected request."""
        async with self._lock:
            if (self.total_spent_usd + estimated_cost) > self.limit_usd:
                remaining = max(0.0, self.limit_usd - self.total_spent_usd)
                raise SpendingLimitExceededException(
                    f"Spending limit reached! Limit: ${self.limit_usd:.4f}, "
                    f"Total Spent: ${self.total_spent_usd:.4f}, "
                    f"Remaining: ${remaining:.4f}. Request rejected to prevent budget overrun."
                )

    async def record_spending(self, actual_cost: float) -> float:
        """Record the actual cost after successful generation and return new total."""
        async with self._lock:
            self.total_spent_usd = round(self.total_spent_usd + actual_cost, 6)
            return self.total_spent_usd

    async def get_status(self) -> dict:
        """Return the current budget and spending metrics."""
        async with self._lock:
            remaining = max(0.0, round(self.limit_usd - self.total_spent_usd, 6))
            return {
                "spending_limit_usd": self.limit_usd,
                "total_spent_usd": self.total_spent_usd,
                "remaining_budget_usd": remaining,
                "budget_exhausted": self.total_spent_usd >= self.limit_usd,
            }

    async def reset(self) -> None:
        """Reset the spent amount back to zero."""
        async with self._lock:
            self.total_spent_usd = 0.0


# Global singleton instance
spending_tracker = SpendingTracker()
