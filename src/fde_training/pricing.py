# Pricing per 1,000,000 (1M) tokens in USD: (prompt_rate, completion_rate)
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # OpenAI
    "gpt-4o": (2.50, 10.00),
    "gpt-4o-mini": (0.15, 0.60),
    "o1": (15.00, 60.00),
    "o1-preview": (15.00, 60.00),
    "o1-mini": (3.00, 12.00),
    "o3-mini": (1.10, 4.40),
    # Anthropic
    "claude-3-5-sonnet-20241022": (3.00, 15.00),
    "claude-3-5-sonnet-latest": (3.00, 15.00),
    "claude-3-5-haiku-20241022": (0.80, 4.00),
    "claude-3-5-haiku-latest": (0.80, 4.00),
    "claude-3-opus-20240229": (15.00, 75.00),
    # Groq (Hosted Open-Weights)
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
    "mixtral-8x7b-32768": (0.24, 0.24),
    "gemma2-9b-it": (0.20, 0.20),
}

DEFAULT_RATE: tuple[float, float] = (1.00, 3.00)


def calculate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """Calculate the estimated USD cost of a request based on token counts."""
    # Normalize model string
    clean_model = model.lower().strip()
    rate = MODEL_PRICING.get(clean_model)

    if not rate:
        # Match by partial prefix if exact model date tag differs
        for key, val in MODEL_PRICING.items():
            if key in clean_model or clean_model in key:
                rate = val
                break

    if not rate:
        rate = DEFAULT_RATE

    prompt_rate, completion_rate = rate
    cost = (prompt_tokens * (prompt_rate / 1_000_000.0)) + (completion_tokens * (completion_rate / 1_000_000.0))
    return round(cost, 6)
