import pytest
from fde_training.pricing import calculate_cost, MODEL_PRICING, DEFAULT_RATE


def test_openai_pricing_gpt4o():
    # gpt-4o: 2.50 / 1M prompt, 10.00 / 1M completion
    prompt_tokens = 1_000_000
    completion_tokens = 1_000_000
    expected = 2.50 + 10.00
    cost = calculate_cost("gpt-4o", prompt_tokens, completion_tokens)
    assert cost == expected


def test_openai_pricing_gpt4o_mini():
    # gpt-4o-mini: 0.15 / 1M prompt, 0.60 / 1M completion
    cost = calculate_cost("gpt-4o-mini", 100_000, 100_000)
    expected = round((100_000 * 0.15 / 1_000_000) + (100_000 * 0.60 / 1_000_000), 6)
    assert cost == expected


def test_anthropic_pricing_claude_sonnet():
    # claude-3-5-sonnet: 3.00 / 1M prompt, 15.00 / 1M completion
    cost = calculate_cost("claude-3-5-sonnet-20241022", 500_000, 200_000)
    expected = round((500_000 * 3.00 / 1_000_000) + (200_000 * 15.00 / 1_000_000), 6)
    assert cost == expected


def test_groq_pricing_llama33():
    # llama-3.3-70b-versatile: 0.59 / 1M prompt, 0.79 / 1M completion
    cost = calculate_cost("llama-3.3-70b-versatile", 10_000, 20_000)
    expected = round((10_000 * 0.59 / 1_000_000) + (20_000 * 0.79 / 1_000_000), 6)
    assert cost == expected


def test_zero_tokens_returns_zero():
    assert calculate_cost("gpt-4o", 0, 0) == 0.0


def test_unknown_model_uses_default_rate():
    # DEFAULT_RATE: (1.00, 3.00)
    prompt_rate, completion_rate = DEFAULT_RATE
    cost = calculate_cost("completely-unknown-custom-model", 1_000_000, 1_000_000)
    assert cost == (prompt_rate + completion_rate)


def test_case_insensitivity_and_whitespace():
    cost1 = calculate_cost("  GPT-4o  ", 1_000, 1_000)
    cost2 = calculate_cost("gpt-4o", 1_000, 1_000)
    assert cost1 == cost2
