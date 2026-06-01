"""Approximate token pricing for cost visibility in traces and evals.

Prices are illustrative USD-per-1K-tokens (input, output) and are easy to update.
The mock provider is free, so keyless runs report zero cost.
"""

from __future__ import annotations

from ..llm.types import TokenUsage

# (input_per_1k, output_per_1k) in USD. Approximate; for demonstration only.
PRICES_PER_1K: dict[str, tuple[float, float]] = {
    "mock-1": (0.0, 0.0),
    "claude-sonnet-4-6": (0.003, 0.015),
    "claude-opus-4-8": (0.015, 0.075),
    "claude-haiku-4-5-20251001": (0.001, 0.005),
    "command-r-plus": (0.0025, 0.010),
    "command-r": (0.00015, 0.0006),
    "gpt-4o": (0.005, 0.015),
    "gpt-4o-mini": (0.00015, 0.0006),
}

# Used when a model id is not in the table (e.g. a future/custom model).
_DEFAULT_PRICE = (0.003, 0.015)


def cost_for(model: str, usage: TokenUsage) -> float:
    """USD cost for `usage` at `model`'s rate; 0.0 for the mock."""
    input_rate, output_rate = PRICES_PER_1K.get(model, _DEFAULT_PRICE)
    dollars = (usage.input_tokens / 1000.0) * input_rate + (
        usage.output_tokens / 1000.0
    ) * output_rate
    return round(dollars, 6)
