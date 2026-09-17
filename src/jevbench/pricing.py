"""Published per-token prices, and cost estimated from measured token usage.

Prices are USD per million tokens (input, output), entered from public rate
tables on the dates below. They are estimates for reporting, never a billed
amount. Update the table and the `as_of` date when a provider reprices.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input_per_mtok: float
    output_per_mtok: float
    as_of: str
    source: str


# Keyed by the model id passed on the CLI. Jev bills input only (output free);
# its output rate is 0.0. Sources recorded so a reviewer can re-check.
PRICES: dict[str, Price] = {
    "jev-1.13.0": Price(0.042, 0.0, "2026-09-17", "https://docs.typesafe.ai/models"),
    "jev-latest": Price(0.042, 0.0, "2026-09-17", "https://docs.typesafe.ai/models"),
    # OpenAI GPT-5.6 tiers and GPT-5, from 2026 rate tables.
    "gpt-5.6-terra": Price(2.0, 12.0, "2026-09-17", "openai 2026 rate table"),
    "gpt-5.6-sol": Price(4.0, 20.0, "2026-09-17", "openai 2026 rate table (promo)"),
    "gpt-5.6-luna": Price(0.20, 1.20, "2026-09-17", "openai 2026 rate table"),
    "gpt-5": Price(1.25, 10.0, "2026-09-17", "openai 2026 rate table"),
}


def price_for(model: str) -> Price | None:
    """Return the price row for a model id, or None if we have no rate for it."""
    return PRICES.get(model)


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Estimated USD cost for the given token counts, or None if unpriced."""
    p = price_for(model)
    if p is None:
        return None
    return input_tokens / 1e6 * p.input_per_mtok + output_tokens / 1e6 * p.output_per_mtok
