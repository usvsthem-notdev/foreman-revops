"""
Shared per-model $/M-token pricing.

Reuses the pricing tables the CSV parsers already maintain (single source of
truth — no drift between "what a bill parser estimates" and "how the burn map
attributes cost by axis"). Used to split each entry's *already reported*
cost_usd into input/output shares, since Foreman stores a single blended cost
per entry but tracks input/output/reasoning token counts separately.
"""
from __future__ import annotations

import os

from src.parsers.anthropic import ANTHROPIC_PRICES
from src.parsers.openai import OPENAI_PRICES

# Model variants missing from the parser tables above. Without an explicit
# entry, substring matching would shadow these behind a shorter sibling key
# with different real pricing (e.g. "gpt-4o-mini" matching "gpt-4o").
_SUPPLEMENTAL_PRICES: dict[str, tuple[float, float]] = {
    "gpt-4o-mini": (0.15, 0.6),
}

# Approximate Google/Gemini pricing ($/M tokens) — July 2026. No dedicated
# Gemini bill parser yet, so the table lives here; Gemini rows arrive via
# live polling and the generic parser. Gemini 3 Pro rates are the ≤200K
# context tier (long-context requests bill higher).
GOOGLE_PRICES: dict[str, tuple[float, float]] = {
    "gemini-3.5-flash": (1.5,  9.0),
    "gemini-3-pro":     (2.0,  12.0),
    "gemini-3-flash":   (0.5,  3.0),
    # Prior generation — historical data
    "gemini-2.5-pro":   (1.25, 10.0),
    "gemini-2.5-flash": (0.3,  2.5),
    "gemini-2.0-flash": (0.1,  0.4),
}

_ALL_PRICES: dict[str, tuple[float, float]] = {
    **ANTHROPIC_PRICES,
    **OPENAI_PRICES,
    **GOOGLE_PRICES,
    **_SUPPLEMENTAL_PRICES,
}

# Longest key first, so e.g. "o1-mini" matches before the shorter "o1" —
# dict/insertion order alone would let "o1" shadow "o1-mini" every time.
_LOOKUP_ORDER: list[str] = sorted(_ALL_PRICES, key=len, reverse=True)

# Mid-tier fallback for models neither pricing table recognizes.
_FALLBACK_PRICE: tuple[float, float] = (3.0, 15.0)

# Every model name this module can price directly — for UI dropdowns etc.
KNOWN_MODELS: list[str] = sorted(_ALL_PRICES)


def price_for_model(model: str) -> tuple[float, float]:
    """Return (input_price, output_price) per million tokens for a model name."""
    model_lower = (model or "").lower()
    for key in _LOOKUP_ORDER:
        if key in model_lower:
            return _ALL_PRICES[key]
    return _FALLBACK_PRICE


# Prompt/prefix cache pricing, as a multiplier on the model's base input
# price. Real, published discount ratios — not illustrative like the base
# price tables' "approximate" tag, but still per-provider approximations
# since Anthropic/OpenAI both round differently model-to-model.
#   Anthropic: cache read  ~10% of input price (90% discount)
#              cache write ~125% of input price (a premium — you pay extra
#              to populate the cache, in exchange for future cheap reads)
#   OpenAI:    GPT-5-era models discount cached input 90% (0.1x); the older
#              gpt-4o generation discounted 50% (0.5x). Caching is automatic
#              with no separate billed "write" line — the first (uncached)
#              hit costs the SAME as normal input, i.e. a 1.0 multiplier,
#              not a discount and not free.
#   Google:    implicit caching bills cached Gemini tokens at ~25% of input
#              price (75% discount); no billed write line. (Explicit context
#              caching adds hourly storage — not modeled here.)
_ANTHROPIC_CACHE_MULT: tuple[float, float] = (0.1, 1.25)
_OPENAI_CACHE_MULT: tuple[float, float] = (0.1, 1.0)
_OPENAI_LEGACY_CACHE_MULT: tuple[float, float] = (0.5, 1.0)
_GOOGLE_CACHE_MULT: tuple[float, float] = (0.25, 1.0)
# No discount/premium at all — used for any model outside the tables above
# (local models, future providers) so an entry with cache tokens against an
# unrecognized model doesn't silently inherit someone else's cache economics.
_NEUTRAL_CACHE_MULT: tuple[float, float] = (1.0, 1.0)

_OPENAI_CURRENT_GEN_PREFIXES = ("gpt-5",)

# Keyed the same way as _ALL_PRICES, resolved through the same _LOOKUP_ORDER
# substring match — one mechanism for both price and cache-multiplier lookup.
_CACHE_MULT_BY_KEY: dict[str, tuple[float, float]] = {
    **{k: _ANTHROPIC_CACHE_MULT for k in ANTHROPIC_PRICES},
    **{
        k: (
            _OPENAI_CACHE_MULT
            if k.startswith(_OPENAI_CURRENT_GEN_PREFIXES)
            else _OPENAI_LEGACY_CACHE_MULT
        )
        for k in OPENAI_PRICES
    },
    **{k: _GOOGLE_CACHE_MULT for k in GOOGLE_PRICES},
    **{k: _OPENAI_LEGACY_CACHE_MULT for k in _SUPPLEMENTAL_PRICES},
}


def cache_multipliers(model: str) -> tuple[float, float]:
    """Return (cache_read_multiplier, cache_creation_multiplier) for a model."""
    model_lower = (model or "").lower()
    for key in _LOOKUP_ORDER:
        if key in model_lower:
            return _CACHE_MULT_BY_KEY[key]
    return _NEUTRAL_CACHE_MULT


def split_cost(
    cost_usd: float,
    input_tokens: int,
    output_tokens: int,
    reasoning_tokens: int,
    model: str,
    cache_read_tokens: int = 0,
    cache_creation_tokens: int = 0,
) -> tuple[float, float]:
    """Split a blended cost_usd into (input_usd, output_usd).

    The split is proportional to each axis's token-weighted list price, so the
    two legs always sum back to the original cost_usd — no drift from the
    real reported/estimated total. Reasoning tokens are priced on the output
    axis, matching _estimate_openai_cost's existing convention.

    cache_read_tokens/cache_creation_tokens are a SUBSET of input_tokens (not
    additive — same convention as the SpendEntry fields) and are weighted at
    their real discounted/premium rate rather than the full input price, so
    cache-heavy entries don't overstate the input axis's true cost share.
    """
    if cost_usd <= 0:
        return 0.0, 0.0
    in_price, out_price = price_for_model(model)
    read_mult, creation_mult = cache_multipliers(model)
    # Defensive: cache_read/creation are documented (and validated on write
    # via SpendEntry) as a subset of input_tokens, not additive. Cap them
    # here too for any row written before that validator existed — otherwise
    # bad data that exceeds input_tokens gets counted at full magnitude on
    # top of a zero-clamped fresh_input, wildly overweighting the input axis.
    cache_read_tokens = max(min(cache_read_tokens, input_tokens), 0)
    cache_creation_tokens = max(min(cache_creation_tokens, input_tokens - cache_read_tokens), 0)
    fresh_input = max(input_tokens - cache_read_tokens - cache_creation_tokens, 0)
    in_weight = (
        fresh_input * in_price
        + cache_read_tokens * in_price * read_mult
        + cache_creation_tokens * in_price * creation_mult
    )
    out_weight = (output_tokens + reasoning_tokens) * out_price
    total_weight = in_weight + out_weight
    if total_weight <= 0:
        return cost_usd, 0.0
    input_usd = cost_usd * in_weight / total_weight
    return input_usd, cost_usd - input_usd


def cache_savings_usd(cache_read_tokens: int, model: str) -> float:
    """What cache_read_tokens actually cost at the discounted rate, subtracted
    from what they would have cost at full fresh-input price — the $ a cache
    hit saved versus paying full price for the same tokens."""
    if cache_read_tokens <= 0:
        return 0.0
    in_price, _ = price_for_model(model)
    read_mult, _ = cache_multipliers(model)
    full_price_usd = cache_read_tokens * in_price / 1_000_000
    discounted_usd = full_price_usd * read_mult
    return full_price_usd - discounted_usd


_DEFAULT_MCP_REFERENCE_MODEL = "claude-sonnet-4"


def mcp_reference_model() -> str:
    """The model whose input price stands in for 'cost of an MCP tool
    response being read back into a calling agent's context', until the
    caller identifies itself. Shared by mcp_server.py (to price a call as
    it happens) and src.analytics.mcp_usage (to reprice historical calls
    against the *current* setting rather than whatever was set when each
    row was recorded)."""
    return os.environ.get("FOREMAN_MCP_REFERENCE_MODEL", _DEFAULT_MCP_REFERENCE_MODEL)
