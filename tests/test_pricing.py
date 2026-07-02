"""Tests for src/analytics/pricing.py — the input/output cost-split bridge."""
from __future__ import annotations

import pytest

from src.analytics.pricing import (
    KNOWN_MODELS,
    _ALL_PRICES,
    _FALLBACK_PRICE,
    cache_multipliers,
    cache_savings_usd,
    price_for_model,
    split_cost,
)


class TestPriceForModel:
    def test_known_anthropic_model(self):
        assert price_for_model("claude-opus-4") == (15.0, 75.0)

    def test_known_openai_model(self):
        assert price_for_model("gpt-4o") == (2.5, 10.0)

    def test_substring_match_on_dated_model_name(self):
        # Real usage exports append dates/suffixes to model names.
        assert price_for_model("claude-sonnet-4-5-20260101") == (3.0, 15.0)

    def test_longest_key_wins_over_shorter_substring(self):
        # "o1" is a substring of "o1-mini" — the longer, more specific key
        # must win, not whichever happens to iterate first.
        assert price_for_model("o1-mini") == (3.0, 12.0)
        assert price_for_model("o1-preview") == (15.0, 60.0)

    def test_gpt_4o_mini_is_not_priced_as_gpt_4o(self):
        # "gpt-4o" is a substring of "gpt-4o-mini" but they're different,
        # differently-priced models — gpt-4o-mini must not silently inherit
        # gpt-4o's (much higher) rate.
        assert price_for_model("gpt-4o-mini") == (0.15, 0.6)
        assert price_for_model("gpt-4o-mini") != price_for_model("gpt-4o")

    def test_unknown_model_falls_back(self):
        assert price_for_model("some-future-model") == _FALLBACK_PRICE

    def test_empty_or_none_model_falls_back(self):
        assert price_for_model("") == _FALLBACK_PRICE
        assert price_for_model(None) == _FALLBACK_PRICE

    def test_known_models_nonempty_and_all_priceable(self):
        assert len(KNOWN_MODELS) > 0
        for model in KNOWN_MODELS:
            assert price_for_model(model) == _ALL_PRICES[model]


class TestSplitCost:
    def test_splits_sum_back_to_original_cost(self):
        cost = 12.3456
        input_usd, output_usd = split_cost(cost, 10_000, 2_000, 0, "claude-sonnet-4")
        assert input_usd + output_usd == pytest.approx(cost)

    def test_output_weighted_more_per_token(self):
        # Same token counts on both axes -> output leg should dominate the
        # split because output is priced several times higher per token.
        input_usd, output_usd = split_cost(1.0, 1000, 1000, 0, "claude-opus-4")
        assert output_usd > input_usd

    def test_reasoning_tokens_priced_on_output_axis(self):
        with_reasoning = split_cost(1.0, 1000, 0, 1000, "gpt-4o")
        without_reasoning = split_cost(1.0, 1000, 1000, 0, "gpt-4o")
        assert with_reasoning == pytest.approx(without_reasoning)

    def test_zero_cost_returns_zero_split(self):
        assert split_cost(0.0, 1000, 200, 0, "claude-opus-4") == (0.0, 0.0)

    def test_zero_tokens_attributes_all_cost_to_input(self):
        # No token-weighted basis to split on (e.g. a flat invoice line) —
        # fall back to crediting the whole amount to input rather than
        # dividing by zero.
        assert split_cost(5.0, 0, 0, 0, "claude-opus-4") == (5.0, 0.0)

    def test_cache_tokens_exceeding_input_are_capped_not_double_counted(self):
        # Defense in depth for rows written before SpendEntry's validator
        # existed — cache_read_tokens vastly exceeding input_tokens must not
        # overweight the input axis; it should behave as if capped at
        # input_tokens.
        cost = 10.0
        overshoot = split_cost(
            cost, 100, 1000, 0, "claude-opus-4", cache_read_tokens=100_000,
        )
        capped = split_cost(
            cost, 100, 1000, 0, "claude-opus-4", cache_read_tokens=100,
        )
        assert overshoot == pytest.approx(capped)

    def test_cache_defaults_do_not_change_existing_behavior(self):
        # Omitting cache_read/cache_creation must be identical to passing 0 —
        # backward compatible with every pre-caching call site.
        with_defaults = split_cost(4.0, 5000, 1000, 0, "claude-opus-4")
        explicit_zero = split_cost(
            4.0, 5000, 1000, 0, "claude-opus-4",
            cache_read_tokens=0, cache_creation_tokens=0,
        )
        assert with_defaults == explicit_zero

    def test_cache_read_tokens_still_sum_back_to_total_cost(self):
        cost = 3.5
        input_usd, output_usd = split_cost(
            cost, 10_000, 2_000, 0, "claude-opus-4", cache_read_tokens=8_000,
        )
        assert input_usd + output_usd == pytest.approx(cost)

    def test_cache_read_tokens_lower_the_input_share(self):
        # Same total tokens, but a version where most of the input is a cheap
        # cache hit should attribute *less* of the cost to the input axis.
        cost = 1.0
        no_cache = split_cost(cost, 10_000, 1_000, 0, "claude-opus-4")
        mostly_cached = split_cost(
            cost, 10_000, 1_000, 0, "claude-opus-4", cache_read_tokens=9_000,
        )
        assert mostly_cached[0] < no_cache[0]

    def test_cache_creation_is_priced_as_a_premium(self):
        # Anthropic cache-write tokens cost *more* per token than fresh
        # input (1.25x) — replacing fresh input with cache-creation tokens
        # at the same count should raise, not lower, the input-axis weight.
        cost = 1.0
        fresh = split_cost(cost, 10_000, 1_000, 0, "claude-opus-4")
        with_creation = split_cost(
            cost, 10_000, 1_000, 0, "claude-opus-4", cache_creation_tokens=10_000,
        )
        assert with_creation[0] > fresh[0]


class TestCacheSavingsUsd:
    def test_no_cache_reads_no_savings(self):
        assert cache_savings_usd(0, "claude-opus-4") == 0.0

    def test_anthropic_savings_are_90_percent_of_full_price(self):
        in_price, _ = price_for_model("claude-opus-4")
        full_price = 100_000 * in_price / 1_000_000
        saved = cache_savings_usd(100_000, "claude-opus-4")
        assert saved == pytest.approx(full_price * 0.9)

    def test_openai_savings_are_50_percent_of_full_price(self):
        in_price, _ = price_for_model("gpt-4o")
        full_price = 100_000 * in_price / 1_000_000
        saved = cache_savings_usd(100_000, "gpt-4o")
        assert saved == pytest.approx(full_price * 0.5)


class TestCacheMultipliers:
    def test_anthropic_model(self):
        assert cache_multipliers("claude-opus-4") == (0.1, 1.25)

    def test_openai_model_creation_mult_is_full_price_not_free(self):
        # OpenAI has no billed cache-write line, but the first (uncached)
        # hit still costs the SAME as normal input — 1.0, not 0.0. A 0.0
        # multiplier would make caching look free on the very first hit.
        assert cache_multipliers("gpt-4o") == (0.5, 1.0)

    def test_unrecognized_model_gets_neutral_multiplier_not_openai_discount(self):
        # A model that isn't in either price table (local model, Gemini,
        # future provider) must not silently inherit someone else's cache
        # economics.
        assert cache_multipliers("gemini-2.5-pro") == (1.0, 1.0)
        assert cache_multipliers("llama-3-70b-local") == (1.0, 1.0)
        assert cache_multipliers("") == (1.0, 1.0)
        assert cache_multipliers(None) == (1.0, 1.0)
