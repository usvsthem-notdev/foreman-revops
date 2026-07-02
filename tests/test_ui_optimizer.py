"""Tests for the pure-logic helpers in src/ui/optimizer.py (no Streamlit runtime needed)."""
from __future__ import annotations

from src.ui.optimizer import _cache_prefix_savings

_PREFIX = "As a very helpful AI assistant, summarize this support ticket."


class TestCachePrefixSavings:
    def test_openai_first_occurrence_has_no_savings(self):
        # OpenAI's cache-write leg costs the same as fresh input (no premium,
        # no discount) — a single occurrence must show zero savings, not a
        # false positive from treating the write as free.
        assert _cache_prefix_savings(_PREFIX, 1, "gpt-4o") == 0.0

    def test_openai_second_occurrence_shows_real_savings(self):
        assert _cache_prefix_savings(_PREFIX, 2, "gpt-4o") > 0.0

    def test_anthropic_first_occurrence_is_a_net_cost_not_a_saving(self):
        # Anthropic's cache-write leg is a real premium (1.25x) — a single
        # occurrence should show a net cost (negative "savings"), matching
        # the UI's caption that caching pays off starting on the 2nd hit.
        assert _cache_prefix_savings(_PREFIX, 1, "claude-opus-4") < 0.0

    def test_anthropic_second_occurrence_shows_real_savings(self):
        assert _cache_prefix_savings(_PREFIX, 2, "claude-opus-4") > 0.0

    def test_no_prefix_or_zero_occurrences_is_zero(self):
        assert _cache_prefix_savings("", 5, "gpt-4o") == 0.0
        assert _cache_prefix_savings(_PREFIX, 0, "gpt-4o") == 0.0
