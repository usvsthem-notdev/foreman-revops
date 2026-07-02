"""Tests for the pure-logic helpers in src/ui/exec_brief.py (no Streamlit runtime needed)."""
from __future__ import annotations

from src.ui.exec_brief import _escape_markdown_dollars


class TestEscapeMarkdownDollars:
    def test_escapes_every_dollar_sign(self):
        text = "You spent **$509.41** over the last 31 days — a run-rate of **$15.92**."
        escaped = _escape_markdown_dollars(text)
        assert "\\$509.41" in escaped
        assert "\\$15.92" in escaped
        assert "$" not in escaped.replace("\\$", "")

    def test_bold_and_dashes_untouched(self):
        # Only the $ characters change — markdown bold markers and the
        # em-dash/hyphen must survive as-is.
        text = "Spend **accelerated 20%** week-over-week ($100.00 vs $80.00)."
        escaped = _escape_markdown_dollars(text)
        assert "**accelerated 20%**" in escaped
        assert "week-over-week" in escaped

    def test_no_dollar_signs_is_a_no_op(self):
        text = "All 5 budgets are on track."
        assert _escape_markdown_dollars(text) == text
