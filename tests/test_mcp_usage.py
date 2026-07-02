"""
Tests for src/analytics/mcp_usage.py — dollar figures must be repriced from
response_tokens at read time against the *current* reference model, not
summed from whatever estimated_usd each row was written with.
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

_TMP = str(Path(tempfile.mktemp(suffix=".db")).resolve())
os.environ["FOREMAN_DB_PATH"] = _TMP

from src.analytics import mcp_usage  # noqa: E402
from src.analytics.pricing import price_for_model  # noqa: E402
from src.db import clear_mcp_calls, init_db, insert_mcp_call  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_db():
    init_db()
    clear_mcp_calls()
    yield
    os.environ.pop("FOREMAN_MCP_REFERENCE_MODEL", None)


class TestReadTimeRepricing:
    def test_stored_estimated_usd_column_is_ignored(self):
        # A deliberately wrong stored estimate — if aggregation summed this
        # column directly, it would leak straight into the totals below.
        insert_mcp_call(
            "get_key_metrics", request_tokens=5, response_tokens=1000,
            estimated_usd=999_999.0,
        )
        os.environ["FOREMAN_MCP_REFERENCE_MODEL"] = "claude-haiku"
        in_price, _ = price_for_model("claude-haiku")
        expected = 1000 * in_price / 1_000_000

        df = mcp_usage.load_dataframe()
        metrics = mcp_usage.key_metrics(df)
        assert metrics["total_estimated_usd"] == pytest.approx(expected)
        assert metrics["total_estimated_usd"] != pytest.approx(999_999.0)

    def test_changing_reference_model_reprices_existing_rows(self):
        insert_mcp_call("get_key_metrics", request_tokens=5, response_tokens=1000, estimated_usd=0)

        os.environ["FOREMAN_MCP_REFERENCE_MODEL"] = "claude-haiku"
        df = mcp_usage.load_dataframe()
        haiku_usd = mcp_usage.key_metrics(df)["total_estimated_usd"]

        os.environ["FOREMAN_MCP_REFERENCE_MODEL"] = "claude-opus-4"
        df = mcp_usage.load_dataframe()
        opus_usd = mcp_usage.key_metrics(df)["total_estimated_usd"]

        # Same stored row, same response_tokens — the dollar figure must
        # move with the current reference model, not stay frozen at
        # whichever model was set when the row was inserted.
        assert opus_usd > haiku_usd

    def test_by_tool_and_daily_also_reprice(self):
        insert_mcp_call(
            "get_burn_by_model", request_tokens=5, response_tokens=2000, estimated_usd=0
        )
        os.environ["FOREMAN_MCP_REFERENCE_MODEL"] = "claude-opus-4"
        in_price, _ = price_for_model("claude-opus-4")
        expected = 2000 * in_price / 1_000_000

        df = mcp_usage.load_dataframe()
        tool_df = mcp_usage.by_tool(df)
        daily_df = mcp_usage.daily_input_tokens(df)

        assert tool_df.iloc[0]["estimated_usd"] == pytest.approx(expected)
        assert daily_df.iloc[0]["estimated_usd"] == pytest.approx(expected)
