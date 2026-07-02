"""Tests for mcp_server.py — the 8 tools and MCP input-burn tracking."""
from __future__ import annotations

import json
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

# Point DB to a temp file before importing mcp_server (it calls init_db() at
# import time), mirroring tests/test_db.py's isolation pattern.
_TMP = str(Path(tempfile.mktemp(suffix=".db")).resolve())
os.environ["FOREMAN_DB_PATH"] = _TMP

import mcp_server  # noqa: E402
from src.db import clear_all_entries, fetch_mcp_calls, insert_entry  # noqa: E402
from src.models import EntrySource, Provider, SpendEntry, WorkloadClass  # noqa: E402


def _seed_entry(**overrides) -> None:
    defaults = dict(
        timestamp=datetime.utcnow(),
        provider=Provider.anthropic,
        model="claude-opus-4",
        workload_class=WorkloadClass.reason,
        input_tokens=1000,
        output_tokens=200,
        cost_usd=1.0,
        team="eng",
        source=EntrySource.manual,
    )
    defaults.update(overrides)
    insert_entry(SpendEntry(**defaults))


@pytest.fixture(autouse=True)
def fresh_db():
    clear_all_entries()
    yield


class TestToolsReturnJsonSafeResults:
    def test_get_key_metrics(self):
        _seed_entry()
        result = mcp_server.get_key_metrics(days=30)
        json.dumps(result)  # raises if numpy scalars leaked through
        assert result["entry_count"] == 1

    def test_get_burn_by_provider(self):
        _seed_entry()
        result = mcp_server.get_burn_by_provider(days=30)
        json.dumps(result)
        assert result[0]["provider"] == "anthropic"

    def test_get_burn_by_model(self):
        _seed_entry()
        result = mcp_server.get_burn_by_model(days=30, limit=5)
        json.dumps(result)
        assert result[0]["model"] == "claude-opus-4"

    def test_get_burn_by_class(self):
        _seed_entry()
        result = mcp_server.get_burn_by_class(days=30)
        json.dumps(result)
        assert result[0]["workload_class"] == "reason"

    def test_get_daily_burn(self):
        _seed_entry()
        result = mcp_server.get_daily_burn(days=30)
        json.dumps(result)
        assert isinstance(result[0]["date"], str)

    def test_get_projection(self):
        _seed_entry()
        result = mcp_server.get_projection(days_ahead=30)
        json.dumps(result)
        assert "projected_total" in result

    def test_get_budget_status_empty_without_budgets(self):
        assert mcp_server.get_budget_status() == []

    def test_get_top_spenders_by_model(self):
        _seed_entry()
        result = mcp_server.get_top_spenders(by="model", limit=5)
        json.dumps(result)
        assert result[0]["model"] == "claude-opus-4"

    def test_get_top_spenders_by_team(self):
        _seed_entry()
        result = mcp_server.get_top_spenders(by="team", limit=5)
        json.dumps(result)
        assert result[0]["team"] == "eng"

    def test_get_top_spenders_rejects_invalid_by(self):
        _seed_entry()
        with pytest.raises(ValueError):
            mcp_server.get_top_spenders(by="teams")  # typo — must not silently mean "model"

    def test_get_top_spenders_by_normalizes_case_and_whitespace(self):
        _seed_entry()
        result = mcp_server.get_top_spenders(by=" Team ", limit=5)
        assert result[0]["team"] == "eng"

    def test_get_top_spenders_empty_without_data(self):
        assert mcp_server.get_top_spenders() == []


class TestDaysZeroIsNotUnfiltered:
    def test_load_with_days_zero_excludes_backdated_entries(self):
        from datetime import timedelta

        _seed_entry(timestamp=datetime.utcnow() - timedelta(days=5))
        # days=0 must mean "as of right now", not "no filter" — a
        # days-old entry must not appear.
        df = mcp_server._load(days=0)
        assert df.empty

    def test_load_with_none_is_unfiltered(self):
        from datetime import timedelta

        _seed_entry(timestamp=datetime.utcnow() - timedelta(days=5))
        df = mcp_server._load(days=None)
        assert len(df) == 1


class TestInputBurnTracking:
    def test_each_call_logs_a_row(self):
        before = len(fetch_mcp_calls())
        mcp_server.get_key_metrics(days=30)
        after = fetch_mcp_calls()
        assert len(after) == before + 1
        assert after[0]["tool"] == "get_key_metrics"
        assert after[0]["response_tokens"] > 0
        assert after[0]["estimated_usd"] >= 0

    def test_response_tokens_scale_with_result_size(self):
        _seed_entry(model="claude-opus-4")
        _seed_entry(model="gpt-4o")
        _seed_entry(model="claude-haiku")

        mcp_server.get_budget_status()  # tiny, empty response
        tiny_tokens = fetch_mcp_calls(limit=1)[0]["response_tokens"]

        mcp_server.get_burn_by_model(days=30, limit=10)
        bigger_tokens = fetch_mcp_calls(limit=1)[0]["response_tokens"]

        assert bigger_tokens > tiny_tokens
