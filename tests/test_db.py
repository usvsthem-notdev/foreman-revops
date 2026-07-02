"""Tests for the database layer — uses a temp SQLite file."""
import os
import tempfile
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import pytest

# Point DB to a temp file before importing src.db.
# Use the resolved tempdir so the path-safety check passes on macOS
# (where /tmp is a symlink to /private/var/folders/...).
_TMP = str(Path(tempfile.mktemp(suffix=".db")).resolve())
os.environ["FOREMAN_DB_PATH"] = _TMP

from src.db import (  # noqa: E402, I001
    clear_all_entries,
    delete_budget,
    delete_entry,
    fetch_budgets,
    fetch_entries,
    get_poll_cursor,
    init_db,
    insert_entries_bulk,
    insert_entry,
    set_poll_cursor,
    upsert_budget,
)
from src.models import Budget, BudgetPeriod, EntrySource, Provider, SpendEntry, WorkloadClass  # noqa: E402


@pytest.fixture(autouse=True)
def fresh_db():
    init_db()
    clear_all_entries()
    yield
    # cleanup budgets
    for b in fetch_budgets():
        delete_budget(b["id"])


def _entry(**kwargs) -> SpendEntry:
    defaults = dict(
        timestamp=datetime(2026, 6, 1),
        provider=Provider.anthropic,
        model="claude-haiku-4-5",
        workload_class=WorkloadClass.extract,
        input_tokens=1000,
        output_tokens=200,
        reasoning_tokens=0,
        cost_usd=0.001,
        is_local=False,
        source=EntrySource.manual,
    )
    defaults.update(kwargs)
    return SpendEntry(**defaults)


class TestInsertFetch:
    def test_insert_and_fetch(self):
        e = _entry()
        insert_entry(e)
        rows = fetch_entries()
        assert len(rows) == 1
        assert rows[0]["model"] == "claude-haiku-4-5"

    def test_bulk_insert(self):
        entries = [_entry(cost_usd=0.001 * i) for i in range(1, 6)]
        count = insert_entries_bulk(entries)
        assert count == 5
        rows = fetch_entries()
        assert len(rows) == 5

    def test_bulk_insert_returns_actual_inserted_not_submitted(self):
        entries = [_entry(cost_usd=0.001 * i) for i in range(1, 6)]
        insert_entries_bulk(entries)
        # Re-importing the same set must return 0, not 5
        count = insert_entries_bulk(entries)
        assert count == 0

    def test_duplicate_id_ignored(self):
        e = _entry()
        insert_entry(e)
        insert_entry(e)  # same id — INSERT OR IGNORE
        assert len(fetch_entries()) == 1

    def test_filter_by_provider(self):
        insert_entry(_entry(provider=Provider.anthropic))
        insert_entry(_entry(provider=Provider.openai))
        rows = fetch_entries(provider="anthropic")
        assert all(r["provider"] == "anthropic" for r in rows)

    def test_filter_by_team(self):
        insert_entry(_entry(team="eng"))
        insert_entry(_entry(team="product"))
        rows = fetch_entries(team="eng")
        assert len(rows) == 1

    def test_filter_by_date_range(self):
        insert_entry(_entry(timestamp=datetime(2026, 5, 1)))
        insert_entry(_entry(timestamp=datetime(2026, 6, 15)))
        rows = fetch_entries(since=datetime(2026, 6, 1))
        assert len(rows) == 1

    def test_delete_entry(self):
        e = _entry()
        insert_entry(e)
        delete_entry(e.id)
        assert len(fetch_entries()) == 0

    def test_clear_all(self):
        insert_entries_bulk([_entry() for _ in range(10)])
        clear_all_entries()
        assert len(fetch_entries()) == 0

    def test_cache_tokens_round_trip(self):
        e = _entry(cache_read_tokens=800, cache_creation_tokens=50)
        insert_entry(e)
        rows = fetch_entries()
        assert rows[0]["cache_read_tokens"] == 800
        assert rows[0]["cache_creation_tokens"] == 50

    def test_cache_tokens_default_to_zero(self):
        insert_entry(_entry())
        rows = fetch_entries()
        assert rows[0]["cache_read_tokens"] == 0
        assert rows[0]["cache_creation_tokens"] == 0

    def test_cache_tokens_exceeding_input_tokens_rejected(self):
        # cache_read/cache_creation are documented as a subset of
        # input_tokens, not additive — constructing a SpendEntry that
        # violates this must fail loudly, not silently corrupt downstream
        # cost attribution.
        with pytest.raises(ValueError, match="subset of input"):
            _entry(input_tokens=100, cache_read_tokens=50000, cache_creation_tokens=0)

    def test_cache_tokens_exactly_equal_to_input_tokens_allowed(self):
        e = _entry(input_tokens=1000, cache_read_tokens=900, cache_creation_tokens=100)
        assert e.cache_read_tokens + e.cache_creation_tokens == e.input_tokens


class TestMigration:
    def test_adds_cache_columns_to_pre_existing_table(self):
        import sqlite3

        from src.db import get_db_path

        # Resolve the path the same way db.py itself does — os.environ can
        # be mutated by other test modules during collection, so a
        # module-level constant captured at import time isn't reliable here.
        db_path = str(get_db_path())

        con = sqlite3.connect(db_path)
        con.execute("DROP TABLE IF EXISTS spend_entries")
        con.execute("""
            CREATE TABLE spend_entries (
                id TEXT PRIMARY KEY, timestamp TEXT NOT NULL, provider TEXT NOT NULL,
                model TEXT NOT NULL, workload_class TEXT NOT NULL DEFAULT 'unknown',
                input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0,
                reasoning_tokens INTEGER NOT NULL DEFAULT 0, cost_usd REAL NOT NULL DEFAULT 0.0,
                is_local INTEGER NOT NULL DEFAULT 0, team TEXT, feature TEXT, notes TEXT,
                source TEXT NOT NULL DEFAULT 'manual'
            )
        """)
        con.commit()
        con.close()

        init_db()

        con = sqlite3.connect(db_path)
        cols = {r[1] for r in con.execute("PRAGMA table_info(spend_entries)")}
        con.close()
        assert "cache_read_tokens" in cols
        assert "cache_creation_tokens" in cols

        # And the table is immediately usable with the new columns.
        insert_entry(_entry(cache_read_tokens=42))
        assert fetch_entries()[0]["cache_read_tokens"] == 42


class TestBudgets:
    def _budget(self, name: str = "test") -> Budget:
        return Budget(
            name=name,
            amount_usd=100.0,
            period=BudgetPeriod.monthly,
        )

    def test_upsert_and_fetch(self):
        upsert_budget(self._budget())
        budgets = fetch_budgets()
        assert len(budgets) == 1
        assert budgets[0]["name"] == "test"

    def test_upsert_updates_amount(self):
        b = self._budget()
        upsert_budget(b)
        b2 = Budget(name="test", amount_usd=200.0, period=BudgetPeriod.monthly)
        upsert_budget(b2)
        budgets = fetch_budgets()
        assert len(budgets) == 1
        assert budgets[0]["amount_usd"] == 200.0

    def test_delete_budget(self):
        b = self._budget()
        upsert_budget(b)
        bid = fetch_budgets()[0]["id"]
        delete_budget(bid)
        assert len(fetch_budgets()) == 0


# ── Path-safety and _conn rollback ───────────────────────────────────────────

class TestDbPathAndConn:
    def test_get_db_path_outside_allowed_raises(self, monkeypatch):
        """Line 38: ValueError when FOREMAN_DB_PATH is outside home/tmp."""
        from src.db import get_db_path
        monkeypatch.setenv("FOREMAN_DB_PATH", "/etc/passwd")
        with pytest.raises(ValueError, match="FOREMAN_DB_PATH must be inside"):
            get_db_path()

    def test_get_db_path_default_when_no_env(self, monkeypatch):
        """Line 40: returns default DB path when env var is unset."""
        from src.db import get_db_path
        monkeypatch.delenv("FOREMAN_DB_PATH", raising=False)
        p = get_db_path()
        assert p.name == "foreman.db"

    def test_conn_rolls_back_and_reraises_on_exception(self):
        """Lines 54-56: _conn context manager rolls back and re-raises."""
        from src import db as db_mod
        with pytest.raises(RuntimeError, match="force-rollback"):
            with db_mod._conn() as con:
                raise RuntimeError("force-rollback")


# ── fetch_entries until filter ────────────────────────────────────────────────

class TestFetchUntil:
    def test_filter_by_until_excludes_later_rows(self):
        """Lines 190-191: until= filter in fetch_entries."""
        insert_entry(_entry(timestamp=datetime(2026, 4, 1)))
        insert_entry(_entry(timestamp=datetime(2026, 7, 1)))
        rows = fetch_entries(until=datetime(2026, 5, 31))
        assert len(rows) == 1
        assert rows[0]["model"] == "claude-haiku-4-5"


# ── Poll cursors ──────────────────────────────────────────────────────────────

class TestPollCursors:
    def test_get_poll_cursor_returns_none_when_absent(self):
        """Lines 254-258: get_poll_cursor returns None for unknown provider."""
        assert get_poll_cursor("nonexistent_zzz") is None

    def test_set_and_get_poll_cursor(self):
        """Lines 254-258, 267-276: set then get a poll cursor."""
        ts = datetime(2026, 6, 15, 10, 0, 0)
        set_poll_cursor("test_prov_xyz", last_polled=ts, since_date=ts, until_date=ts)
        result = get_poll_cursor("test_prov_xyz")
        assert result is not None
        assert result["provider"] == "test_prov_xyz"
        assert "2026-06-15" in result["last_polled"]
