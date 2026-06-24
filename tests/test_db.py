"""Tests for the database layer — uses a temp SQLite file."""
import os
import tempfile
from datetime import datetime
from pathlib import Path

import pytest

# Point DB to a temp file before importing src.db.
# Use the resolved tempdir so the path-safety check passes on macOS
# (where /tmp is a symlink to /private/var/folders/...).
_TMP = str(Path(tempfile.mktemp(suffix=".db")).resolve())
os.environ["FOREMAN_DB_PATH"] = _TMP

from src.db import (
    clear_all_entries,
    delete_budget,
    delete_entry,
    fetch_budgets,
    fetch_entries,
    init_db,
    insert_entries_bulk,
    insert_entry,
    upsert_budget,
)
from src.models import Budget, BudgetPeriod, EntrySource, Provider, SpendEntry, WorkloadClass


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
