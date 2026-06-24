"""
SQLite persistence layer.  All user-supplied values are bound via ? parameters.
The WHERE clause skeletons are assembled from hard-coded string literals (never
user input), so there is no SQL injection surface.
"""
from __future__ import annotations

import logging
import os
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from src.models import Budget, SpendEntry

log = logging.getLogger(__name__)

_DB_PATH_ENV = "FOREMAN_DB_PATH"
_DEFAULT_DB = Path(__file__).parent.parent / "data" / "foreman.db"


def get_db_path() -> Path:
    raw = os.environ.get(_DB_PATH_ENV, "")
    if raw:
        import tempfile
        p = Path(raw).resolve()
        # Use is_relative_to for boundary-aware path checks (no prefix-collision bypass).
        # Include both gettempdir() and /tmp because on macOS /tmp → /private/tmp while
        # gettempdir() returns the per-session Launchd temp dir.
        allowed = [
            Path.home().resolve(),
            Path(tempfile.gettempdir()).resolve(),
            Path("/tmp").resolve(),   # noqa: S108
            Path("/app").resolve(),   # Docker working directory
        ]
        if not any(p == a or p.is_relative_to(a) for a in allowed):
            raise ValueError(f"FOREMAN_DB_PATH must be inside home, temp, or /app dir, got: {p}")
        return p
    return _DEFAULT_DB.resolve()


@contextmanager
def _conn() -> Generator[sqlite3.Connection, None, None]:
    path = get_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(str(path), detect_types=sqlite3.PARSE_DECLTYPES)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA foreign_keys=ON")
    try:
        yield con
        con.commit()
    except Exception:
        con.rollback()
        raise
    finally:
        con.close()


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS spend_entries (
    id              TEXT PRIMARY KEY,
    timestamp       TEXT NOT NULL,
    provider        TEXT NOT NULL,
    model           TEXT NOT NULL,
    workload_class  TEXT NOT NULL DEFAULT 'unknown',
    input_tokens    INTEGER NOT NULL DEFAULT 0,
    output_tokens   INTEGER NOT NULL DEFAULT 0,
    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
    cost_usd        REAL NOT NULL DEFAULT 0.0,
    is_local        INTEGER NOT NULL DEFAULT 0,
    team            TEXT,
    feature         TEXT,
    notes           TEXT,
    source          TEXT NOT NULL DEFAULT 'manual'
);

CREATE TABLE IF NOT EXISTS budgets (
    id              TEXT PRIMARY KEY,
    name            TEXT NOT NULL UNIQUE,
    amount_usd      REAL NOT NULL,
    period          TEXT NOT NULL,
    provider        TEXT,
    team            TEXT,
    alert_threshold REAL NOT NULL DEFAULT 0.8,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_spend_timestamp ON spend_entries(timestamp);
CREATE INDEX IF NOT EXISTS idx_spend_provider  ON spend_entries(provider);
CREATE INDEX IF NOT EXISTS idx_spend_team      ON spend_entries(team);

CREATE TABLE IF NOT EXISTS poll_cursors (
    provider    TEXT PRIMARY KEY,
    last_polled TEXT NOT NULL,
    since_date  TEXT NOT NULL,
    until_date  TEXT NOT NULL
);
"""


def init_db() -> None:
    with _conn() as con:
        con.executescript(_SCHEMA)
    log.info("Database initialised at %s", get_db_path())


# ---------------------------------------------------------------------------
# Spend entries
# ---------------------------------------------------------------------------

def insert_entry(entry: SpendEntry) -> None:
    sql = """
        INSERT OR IGNORE INTO spend_entries
            (id, timestamp, provider, model, workload_class,
             input_tokens, output_tokens, reasoning_tokens,
             cost_usd, is_local, team, feature, notes, source)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """
    with _conn() as con:
        con.execute(sql, (
            entry.id,
            entry.timestamp.isoformat(),
            entry.provider.value,
            entry.model,
            entry.workload_class.value,
            entry.input_tokens,
            entry.output_tokens,
            entry.reasoning_tokens,
            entry.cost_usd,
            int(entry.is_local),
            entry.team,
            entry.feature,
            entry.notes,
            entry.source.value,
        ))


def insert_entries_bulk(entries: list[SpendEntry]) -> int:
    """Return the number of rows *actually inserted* (skips already-existing IDs)."""
    sql = """
        INSERT OR IGNORE INTO spend_entries
            (id, timestamp, provider, model, workload_class,
             input_tokens, output_tokens, reasoning_tokens,
             cost_usd, is_local, team, feature, notes, source)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """
    rows = [
        (e.id, e.timestamp.isoformat(), e.provider.value, e.model,
         e.workload_class.value, e.input_tokens, e.output_tokens,
         e.reasoning_tokens, e.cost_usd, int(e.is_local),
         e.team, e.feature, e.notes, e.source.value)
        for e in entries
    ]
    with _conn() as con:
        # total_changes counts across all rows in executemany; changes() only
        # reflects the final iteration.
        before = con.total_changes
        con.executemany(sql, rows)
        inserted = con.total_changes - before
    return inserted


def fetch_entries(
    *,
    provider: str | None = None,
    team: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    limit: int = 10_000,
) -> list[dict]:
    clauses = []
    params: list = []

    if provider:
        clauses.append("provider = ?")
        params.append(provider)
    if team:
        clauses.append("team = ?")
        params.append(team)
    if since:
        clauses.append("timestamp >= ?")
        params.append(since.isoformat())
    if until:
        clauses.append("timestamp <= ?")
        params.append(until.isoformat())

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    # LIMIT via a bound parameter — keeps every value out of the SQL string.
    params.append(int(limit))
    sql = f"SELECT * FROM spend_entries {where} ORDER BY timestamp DESC LIMIT ?"  # noqa: S608

    with _conn() as con:
        rows = con.execute(sql, params).fetchall()
    return [dict(r) for r in rows]


def delete_entry(entry_id: str) -> None:
    with _conn() as con:
        con.execute("DELETE FROM spend_entries WHERE id = ?", (entry_id,))


def clear_all_entries() -> None:
    with _conn() as con:
        con.execute("DELETE FROM spend_entries")


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------

def upsert_budget(budget: Budget) -> None:
    sql = """
        INSERT INTO budgets
            (id, name, amount_usd, period, provider, team, alert_threshold, created_at)
        VALUES (?,?,?,?,?,?,?,?)
        ON CONFLICT(name) DO UPDATE SET
            amount_usd=excluded.amount_usd,
            period=excluded.period,
            provider=excluded.provider,
            team=excluded.team,
            alert_threshold=excluded.alert_threshold
    """
    with _conn() as con:
        con.execute(sql, (
            budget.id, budget.name, budget.amount_usd, budget.period.value,
            budget.provider.value if budget.provider else None,
            budget.team, budget.alert_threshold,
            budget.created_at.isoformat(),
        ))


def fetch_budgets() -> list[dict]:
    with _conn() as con:
        rows = con.execute("SELECT * FROM budgets ORDER BY name").fetchall()
    return [dict(r) for r in rows]


def delete_budget(budget_id: str) -> None:
    with _conn() as con:
        con.execute("DELETE FROM budgets WHERE id = ?", (budget_id,))


# ---------------------------------------------------------------------------
# Poll cursors — track last successful API poll per provider
# ---------------------------------------------------------------------------

def get_poll_cursor(provider: str) -> dict | None:
    with _conn() as con:
        row = con.execute(
            "SELECT * FROM poll_cursors WHERE provider = ?", (provider,)
        ).fetchone()
    return dict(row) if row else None


def set_poll_cursor(
    provider: str,
    last_polled: datetime,
    since_date: datetime,
    until_date: datetime,
) -> None:
    sql = """
        INSERT INTO poll_cursors (provider, last_polled, since_date, until_date)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(provider) DO UPDATE SET
            last_polled = excluded.last_polled,
            since_date  = excluded.since_date,
            until_date  = excluded.until_date
    """
    with _conn() as con:
        con.execute(sql, (
            provider,
            last_polled.isoformat(),
            since_date.isoformat(),
            until_date.isoformat(),
        ))
