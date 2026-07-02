"""
Foreman MCP server — exposes burn-map analytics as tools over stdio so an
editor agent (e.g. Cursor) can query LLM spend without leaving the editor.

Run:
    .venv/bin/python mcp_server.py

Add to ~/.cursor/mcp.json per README.md. FOREMAN_DB_PATH controls which
foreman.db is read, same as the Streamlit app.

Every tool response gets read back into the calling agent's own context as
input tokens on its next turn — a real cost, invisible to provider billing.
Each call here is logged to mcp_tool_calls (src/db.py) using the same free
token-estimation heuristic foreman_optimizer uses, so that burn is visible
in the app's "MCP TOOL-CALL INPUT BURN" section instead of disappearing.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timedelta
from functools import wraps
from typing import Any, Callable

from mcp.server.fastmcp import FastMCP

from foreman_optimizer.ir import estimate_tokens
from src.analytics import burn_map
from src.analytics.pricing import mcp_reference_model, price_for_model
from src.db import fetch_budgets, get_db_path, init_db, insert_mcp_call

init_db()

mcp = FastMCP("foreman")

# One connection reused for every burn-tracking insert, instead of a fresh
# connect/pragma/commit/close round trip per tool call. Opened lazily (on
# first tool call, not at import) so it resolves FOREMAN_DB_PATH the same
# way every other read/write in this codebase does — at call time, not at
# module-import time — rather than pinning to whatever the env var happened
# to be the moment this module was first imported. FastMCP may run sync
# tools off the main thread, so allow cross-thread use and serialize writes
# with a lock (sqlite3 connections aren't otherwise safe to share across
# threads).
_mcp_call_conn: sqlite3.Connection | None = None
_mcp_call_lock = threading.Lock()


def _get_mcp_call_conn() -> sqlite3.Connection:
    global _mcp_call_conn
    if _mcp_call_conn is None:
        _mcp_call_conn = sqlite3.connect(str(get_db_path()), check_same_thread=False)
        _mcp_call_conn.execute("PRAGMA journal_mode=WAL")
        _mcp_call_conn.execute("PRAGMA foreign_keys=ON")
    return _mcp_call_conn


def _native(obj: Any) -> Any:
    """Recursively unwrap numpy/pandas scalars so tool results are plain
    JSON-safe Python values."""
    if isinstance(obj, dict):
        return {k: _native(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_native(v) for v in obj]
    if hasattr(obj, "item") and not isinstance(obj, (str, bytes)):
        return obj.item()
    return obj


def _track(tool_name: str) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            result = _native(fn(*args, **kwargs))
            call_repr = json.dumps({"args": args, "kwargs": kwargs}, default=str)
            request_tokens = estimate_tokens(call_repr)
            response_tokens = estimate_tokens(json.dumps(result, default=str))
            # estimated_usd here is a point-in-time value for this row's own
            # record; src.analytics.mcp_usage repriced from response_tokens
            # against the *current* reference model for anything aggregated,
            # so a later env var change doesn't leave stale dollars mixed in.
            in_price, _ = price_for_model(mcp_reference_model())
            with _mcp_call_lock:
                insert_mcp_call(
                    tool_name,
                    request_tokens=request_tokens,
                    response_tokens=response_tokens,
                    estimated_usd=response_tokens * in_price / 1_000_000,
                    conn=_get_mcp_call_conn(),
                )
            return result
        return wrapper
    return decorator


def _load(days: int | None = None):
    # `is not None` (not truthiness) — days=0 means "right now", a valid
    # zero-width window, not "no filter".
    since = datetime.utcnow() - timedelta(days=days) if days is not None else None
    return burn_map.load_dataframe(since=since)


@mcp.tool()
@_track("get_key_metrics")
def get_key_metrics(days: int = 30) -> dict:
    """Total cost, token counts, local-absorption %, and entry count for the last N days."""
    return burn_map.key_metrics(_load(days=days))


@mcp.tool()
@_track("get_burn_by_provider")
def get_burn_by_provider(days: int = 30) -> list[dict]:
    """Cost breakdown per provider for the last N days."""
    return burn_map.burn_by_provider(_load(days=days)).to_dict("records")


@mcp.tool()
@_track("get_burn_by_model")
def get_burn_by_model(days: int = 30, limit: int = 10) -> list[dict]:
    """Cost breakdown per model (top `limit`) for the last N days."""
    return burn_map.burn_by_model(_load(days=days), top_n=limit).to_dict("records")


@mcp.tool()
@_track("get_burn_by_class")
def get_burn_by_class(days: int = 30) -> list[dict]:
    """Cost breakdown per workload class (extract/rag/reason/agents/coding), local vs frontier."""
    return burn_map.burn_by_class(_load(days=days)).to_dict("records")


@mcp.tool()
@_track("get_daily_burn")
def get_daily_burn(days: int = 30) -> list[dict]:
    """Day-by-day spend, split local vs frontier, for the last N days."""
    df = burn_map.daily_burn(_load(days=days))
    if df.empty:
        return []
    df = df.copy()
    df["date"] = df["date"].astype(str)
    return df.to_dict("records")


@mcp.tool()
@_track("get_projection")
def get_projection(days_ahead: int = 30) -> dict:
    """Linear spend forecast `days_ahead` days out, from the recent daily average."""
    return burn_map.burn_rate_projection(_load(), days_ahead=days_ahead)


@mcp.tool()
@_track("get_budget_status")
def get_budget_status() -> list[dict]:
    """Each configured budget's spend so far, remaining, and over-threshold status."""
    budgets = fetch_budgets()
    if not budgets:
        return []
    return burn_map.budget_status(_load(), budgets)


@mcp.tool()
@_track("get_top_spenders")
def get_top_spenders(by: str = "model", days: int = 30, limit: int = 10) -> list[dict]:
    """Rank models or teams by total cost. `by` is "model" or "team"."""
    group_col = by.strip().lower()
    if group_col not in ("model", "team"):
        raise ValueError(f'by must be "model" or "team", got {by!r}')
    df = _load(days=days)
    if df.empty:
        return []
    ranked = (
        df.fillna({group_col: "(untagged)"})
        .groupby(group_col)["cost_usd"]
        .sum()
        .nlargest(limit)
        .reset_index()
        .rename(columns={"cost_usd": "total_usd"})
    )
    return ranked.to_dict("records")


if __name__ == "__main__":
    mcp.run(transport="stdio")
