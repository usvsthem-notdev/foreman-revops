"""
MCP tool-call usage analytics.

Separate from src/analytics/burn_map.py on purpose: mcp_tool_calls rows are
not provider spend, they're an *estimate* of the input tokens an MCP tool
response adds to the calling agent's own context on its next turn. Same
free/local token-estimation approach as foreman_optimizer (see mcp_server.py),
just aggregated the same way the burn map aggregates real spend.

Dollar figures are always recomputed here from the stored `response_tokens`
(a durable fact) against the *current* FOREMAN_MCP_REFERENCE_MODEL, rather
than summed from the `estimated_usd` column each row was written with — that
column is priced at insert time, so summing it directly would silently mix
old- and new-reference-model dollars in the same aggregate the moment
FOREMAN_MCP_REFERENCE_MODEL changes.
"""
from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.analytics.pricing import mcp_reference_model, price_for_model
from src.db import fetch_mcp_calls


def _priced_usd(response_tokens: pd.Series) -> pd.Series:
    in_price, _ = price_for_model(mcp_reference_model())
    return response_tokens * in_price / 1_000_000


def load_dataframe(since: datetime | None = None) -> pd.DataFrame:
    rows = fetch_mcp_calls(since=since)
    if not rows:
        return _empty_df()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["request_tokens"] = (
        pd.to_numeric(df["request_tokens"], errors="coerce").fillna(0).astype(int)
    )
    df["response_tokens"] = (
        pd.to_numeric(df["response_tokens"], errors="coerce").fillna(0).astype(int)
    )
    df["estimated_usd"] = pd.to_numeric(df["estimated_usd"], errors="coerce").fillna(0.0)
    return df


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "id", "timestamp", "tool", "request_tokens", "response_tokens", "estimated_usd",
    ])


def key_metrics(df: pd.DataFrame) -> dict:
    """total_input_tokens = tool *responses* only — that's what gets read back
    into the calling agent's context as input on its next turn. request_tokens
    (the size of the tool call itself) is tracked separately, not folded in,
    to keep this number an honest read on the "high input burn" concern."""
    if df.empty:
        return {
            "call_count": 0,
            "total_input_tokens": 0,
            "total_request_tokens": 0,
            "total_estimated_usd": 0.0,
            "avg_response_tokens": 0.0,
        }
    return {
        "call_count": len(df),
        "total_input_tokens": int(df["response_tokens"].sum()),
        "total_request_tokens": int(df["request_tokens"].sum()),
        "total_estimated_usd": float(_priced_usd(df["response_tokens"]).sum()),
        "avg_response_tokens": float(df["response_tokens"].mean()),
    }


def by_tool(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if df.empty:
        return df
    priced = df.assign(estimated_usd=_priced_usd(df["response_tokens"]))
    return (
        priced.groupby("tool")
        .agg(
            calls=("tool", "count"),
            input_tokens=("response_tokens", "sum"),
            estimated_usd=("estimated_usd", "sum"),
        )
        .reset_index()
        .sort_values("input_tokens", ascending=False)
        .head(top_n)
    )


def daily_input_tokens(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    daily = df.assign(estimated_usd=_priced_usd(df["response_tokens"]))
    daily["date"] = daily["timestamp"].dt.date
    return (
        daily.groupby("date")
        .agg(input_tokens=("response_tokens", "sum"), estimated_usd=("estimated_usd", "sum"))
        .reset_index()
        .sort_values("date")
    )
