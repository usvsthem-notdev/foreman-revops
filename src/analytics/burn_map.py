"""
Burn Map analytics — aggregation and series generation for charts.
"""
from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd

from src.db import fetch_entries


def load_dataframe(
    *,
    provider: Optional[str] = None,
    team: Optional[str] = None,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
) -> pd.DataFrame:
    rows = fetch_entries(provider=provider, team=team, since=since, until=until)
    if not rows:
        return _empty_df()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["cost_usd"] = pd.to_numeric(df["cost_usd"], errors="coerce").fillna(0)
    df["input_tokens"] = pd.to_numeric(df["input_tokens"], errors="coerce").fillna(0).astype(int)
    df["output_tokens"] = pd.to_numeric(df["output_tokens"], errors="coerce").fillna(0).astype(int)
    df["reasoning_tokens"] = pd.to_numeric(df["reasoning_tokens"], errors="coerce").fillna(0).astype(int)
    df["is_local"] = df["is_local"].astype(bool)
    return df


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "id", "timestamp", "provider", "model", "workload_class",
        "input_tokens", "output_tokens", "reasoning_tokens",
        "cost_usd", "is_local", "team", "feature", "notes", "source",
    ])


def daily_burn(df: pd.DataFrame) -> pd.DataFrame:
    """Daily cost totals, split by is_local."""
    if df.empty:
        return df
    df = df.copy()
    df["date"] = df["timestamp"].dt.date
    pivot = (
        df.groupby(["date", "is_local"])["cost_usd"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    pivot.columns.name = None
    pivot = pivot.rename(columns={True: "absorbed_usd", False: "frontier_usd"})
    if "absorbed_usd" not in pivot.columns:
        pivot["absorbed_usd"] = 0.0
    if "frontier_usd" not in pivot.columns:
        pivot["frontier_usd"] = 0.0
    pivot["total_usd"] = pivot["absorbed_usd"] + pivot["frontier_usd"]
    return pivot.sort_values("date")


def burn_by_class(df: pd.DataFrame) -> pd.DataFrame:
    """Cost per workload class, split absorbed vs frontier."""
    if df.empty:
        return df
    pivot = (
        df.groupby(["workload_class", "is_local"])["cost_usd"]
        .sum()
        .unstack(fill_value=0)
        .reset_index()
    )
    pivot.columns.name = None
    pivot = pivot.rename(columns={True: "absorbed_usd", False: "frontier_usd"})
    if "absorbed_usd" not in pivot.columns:
        pivot["absorbed_usd"] = 0.0
    if "frontier_usd" not in pivot.columns:
        pivot["frontier_usd"] = 0.0
    return pivot


def burn_by_model(df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.groupby("model")["cost_usd"]
        .sum()
        .nlargest(top_n)
        .reset_index()
        .rename(columns={"cost_usd": "total_usd"})
    )


def burn_by_provider(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    return (
        df.groupby("provider")["cost_usd"]
        .sum()
        .reset_index()
        .rename(columns={"cost_usd": "total_usd"})
        .sort_values("total_usd", ascending=False)
    )


def cumulative_burn(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    daily = daily_burn(df)
    daily["cumulative_usd"] = daily["total_usd"].cumsum()
    return daily


def key_metrics(df: pd.DataFrame) -> dict:
    if df.empty:
        return {
            "total_cost_usd": 0.0,
            "frontier_cost_usd": 0.0,
            "absorbed_cost_usd": 0.0,
            "local_pct": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "total_reasoning_tokens": 0,
            "entry_count": 0,
            "cost_per_1k_tokens": 0.0,
        }
    total = df["cost_usd"].sum()
    absorbed = df.loc[df["is_local"], "cost_usd"].sum()
    frontier = df.loc[~df["is_local"], "cost_usd"].sum()
    total_tok = (
        df["input_tokens"].sum()
        + df["output_tokens"].sum()
        + df["reasoning_tokens"].sum()
    )
    return {
        "total_cost_usd": total,
        "frontier_cost_usd": frontier,
        "absorbed_cost_usd": absorbed,
        "local_pct": (absorbed / total * 100) if total > 0 else 0.0,
        "total_input_tokens": int(df["input_tokens"].sum()),
        "total_output_tokens": int(df["output_tokens"].sum()),
        "total_reasoning_tokens": int(df["reasoning_tokens"].sum()),
        "entry_count": len(df),
        "cost_per_1k_tokens": (total / total_tok * 1000) if total_tok > 0 else 0.0,
    }


def burn_rate_projection(df: pd.DataFrame, days_ahead: int = 30) -> dict:
    """Linear extrapolation of daily spend."""
    if df.empty or len(df) < 2:
        return {"projected_total": 0.0, "daily_avg": 0.0, "days_of_data": 0}
    daily = daily_burn(df)
    avg = daily["total_usd"].mean()
    return {
        "projected_total": avg * days_ahead,
        "daily_avg": avg,
        "days_of_data": len(daily),
    }


def budget_status(df: pd.DataFrame, budgets: list[dict]) -> list[dict]:
    """Return each budget with spend so far in its current period."""
    now = datetime.utcnow()
    results = []
    for b in budgets:
        period = b["period"]
        if period == "daily":
            since = now.replace(hour=0, minute=0, second=0, microsecond=0)
        elif period == "weekly":
            since = now - timedelta(days=now.weekday())
            since = since.replace(hour=0, minute=0, second=0, microsecond=0)
        else:  # monthly
            since = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

        mask = df["timestamp"] >= pd.Timestamp(since)
        if b.get("provider"):
            mask &= df["provider"] == b["provider"]
        if b.get("team"):
            mask &= df["team"] == b["team"]

        spent = df.loc[mask, "cost_usd"].sum() if not df.empty else 0.0
        pct = (spent / b["amount_usd"]) if b["amount_usd"] > 0 else 0.0
        results.append({
            **b,
            "spent_usd": spent,
            "remaining_usd": max(0, b["amount_usd"] - spent),
            "pct_used": min(pct, 1.0),
            "over_threshold": pct >= b.get("alert_threshold", 0.8),
        })
    return results
