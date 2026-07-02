"""
Burn Map analytics — aggregation and series generation for charts.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

from src.analytics.pricing import cache_multipliers, price_for_model
from src.db import fetch_entries


def load_dataframe(
    *,
    provider: str | None = None,
    team: str | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
) -> pd.DataFrame:
    rows = fetch_entries(provider=provider, team=team, since=since, until=until)
    if not rows:
        return _empty_df()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["cost_usd"] = pd.to_numeric(df["cost_usd"], errors="coerce").fillna(0)
    df["input_tokens"] = pd.to_numeric(df["input_tokens"], errors="coerce").fillna(0).astype(int)
    df["output_tokens"] = pd.to_numeric(df["output_tokens"], errors="coerce").fillna(0).astype(int)
    df["reasoning_tokens"] = (
        pd.to_numeric(df["reasoning_tokens"], errors="coerce").fillna(0).astype(int)
    )
    df["cache_read_tokens"] = (
        pd.to_numeric(df["cache_read_tokens"], errors="coerce").fillna(0).astype(int)
    )
    df["cache_creation_tokens"] = (
        pd.to_numeric(df["cache_creation_tokens"], errors="coerce").fillna(0).astype(int)
    )
    df["is_local"] = df["is_local"].astype(bool)
    return df


def _empty_df() -> pd.DataFrame:
    return pd.DataFrame(columns=[
        "id", "timestamp", "provider", "model", "workload_class",
        "input_tokens", "output_tokens", "reasoning_tokens",
        "cache_read_tokens", "cache_creation_tokens",
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


def _price_maps(df: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Resolve (in_price, out_price, cache_read_mult, cache_creation_mult) as
    per-row Series, priced once per unique model rather than once per row.
    Shared by cost_by_axis/cache_savings_total so both don't independently
    rebuild identical per-model lookup dicts on every key_metrics() call."""
    prices = {m: price_for_model(m) for m in df["model"].unique()}
    cache_mults = {m: cache_multipliers(m) for m in df["model"].unique()}
    in_price = df["model"].map(lambda m: prices[m][0])
    out_price = df["model"].map(lambda m: prices[m][1])
    read_mult = df["model"].map(lambda m: cache_mults[m][0])
    creation_mult = df["model"].map(lambda m: cache_mults[m][1])
    return in_price, out_price, read_mult, creation_mult


def _clamped_cache_tokens(df: pd.DataFrame) -> tuple[pd.Series, pd.Series]:
    """Cap cache_read/cache_creation at input_tokens defensively — the same
    subset invariant SpendEntry's validator enforces on write, covering rows
    that were persisted before that validator existed. Without this, bad
    data (cache tokens exceeding input_tokens) gets counted at full
    magnitude on top of a zero-clamped fresh_input, wildly overweighting the
    input axis."""
    cache_read = df["cache_read_tokens"].clip(lower=0, upper=df["input_tokens"])
    remaining = (df["input_tokens"] - cache_read).clip(lower=0)
    cache_creation = df["cache_creation_tokens"].clip(lower=0, upper=remaining)
    return cache_read, cache_creation


def cost_by_axis(df: pd.DataFrame) -> tuple[float, float]:
    """Split total cost_usd into (input_usd, output_usd) using per-model pricing.

    Reasoning tokens are folded into the output leg. cache_read_tokens/
    cache_creation_tokens (a subset of input_tokens) are weighted at their
    real discounted/premium rate rather than full input price, so cache-heavy
    entries don't overstate the input axis. Prices are resolved once per
    unique model (not per row) and the split runs as vectorized pandas ops —
    key_metrics() calls this on every Burn Map render, so a per-row Python
    loop would degrade with live-polling data volumes. Mirrors
    pricing.split_cost's per-row semantics exactly, including its zero-cost
    and zero-token-weight fallbacks (see tests/test_burn_map.py)."""
    if df.empty:
        return 0.0, 0.0

    in_price, out_price, read_mult, creation_mult = _price_maps(df)
    cache_read, cache_creation = _clamped_cache_tokens(df)

    fresh_input = (df["input_tokens"] - cache_read - cache_creation).clip(lower=0)
    in_weight = (
        fresh_input * in_price
        + cache_read * in_price * read_mult
        + cache_creation * in_price * creation_mult
    )
    out_weight = (df["output_tokens"] + df["reasoning_tokens"]) * out_price
    total_weight = in_weight + out_weight

    has_cost = df["cost_usd"] > 0
    has_weight = has_cost & (total_weight > 0)

    input_usd = pd.Series(0.0, index=df.index)
    input_usd[has_weight] = (
        df.loc[has_weight, "cost_usd"] * in_weight[has_weight] / total_weight[has_weight]
    )
    # cost > 0 but no token weight to split on -> credit the whole amount to
    # input, matching split_cost's zero-weight fallback.
    zero_weight_with_cost = has_cost & ~has_weight
    input_usd[zero_weight_with_cost] = df.loc[zero_weight_with_cost, "cost_usd"]

    output_usd = df["cost_usd"].where(has_cost, 0.0) - input_usd
    return float(input_usd.sum()), float(output_usd.sum())


def cache_savings_total(df: pd.DataFrame) -> float:
    """Total $ saved by cache_read_tokens landing at the discounted cache
    rate instead of full fresh-input price. Vectorized version of
    pricing.cache_savings_usd for the same reason cost_by_axis is."""
    if df.empty:
        return 0.0
    in_price, _, read_mult, _ = _price_maps(df)
    cache_read, _ = _clamped_cache_tokens(df)
    full_price_usd = cache_read * in_price / 1_000_000
    saved = full_price_usd * (1 - read_mult)
    return float(saved.sum())


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
            "input_cost_usd": 0.0,
            "output_cost_usd": 0.0,
            "input_token_share": 0.0,
            "input_cost_share": 0.0,
            "total_cache_read_tokens": 0,
            "total_cache_creation_tokens": 0,
            "cache_hit_rate": 0.0,
            "cache_savings_usd": 0.0,
        }
    total = df["cost_usd"].sum()
    absorbed = df.loc[df["is_local"], "cost_usd"].sum()
    frontier = df.loc[~df["is_local"], "cost_usd"].sum()
    total_input_tok = int(df["input_tokens"].sum())
    total_output_tok = int(df["output_tokens"].sum())
    total_reasoning_tok = int(df["reasoning_tokens"].sum())
    total_cache_read_tok = int(df["cache_read_tokens"].sum())
    total_cache_creation_tok = int(df["cache_creation_tokens"].sum())
    total_tok = total_input_tok + total_output_tok + total_reasoning_tok
    input_usd, output_usd = cost_by_axis(df)
    return {
        "total_cost_usd": total,
        "frontier_cost_usd": frontier,
        "absorbed_cost_usd": absorbed,
        "local_pct": (absorbed / total * 100) if total > 0 else 0.0,
        "total_input_tokens": total_input_tok,
        "total_output_tokens": total_output_tok,
        "total_reasoning_tokens": total_reasoning_tok,
        "entry_count": len(df),
        "cost_per_1k_tokens": (total / total_tok * 1000) if total_tok > 0 else 0.0,
        "input_cost_usd": input_usd,
        "output_cost_usd": output_usd,
        "input_token_share": (total_input_tok / total_tok * 100) if total_tok > 0 else 0.0,
        "input_cost_share": (input_usd / total * 100) if total > 0 else 0.0,
        "total_cache_read_tokens": total_cache_read_tok,
        "total_cache_creation_tokens": total_cache_creation_tok,
        "cache_hit_rate": (
            (total_cache_read_tok / total_input_tok * 100) if total_input_tok > 0 else 0.0
        ),
        "cache_savings_usd": cache_savings_total(df),
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
