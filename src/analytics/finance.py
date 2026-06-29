"""
Financial analytics for COO/CFO consumption.

Three outputs:
  unit_economics()      — cost per MAU/transaction, LLM as % of revenue
  period_summary()      — MoM spend by team with budget variance
  gl_export_df()        — GL-ready rows mapped to chart of accounts
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd

# ---------------------------------------------------------------------------
# GL account mapping
# Workload classes map to standard COGS/R&D accounts.
# Teams override to R&D when the workload is 'coding' (developer tooling).
# ---------------------------------------------------------------------------

_GL_ACCOUNTS: dict[str, tuple[str, str]] = {
    # workload_class → (account_code, account_name)
    "reason":   ("5100", "COGS — AI Inference: Reasoning"),
    "agents":   ("5110", "COGS — AI Inference: Agents"),
    "extract":  ("5120", "COGS — AI Inference: Data Processing"),
    "rag":      ("5130", "COGS — AI Inference: Knowledge Retrieval"),
    "coding":   ("6200", "R&D — AI Development Tools"),
    "unknown":  ("5190", "COGS — AI Inference: Unclassified"),
}
_LOCAL_GL = ("5200", "COGS — Infrastructure: Local Model Compute")


def unit_economics(
    df: pd.DataFrame,
    mau: int,
    mrr_usd: float,
    transactions: int | None = None,
) -> dict:
    """
    Return LLM cost metrics relative to business volume inputs.

    Parameters
    ----------
    df           : spend DataFrame (standard schema)
    mau          : monthly active users
    mrr_usd      : monthly recurring revenue in USD
    transactions : optional — API calls, orders, completions, etc.
    """
    if df.empty or mrr_usd <= 0 or mau <= 0:
        return {}

    # Normalise to a single calendar month's worth of data
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    span_days = max((df["timestamp"].max() - df["timestamp"].min()).days, 1)
    months = span_days / 30.44

    total_usd      = df["cost_usd"].sum()
    frontier_usd   = df[~df["is_local"]]["cost_usd"].sum()
    monthly_usd    = total_usd / months

    result = {
        "monthly_llm_cost_usd":        round(monthly_usd, 2),
        "cost_per_mau_usd":            round(monthly_usd / mau, 4),
        "llm_pct_of_mrr":              round(monthly_usd / mrr_usd * 100, 2),
        "frontier_pct_of_mrr":         round(frontier_usd / months / mrr_usd * 100, 2),
        # Remaining gross margin assuming LLM is the only COGS line shown here.
        "effective_gross_margin_pct":  round((mrr_usd - monthly_usd) / mrr_usd * 100, 2),
        "monthly_frontier_usd":        round(frontier_usd / months, 2),
        "monthly_absorbed_usd":        round((total_usd - frontier_usd) / months, 2),
        "short_range":                 span_days < 7,
    }

    if transactions and transactions > 0:
        result["cost_per_transaction_usd"] = round(monthly_usd / transactions, 6)

    return result


def period_summary(
    df: pd.DataFrame,
    budgets: list[dict] | None = None,
    n_months: int = 3,
) -> pd.DataFrame:
    """
    Monthly spend by team for the last n_months, with budget variance.

    Returns a DataFrame with columns:
      team, month, spend_usd, budget_usd, variance_usd, variance_pct, status
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["month"] = df["timestamp"].dt.to_period("M")
    # Normalise team names so "Sales", "sales", " sales" group together
    df["team"] = df["team"].fillna("(untagged)").str.strip().str.lower()

    cutoff = pd.Period(datetime.utcnow(), "M") - n_months
    df = df[df["month"] > cutoff]

    summary = (
        df.groupby(["team", "month"])["cost_usd"]
        .sum()
        .reset_index()
        .rename(columns={"cost_usd": "spend_usd"})
    )
    summary["month"] = summary["month"].astype(str)
    summary["budget_usd"] = None
    summary["variance_usd"] = None
    summary["variance_pct"] = None
    summary["status"] = "no budget"

    if budgets:
        team_budgets: dict[str, float] = {}
        for b in budgets:
            team = (b.get("team") or "all").strip().lower()
            amount = float(b["amount_usd"])
            period = b.get("period", "monthly")
            # Prorate daily and weekly budgets to a monthly equivalent so they
            # appear in the variance table alongside monthly budgets.
            if period == "daily":
                amount = amount * 30.44
            elif period == "weekly":
                amount = amount * 4.33
            # "monthly" needs no adjustment
            team_budgets[team] = amount

        def _apply_budget(row):
            team_key = (row["team"] or "").strip().lower()
            budget = team_budgets.get(team_key, team_budgets.get("all"))
            if budget is None:
                return row
            row["budget_usd"]    = budget
            row["variance_usd"]  = round(row["spend_usd"] - budget, 2)
            row["variance_pct"]  = round((row["spend_usd"] - budget) / budget * 100, 1)
            threshold = next(
                (b.get("alert_threshold", 0.8) for b in budgets
                 if (b.get("team") or "all") == (row["team"] or "all")),
                0.8,
            )
            if row["spend_usd"] > budget:
                row["status"] = "over"
            elif row["spend_usd"] / budget >= threshold:
                row["status"] = "at risk"
            else:
                row["status"] = "on track"
            return row

        summary = summary.apply(_apply_budget, axis=1)

    return summary.sort_values(["month", "team"], ascending=[False, True])


def gl_export_df(df: pd.DataFrame, period: str = "month") -> pd.DataFrame:
    """
    Return a GL-ready DataFrame suitable for import into NetSuite, QuickBooks,
    Sage, or any double-entry accounting system.

    period : "month" | "quarter" | "week"

    Columns: period, gl_account, account_name, department, description,
             provider, debit_usd, credit_usd, memo
    """
    if df.empty:
        return pd.DataFrame()

    _FORMULA_CHARS = ("=", "+", "-", "@")

    def _safe(val: str) -> str:
        """Prefix values that would execute as spreadsheet formulas."""
        s = str(val) if val is not None else ""
        return ("'" + s) if s.startswith(_FORMULA_CHARS) else s

    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["team"] = df["team"].fillna("unassigned").replace("", "unassigned")

    freq_map = {"month": "M", "quarter": "Q", "week": "W"}
    freq = freq_map.get(period, "M")
    df["period"] = df["timestamp"].dt.to_period(freq).astype(str)

    rows = []
    grouped = df.groupby(["period", "workload_class", "team", "provider", "is_local"])

    for (period_str, wc, team, provider, is_local), grp in grouped:
        if is_local:
            code, name = _LOCAL_GL
        else:
            code, name = _GL_ACCOUNTS.get(wc, _GL_ACCOUNTS["unknown"])

        amount = round(grp["cost_usd"].sum(), 4)
        if amount == 0:
            continue

        dept = team or "unassigned"
        rows.append({
            "period":       period_str,
            "gl_account":   code,
            "account_name": name,
            "department":   _safe(dept),
            "provider":     _safe(provider),
            "workload":     _safe(wc),
            "debit_usd":    amount,
            "credit_usd":   0.0,
            "memo":         f"LLM spend — {_safe(provider)} / {_safe(wc)} / {_safe(dept)}",
        })

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(["period", "gl_account", "department"])
