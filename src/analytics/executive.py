"""
Executive Brief analytics — the one-screen answer for a COO/CFO.

Everything here is deterministic and derived from data the other analytics
modules already compute: no LLM call, no randomness, so the brief is free to
generate and identical on every rerun. The UI layer (src/ui/exec_brief.py)
renders the dict this module returns; keeping the logic here keeps it
testable without Streamlit.
"""
from __future__ import annotations

import pandas as pd

from src.analytics.burn_map import burn_rate_projection, key_metrics
from src.analytics.intelligence import IntelligenceReport

_WOW_DAYS = 7


def week_over_week(df: pd.DataFrame) -> dict:
    """Spend for the trailing 7 days vs the 7 days before that, anchored to
    the newest timestamp in the filtered data (not wall-clock now, so a
    historical date-range selection still produces a sensible comparison)."""
    if df.empty:
        return {"recent_usd": 0.0, "prior_usd": 0.0, "delta_pct": None}
    ts = pd.to_datetime(df["timestamp"])
    anchor = ts.max()
    cutoff = anchor - pd.Timedelta(days=_WOW_DAYS)
    prior_cutoff = cutoff - pd.Timedelta(days=_WOW_DAYS)
    recent = float(df.loc[ts > cutoff, "cost_usd"].sum())
    prior = float(df.loc[(ts > prior_cutoff) & (ts <= cutoff), "cost_usd"].sum())
    delta = ((recent - prior) / prior * 100) if prior > 0 else None
    return {"recent_usd": recent, "prior_usd": prior, "delta_pct": delta}


def top_cost_center(df: pd.DataFrame) -> tuple[str, float] | None:
    """(model, spend) for the single most expensive model in the period."""
    if df.empty:
        return None
    by_model = df.groupby("model")["cost_usd"].sum()
    if by_model.empty or by_model.max() <= 0:
        return None
    return str(by_model.idxmax()), float(by_model.max())


def budget_health(budgets_status: list[dict]) -> dict:
    over = [b for b in budgets_status if b["spent_usd"] > b["amount_usd"]]
    at_risk = [
        b for b in budgets_status
        if b["over_threshold"] and b["spent_usd"] <= b["amount_usd"]
    ]
    worst = None
    if budgets_status:
        worst = max(budgets_status, key=lambda b: b["pct_used"])
    return {
        "total": len(budgets_status),
        "over": len(over),
        "at_risk": len(at_risk),
        "worst": worst,
    }


def _narrative(
    metrics: dict,
    proj: dict,
    wow: dict,
    budgets: dict,
    report: IntelligenceReport,
    span_days: int,
) -> list[str]:
    """Plain-English bottom line, most important sentence first."""
    lines: list[str] = []

    lines.append(
        f"You spent **${metrics['total_cost_usd']:,.2f}** over the last "
        f"{span_days} day{'s' if span_days != 1 else ''} — a daily run-rate of "
        f"**${proj['daily_avg']:,.2f}**, tracking to "
        f"**${proj['projected_total']:,.2f}** over the next 30 days."
    )

    delta = wow["delta_pct"]
    if delta is not None and abs(delta) >= 15:
        direction = "accelerated" if delta > 0 else "declined"
        lines.append(
            f"Spend {direction} **{abs(delta):.0f}%** week-over-week "
            f"(${wow['recent_usd']:,.2f} vs ${wow['prior_usd']:,.2f})."
        )

    if budgets["over"] > 0:
        lines.append(
            f"**{budgets['over']} of {budgets['total']} budgets are over** for "
            "the current period — details below."
        )
    elif budgets["at_risk"] > 0:
        lines.append(
            f"{budgets['at_risk']} of {budgets['total']} budgets have crossed "
            "their alert threshold; none are over yet."
        )
    elif budgets["total"] > 0:
        lines.append(f"All {budgets['total']} budgets are on track.")

    savings = report.total_potential_savings_usd
    if savings > 0 and proj["projected_total"] > 0 and span_days > 0:
        # Proposals measure savings over the observed window — normalize to a
        # 30-day equivalent before comparing against the 30-day forecast.
        savings_30d = savings / span_days * 30
        pct = min(savings_30d / proj["projected_total"] * 100, 100)
        top = report.proposals[0].title if report.proposals else None
        sentence = (
            f"**${savings:,.2f}** in savings is identified and actionable over "
            f"this period — roughly **${savings_30d:,.2f}/month** "
            f"({pct:.0f}% of the 30-day forecast)"
        )
        sentence += f". Start with: *{top}*." if top else "."
        lines.append(sentence)

    if metrics["absorbed_cost_usd"] > 0:
        lines.append(
            f"Local models absorbed **{metrics['local_pct']:.0f}%** of total "
            "workload cost that would otherwise be API spend."
        )

    return lines


def build_brief(
    df: pd.DataFrame,
    budgets_status: list[dict],
    report: IntelligenceReport,
) -> dict:
    """Assemble every number the Executive Brief screen shows."""
    metrics = key_metrics(df)
    proj = burn_rate_projection(df)
    wow = week_over_week(df)
    budgets = budget_health(budgets_status)

    span_days = 0
    if not df.empty:
        ts = pd.to_datetime(df["timestamp"])
        span_days = max((ts.max() - ts.min()).days, 1)

    top_actions = report.proposals[:3]

    savings = report.total_potential_savings_usd
    savings_30d = (savings / span_days * 30) if span_days > 0 else savings

    return {
        "metrics": metrics,
        "projection": proj,
        "wow": wow,
        "budgets": budgets,
        "top_cost_center": top_cost_center(df),
        "top_actions": top_actions,
        "total_savings_usd": savings,
        "savings_30d_usd": savings_30d,
        "narrative": _narrative(metrics, proj, wow, budgets, report, span_days),
        "span_days": span_days,
    }
