"""
Executive Brief — the first screen a COO/CFO sees.

One screen, four questions, no scrolling required for the answer:
  1. What are we spending?        (KPI row)
  2. Is it under control?         (trend + budget health)
  3. What should we do about it?  (top actions, ranked by $)
  4. The bottom line              (plain-English narrative)
"""
from __future__ import annotations

import html as _html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics.burn_map import daily_burn
from src.analytics.executive import build_brief
from src.analytics.intelligence import generate_report
from src.ui.theme import CLAY, PLOTLY_LAYOUT, PLOTLY_YAXIS, SAGE


def render(df: pd.DataFrame, budgets_status: list[dict]) -> None:
    if df.empty:
        st.info(
            "No spend data yet. Upload a billing export in **Bill Analyzer** "
            "or connect keys in **Live API** — the brief builds itself from there."
        )
        return

    report = generate_report(df)
    brief = build_brief(df, budgets_status, report)
    metrics = brief["metrics"]
    proj = brief["projection"]
    wow = brief["wow"]

    # ---- 1 · What are we spending? ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Period spend", f"${metrics['total_cost_usd']:,.2f}")

    delta = wow["delta_pct"]
    c2.metric(
        "Daily run-rate",
        f"${proj['daily_avg']:,.2f}",
        delta=f"{delta:+.0f}% WoW" if delta is not None else None,
        delta_color="inverse",  # spend going up is bad → red
    )
    c3.metric(
        "30-day forecast",
        f"${proj['projected_total']:,.2f}",
        help="Linear extrapolation of the period's average daily spend.",
    )
    c4.metric(
        "Identified savings",
        f"${brief['total_savings_usd']:,.2f}",
        delta=(
            f"~${brief['savings_30d_usd']:,.0f}/mo "
            f"({min(brief['savings_30d_usd'] / proj['projected_total'] * 100, 100):.0f}% of forecast)"
            if proj["projected_total"] > 0 and brief["total_savings_usd"] > 0
            else None
        ),
        delta_color="off",
        help=(
            "Sum of the actionable proposals in Spend Intelligence for this "
            "period, with a 30-day equivalent for comparison to the forecast."
        ),
    )

    # ---- The bottom line ----
    st.markdown('<div class="foreman-section">THE BOTTOM LINE</div>', unsafe_allow_html=True)
    for line in brief["narrative"]:
        st.markdown(f"- {line}")

    # ---- 2 · Is it under control? ----
    col_trend, col_budget = st.columns([3, 2])

    with col_trend:
        st.markdown('<div class="foreman-section">DAILY SPEND</div>', unsafe_allow_html=True)
        daily = daily_burn(df)
        if not daily.empty:
            fig = go.Figure()
            fig.add_scatter(
                name="API / external",
                x=daily["date"].astype(str),
                y=daily["frontier_usd"],
                mode="lines",
                stackgroup="burn",
                line=dict(color=CLAY, width=1),
            )
            fig.add_scatter(
                name="Local / on-prem",
                x=daily["date"].astype(str),
                y=daily["absorbed_usd"],
                mode="lines",
                stackgroup="burn",
                line=dict(color=SAGE, width=1),
            )
            fig.update_layout(
                **PLOTLY_LAYOUT,
                height=260,
                yaxis=dict(**PLOTLY_YAXIS, tickprefix="$"),
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_budget:
        st.markdown('<div class="foreman-section">BUDGET HEALTH</div>', unsafe_allow_html=True)
        budgets = brief["budgets"]
        if budgets["total"] == 0:
            st.caption("No budgets set — add one in **Settings** to get alerts here.")
        else:
            on_track = budgets["total"] - budgets["over"] - budgets["at_risk"]
            b1, b2, b3 = st.columns(3)
            b1.metric("On track", on_track)
            b2.metric("At risk", budgets["at_risk"])
            b3.metric("Over", budgets["over"])
            worst = budgets["worst"]
            if worst:
                st.markdown(
                    f"Tightest: **{_html.escape(worst['name'])}** — "
                    f"${worst['spent_usd']:,.2f} of ${worst['amount_usd']:,.2f} "
                    f"({worst['pct_used']:.0%})"
                )
                st.progress(float(worst["pct_used"]))
        top = brief["top_cost_center"]
        if top:
            model, spend = top
            share = spend / metrics["total_cost_usd"] * 100 if metrics["total_cost_usd"] else 0
            st.caption(
                f"Largest cost center: `{_html.escape(model)}` — "
                f"${spend:,.2f} ({share:.0f}% of period spend)"
            )

    # ---- 3 · What should we do about it? ----
    st.markdown(
        '<div class="foreman-section">TOP ACTIONS — RANKED BY $ IMPACT</div>',
        unsafe_allow_html=True,
    )
    if not brief["top_actions"]:
        st.success("No material savings opportunities detected — spend looks efficient.")
    else:
        for i, p in enumerate(brief["top_actions"], start=1):
            with st.container(border=True):
                col_t, col_s = st.columns([4, 1])
                col_t.markdown(f"**{i} · {p.title}**")
                col_s.metric("Est. savings", f"${p.estimated_savings_usd:,.2f}")
                col_t.caption(p.action)
        st.caption(
            "Full findings, guardrails, and routing policies live in "
            "**Spend Intelligence**. Nothing auto-applies without a quality-floor hold."
        )

    # ---- How to read these numbers ----
    st.markdown(
        '<div class="foreman-section">HOW WE RANK — $ PER TASK, NOT PER TOKEN</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        "Price per token is the wrong unit; what matters is **dollars per task "
        "solved**. A lower token price can be the more expensive choice — a "
        "premium model that solves in fewer, less verbose turns often costs "
        "less per solved task. Two levers move real cost more than token price "
        "does:  \n"
        "1. **Prompt caching** — in agentic loops the prompt is mostly stable "
        "across calls, so hit rates approach 90%. Verify it's actually on: "
        "silently invalidated caches eat the discount you think you're getting "
        "(we watch for this above).  \n"
        "2. **Model choice by total tokens to solution** — you pay for every "
        "turn until the task is done, not for any single call.  \n\n"
        "The savings above are list-price estimates. Before any of them is "
        "applied, it must clear a golden-eval benchmarked on "
        "**cost-per-solved-task against your own workload, caching measured** — "
        "never on published token prices."
    )
