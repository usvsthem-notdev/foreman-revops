"""
Finance tab — COO/CFO view of LLM burn.

Three sections:
  01 · UNIT ECONOMICS  — cost/MAU, LLM% of MRR, gross margin impact
  02 · PERIOD VARIANCE — MoM spend by team vs budget
  03 · GL EXPORT       — NetSuite/QuickBooks-ready CSV download
"""
from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from src.analytics.finance import gl_export_df, period_summary, unit_economics


def render(df: pd.DataFrame, budgets_raw: list[dict]) -> None:
    if df.empty:
        st.info("No spend data in the selected range. Adjust the date filter or import data.")
        return

    # ── 01 · UNIT ECONOMICS ────────────────────────────────────────────────
    st.markdown('<div class="foreman-section">01 · UNIT ECONOMICS</div>', unsafe_allow_html=True)
    st.caption("Normalised to a single calendar month across the selected date range.")

    col_inputs, col_metrics = st.columns([1, 2])

    with col_inputs:
        mau = st.number_input("Monthly Active Users (MAU)", min_value=1, value=1000, step=100)
        mrr = st.number_input("MRR (USD)", min_value=0.0, value=50_000.0, step=1_000.0, format="%.2f")
        txn = st.number_input(
            "Transactions / month (optional)",
            min_value=0, value=0, step=1,
            help="API calls, completions, orders — leave 0 to skip cost-per-transaction.",
        )

    ue = unit_economics(df, mau=int(mau), mrr_usd=float(mrr), transactions=int(txn) or None)

    with col_metrics:
        if not ue:
            st.warning("Could not compute unit economics — check MAU and MRR inputs.")
        else:
            if ue.get("short_range"):
                st.warning("Date range is under 7 days — monthly projections are unreliable.")

            r1c1, r1c2, r1c3 = st.columns(3)
            r1c1.metric("Monthly LLM cost",       f"${ue['monthly_llm_cost_usd']:,.2f}")
            r1c2.metric("Cost / MAU",              f"${ue['cost_per_mau_usd']:.4f}")
            r1c3.metric("LLM % of MRR",            f"{ue['llm_pct_of_mrr']:.2f}%")

            r2c1, r2c2, r2c3 = st.columns(3)
            r2c1.metric("Effective gross margin",  f"{ue['effective_gross_margin_pct']:.2f}%",
                        help="(MRR − monthly LLM cost) / MRR — excludes all other COGS.")
            r2c2.metric("Frontier spend / mo",     f"${ue['monthly_frontier_usd']:,.2f}")
            r2c3.metric("Absorbed (local) / mo",   f"${ue['monthly_absorbed_usd']:,.2f}")

            if "cost_per_transaction_usd" in ue:
                st.metric("Cost / transaction", f"${ue['cost_per_transaction_usd']:.6f}")

    # ── 02 · PERIOD VARIANCE ───────────────────────────────────────────────
    st.markdown('<div class="foreman-section">02 · PERIOD VARIANCE</div>', unsafe_allow_html=True)

    n_months = st.slider("Months to show", min_value=1, max_value=12, value=3)
    summary = period_summary(df, budgets=budgets_raw if budgets_raw else None, n_months=n_months)

    if summary.empty:
        st.info("No team spend data for the selected period.")
    else:
        def _status_colour(val: str) -> str:
            colours = {"over": "#FF4B4B", "at risk": "#FFA500", "on track": "#21C354", "no budget": "#8A9BB0"}
            return f"color: {colours.get(val, '#8A9BB0')}"

        styled = (
            summary.style
            .format({
                "spend_usd":    "${:,.2f}",
                "budget_usd":   lambda v: f"${v:,.2f}" if pd.notna(v) else "—",
                "variance_usd": lambda v: f"${v:,.2f}" if pd.notna(v) else "—",
                "variance_pct": lambda v: f"{v:.1f}%" if pd.notna(v) else "—",
            })
            .map(_status_colour, subset=["status"])
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

    # ── 03 · GL EXPORT ─────────────────────────────────────────────────────
    st.markdown('<div class="foreman-section">03 · GL EXPORT</div>', unsafe_allow_html=True)
    st.caption("Double-entry rows mapped to COGS (5xxx) and R&D (6xxx) accounts. "
               "Compatible with NetSuite, QuickBooks, Sage, and Xero.")

    gl_period = st.radio("Aggregation period", ["month", "quarter", "week"], horizontal=True, key="gl_period_radio")
    gl_df = gl_export_df(df, period=gl_period)

    if gl_df.empty:
        st.info("No GL rows to export.")
    else:
        st.dataframe(gl_df, use_container_width=True, hide_index=True)

        csv_bytes = gl_df.to_csv(index=False).encode()
        st.download_button(
            label="Download GL CSV",
            data=csv_bytes,
            file_name=f"foreman_gl_export_{gl_period}.csv",
            mime="text/csv",
        )
