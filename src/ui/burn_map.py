"""
Burn Map page — live spend by class, provider, model, and over time.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics.burn_map import (
    burn_by_class,
    burn_by_model,
    burn_by_provider,
    burn_rate_projection,
    cumulative_burn,
    key_metrics,
)
from src.ui.theme import CLAY, PLOTLY_COLORS, PLOTLY_LAYOUT, PLOTLY_YAXIS, SAGE, SLATE


def render(df: pd.DataFrame, budgets_status: list[dict]) -> None:
    metrics = key_metrics(df)

    # ---- KPI row ----
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Spend", f"${metrics['total_cost_usd']:,.2f}")
    c2.metric(
        "Frontier Spend",
        f"${metrics['frontier_cost_usd']:,.2f}",
        delta=(
            f"-${metrics['absorbed_cost_usd']:,.2f} absorbed"
            if metrics["absorbed_cost_usd"] > 0
            else None
        ),
        delta_color="inverse",
    )
    c3.metric("Local Absorbed", f"{metrics['local_pct']:.1f}%")
    c4.metric("Entries", f"{metrics['entry_count']:,}")

    if df.empty:
        st.info("No spend data yet. Upload a billing export or add entries manually.")
        return

    # ---- Budget progress ----
    if budgets_status:
        st.markdown('<div class="foreman-section">BUDGETS</div>', unsafe_allow_html=True)
        for b in budgets_status:
            label = (
                f"{b['name']}  ·  {b['period']}"
                f"  ·  ${b['spent_usd']:,.2f} / ${b['amount_usd']:,.2f}"
            )
            color = "🔴" if b["over_threshold"] else "🟢"
            st.markdown(f"{color} **{label}**")
            st.progress(float(b["pct_used"]))

    # ---- Burn by workload class (FIG. 03 main chart) ----
    st.markdown(
        '<div class="foreman-section">BURN MAP — LIVE SPEND BY CLASS</div>',
        unsafe_allow_html=True,
    )
    class_df = burn_by_class(df)
    if not class_df.empty:
        fig = go.Figure()
        fig.add_bar(
            name="Absorbed locally",
            x=class_df["workload_class"],
            y=class_df["absorbed_usd"],
            marker_color=SAGE,
        )
        fig.add_bar(
            name="Frontier spend",
            x=class_df["workload_class"],
            y=class_df["frontier_usd"],
            marker_color=CLAY,
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            barmode="stack",
            height=320,
            xaxis_title="Workload class",
            yaxis=dict(**PLOTLY_YAXIS, title="Cost (USD)", tickprefix="$"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("**Sage** = absorbed locally  ·  **Clay** = frontier spend")

    # ---- Daily burn + cumulative ----
    st.markdown('<div class="foreman-section">DAILY BURN</div>', unsafe_allow_html=True)
    cum_df = cumulative_burn(df)
    if not cum_df.empty:
        fig2 = go.Figure()
        fig2.add_bar(
            name="Absorbed",
            x=cum_df["date"].astype(str),
            y=cum_df["absorbed_usd"],
            marker_color=SAGE,
            opacity=0.85,
        )
        fig2.add_bar(
            name="Frontier",
            x=cum_df["date"].astype(str),
            y=cum_df["frontier_usd"],
            marker_color=CLAY,
            opacity=0.85,
        )
        fig2.add_scatter(
            name="Cumulative",
            x=cum_df["date"].astype(str),
            y=cum_df["cumulative_usd"],
            mode="lines",
            line=dict(color=SLATE, width=1.5, dash="dot"),
            yaxis="y2",
        )
        fig2.update_layout(
            **PLOTLY_LAYOUT,
            barmode="stack",
            height=300,
            yaxis=dict(**PLOTLY_YAXIS, title="Daily cost (USD)", tickprefix="$"),
            yaxis2=dict(
                title="Cumulative (USD)",
                tickprefix="$",
                overlaying="y",
                side="right",
                showgrid=False,
            ),
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ---- Projection ----
    proj = burn_rate_projection(df)
    if proj["daily_avg"] > 0:
        st.info(
            f"Daily avg: **${proj['daily_avg']:,.2f}**  ·  "
            f"30-day projection: **${proj['projected_total']:,.2f}**  "
            f"(based on {proj['days_of_data']} days of data)"
        )

    # ---- By provider + by model side by side ----
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="foreman-section">BY PROVIDER</div>', unsafe_allow_html=True)
        prov_df = burn_by_provider(df)
        if not prov_df.empty:
            fig3 = go.Figure(go.Bar(
                x=prov_df["total_usd"],
                y=prov_df["provider"],
                orientation="h",
                marker_color=PLOTLY_COLORS[:len(prov_df)],
            ))
            fig3.update_layout(**PLOTLY_LAYOUT, height=250, xaxis_tickprefix="$")
            st.plotly_chart(fig3, use_container_width=True)

    with col_b:
        st.markdown('<div class="foreman-section">TOP MODELS</div>', unsafe_allow_html=True)
        model_df = burn_by_model(df, top_n=8)
        if not model_df.empty:
            fig4 = go.Figure(go.Bar(
                x=model_df["total_usd"],
                y=model_df["model"],
                orientation="h",
                marker_color=CLAY,
            ))
            fig4.update_layout(**PLOTLY_LAYOUT, height=280, xaxis_tickprefix="$")
            st.plotly_chart(fig4, use_container_width=True)

    # ---- Token breakdown ----
    st.markdown('<div class="foreman-section">TOKEN BREAKDOWN</div>', unsafe_allow_html=True)
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("Input tokens",     f"{metrics['total_input_tokens']:,}")
    t2.metric("Output tokens",    f"{metrics['total_output_tokens']:,}")
    t3.metric("Reasoning tokens", f"{metrics['total_reasoning_tokens']:,}")
    t4.metric("Cost / 1K tokens", f"${metrics['cost_per_1k_tokens']:.4f}")
