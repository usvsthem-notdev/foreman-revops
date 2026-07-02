"""
Burn Map page — live spend by class, provider, model, and over time.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics import mcp_usage
from src.analytics.burn_map import (
    burn_by_class,
    burn_by_model,
    burn_by_provider,
    burn_rate_projection,
    cumulative_burn,
    key_metrics,
)
from src.ui.theme import CLAY, PLOTLY_COLORS, PLOTLY_LAYOUT, PLOTLY_YAXIS, SAGE, SLATE, WC_LABELS


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
    c3.metric(
        "Local Absorbed",
        f"${metrics['absorbed_cost_usd']:,.2f}",
        delta=f"{metrics['local_pct']:.1f}% of total",
        delta_color="off",
    )
    c4.metric("Entries", f"{metrics['entry_count']:,}")

    if df.empty:
        st.info("No spend data yet. Upload a billing export or add entries manually.")
        return

    # ---- Budget progress ----
    if budgets_status:
        st.markdown('<div class="foreman-section">BUDGETS</div>', unsafe_allow_html=True)
        for b in budgets_status:
            remaining = b["amount_usd"] - b["spent_usd"]
            over = remaining < 0
            color = "🔴" if b["over_threshold"] else "🟢"
            label = (
                f"{b['name']}  ·  {b['period']}"
                f"  ·  ${b['spent_usd']:,.2f} / ${b['amount_usd']:,.2f}"
            )
            col_bar, col_rem = st.columns([5, 1])
            with col_bar:
                st.markdown(f"{color} **{label}**")
                st.progress(float(b["pct_used"]))
            with col_rem:
                st.metric(
                    "Over" if over else "Left",
                    f"${abs(remaining):,.2f}",
                )

    # ---- Burn by workload class (FIG. 03 main chart) ----
    st.markdown(
        '<div class="foreman-section">BURN MAP — LIVE SPEND BY CLASS</div>',
        unsafe_allow_html=True,
    )
    class_df = burn_by_class(df)
    if not class_df.empty:
        class_df = class_df.copy()
        class_df["workload_class"] = class_df["workload_class"].map(
            lambda x: WC_LABELS.get(x, x)
        )
        fig = go.Figure()
        fig.add_bar(
            name="Local / on-prem",
            x=class_df["workload_class"],
            y=class_df["absorbed_usd"],
            marker_color=SAGE,
        )
        fig.add_bar(
            name="API / external",
            x=class_df["workload_class"],
            y=class_df["frontier_usd"],
            marker_color=CLAY,
        )
        fig.update_layout(
            **PLOTLY_LAYOUT,
            barmode="stack",
            height=320,
            xaxis_title="AI use category",
            yaxis=dict(**PLOTLY_YAXIS, title="Cost (USD)", tickprefix="$"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Green = Local / on-prem models  ·  Orange = API / external spend")

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

    # ---- Prompt / prefix cache ----
    # cache_read_tokens is a subset of input_tokens billed at a steep
    # discount (Anthropic ~90% off, OpenAI ~50% off) instead of full input
    # price — surfaced separately so caching's $ impact isn't hidden inside
    # the blended input-axis total above.
    if metrics["total_cache_read_tokens"] > 0 or metrics["total_cache_creation_tokens"] > 0:
        st.markdown(
            '<div class="foreman-section">PROMPT / PREFIX CACHE</div>', unsafe_allow_html=True
        )
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Cache hit rate", f"{metrics['cache_hit_rate']:.0f}%")
        c2.metric("$ saved via caching", f"${metrics['cache_savings_usd']:,.2f}")
        c3.metric("Cache read tokens", f"{metrics['total_cache_read_tokens']:,}")
        c4.metric("Cache write tokens", f"{metrics['total_cache_creation_tokens']:,}")
        st.caption(
            "Cache reads are priced far below fresh input tokens (Anthropic "
            "~90% off, OpenAI ~50% off) — the input $ split above already "
            "prices them correctly, this is the discount's dollar impact."
        )
        if metrics["total_cost_usd"] <= 0 and metrics["cache_savings_usd"] > 0:
            st.caption(
                "⚠️ Total Spend for this period is $0, so '$ saved via caching' "
                "is a list-price estimate from token volume alone, independent "
                "of recorded cost — some sources (e.g. live-polled usage data) "
                "don't report per-entry cost, so recorded spend here is "
                "incomplete rather than truly zero."
            )

    # ---- Input vs output cost split ----
    # Token *count* share and dollar *cost* share diverge because output is
    # priced ~4-8x higher per token than input — this section makes that gap
    # visible instead of leaving it buried inside a single cost_usd total.
    st.markdown('<div class="foreman-section">INPUT vs OUTPUT COST</div>', unsafe_allow_html=True)
    gap = metrics["input_token_share"] - metrics["input_cost_share"]
    io1, io2, io3 = st.columns(3)
    io1.metric(
        "Input $ share",
        f"{metrics['input_cost_share']:.0f}%",
        delta=f"{metrics['input_token_share']:.0f}% of tokens",
        delta_color="off",
    )
    io2.metric(
        "Output $ share",
        f"{100 - metrics['input_cost_share']:.0f}%",
        delta=f"{100 - metrics['input_token_share']:.0f}% of tokens",
        delta_color="off",
    )
    io3.metric("Input $", f"${metrics['input_cost_usd']:,.2f}")
    has_token_data = (
        metrics["total_input_tokens"]
        + metrics["total_output_tokens"]
        + metrics["total_reasoning_tokens"]
    ) > 0
    if has_token_data and abs(gap) >= 10:
        st.caption(
            f"Input is **{metrics['input_token_share']:.0f}%** of tokens but only "
            f"**{metrics['input_cost_share']:.0f}%** of dollars — output tokens are "
            "priced several times higher per token, so cutting output length moves "
            "the bill more than cutting input volume."
        )
    elif not has_token_data and metrics["input_cost_usd"] > 0:
        st.caption(
            "This period has cost data but no recorded token counts (e.g. "
            "flat-rate/invoice entries) — the input/output split above isn't "
            "backed by real token volume."
        )

    # ---- MCP tool-call input burn ----
    # Every MCP tool response gets read back into the calling agent's own
    # context as input tokens on its next turn — a burn source that's real
    # but invisible in provider billing, so it's tracked and shown separately.
    mcp_df = mcp_usage.load_dataframe()
    if not mcp_df.empty:
        st.markdown(
            '<div class="foreman-section">MCP TOOL-CALL INPUT BURN</div>', unsafe_allow_html=True
        )
        mcp_metrics = mcp_usage.key_metrics(mcp_df)
        m1, m2, m3 = st.columns(3)
        m1.metric("Tool calls", f"{mcp_metrics['call_count']:,}")
        m2.metric("Est. input tokens fed back", f"{mcp_metrics['total_input_tokens']:,}")
        m3.metric("Est. $ (input-priced)", f"${mcp_metrics['total_estimated_usd']:,.4f}")
        st.caption(
            "Estimated tokens an MCP tool's JSON response adds to the calling "
            "agent's context — not provider spend, a local estimate via the same "
            "free heuristic foreman_optimizer uses."
        )

        col_tool, col_daily = st.columns(2)
        with col_tool:
            st.markdown("**By tool**")
            tool_df = mcp_usage.by_tool(mcp_df)
            if not tool_df.empty:
                fig5 = go.Figure(go.Bar(
                    x=tool_df["input_tokens"],
                    y=tool_df["tool"],
                    orientation="h",
                    marker_color=SLATE,
                ))
                fig5.update_layout(**PLOTLY_LAYOUT, height=250, xaxis_title="Input tokens")
                st.plotly_chart(fig5, use_container_width=True)

        with col_daily:
            st.markdown("**Daily trend**")
            daily_df = mcp_usage.daily_input_tokens(mcp_df)
            if not daily_df.empty:
                fig6 = go.Figure(go.Bar(
                    x=daily_df["date"].astype(str),
                    y=daily_df["input_tokens"],
                    marker_color=CLAY,
                ))
                fig6.update_layout(**PLOTLY_LAYOUT, height=250, yaxis_title="Input tokens")
                st.plotly_chart(fig6, use_container_width=True)
