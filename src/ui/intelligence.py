"""
Spend Intelligence page — FIG. 03 flow: Detect → Propose → Guardrails → Workload Library.
"""
from __future__ import annotations

import html as _html
import pandas as pd
import streamlit as st

from src.analytics.intelligence import Finding, IntelligenceReport, Proposal, generate_report
from src.ui.theme import CLAY, SAGE


def render(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("Import spend data to generate intelligence findings.")
        return

    report = generate_report(df)

    # ---- Header metrics ----
    c1, c2, c3 = st.columns(3)
    c1.metric("Findings",             len(report.findings))
    c2.metric("Proposals",            len(report.proposals))
    c3.metric("Est. potential savings", f"${report.total_potential_savings_usd:,.2f}")

    # ---- Step 01: Detect ----
    st.markdown('<div class="foreman-section">01 · DETECT</div>', unsafe_allow_html=True)
    st.caption("concentration · drift · waste · untagged")

    if not report.findings:
        st.success("No significant findings. Spend looks healthy.")
    else:
        for f in report.findings:
            _render_finding(f)

    # ---- Step 02: Propose ----
    st.markdown('<div class="foreman-section">02 · PROPOSE</div>', unsafe_allow_html=True)
    st.caption("backtested vs golden eval")

    if not report.proposals:
        st.info("No proposals generated.")
    else:
        for p in report.proposals:
            _render_proposal(p)

    # ---- Step 03: Guardrails ----
    st.markdown('<div class="foreman-section">03 · GUARDRAILS</div>', unsafe_allow_html=True)
    st.caption("floor · suggest / auto · rollback")

    st.info(
        "**All proposals are in suggest mode by default.**  \n"
        "Promote to auto-apply only after a 7-day golden-eval quality floor hold.  \n"
        "Every routing change is backtested, not guessed."
    )

    col_a, col_b = st.columns(2)
    with col_a:
        quality_floor = st.slider(
            "Quality floor (task pass rate %)",
            min_value=70, max_value=100, value=95, step=1,
            help="Proposals will not ship if the golden eval pass rate drops below this threshold.",
        )
    with col_b:
        mode = st.radio(
            "Apply mode",
            options=["Suggest only", "Auto-apply (after eval hold)"],
            index=0,
        )

    if mode == "Auto-apply (after eval hold)":
        st.warning(
            "Auto-apply requires a verified 7-day golden eval hold above "
            f"{quality_floor}% pass rate. Rollback is one click."
        )

    # ---- Step 04: Workload Library ----
    st.markdown('<div class="foreman-section">04 · WORKLOAD LIBRARY</div>', unsafe_allow_html=True)
    st.caption("approved policy — class-level routing guidance")

    for cls, rec in report.workload_library.items():
        with st.expander(f"**{cls.upper()}** — {rec[:60]}…" if len(rec) > 60 else f"**{cls.upper()}**"):
            st.write(rec)

    # ---- Step 05: Policy Router ----
    st.markdown('<div class="foreman-section">05 · POLICY ROUTER</div>', unsafe_allow_html=True)
    st.caption("verify realized vs projected — loops back to Detect")

    st.markdown(
        "When a routing policy ships, the Policy Router compares realized savings to the "
        "projected savings from the proposal. Divergence triggers a new Detect cycle.  \n\n"
        "> *See the burn, follow the burn. Routing changes ship backtested, not guessed.*"
    )


def _render_finding(f: Finding) -> None:
    css_class = f"finding-{f.severity}"
    # Escape user-sourced strings (model names from CSV) before embedding in HTML.
    title_safe = _html.escape(f.title)
    detail_safe = _html.escape(f.detail)
    savings_note = (
        f"  Est. savings: <strong>${f.estimated_savings_usd:,.2f}</strong>"
        if f.estimated_savings_usd > 0 else ""
    )
    markup = f"""
    <div class="{css_class}">
        <strong>[{_html.escape(f.severity.upper())}] {title_safe}</strong><br>
        <span style="font-size:0.85rem">{detail_safe}{savings_note}</span>
    </div>
    """
    st.markdown(markup, unsafe_allow_html=True)


def _render_proposal(p: Proposal) -> None:
    with st.container(border=True):
        col_t, col_s = st.columns([3, 1])
        col_t.markdown(f"**{p.title}**")
        col_s.metric("Est. savings", f"${p.estimated_savings_usd:,.2f}")
        st.markdown(p.action)
        st.caption(f"Guardrail: {p.guardrail}")
        if p.affected_models:
            st.caption(f"Affected models: {', '.join(p.affected_models[:5])}")
