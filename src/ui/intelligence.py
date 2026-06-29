"""
Spend Intelligence page — FIG. 03 flow: Detect → Propose → Guardrails → Workload Library.
"""
from __future__ import annotations

import html as _html

import pandas as pd
import streamlit as st

from src.analytics.classifier import REVIEW_THRESHOLD, classify_pending
from src.analytics.intelligence import Finding, Proposal, generate_report
from src.db import fetch_pending_review, fetch_unclassified, tag_entry
from src.models import AICategory
from src.ui.theme import WC_LABELS


def render(df: pd.DataFrame) -> None:
    # Fetch counts before rendering so we can include pending in the header row
    pending_count    = len(fetch_pending_review(limit=200))
    unclassified_any = bool(fetch_unclassified(limit=1))

    if not df.empty:
        report = generate_report(df)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Findings",              len(report.findings))
        c2.metric("Proposals",             len(report.proposals))
        c3.metric("Est. potential savings", f"${report.total_potential_savings_usd:,.2f}")
        c4.metric("Pending review",        pending_count,
                  help="Low-confidence AI category tags awaiting your confirmation.")

    _render_pending_review(pending_count, unclassified_any)

    if df.empty:
        st.info("Import spend data to generate intelligence findings.")
        return

    # ---- Step 01: Detect ----
    st.markdown('<div class="foreman-section">01 · DETECT</div>', unsafe_allow_html=True)
    st.caption("Single-model risk · Spend acceleration · Token waste · Unclassified entries")

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
        label = WC_LABELS.get(cls, cls.upper())
        with st.expander(label):
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


# ---------------------------------------------------------------------------
# Pending review — Need mark equivalent
# ---------------------------------------------------------------------------

_CATEGORY_OPTIONS = [c.value for c in AICategory]


def _render_pending_review(pending_count: int, unclassified_any: bool) -> None:
    pending = fetch_pending_review(limit=200) if pending_count > 0 else []

    if not unclassified_any and not pending:
        return

    st.markdown(
        '<div class="foreman-section">00 · PENDING REVIEW</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        f"Classifier confidence below {REVIEW_THRESHOLD:.0%} — confirm or override "
        "before these tags are treated as ground truth."
    )

    if unclassified_any:
        col_btn, col_note = st.columns([1, 3])
        with col_btn:
            if st.button("Run classifier", type="primary", key="run_classifier"):
                with st.spinner("Classifying…"):
                    n = classify_pending()
                st.success(f"Classified {n} entries.")
                st.rerun()
        with col_note:
            st.caption(
                "Assigns ai_category from provider + model + workload class. "
                "Results below the confidence threshold land here for review."
            )
        if pending:
            st.divider()

    if pending:
        st.write(f"**{len(pending)}** {'entry' if len(pending) == 1 else 'entries'} "
                 "awaiting confirmation:")
        for entry in pending:
            _render_review_row(entry)

    st.divider()


def _render_review_row(entry: dict) -> None:
    eid        = entry["id"]
    confidence = entry.get("tag_confidence") or 0.0
    proposed   = entry.get("ai_category", "unknown")
    provider   = entry.get("provider", "")
    model      = (entry.get("model") or "")[:48]
    wc         = entry.get("workload_class", "")
    team       = entry.get("team") or entry.get("user_id") or "untagged"
    date       = (entry.get("timestamp") or "")[:10]

    with st.container(border=True):
        col_info, col_action = st.columns([3, 2])

        with col_info:
            st.markdown(f"**{_html.escape(provider)}** · `{_html.escape(model)}`")
            st.caption(
                f"workload: {_html.escape(wc)} · "
                f"team: {_html.escape(str(team))} · {date}"
            )
            st.progress(
                confidence,
                text=f"Confidence {confidence:.0%} → proposed: **{proposed}**",
            )

        with col_action:
            try:
                default_idx = _CATEGORY_OPTIONS.index(proposed)
            except ValueError:
                default_idx = _CATEGORY_OPTIONS.index("unknown")

            chosen = st.selectbox(
                "Category",
                options=_CATEGORY_OPTIONS,
                index=default_idx,
                key=f"cat_{eid}",
                label_visibility="collapsed",
            )
            if st.button("Confirm", key=f"confirm_{eid}"):
                tag_entry(
                    eid,
                    ai_category=AICategory(chosen),
                    confidence=confidence,
                    needs_review=False,
                )
                st.rerun()
