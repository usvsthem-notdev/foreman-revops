"""
AI Category Classifier tab.

Three sections:
  01 · COVERAGE   — classified / pending / unclassified counts + run button
  02 · BREAKDOWN  — cost and entry count by category, confidence histogram
  03 · RULES      — the classifier rule table (read-only reference)
"""
from __future__ import annotations

import html as _html

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.analytics.classifier import REVIEW_THRESHOLD, classify_pending, get_rules
from src.db import fetch_unclassified
from src.ui.theme import (
    AI_CAT_LABELS, BONE, CLAY, MUTED, PLOTLY_COLORS, PLOTLY_LAYOUT,
    PLOTLY_YAXIS, SAGE, SAND, SLATE, WC_LABELS,
)

# Colour per category — stable so charts are consistent across rerenders
_CAT_COLORS: dict[str, str] = {
    "code_gen":        CLAY,
    "research":        SAGE,
    "document_office": "#7B9EC9",
    "unknown":         MUTED,
}


def render(df: pd.DataFrame) -> None:
    if df.empty:
        st.info("No spend data. Import entries or connect a live API key to run the classifier.")
        return

    # Coerce DB int flags to bool/float so pandas doesn't choke
    df = df.copy()
    if "tag_needs_review" in df.columns:
        df["tag_needs_review"] = df["tag_needs_review"].fillna(0).astype(bool)
    if "tag_confidence" in df.columns:
        df["tag_confidence"] = pd.to_numeric(df["tag_confidence"], errors="coerce")
    if "ai_category" not in df.columns:
        df["ai_category"] = "unknown"

    _render_coverage(df)
    _render_breakdown(df)
    _render_rules()


# ---------------------------------------------------------------------------
# 01 · COVERAGE
# ---------------------------------------------------------------------------

def _render_coverage(df: pd.DataFrame) -> None:
    st.markdown('<div class="foreman-section">01 · COVERAGE</div>', unsafe_allow_html=True)
    st.caption("How many entries have been through the classifier and how many still need review.")

    total       = len(df)
    classified  = int(df["tag_confidence"].notna().sum())
    unclassified = total - classified
    needs_review = int(df["tag_needs_review"].sum()) if "tag_needs_review" in df.columns else 0
    confirmed   = classified - needs_review

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total entries",   f"{total:,}")
    c2.metric("Classified",      f"{classified:,}",
              delta=f"{classified / total:.0%} of total" if total else None,
              delta_color="off")
    c3.metric("Confirmed",       f"{confirmed:,}",
              help="Classified with confidence ≥ threshold, or manually confirmed.")
    c4.metric("Pending review",  f"{needs_review:,}",
              help="Confidence below threshold — confirm in Spend Intelligence.")

    st.divider()

    col_btn, col_note = st.columns([1, 3])
    with col_btn:
        unclassified_db = fetch_unclassified(limit=1)
        if unclassified_db:
            if st.button("Run classifier", type="primary", key="clf_run_btn"):
                with st.spinner("Classifying…"):
                    n = classify_pending(limit=50_000)
                st.success(f"Classified {n:,} entries. Reload the page to see updated counts.")
                st.rerun()
        else:
            st.button("Run classifier", disabled=True, key="clf_run_btn_disabled",
                      help="All entries have already been classified.")

    with col_note:
        st.caption(
            f"Assigns **ai_category** from provider + workload class using the rule table below.  \n"
            f"Entries with confidence < **{REVIEW_THRESHOLD:.0%}** are flagged for review "
            f"and never silently committed as ground truth."
        )
        if unclassified > 0:
            st.warning(f"{unclassified:,} entries have not been classified yet — click **Run classifier**.")


# ---------------------------------------------------------------------------
# 02 · BREAKDOWN
# ---------------------------------------------------------------------------

def _render_breakdown(df: pd.DataFrame) -> None:
    st.markdown('<div class="foreman-section">02 · BREAKDOWN</div>', unsafe_allow_html=True)

    col_cost, col_conf = st.columns(2)

    with col_cost:
        st.caption("Cost by AI category")
        cat_cost = (
            df.groupby("ai_category")["cost_usd"]
            .sum()
            .reset_index()
            .sort_values("cost_usd", ascending=True)
        )
        cat_cost["label"] = cat_cost["ai_category"].map(
            lambda x: AI_CAT_LABELS.get(x, x)
        )
        cat_cost["color"] = cat_cost["ai_category"].map(
            lambda x: _CAT_COLORS.get(x, MUTED)
        )
        fig = go.Figure(go.Bar(
            x=cat_cost["cost_usd"],
            y=cat_cost["label"],
            orientation="h",
            marker_color=cat_cost["color"].tolist(),
            text=cat_cost["cost_usd"].map("${:,.2f}".format),
            textposition="outside",
        ))
        fig.update_layout(**PLOTLY_LAYOUT, height=220, xaxis_tickprefix="$",
                          margin=dict(l=130, r=60, t=20, b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col_conf:
        st.caption(f"Confidence distribution (threshold at {REVIEW_THRESHOLD:.0%})")
        classified_df = df.dropna(subset=["tag_confidence"])
        if classified_df.empty:
            st.info("Run the classifier to see confidence distribution.")
        else:
            fig2 = go.Figure(go.Histogram(
                x=classified_df["tag_confidence"],
                nbinsx=20,
                marker_color=SAGE,
                opacity=0.85,
                name="Confidence",
            ))
            fig2.add_vline(
                x=REVIEW_THRESHOLD,
                line_dash="dash",
                line_color=CLAY,
                annotation_text=f"Review threshold ({REVIEW_THRESHOLD:.0%})",
                annotation_position="top right",
                annotation_font_color=CLAY,
            )
            fig2.update_layout(
                **PLOTLY_LAYOUT,
                height=220,
                xaxis=dict(tickformat=".0%", range=[0, 1]),
                yaxis=dict(**PLOTLY_YAXIS, title="Entries"),
                showlegend=False,
                margin=dict(l=40, r=20, t=20, b=20),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Entry explorer ──
    st.caption("Entries by category")

    cat_options = ["All"] + [
        AI_CAT_LABELS.get(c, c)
        for c in df["ai_category"].unique()
        if c in AI_CAT_LABELS
    ]
    chosen_label = st.selectbox("Filter by category", cat_options,
                                key="clf_cat_filter", label_visibility="collapsed")
    chosen_key   = next(
        (k for k, v in AI_CAT_LABELS.items() if v == chosen_label),
        None,
    )

    filtered = df if chosen_key is None else df[df["ai_category"] == chosen_key]

    display = (
        filtered[["timestamp", "provider", "model", "workload_class",
                  "ai_category", "tag_confidence", "tag_needs_review",
                  "cost_usd", "team"]]
        .copy()
        .sort_values("timestamp", ascending=False)
        .head(500)
    )
    display["ai_category"]    = display["ai_category"].map(lambda x: AI_CAT_LABELS.get(x, x))
    display["workload_class"] = display["workload_class"].map(lambda x: WC_LABELS.get(x, x))
    display["tag_confidence"] = display["tag_confidence"].map(
        lambda v: f"{v:.0%}" if pd.notna(v) else "—"
    )
    display["tag_needs_review"] = display["tag_needs_review"].map(
        {True: "⚠ review", False: "✓", 1: "⚠ review", 0: "✓"}
    ).fillna("—")
    display["cost_usd"] = display["cost_usd"].map("${:.4f}".format)
    display["timestamp"] = pd.to_datetime(display["timestamp"]).dt.date

    display.columns = [
        "Date", "Provider", "Model", "Workload",
        "AI Category", "Confidence", "Status", "Cost", "Team"
    ]
    st.dataframe(display, use_container_width=True, height=320, hide_index=True)
    if len(filtered) > 500:
        st.caption(f"Showing 500 of {len(filtered):,} entries.")


# ---------------------------------------------------------------------------
# 03 · RULES
# ---------------------------------------------------------------------------

def _render_rules() -> None:
    st.markdown('<div class="foreman-section">03 · RULES</div>', unsafe_allow_html=True)
    st.caption(
        "Rules are evaluated top-to-bottom — first match wins. "
        f"Entries below **{REVIEW_THRESHOLD:.0%}** confidence are flagged for human review "
        "before the tag is treated as ground truth."
    )

    rules = get_rules()
    rows = []
    for r in rules:
        rows.append({
            "Provider match": r["provider"],
            "Workload match": WC_LABELS.get(r["workload"], r["workload"]),
            "→ AI Category":  AI_CAT_LABELS.get(r["category"], r["category"]),
            "Confidence":     f"{r['confidence']:.0%}",
            "Auto-confirm":   "No — needs review" if r["needs_review"] else "Yes",
        })

    rules_df = pd.DataFrame(rows)

    def _row_style(row):
        if "No" in str(row.get("Auto-confirm", "")):
            return ["background-color: #FDF4F0"] * len(row)
        return [""] * len(row)

    st.dataframe(
        rules_df.style.apply(_row_style, axis=1),
        use_container_width=True,
        hide_index=True,
    )
    st.caption(
        "To add or tune rules, edit `src/analytics/classifier.py` → `_RULES`. "
        "Re-run the classifier after any change to propagate updates to existing entries."
    )
