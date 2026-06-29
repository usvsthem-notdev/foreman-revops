"""
AI Category Classifier tab.

Three sections:
  01 · COVERAGE   — classified / pending / unclassified counts + run button
  02 · BREAKDOWN  — cost and entry count by category, confidence histogram
  03 · RULES      — the classifier rule table (read-only reference)
"""
from __future__ import annotations

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
    st.caption("How many entries have been categorized and how many still need review.")

    total        = len(df)
    classified   = int(df["tag_confidence"].notna().sum())
    needs_review = int(df["tag_needs_review"].sum()) if "tag_needs_review" in df.columns else 0
    high_conf    = classified - needs_review

    # DB is authoritative for unclassified count (avoids NaN mismatch with filtered df)
    unclassified_db = fetch_unclassified(limit=1)
    has_unclassified = bool(unclassified_db)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total entries",    f"{total:,}")
    c2.metric("Categorized",      f"{classified:,}",
              delta=f"{classified / total:.0%} of total" if total else None,
              delta_color="off")
    c3.metric("High-confidence",  f"{high_conf:,}",
              help=f"Categorized with confidence ≥ {REVIEW_THRESHOLD:.0%}.")
    c4.metric("Needs review",     f"{needs_review:,}",
              help="Lower-confidence categories — confirm in Spend Intelligence before treating as final.")

    st.divider()

    col_btn, col_note = st.columns([1, 3])
    with col_btn:
        if has_unclassified:
            if st.button("Run categorizer", type="primary", key="clf_run_btn"):
                with st.spinner("Categorizing…"):
                    n = classify_pending(limit=50_000)
                st.session_state["clf_last_run_count"] = n
                st.rerun()
        else:
            st.button("Run categorizer", disabled=True, key="clf_run_btn_disabled",
                      help="All entries have already been categorized.")

    if "clf_last_run_count" in st.session_state:
        n = st.session_state.pop("clf_last_run_count")
        st.success(f"Categorized {n:,} entries.")

    with col_note:
        st.caption(
            f"Each entry is assigned a spend category based on the provider and how the tool was used.  \n"
            f"Entries with confidence below **{REVIEW_THRESHOLD:.0%}** are flagged for human review "
            f"before the category is treated as final."
        )
        if has_unclassified:
            st.warning("Some entries have not been categorized yet — click **Run categorizer**.")


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
        fig.update_layout(**PLOTLY_LAYOUT, height=220, xaxis_tickprefix="$")
        fig.update_layout(margin=dict(l=130, r=60, t=20, b=20))
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
            fig2.update_layout(**PLOTLY_LAYOUT, height=220, showlegend=False)
            fig2.update_layout(
                xaxis=dict(tickformat=".0%", range=[0, 1]),
                yaxis=dict(**PLOTLY_YAXIS, title="Entries"),
                margin=dict(l=40, r=20, t=20, b=20),
            )
            st.plotly_chart(fig2, use_container_width=True)

    # ── Entry explorer ──
    st.caption("Entries by category")

    # Include all category values in the filter, even if unmapped in AI_CAT_LABELS
    all_cats = sorted(df["ai_category"].dropna().unique().tolist())
    cat_options = ["All"] + [AI_CAT_LABELS.get(c, c) for c in all_cats]
    # Map display label → raw key (covers both known and unknown category values)
    label_to_key: dict[str, str] = {AI_CAT_LABELS.get(c, c): c for c in all_cats}

    chosen_label = st.selectbox("Filter by category", cat_options,
                                key="clf_cat_filter", label_visibility="collapsed")
    chosen_key   = label_to_key.get(chosen_label)  # None when "All" selected

    filtered = df if chosen_key is None else df[df["ai_category"] == chosen_key]

    # Pre-flight: only keep columns that are actually present to avoid KeyError on older DBs
    _WANTED = ["timestamp", "provider", "model", "workload_class",
               "ai_category", "tag_confidence", "tag_needs_review", "cost_usd", "team"]
    _COLS   = [c for c in _WANTED if c in filtered.columns]

    display = (
        filtered[_COLS]
        .copy()
        .sort_values("timestamp", ascending=False)
        .head(500)
    )
    if "ai_category" in display.columns:
        display["ai_category"]    = display["ai_category"].map(lambda x: AI_CAT_LABELS.get(x, x))
    if "workload_class" in display.columns:
        display["workload_class"] = display["workload_class"].map(lambda x: WC_LABELS.get(x, x))
    if "tag_confidence" in display.columns:
        display["tag_confidence"] = display["tag_confidence"].map(
            lambda v: f"{v:.0%}" if pd.notna(v) else "—"
        )
    if "tag_needs_review" in display.columns:
        display["tag_needs_review"] = display["tag_needs_review"].map(
            {True: "⚠ review", False: "✓", 1: "⚠ review", 0: "✓"}
        ).fillna("—")
    if "cost_usd" in display.columns:
        display["cost_usd"] = display["cost_usd"].map("${:.4f}".format)
    if "timestamp" in display.columns:
        display["timestamp"] = pd.to_datetime(display["timestamp"]).dt.date

    _RENAME = {
        "timestamp": "Date", "provider": "Provider", "model": "Model",
        "workload_class": "Workload", "ai_category": "Category",
        "tag_confidence": "Confidence", "tag_needs_review": "Status",
        "cost_usd": "Cost", "team": "Team",
    }
    display.rename(columns=_RENAME, inplace=True)
    st.dataframe(display, use_container_width=True, height=320, hide_index=True)
    if len(filtered) > 500:
        st.caption(f"Showing 500 of {len(filtered):,} entries.")


# ---------------------------------------------------------------------------
# 03 · RULES
# ---------------------------------------------------------------------------

def _render_rules() -> None:
    st.markdown('<div class="foreman-section">03 · RULES</div>', unsafe_allow_html=True)
    st.caption(
        "Categorization logic runs in priority order — first match wins. "
        f"Entries below **{REVIEW_THRESHOLD:.0%}** confidence are flagged for human review "
        "before the category is treated as final."
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
        "Categories are assigned automatically based on provider and usage type. "
        "Entries highlighted in red are below the confidence threshold and require human sign-off "
        "before the category is treated as final. Contact your admin to adjust categorization rules."
    )
