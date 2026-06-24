"""
Bill Analyzer page — upload Anthropic / OpenAI billing CSVs.
"""
from __future__ import annotations

import logging

import pandas as pd
import streamlit as st

from src.db import insert_entries_bulk
from src.parsers.anthropic import parse_anthropic_csv
from src.parsers.generic import parse_auto
from src.parsers.openai import parse_openai_csv

log = logging.getLogger(__name__)

_ALLOWED_TYPES = ["text/csv", "application/csv", "text/plain", "application/vnd.ms-excel"]
_MAX_SIZE_MB = 50


def render() -> None:
    st.markdown(
        "Upload a billing export from Anthropic or OpenAI. "
        "The analyzer parses it locally — **no data leaves your machine.**"
    )

    col_prov, col_upload = st.columns([1, 3])

    with col_prov:
        provider_choice = st.selectbox(
            "Provider",
            options=["Auto-detect", "Anthropic", "OpenAI"],
            index=0,
        )

    with col_upload:
        uploaded = st.file_uploader(
            "Billing CSV",
            type=["csv", "txt"],
            help=f"Max {_MAX_SIZE_MB} MB. CSV export from your provider's billing console.",
            accept_multiple_files=False,
        )

    if uploaded is None:
        _render_format_guide()
        return

    # Security: check declared size before reading into memory.
    # Streamlit's UploadedFile exposes .size (bytes) without calling .read().
    if uploaded.size > _MAX_SIZE_MB * 1024 * 1024:
        st.error(f"File too large ({uploaded.size / 1e6:.1f} MB). Max is {_MAX_SIZE_MB} MB.")
        return

    data = uploaded.read()

    # Security: reject non-UTF-8 content (binary files, wrong encoding).
    try:
        data.decode("utf-8-sig")
    except UnicodeDecodeError:
        st.error("File is not valid UTF-8. Please export as CSV and try again.")
        return

    with st.spinner("Parsing…"):
        try:
            if provider_choice == "Anthropic":
                bill = parse_anthropic_csv(data, uploaded.name)
            elif provider_choice == "OpenAI":
                bill = parse_openai_csv(data, uploaded.name)
            else:
                bill = parse_auto(data, uploaded.name)
        except Exception as exc:
            st.error(f"Parse error: {exc}")
            log.exception("Bill parse failed for %s", uploaded.name)
            return

    # ---- Summary ----
    summary = bill.summarize()
    st.success(
        f"Parsed **{summary['entry_count']:,}** entries from **{summary['provider']}** "
        f"— Total: **${summary['total_cost_usd']:,.4f}**"
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total cost",      f"${summary['total_cost_usd']:,.2f}")
    m2.metric("Input tokens",    f"{summary['total_input_tokens']:,}")
    m3.metric("Output tokens",   f"{summary['total_output_tokens']:,}")
    m4.metric("Reasoning tokens",f"{summary['total_reasoning_tokens']:,}")

    if bill.parse_warnings:
        with st.expander(f"{len(bill.parse_warnings)} parse warning(s)"):
            for w in bill.parse_warnings:
                st.warning(w)

    # ---- Absorbable spend estimate ----
    if bill.entries:
        absorbable_entries = [
            e for e in bill.entries if e.workload_class.value in ("extract", "rag")
        ]
        absorbable_cost = sum(e.cost_usd for e in absorbable_entries)
        total_cost = summary["total_cost_usd"]
        absorbable_pct = absorbable_cost / total_cost if total_cost > 0 else 0
        if absorbable_cost > 0:
            st.info(
                f"**Estimated absorbable spend:** ${absorbable_cost:,.2f} "
                f"({absorbable_pct:.0%} of total) — extract and RAG workloads that could "
                f"run on a local model."
            )

    # ---- Preview table ----
    if bill.entries:
        st.markdown('<div class="foreman-section">ENTRY PREVIEW</div>', unsafe_allow_html=True)
        rows = [
            {
                "Date": e.timestamp.date(),
                "Provider": e.provider.value,
                "Model": e.model,
                "Class": e.workload_class.value,
                "Input tok": f"{e.input_tokens:,}",
                "Output tok": f"{e.output_tokens:,}",
                "Reason tok": f"{e.reasoning_tokens:,}",
                "Cost USD": f"${e.cost_usd:.6f}",
                "Local": "✓" if e.is_local else "",
                "Feature": e.feature or "",
            }
            for e in bill.entries[:200]
        ]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, height=280)
        if len(bill.entries) > 200:
            st.caption(f"Showing first 200 of {len(bill.entries):,} entries.")

    # ---- Import button ----
    if bill.entries:
        st.divider()
        if st.button("Import to Burn Map", type="primary", icon="📥"):
            with st.spinner("Importing…"):
                count = insert_entries_bulk(bill.entries)
            skipped = len(bill.entries) - count
            if count > 0:
                msg = f"Imported {count:,} entries."
                if skipped:
                    msg += f" {skipped:,} duplicate(s) skipped."
                st.success(msg + " Visit the Burn Map tab to see your data.")
            else:
                st.info(f"All {skipped:,} entries already exist — nothing new imported.")
            st.rerun()


def _render_format_guide() -> None:
    with st.expander("How to export your billing CSV"):
        st.markdown("""
**Anthropic Console**
1. Go to [console.anthropic.com](https://console.anthropic.com) → **Billing** → **Usage**
2. Select date range → **Export CSV**
3. Columns: Date, Organization, Project, Model, Input tokens, Output tokens,\
 Cache read tokens, Cache write tokens, Cost (USD)

---

**OpenAI Platform**
1. Go to [platform.openai.com](https://platform.openai.com) → **Usage** → **Export**
2. Or: **Billing** → **Usage this month** → **Download CSV**
3. Columns vary by plan — the parser handles multiple OpenAI export formats.
        """)
