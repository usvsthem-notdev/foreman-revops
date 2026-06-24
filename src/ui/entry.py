"""
Manual entry page — log individual API calls or bulk-paste from logs.
"""
from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import streamlit as st

from src.db import delete_entry, fetch_entries, insert_entry
from src.models import EntrySource, Provider, SpendEntry, WorkloadClass
from src.parsers.base import infer_is_local, infer_workload_class

log = logging.getLogger(__name__)


def render() -> None:
    tab_single, tab_table = st.tabs(["Single Entry", "Recent Entries"])

    with tab_single:
        _render_single_form()

    with tab_table:
        _render_recent_table()


def _render_single_form() -> None:
    with st.form("manual_entry", clear_on_submit=True):
        c1, c2 = st.columns(2)

        timestamp = c1.date_input("Date", value=datetime.utcnow().date())
        provider  = c2.selectbox("Provider", [p.value for p in Provider])

        c3, c4 = st.columns(2)
        model = c3.text_input("Model name", placeholder="claude-haiku-4-5", max_chars=128)
        workload = c4.selectbox("Workload class", [w.value for w in WorkloadClass])

        c5, c6, c7 = st.columns(3)
        input_tok    = c5.number_input("Input tokens",    min_value=0, step=100)
        output_tok   = c6.number_input("Output tokens",   min_value=0, step=100)
        reasoning_tok= c7.number_input("Reasoning tokens",min_value=0, step=100)

        c8, c9 = st.columns(2)
        cost = c8.number_input("Cost (USD)", min_value=0.0, step=0.0001, format="%.6f")
        is_local = c9.checkbox("Absorbed locally (ran on local model)")

        c10, c11 = st.columns(2)
        team    = c10.text_input("Team / project", max_chars=64)
        feature = c11.text_input("Feature", max_chars=128)

        notes = st.text_area("Notes (optional)", max_chars=512, height=80)

        submitted = st.form_submit_button("Add Entry", type="primary")

    if submitted:
        if not model.strip():
            st.error("Model name is required.")
            return
        if cost == 0 and (input_tok + output_tok + reasoning_tok) == 0:
            st.error("Provide either a cost or token counts.")
            return

        inferred_class = WorkloadClass(workload) if workload != "unknown" else infer_workload_class(model)
        inferred_local = is_local or infer_is_local(model)

        entry = SpendEntry(
            timestamp=datetime.combine(timestamp, datetime.min.time()),
            provider=Provider(provider),
            model=model.strip(),
            workload_class=inferred_class,
            input_tokens=int(input_tok),
            output_tokens=int(output_tok),
            reasoning_tokens=int(reasoning_tok),
            cost_usd=float(cost),
            is_local=inferred_local,
            team=team.strip() or None,
            feature=feature.strip() or None,
            notes=notes.strip() or None,
            source=EntrySource.manual,
        )
        insert_entry(entry)
        st.success("Entry added.")
        st.rerun()


def _render_recent_table() -> None:
    rows = fetch_entries(limit=500)
    if not rows:
        st.info("No entries yet.")
        return

    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).dt.date

    display = df[["timestamp","provider","model","workload_class","cost_usd",
                  "input_tokens","output_tokens","reasoning_tokens","is_local","team","feature"]].copy()
    display["cost_usd"] = display["cost_usd"].map("${:.6f}".format)
    display["is_local"] = display["is_local"].map({1: "✓", 0: "", True: "✓", False: ""})

    st.dataframe(display, use_container_width=True, height=400)

    # Delete by ID
    with st.expander("Delete an entry"):
        del_id = st.text_input("Entry ID to delete", max_chars=36)
        if st.button("Delete", type="secondary") and del_id.strip():
            # Validate it's a UUID-shaped string before hitting DB
            import re
            if re.match(r"^[0-9a-f\-]{36}$", del_id.strip()):
                delete_entry(del_id.strip())
                st.success("Deleted.")
                st.rerun()
            else:
                st.error("Invalid entry ID format.")
