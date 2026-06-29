"""
Settings page — budgets, data export/import, and about.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime

import pandas as pd
import streamlit as st

from src.db import clear_all_entries, delete_budget, fetch_budgets, fetch_entries, upsert_budget
from src.models import Budget, BudgetPeriod, Provider

log = logging.getLogger(__name__)


def render() -> None:
    tab_budget, tab_data, tab_about = st.tabs(["Budgets", "Data", "About"])

    with tab_budget:
        _render_budgets()

    with tab_data:
        _render_data()

    with tab_about:
        _render_about()


def _render_budgets() -> None:
    st.markdown('<div class="foreman-section">ADD BUDGET</div>', unsafe_allow_html=True)

    with st.form("add_budget", clear_on_submit=True):
        c1, c2 = st.columns(2)
        name   = c1.text_input("Budget name", placeholder="Monthly total", max_chars=64)
        amount = c2.number_input("Amount (USD)", min_value=0.01, step=10.0, format="%.2f")

        c3, c4, c5 = st.columns(3)
        period    = c3.selectbox("Period", [p.value for p in BudgetPeriod])
        provider  = c4.selectbox("Limit to provider", ["(all)"] + [p.value for p in Provider])
        threshold = c5.slider("Alert at", 50, 100, 80, step=5, format="%d%%")

        team = st.text_input("Limit to team (optional)", max_chars=64)

        if st.form_submit_button("Save Budget", type="primary"):
            if not name.strip():
                st.error("Name is required.")
            else:
                budget = Budget(
                    name=name.strip(),
                    amount_usd=float(amount),
                    period=BudgetPeriod(period),
                    provider=Provider(provider) if provider != "(all)" else None,
                    team=team.strip() or None,
                    alert_threshold=threshold / 100,
                )
                upsert_budget(budget)
                st.success(f"Budget '{budget.name}' saved.")
                st.rerun()

    st.markdown('<div class="foreman-section">EXISTING BUDGETS</div>', unsafe_allow_html=True)
    budgets = fetch_budgets()
    if not budgets:
        st.info("No budgets set.")
    else:
        for b in budgets:
            col_name, col_amount, col_period, col_prov, col_team, col_thresh, col_del = \
                st.columns([3, 2, 1, 2, 2, 1, 1])
            col_name.write(f"**{b['name']}**")
            col_amount.write(f"${b['amount_usd']:,.2f}")
            col_period.write(b["period"])
            col_prov.write(b.get("provider") or "All")
            col_team.write(b.get("team") or "All")
            col_thresh.write(f"{b.get('alert_threshold', 0.8):.0%}")
            if col_del.button("✕", key=f"del_budget_{b['id']}"):
                delete_budget(b["id"])
                st.rerun()


def _render_data() -> None:
    st.markdown('<div class="foreman-section">EXPORT</div>', unsafe_allow_html=True)
    rows = fetch_entries(limit=100_000)
    if rows:
        df = pd.DataFrame(rows)
        csv = df.to_csv(index=False).encode()
        st.download_button(
            "Download all entries as CSV",
            data=csv,
            file_name=f"foreman_spend_{datetime.utcnow().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
        json_data = json.dumps(rows, default=str, indent=2).encode()
        st.download_button(
            "Download all entries as JSON",
            data=json_data,
            file_name=f"foreman_spend_{datetime.utcnow().strftime('%Y%m%d')}.json",
            mime="application/json",
        )
    else:
        st.info("No data to export.")

    st.markdown('<div class="foreman-section">DANGER ZONE</div>', unsafe_allow_html=True)
    st.warning("This will permanently delete all spend entries from the local database.")
    confirm = st.text_input("Type DELETE to confirm", max_chars=10)
    if st.button("Clear all entries", type="secondary"):
        if confirm.strip() == "DELETE":
            clear_all_entries()
            st.success("All entries cleared.")
            st.rerun()
        else:
            st.error("Type DELETE to confirm.")


def _render_about() -> None:
    st.markdown("""
## Foreman RevOps Tracker

**The FinOps view that does not yet exist for LLM spend.**

See the burn, follow the burn.

---

### What this is

An open-source spend tracker for LLM API costs — built as the free
Bill Analyzer seed product described in the [Foreman](https://github.com/usvsthem-notdev/foreman-revops)
architecture docs.

- **Burn Map** — live spend by workload class, provider, and model
- **Bill Analyzer** — parse Anthropic / OpenAI billing CSVs locally
- **Spend Intelligence** — detect concentration, drift, and waste; propose routing policies

All data stays on your machine. No telemetry.

---

### Workload classes

| Class | Description |
|-------|-------------|
| `extract` | Structured extraction, summarization, classification |
| `rag` | Retrieval-augmented generation, embedding lookups |
| `reason` | Complex multi-step reasoning, planning |
| `agents` | Long-horizon agentic workflows |
| `coding` | Code generation, review, and debugging |

**Sage** = absorbed locally · **Clay** = frontier spend

---

### License · Apache-2.0
### Version · 0.1.0-mvp
    """)
