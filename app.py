"""
Foreman RevOps Tracker — MVP entry point.

Run:  streamlit run app.py
HF:   This file is the Hugging Face Spaces entry point.
"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

import streamlit as st

from src.analytics.burn_map import budget_status, load_dataframe

# Bootstrap DB before any other src imports that might read it
from src.db import fetch_budgets, init_db
from src.demo import seed_if_empty
from src.ui import (
    api_poll,
    bill_analyzer,
    burn_map,
    classifier,
    entry,
    finance,
    intelligence,
    optimizer,
    settings,
)
from src.ui.theme import BONE, CSS, PROVIDER_LABELS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Foreman · RevOps Tracker",
    page_icon="🔥",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get help": "https://github.com/usvsthem-notdev/foreman-revops",
        "Report a bug": "https://github.com/usvsthem-notdev/foreman-revops/issues",
        "About": "Foreman RevOps Tracker — open-source LLM spend intelligence.",
    },
)

st.markdown(CSS, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# DB init (idempotent)
# ---------------------------------------------------------------------------

@st.cache_resource
def _init() -> bool:
    init_db()
    return seed_if_empty()

_demo_mode = _init()

# ---------------------------------------------------------------------------
# Sidebar — filters
# ---------------------------------------------------------------------------

with st.sidebar:
    st.markdown(
        f"<h2 style='color:{BONE}; letter-spacing:0.05em; font-size:1.1rem;'>"
        f"FOREMAN</h2>"
        f"<p style='color:#8A9BB0; font-size:0.7rem; letter-spacing:0.12em; "
        f"text-transform:uppercase; margin-top:-0.5rem;'>RevOps Tracker</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    # Date range
    today = datetime.utcnow().date()
    date_options = {
        "Last 7 days":   today - timedelta(days=7),
        "Last 30 days":  today - timedelta(days=30),
        "Last 90 days":  today - timedelta(days=90),
        "All time":      None,
        "Custom":        None,
    }
    range_choice = st.selectbox("Date range", list(date_options.keys()), index=1)

    if range_choice == "Custom":
        since_date = st.date_input("From", value=today - timedelta(days=30))
        until_date = st.date_input("To",   value=today)
        since_dt = datetime.combine(since_date, datetime.min.time())
        until_dt = datetime.combine(until_date, datetime.max.time().replace(microsecond=0))
    elif date_options[range_choice] is not None:
        since_dt = datetime.combine(date_options[range_choice], datetime.min.time())
        until_dt = None
    else:
        since_dt = None
        until_dt = None

    # Provider filter
    provider_filter = st.selectbox(
        "Provider",
        options=["All", "anthropic", "openai", "google", "gemini", "cursor", "mistral", "together", "other"],
        format_func=lambda x: PROVIDER_LABELS.get(x, x),
    )

    # Team filter
    team_filter = st.text_input("Team", placeholder="all teams", max_chars=64)

    st.divider()
    if _demo_mode:
        st.info(
            "**Sample data** — 60-day simulation across 4 providers.  \n"
            "Connect real keys in the **Live API** tab to see your own spend.",
            icon="ℹ️",
        )
    else:
        st.caption("Data is stored locally. No telemetry.")

# ---------------------------------------------------------------------------
# Load data (cached per filter combination)
# ---------------------------------------------------------------------------

@st.cache_data(ttl=30)
def _load(provider: str, team: str, since: str, until: str):
    return load_dataframe(
        provider=provider if provider != "All" else None,
        team=team.strip() or None,
        since=datetime.fromisoformat(since) if since else None,
        until=datetime.fromisoformat(until) if until else None,
    )


df = _load(
    provider_filter,
    team_filter,
    since_dt.isoformat() if since_dt else "",
    until_dt.isoformat() if until_dt else "",
)

budgets_raw  = fetch_budgets()
budgets_stat = budget_status(df, budgets_raw) if not df.empty else []

# ---------------------------------------------------------------------------
# Main navigation
# ---------------------------------------------------------------------------

TABS = [
    "Burn Map", "Finance", "Spend Intelligence", "AI Categories",
    "Prompt Optimizer", "Bill Analyzer", "Live API", "Manual Entry", "Settings",
]
(
    tab_burn, tab_finance, tab_intel, tab_clf,
    tab_optimizer, tab_bill, tab_api, tab_entry, tab_settings,
) = st.tabs(TABS)

with tab_burn:
    burn_map.render(df, budgets_stat)

with tab_finance:
    finance.render(df, budgets_raw)

with tab_intel:
    intelligence.render(df)

with tab_clf:
    classifier.render(df)

with tab_optimizer:
    optimizer.render()

with tab_bill:
    bill_analyzer.render()

with tab_api:
    api_poll.render()

with tab_entry:
    entry.render()

with tab_settings:
    settings.render()
