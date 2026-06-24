"""
Live API Polling tab — pull usage directly from Anthropic / OpenAI APIs.

Security notes that inform this UI:
  - Keys are entered via st.text_input(type="password") — never plain-text.
  - Keys are stored in .env.local (0o600) or environment variables only.
  - The raw key value is never displayed or echoed back; only a masked form is shown.
  - Polling is gated behind a minimum interval to prevent accidental rate-limit lockouts.
  - All HTTP calls are routed through polling.base.safe_get which enforces the host allowlist.
"""
from __future__ import annotations

import html
import logging
from datetime import date, datetime, timedelta

import streamlit as st

from src.db import get_poll_cursor, insert_entries_bulk, set_poll_cursor
from src.polling import anthropic as anthropic_poller
from src.polling import openai as openai_poller
from src.polling.base import mask_key, validate_key_format
from src.polling.key_store import clear_key, get_key, has_key, key_source, set_key

log = logging.getLogger(__name__)

_MIN_POLL_SECONDS = 60
_PROVIDERS = {
    "anthropic": {
        "label":   "Anthropic",
        "poller":  anthropic_poller.poll,
        "key_hint": "sk-ant-api03-…",
        "scope_note": (
            "Requires a key with **usage:read** scope. Create one at "
            "[console.anthropic.com](https://console.anthropic.com) → API Keys."
        ),
    },
    "openai": {
        "label":  "OpenAI",
        "poller": openai_poller.poll,
        "key_hint": "sk-… or sk-proj-…",
        "scope_note": (
            "Requires a key with **Read usage data** permission. Create one at "
            "[platform.openai.com/api-keys](https://platform.openai.com/api-keys)."
        ),
    },
}


def render() -> None:
    st.markdown(
        '<div class="foreman-section">LIVE API POLLING</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Pull usage data directly from provider APIs. "
        "Keys are stored locally in .env.local (0o600) — never in the database."
    )

    tab_anthropic, tab_openai = st.tabs(["Anthropic", "OpenAI"])

    with tab_anthropic:
        _render_provider("anthropic")

    with tab_openai:
        _render_provider("openai")


# ── Per-provider section ─────────────────────────────────────────────────────

def _render_provider(provider: str) -> None:
    cfg = _PROVIDERS[provider]
    label = cfg["label"]

    _render_key_section(provider, cfg)
    st.divider()

    if not has_key(provider):
        st.info(f"Add a {label} API key above to enable polling.")
        return

    _render_poll_section(provider, cfg)


def _render_key_section(provider: str, cfg: dict) -> None:
    label = cfg["label"]
    source = key_source(provider)

    st.markdown(f"##### {label} API Key")
    st.caption(cfg["scope_note"])

    if source == "env":
        current_key = get_key(provider) or ""
        st.success(
            f"Key active from environment variable "
            f"(`{_env_name(provider)}`): **{mask_key(current_key)}**  \n"
            "To change it, update the environment variable and restart the app."
        )
        return

    if source == "file":
        current_key = get_key(provider) or ""
        col_status, col_clear = st.columns([4, 1])
        col_status.success(
            f"Key stored in .env.local: **{mask_key(current_key)}**"
        )
        if col_clear.button("Clear", key=f"clear_{provider}", type="secondary"):
            clear_key(provider)
            st.success(f"{label} key removed.")
            st.rerun()

    with st.form(f"key_form_{provider}", clear_on_submit=True):
        action = "Update" if source == "file" else "Save"
        new_key = st.text_input(
            f"{action} {label} key",
            type="password",
            placeholder=cfg["key_hint"],
            help="The key is stored in .env.local (0o600) — never in the database.",
        )
        if st.form_submit_button(f"{action} Key", type="primary"):
            if not new_key.strip():
                st.error("Key cannot be empty.")
            else:
                err = validate_key_format(provider, new_key)
                if err:
                    st.error(err)
                else:
                    set_key(provider, new_key.strip())
                    st.success(
                        f"{label} key saved. "
                        f"Stored as: **{mask_key(new_key.strip())}**"
                    )
                    st.rerun()


def _render_poll_section(provider: str, cfg: dict) -> None:
    label = cfg["label"]
    st.markdown(f"##### Poll {label} Usage")

    cursor = get_poll_cursor(provider)
    if cursor:
        last_dt = datetime.fromisoformat(cursor["last_polled"])
        st.caption(
            f"Last polled: **{last_dt.strftime('%Y-%m-%d %H:%M UTC')}**  "
            f"· range {cursor['since_date'][:10]} → {cursor['until_date'][:10]}"
        )
        seconds_ago = (datetime.utcnow() - last_dt).total_seconds()
        if seconds_ago < _MIN_POLL_SECONDS:
            remaining = int(_MIN_POLL_SECONDS - seconds_ago)
            st.warning(
                f"Please wait {remaining}s before polling again "
                "(rate-limit protection)."
            )
            return

    today = date.today()
    c1, c2 = st.columns(2)
    since = c1.date_input(
        "From",
        value=today - timedelta(days=30),
        max_value=today,
        key=f"since_{provider}",
    )
    until = c2.date_input(
        "To",
        value=today,
        max_value=today,
        key=f"until_{provider}",
    )

    if since > until:
        st.error("'From' must be before 'To'.")
        return

    if st.button(f"Poll {label} now", type="primary", key=f"poll_{provider}"):
        _run_poll(provider, cfg, since, until)


def _run_poll(provider: str, cfg: dict, since: date, until: date) -> None:
    label = cfg["label"]
    api_key = get_key(provider)
    if not api_key:
        st.error("No API key found.")
        return

    with st.spinner(f"Fetching {label} usage {since} → {until}…"):
        try:
            entries, errors = cfg["poller"](api_key, since=since, until=until)
        except Exception as exc:
            st.error(f"Unexpected error: {type(exc).__name__}: {exc}")
            log.exception("Poll failed for provider=%s", provider)
            return

    now = datetime.utcnow()
    set_poll_cursor(
        provider,
        last_polled=now,
        since_date=datetime.combine(since, datetime.min.time()),
        until_date=datetime.combine(until, datetime.max.time().replace(microsecond=0)),
    )

    if entries:
        inserted = insert_entries_bulk(entries)
        skipped = len(entries) - inserted
        msg = f"Inserted **{inserted:,}** new entries."
        if skipped:
            msg += f" {skipped:,} duplicate(s) skipped."
        st.success(msg)
    else:
        st.info("No new usage data returned for this date range.")

    if errors:
        st.markdown("##### Warnings / errors")
        for err in errors:
            safe_err = html.escape(err)
            st.markdown(
                f'<div class="finding-medium">{safe_err}</div>',
                unsafe_allow_html=True,
            )

    if entries or not errors:
        st.caption(
            "Refresh the Burn Map tab to see updated charts. "
            "Poll again tomorrow to capture today's final usage."
        )


# ── Utilities ────────────────────────────────────────────────────────────────

def _env_name(provider: str) -> str:
    return "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY"
