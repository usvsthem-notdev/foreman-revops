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
import os
from datetime import date, datetime, timedelta

import streamlit as st

from src.db import get_poll_cursor, insert_entries_bulk, set_poll_cursor
from src.polling import anthropic as anthropic_poller
from src.polling import cursor as cursor_poller
from src.polling import openai as openai_poller
from src.polling.base import mask_key, validate_key_format
from src.polling.key_store import clear_key, get_key, has_key, key_source, set_key
from src.polling.scheduler import read_heartbeat

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
    "cursor": {
        "label":  "Cursor",
        "poller": cursor_poller.poll,
        "key_hint": "crsr_…",
        "scope_note": (
            "Requires a **Team or Business plan** and an admin API key. "
            "Create one at [cursor.com/dashboard](https://cursor.com/dashboard) "
            "→ Settings → Advanced → Admin API Keys."
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

    # HuggingFace Spaces sets SPACE_ID — the filesystem is shared across all
    # browser sessions, so storing keys in .env.local would expose them to
    # anyone who visits the URL.  Key entry is disabled; owners should use
    # HuggingFace Space secrets (Settings → Variables and secrets) which land
    # in os.environ as private, per-Space environment variables.
    if os.environ.get("SPACE_ID"):
        st.error(
            "**Key entry is disabled on shared deployments.**  \n"
            "Set your API keys as **Space secrets** in the HuggingFace settings "
            "(Settings → Variables and secrets). Secret values are injected as "
            "environment variables and are never visible to visitors.",
            icon="🔒",
        )
        _render_scheduler_status()
        return

    _render_scheduler_status()
    st.divider()

    tab_anthropic, tab_openai, tab_cursor, tab_gemini = st.tabs(
        ["Anthropic", "OpenAI", "Cursor", "Gemini"]
    )

    with tab_anthropic:
        _render_provider("anthropic")

    with tab_openai:
        _render_provider("openai")

    with tab_cursor:
        _render_provider("cursor")

    with tab_gemini:
        _render_gemini_info()


# ── Scheduler status banner ──────────────────────────────────────────────────

def _render_scheduler_status() -> None:
    """Show whether the background scheduler is running and when it last ran."""
    hb = read_heartbeat()
    if hb is None:
        st.info(
            "**Auto-poll scheduler is not running.**  \n"
            "Start it with `python scheduler.py` (or via Docker Compose) "
            "to poll automatically on a schedule.  \n"
            "Use the manual poll buttons below to fetch data on demand."
        )
        return

    try:
        last_ts = datetime.fromisoformat(hb["timestamp"])
        age_minutes = (datetime.utcnow() - last_ts).total_seconds() / 60
        providers = hb.get("providers", "—")
        errors = int(hb.get("errors", 0))
    except (KeyError, ValueError):
        st.warning("Scheduler heartbeat file exists but could not be parsed.")
        return

    interval_h = int(os.environ.get("FOREMAN_POLL_INTERVAL_HOURS", 6))
    stale_threshold_minutes = interval_h * 60 * 1.5  # 1.5× interval = overdue

    if age_minutes > stale_threshold_minutes:
        st.warning(
            f"**Scheduler may have stopped.** Last heartbeat: "
            f"{last_ts.strftime('%Y-%m-%d %H:%M UTC')} "
            f"({age_minutes:.0f} min ago) — expected within {interval_h * 60:.0f} min."
        )
    else:
        err_note = f" · {errors} error(s) last cycle" if errors else ""
        st.success(
            f"**Auto-poll scheduler running.**  "
            f"Last ran: {last_ts.strftime('%Y-%m-%d %H:%M UTC')} "
            f"({age_minutes:.0f} min ago){err_note}  \n"
            f"Providers: `{providers}` · "
            f"Interval: `{interval_h}h` · "
            f"Lookback: `{os.environ.get('FOREMAN_POLL_LOOKBACK_DAYS', 2)}d`"
        )


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
        # Auto-clear a key that has been revoked so the scheduler stops retrying it
        auth_failure = any("401" in e or "403" in e for e in errors)
        if auth_failure:
            clear_key(provider)
            st.error(
                f"Authentication failed — the {label} key has been removed. "
                "Enter a new key below."
            )

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


# ── Gemini info panel ─────────────────────────────────────────────────────────

def _render_gemini_info() -> None:
    st.markdown("##### Gemini / Google AI Studio")
    st.info(
        "**Google does not expose a usage API for AI Studio keys.**  \n"
        "The Generative Language API (`AIzaSy…` keys) returns token counts "
        "inline with each inference call but provides no historical usage endpoint.  \n\n"
        "**To import Gemini spend data:**  \n"
        "1. In [Google Cloud Console](https://console.cloud.google.com) → Billing → "
        "Export, configure a BigQuery billing export.  \n"
        "2. Query the `gcp_billing_export_v1_*` table filtering on the Gemini SKU "
        "and download a CSV.  \n"
        "3. Import it in the **Bill Analyzer** tab.  \n\n"
        "You can also add Gemini entries manually via the **Add Entry** tab and "
        "select *Gemini* as the provider."
    )
    st.markdown("##### Recognized Gemini models")
    st.caption(
        "These models are already recognized for workload-class inference and charting:"
    )
    st.code(
        "gemini-2.5-pro\n"
        "gemini-2.5-flash\n"
        "gemini-2.0-flash\n"
        "gemini-1.5-pro\n"
        "gemini-1.5-flash\n"
        "gemini-1.0-pro",
        language=None,
    )


# ── Utilities ────────────────────────────────────────────────────────────────

_ENV_NAME_MAP = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai":    "OPENAI_API_KEY",
    "cursor":    "CURSOR_API_KEY",
    "gemini":    "GEMINI_API_KEY",
}


def _env_name(provider: str) -> str:
    return _ENV_NAME_MAP.get(provider, f"{provider.upper()}_API_KEY")
