"""
Foreman auto-poll scheduler.

Runs as a long-lived process alongside the Streamlit app.  On each tick it
checks whether each configured provider is due for a poll (based on
FOREMAN_POLL_INTERVAL_HOURS) and, if so, fetches the last
FOREMAN_POLL_LOOKBACK_DAYS worth of usage data and inserts new entries.

Configuration — all via environment variables, no file writes required:

  FOREMAN_POLL_INTERVAL_HOURS   How often to poll each provider.
                                 Default: 6.  Minimum: 1.
  FOREMAN_POLL_PROVIDERS        Comma-separated provider names.
                                 Default: "anthropic,openai".
  FOREMAN_POLL_LOOKBACK_DAYS    Days of history fetched on every run.
                                 Default: 2 (catches any late-arriving data
                                 from the previous day).  Maximum: 7.
  ANTHROPIC_API_KEY             Required when "anthropic" is in providers.
  OPENAI_API_KEY                Required when "openai" is in providers.
  FOREMAN_DB_PATH               Path to the SQLite database (shared with app).

Security:
  - All HTTP calls go through polling.base.safe_get (host allowlist + SSL).
  - Key values are never logged; only mask_key() output appears in logs.
  - The scheduler refuses to start if no keys are configured and logs a clear
    message rather than silently doing nothing.
  - A heartbeat file is written after each successful cycle so the UI can
    display scheduler liveness without requiring IPC.
"""
from __future__ import annotations

import logging
import os
import signal
import time
from datetime import date, datetime, timedelta
from pathlib import Path

from src.db import get_poll_cursor, insert_entries_bulk, set_poll_cursor
from src.polling import anthropic as anthropic_poller
from src.polling import openai as openai_poller
from src.polling.base import mask_key
from src.polling.key_store import get_key

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

_MIN_INTERVAL_HOURS  = 1
_MAX_LOOKBACK_DAYS   = 7
_HEARTBEAT_FILENAME  = ".scheduler_heartbeat"


def load_config() -> dict:
    """
    Read scheduler settings from environment variables.
    Returns a validated config dict; raises ValueError with a clear message
    if the configuration is unusable (no valid providers).
    """
    raw_interval = os.environ.get("FOREMAN_POLL_INTERVAL_HOURS", "6")
    try:
        interval_hours = max(_MIN_INTERVAL_HOURS, int(raw_interval))
    except ValueError:
        log.warning(
            "FOREMAN_POLL_INTERVAL_HOURS=%r is not an integer; using default 6.",
            raw_interval,
        )
        interval_hours = 6

    raw_lookback = os.environ.get("FOREMAN_POLL_LOOKBACK_DAYS", "2")
    try:
        lookback_days = max(1, min(_MAX_LOOKBACK_DAYS, int(raw_lookback)))
    except ValueError:
        log.warning(
            "FOREMAN_POLL_LOOKBACK_DAYS=%r is not an integer; using default 2.",
            raw_lookback,
        )
        lookback_days = 2

    raw_providers = os.environ.get("FOREMAN_POLL_PROVIDERS", "anthropic,openai")
    requested = [p.strip().lower() for p in raw_providers.split(",") if p.strip()]

    _pollers = {
        "anthropic": anthropic_poller.poll,
        "openai":    openai_poller.poll,
    }

    # Only include providers that both have a registered poller AND a key.
    active: dict[str, object] = {}
    for provider in requested:
        if provider not in _pollers:
            log.warning("Unknown provider %r — skipped.", provider)
            continue
        key = get_key(provider)
        if not key:
            log.warning(
                "No API key for provider=%s "
                "(set %s or configure it via the Live API tab).",
                provider,
                "ANTHROPIC_API_KEY" if provider == "anthropic" else "OPENAI_API_KEY",
            )
            continue
        active[provider] = _pollers[provider]
        log.info("Provider %s active (key=%s).", provider, mask_key(key))

    if not active:
        raise ValueError(
            "No providers configured with valid API keys. "
            "Set ANTHROPIC_API_KEY and/or OPENAI_API_KEY, then restart the scheduler."
        )

    return {
        "interval_hours": interval_hours,
        "lookback_days":  lookback_days,
        "providers":      active,
    }


# ---------------------------------------------------------------------------
# Heartbeat
# ---------------------------------------------------------------------------

def _heartbeat_path() -> Path:
    db_path = Path(os.environ.get("FOREMAN_DB_PATH", "data/foreman.db"))
    return db_path.parent / _HEARTBEAT_FILENAME


def write_heartbeat(providers_polled: list[str], errors: int) -> None:
    """
    Write a small status file next to the database.
    The Streamlit UI reads this to show scheduler liveness.
    Format is intentionally simple — one key=value per line.
    """
    path = _heartbeat_path()
    lines = [
        f"timestamp={datetime.utcnow().isoformat()}",
        f"providers={','.join(providers_polled)}",
        f"errors={errors}",
    ]
    try:
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    except OSError as exc:
        log.warning("Could not write heartbeat file: %s", exc)


def read_heartbeat() -> dict | None:
    """
    Read the heartbeat file written by the scheduler.
    Returns None if the file does not exist or cannot be parsed.
    """
    path = _heartbeat_path()
    if not path.exists():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    result: dict[str, str] = {}
    for line in text.splitlines():
        if "=" in line:
            k, _, v = line.partition("=")
            result[k.strip()] = v.strip()
    return result or None


# ---------------------------------------------------------------------------
# Core poll logic
# ---------------------------------------------------------------------------

def _due_for_poll(provider: str, interval_hours: int) -> bool:
    """Return True if the provider hasn't been polled within interval_hours."""
    cursor = get_poll_cursor(provider)
    if cursor is None:
        return True
    try:
        last = datetime.fromisoformat(cursor["last_polled"])
    except (KeyError, ValueError):
        return True
    return (datetime.utcnow() - last).total_seconds() >= interval_hours * 3600


def run_once(config: dict) -> tuple[list[str], int]:
    """
    Check each provider and poll if due.
    Returns (providers_polled, total_error_count).
    """
    polled: list[str] = []
    total_errors = 0

    for provider, poller in config["providers"].items():
        if not _due_for_poll(provider, config["interval_hours"]):
            log.debug("Provider %s not due yet — skipping.", provider)
            continue

        key = get_key(provider)
        if not key:
            log.warning("Key for provider=%s disappeared since startup — skipping.", provider)
            continue

        lookback = config["lookback_days"]
        until = date.today()
        since = until - timedelta(days=lookback)

        log.info("Polling %s: %s → %s (key=%s).", provider, since, until, mask_key(key))

        try:
            entries, errors = poller(key, since=since, until=until)
        except Exception as exc:
            log.error("Unexpected error polling %s: %s", provider, exc, exc_info=True)
            total_errors += 1
            continue

        if errors:
            for err in errors:
                log.warning("[%s] %s", provider, err)
            total_errors += len(errors)

        if entries:
            inserted = insert_entries_bulk(entries)
            skipped = len(entries) - inserted
            log.info(
                "%s: inserted=%d skipped=%d (of %d fetched).",
                provider, inserted, skipped, len(entries),
            )
        else:
            log.info("%s: no new entries returned.", provider)

        now = datetime.utcnow()
        set_poll_cursor(
            provider,
            last_polled=now,
            since_date=datetime.combine(since, datetime.min.time()),
            until_date=datetime.combine(until, datetime.max.time().replace(microsecond=0)),
        )
        polled.append(provider)

    return polled, total_errors


# ---------------------------------------------------------------------------
# Scheduler loop
# ---------------------------------------------------------------------------

_SHUTDOWN = False


def _handle_signal(signum: int, _frame: object) -> None:
    global _SHUTDOWN  # noqa: PLW0603
    log.info("Signal %d received — shutting down after current cycle.", signum)
    _SHUTDOWN = True


def run(config: dict, tick_seconds: int = 60) -> None:
    """
    Main loop.  Wakes every `tick_seconds` and checks whether any provider
    is due.  Handles SIGTERM/SIGINT for graceful shutdown.
    """
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT,  _handle_signal)

    interval_h = config["interval_hours"]
    providers  = list(config["providers"])
    log.info(
        "Scheduler started — providers=%s interval=%dh lookback=%dd tick=%ds.",
        providers, interval_h, config["lookback_days"], tick_seconds,
    )

    while not _SHUTDOWN:
        try:
            polled, errors = run_once(config)
            if polled:
                write_heartbeat(polled, errors)
        except Exception as exc:
            log.error("Unexpected scheduler error: %s", exc, exc_info=True)

        # Sleep in short increments so SIGTERM is handled promptly.
        for _ in range(tick_seconds):
            if _SHUTDOWN:
                break
            time.sleep(1)

    log.info("Scheduler stopped cleanly.")
