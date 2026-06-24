"""
OpenAI Usage API poller.

Endpoint: GET https://api.openai.com/v1/usage?date=YYYY-MM-DD
Auth:     Authorization: Bearer <key>
Docs:     https://platform.openai.com/docs/api-reference/usage

Iterates day-by-day over [since, until] to build a complete picture.
Stops early on 401/403 to avoid lockout; rate-limit 429 per day is skipped
with a warning rather than aborting the full range.

API key value is never logged.
"""
from __future__ import annotations

import logging
from datetime import date, datetime, timedelta
from typing import Any

import httpx

from src.models import EntrySource, Provider, SpendEntry
from src.parsers.base import infer_is_local, infer_workload_class
from src.polling.base import mask_key, safe_get

log = logging.getLogger(__name__)

_USAGE_URL = "https://api.openai.com/v1/usage"


def _auth_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "accept": "application/json",
    }


def poll(
    api_key: str,
    since: date | None = None,
    until: date | None = None,
) -> tuple[list[SpendEntry], list[str]]:
    """
    Fetch daily usage from OpenAI for each day in [since, until].
    Returns (entries, errors).  api_key value is never logged.
    """
    if since is None:
        since = date.today() - timedelta(days=30)
    if until is None:
        until = date.today()

    all_entries: list[SpendEntry] = []
    all_errors:  list[str] = []

    current = since
    while current <= until:
        entries, errors = _poll_day(api_key, current)
        all_entries.extend(entries)
        # Auth errors: stop immediately — polling more days won't help.
        if any("401" in e or "403" in e for e in errors):
            all_errors.extend(errors)
            break
        all_errors.extend(errors)
        current += timedelta(days=1)

    log.info(
        "OpenAI poll: %d entries fetched, %d errors (key=%s)",
        len(all_entries), len(all_errors), mask_key(api_key),
    )
    return all_entries, all_errors


def _poll_day(api_key: str, day: date) -> tuple[list[SpendEntry], list[str]]:
    errors: list[str] = []
    entries: list[SpendEntry] = []

    try:
        resp = safe_get(
            _USAGE_URL,
            headers=_auth_headers(api_key),
            params={"date": day.strftime("%Y-%m-%d")},
        )
    except ValueError as exc:
        return entries, [str(exc)]
    except httpx.TimeoutException:
        return entries, [f"Timeout fetching {day} — skipped."]
    except httpx.RequestError as exc:
        return entries, [f"Network error for {day}: {type(exc).__name__}"]

    day_errors = _check_status(resp, day)
    if day_errors:
        return entries, day_errors

    try:
        body = resp.json()
    except Exception:
        return entries, [f"Could not parse response for {day}."]

    # OpenAI wraps completions in 'data'; DALL-E/Whisper/TTS in side lists.
    # We currently only pull completion (LLM) usage.
    for item in body.get("data", []):
        try:
            entry = _to_entry(item, day)
            if entry is not None:
                entries.append(entry)
        except Exception as exc:
            log.debug("Skipped OpenAI item %s: %s", item, exc)

    return entries, errors


def _check_status(resp: httpx.Response, day: date) -> list[str]:
    if resp.is_success:
        return []
    code = resp.status_code
    if code == 401:
        return [
            "Invalid API key (401). Verify your OpenAI key at platform.openai.com/api-keys."
        ]
    if code == 403:
        return [
            "Permission denied (403). Your key may not have usage read access. "
            "In the OpenAI platform, ensure the key has no 'Read usage data' restriction disabled."
        ]
    if code == 429:
        return [f"Rate limited (429) for {day} — skipped."]
    return [f"API error {code} for {day}: {resp.text[:200]}"]


def _to_entry(item: dict[str, Any], day: date) -> SpendEntry | None:
    model: str = (
        item.get("snapshot_id")
        or item.get("model")
        or item.get("engine")
        or ""
    )
    if not model:
        return None

    input_tok  = int(item.get("n_context_tokens_total",   0) or 0)
    output_tok = int(item.get("n_generated_tokens_total", 0) or 0)
    # n_cached_context_tokens_total is a subset of input already counted;
    # store as reasoning_tokens to surface cache-hit volume in the UI.
    cached_tok = int(item.get("n_cached_context_tokens_total", 0) or 0)

    if input_tok == 0 and output_tok == 0:
        return None

    ts_raw = item.get("aggregation_timestamp")
    ts = (
        datetime.utcfromtimestamp(ts_raw)
        if isinstance(ts_raw, int | float)
        else datetime.combine(day, datetime.min.time())
    )

    return SpendEntry(
        timestamp=ts,
        provider=Provider.openai,
        model=model,
        workload_class=infer_workload_class(model),
        input_tokens=input_tok,
        output_tokens=output_tok,
        reasoning_tokens=cached_tok,
        cost_usd=0.0,   # usage endpoint does not return per-model cost
        is_local=infer_is_local(model),
        source=EntrySource.api,
    )
