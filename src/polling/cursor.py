"""
Cursor Team Usage API poller.

Endpoint: POST https://api.cursor.com/teams/filtered-usage-events
Auth:     HTTP Basic Auth — admin API key as username, empty password.
          Key format: crsr_<64+ chars>
          Create at: cursor.com/dashboard → Settings → Advanced → Admin API Keys
Docs:     https://cursor.com/docs/account/teams/admin-api

Requires a Cursor Team or Business plan.  Individual accounts are not
supported — the API returns 403 for non-team keys.

Returns per-event token counts and cost in cents.  Iterates pages until
exhausted.  Stops immediately on 401/403 to avoid lockout.
"""
from __future__ import annotations

import logging
from datetime import UTC, date, datetime, timedelta
from typing import Any

import httpx

from src.models import AICategory, EntrySource, Provider, SpendEntry
from src.parsers.base import infer_is_local, infer_workload_class
from src.polling.base import mask_key, safe_post

log = logging.getLogger(__name__)

_BASE_URL   = "https://api.cursor.com"
_EVENTS_URL = f"{_BASE_URL}/teams/filtered-usage-events"
_PAGE_SIZE  = 500


def _auth(api_key: str) -> tuple[str, str]:
    return (api_key, "")


def _ms(d: date) -> int:
    """Convert a date to epoch milliseconds (midnight UTC)."""
    return int(datetime(d.year, d.month, d.day, tzinfo=UTC).timestamp() * 1000)


def poll(
    api_key: str,
    since: date | None = None,
    until: date | None = None,
) -> tuple[list[SpendEntry], list[str]]:
    """
    Fetch per-event usage from the Cursor team API.

    Returns (entries, error_messages).
    api_key value is never logged; only mask_key(api_key) appears in logs.
    """
    if since is None:
        since = date.today() - timedelta(days=30)
    if until is None:
        until = date.today()

    all_entries: list[SpendEntry] = []
    all_errors: list[str] = []

    page = 1
    while True:
        entries, errors, has_more = _fetch_page(api_key, since, until, page)
        all_entries.extend(entries)
        all_errors.extend(errors)

        if any("401" in e or "403" in e for e in errors):
            break
        if not has_more:
            break
        page += 1

    log.info(
        "Cursor poll: %d entries fetched, %d errors (key=%s)",
        len(all_entries), len(all_errors), mask_key(api_key),
    )
    return all_entries, all_errors


def _fetch_page(
    api_key: str,
    since: date,
    until: date,
    page: int,
) -> tuple[list[SpendEntry], list[str], bool]:
    """Fetch one page.  Returns (entries, errors, has_more)."""
    payload = {
        "startDate": _ms(since),
        "endDate":   _ms(until + timedelta(days=1)),  # end of 'until' day
        "page":      page,
        "pageSize":  _PAGE_SIZE,
    }

    try:
        resp = safe_post(
            _EVENTS_URL,
            headers={"accept": "application/json"},
            json=payload,
            auth=_auth(api_key),
        )
    except ValueError as exc:
        return [], [str(exc)], False
    except httpx.TimeoutException:
        return [], ["Request timed out. Check your network and try again."], False
    except httpx.RequestError as exc:
        return [], [f"Network error: {type(exc).__name__}"], False

    errors = _check_status(resp)
    if errors:
        return [], errors, False

    try:
        body = resp.json()
    except Exception:
        return [], ["Could not parse API response as JSON."], False

    raw_events: list[Any] = body.get("usageEvents", [])
    if not isinstance(raw_events, list):
        return [], [f"Unexpected response shape: {str(body)[:200]}"], False

    entries: list[SpendEntry] = []
    for event in raw_events:
        try:
            entry = _to_entry(event)
            if entry is not None:
                entries.append(entry)
        except Exception as exc:
            log.debug("Skipped Cursor event %s: %s", event, exc)

    # Paginate while we get a full page
    has_more = len(raw_events) == _PAGE_SIZE
    return entries, [], has_more


def _check_status(resp: httpx.Response) -> list[str]:
    if resp.is_success:
        return []
    code = resp.status_code
    if code == 401:
        return [
            "Invalid API key (401). Verify your Cursor admin key at "
            "cursor.com/dashboard → Settings → Advanced → Admin API Keys."
        ]
    if code == 403:
        return [
            "Permission denied (403). The Cursor usage API requires a Team or Business plan "
            "and an admin-level API key (crsr_...). Individual accounts are not supported."
        ]
    if code == 429:
        return ["Rate limited (429). Wait before polling again (100 req/min limit)."]
    return [f"API error {code}: {resp.text[:200]}"]


def _to_entry(event: dict[str, Any]) -> SpendEntry | None:
    # Only process token-based, chargeable calls
    if not event.get("isTokenBasedCall"):
        return None

    model: str = event.get("model", "") or ""
    if not model:
        return None

    usage = event.get("tokenUsage") or {}
    input_tok  = int(usage.get("inputTokens",      0) or 0)
    output_tok = int(usage.get("outputTokens",     0) or 0)
    cache_tok  = int(usage.get("cacheReadTokens",  0) or 0)

    if input_tok == 0 and output_tok == 0:
        return None

    # totalCents is a float in dollars-as-cents (e.g. 0.05 = $0.05 / 100 = $0.0005)
    # chargedCents is an int in integer cents
    charged = event.get("chargedCents")
    total_c  = usage.get("totalCents")
    if charged is not None:
        cost_usd = int(charged) / 100
    elif total_c is not None:
        cost_usd = float(total_c) / 100
    else:
        cost_usd = 0.0

    ts_raw = event.get("timestamp")
    ts = _parse_ts(ts_raw)
    if ts is None:
        return None

    user_email = event.get("userEmail")
    feature    = event.get("kind", "")    # "chat", "agent", "cmd", etc.

    return SpendEntry(
        timestamp=ts,
        provider=Provider.cursor,
        model=model,
        workload_class=infer_workload_class(model, feature),
        input_tokens=input_tok,
        output_tokens=output_tok,
        reasoning_tokens=cache_tok,
        cost_usd=round(cost_usd, 8),
        is_local=infer_is_local(model),
        team=user_email,           # kept for backwards compat with existing filters
        feature=feature,
        source=EntrySource.cursor_api,
        user_id=user_email,        # canonical user attribution field
        ai_category=AICategory.code_gen,  # Cursor is code gen by definition
    )


def _parse_ts(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, int | float):
        # Cursor timestamps are in milliseconds
        return datetime.utcfromtimestamp(raw / 1000)
    s = str(raw).strip().rstrip("Z").replace("T", " ").split("+")[0].split(".")[0]
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None
