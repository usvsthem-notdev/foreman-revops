"""
Anthropic Usage API poller.

Endpoint: GET https://api.anthropic.com/v1/usage/models
Auth:     x-api-key header  (key with usage:read / billing scope)
Docs:     https://docs.anthropic.com/en/api/usage

If your key does not have the usage scope, the API returns 403 and this
poller surfaces that with a clear message — no key value is ever logged.
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

_BASE             = "https://api.anthropic.com"
_USAGE_URL        = f"{_BASE}/v1/usage/models"
_ANTHROPIC_VERSION = "2023-06-01"
_MAX_DAYS_PER_CALL = 90


def _auth_headers(api_key: str) -> dict[str, str]:
    return {
        "x-api-key": api_key,
        "anthropic-version": _ANTHROPIC_VERSION,
        "accept": "application/json",
    }


def poll(
    api_key: str,
    since: date | None = None,
    until: date | None = None,
) -> tuple[list[SpendEntry], list[str]]:
    """
    Fetch model-level usage from Anthropic.

    Returns (entries, error_messages).
    The api_key value is never logged; only mask_key(api_key) appears in logs.
    """
    errors: list[str] = []
    entries: list[SpendEntry] = []

    if since is None:
        since = date.today() - timedelta(days=30)
    if until is None:
        until = date.today()

    # Anthropic accepts ISO-8601 timestamps as start_time / end_time.
    params = {
        "start_time": f"{since.isoformat()}T00:00:00Z",
        "end_time":   f"{until.isoformat()}T23:59:59Z",
    }

    try:
        resp = safe_get(_USAGE_URL, headers=_auth_headers(api_key), params=params)
    except ValueError as exc:
        errors.append(str(exc))
        return entries, errors
    except httpx.TimeoutException:
        errors.append("Request timed out. Check your network and try again.")
        return entries, errors
    except httpx.RequestError as exc:
        errors.append(f"Network error: {type(exc).__name__}")
        return entries, errors

    errors.extend(_check_status(resp))
    if errors:
        return entries, errors

    try:
        body = resp.json()
    except Exception:
        errors.append("Could not parse API response as JSON.")
        return entries, errors

    raw_items: list[Any] = body.get("data", body.get("models", []))
    if not isinstance(raw_items, list):
        errors.append(f"Unexpected response shape — missing 'data' list: {str(body)[:200]}")
        return entries, errors

    for item in raw_items:
        try:
            entry = _to_entry(item)
            if entry is not None:
                entries.append(entry)
        except Exception as exc:
            log.debug("Skipped Anthropic item %s: %s", item, exc)

    log.info(
        "Anthropic poll: %d entries fetched, %d errors (key=%s)",
        len(entries), len(errors), mask_key(api_key),
    )
    return entries, errors


def _check_status(resp: httpx.Response) -> list[str]:
    if resp.is_success:
        return []
    code = resp.status_code
    if code == 401:
        return ["Invalid API key (401). Verify your Anthropic key in console.anthropic.com."]
    if code == 403:
        return [
            "Permission denied (403). The usage API requires a key with billing/usage "
            "read scope. In the Anthropic console, create a key with 'usage:read' permission "
            "or use an admin key. If your plan does not expose the usage API, use the CSV "
            "export from the Bill Analyzer tab instead."
        ]
    if code == 404:
        return [
            "Usage endpoint not found (404). This endpoint may not be available for your "
            "account tier. Use the Anthropic console → Billing → Export CSV and import it "
            "in the Bill Analyzer tab."
        ]
    if code == 429:
        return ["Rate limited (429). Wait a minute before polling again."]
    return [f"API error {code}: {resp.text[:200]}"]


def _to_entry(item: dict[str, Any]) -> SpendEntry | None:
    model: str = (
        item.get("model")
        or item.get("snapshot_id")
        or item.get("model_id")
        or ""
    )
    if not model:
        return None

    input_tok  = int(item.get("input_tokens",  item.get("n_context_tokens_total",   0)) or 0)
    output_tok = int(item.get("output_tokens", item.get("n_generated_tokens_total",  0)) or 0)
    # Anthropic's usage API has no reasoning-token concept (extended thinking
    # is billed as output) — these are real prompt-cache counts, not reasoning.
    cache_read_tok = int(item.get("cache_read_tokens", 0) or 0)
    cache_creation_tok = int(item.get("cache_creation_tokens", 0) or 0)
    # The API reports input_tokens as fresh-only, with cache tokens as
    # separate additive fields — fold them in so input_tokens matches this
    # app's "cache tokens are a subset of input_tokens" convention (same as
    # the CSV parser's effective_input).
    input_tok += cache_read_tok + cache_creation_tok

    if input_tok == 0 and output_tok == 0:
        return None

    # Timestamps may arrive as ISO strings or Unix ints.
    ts_raw = item.get("timestamp") or item.get("aggregation_timestamp") or item.get("date")
    ts = _parse_ts(ts_raw)
    if ts is None:
        return None

    cost = float(item.get("cost_usd", item.get("cost", 0.0)) or 0.0)

    return SpendEntry(
        timestamp=ts,
        provider=Provider.anthropic,
        model=model,
        workload_class=infer_workload_class(model),
        input_tokens=input_tok,
        output_tokens=output_tok,
        reasoning_tokens=0,
        cache_read_tokens=cache_read_tok,
        cache_creation_tokens=cache_creation_tok,
        cost_usd=cost,
        is_local=infer_is_local(model),
        source=EntrySource.api,
    )


def _parse_ts(raw: Any) -> datetime | None:
    if raw is None:
        return None
    if isinstance(raw, int | float):
        return datetime.utcfromtimestamp(raw)
    s = str(raw).strip().rstrip("Z").replace("T", " ").split("+")[0].split(".")[0]
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(s, fmt)
        except ValueError:
            continue
    return None
