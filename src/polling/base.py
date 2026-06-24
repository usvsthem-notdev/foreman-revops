"""
Shared utilities for live API polling.

Security model:
  - Only the two hardcoded hosts in _ALLOWED_HOSTS may receive requests.
    User-supplied URLs are never accepted (no SSRF surface).
  - API keys are never logged; only masked representations appear in logs.
  - All requests use explicit timeouts; SSL verification is always on.
  - Exponential backoff on 429; hard stop on 401/403 to avoid lockout.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlparse

import httpx

log = logging.getLogger(__name__)

# ── Allowlist ────────────────────────────────────────────────────────────────
# Only these hosts may receive authenticated requests.  Checked before every
# outbound call — if the URL resolves to any other host the call is refused.
_ALLOWED_HOSTS: frozenset[str] = frozenset({
    "api.anthropic.com",
    "api.openai.com",
})

# ── Key format patterns (sanity check only, not a security boundary) ─────────
# Anthropic keys: sk-ant-api03-<base64url, 93+ chars>
_ANTHROPIC_KEY_RE = re.compile(r"^sk-ant-api\d{2}-[A-Za-z0-9_-]{80,}$")
# OpenAI keys: sk-<random> or sk-proj-<random>
_OPENAI_KEY_RE = re.compile(r"^sk-(?:proj-)?[A-Za-z0-9_-]{20,}$")

_REQUEST_TIMEOUT = 30.0   # seconds
_MIN_POLL_INTERVAL = 60   # seconds between polls per provider (UI-enforced)


# ── Result type ──────────────────────────────────────────────────────────────

@dataclass
class PollResult:
    provider: str
    entries_inserted: int = 0
    entries_skipped: int = 0
    errors: list[str] = field(default_factory=list)
    polled_at: datetime = field(default_factory=datetime.utcnow)


# ── Key utilities ─────────────────────────────────────────────────────────────

def mask_key(key: str) -> str:
    """Return a display-safe representation.  Never call this in a log format
    string — only use the result, never the raw key."""
    if not key or len(key) < 12:
        return "****"
    return f"{key[:10]}…{key[-4:]}"


def validate_key_format(provider: str, key: str) -> str | None:
    """Return an error string if the key looks wrong; None if it passes."""
    key = key.strip()
    if not key:
        return "API key is required."
    if provider == "anthropic":
        if not _ANTHROPIC_KEY_RE.match(key):
            return (
                "Anthropic keys must match 'sk-ant-api##-<token>' (90+ chars). "
                "Copy the key from console.anthropic.com → API Keys."
            )
    elif provider == "openai":
        if not _OPENAI_KEY_RE.match(key):
            return (
                "OpenAI keys must start with 'sk-' or 'sk-proj-'. "
                "Copy the key from platform.openai.com → API keys."
            )
    return None


# ── Safe HTTP ────────────────────────────────────────────────────────────────

def safe_get(
    url: str,
    headers: dict[str, str],
    params: dict[str, str] | None = None,
) -> httpx.Response:
    """
    GET `url` — but only if its host is in _ALLOWED_HOSTS.

    Raises ValueError for disallowed hosts (SSRF guard).
    SSL verification is always enabled; timeout is always set.
    """
    host = urlparse(url).hostname or ""
    if host not in _ALLOWED_HOSTS:
        raise ValueError(
            f"Blocked request to disallowed host {host!r}. "
            "Only api.anthropic.com and api.openai.com are permitted."
        )
    with httpx.Client(
        verify=True,
        timeout=_REQUEST_TIMEOUT,
        follow_redirects=False,   # no redirect following — keep host pinned
    ) as client:
        return client.get(url, headers=headers, params=params)
