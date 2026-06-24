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
    "api.cursor.com",
})

# ── Key format patterns (sanity check only, not a security boundary) ─────────
# Anthropic keys: sk-ant-api03-<base64url, 93+ chars>
_ANTHROPIC_KEY_RE = re.compile(r"^sk-ant-api\d{2}-[A-Za-z0-9_-]{80,}$")
# OpenAI keys: sk-<random> or sk-proj-<random>
_OPENAI_KEY_RE = re.compile(r"^sk-(?:proj-)?[A-Za-z0-9_-]{20,}$")
# Cursor admin keys: crsr_<64 hex/base64 chars>
_CURSOR_KEY_RE = re.compile(r"^crsr_[A-Za-z0-9_-]{32,}$")
# Gemini / Google AI Studio keys: AIzaSy<33 chars>
_GEMINI_KEY_RE = re.compile(r"^AIzaSy[A-Za-z0-9_-]{33}$")

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
    elif provider == "cursor":
        if not _CURSOR_KEY_RE.match(key):
            return (
                "Cursor admin keys start with 'crsr_'. "
                "Create one at cursor.com/dashboard → Settings → Advanced → Admin API Keys. "
                "Requires a Team or Business plan."
            )
    elif provider == "gemini":
        if not _GEMINI_KEY_RE.match(key):
            return (
                "Gemini API keys start with 'AIzaSy' and are 39 chars total. "
                "Create one at aistudio.google.com → Get API key."
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


def safe_post(
    url: str,
    headers: dict[str, str],
    json: dict | None = None,
    params: dict[str, str] | None = None,
    auth: tuple[str, str] | None = None,
) -> httpx.Response:
    """
    POST `url` — only if its host is in _ALLOWED_HOSTS.

    `auth` is a (username, password) tuple for HTTP Basic Auth.
    SSL verification always on; redirects never followed.
    """
    host = urlparse(url).hostname or ""
    if host not in _ALLOWED_HOSTS:
        raise ValueError(
            f"Blocked request to disallowed host {host!r}. "
            "Only api.anthropic.com, api.openai.com and api.cursor.com are permitted."
        )
    httpx_auth = httpx.BasicAuth(*auth) if auth else None
    with httpx.Client(
        verify=True,
        timeout=_REQUEST_TIMEOUT,
        follow_redirects=False,
    ) as client:
        return client.post(
            url, headers=headers, json=json, params=params, auth=httpx_auth
        )
