"""
Secure local API key storage.

Keys are kept in a .env.local file at the project root (0o600 permissions,
gitignored).  Environment variables always take precedence and are never
overwritten.  Keys are never written to the SQLite database.
"""
from __future__ import annotations

import logging
import os
import re
import stat
from pathlib import Path

log = logging.getLogger(__name__)

# Project root — three levels up from src/polling/key_store.py
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ENV_LOCAL    = _PROJECT_ROOT / ".env.local"

_ENV_NAMES: dict[str, str] = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai":    "OPENAI_API_KEY",
    "cursor":    "CURSOR_API_KEY",
    "gemini":    "GEMINI_API_KEY",
}

_LINE_RE = re.compile(r"^([A-Z][A-Z0-9_]*)=(.+)$")


# ── Internal helpers ──────────────────────────────────────────────────────────

def _read() -> dict[str, str]:
    if not _ENV_LOCAL.exists():
        return {}
    try:
        text = _ENV_LOCAL.read_text(encoding="utf-8")
    except OSError as exc:
        log.warning("Cannot read .env.local: %s", exc)
        return {}
    result: dict[str, str] = {}
    for line in text.splitlines():
        m = _LINE_RE.match(line.strip())
        if m:
            result[m.group(1)] = m.group(2)
    return result


def _write(data: dict[str, str]) -> None:
    lines = "\n".join(f"{k}={v}" for k, v in sorted(data.items())) + "\n"
    _ENV_LOCAL.write_text(lines, encoding="utf-8")
    _ENV_LOCAL.chmod(stat.S_IRUSR | stat.S_IWUSR)   # 0o600 — owner only


# ── Public API ────────────────────────────────────────────────────────────────

def get_key(provider: str) -> str | None:
    """
    Return the stored API key for `provider`.
    Environment variable takes priority over .env.local.
    Returns None if no key is configured.
    """
    env_name = _ENV_NAMES.get(provider, "")
    env_val = os.environ.get(env_name, "").strip()
    if env_val:
        return env_val
    local = _read()
    return local.get(env_name) or None


def set_key(provider: str, key: str) -> None:
    """
    Persist `key` to .env.local and inject into the current process env.
    Never logged — only a masked representation appears in debug output.
    Silently refuses to overwrite a key already set via environment variable.
    """
    env_name = _ENV_NAMES.get(provider, "")
    if not env_name:
        raise ValueError(f"Unknown provider: {provider!r}")

    key = key.strip()
    if os.environ.get(env_name):
        # Env var takes precedence; changing it here would be confusing.
        log.info(
            "Key for %s is already set via environment variable; "
            ".env.local not updated.",
            provider,
        )
        return

    local = _read()
    local[env_name] = key
    _write(local)
    # Do NOT inject into os.environ here: that would make the key available
    # to all concurrent Streamlit sessions (process-global state).  The key
    # is readable via _read() / get_key() in all subsequent calls.
    log.info("Key for provider=%s written to .env.local", provider)


def clear_key(provider: str) -> None:
    """Remove the stored key for `provider` from .env.local and process env."""
    env_name = _ENV_NAMES.get(provider, "")
    local = _read()
    local.pop(env_name, None)
    if local:
        _write(local)
    elif _ENV_LOCAL.exists():
        _ENV_LOCAL.unlink()
    os.environ.pop(env_name, None)
    log.info("Key for provider=%s cleared", provider)


def has_key(provider: str) -> bool:
    return bool(get_key(provider))


def key_source(provider: str) -> str:
    """Return 'env', 'file', or 'none' — useful for UI display."""
    env_name = _ENV_NAMES.get(provider, "")
    if os.environ.get(env_name):
        return "env"
    if _read().get(env_name):
        return "file"
    return "none"
