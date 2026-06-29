"""
Content policy for user-supplied and externally-sourced strings.

Inspired by the markspace ContentPolicy pattern (ALLOW / REWRITE / REJECT / FLAG).
Applied at the model-validation boundary so injection payloads never reach storage or
an LLM context regardless of which ingestion path brought them in.

Verdicts:
  ALLOW  — clean, pass through unchanged
  REWRITE — stripped/escaped to a safe form; logged at DEBUG
  REJECT  — refused entirely; raises ValueError so Pydantic surfaces it to the caller
  FLAG    — allowed but logged at WARNING for operator review
"""
from __future__ import annotations

import logging
import re
import unicodedata
from enum import Enum
from typing import NamedTuple

log = logging.getLogger(__name__)


class Verdict(str, Enum):
    ALLOW   = "ALLOW"
    REWRITE = "REWRITE"
    REJECT  = "REJECT"
    FLAG    = "FLAG"


class PolicyResult(NamedTuple):
    verdict: Verdict
    value: str
    reason: str = ""


# Patterns that look like prompt-injection attempts.
# We don't claim to be exhaustive — the goal is to flag obvious attacks and
# rewrite/reject them before they reach a downstream LLM call.
_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+)?previous\s+instructions?", re.I),
    re.compile(r"you\s+are\s+now\s+(a|an)\s+", re.I),
    re.compile(r"system\s*prompt", re.I),
    re.compile(r"<\|.*?\|>"),                           # OpenAI special tokens
    re.compile(r"\[\[.*?\]\]"),                          # Llama instruction markers
    re.compile(r"###\s*(Instruction|System|Human|Assistant)\b", re.I),
]

# Control characters that have no place in display strings.
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")


def _strip_controls(s: str) -> str:
    """Remove ASCII control characters (keep tab \x09, newline \x0a, CR \x0d)."""
    return _CTRL_RE.sub("", s)


def _has_injection(s: str) -> bool:
    return any(p.search(s) for p in _INJECTION_PATTERNS)


def apply(value: str, *, field: str = "", max_length: int = 512) -> PolicyResult:
    """
    Evaluate a single string value against the content policy.

    Parameters
    ----------
    value:      The candidate string.
    field:      Field name — used in log messages only.
    max_length: Hard cap enforced via REJECT.
    """
    if not isinstance(value, str):
        return PolicyResult(Verdict.ALLOW, value)

    # 1. Normalize unicode (NFC) and strip ASCII control characters
    normalized = unicodedata.normalize("NFC", value)
    cleaned = _strip_controls(normalized)
    if cleaned != value:
        log.debug("ContentPolicy REWRITE field=%r: stripped control characters", field)
        return PolicyResult(Verdict.REWRITE, cleaned, "control characters stripped")

    # 2. Hard length cap
    if len(value) > max_length:
        log.warning("ContentPolicy REJECT field=%r: length %d > %d", field, len(value), max_length)
        raise ValueError(f"{field}: value exceeds maximum length of {max_length} characters")

    # 3. Prompt injection patterns — FLAG (allow but warn; do not silently store without logging)
    if _has_injection(value):
        log.warning(
            "ContentPolicy FLAG field=%r: possible prompt injection payload detected — "
            "value stored but flagged for review: %r",
            field, value[:120],
        )
        return PolicyResult(Verdict.FLAG, value, "possible prompt injection")

    return PolicyResult(Verdict.ALLOW, value)


def sanitize(value: str | None, *, field: str = "", max_length: int = 512) -> str | None:
    """
    Convenience wrapper: apply the policy and return the (possibly rewritten) value.

    Call this at every trust boundary — model validators, API response parsers,
    CSV importers — before writing to the database or passing to an LLM.

    Raises ValueError on REJECT so Pydantic surfaces it cleanly.
    Returns None unchanged.
    """
    if value is None:
        return None
    result = apply(value, field=field, max_length=max_length)
    return result.value
