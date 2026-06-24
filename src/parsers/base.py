"""
Shared parsing utilities and workload class heuristics.
"""
from __future__ import annotations

import re
from datetime import datetime

from src.models import WorkloadClass

# Map model name fragments → workload class heuristic
_MODEL_CLASS_HINTS: list[tuple[re.Pattern, WorkloadClass]] = [
    (re.compile(r"o[13]-?(mini|pro|preview)?", re.I), WorkloadClass.reason),
    (re.compile(r"claude.*(opus|sonnet)", re.I),       WorkloadClass.reason),
    (re.compile(r"claude.*haiku", re.I),               WorkloadClass.extract),
    (re.compile(r"gpt-4o", re.I),                      WorkloadClass.agents),
    (re.compile(r"gpt-4[^o]", re.I),                   WorkloadClass.reason),
    (re.compile(r"gpt-3\.5", re.I),                    WorkloadClass.extract),
    (re.compile(r"embed|text-embed", re.I),            WorkloadClass.rag),
    (re.compile(r"code|codex|starcoder", re.I),        WorkloadClass.coding),
    (re.compile(r"gemini.*(2\.5|ultra|pro)", re.I),     WorkloadClass.reason),
    (re.compile(r"gemini.*(flash|nano)", re.I),        WorkloadClass.extract),
    (re.compile(r"cursor-small", re.I),                WorkloadClass.coding),
    (re.compile(r"mistral.*large", re.I),              WorkloadClass.reason),
    (re.compile(r"mistral.*(7b|small|8x)", re.I),      WorkloadClass.extract),
]

# Models that run locally = absorbed (sage)
_LOCAL_MODEL_PATTERNS = re.compile(
    r"qwen|llama|phi|mistral-local|r1-distill|gemma|deepseek|"
    r"local|on-prem|self-hosted",
    re.I,
)

_MAX_FILE_BYTES = 50 * 1024 * 1024  # 50 MB upload limit


def infer_workload_class(model: str, description: str = "") -> WorkloadClass:
    text = f"{model} {description}"
    for pattern, cls in _MODEL_CLASS_HINTS:
        if pattern.search(text):
            return cls
    return WorkloadClass.unknown


def infer_is_local(model: str) -> bool:
    return bool(_LOCAL_MODEL_PATTERNS.search(model))


def parse_date_flexible(raw: str) -> datetime | None:
    """Try several common date formats."""
    formats = [
        "%Y-%m-%d",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%b %d, %Y",
        "%B %d, %Y",
    ]
    raw = raw.strip()
    for fmt in formats:
        try:
            return datetime.strptime(raw, fmt)
        except ValueError:
            continue
    return None


def validate_upload_size(data: bytes) -> None:
    if len(data) > _MAX_FILE_BYTES:
        raise ValueError(
            f"File too large: {len(data) / 1e6:.1f} MB. Max allowed: "
            f"{_MAX_FILE_BYTES / 1e6:.0f} MB."
        )


def safe_float(val: str, default: float = 0.0) -> float:
    try:
        return float(str(val).replace(",", "").replace("$", "").strip())
    except (ValueError, TypeError):
        return default


def safe_int(val: str, default: int = 0) -> int:
    try:
        return int(str(val).replace(",", "").strip())
    except (ValueError, TypeError):
        return default
