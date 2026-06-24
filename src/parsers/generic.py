"""
Generic / auto-detect CSV parser.

Sniffs the provider from headers or content, then delegates to the
appropriate specialized parser. Falls back to a best-effort generic parse.
"""
from __future__ import annotations

import csv
import io
import logging

from src.models import ParsedBill, Provider
from src.parsers.anthropic import parse_anthropic_csv
from src.parsers.base import validate_upload_size
from src.parsers.openai import parse_openai_csv

log = logging.getLogger(__name__)

_ANTHROPIC_SIGNALS = {"cache read tokens", "cache write tokens", "cache_creation_input_tokens",
                      "claude", "anthropic"}
_OPENAI_SIGNALS    = {"aggregation_timestamp", "snapshot_id", "context_tokens_input",
                      "credits", "openai", "gpt", "o1", "o3"}


def detect_provider(data: bytes) -> Provider:
    try:
        text = data.decode("utf-8-sig", errors="strict")
        reader = csv.reader(io.StringIO(text))
        header_row = next(reader, [])
        first_data  = next(reader, [])
    except Exception:
        return Provider.other

    combined = " ".join(header_row + first_data).lower()

    for sig in _ANTHROPIC_SIGNALS:
        if sig in combined:
            return Provider.anthropic
    for sig in _OPENAI_SIGNALS:
        if sig in combined:
            return Provider.openai

    return Provider.other


def parse_auto(data: bytes, filename: str = "upload.csv") -> ParsedBill:
    validate_upload_size(data)
    provider = detect_provider(data)
    log.info("Auto-detected provider: %s for file: %s", provider, filename)

    if provider == Provider.anthropic:
        return parse_anthropic_csv(data, filename)
    if provider == Provider.openai:
        return parse_openai_csv(data, filename)

    # Generic fallback — try Anthropic parser (most permissive), note it
    bill = parse_anthropic_csv(data, filename)
    bill.provider = Provider.other
    bill.parse_warnings.insert(
        0,
        "Provider not auto-detected. Parsed as generic CSV. "
        "Results may be incomplete — try selecting your provider manually."
    )
    return bill
