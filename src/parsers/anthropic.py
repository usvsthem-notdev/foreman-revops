"""
Anthropic billing CSV parser.

Expected columns (Console → Usage export):
  Date, Organization, Project, Model, Input tokens, Output tokens, Cache read tokens,
  Cache write tokens, Cost (USD)

The parser is tolerant: it tries several column name variants and emits
a warning for each row it cannot map rather than failing the whole upload.
"""
from __future__ import annotations

import csv
import io
import logging
from datetime import datetime

from src.models import EntrySource, ParsedBill, Provider, SpendEntry
from src.parsers.base import (
    infer_is_local,
    infer_workload_class,
    parse_date_flexible,
    safe_float,
    safe_int,
    validate_upload_size,
)

log = logging.getLogger(__name__)

# Column name aliases (lowercase) → canonical key
_COL_MAP = {
    "date":                   "date",
    "timestamp":              "date",
    "time":                   "date",
    "model":                  "model",
    "model name":             "model",
    "input tokens":           "input_tokens",
    "input_tokens":           "input_tokens",
    "prompt tokens":          "input_tokens",
    "output tokens":          "output_tokens",
    "output_tokens":          "output_tokens",
    "completion tokens":      "output_tokens",
    "cache read tokens":      "cache_read",
    "cache_read_input_tokens":"cache_read",
    "cache write tokens":     "cache_write",
    "cache_creation_input_tokens": "cache_write",
    "cost (usd)":             "cost_usd",
    "cost":                   "cost_usd",
    "amount":                 "cost_usd",
    "total cost":             "cost_usd",
    "project":                "project",
    "workspace":              "project",
    "organization":           "org",
}


def _map_headers(headers: list[str]) -> dict[str, int]:
    """Return canonical_key → column_index."""
    mapping: dict[str, int] = {}
    for i, h in enumerate(headers):
        canonical = _COL_MAP.get(h.lower().strip())
        if canonical and canonical not in mapping:
            mapping[canonical] = i
    return mapping


def parse_anthropic_csv(data: bytes, filename: str = "upload.csv") -> ParsedBill:
    validate_upload_size(data)

    warnings: list[str] = []
    entries: list[SpendEntry] = []

    try:
        text = data.decode("utf-8-sig", errors="strict")
    except UnicodeDecodeError as exc:
        raise ValueError(
            f"File is not valid UTF-8 (byte offset {exc.start}). "
            "Re-export as UTF-8 CSV and try again."
        ) from exc
    reader = csv.reader(io.StringIO(text))

    try:
        raw_headers = next(reader)
    except StopIteration:
        return ParsedBill(
            source_file=filename,
            provider=Provider.anthropic,
            entries=[],
            parse_warnings=["File appears to be empty."],
        )

    col = _map_headers(raw_headers)

    if "model" not in col:
        warnings.append("Could not find a 'model' column — rows may not parse correctly.")
    if "cost_usd" not in col and "input_tokens" not in col:
        warnings.append(
            "No cost or token columns detected."
            " Check that this is an Anthropic billing export."
        )

    for line_num, row in enumerate(reader, start=2):
        if not any(row):
            continue
        try:
            entry = _parse_row(row, col, filename, line_num, warnings)
            if entry:
                entries.append(entry)
        except Exception as exc:
            warnings.append(f"Row {line_num}: skipped ({exc})")

    bill = ParsedBill(
        source_file=filename,
        provider=Provider.anthropic,
        entries=entries,
        parse_warnings=warnings,
    )
    bill.total_cost_usd = sum(e.cost_usd for e in entries)
    bill.total_input_tokens = sum(e.input_tokens for e in entries)
    bill.total_output_tokens = sum(e.output_tokens for e in entries)
    bill.total_reasoning_tokens = sum(e.reasoning_tokens for e in entries)
    return bill


def _parse_row(
    row: list[str],
    col: dict[str, int],
    filename: str,
    line_num: int,
    warnings: list[str],
) -> SpendEntry | None:
    def get(key: str, default: str = "") -> str:
        idx = col.get(key)
        if idx is None or idx >= len(row):
            return default
        return row[idx].strip()

    date_raw = get("date")
    ts = parse_date_flexible(date_raw) if date_raw else None
    if ts is None:
        ts = datetime.utcnow()
        if date_raw:
            warnings.append(f"Row {line_num}: could not parse date '{date_raw}', using now.")

    model = get("model") or "unknown"
    input_tok = safe_int(get("input_tokens"))
    output_tok = safe_int(get("output_tokens"))
    cache_read = safe_int(get("cache_read"))
    cache_write = safe_int(get("cache_write"))
    cost = safe_float(get("cost_usd"))

    # Anthropic cache tokens are a form of "input" — add to input count
    effective_input = input_tok + cache_read + cache_write

    # Rough cost estimate if missing (current Anthropic list prices, June 2026)
    if cost == 0.0 and (effective_input + output_tok) > 0:
        cost = _estimate_anthropic_cost(model, effective_input, output_tok)

    feature = get("project") or None

    return SpendEntry(
        timestamp=ts,
        provider=Provider.anthropic,
        model=model,
        workload_class=infer_workload_class(model),
        input_tokens=effective_input,
        output_tokens=output_tok,
        reasoning_tokens=0,
        cache_read_tokens=cache_read,
        cache_creation_tokens=cache_write,
        cost_usd=cost,
        is_local=infer_is_local(model),
        feature=feature,
        source=EntrySource.anthropic_csv,
    )


# Approximate Anthropic pricing ($/M tokens) — July 2026
# Public: also consumed by src.analytics.pricing for burn-map cost attribution.
# Retired/renamed models stay in the table so historical bill exports still
# price at the rate that applied when the spend happened.
ANTHROPIC_PRICES: dict[str, tuple[float, float]] = {
    # Current generation
    "claude-fable-5":    (10.0, 50.0),
    "claude-opus-4-8":   (5.0,  25.0),
    "claude-opus-4-7":   (5.0,  25.0),
    "claude-opus-4-6":   (5.0,  25.0),
    "claude-opus-4-5":   (5.0,  25.0),   # Opus price cut landed with 4.5
    "claude-sonnet-4-6": (3.0,  15.0),
    "claude-sonnet-4-5": (3.0,  15.0),
    "claude-haiku-4-5":  (1.0,  5.0),
    # Prior generation — historical bills
    "claude-opus-4-1":   (15.0, 75.0),
    "claude-opus-4":     (15.0, 75.0),
    "claude-opus":       (15.0, 75.0),
    "claude-sonnet-4":   (3.0,  15.0),
    "claude-sonnet":     (3.0,  15.0),
    "claude-haiku":      (1.0,  5.0),
    "claude-3-5-sonnet": (3.0,  15.0),
    "claude-3-5-haiku":  (0.8,  4.0),
    "claude-3-opus":     (15.0, 75.0),
    "claude-3-haiku":    (0.25, 1.25),
}

# Longest key first so "claude-opus-4-8" can't be shadowed by "claude-opus-4".
_PRICE_LOOKUP_ORDER = sorted(ANTHROPIC_PRICES, key=len, reverse=True)


def _estimate_anthropic_cost(model: str, input_tok: int, output_tok: int) -> float:
    model_lower = model.lower()
    for key in _PRICE_LOOKUP_ORDER:
        if key in model_lower:
            in_price, out_price = ANTHROPIC_PRICES[key]
            return (input_tok * in_price + output_tok * out_price) / 1_000_000
    # Default fallback — mid-tier price
    return (input_tok * 3.0 + output_tok * 15.0) / 1_000_000
