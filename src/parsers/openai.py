"""
OpenAI billing CSV parser.

OpenAI exports vary by account type and date range. We handle:
  - Activity / usage exports (model, tokens, cost)
  - Invoice line-item exports
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

_COL_MAP = {
    "date":                   "date",
    "timestamp":              "date",
    "aggregation_timestamp":  "date",
    "model":                  "model",
    "model_id":               "model",
    "snapshot_id":            "model",
    "context_tokens_input":   "input_tokens",
    "context_tokens_prompt":  "input_tokens",
    "input_tokens":           "input_tokens",
    "prompt_tokens":          "input_tokens",
    "generated_tokens":       "output_tokens",
    "context_tokens_output":  "output_tokens",
    "output_tokens":          "output_tokens",
    "completion_tokens":      "output_tokens",
    "cached_context_tokens_input": "cached_tokens",
    "reasoning_tokens":       "reasoning_tokens",
    "amount":                 "cost_usd",
    "cost":                   "cost_usd",
    "credits":                "cost_usd",
    "cost_in_major_units":    "cost_usd",
    "total_cost":             "cost_usd",
    "usage_type":             "usage_type",
    "product":                "usage_type",
    "project_name":           "project",
    "project":                "project",
    "organization_name":      "org",
}


def _map_headers(headers: list[str]) -> dict[str, int]:
    mapping: dict[str, int] = {}
    for i, h in enumerate(headers):
        canonical = _COL_MAP.get(h.lower().strip())
        if canonical and canonical not in mapping:
            mapping[canonical] = i
    return mapping


def parse_openai_csv(data: bytes, filename: str = "upload.csv") -> ParsedBill:
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
            provider=Provider.openai,
            entries=[],
            parse_warnings=["File appears to be empty."],
        )

    col = _map_headers(raw_headers)

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
        provider=Provider.openai,
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
            warnings.append(f"Row {line_num}: could not parse date '{date_raw}'.")

    model = get("model") or get("usage_type") or "openai-unknown"
    input_tok = safe_int(get("input_tokens"))
    output_tok = safe_int(get("output_tokens"))
    reasoning_tok = safe_int(get("reasoning_tokens"))
    cost_raw = get("cost_usd")

    # OpenAI exports cost as negative credits in some formats
    cost = abs(safe_float(cost_raw))
    if cost == 0.0 and (input_tok + output_tok) > 0:
        # Use input_tok directly; cached_tok is already included in that count.
        cost = _estimate_openai_cost(model, input_tok, output_tok, reasoning_tok)

    feature = get("project") or None

    return SpendEntry(
        timestamp=ts,
        provider=Provider.openai,
        model=model,
        workload_class=infer_workload_class(model),
        input_tokens=input_tok,   # cached tokens are a subset, not additive
        output_tokens=output_tok,
        reasoning_tokens=reasoning_tok,
        cost_usd=cost,
        is_local=infer_is_local(model),
        feature=feature,
        source=EntrySource.openai_csv,
    )


# Approximate OpenAI pricing ($/M tokens) — June 2026
_OPENAI_PRICES: dict[str, tuple[float, float]] = {
    "o3":           (10.0, 40.0),
    "o1":           (15.0, 60.0),
    "o1-mini":      (3.0,  12.0),
    "gpt-4o":       (2.5,  10.0),
    "gpt-4-turbo":  (10.0, 30.0),
    "gpt-4":        (30.0, 60.0),
    "gpt-3.5":      (0.5,  1.5),
    "text-embedding": (0.1, 0.0),
}


def _estimate_openai_cost(model: str, input_tok: int, output_tok: int, reasoning_tok: int) -> float:
    model_lower = model.lower()
    for key, (in_price, out_price) in _OPENAI_PRICES.items():
        if key in model_lower:
            return (input_tok * in_price + (output_tok + reasoning_tok) * out_price) / 1_000_000
    return (input_tok * 2.5 + output_tok * 10.0) / 1_000_000
