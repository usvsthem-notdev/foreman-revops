"""
Gemini billing CSV parser (Google Cloud Billing / BigQuery export format).

To export Gemini spend data:
  1. console.cloud.google.com → Billing → Billing export → BigQuery export
  2. Query usage for service "aiplatform.googleapis.com" or "generativelanguage.googleapis.com"
  3. Download a date-range CSV and import it here via the Bill Analyzer tab.

Column names in BigQuery exports vary by setup; the generic parser is often
a better first attempt.  This module provides the Gemini pricing table used
by cost estimation when a row has no cost value.
"""
from __future__ import annotations

from src.models import ParsedBill, Provider
from src.parsers.base import validate_upload_size

# Gemini pricing ($/M tokens) — June 2026
# Tuple: (input_price, cache_read_price, thinking_price, output_price)
# thinking_price = 0.0 for models that do not support thinking tokens.
_GEMINI_PRICES: dict[str, tuple[float, float, float, float]] = {
    "gemini-2.5-pro":   (1.25,   0.3125, 3.50,  10.00),
    "gemini-2.5-flash": (0.15,   0.0375, 3.50,   0.60),
    "gemini-1.5-pro":   (1.25,   0.3125, 0.0,    5.00),
    "gemini-1.5-flash": (0.075,  0.01875, 0.0,   0.30),
    "gemini-1.0-pro":   (0.50,   0.0,    0.0,    1.50),
}


def _estimate_gemini_cost(
    model: str,
    input_tok: int,
    cache_read: int,
    thinking_tok: int,
    output_tok: int,
) -> float:
    """
    Gemini token pricing (June 2026):
      - Regular input:       input_price     (standard rate)
      - Context cache read:  cache_read_price (~25% of input price — cheaper)
      - Thinking tokens:     thinking_price  (higher rate; reasoning output)
      - Regular output:      output_price
    Thinking tokens are priced separately from regular output on models that
    support them (Gemini 2.5 Pro/Flash).
    """
    model_lower = model.lower()
    for key, (in_price, cache_price, think_price, out_price) in _GEMINI_PRICES.items():
        if key in model_lower:
            return (
                input_tok    * in_price
                + cache_read   * cache_price
                + thinking_tok * think_price
                + output_tok   * out_price
            ) / 1_000_000
    # Unknown Gemini model — use Flash as conservative fallback
    return (input_tok * 0.15 + thinking_tok * 3.50 + output_tok * 0.60) / 1_000_000


def parse_gemini_csv(data: bytes, filename: str = "upload.csv") -> ParsedBill:
    """
    Parse a Google Cloud Billing BigQuery CSV export for Gemini usage.

    BigQuery export schemas vary significantly by project setup and date range.
    This stub returns a clear warning directing users to the generic parser or
    manual entry.  A full implementation requires knowing the exact column names
    from the user's BigQuery export.
    """
    validate_upload_size(data)
    return ParsedBill(
        source_file=filename,
        provider=Provider.gemini,
        entries=[],
        parse_warnings=[
            "Gemini BigQuery CSV parsing is not yet fully implemented. "
            "Try the generic parser tab, or add entries manually. "
            "Cost estimation uses _estimate_gemini_cost() when a row has no cost value."
        ],
    )
