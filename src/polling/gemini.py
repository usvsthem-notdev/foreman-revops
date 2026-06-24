"""
Google Gemini provider stub.

Google does not expose a usage or billing REST API for AI Studio API keys
(AIzaSy...).  The Generative Language API returns token counts inline with
each inference response but provides no historical usage endpoint.

To get Gemini cost data:
  1. Export billing to BigQuery (console.cloud.google.com → Billing → Export)
     then download a date-range CSV and import it via the Bill Analyzer tab.
  2. Use the Cloud Monitoring API with a GCP service account (complex OAuth
     flow, request counts only — no token-level data).

The key is stored here so Gemini can be recognized as a provider for manual
entries and future integrations.
"""
from __future__ import annotations

from datetime import date

from src.models import SpendEntry

_NO_USAGE_API_MSG = (
    "Gemini polling is not available: Google does not expose a usage or billing "
    "REST API for AI Studio API keys. To import Gemini spend data, export your "
    "Google Cloud billing to BigQuery, download a CSV for the date range, and "
    "import it via the Bill Analyzer tab."
)


def poll(
    api_key: str,  # noqa: ARG001
    since: date | None = None,  # noqa: ARG001
    until: date | None = None,  # noqa: ARG001
) -> tuple[list[SpendEntry], list[str]]:
    """Always returns an empty result with a clear explanation."""
    return [], [_NO_USAGE_API_MSG]
