"""
Rule-based AI category classifier.

Maps (provider, workload_class, feature) → (AICategory, confidence).
Entries below REVIEW_THRESHOLD are written with tag_needs_review=True,
surfacing them in the Spend Intelligence review queue for human confirmation.

Rules are evaluated top-to-bottom; first match wins.
"""
from __future__ import annotations

import logging

from src.models import AICategory

log = logging.getLogger(__name__)

REVIEW_THRESHOLD = 0.70  # confidence < this → tag_needs_review = True

# Each rule: (provider_match, workload_match, category, confidence)
# None in provider_match or workload_match means "any".
_RULES: list[tuple[str | None, str | None, AICategory, float]] = [
    # Provider-certain: Cursor is exclusively a coding tool
    ("cursor",    None,       AICategory.code_gen,        1.00),
    # Workload-certain
    ("*",         "coding",   AICategory.code_gen,        0.85),
    ("*",         "rag",      AICategory.research,        0.80),
    # Reasoning models are mostly research/analysis — below threshold to catch edge cases
    ("*",         "reason",   AICategory.research,        0.70),
    # Extract is ambiguous: structured data work or document processing
    ("*",         "extract",  AICategory.research,        0.60),
    # Agents pattern is mixed: code gen automations or research pipelines
    ("*",         "agents",   AICategory.code_gen,        0.55),
    # Catch-all
    ("*",         "unknown",  AICategory.unknown,         0.25),
]


def get_rules() -> list[dict]:
    """Return the rule table as plain dicts for display in the UI."""
    return [
        {
            "provider":       prov if prov and prov != "*" else "any",
            "workload":       wc   if wc   and wc   != "*" else "any",
            "category":       cat.value,
            "confidence":     conf,
            "needs_review":   conf < REVIEW_THRESHOLD,
        }
        for prov, wc, cat, conf in _RULES
    ]


def classify(
    provider: str,
    workload_class: str,
    feature: str | None = None,  # reserved for future feature-hint rules
) -> tuple[AICategory, float]:
    """Return (category, confidence) for a single entry."""
    for prov_match, wc_match, category, confidence in _RULES:
        if prov_match is not None and prov_match != "*" and prov_match != provider:
            continue
        if wc_match is not None and wc_match != "*" and wc_match != workload_class:
            continue
        return category, confidence
    return AICategory.unknown, 0.25


def classify_pending(limit: int = 500) -> int:
    """
    Classify entries that haven't been through the classifier yet.

    Reads entries where tag_confidence IS NULL, runs classify() on each,
    writes the result back via tag_entry() (restricted write — only tagging
    columns are touched). Returns the number of entries processed.
    """
    from src.db import fetch_unclassified, tag_entry

    entries = fetch_unclassified(limit=limit)
    if not entries:
        return 0

    for row in entries:
        category, confidence = classify(
            provider=row.get("provider", ""),
            workload_class=row.get("workload_class", "unknown"),
            feature=row.get("feature"),
        )
        needs_review = confidence < REVIEW_THRESHOLD
        try:
            tag_entry(
                row["id"],
                ai_category=category,
                confidence=confidence,
                needs_review=needs_review,
            )
        except Exception:
            log.exception("Failed to tag entry %s", row["id"])

    log.info("Classifier: processed %d entries", len(entries))
    return len(entries)
