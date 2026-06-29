"""
LLM-assisted classifier for low-confidence spend entries.

Complements the rule-based classifier: rules handle obvious cases for free
and deterministically; this module sends only the needs_review tail to a
Claude or GPT model for context-aware reclassification.

The model sees the full entry context — model name, feature, notes, team —
which the rule-based classifier ignores.  That's where it earns its keep.
"""
from __future__ import annotations

import json
import logging
import re
import time

import httpx

from src.analytics.classifier import REVIEW_THRESHOLD
from src.models import AICategory

log = logging.getLogger(__name__)

# Models chosen for speed and cost — classification is a simple structured task.
_ANTHROPIC_MODEL = "claude-haiku-4-5-20251001"
_OPENAI_MODEL    = "gpt-4o-mini"

_ANTHROPIC_URL   = "https://api.anthropic.com/v1/messages"
_OPENAI_URL      = "https://api.openai.com/v1/chat/completions"

_VALID_CATEGORIES = {c.value for c in AICategory}

_PROMPT = """\
You are classifying AI API spend entries for a finance ops tracker.
Assign the entry to exactly one category based on what the tool was used for.
Use the model name, feature, and notes as your primary signal.

Categories (reply with the exact key):
  code_gen        — coding, debugging, code review, software development
  research        — analysis, summarisation, RAG, retrieval, reasoning, data extraction
  document_office — document drafting, editing, email, translation, office tasks
  unknown         — not enough context to determine

Entry:
  provider:       {provider}
  model:          {model}
  workload_class: {workload_class}
  feature:        {feature}
  notes:          {notes}
  team:           {team}

Respond with JSON only, no markdown fences:
{{"category": "<key>", "confidence": <0.0-1.0>, "reasoning": "<one sentence>"}}"""


# ---------------------------------------------------------------------------
# Single-entry classification
# ---------------------------------------------------------------------------

def classify_entry_with_llm(
    entry: dict,
    api_key: str,
    llm_provider: str = "anthropic",
) -> tuple[AICategory, float, str]:
    """
    Return (category, confidence, reasoning) for a single entry.

    Raises ValueError if the API call fails or the response is unparseable.
    The api_key is never logged.
    """
    from src.polling.base import mask_key, safe_post

    prompt = _PROMPT.format(
        provider=entry.get("provider", ""),
        model=entry.get("model", ""),
        workload_class=entry.get("workload_class", "unknown"),
        feature=entry.get("feature") or "not specified",
        notes=entry.get("notes") or "none",
        team=entry.get("team") or "not specified",
    )

    if llm_provider == "anthropic":
        resp = safe_post(
            _ANTHROPIC_URL,
            headers={
                "x-api-key":          api_key,
                "anthropic-version":  "2023-06-01",
                "content-type":       "application/json",
            },
            json={
                "model":      _ANTHROPIC_MODEL,
                "max_tokens": 256,
                "messages":   [{"role": "user", "content": prompt}],
            },
        )
        if not resp.is_success:
            raise ValueError(f"Anthropic API error {resp.status_code}: {resp.text[:200]}")
        raw_text = resp.json()["content"][0]["text"]

    elif llm_provider == "openai":
        resp = safe_post(
            _OPENAI_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "content-type":  "application/json",
            },
            json={
                "model":           _OPENAI_MODEL,
                "max_tokens":      256,
                "messages":        [{"role": "user", "content": prompt}],
                "response_format": {"type": "json_object"},
            },
        )
        if not resp.is_success:
            raise ValueError(f"OpenAI API error {resp.status_code}: {resp.text[:200]}")
        raw_text = resp.json()["choices"][0]["message"]["content"]

    else:
        raise ValueError(f"Unsupported LLM provider: {llm_provider!r}")

    log.debug("LLM classify (key=%s): %s", mask_key(api_key), raw_text[:120])
    return _parse_response(raw_text)


def _parse_response(text: str) -> tuple[AICategory, float, str]:
    """Extract (category, confidence, reasoning) from the model's JSON reply."""
    # Strip markdown fences if the model ignored the instruction
    text = re.sub(r"```[a-z]*\n?", "", text).strip()

    # Find the first {...} block
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:200]!r}")

    try:
        data = json.loads(match.group())
    except json.JSONDecodeError as exc:
        raise ValueError(f"Could not parse JSON: {exc}  raw={text[:200]!r}") from exc

    category_str = str(data.get("category", "unknown")).strip().lower()
    if category_str not in _VALID_CATEGORIES:
        log.warning("LLM returned unknown category %r — falling back to unknown", category_str)
        category_str = "unknown"

    confidence = float(data.get("confidence", 0.5))
    confidence = max(0.0, min(0.95, confidence))   # cap — no LLM is 100% certain

    reasoning = str(data.get("reasoning", "")).strip()[:512]

    return AICategory(category_str), confidence, reasoning


# ---------------------------------------------------------------------------
# Batch: reclassify all needs_review entries
# ---------------------------------------------------------------------------

def classify_needs_review(
    llm_provider: str = "anthropic",
    limit: int = 200,
) -> tuple[int, list[str]]:
    """
    Re-classify entries flagged as needs_review using an LLM.

    Returns (n_classified, error_messages).
    Automatically clears needs_review when confidence >= REVIEW_THRESHOLD.
    """
    from src.db import fetch_pending_review, tag_entry
    from src.polling.key_store import get_key

    api_key = get_key(llm_provider)
    if not api_key:
        return 0, [f"No {llm_provider} API key configured — add one in the Live API tab."]

    entries = fetch_pending_review(limit=limit)
    if not entries:
        return 0, []

    classified = 0
    errors: list[str] = []

    for entry in entries:
        try:
            category, confidence, reasoning = classify_entry_with_llm(
                entry, api_key, llm_provider=llm_provider
            )
            needs_review = confidence < REVIEW_THRESHOLD
            tag_entry(
                entry["id"],
                ai_category=category,
                confidence=confidence,
                needs_review=needs_review,
            )
            classified += 1
            log.info(
                "LLM reclassified entry %s → %s (conf=%.2f, review=%s) — %s",
                entry["id"][:8], category.value, confidence, needs_review, reasoning,
            )
        except httpx.TimeoutException:
            errors.append(f"Timeout on entry {entry['id'][:8]} — skipped.")
        except Exception as exc:
            errors.append(f"Entry {entry['id'][:8]}: {exc}")

        # Polite pause — avoid hammering rate limits
        time.sleep(0.25)

    log.info("LLM classifier: %d classified, %d errors", classified, len(errors))
    return classified, errors
