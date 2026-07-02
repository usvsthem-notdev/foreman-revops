"""
Auto-seed demo data on first run.

Called from app.py at startup. Seeds 60 days of realistic multi-provider
spend so the app shows something meaningful without real API keys or
billing exports. Runs only when the DB is empty — user data is never
overwritten.
"""
from __future__ import annotations

import random
from datetime import datetime, timedelta

from src.analytics.classifier import classify
from src.db import fetch_entries, insert_entries_bulk, upsert_budget
from src.models import (
    Budget,
    BudgetPeriod,
    EntrySource,
    Provider,
    SpendEntry,
    WorkloadClass,
)

_SEED = 42

_TEMPLATES = [
    # (workload_class, provider, model, is_local, in_rate_per_1k, out_rate_per_1k)
    # Rates mirror the July 2026 tables in src/parsers + src/analytics/pricing.
    (WorkloadClass.extract, Provider.anthropic, "claude-haiku-4-5",        False, 0.001,   0.005),
    (WorkloadClass.extract, Provider.anthropic, "bge-large-local",         True,  0.0,     0.0),
    (WorkloadClass.rag,     Provider.anthropic, "claude-haiku-4-5",        False, 0.001,   0.005),
    (WorkloadClass.rag,     Provider.anthropic, "bge-large-local",         True,  0.0,     0.0),
    (WorkloadClass.rag,     Provider.openai,    "text-embedding-3-small",  False, 0.0001,  0.0),
    (WorkloadClass.reason,  Provider.anthropic, "claude-opus-4-6",         False, 0.005,   0.025),
    (WorkloadClass.reason,  Provider.anthropic, "claude-sonnet-4-6",       False, 0.003,   0.015),
    (WorkloadClass.reason,  Provider.openai,    "gpt-5.4",                 False, 0.0025,  0.015),
    (WorkloadClass.reason,  Provider.gemini,    "gemini-3-pro",            False, 0.002,   0.012),
    (WorkloadClass.agents,  Provider.anthropic, "claude-sonnet-4-6",       False, 0.003,   0.015),
    (WorkloadClass.agents,  Provider.openai,    "gpt-5.4",                 False, 0.0025,  0.015),
    (WorkloadClass.agents,  Provider.anthropic, "qwen3-32b-local",         True,  0.0,     0.0),
    (WorkloadClass.coding,  Provider.anthropic, "claude-sonnet-4-6",       False, 0.003,   0.015),
    (WorkloadClass.coding,  Provider.openai,    "gpt-5.4-mini",            False, 0.00075, 0.0045),
    (WorkloadClass.coding,  Provider.cursor,    "claude-sonnet-4-6",       False, 0.003,   0.015),
    (WorkloadClass.coding,  Provider.cursor,    "cursor-small",            False, 0.0002,  0.0008),
    (WorkloadClass.coding,  Provider.gemini,    "gemini-3-flash",          False, 0.0005,  0.003),
    (WorkloadClass.coding,  Provider.anthropic, "deepseek-coder-local",    True,  0.0,     0.0),
]

_TEAMS    = ["eng", "eng", "eng", "product", "data", "infra"]
_FEATURES = ["chat", "search", "summarize", "review", "classify", "embed", "plan", "generate"]


def seed_if_empty() -> bool:
    """
    Seed the DB with 60 days of demo data if it contains no entries.
    Returns True if seeding ran, False if data already existed.
    """
    if fetch_entries(limit=1):
        return False

    rng = random.Random(_SEED)
    today = datetime.utcnow()
    entries: list[SpendEntry] = []

    for day_offset in range(60):
        ts_base = today - timedelta(days=59 - day_offset)
        growth  = 1.0 + (day_offset / 60) * 1.8
        n_calls = int(rng.gauss(120 * growth, 20))

        for _ in range(n_calls):
            cls, prov, model, is_local, in_rate, out_rate = rng.choice(_TEMPLATES)
            ts = ts_base + timedelta(
                hours=rng.randint(0, 23),
                minutes=rng.randint(0, 59),
            )

            cache_read_tok = 0
            cache_creation_tok = 0

            if cls == WorkloadClass.extract:
                in_tok  = rng.randint(500,    4_000)
                out_tok = rng.randint(100,      500)
                r_tok   = 0
            elif cls == WorkloadClass.rag:
                in_tok  = rng.randint(2_000, 12_000)
                out_tok = rng.randint(200,   1_500)
                r_tok   = 0
                # Repeated retrieval context (same doc chunks reused across
                # queries) makes RAG a realistic prompt-caching candidate.
                if "claude" in model and not is_local:
                    cache_read_tok = rng.randint(int(in_tok * 0.3), int(in_tok * 0.7))
                    cache_creation_tok = rng.randint(0, int(in_tok * 0.05))
            elif cls == WorkloadClass.reason:
                in_tok  = rng.randint(5_000, 30_000)
                out_tok = rng.randint(500,   3_000)
                r_tok   = rng.randint(2_000, 15_000) if "gpt-5" in model or "opus" in model else 0
            elif cls == WorkloadClass.agents:
                in_tok  = rng.randint(10_000, 80_000)
                out_tok = rng.randint(1_000,  8_000)
                r_tok   = rng.randint(5_000,  40_000) if "sonnet" in model else 0
                # Agent loops re-send a large, mostly-static system prompt +
                # tool definitions on every step — a strong caching fit.
                if "claude" in model and not is_local:
                    cache_read_tok = rng.randint(int(in_tok * 0.4), int(in_tok * 0.85))
                    cache_creation_tok = rng.randint(0, int(in_tok * 0.05))
            else:
                in_tok  = rng.randint(3_000, 20_000)
                out_tok = rng.randint(500,   4_000)
                r_tok   = 0

            cost = (
                (in_tok + out_tok + r_tok) * 0.0000002  # ~L40S amortised GPU rate
                if is_local
                else ((in_tok + r_tok) * in_rate + out_tok * out_rate) / 1_000
            )

            ai_cat, confidence = classify(prov.value, cls.value)
            entries.append(SpendEntry(
                timestamp=ts,
                provider=prov,
                model=model,
                workload_class=cls,
                input_tokens=in_tok,
                output_tokens=out_tok,
                reasoning_tokens=r_tok,
                cache_read_tokens=cache_read_tok,
                cache_creation_tokens=cache_creation_tok,
                cost_usd=round(cost, 8),
                is_local=is_local,
                team=rng.choice(_TEAMS),
                feature=rng.choice(_FEATURES),
                source=EntrySource.manual,
                ai_category=ai_cat,
                tag_confidence=confidence,
                tag_needs_review=confidence < 0.70,
            ))

    insert_entries_bulk(entries)

    for b in [
        Budget(name="Monthly total",    amount_usd=4_000, period=BudgetPeriod.monthly, alert_threshold=0.80),
        Budget(name="Frontier only",    amount_usd=3_500, period=BudgetPeriod.monthly, alert_threshold=0.85),
        Budget(name="Daily cap",        amount_usd=200,   period=BudgetPeriod.daily,   alert_threshold=0.90),
        Budget(name="Eng team monthly", amount_usd=2_500, period=BudgetPeriod.monthly, team="eng",     alert_threshold=0.80),
        Budget(name="Product weekly",   amount_usd=500,   period=BudgetPeriod.weekly,  team="product", alert_threshold=0.75),
    ]:
        upsert_budget(b)

    return True
