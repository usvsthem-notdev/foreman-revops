"""Seed the local DB with a realistic 60-day spend simulation."""
import os, sys, random
from datetime import datetime, timedelta
from pathlib import Path

os.environ.setdefault("FOREMAN_DB_PATH", str(Path(__file__).parent / "data" / "foreman.db"))
sys.path.insert(0, str(Path(__file__).parent))

from src.db import clear_all_entries, init_db, insert_entries_bulk, upsert_budget
from src.models import (
    Budget, BudgetPeriod, EntrySource, Provider,
    SpendEntry, WorkloadClass,
)


random.seed(42)
init_db()
clear_all_entries()

TODAY = datetime.utcnow()

# ── Workload templates (class, provider, model, is_local, cost_per_1k_in, cost_per_1k_out)
TEMPLATES = [
    # High-volume extraction — should be absorbed locally
    (WorkloadClass.extract, Provider.anthropic, "claude-haiku-4-5",        False, 0.00025, 0.00125),
    (WorkloadClass.extract, Provider.anthropic, "claude-haiku-4-5",        True,  0.0,     0.0),     # local
    # RAG — mix of local embedding + haiku
    (WorkloadClass.rag,     Provider.anthropic, "claude-haiku-4-5",        False, 0.00025, 0.00125),
    (WorkloadClass.rag,     Provider.anthropic, "bge-large-local",         True,  0.0,     0.0),     # local
    (WorkloadClass.rag,     Provider.openai,    "text-embedding-3-small",  False, 0.0001,  0.0),
    # Reasoning — expensive frontier
    (WorkloadClass.reason,  Provider.anthropic, "claude-opus-4",           False, 0.015,   0.075),
    (WorkloadClass.reason,  Provider.anthropic, "claude-sonnet-4-5",       False, 0.003,   0.015),
    (WorkloadClass.reason,  Provider.openai,    "o1-mini",                 False, 0.003,   0.012),
    (WorkloadClass.reason,  Provider.gemini,    "gemini-2.5-pro",          False, 0.00125, 0.01),
    # Agents — long horizon, many tokens
    (WorkloadClass.agents,  Provider.anthropic, "claude-sonnet-4-5",       False, 0.003,   0.015),
    (WorkloadClass.agents,  Provider.openai,    "gpt-4o",                  False, 0.0025,  0.01),
    (WorkloadClass.agents,  Provider.anthropic, "qwen3-32b-local",         True,  0.0,     0.0),     # local
    # Coding — mix including Cursor IDE
    (WorkloadClass.coding,  Provider.anthropic, "claude-sonnet-4-5",       False, 0.003,   0.015),
    (WorkloadClass.coding,  Provider.openai,    "gpt-4o",                  False, 0.0025,  0.01),
    (WorkloadClass.coding,  Provider.cursor,    "claude-3.5-sonnet",       False, 0.003,   0.015),
    (WorkloadClass.coding,  Provider.cursor,    "cursor-small",            False, 0.0002,  0.0008),
    (WorkloadClass.coding,  Provider.gemini,    "gemini-2.5-flash",        False, 0.00015, 0.0006),
    (WorkloadClass.coding,  Provider.anthropic, "deepseek-coder-local",    True,  0.0,     0.0),     # local
]

TEAMS    = ["eng", "eng", "eng", "product", "data", "infra"]
FEATURES = ["chat", "search", "summarize", "review", "classify", "embed", "plan", "generate"]

# Volume ramps up over 60 days (simulating growth)
entries = []
for day_offset in range(60):
    ts_base = TODAY - timedelta(days=59 - day_offset)
    growth  = 1.0 + (day_offset / 60) * 1.8          # 1x → 2.8x over 60 days
    n_calls = int(random.gauss(120 * growth, 20))

    for _ in range(n_calls):
        tmpl = random.choice(TEMPLATES)
        cls, prov, model, is_local, in_rate, out_rate = tmpl

        ts = ts_base + timedelta(
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
        )

        # Token volumes by class
        if cls == WorkloadClass.extract:
            in_tok  = random.randint(500,  4_000)
            out_tok = random.randint(100,    500)
            r_tok   = 0
        elif cls == WorkloadClass.rag:
            in_tok  = random.randint(2_000, 12_000)
            out_tok = random.randint(200,   1_500)
            r_tok   = 0
        elif cls == WorkloadClass.reason:
            in_tok  = random.randint(5_000, 30_000)
            out_tok = random.randint(500,   3_000)
            r_tok   = random.randint(2_000, 15_000) if "o1" in model or "opus" in model else 0
        elif cls == WorkloadClass.agents:
            in_tok  = random.randint(10_000, 80_000)
            out_tok = random.randint(1_000,  8_000)
            r_tok   = random.randint(5_000,  40_000) if "sonnet" in model else 0
        else:  # coding
            in_tok  = random.randint(3_000, 20_000)
            out_tok = random.randint(500,   4_000)
            r_tok   = 0

        if is_local:
            # Notional GPU compute cost (~$0.0002/1K tokens — L40S amortised)
            cost = (in_tok + out_tok + r_tok) * 0.0000002
        else:
            cost = ((in_tok + r_tok) * in_rate + out_tok * out_rate) / 1_000

        entries.append(SpendEntry(
            timestamp=ts,
            provider=prov,
            model=model,
            workload_class=cls,
            input_tokens=in_tok,
            output_tokens=out_tok,
            reasoning_tokens=r_tok,
            cost_usd=round(cost, 8),
            is_local=is_local,
            team=random.choice(TEAMS),
            feature=random.choice(FEATURES),
            source=EntrySource.manual,
        ))

inserted = insert_entries_bulk(entries)
print(f"Inserted {inserted:,} entries")

# Budgets
for b in [
    Budget(name="Monthly total",     amount_usd=4_000, period=BudgetPeriod.monthly, alert_threshold=0.80),
    Budget(name="Frontier only",     amount_usd=3_500, period=BudgetPeriod.monthly, alert_threshold=0.85),
    Budget(name="Daily cap",         amount_usd=200,   period=BudgetPeriod.daily,   alert_threshold=0.90),
    Budget(name="Eng team monthly",  amount_usd=2_500, period=BudgetPeriod.monthly, team="eng",     alert_threshold=0.80),
    Budget(name="Product weekly",    amount_usd=500,   period=BudgetPeriod.weekly,  team="product", alert_threshold=0.75),
]:
    upsert_budget(b)
print("Budgets set")

total = sum(e.cost_usd for e in entries)
frontier = sum(e.cost_usd for e in entries if not e.is_local)
local_pct = (1 - frontier/total) * 100 if total else 0
print(f"Total spend: ${total:,.2f}  |  Frontier: ${frontier:,.2f}  |  Local absorbed: {local_pct:.1f}%")
