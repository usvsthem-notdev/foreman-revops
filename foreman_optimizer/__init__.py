# Copyright 2026 Foreman contributors
# SPDX-License-Identifier: Apache-2.0
"""
foreman_optimizer — a token-cost optimization layer that runs *before* prompt
deconstruction in Foreman.

Quick start:

    from foreman_optimizer import ForemanOptimizer

    opt = ForemanOptimizer()
    result = opt.optimize(raw_prompt)

    send_to_model(result.optimized_prompt)      # feed Foreman deconstruction
    foreman.ingest(result.record.to_dict())     # spend + savings, categorized
"""

from .ir import PromptIR, Clause, CostDriver, ClauseKind, Tag, parse, set_token_estimator
from .rules import Tier0Config, Tier0Result, run_tier0
from .fingerprint import (
    FrequencyPromoter, InMemoryStore, SQLiteStore,
    fingerprint, skeleton, cache_prefix,
    TIER_0, TIER_1_ELIGIBLE, TIER_1_CACHED,
)
from .categories import (
    SpendCategory, ModelTier, PRICING, cost_usd,
    SavingsReport, NodeReport, ForemanRecord, savings_from_rules,
)
from .pipeline import ForemanOptimizer, OptimizerConfig, OptimizationResult, Tier1Hook

__version__ = "0.1.0"

__all__ = [
    "ForemanOptimizer", "OptimizerConfig", "OptimizationResult", "Tier1Hook",
    "PromptIR", "Clause", "CostDriver", "ClauseKind", "Tag", "parse", "set_token_estimator",
    "Tier0Config", "Tier0Result", "run_tier0",
    "FrequencyPromoter", "InMemoryStore", "SQLiteStore",
    "fingerprint", "skeleton", "cache_prefix",
    "TIER_0", "TIER_1_ELIGIBLE", "TIER_1_CACHED",
    "SpendCategory", "ModelTier", "PRICING", "cost_usd",
    "SavingsReport", "NodeReport", "ForemanRecord", "savings_from_rules",
]
