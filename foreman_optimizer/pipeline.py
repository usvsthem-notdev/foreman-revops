# Copyright 2026 Foreman contributors
# SPDX-License-Identifier: Apache-2.0
"""
Pipeline orchestrator.

The full pre-deconstruction pass Foreman runs on every prompt:

    parse (free)  ->  Tier-0 rules (free)  ->  fingerprint + observe
                                                     |
                                          hot? -> mark Tier-1 eligible
                                                     |
                                          build ForemanRecord (savings + hook)

The optimized prompt this returns is what flows into Foreman's existing prompt
*deconstruction* step. Nothing here calls an LLM; the Tier-1 LLM rewrite is a
hook you wire to your own optimizer (DSPy/GEPA, LLMLingua, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from . import categories as cat
from .fingerprint import FrequencyPromoter, Observation, TIER_1_ELIGIBLE, cache_prefix
from .ir import PromptIR, estimate_tokens, parse
from .rules import Tier0Config, Tier0Result, run_tier0


# A Tier-1 rewrite hook: (optimized_tier0_prompt, ir) -> better_prompt.
# Wire this to your LLM optimizer. Returning None means "no rewrite produced".
Tier1Hook = Callable[[str, PromptIR], Optional[str]]


@dataclass
class OptimizationResult:
    optimized_prompt: str        # feeds Foreman's deconstruction step
    ir: PromptIR
    tier0: Tier0Result
    observation: Observation
    record: cat.ForemanRecord
    cacheable_prefix: str

    @property
    def is_hot(self) -> bool:
        return self.observation.tier in (TIER_1_ELIGIBLE, "tier-1-cached")


@dataclass
class OptimizerConfig:
    tier0: Tier0Config = None
    promote_threshold: int = 5
    reference_tier: str = cat.ModelTier.MID.value

    def __post_init__(self):
        if self.tier0 is None:
            self.tier0 = Tier0Config()


class ForemanOptimizer:
    """First-layer optimizer. One instance per Foreman process."""

    def __init__(self, promoter: Optional[FrequencyPromoter] = None,
                 config: Optional[OptimizerConfig] = None,
                 tier1_hook: Optional[Tier1Hook] = None):
        self.config = config or OptimizerConfig()
        self.promoter = promoter or FrequencyPromoter(threshold=self.config.promote_threshold)
        self.tier1_hook = tier1_hook

    def optimize(self, prompt: str) -> OptimizationResult:
        # 1. Parse into cost-annotated IR (free).
        ir = parse(prompt)

        # 2. Tier-0 deterministic rules (free, every prompt).
        tier0 = run_tier0(ir, self.config.tier0)
        optimized = tier0.optimized

        # 3. Fingerprint + observe frequency.
        obs = self.promoter.observe(
            prompt=prompt,
            tokens=ir.total_tokens,
            saved=tier0.tokens_saved_input,
            ir=ir,
        )

        # 4. If hot AND we have an LLM rewrite hook AND none cached yet, run it.
        if obs.tier == TIER_1_ELIGIBLE and self.tier1_hook and obs.record.optimized is None:
            better = self.tier1_hook(optimized, ir)
            if better:
                optimized = better
                self.promoter.store_optimized(obs.fp, better)
        elif obs.record.optimized is not None:
            optimized = obs.record.optimized   # reuse the amortized rewrite

        # 5. Build the Foreman record (savings attributed by category).
        savings = cat.savings_from_rules(tier0.results, self.config.reference_tier)
        record = cat.ForemanRecord(
            fingerprint=obs.fp,
            tier=obs.tier,
            tokens_in_raw=estimate_tokens(prompt),
            tokens_in_optimized=estimate_tokens(optimized),
            savings=savings,
        )

        return OptimizationResult(
            optimized_prompt=optimized,
            ir=ir,
            tier0=tier0,
            observation=obs,
            record=record,
            cacheable_prefix=cache_prefix(prompt, ir),
        )

    def attach_node_reports(self, result: OptimizationResult,
                            nodes: list[cat.NodeReport]) -> None:
        """After Foreman deconstructs + routes, feed the per-node spend back in
        so the record carries both savings and spend."""
        result.record.nodes.extend(nodes)
