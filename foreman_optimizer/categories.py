# Copyright 2026 Foreman contributors
# SPDX-License-Identifier: Apache-2.0
"""
Foreman category mapping.

Every unit of spend (and every unit *saved*) is attributed to a category so
Foreman can answer "where did the tokens go, and where did we save them?" at
the granularity of a decomposed node rather than a whole request.

Two report types:

  - SavingsReport: emitted by the Tier-0 rule pass. Attributes tokens *saved*
    to a reason (filler_removal, output_shaping, ...).
  - NodeReport: emitted per decomposed subtask. Attributes tokens *spent* to a
    routed model tier, so cheap-model routing shows up as a spend line.

A small, user-editable pricing table converts tokens -> dollars. Prices are
per-million-token and are illustrative tiers, not vendor quotes — edit to match
your actual model roster. Output is priced above input to reflect the ~4:1
median generation multiplier.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum


class SpendCategory(str, Enum):
    # --- input-side savings ---
    FILLER_REMOVAL = "filler_removal"
    REDUNDANCY_REMOVAL = "redundancy_removal"
    WHITESPACE = "whitespace"
    CONTEXT_COMPRESSION = "context_compression"   # LLMLingua-style (Tier-1+)
    CACHE_PREFIX = "cache_prefix"                  # moved to cached prefix
    # --- output-side ---
    OUTPUT_SHAPING = "output_shaping"              # length/format directives
    # --- routing / structural ---
    MODEL_ROUTING = "model_routing"                # sent to cheaper tier
    TEMPLATE_REUSE = "template_reuse"              # Tier-1 rewrite amortized


class ModelTier(str, Enum):
    BUDGET = "budget"        # simple classification/extraction
    MID = "mid"
    FLAGSHIP = "flagship"    # complex reasoning only


# Per-million-token prices (input, output). EDIT to match your roster.
PRICING: dict[str, tuple[float, float]] = {
    ModelTier.BUDGET.value:   (0.25, 1.25),
    ModelTier.MID.value:      (2.50, 10.00),
    ModelTier.FLAGSHIP.value: (10.00, 50.00),
}


def cost_usd(input_tokens: int, output_tokens: int, tier: str) -> float:
    pin, pout = PRICING.get(tier, PRICING[ModelTier.MID.value])
    return (input_tokens * pin + output_tokens * pout) / 1_000_000


@dataclass
class SavingsReport:
    """One line of 'we saved N tokens because X'. Foreman sums these."""
    category: str
    tokens_saved: int
    axis: str                      # "input" | "output"
    # Value the saving in dollars at a reference tier (default: mid).
    reference_tier: str = ModelTier.MID.value

    @property
    def usd_saved(self) -> float:
        if self.axis == "output":
            return cost_usd(0, self.tokens_saved, self.reference_tier)
        return cost_usd(self.tokens_saved, 0, self.reference_tier)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["usd_saved"] = round(self.usd_saved, 6)
        return d


@dataclass
class NodeReport:
    """One decomposed subtask's spend. Emitted after deconstruction/routing."""
    node_id: str
    category: str
    model_tier: str
    input_tokens: int
    output_tokens: int = 0        # estimate or actual, post-call

    @property
    def usd(self) -> float:
        return cost_usd(self.input_tokens, self.output_tokens, self.model_tier)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["usd"] = round(self.usd, 6)
        return d


@dataclass
class ForemanRecord:
    """The complete pre-deconstruction record Foreman ingests for one prompt."""
    fingerprint: str
    tier: str
    tokens_in_raw: int
    tokens_in_optimized: int
    savings: list[SavingsReport] = field(default_factory=list)
    nodes: list[NodeReport] = field(default_factory=list)

    @property
    def total_saved_usd(self) -> float:
        return sum(s.usd_saved for s in self.savings)

    @property
    def total_spend_usd(self) -> float:
        return sum(n.usd for n in self.nodes)

    def to_dict(self) -> dict:
        return {
            "fingerprint": self.fingerprint,
            "tier": self.tier,
            "tokens_in_raw": self.tokens_in_raw,
            "tokens_in_optimized": self.tokens_in_optimized,
            "input_tokens_saved": self.tokens_in_raw - self.tokens_in_optimized,
            "total_saved_usd": round(self.total_saved_usd, 6),
            "total_spend_usd": round(self.total_spend_usd, 6),
            "savings": [s.to_dict() for s in self.savings],
            "nodes": [n.to_dict() for n in self.nodes],
        }


# Map a Tier-0 rule category string to (SpendCategory, axis).
_RULE_TO_CATEGORY = {
    "filler_removal": (SpendCategory.FILLER_REMOVAL, "input"),
    "redundancy_removal": (SpendCategory.REDUNDANCY_REMOVAL, "input"),
    "whitespace": (SpendCategory.WHITESPACE, "input"),
    "output_shaping": (SpendCategory.OUTPUT_SHAPING, "output"),
}


def savings_from_rules(rule_results, reference_tier: str = ModelTier.MID.value) -> list[SavingsReport]:
    """Convert Tier-0 RuleResults into Foreman SavingsReports.

    Output-shaping is special: it *adds* input tokens to remove pricier output
    tokens. We record the input cost as negative saving and estimate the output
    reduction conservatively (assume it caps an otherwise-unbounded response).
    """
    reports: list[SavingsReport] = []
    for r in rule_results:
        cat, axis = _RULE_TO_CATEGORY.get(r.category, (SpendCategory.WHITESPACE, "input"))
        if r.category == "output_shaping":
            # Input tokens added by the directive (a small negative saving)...
            added = r.tokens_after - r.tokens_before
            if added > 0:
                reports.append(SavingsReport(cat.value, -added, "input", reference_tier))
            # ...offset by an estimated output reduction. Conservative default:
            # a concise directive trims ~40 output tokens vs an unbounded reply.
            reports.append(SavingsReport(cat.value, 40, "output", reference_tier))
        elif r.tokens_saved:
            reports.append(SavingsReport(cat.value, r.tokens_saved, axis, reference_tier))
    return reports
