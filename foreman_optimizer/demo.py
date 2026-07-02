# Copyright 2026 Foreman contributors
# SPDX-License-Identifier: Apache-2.0
"""
End-to-end demo. Run: python -m foreman_optimizer.demo
"""

from __future__ import annotations

import json

from . import (
    ForemanOptimizer, OptimizerConfig, Tier0Config,
    NodeReport, ModelTier, PromptIR,
)


def _tier1_stub(optimized: str, ir: PromptIR) -> str:
    """Stand-in for a real LLM rewrite (DSPy/GEPA/LLMLingua). Here it just tags
    the prompt so you can see when the hot path fires."""
    return "[tier-1 rewrite] " + optimized


def banner(t):
    print("\n" + "=" * 68 + f"\n{t}\n" + "=" * 68)


def main():
    opt = ForemanOptimizer(
        config=OptimizerConfig(
            tier0=Tier0Config(prefer_structured_output=False),
            promote_threshold=3,          # low, so the demo promotes quickly
        ),
        tier1_hook=_tier1_stub,
    )

    # A verbose, one-off-looking prompt with a variable slot (the order id).
    verbose = (
        "As a very helpful AI assistant, I would like you to please carefully "
        "analyze the following customer support ticket and could you kindly "
        'determine the sentiment. The ticket is: "order 48213 arrived broken". '
        "Please note that I want you to be thorough. Please note that I want you "
        "to be thorough."
    )

    banner("1. TIER-0 on a single prompt")
    r = opt.optimize(verbose)
    print("RAW      :", verbose)
    print("\nOPTIMIZED:", r.optimized_prompt)
    print("\nTier:", r.observation.tier, "| fingerprint:", r.record.fingerprint)
    print("Cacheable prefix:", repr(r.cacheable_prefix[:60] + "..."))

    banner("2. COST-ANNOTATED IR (what Foreman deconstruction consumes)")
    print(json.dumps(r.ir.to_dict(), indent=2)[:1400], "...")

    banner("3. FOREMAN RECORD (savings attributed by category)")
    print(json.dumps(r.record.to_dict(), indent=2))

    banner("4. TEMPLATE PROMOTION (same skeleton, different slot values)")
    for oid in ("48213", "99001", "10774", "55555"):
        p = verbose.replace("48213", oid)
        res = opt.optimize(p)
        print(f"order {oid}: tier={res.observation.tier:16s} count={res.observation.count}")

    banner("5. ATTACH DECOMPOSED NODE SPEND (post-routing)")
    res = opt.optimize(verbose)
    opt.attach_node_reports(res, [
        NodeReport("extract_entities", "model_routing", ModelTier.BUDGET.value, input_tokens=40, output_tokens=15),
        NodeReport("classify_sentiment", "model_routing", ModelTier.BUDGET.value, input_tokens=30, output_tokens=5),
        NodeReport("summarize", "model_routing", ModelTier.MID.value, input_tokens=25, output_tokens=20),
    ])
    print(json.dumps(res.record.to_dict(), indent=2))

    banner("6. HOT TEMPLATES (what Foreman surfaces as spend concentration)")
    for t in opt.promoter.hot_templates(5):
        print(f"count={t.count:3d}  tokens={t.total_tokens:5d}  saved={t.saved_tokens:4d}  "
              f"promoted={t.promoted}  skeleton={t.skeleton[:50]!r}")


if __name__ == "__main__":
    main()
