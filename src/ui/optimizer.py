"""
Prompt Optimizer page — first-layer, pre-deconstruction token-cost pass.

Lets a user paste a prompt they're about to send (from their own app, a
Cursor rule, an agent template, ...) and see the free Tier-0 rewrite, which
clauses drive input vs output cost, and — once the same prompt template has
been pasted enough times — whether it has earned a Tier-1 rewrite.

Dollar estimates use Foreman's own per-model pricing (src.analytics.pricing),
not foreman_optimizer.categories.PRICING — that table is an illustrative
placeholder by design (see foreman_optimizer/CLAUDE.md), so real Foreman
prices are applied here instead.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from foreman_optimizer import (
    TIER_0,
    TIER_1_CACHED,
    TIER_1_ELIGIBLE,
    ForemanOptimizer,
    FrequencyPromoter,
    SQLiteStore,
)
from foreman_optimizer.ir import estimate_tokens
from src.analytics.pricing import KNOWN_MODELS, cache_multipliers, price_for_model
from src.db import get_db_path

_TIER_LABELS = {
    TIER_0: "Tier-0 (one-off)",
    TIER_1_ELIGIBLE: "Tier-1 eligible (hot template)",
    TIER_1_CACHED: "Tier-1 cached rewrite",
}


@st.cache_resource
def _get_optimizer() -> ForemanOptimizer:
    """One optimizer instance per process — its SQLite-backed frequency
    promoter persists into the same foreman.db Foreman already uses."""
    store = SQLiteStore(str(get_db_path()))
    return ForemanOptimizer(promoter=FrequencyPromoter(store))


def _usd_for_savings(savings, model: str) -> tuple[float, dict[str, float]]:
    """Value each SavingsReport at Foreman's real per-model price instead of
    the package's illustrative ModelTier pricing."""
    in_price, out_price = price_for_model(model)
    by_category: dict[str, float] = {}
    total = 0.0
    for s in savings:
        price = out_price if s.axis == "output" else in_price
        usd = s.tokens_saved * price / 1_000_000
        by_category[s.category] = by_category.get(s.category, 0.0) + usd
        total += usd
    return total, by_category


def _cache_prefix_savings(prefix: str, occurrence_count: int, model: str) -> float:
    """Net $ saved if this static prefix were prompt-cached, across the times
    it's actually been seen: one cache-creation write (a premium over fresh
    input) followed by (occurrence_count - 1) cheap cache reads, versus
    paying full fresh-input price every time. Negative until a prefix has
    been seen at least twice — a single hit only pays the write premium."""
    if not prefix or occurrence_count < 1:
        return 0.0
    prefix_tokens = estimate_tokens(prefix)
    in_price, _ = price_for_model(model)
    read_mult, creation_mult = cache_multipliers(model)
    baseline = occurrence_count * prefix_tokens * in_price / 1_000_000
    with_caching = (
        prefix_tokens * in_price / 1_000_000 * creation_mult
        + (occurrence_count - 1) * prefix_tokens * in_price / 1_000_000 * read_mult
    )
    return baseline - with_caching


def render() -> None:
    opt = _get_optimizer()

    st.markdown(
        '<div class="foreman-section">ANALYZE A PROMPT</div>', unsafe_allow_html=True
    )
    st.caption(
        "Paste a prompt you're about to send (app template, agent system "
        "prompt, Cursor rule, ...). Tier-0 rewrites run for free, locally, "
        "on every prompt — no LLM call is made here."
    )

    prompt = st.text_area("Prompt", height=180, max_chars=20_000, key="optimizer_prompt")
    model = st.selectbox(
        "Price savings against",
        KNOWN_MODELS,
        index=KNOWN_MODELS.index("claude-sonnet-4") if "claude-sonnet-4" in KNOWN_MODELS else 0,
    )
    analyze = st.button("Analyze", type="primary", disabled=not prompt.strip())

    if not analyze:
        _render_hot_templates(opt)
        return

    result = opt.optimize(prompt)
    total_usd, by_category = _usd_for_savings(result.record.savings, model)

    tier = result.observation.tier
    c1, c2, c3 = st.columns(3)
    c1.metric("Tier", _TIER_LABELS.get(tier, tier))
    c2.metric("Times seen", result.observation.count)
    c3.metric("Est. $ saved / call", f"${total_usd:,.6f}")

    col_raw, col_opt = st.columns(2)
    with col_raw:
        st.markdown("**Raw**")
        st.text_area("raw_prompt", prompt, height=220, disabled=True, label_visibility="collapsed")
    with col_opt:
        st.markdown("**Tier-0 optimized**")
        st.text_area(
            "optimized_prompt", result.optimized_prompt, height=220,
            disabled=True, label_visibility="collapsed",
        )

    st.markdown('<div class="foreman-section">SAVINGS BY CATEGORY</div>', unsafe_allow_html=True)
    if by_category:
        cat_df = pd.DataFrame(
            [{"category": k, "usd_saved": v} for k, v in by_category.items()]
        ).sort_values("usd_saved", ascending=False)
        st.dataframe(cat_df, use_container_width=True, hide_index=True)
    else:
        st.caption("No rewrite categories fired — prompt is already tight.")

    ir = result.ir
    tokens_by_driver = ir.tokens_by_driver()
    st.markdown('<div class="foreman-section">COST DRIVER (IR)</div>', unsafe_allow_html=True)
    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Input-driving tokens", tokens_by_driver.get("input", 0))
    d2.metric("Output-driving tokens", tokens_by_driver.get("output", 0))
    d3.metric("Both", tokens_by_driver.get("both", 0))
    d4.metric("Filler (free to cut)", tokens_by_driver.get("none", 0))

    with st.expander("Cacheable prefix"):
        st.code(result.cacheable_prefix or "(no stable static prefix detected)")
        if result.cacheable_prefix:
            occurrence_count = result.observation.count
            savings = _cache_prefix_savings(result.cacheable_prefix, occurrence_count, model)
            if savings > 0:
                st.metric(
                    f"Est. $ saved if cached (seen {occurrence_count}x)",
                    f"${savings:,.6f}",
                )
            else:
                st.caption(
                    f"Seen {occurrence_count}x so far — caching this prefix pays "
                    "off starting on the 2nd hit (the 1st is a cache-write "
                    "premium, not a discount)."
                )

    st.divider()
    _render_hot_templates(opt)


def _render_hot_templates(opt: ForemanOptimizer) -> None:
    st.markdown('<div class="foreman-section">HOT TEMPLATES</div>', unsafe_allow_html=True)
    st.caption(
        "Recurring prompt skeletons by cumulative token volume — the "
        "concentration Tier-1 rewriting and prompt caching should target."
    )
    templates = opt.promoter.hot_templates(10)
    if not templates:
        st.info("No prompts analyzed yet.")
        return
    df = pd.DataFrame(
        [
            {
                "skeleton": t.skeleton[:80],
                "count": t.count,
                "total_tokens": t.total_tokens,
                "tier0_saved_tokens": t.saved_tokens,
                "promoted": t.promoted,
            }
            for t in templates
        ]
    )
    st.dataframe(df, use_container_width=True, hide_index=True)
