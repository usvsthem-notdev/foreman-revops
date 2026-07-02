"""
Spend Intelligence — Detect, Propose, Guardrails.

Based on FIG. 03 of the Foreman design:
  Burn Map → 01 Detect → 02 Propose → 03 Guardrails → 04 Workload Library → 05 Policy Router (loop)
"""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from src.analytics.pricing import cache_multipliers, price_for_model

# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------

_CONCENTRATION_THRESHOLD = 0.5   # single model > 50% of spend = concentration risk
_REASONING_WASTE_THRESHOLD = 0.3 # reasoning tokens > 30% of total = potential waste
_DRIFT_DAYS = 7                  # compare last 7 days to prior 7
_CACHE_HIT_FLOOR = 0.30          # below this hit rate on cache-friendly work = opportunity
_CACHE_DEGRADATION_FLOOR = 0.2   # prior hit rate must be at least this to "degrade"
_CACHE_DEGRADATION_DROP = 0.3    # relative hit-rate drop that counts as degradation
# Repeat-heavy classes where the same prefix (system prompt, tools, retrieved
# docs) is re-sent every call — where prompt caching actually pays off.
_CACHE_FRIENDLY_CLASSES = ("rag", "agents", "coding")
# Share of repeat-heavy input assumed to be a cacheable stable prefix.
_CACHEABLE_INPUT_SHARE = 0.6
# Latency-tolerant classes eligible for provider batch endpoints (50% off).
_BATCH_ELIGIBLE_CLASSES = ("extract", "rag")
_BATCH_DISCOUNT = 0.5
# Not every extract/rag call can wait up to 24h — assume this share can.
_BATCHABLE_SHARE = 0.8

# Cheaper alternative suggestions (model prefix → suggested alternative,
# approximate remaining-cost ratio from the July 2026 list prices in
# src.analytics.pricing, blended toward the input axis).
_CHEAPER_ALTERNATIVES: dict[str, tuple[str, float]] = {
    "claude-fable-5":   ("claude-sonnet-4-6", 0.30),
    "claude-opus":      ("claude-haiku-4-5",  0.20),
    "claude-sonnet":    ("claude-haiku-4-5",  0.33),
    "gpt-5.5":          ("gpt-5.4-mini",      0.15),
    "gpt-5.4":          ("gpt-5.4-mini",      0.30),
    "gpt-5":            ("gpt-5-mini",        0.20),
    "gpt-4o":           ("gpt-5.4-mini",      0.30),
    "o1":               ("gpt-5.4",           0.20),
    "o3":               ("gpt-5.4-mini",      0.38),
    "gemini-3-pro":     ("gemini-3-flash",    0.25),
    "gemini-3.5-flash": ("gemini-3-flash",    0.33),
}
# Longest key first so "gpt-5.4" wins over "gpt-5" for a gpt-5.4 model.
_ALT_LOOKUP_ORDER = sorted(_CHEAPER_ALTERNATIVES, key=len, reverse=True)

_WORKLOAD_RECOMMENDATIONS: dict[str, str] = {
    "extract": (
        "claude-haiku-4-5 or gpt-5.4-nano — extraction is structured and rarely "
        "needs frontier reasoning. Anything that can wait belongs on the Batch "
        "API: every major provider prices batch traffic at 50% off."
    ),
    "rag": (
        "A local embedding model (BGE-large) + haiku-class generation covers most "
        "RAG workloads. Cache the shared retrieval/system prefix — cached input "
        "is 75–90% off depending on provider."
    ),
    "reason": (
        "Price per token is the wrong unit here — what matters is dollars per "
        "task solved. Opus 4.8 bills more per token than Sonnet, but when it "
        "solves in fewer, less verbose turns it comes out cheaper per solved "
        "task. Benchmark both on your own workload (caching measured) and pick "
        "on total tokens to solution; tune effort/reasoning depth down where "
        "quality holds."
    ),
    "agents": (
        "Agent loops re-send the same system prompt and tool definitions every "
        "step — prompt caching is the single biggest lever here (90% off cached "
        "input on Anthropic and GPT-5-era OpenAI). Enforce per-step budgets."
    ),
    "coding": (
        "claude-sonnet-4-6 / gpt-5.4-mini cover most code tasks. Reserve "
        "Opus-tier for hard migrations and proofs."
    ),
    "unknown": (
        "Tag these entries with a workload_class to unlock class-specific recommendations."
    ),
}


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class Finding:
    severity: str           # "high" | "medium" | "low"
    category: str           # "concentration" | "waste" | "drift" | "untagged"
    title: str
    detail: str
    estimated_savings_usd: float = 0.0


@dataclass
class Proposal:
    title: str
    action: str
    estimated_savings_usd: float
    affected_models: list[str]
    guardrail: str          # the quality floor / rollback note


@dataclass
class IntelligenceReport:
    findings: list[Finding] = field(default_factory=list)
    proposals: list[Proposal] = field(default_factory=list)
    workload_library: dict[str, str] = field(default_factory=dict)
    total_potential_savings_usd: float = 0.0


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

def detect(df: pd.DataFrame) -> list[Finding]:
    if df.empty:
        return []
    findings: list[Finding] = []
    findings.extend(_detect_concentration(df))
    findings.extend(_detect_reasoning_waste(df))
    findings.extend(_detect_drift(df))
    findings.extend(_detect_untagged(df))
    findings.extend(_detect_cache_opportunity(df))
    findings.extend(_detect_cache_degradation(df))
    findings.extend(_detect_batch_opportunity(df))
    return sorted(findings, key=lambda f: {"high": 0, "medium": 1, "low": 2}[f.severity])


def _detect_concentration(df: pd.DataFrame) -> list[Finding]:
    findings = []
    total = df["cost_usd"].sum()
    if total == 0:
        return findings
    by_model = df.groupby("model")["cost_usd"].sum()
    for model, spend in by_model.items():
        pct = spend / total
        if pct > _CONCENTRATION_THRESHOLD:
            findings.append(Finding(
                severity="high" if pct > 0.7 else "medium",
                category="concentration",
                title=f"High concentration on {model}",
                detail=(
                    f"{pct:.0%} of spend (${spend:,.2f}) flows through a single model. "
                    "Consider splitting workload classes to cheaper alternatives."
                ),
                estimated_savings_usd=_estimate_model_savings(model, spend),
            ))
    return findings


def _detect_reasoning_waste(df: pd.DataFrame) -> list[Finding]:
    findings = []
    total_tok = df["input_tokens"].sum() + df["output_tokens"].sum() + df["reasoning_tokens"].sum()
    if total_tok == 0:
        return findings
    reason_tok = df["reasoning_tokens"].sum()
    pct = reason_tok / total_tok if total_tok > 0 else 0
    if pct > _REASONING_WASTE_THRESHOLD:
        reasoning_df = df[df["reasoning_tokens"] > 0]
        reasoning_spend = reasoning_df["cost_usd"].sum()
        findings.append(Finding(
            severity="high" if pct > 0.5 else "medium",
            category="waste",
            title="High reasoning token volume",
            detail=(
                f"{pct:.0%} of tokens are reasoning tokens ({reason_tok:,} tokens). "
                f"Reasoning tokens are billed but hidden — absorbing planning steps locally "
                f"could recover an estimated ${reasoning_spend * 0.6:,.2f}."
            ),
            estimated_savings_usd=reasoning_spend * 0.6,
        ))
    return findings


def _detect_drift(df: pd.DataFrame) -> list[Finding]:
    findings = []
    if df.empty:
        return findings
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    now = df["timestamp"].max()
    cutoff = now - pd.Timedelta(days=_DRIFT_DAYS)
    prior_cutoff = cutoff - pd.Timedelta(days=_DRIFT_DAYS)

    recent = df[df["timestamp"] >= cutoff]["cost_usd"].sum()
    prior  = df[(df["timestamp"] >= prior_cutoff) & (df["timestamp"] < cutoff)]["cost_usd"].sum()

    if prior > 0:
        change_pct = (recent - prior) / prior
        if change_pct > 0.3:
            findings.append(Finding(
                severity="high" if change_pct > 0.6 else "medium",
                category="drift",
                title=f"Spend up {change_pct:.0%} vs prior {_DRIFT_DAYS} days",
                detail=(
                    f"Last {_DRIFT_DAYS} days: ${recent:,.2f} vs prior period: ${prior:,.2f}. "
                    "Review recent deploys or new agent workflows."
                ),
            ))
        elif change_pct < -0.3:
            findings.append(Finding(
                severity="low",
                category="drift",
                title=f"Spend down {abs(change_pct):.0%} vs prior {_DRIFT_DAYS} days",
                detail="Notable spend decrease — verify this is expected and not a monitoring gap.",
            ))
    return findings


def _detect_untagged(df: pd.DataFrame) -> list[Finding]:
    findings = []
    if df.empty:
        return findings
    untagged = df[df["workload_class"] == "unknown"]
    if len(untagged) == 0:
        return findings
    pct = len(untagged) / len(df)
    spend = untagged["cost_usd"].sum()
    if pct > 0.2:
        findings.append(Finding(
            severity="low",
            category="untagged",
            title=f"{pct:.0%} of entries have no workload class",
            detail=(
                f"{len(untagged):,} entries (${spend:,.2f}) are tagged 'unknown'. "
                "Tagging enables class-specific routing recommendations."
            ),
        ))
    return findings


def _uncached_input_savings(df: pd.DataFrame) -> tuple[float, float, float]:
    """(potential_savings_usd, cache_hit_rate, frontier_input_tokens) for
    cache-friendly frontier work. Savings = list price of the uncached input
    tokens on cache-capable models × the assumed stable-prefix share × that
    model's cache-read discount."""
    work = df[~df["is_local"] & df["workload_class"].isin(_CACHE_FRIENDLY_CLASSES)]
    if work.empty or work["input_tokens"].sum() == 0:
        return 0.0, 0.0, 0.0

    cache_read = (
        work["cache_read_tokens"]
        if "cache_read_tokens" in work.columns
        else pd.Series(0, index=work.index)
    )
    total_input = float(work["input_tokens"].sum())
    hit_rate = float(cache_read.sum()) / total_input

    savings = 0.0
    for model, grp in work.groupby("model"):
        read_mult, _ = cache_multipliers(model)
        if read_mult >= 1.0:
            continue  # model has no cache discount to capture
        in_price, _ = price_for_model(model)
        grp_read = (
            grp["cache_read_tokens"].sum() if "cache_read_tokens" in grp.columns else 0
        )
        uncached = max(float(grp["input_tokens"].sum()) - float(grp_read), 0.0)
        savings += (
            uncached * _CACHEABLE_INPUT_SHARE * in_price * (1 - read_mult) / 1_000_000
        )
    return savings, hit_rate, total_input


def _detect_cache_opportunity(df: pd.DataFrame) -> list[Finding]:
    savings, hit_rate, total_input = _uncached_input_savings(df)
    if savings < 1.0 or hit_rate >= _CACHE_HIT_FLOOR:
        return []
    return [Finding(
        severity="high" if hit_rate < 0.1 else "medium",
        category="caching",
        title=f"Prompt cache hit rate is {hit_rate:.0%} on repeat-heavy workloads",
        detail=(
            f"RAG, agent, and coding calls re-send the same prefix every request, "
            f"but only {hit_rate:.0%} of their {total_input:,.0f} input tokens hit "
            f"the cache. Cached input is billed at 10–25% of list price — an "
            f"estimated ${savings:,.2f} is recoverable at current volume."
        ),
        estimated_savings_usd=savings,
    )]


def _detect_cache_degradation(df: pd.DataFrame) -> list[Finding]:
    """Prompt caches invalidate silently — a prefix edit, reordered tool
    definitions, or dynamic content injected early in the prompt kills the
    hit rate while everyone keeps assuming the discount. Compare the trailing
    7 days' hit rate on cache-friendly work to the prior 7 days and flag a
    meaningful relative drop."""
    work = df[~df["is_local"] & df["workload_class"].isin(_CACHE_FRIENDLY_CLASSES)]
    if work.empty or "cache_read_tokens" not in work.columns:
        return []

    ts = pd.to_datetime(work["timestamp"])
    anchor = ts.max()
    cutoff = anchor - pd.Timedelta(days=_DRIFT_DAYS)
    prior_cutoff = cutoff - pd.Timedelta(days=_DRIFT_DAYS)
    recent = work[ts > cutoff]
    prior = work[(ts > prior_cutoff) & (ts <= cutoff)]

    def _hit_rate(grp: pd.DataFrame) -> float | None:
        total = float(grp["input_tokens"].sum())
        return float(grp["cache_read_tokens"].sum()) / total if total > 0 else None

    r_recent, r_prior = _hit_rate(recent), _hit_rate(prior)
    if r_recent is None or r_prior is None or r_prior < _CACHE_DEGRADATION_FLOOR:
        return []
    rel_drop = (r_prior - r_recent) / r_prior
    if rel_drop < _CACHE_DEGRADATION_DROP:
        return []

    # $ lost = the discount the now-uncached share of recent input gave up.
    lost = 0.0
    for model, grp in recent.groupby("model"):
        read_mult, _ = cache_multipliers(model)
        if read_mult >= 1.0:
            continue
        in_price, _ = price_for_model(model)
        delta_tokens = float(grp["input_tokens"].sum()) * (r_prior - r_recent)
        lost += delta_tokens * in_price * (1 - read_mult) / 1_000_000

    return [Finding(
        severity="high" if rel_drop > 0.5 else "medium",
        category="cache_degradation",
        title=(
            f"Cache hit rate fell {r_prior:.0%} → {r_recent:.0%} on "
            "repeat-heavy workloads"
        ),
        detail=(
            "Prompt caches invalidate silently — an edited system prompt, "
            "reordered tool definitions, or dynamic content injected before "
            "the stable prefix kills the hit rate while you keep assuming the "
            f"discount. Estimated ${lost:,.2f} of discount given up in the "
            f"last {_DRIFT_DAYS} days. Check recent deploys that touch prompts."
        ),
        estimated_savings_usd=lost,
    )]


def _batch_eligible_spend(df: pd.DataFrame) -> float:
    """Frontier spend in latency-tolerant classes eligible for batch pricing."""
    eligible = df[~df["is_local"] & df["workload_class"].isin(_BATCH_ELIGIBLE_CLASSES)]
    return float(eligible["cost_usd"].sum())


def _detect_batch_opportunity(df: pd.DataFrame) -> list[Finding]:
    spend = _batch_eligible_spend(df)
    savings = spend * _BATCHABLE_SHARE * _BATCH_DISCOUNT
    if savings < 1.0:
        return []
    total = df["cost_usd"].sum()
    return [Finding(
        severity="medium" if spend / total > 0.2 else "low",
        category="batch",
        title=f"${spend:,.2f} of latency-tolerant spend is paying real-time prices",
        detail=(
            "Extraction and RAG-indexing traffic rarely needs a synchronous "
            "response. Every major provider prices batch endpoints at 50% off "
            f"with ≤24h turnaround — an estimated ${savings:,.2f} is recoverable "
            "by queueing this work."
        ),
        estimated_savings_usd=savings,
    )]


# ---------------------------------------------------------------------------
# Proposals
# ---------------------------------------------------------------------------

def propose(df: pd.DataFrame, findings: list[Finding]) -> list[Proposal]:
    proposals = []
    if df.empty:
        return proposals

    by_model = df.groupby("model")["cost_usd"].sum().to_dict()

    for model, spend in by_model.items():
        alt = _find_cheaper_alternative(model)
        if alt and spend > 1.0:
            alt_name, ratio = alt
            saved = spend * (1 - ratio)
            proposals.append(Proposal(
                title=f"Route {model} → {alt_name} for eligible tasks",
                action=(
                    f"Redirect extract, rag, and simple coding sub-tasks"
                    f" from {model} to {alt_name}."
                    f" Estimate: ${saved:,.2f} savings on ${spend:,.2f} spend."
                ),
                estimated_savings_usd=saved,
                affected_models=[model],
                guardrail=(
                    "Benchmark on cost-per-solved-task against your own workload, "
                    "caching measured — a lower per-token price that needs more "
                    "turns to solve the task is the more expensive choice. Run a "
                    "backtested golden-eval before auto-routing; promote to auto "
                    "after a 7-day quality-floor hold."
                ),
            ))

    # Absorb reasoning tokens locally
    reasoning_spend = df[df["reasoning_tokens"] > 0]["cost_usd"].sum()
    if reasoning_spend > 1.0:
        proposals.append(Proposal(
            title="Absorb planning steps with a local reasoning model",
            action=(
                "Deploy R1-Distill-14B or Qwen3-32B locally to handle decomposition and "
                f"planning steps. Estimated 60–80% of reasoning spend"
                f" (${reasoning_spend * 0.7:,.2f}) is absorbable."
            ),
            estimated_savings_usd=reasoning_spend * 0.7,
            affected_models=[m for m in df[df["reasoning_tokens"] > 0]["model"].unique()],
            guardrail="Quality-gate on golden eval for each task class before enabling absorption.",
        ))

    # Prompt caching on repeat-heavy workloads
    cache_savings, hit_rate, _ = _uncached_input_savings(df)
    if cache_savings > 1.0 and hit_rate < _CACHE_HIT_FLOOR:
        cache_models = sorted(
            df[~df["is_local"]
               & df["workload_class"].isin(_CACHE_FRIENDLY_CLASSES)]["model"].unique()
        )
        proposals.append(Proposal(
            title="Enable prompt caching on RAG, agent, and coding calls",
            action=(
                "Mark the stable prefix (system prompt, tool definitions, shared "
                "retrieval context) with a cache breakpoint and keep volatile "
                "content after it. Cached input bills at 10–25% of list price — "
                f"estimated ${cache_savings:,.2f} recoverable at current volume."
            ),
            estimated_savings_usd=cache_savings,
            affected_models=list(cache_models),
            guardrail=(
                "Zero quality risk — caching changes what you pay, not what the "
                "model sees. Verify hit rate via usage.cache_read tokens after rollout."
            ),
        ))

    # Batch API for latency-tolerant classes
    batch_spend = _batch_eligible_spend(df)
    batch_savings = batch_spend * _BATCHABLE_SHARE * _BATCH_DISCOUNT
    if batch_savings > 1.0:
        batch_models = sorted(
            df[~df["is_local"]
               & df["workload_class"].isin(_BATCH_ELIGIBLE_CLASSES)]["model"].unique()
        )
        proposals.append(Proposal(
            title="Move latency-tolerant extraction/RAG-indexing to the Batch API",
            action=(
                f"Queue asynchronous extraction and indexing jobs through provider "
                f"batch endpoints (50% off list, ≤24h turnaround). Estimated "
                f"${batch_savings:,.2f} savings on ${batch_spend:,.2f} eligible spend."
            ),
            estimated_savings_usd=batch_savings,
            affected_models=list(batch_models),
            guardrail=(
                "Same models, same outputs — only the SLA changes. Keep a "
                "real-time fallback path for jobs that exceed the queue deadline."
            ),
        ))

    return sorted(proposals, key=lambda p: p.estimated_savings_usd, reverse=True)


def _find_cheaper_alternative(model: str) -> tuple[str, float] | None:
    model_lower = model.lower()
    for prefix in _ALT_LOOKUP_ORDER:
        if prefix in model_lower:
            alt_name, ratio = _CHEAPER_ALTERNATIVES[prefix]
            # Already on (or below) the suggested alternative — nothing to route.
            if alt_name in model_lower:
                return None
            return alt_name, ratio
    return None


def _estimate_model_savings(model: str, spend: float) -> float:
    alt = _find_cheaper_alternative(model)
    if alt:
        _, ratio = alt
        return spend * (1 - ratio)
    return 0.0


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------

def generate_report(df: pd.DataFrame) -> IntelligenceReport:
    findings = detect(df)
    proposals = propose(df, findings)
    total_savings = sum(p.estimated_savings_usd for p in proposals)
    return IntelligenceReport(
        findings=findings,
        proposals=proposals,
        workload_library=_WORKLOAD_RECOMMENDATIONS.copy(),
        total_potential_savings_usd=total_savings,
    )
