"""
Spend Intelligence — Detect, Propose, Guardrails.

Based on FIG. 03 of the Foreman design:
  Burn Map → 01 Detect → 02 Propose → 03 Guardrails → 04 Workload Library → 05 Policy Router (loop)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import pandas as pd

# ---------------------------------------------------------------------------
# Detection thresholds
# ---------------------------------------------------------------------------

_CONCENTRATION_THRESHOLD = 0.5   # single model > 50% of spend = concentration risk
_REASONING_WASTE_THRESHOLD = 0.3 # reasoning tokens > 30% of total = potential waste
_DRIFT_DAYS = 7                  # compare last 7 days to prior 7

# Cheaper alternative suggestions (model prefix → suggested alternative)
_CHEAPER_ALTERNATIVES: dict[str, tuple[str, float]] = {
    "claude-opus":      ("claude-haiku",    0.017),   # ~98% cheaper input
    "claude-sonnet":    ("claude-haiku",    0.083),
    "gpt-4o":           ("gpt-3.5-turbo",  0.20),
    "gpt-4":            ("gpt-3.5-turbo",  0.017),
    "o1":               ("gpt-4o",          0.25),
    "o3":               ("gpt-4o",          0.25),
}

_WORKLOAD_RECOMMENDATIONS: dict[str, str] = {
    "extract":  "claude-haiku or gpt-3.5-turbo — extraction is structured and rarely needs frontier reasoning.",
    "rag":      "A local embedding model (BGE-large) + haiku-class generation covers most RAG workloads.",
    "reason":   "Appropriate for frontier models, but consider o1-mini or Sonnet before Opus/o3.",
    "agents":   "Agent loops compound cost. Enforce per-step budgets and absorb planning steps locally.",
    "coding":   "claude-sonnet-4 / gpt-4o-mini cover most code tasks. Reserve Opus/o3 for hard proofs.",
    "unknown":  "Tag these entries with a workload_class to unlock class-specific recommendations.",
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


# ---------------------------------------------------------------------------
# Proposals
# ---------------------------------------------------------------------------

def propose(df: pd.DataFrame, findings: list[Finding]) -> list[Proposal]:
    proposals = []
    if df.empty:
        return proposals

    total = df["cost_usd"].sum()
    by_model = df.groupby("model")["cost_usd"].sum().to_dict()

    for model, spend in by_model.items():
        alt = _find_cheaper_alternative(model)
        if alt and spend > 1.0:
            alt_name, ratio = alt
            saved = spend * (1 - ratio)
            proposals.append(Proposal(
                title=f"Route {model} → {alt_name} for eligible tasks",
                action=(
                    f"Redirect extract, rag, and simple coding sub-tasks from {model} to {alt_name}. "
                    f"Estimate: ${saved:,.2f} savings on ${spend:,.2f} spend."
                ),
                estimated_savings_usd=saved,
                affected_models=[model],
                guardrail=(
                    "Run a backtested golden-eval before auto-routing. "
                    "Apply to suggest mode first; promote to auto after 7-day quality floor hold."
                ),
            ))

    # Absorb reasoning tokens locally
    reasoning_spend = df[df["reasoning_tokens"] > 0]["cost_usd"].sum()
    if reasoning_spend > 1.0:
        proposals.append(Proposal(
            title="Absorb planning steps with a local reasoning model",
            action=(
                "Deploy R1-Distill-14B or Qwen3-32B locally to handle decomposition and "
                f"planning steps. Estimated 60–80% of reasoning spend (${reasoning_spend * 0.7:,.2f}) "
                "is absorbable."
            ),
            estimated_savings_usd=reasoning_spend * 0.7,
            affected_models=[m for m in df[df["reasoning_tokens"] > 0]["model"].unique()],
            guardrail="Quality-gate on golden eval for each task class before enabling absorption.",
        ))

    return sorted(proposals, key=lambda p: p.estimated_savings_usd, reverse=True)


def _find_cheaper_alternative(model: str) -> Optional[tuple[str, float]]:
    model_lower = model.lower()
    for prefix, alt in _CHEAPER_ALTERNATIVES.items():
        if prefix in model_lower:
            return alt
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
