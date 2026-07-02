# Copyright 2026 Foreman contributors
# SPDX-License-Identifier: Apache-2.0
"""
Tier-0 rule pass: deterministic, free, non-destructive.

Runs unconditionally on *every* prompt (templates and one-offs alike) because
it costs zero tokens. Every rule is:

  - conservative (extractive, never rewrites meaning),
  - structure-safe (code/quoted spans are masked out first — aggressive pruning
    corrupts structured content, e.g. text-to-SQL accuracy collapses),
  - self-reporting (each rule emits tokens saved + a Foreman category so the
    savings are attributable, not just observed).

The LLM-based Tier-1 rewrite is intentionally *not* here — it lives behind the
frequency promoter and only fires for hot templates. See `pipeline.py`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from .ir import PromptIR, estimate_tokens


@dataclass
class RuleResult:
    rule: str
    category: str           # Foreman spend category (see categories.py)
    tokens_before: int
    tokens_after: int
    note: str = ""

    @property
    def tokens_saved(self) -> int:
        return self.tokens_before - self.tokens_after


@dataclass
class Tier0Result:
    original: str
    optimized: str
    results: list[RuleResult] = field(default_factory=list)

    @property
    def tokens_saved(self) -> int:
        return estimate_tokens(self.original) - estimate_tokens(self.optimized)

    @property
    def tokens_saved_input(self) -> int:
        return sum(r.tokens_saved for r in self.results if r.category != "output_shaping")

    @property
    def output_tokens_shaped(self) -> bool:
        return any(r.category == "output_shaping" for r in self.results)


# --------------------------------------------------------------------------- #
# Structure protection: mask code / quotes before mutating, restore after.
# --------------------------------------------------------------------------- #

_PROTECT = [
    re.compile(r"```.*?```", re.DOTALL),
    re.compile(r"`[^`]+`"),
    re.compile(r'"[^"]*"'),
    re.compile(r"'[^']*'"),
]


def _mask(text: str) -> tuple[str, dict[str, str]]:
    store: dict[str, str] = {}
    i = 0
    for pat in _PROTECT:
        def repl(m, _i=[i]):
            key = f"\x00P{_i[0]}\x00"
            store[key] = m.group(0)
            _i[0] += 1
            return key
        text = pat.sub(repl, text)
        i = len(store)
    return text, store


def _unmask(text: str, store: dict[str, str]) -> str:
    for key, val in store.items():
        text = text.replace(key, val)
    return text


# --------------------------------------------------------------------------- #
# Individual rules. Each: (name, category, fn(masked_text) -> new_masked_text)
# --------------------------------------------------------------------------- #

# 1. Filler / boilerplate removal.
_FILLER_SUBS = [
    (re.compile(r"(?i)\bas an?\s+(?:very\s+)?(?:helpful|friendly|expert)?\s*(?:ai\s+)?assistant,?\s*"), ""),
    (re.compile(r"(?i)\bi (?:would like|want) you to\s*"), ""),
    (re.compile(r"(?i)\byour task is to\s*"), ""),
    (re.compile(r"(?i)\bplease note that\s*"), ""),
    (re.compile(r"(?i)\bit would be great if you could\s*"), ""),
    (re.compile(r"(?i)\bcould you (?:please\s+)?"), ""),
    (re.compile(r"(?i)\bcan you (?:please\s+)?"), ""),
    (re.compile(r"(?i)\bkindly\s+"), ""),
    (re.compile(r"(?i)\bin order to\b"), "to"),
    (re.compile(r"(?i)\bthe following\b"), ""),
    (re.compile(r"(?i)\bmake sure (?:that\s+)?(?:you\s+)?"), ""),
]

# 2. Whitespace / repetition normalization.
_WS_MULTISPACE = re.compile(r"[ \t]{2,}")
_WS_MULTINEWLINE = re.compile(r"\n{3,}")


def rule_strip_filler(text: str) -> str:
    for pat, sub in _FILLER_SUBS:
        text = pat.sub(sub, text)
    # Capitalize the first surviving letter (avoid "provide..." lowercase starts).
    text = re.sub(r"^\s*([a-z])", lambda m: m.group(1).upper(), text)
    return text


def rule_normalize_whitespace(text: str) -> str:
    text = _WS_MULTISPACE.sub(" ", text)
    text = _WS_MULTINEWLINE.sub("\n\n", text)
    return "\n".join(line.rstrip() for line in text.split("\n")).strip()


def rule_dedupe_sentences(text: str) -> str:
    seen: set[str] = set()
    out: list[str] = []
    for sent in re.split(r"(?<=[.!?])\s+", text):
        key = sent.strip().lower()
        if key and key in seen:
            continue
        seen.add(key)
        out.append(sent)
    return " ".join(out)


# --------------------------------------------------------------------------- #
# Output-shaping rule (the high-value lever — output is ~4x pricier per token).
# This one *adds* a few input tokens to remove many pricier output tokens.
# --------------------------------------------------------------------------- #

_CONCISE_DIRECTIVE = "Be concise. Return only the answer, no preamble."
_STRUCTURED_DIRECTIVE = "Output as a compact JSON object. No prose."


def make_output_shaper(prefer_structured: bool = False) -> Callable[[PromptIR, str], tuple[str, str]]:
    """Returns a shaper that appends a terse output directive when the prompt
    lacks a length cap / format spec. Returns (new_text, note)."""
    def shaper(ir: PromptIR, text: str) -> tuple[str, str]:
        oc = ir.output_contract
        if oc.has_length_cap and oc.has_format_spec:
            return text, "already constrained"
        directive = _STRUCTURED_DIRECTIVE if prefer_structured and not oc.has_format_spec else _CONCISE_DIRECTIVE
        if directive.split(".")[0].lower() in text.lower():
            return text, "directive already present"
        return text.rstrip() + "\n\n" + directive, "appended output directive"
    return shaper


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #

@dataclass
class Tier0Config:
    strip_filler: bool = True
    normalize_whitespace: bool = True
    dedupe_sentences: bool = True
    shape_output: bool = True
    prefer_structured_output: bool = False


def run_tier0(ir: PromptIR, config: Tier0Config | None = None) -> Tier0Result:
    config = config or Tier0Config()
    original = ir.raw
    masked, store = _mask(original)
    results: list[RuleResult] = []

    def apply(name: str, category: str, fn: Callable[[str], str], txt: str, note: str = "") -> str:
        before = estimate_tokens(_unmask(txt, store))
        new = fn(txt)
        after = estimate_tokens(_unmask(new, store))
        results.append(RuleResult(name, category, before, after, note))
        return new

    text = masked
    if config.strip_filler:
        text = apply("strip_filler", "filler_removal", rule_strip_filler, text)
    if config.dedupe_sentences:
        text = apply("dedupe_sentences", "redundancy_removal", rule_dedupe_sentences, text)
    if config.normalize_whitespace:
        text = apply("normalize_whitespace", "whitespace", rule_normalize_whitespace, text)

    # Output shaping operates on the restored text (it appends a directive).
    text = _unmask(text, store)
    if config.shape_output:
        shaper = make_output_shaper(config.prefer_structured_output)
        before = estimate_tokens(text)
        text, note = shaper(ir, text)
        after = estimate_tokens(text)
        results.append(RuleResult("shape_output", "output_shaping", before, after, note))

    return Tier0Result(original=original, optimized=text, results=results)
