# Copyright 2026 Foreman contributors
# SPDX-License-Identifier: Apache-2.0
"""
Cost-annotated intermediate representation (IR).

This is the *first layer* Foreman sees, before prompt deconstruction. A raw
prompt is segmented into clauses, each annotated with:

  - which cost axis it drives (INPUT vs OUTPUT — output is ~4x pricier/token),
  - whether it is compressible / routable / cacheable,
  - a rough token estimate.

The parser here is deliberately *heuristic and free* (regex/stdlib only). It
never calls an LLM, because Tier-0 must run on every prompt including one-offs.
The IR doubles as the slot-masking source for the fingerprinter.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Callable, Optional


# --------------------------------------------------------------------------- #
# Token estimation (pluggable — swap in a real tokenizer in production)
# --------------------------------------------------------------------------- #

# Default: ~4 chars/token. Model-agnostic and free. Note that recent tokenizers
# (e.g. Opus 4.x) run ~35% heavier on code/structured data, so for code-dense
# workloads inject a real tokenizer via `set_token_estimator`.
_CHARS_PER_TOKEN = 4.0
_token_estimator: Callable[[str], int] = lambda s: max(1, round(len(s) / _CHARS_PER_TOKEN)) if s else 0


def set_token_estimator(fn: Callable[[str], int]) -> None:
    """Install a real tokenizer, e.g. tiktoken/anthropic counts."""
    global _token_estimator
    _token_estimator = fn


def estimate_tokens(text: str) -> int:
    return _token_estimator(text)


# --------------------------------------------------------------------------- #
# Enums
# --------------------------------------------------------------------------- #

class CostDriver(str, Enum):
    """Which price axis a clause moves."""
    INPUT = "input"        # adds prompt tokens only
    OUTPUT = "output"      # shapes generation length (4-8x pricier per token)
    BOTH = "both"
    NONE = "none"          # pure filler; removing it is free savings


class ClauseKind(str, Enum):
    ROLE = "role"                 # "You are a..."
    INSTRUCTION = "instruction"   # the actual ask
    CONTEXT = "context"           # background/data the model reads
    EXAMPLE = "example"           # few-shot demonstration
    QUESTION = "question"         # the concrete query
    OUTPUT_SPEC = "output_spec"   # format/length directives
    CONSTRAINT = "constraint"     # hard/soft rules
    FILLER = "filler"             # politeness/boilerplate, safe to drop


class Tag(str, Enum):
    COMPRESSIBLE = "compressible"   # extractive compression candidate
    ROUTABLE = "routable"           # subtask could go to a cheaper model
    CACHEABLE = "cacheable"         # static; belongs in the cached prefix
    VOLATILE = "volatile"           # varies per call; the cache-busting tail
    PROTECTED = "protected"         # code/quoted; never mutate (corruption risk)


# --------------------------------------------------------------------------- #
# Clause + PromptIR
# --------------------------------------------------------------------------- #

@dataclass
class Clause:
    text: str
    kind: ClauseKind
    cost_driver: CostDriver
    tags: set[Tag] = field(default_factory=set)
    tokens: int = 0

    def __post_init__(self):
        if not self.tokens:
            self.tokens = estimate_tokens(self.text)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["kind"] = self.kind.value
        d["cost_driver"] = self.cost_driver.value
        d["tags"] = sorted(t.value for t in self.tags)
        return d


@dataclass
class OutputContract:
    """What the prompt currently asks the model to produce."""
    has_length_cap: bool = False
    has_format_spec: bool = False
    format_hint: Optional[str] = None   # "json" | "list" | "prose" | ...
    requests_reasoning: bool = False    # invites long CoT (expensive output)

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class PromptIR:
    raw: str
    clauses: list[Clause] = field(default_factory=list)
    output_contract: OutputContract = field(default_factory=OutputContract)
    # (start, end) char spans of variable slots, for fingerprint masking.
    slots: list[tuple[int, int]] = field(default_factory=list)

    @property
    def total_tokens(self) -> int:
        return sum(c.tokens for c in self.clauses) or estimate_tokens(self.raw)

    def tokens_by_driver(self) -> dict[str, int]:
        out = {d.value: 0 for d in CostDriver}
        for c in self.clauses:
            out[c.cost_driver.value] += c.tokens
        return out

    def to_dict(self) -> dict:
        return {
            "raw": self.raw,
            "total_tokens": self.total_tokens,
            "tokens_by_driver": self.tokens_by_driver(),
            "output_contract": self.output_contract.to_dict(),
            "clauses": [c.to_dict() for c in self.clauses],
            "slots": self.slots,
        }


# --------------------------------------------------------------------------- #
# Heuristic parser
# --------------------------------------------------------------------------- #

# Signals that classify a clause's kind. Order matters (first hit wins).
_ROLE_RE = re.compile(r"(?i)^\s*(you are|act as|as an? .*assistant)\b")
_OUTPUT_RE = re.compile(
    r"(?i)\b(respond in|output (?:as|in)|format(?:ted)? as|return (?:a|an|only)|"
    r"in json|as a list|bullet points?|no more than|at most \d+|within \d+ (?:words|sentences))\b"
)
_CONSTRAINT_RE = re.compile(r"(?i)\b(must|do not|don't|never|always|only|ensure that|required to)\b")
_QUESTION_RE = re.compile(r"\?\s*$")
_EXAMPLE_RE = re.compile(r"(?i)^\s*(example|e\.g\.|for instance|input:|output:)\b")
_REASONING_RE = re.compile(r"(?i)\b(think step by step|reason through|explain your reasoning|show your work|chain of thought)\b")

# Filler: politeness/boilerplate that carries no task signal.
_FILLER_RE = re.compile(
    r"(?i)^\s*(please\b.*|thanks?\b.*|thank you\b.*|i would (?:like|appreciate)\b.*|"
    r"i want you to\b.*|it would be great if\b.*)$"
)

# Slot patterns for masking (numbers, quotes, urls, emails, dates, uuids).
_SLOT_PATTERNS = [
    re.compile(r"```.*?```", re.DOTALL),                     # fenced code
    re.compile(r"`[^`]+`"),                                  # inline code
    re.compile(r'"[^"]*"'),                                  # double-quoted
    re.compile(r"'[^']*'"),                                  # single-quoted
    re.compile(r"https?://\S+"),                             # urls
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),             # emails
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),                    # iso dates
    re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"),  # uuid
    re.compile(r"\b\d[\d,.]*\b"),                            # numbers
]


def _classify(text: str) -> ClauseKind:
    if _FILLER_RE.match(text):
        return ClauseKind.FILLER
    if _ROLE_RE.match(text):
        return ClauseKind.ROLE
    if _EXAMPLE_RE.match(text):
        return ClauseKind.EXAMPLE
    if _OUTPUT_RE.search(text):
        return ClauseKind.OUTPUT_SPEC
    if _CONSTRAINT_RE.search(text):
        return ClauseKind.CONSTRAINT
    if _QUESTION_RE.search(text):
        return ClauseKind.QUESTION
    return ClauseKind.INSTRUCTION


def _cost_driver(kind: ClauseKind, text: str) -> CostDriver:
    # OUTPUT_SPEC and reasoning invitations move the (expensive) output axis.
    if kind == ClauseKind.OUTPUT_SPEC:
        return CostDriver.OUTPUT
    if _REASONING_RE.search(text):
        return CostDriver.OUTPUT
    if kind == ClauseKind.FILLER:
        return CostDriver.NONE
    # Everything the model reads is input-side; questions can imply both.
    if kind == ClauseKind.QUESTION:
        return CostDriver.BOTH
    return CostDriver.INPUT


def _tag(kind: ClauseKind, text: str) -> set[Tag]:
    tags: set[Tag] = set()
    if "```" in text or "`" in text or '"' in text:
        tags.add(Tag.PROTECTED)          # structured content: don't aggressively prune
    if kind in (ClauseKind.CONTEXT, ClauseKind.EXAMPLE) and Tag.PROTECTED not in tags:
        tags.add(Tag.COMPRESSIBLE)
    if kind in (ClauseKind.ROLE, ClauseKind.INSTRUCTION, ClauseKind.OUTPUT_SPEC, ClauseKind.CONSTRAINT):
        tags.add(Tag.CACHEABLE)          # stable across calls -> cached prefix
    if kind in (ClauseKind.EXAMPLE, ClauseKind.QUESTION):
        tags.add(Tag.ROUTABLE)           # candidate for a cheaper model tier
    return tags


def _find_slots(text: str) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    for pat in _SLOT_PATTERNS:
        for m in pat.finditer(text):
            spans.append((m.start(), m.end()))
    # merge overlaps
    spans.sort()
    merged: list[tuple[int, int]] = []
    for s, e in spans:
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def parse(prompt: str) -> PromptIR:
    """Segment a raw prompt into a cost-annotated IR. Free; no LLM call."""
    ir = PromptIR(raw=prompt)

    # Split on blank lines first, then sentences, keeping fenced code intact.
    blocks = re.split(r"\n\s*\n", prompt.strip())
    for block in blocks:
        block = block.strip()
        if not block:
            continue
        # Keep code blocks whole; otherwise split into sentence-ish clauses.
        if block.startswith("```"):
            pieces = [block]
        else:
            pieces = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", block)
        for piece in pieces:
            piece = piece.strip()
            if not piece:
                continue
            kind = _classify(piece)
            ir.clauses.append(
                Clause(
                    text=piece,
                    kind=kind,
                    cost_driver=_cost_driver(kind, piece),
                    tags=_tag(kind, piece),
                )
            )

    # Output contract summary.
    oc = ir.output_contract
    oc.has_format_spec = any(c.kind == ClauseKind.OUTPUT_SPEC for c in ir.clauses)
    oc.requests_reasoning = bool(_REASONING_RE.search(prompt))
    if re.search(r"(?i)\b(no more than|at most|within \d+ (?:words|sentences|tokens))\b", prompt):
        oc.has_length_cap = True
    if re.search(r"(?i)\bjson\b", prompt):
        oc.format_hint = "json"
    elif re.search(r"(?i)\b(list|bullet)\b", prompt):
        oc.format_hint = "list"

    ir.slots = _find_slots(prompt)
    return ir
