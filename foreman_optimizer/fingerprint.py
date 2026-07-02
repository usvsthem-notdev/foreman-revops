# Copyright 2026 Foreman contributors
# SPDX-License-Identifier: Apache-2.0
"""
Fingerprint + frequency promoter.

Turns "is this a reusable template?" from a guess into a rolling counter. Each
incoming prompt is normalized by masking its variable slots (numbers, quoted
strings, entities, urls...), leaving a skeleton; the skeleton is hashed. Count
skeletons; when one crosses a threshold, promote it to Tier-1 (LLM rewrite).

This is the Drain log-template-mining idea applied to prompts. We use exact
skeleton hashing rather than Drain's full prefix tree — simpler, and sufficient
until skeleton cardinality explodes, at which point swap in a real Drain tree
behind the same `observe()` interface.

The masked (static) portion of a hot skeleton is exactly the prompt-cache
prefix; the slots are the cache-busting tail. So the fingerprinter also answers
"what is safely cacheable?" for free.

Storage is pluggable: in-memory by default, SQLite for local-first persistence
(Foreman already lives on the user's machine, so state survives restarts).
"""

from __future__ import annotations

import hashlib
import re
import sqlite3
import time
from dataclasses import dataclass, field
from typing import Optional, Protocol

from .ir import PromptIR


# --------------------------------------------------------------------------- #
# Normalization -> skeleton -> fingerprint
# --------------------------------------------------------------------------- #

_SLOT = "\x00S\x00"

# Same slot family as ir._SLOT_PATTERNS, applied for masking here.
_MASK_PATTERNS = [
    re.compile(r"```.*?```", re.DOTALL),
    re.compile(r"`[^`]+`"),
    re.compile(r'"[^"]*"'),
    re.compile(r"'[^']*'"),
    re.compile(r"https?://\S+"),
    re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b"),
    re.compile(r"\b\d{4}-\d{2}-\d{2}\b"),
    re.compile(r"\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b"),
    re.compile(r"\b\d[\d,.]*\b"),
]


def skeleton(prompt: str, ir: Optional[PromptIR] = None) -> str:
    """Mask variable slots to produce a stable template skeleton."""
    text = prompt
    for pat in _MASK_PATTERNS:
        text = pat.sub(_SLOT, text)
    # If an IR is supplied, also mask its detected entity spans (right-to-left
    # so earlier offsets stay valid).
    if ir is not None and ir.slots:
        for start, end in sorted(ir.slots, reverse=True):
            text = text[:start] + _SLOT + text[end:]
    # Collapse whitespace and repeated slots so trivial variance doesn't fork
    # the template.
    text = re.sub(r"\s+", " ", text).strip().lower()
    text = re.sub(r"(?:\x00S\x00\s*){2,}", _SLOT + " ", text)
    return text


def fingerprint(prompt: str, ir: Optional[PromptIR] = None) -> str:
    sk = skeleton(prompt, ir)
    return hashlib.blake2b(sk.encode("utf-8"), digest_size=12).hexdigest()


def cache_prefix(prompt: str, ir: Optional[PromptIR] = None) -> str:
    """The static leading portion up to the first variable slot — safe to place
    in a cached prompt prefix."""
    text = prompt
    first = len(text)
    for pat in _MASK_PATTERNS:
        m = pat.search(text)
        if m:
            first = min(first, m.start())
    if ir and ir.slots:
        first = min(first, min(s for s, _ in ir.slots))
    return prompt[:first].rstrip()


# --------------------------------------------------------------------------- #
# Store protocol + implementations
# --------------------------------------------------------------------------- #

@dataclass
class TemplateRecord:
    fp: str
    skeleton: str
    count: int = 0
    total_tokens: int = 0        # cumulative tokens seen (spend concentration)
    saved_tokens: int = 0        # cumulative Tier-0 savings on this template
    promoted: bool = False
    optimized: Optional[str] = None   # cached Tier-1 rewrite, once produced
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)


class TemplateStore(Protocol):
    def get(self, fp: str) -> Optional[TemplateRecord]: ...
    def put(self, rec: TemplateRecord) -> None: ...
    def top(self, n: int) -> list[TemplateRecord]: ...


class InMemoryStore:
    def __init__(self):
        self._d: dict[str, TemplateRecord] = {}

    def get(self, fp: str) -> Optional[TemplateRecord]:
        return self._d.get(fp)

    def put(self, rec: TemplateRecord) -> None:
        self._d[rec.fp] = rec

    def top(self, n: int) -> list[TemplateRecord]:
        return sorted(self._d.values(), key=lambda r: r.total_tokens, reverse=True)[:n]


class SQLiteStore:
    """Local-first persistence. Point it at Foreman's existing db if desired."""
    def __init__(self, path: str = "foreman_templates.db"):
        self.conn = sqlite3.connect(path)
        self.conn.execute(
            """CREATE TABLE IF NOT EXISTS templates (
                   fp TEXT PRIMARY KEY, skeleton TEXT, count INTEGER,
                   total_tokens INTEGER, saved_tokens INTEGER, promoted INTEGER,
                   optimized TEXT, first_seen REAL, last_seen REAL)"""
        )
        self.conn.commit()

    def get(self, fp: str) -> Optional[TemplateRecord]:
        row = self.conn.execute("SELECT * FROM templates WHERE fp=?", (fp,)).fetchone()
        if not row:
            return None
        return TemplateRecord(row[0], row[1], row[2], row[3], row[4], bool(row[5]), row[6], row[7], row[8])

    def put(self, rec: TemplateRecord) -> None:
        self.conn.execute(
            "REPLACE INTO templates VALUES (?,?,?,?,?,?,?,?,?)",
            (rec.fp, rec.skeleton, rec.count, rec.total_tokens, rec.saved_tokens,
             int(rec.promoted), rec.optimized, rec.first_seen, rec.last_seen),
        )
        self.conn.commit()

    def top(self, n: int) -> list[TemplateRecord]:
        rows = self.conn.execute(
            "SELECT * FROM templates ORDER BY total_tokens DESC LIMIT ?", (n,)
        ).fetchall()
        return [TemplateRecord(r[0], r[1], r[2], r[3], r[4], bool(r[5]), r[6], r[7], r[8]) for r in rows]


# --------------------------------------------------------------------------- #
# Promoter
# --------------------------------------------------------------------------- #

@dataclass
class Tier(str):
    pass


TIER_0 = "tier-0"        # deterministic only (one-offs stay here forever)
TIER_1_ELIGIBLE = "tier-1-eligible"   # hot enough to justify an LLM rewrite
TIER_1_CACHED = "tier-1-cached"       # optimized skeleton already stored


@dataclass
class Observation:
    fp: str
    tier: str
    count: int
    record: TemplateRecord


class FrequencyPromoter:
    """Observes prompts and decides when a template earns a Tier-1 rewrite."""

    def __init__(self, store: Optional[TemplateStore] = None, threshold: int = 5):
        self.store = store or InMemoryStore()
        self.threshold = threshold

    def observe(self, prompt: str, tokens: int, saved: int = 0, ir: Optional[PromptIR] = None) -> Observation:
        fp = fingerprint(prompt, ir)
        rec = self.store.get(fp)
        if rec is None:
            rec = TemplateRecord(fp=fp, skeleton=skeleton(prompt, ir))
        rec.count += 1
        rec.total_tokens += tokens
        rec.saved_tokens += saved
        rec.last_seen = time.time()

        if rec.optimized is not None:
            tier = TIER_1_CACHED
        elif rec.count >= self.threshold:
            rec.promoted = True
            tier = TIER_1_ELIGIBLE
        else:
            tier = TIER_0

        self.store.put(rec)
        return Observation(fp=fp, tier=tier, count=rec.count, record=rec)

    def store_optimized(self, fp: str, optimized: str) -> None:
        rec = self.store.get(fp)
        if rec:
            rec.optimized = optimized
            self.store.put(rec)

    def hot_templates(self, n: int = 10) -> list[TemplateRecord]:
        """Highest spend-concentration templates — what Foreman should surface."""
        return self.store.top(n)
