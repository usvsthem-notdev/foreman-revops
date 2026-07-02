# CLAUDE.md — foreman_optimizer

Context for Claude Code working on this package. Read before editing.

## What this is

`foreman_optimizer` is the **first-layer, pre-deconstruction pass** for Foreman
(a local-first, Apache-2.0 FinOps tool for tracking/categorizing LLM spend). It
rewrites a raw prompt to cut token cost, then hands the optimized prompt to
Foreman's existing deconstruction step and emits a categorized spend/savings
record.

It attacks two cost axes:
- **Input** — the bulk of token *volume* in input-heavy workloads.
- **Output** — priced ~4:1 (up to 8:1 on reasoning models) per token, so a
  small output-shaping directive outweighs many stripped input tokens.

## Hard constraints (do not break)

- **Stdlib only.** No third-party runtime deps — Foreman must stay lean. Real
  tokenizers / LLM optimizers / LLMLingua enter *only* through the extension
  hooks below, never as hard imports.
- **Apache-2.0 header** on every `.py` file (`SPDX-License-Identifier: Apache-2.0`).
- **Tier-0 never calls an LLM.** It runs on every prompt including one-offs; it
  must stay free and deterministic.
- **Never mutate structured content.** Code fences and quoted strings are masked
  before any rule runs and restored after (see `rules._mask`). Aggressive
  pruning corrupts structured content (e.g. text-to-SQL accuracy collapse).
- **One-offs stay Tier-0 forever.** Only skeletons past the frequency threshold
  earn the expensive Tier-1 LLM rewrite, amortized across future hits. Don't add
  logic that rewrites unseen prompts with an LLM.

## Module map

| File | Responsibility | Key symbols |
|---|---|---|
| `ir.py` | Cost-annotated IR. Heuristic free parse → clauses tagged by cost driver + tags + tokens. | `parse()`, `PromptIR`, `Clause`, `CostDriver`, `ClauseKind`, `Tag`, `set_token_estimator()` |
| `rules.py` | Tier-0 deterministic rule pass. Structure-safe, self-reporting. | `run_tier0()`, `Tier0Config`, `Tier0Result`, `RuleResult`, `make_output_shaper()` |
| `fingerprint.py` | Slot-mask → skeleton → hash (Drain-style). Frequency promoter + stores. | `fingerprint()`, `skeleton()`, `cache_prefix()`, `FrequencyPromoter`, `InMemoryStore`, `SQLiteStore`, `TIER_0/TIER_1_ELIGIBLE/TIER_1_CACHED` |
| `categories.py` | Foreman spend mapping. Savings + per-node spend → one record. | `ForemanRecord`, `SavingsReport`, `NodeReport`, `SpendCategory`, `ModelTier`, `PRICING`, `cost_usd()`, `savings_from_rules()` |
| `pipeline.py` | Orchestrator: parse → Tier-0 → observe → build record. | `ForemanOptimizer`, `OptimizerConfig`, `OptimizationResult`, `Tier1Hook`, `attach_node_reports()` |
| `demo.py` | Runnable end-to-end example (6 stages). | `python -m foreman_optimizer.demo` |

## Data flow

```
raw prompt
   │  parse()            (free, heuristic)
   ▼
PromptIR  ──► run_tier0()          (free, deterministic, every prompt)
   │            │
   │            ▼  optimized prompt ──► Foreman deconstruction step
   │
   ├─► fingerprint()+observe()     (frequency counter)
   │        │
   │        └─ hot? ──► tier1_hook()  (LLM rewrite, amortized, cached)
   ▼
ForemanRecord {savings[], nodes[]} ──► foreman.ingest(...)
```

## Conventions

- `from __future__ import annotations` at top of every module.
- Full type hints. `dataclass` for all structured types; enums subclass `str`.
- Token estimates flow through `ir.estimate_tokens` — never hardcode a divisor
  elsewhere. Default is ~4 chars/token; production installs a real tokenizer
  via `set_token_estimator`.
- Rules must emit a `RuleResult` (tokens before/after + a Foreman category) so
  every saving is attributable. New categories go in `categories.SpendCategory`
  and `categories._RULE_TO_CATEGORY`.

## Open work / TODO (in rough priority order)

1. **Recompute cache prefix on the optimized skeleton.** `cache_prefix()` runs
   on the *raw* prompt today; once a template is hot, the cached block should be
   the Tier-0-cleaned version. Small change in `pipeline.optimize`.
2. **Real Tier-1 hook.** `demo._tier1_stub` is a placeholder. Wire a genuine
   optimizer (DSPy/GEPA) or LLMLingua-2 context compression behind
   `Tier1Hook`. Must respect `PROTECTED` spans.
3. **Eval-set graduation.** Let a `TemplateRecord` optionally carry a small
   labeled eval set; when present, that template switches from heuristic
   rewriting to true optimization (metric-driven). Add the field + a branch in
   the promoter.
4. **Drain upgrade path.** Exact skeleton hashing is fine until skeleton
   cardinality explodes. If it does, swap a Drain prefix tree behind
   `FrequencyPromoter.observe` without changing its signature.
5. **Output-reduction estimate.** `savings_from_rules` assumes a flat ~40-token
   output reduction for the concise directive. Replace with a measured
   per-template delta once real generation lengths are observed.
6. **Tests.** No test suite yet. Add `tests/` covering: filler idempotence,
   structure-protection (code/quotes survive), skeleton stability across slot
   values, promotion threshold, and savings accounting (no double-counting
   between filler_removal and redundancy_removal).

## Gotchas

- Filler removal runs *before* dedupe, so by dedupe time the duplicated clauses
  are already shortened — savings are split across two categories by design, not
  double-counted. Preserve this ordering.
- Output shaping is booked as a **negative** input saving (directive adds
  tokens) plus a positive output saving. Keep both legs.
- `ModelTier`/`PRICING` values are illustrative tiers, not vendor quotes. Treat
  as user-editable config, not facts to "correct."

## Honest limit to preserve in messaging

Without labeled data the fidelity guard verifies *"still satisfies the stated
requirements,"* not *"real task quality held."* Keep Tier-1 rewrites extractive
and verify against the Stage-1 requirements until a template earns an eval set.
Don't let the code or docs imply verified quality where there is none.
