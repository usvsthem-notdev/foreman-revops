# foreman_optimizer

A token-cost optimization layer that runs as the **first pass before Foreman's
prompt deconstruction**. Stdlib-only, local-first, Apache-2.0 — no new runtime
dependencies for Foreman.

It attacks cost on two axes: **input** (the bulk of token *volume* in
input-heavy workloads) and **output** (priced ~4x higher per token, so a small
directive here is worth many stripped input tokens).

## The four pieces

| Deliverable | Module | What it does |
|---|---|---|
| Cost-annotated IR | `ir.py` | Heuristic (free) parse of a prompt into clauses, each tagged with cost driver (input/output/none), tags (compressible/routable/cacheable/protected), and a token estimate. This is what deconstruction consumes. |
| Tier-0 rule pass | `rules.py` | Deterministic, free, structure-safe rewrites: filler removal, dedupe, whitespace, and output-shaping directive injection. Runs on *every* prompt. Self-reports tokens saved per category. |
| Fingerprint / frequency promoter | `fingerprint.py` | Masks variable slots → skeleton → hash (Drain-style). Counts skeletons; promotes to Tier-1 past a threshold. In-memory or SQLite store. Also yields the cacheable prefix. |
| Foreman category mapping | `categories.py` | `SavingsReport` (tokens saved, by reason) + `NodeReport` (per decomposed subtask spend, by model tier) → one `ForemanRecord` per prompt. Editable pricing table. |

`pipeline.py` wires them together; `demo.py` runs the whole thing.

## Integration

```python
from foreman_optimizer import ForemanOptimizer, SQLiteStore, FrequencyPromoter

opt = ForemanOptimizer(
    promoter=FrequencyPromoter(SQLiteStore("foreman_templates.db"), threshold=5),
    tier1_hook=my_llm_rewrite,   # optional; only fires for hot templates
)

result = opt.optimize(raw_prompt)
model_input = result.optimized_prompt        # -> Foreman deconstruction step
foreman.ingest(result.record.to_dict())      # savings, categorized

# after Foreman deconstructs + routes, feed spend back for full attribution:
opt.attach_node_reports(result, node_reports)
```

## Design invariants

- **One-offs stay Tier-0 forever.** An LLM rewrite of a prompt seen once loses
  money; only skeletons crossing the frequency threshold earn the expensive
  Tier-1 rewrite, amortized across every future hit.
- **Structure is never mutated.** Code/quoted spans are masked before any rule
  runs and restored after — aggressive pruning corrupts structured content
  (e.g. text-to-SQL accuracy collapse), so those spans are `PROTECTED`.
- **Output shaping is booked honestly.** The directive *adds* a few input tokens
  (negative saving) offset by an estimated output reduction on the pricier axis.

## Extension hooks (where the deeper research plugs in)

- `set_token_estimator(fn)` — swap the ~4 char/token default for a real
  tokenizer (code-dense inputs run heavier on current tokenizers).
- `tier1_hook` — wire an LLM optimizer (DSPy/GEPA) or LLMLingua-2 context
  compression here; it fires only on hot templates.
- `SQLiteStore` — point at Foreman's existing db for persistence.
- Skeleton hashing → full **Drain** prefix tree if skeleton cardinality grows.
- A template can later carry a small labeled eval set and graduate from
  heuristic rewriting to true optimization — its quality ceiling jumps.

## Honest limit

Without labeled data, the fidelity guard verifies *"still satisfies the stated
requirements,"* not *"real task quality held."* Keep Tier-1 rewrites extractive
and verify against the Stage-1 requirements until a template earns an eval set.
```
python -m foreman_optimizer.demo
```
