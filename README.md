---
title: Foreman RevOps Tracker
emoji: 🔥
colorFrom: gray
colorTo: red
sdk: docker
app_port: 7860
pinned: false
license: apache-2.0
short_description: Open-source LLM burn map and spend tracker
---

# Foreman RevOps Tracker

> See the burn, follow the burn.

Open-source LLM spend tracker. Upload billing CSVs or wire up live API keys — the burn map updates either way. All data stays local.

---

## Design principle: dollars per task solved, not price per token

A lower token price can be the more expensive choice: a premium model that solves the task in fewer, less verbose turns often comes out cheaper *per solved task*. Two levers move real cost more than token price does — **prompt caching** (agentic loops re-send a mostly-stable prefix, so hit rates approach 90%; Foreman watches for silently invalidated caches) and **model choice measured by total tokens to solution** (you pay for every turn until the task is done, not for any single call). Every routing proposal ships in suggest mode and must clear a golden-eval benchmarked on cost-per-solved-task against your own workload, caching measured.

---

## What it does

**Executive Brief** — the one-screen answer for a COO/CFO: period spend, daily run-rate with week-over-week delta, 30-day forecast, identified savings (with a monthly equivalent), budget health, and the top three actions ranked by dollar impact — plus a plain-English "bottom line" generated deterministically from the data. No LLM call, free to render, identical on every rerun.

**Burn Map** — stacked bar chart split between frontier spend and locally-absorbed cost. Breaks down by workload class (extract · rag · reason · agents · coding), provider, and model. Shows daily burn with a 30-day projection and budget progress bars, plus an input-vs-output cost split (output is priced several times higher per token, so the token-count split and the dollar split usually don't match).

**Bill Analyzer** — drop in a CSV from Anthropic, OpenAI, Cursor, or Gemini. The parser auto-detects the provider from headers, handles format variants, and estimates how much of the spend could move to a local model.

**Live Polling** — background scheduler fetches usage from Anthropic, OpenAI, and Cursor APIs on a configurable interval (default: every 6 hours). Stores results to the same SQLite DB the burn map reads from.

**Spend Intelligence** — detect → propose → guardrails loop. Flags concentration risk, reasoning token waste, spend drift, low cache hit rates on repeat-heavy workloads, silently degraded caches, batch-eligible spend paying real-time prices, and untagged entries. Generates routing proposals against current (July 2026) model tiers with estimated savings and a quality-floor guardrail note.

**Prompt Optimizer** — paste a prompt you're about to send and get a free, local, deterministic (Tier-0) rewrite before it ever reaches an LLM: filler stripped, duplicate clauses removed, an output-shaping directive added, all priced against Foreman's real per-model rates. Repeated prompt templates are fingerprinted and tracked so recurring ones surface as "hot" — candidates for a heavier Tier-1 rewrite or a cached prefix. Built on the vendored, stdlib-only [`foreman_optimizer`](foreman_optimizer/) package.

**Cursor MCP Server** — exposes 8 analytics tools over stdio so Cursor can query spend data directly from the editor. Every tool call is itself logged: MCP responses get read back into the calling agent's own context as input tokens on its next turn, which is real burn that's invisible to provider billing — the burn map's "MCP Tool-Call Input Burn" section tracks it using the same free token-estimation heuristic as the Prompt Optimizer. See setup below.

---

## Quick start

```bash
git clone https://github.com/usvsthem-notdev/foreman-revops
cd foreman-revops
pip install -r requirements.txt
streamlit run app.py
```

### Docker

```bash
docker compose up
# → http://localhost:8501
```

### Hugging Face Spaces

The YAML frontmatter above deploys this repo as a Streamlit Space.  
Note: Spaces has an ephemeral filesystem — data won't persist between restarts. Run locally or with Docker for persistence.

---

## Live polling setup

Set API keys and start the scheduler alongside the app:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export CURSOR_API_KEY=crsr_...
python -m src.polling.scheduler &
streamlit run app.py
```

| Variable | Default | Notes |
|---|---|---|
| `FOREMAN_POLL_INTERVAL_HOURS` | `6` | Minimum 1 |
| `FOREMAN_POLL_LOOKBACK_DAYS` | `2` | Maximum 7 |
| `FOREMAN_POLL_PROVIDERS` | `anthropic,openai` | Comma-separated |
| `FOREMAN_DB_PATH` | `data/foreman.db` | Must be inside home or `/tmp` |

You can also set keys from the **Live API** tab in the app — they write to `.env.local` and never touch the database.

---

## Cursor MCP integration

Add to `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "foreman": {
      "command": "/path/to/foreman-revops/.venv/bin/python",
      "args": ["/path/to/foreman-revops/mcp_server.py"],
      "env": { "FOREMAN_DB_PATH": "/path/to/foreman-revops/data/foreman.db" }
    }
  }
}
```

Available tools in Cursor (all take a `days` window where relevant, default 30):

| Tool | What it returns |
|---|---|
| `get_key_metrics` | Total cost, token counts, local absorption %, entry count |
| `get_burn_by_provider` | Cost breakdown per provider |
| `get_burn_by_model` | Cost breakdown per model |
| `get_burn_by_class` | Cost breakdown per workload class |
| `get_daily_burn` | Day-by-day spend for the last N days |
| `get_projection` | Spend forecast from recent average |
| `get_budget_status` | Each budget's used/remaining/over-threshold status |
| `get_top_spenders` | Models or teams ranked by cost (`by="model"` or `by="team"`) |

Every call is logged to a local `mcp_tool_calls` table — not provider spend,
but an estimate (via the same free heuristic `foreman_optimizer` uses) of the
input tokens the tool's JSON response adds to the calling agent's next turn.
Set `FOREMAN_MCP_REFERENCE_MODEL` (default `claude-sonnet-4`) to price that
estimate against a different model's input rate.

---

## Supported providers

| Provider | CSV upload | Live polling |
|---|---|---|
| Anthropic | Yes | Yes |
| OpenAI | Yes | Yes |
| Cursor | No | Yes |
| Gemini | Partial (stub) | No — export via BigQuery |

---

## Workload classes

| Class | Typical models | Absorbable? |
|---|---|---|
| `extract` | haiku-4-5, gpt-5.4-nano | High — structured output, local models match quality; batch-eligible |
| `rag` | haiku-4-5 + embeddings | High — local embedding + small generator works well; cache the shared prefix |
| `reason` | opus-4-8, gpt-5.5, gemini-3-pro | Partial — planning steps can move locally, final synthesis often can't |
| `agents` | sonnet-4-6, gpt-5.4 | Partial — sub-task planning is a good local candidate; caching is the biggest lever |
| `coding` | sonnet-4-6, gpt-5.4-mini | Partial — most code tasks; reserve frontier for hard proofs |

---

## Security

- All SQL uses parameterized queries
- File uploads: 50 MB limit, UTF-8 validation, no disk writes
- `FOREMAN_DB_PATH` validated to home dir or `/tmp` — no path traversal
- API keys stored in `.env.local` (0o600), never written to the database
- Outbound network calls: the scheduler and live polling modules make HTTPS requests to provider APIs; the Streamlit app UI makes no outbound calls
- Docker: non-root user, `no-new-privileges`, read-only root FS

See [SECURITY.md](.github/SECURITY.md) for the full policy.

---

## Roadmap

- [ ] Slack / email budget alerts
- [ ] Team-level dashboards with RBAC
- [ ] Golden eval harness for routing policy backtesting
- [ ] Gemini BigQuery CSV parser
- [ ] PostgreSQL backend for multi-user deployments

---

## Acknowledgements

The content policy in `src/analytics/content_policy.py` (ALLOW / REWRITE / REJECT / FLAG verdict pattern) is adapted from [opinionated-systems/markspace](https://github.com/opinionated-systems/markspace). Credit to the markspace contributors for the ContentPolicy abstraction.

---

## License

Apache-2.0 — see [LICENSE](LICENSE).

Built by Connor Drexler · Brooklyn, NY.
