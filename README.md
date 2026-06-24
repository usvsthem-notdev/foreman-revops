---
title: Foreman RevOps Tracker
emoji: 🔥
colorFrom: gray
colorTo: orange
sdk: streamlit
sdk_version: 1.45.1
app_file: app.py
pinned: false
license: apache-2.0
short_description: Open-source LLM burn map and spend tracker
---

# Foreman RevOps Tracker

> See the burn, follow the burn.

Open-source LLM spend tracker. Upload billing CSVs or wire up live API keys — the burn map updates either way. All data stays local.

---

## What it does

**Burn Map** — stacked bar chart split between frontier spend and locally-absorbed cost. Breaks down by workload class (extract · rag · reason · agents · coding), provider, and model. Shows daily burn with a 30-day projection and budget progress bars.

**Bill Analyzer** — drop in a CSV from Anthropic, OpenAI, Cursor, or Gemini. The parser auto-detects the provider from headers, handles format variants, and estimates how much of the spend could move to a local model.

**Live Polling** — background scheduler fetches usage from Anthropic, OpenAI, and Cursor APIs on a configurable interval (default: every 6 hours). Stores results to the same SQLite DB the burn map reads from.

**Spend Intelligence** — detect → propose → guardrails loop. Flags concentration risk, reasoning token waste, spend drift, and untagged entries. Generates routing proposals with estimated savings and a quality-floor guardrail note.

**Cursor MCP Server** — exposes 8 analytics tools over stdio so Cursor can query spend data directly from the editor. See setup below.

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

Available tools in Cursor:

| Tool | What it returns |
|---|---|
| `get_key_metrics` | Total cost, token counts, local absorption %, entry count |
| `get_burn_by_provider` | Cost breakdown per provider |
| `get_burn_by_model` | Cost breakdown per model |
| `get_burn_by_class` | Cost breakdown per workload class |
| `get_daily_burn` | Day-by-day spend for the last N days |
| `get_projection` | 30-day spend forecast from recent average |
| `get_budget_status` | Each budget's used/remaining/over-threshold status |
| `get_top_spenders` | Models or teams ranked by cost |

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
| `extract` | haiku, gpt-3.5 | High — structured output, local models match quality |
| `rag` | haiku + embeddings | High — local embedding + small generator works well |
| `reason` | opus, o1, o3 | Partial — planning steps can move locally, final synthesis often can't |
| `agents` | sonnet, gpt-4o | Partial — sub-task planning is a good local candidate |
| `coding` | sonnet, gpt-4o | Partial — most code tasks; reserve frontier for hard proofs |

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

## License

Apache-2.0 — see [LICENSE](LICENSE).

Built by Connor Drexler · Brooklyn, NY.
