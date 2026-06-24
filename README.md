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

> **The FinOps view that does not yet exist for LLM spend.**  
> See the burn, follow the burn.

An open-source burn map and spend tracker for LLM API costs — built as the free
Bill Analyzer described in the [Foreman](https://github.com/usvsthem-notdev/foreman-revops)
architecture.

All data stays on your machine. No telemetry.

---

## Supported providers

| Provider | Live polling | CSV import | Notes |
|---|---|---|---|
| **Anthropic** | Yes — usage API | Yes | Cache read/write tokens priced separately |
| **OpenAI** | Yes — usage API | Yes | Cached input and reasoning tokens priced separately |
| **Cursor** | Yes — team admin API | — | Requires Team/Business plan + admin key (`crsr_…`) |
| **Gemini** | — | BigQuery export | No usage REST API for AI Studio keys; import via Bill Analyzer |

---

## Features

### Burn Map
Live spend visualization by workload class, provider, and model — split between
**absorbed locally** (sage) and **frontier spend** (clay), matching FIG. 03 of the
Foreman architecture.

- Stacked bar chart: absorbed vs frontier by workload class (extract · rag · reason · agents · coding)
- Daily burn with cumulative overlay
- 30-day spend projection
- Budget progress tracking with configurable alert thresholds

### Live API Polling
Auto-fetch usage data directly from provider APIs on a configurable schedule.

- **Anthropic**: pulls from the Anthropic usage API with date-range pagination
- **OpenAI**: polls the per-day usage endpoint across the lookback window
- **Cursor**: fetches per-event usage from the Cursor team admin API (`POST /teams/filtered-usage-events`)
- **Gemini**: key stored for future integrations; historical data via BigQuery CSV export
- Configurable poll interval (minimum 1 hour) and lookback window (capped at 7 days)
- Poll cursors stored locally — no duplicate inserts on re-poll
- Keys stored in `.env.local` (never committed); encrypted at rest via OS keychain when available

### Bill Analyzer
Upload billing CSVs — parsed entirely in-process, no data leaves your machine.

- Auto-detects provider from file headers
- Handles multiple export formats per provider (Anthropic Console, OpenAI activity + invoice exports)
- Estimates missing cost values using per-provider pricing tables with differential token rates:
  - Anthropic: cache reads at 10% of input price, cache writes at 125%
  - OpenAI: cached input at 50% discount; reasoning tokens at output rate
  - Cursor: Claude-routed calls use Anthropic cache rates; GPT-routed calls use OpenAI rates
  - Gemini: context cache reads at ~25% of input; thinking tokens at a higher output rate
- One-click import to Burn Map

### Spend Intelligence
FIG. 03 loop: **Detect → Propose → Guardrails → Workload Library → Policy Router**

- **Detect**: concentration, drift, reasoning waste, untagged entries
- **Propose**: backtested routing policy proposals with estimated savings
- **Guardrails**: quality floor slider, suggest vs auto-apply mode, rollback notes
- **Workload Library**: class-level routing guidance

### Manual Entry + Data Export
- Add individual API calls with team/feature attribution
- Export all data as CSV or JSON

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

This repo is structured to deploy directly as a Hugging Face Space (Streamlit SDK).
The YAML frontmatter above is read by the Spaces runtime.

**Note:** HuggingFace Spaces has an ephemeral filesystem — data will not persist
between restarts. For persistent storage, run locally or with Docker.

---

## Configuration

Environment variables (or `.env.local` in the project root):

| Variable | Default | Description |
|---|---|---|
| `ANTHROPIC_API_KEY` | — | Anthropic API key (`sk-ant-api03-…`) |
| `OPENAI_API_KEY` | — | OpenAI API key (`sk-…`) |
| `CURSOR_API_KEY` | — | Cursor admin key (`crsr_…`); requires Team/Business plan |
| `GEMINI_API_KEY` | — | Gemini AI Studio key (stored for future use; polling not available) |
| `FOREMAN_POLL_PROVIDERS` | `anthropic,openai` | Comma-separated list of providers to poll |
| `FOREMAN_POLL_INTERVAL_HOURS` | `6` | Hours between polls (minimum 1) |
| `FOREMAN_POLL_LOOKBACK_DAYS` | `2` | Days of history to fetch per poll (maximum 7) |
| `FOREMAN_DB_PATH` | `~/.foreman/foreman.db` | SQLite database path |

---

## Workload classes

| Class | Models typically used | Absorbable? |
|-------|-----------------------|-------------|
| `extract` | haiku, gpt-3.5, gemini-flash | High — structured output, local models match quality |
| `rag` | haiku + embeddings | High — local embedding + small generator works well |
| `reason` | opus, o1, o3, gemini-pro | Partial — planning steps absorbable, final synthesis often needs frontier |
| `agents` | sonnet, gpt-4o | Partial — sub-task planning absorbable locally |
| `coding` | sonnet, gpt-4o, cursor-small | Partial — most code tasks, reserve frontier for hard proofs |

**Sage** = absorbed locally · **Clay** = frontier spend

---

## Security

- All SQL uses parameterized queries (no SQL injection surface)
- File uploads: 50 MB limit, UTF-8 validation, no disk writes
- `FOREMAN_DB_PATH` validated to home dir or `/tmp` (no path traversal)
- Live polling uses an SSRF allowlist — only `api.anthropic.com`, `api.openai.com`, and `api.cursor.com` are reachable; redirects disabled
- API keys validated by format before any network request is made; keys appear only as masked strings in logs
- Docker: non-root user, `no-new-privileges`, read-only root FS

See [SECURITY.md](.github/SECURITY.md) for the full policy and how to report vulnerabilities.

---

## Supported billing export formats

### Anthropic Console
`Billing → Usage → Export CSV`  
Columns: Date, Organization, Project, Model, Input tokens, Output tokens, Cache read tokens, Cache write tokens, Cost (USD)

### OpenAI Platform
`Usage → Export` or `Billing → Download CSV`  
Multiple formats supported — the parser handles column name variations across activity and invoice exports.

### Gemini / Google Cloud
`console.cloud.google.com → Billing → BigQuery export` → download CSV for the desired date range.  
The generic parser handles common BigQuery billing column layouts.

---

## Roadmap

- [ ] Slack / email budget alerts
- [ ] Team-level dashboards with RBAC
- [ ] Golden eval harness for routing policy backtesting
- [ ] Full Gemini BigQuery CSV parser
- [ ] PostgreSQL backend for multi-user deployments

---

## License

Apache-2.0 — see [LICENSE](LICENSE).

Built on the Foreman architecture by Connor Drexler · Brooklyn, NY.
