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

All data stays on your machine. No telemetry. No external calls.

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

### Bill Analyzer
Upload billing CSVs from Anthropic or OpenAI — parsed entirely in-process.

- Auto-detects provider from file headers
- Handles multiple export formats per provider
- Estimates "absorbable spend" — workloads that could run on a local model
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

## Workload classes

| Class | Models typically used | Absorbable? |
|-------|-----------------------|-------------|
| `extract` | haiku, gpt-3.5 | High — structured output, local models match quality |
| `rag` | haiku + embeddings | High — local embedding + small generator works well |
| `reason` | opus, o1, o3 | Partial — planning steps absorbable, final synthesis often needs frontier |
| `agents` | sonnet, gpt-4o | Partial — sub-task planning absorbable locally |
| `coding` | sonnet, gpt-4o | Partial — most code tasks, reserve frontier for hard proofs |

**Sage** = absorbed locally · **Clay** = frontier spend

---

## Security

- All SQL uses parameterized queries (no SQL injection surface)
- File uploads: 50 MB limit, UTF-8 validation, no disk writes
- `FOREMAN_DB_PATH` validated to home dir or `/tmp` (no path traversal)
- Docker: non-root user, `no-new-privileges`, read-only root FS
- No outbound network calls from the app

See [SECURITY.md](.github/SECURITY.md) for the full policy and how to report vulnerabilities.

---

## Supported billing export formats

### Anthropic Console
`Billing → Usage → Export CSV`  
Columns: Date, Organization, Project, Model, Input tokens, Output tokens, Cache read tokens, Cache write tokens, Cost (USD)

### OpenAI Platform
`Usage → Export` or `Billing → Download CSV`  
Multiple formats supported — the parser handles column name variations.

---

## Roadmap

- [ ] Live API polling (Anthropic / OpenAI usage APIs)
- [ ] Slack / email budget alerts
- [ ] Team-level dashboards with RBAC
- [ ] Golden eval harness for routing policy backtesting
- [ ] Google Cloud / Vertex billing support
- [ ] PostgreSQL backend for multi-user deployments

---

## License

Apache-2.0 — see [LICENSE](LICENSE).

Built on the Foreman architecture by Connor Drexler · Brooklyn, NY.
