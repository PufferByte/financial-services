# Earnings Reviewer — managed-agent template

## Overview

Earnings call + filings → model update → note draft. Same source as the [`earnings-reviewer`](../../plugins/agent-plugins/earnings-reviewer) Cowork plugin — this directory is the Managed Agent cookbook for `POST /v1/agents`.

**Data source:** free Yahoo Finance data via the bundled `yfinance-earnings-data` skill (runs via `bash`, no MCP connector, no subscription). Originally built around paid FactSet/Daloopa MCP tools; swapped out to remove that cost. See [`yfinance-earnings-data`](../../plugins/vertical-plugins/equity-research/skills/yfinance-earnings-data/SKILL.md) for the free-data trade-offs (delayed/best-effort data, GAAP-vs-street EPS mismatches) and its bundled historical beat/miss backtest (`predict.py`).

## Deploy

```bash
export ANTHROPIC_API_KEY=sk-ant-...
# no FACTSET_MCP_URL / DALOOPA_MCP_URL needed -- only outbound internet access
# for Yahoo Finance (yfinance)
../../scripts/deploy-managed-agent.sh earnings-reviewer
```

## Steering events

See [`steering-examples.json`](./steering-examples.json). Fan out across a coverage list from your orchestration layer — one session per ticker.

## Security & handoffs

Transcripts and press releases are untrusted. Three-tier isolation:

| Tier | Touches untrusted docs? | Tools | Connectors |
|---|---|---|---|
| **`transcript-reader`** | **Yes** | `Read`, `Grep` only | None |
| `model-updater` / Orchestrator | No | `Read`, `Grep`, `Glob`, `Bash`, `Agent` | None (free `yfinance` data via bash, no MCP) |
| **`note-writer`** (Write-holder) | No | `Read`, `Write`, `Edit` | None |

`transcript-reader` returns length-capped, schema-validated JSON. `note-writer` produces `./out/note-<ticker>.docx` and the updated model at `./out/model-<ticker>.xlsx`.

**Handoff:** to rebuild a DCF after an earnings-driven thesis change, emit a `handoff_request` for `model-builder`; `scripts/orchestrate.py` routes it as a new steering event.
