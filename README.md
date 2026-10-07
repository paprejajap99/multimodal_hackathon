# Track C — Contrastive Speech Analytics & Temporal Flaw Grounding

Multimodal AI Hackathon 2026 · Track C

## Problem
Judging spoken performance is subjective and feedback is vague. There is no standard dataset that pairs good rhetorical delivery with a *spectrum* of flawed deliveries of the exact same text.

## Proposed solution
1. A small, cleanly aligned **contrastive dataset**: a good recording of a speech plus the same text with controlled delivery flaws (6 flaw types × 5 severity levels) and exact timestamp labels, plus a few real human flawed recordings.
2. An analysis pipeline: **forced alignment → acoustic features → speaker normalization → word-aligned comparison to the good reference → temporal flaw detection → evidence-backed explanation → reproducible rubric score**.
3. An **interactive dashboard** showing reference-vs-participant time-series with flaw regions and causal explanations.

Flaw types: `pace_fast`, `pace_slow`, `monotone`, `low_volume`, `short_pauses`, `long_pauses` (see `config/flaw_types.yaml`).

## Architecture (high level)
`audio + transcript → preprocessing → forced alignment → features → normalization → reference comparison → flaw detection → evidence → explanation → rubric score → analysis JSON → dashboard`. Details: `docs/ARCHITECTURE.md`.

## Current status
**Phase 0 complete (specs and contracts only).** The audio-analysis system, dataset, API, dashboard and evaluation are **not implemented yet**. What exists: agent instructions, flaw taxonomy, analysis JSON schema, a schema-valid example fixture, specs/task board, and a contract validator. See `docs/STATUS.md`.

## Repository structure
```
AGENTS.md  docs/{SPEC,ARCHITECTURE,TASKS,STATUS,DECISIONS}.md
config/flaw_types.yaml   docs/contracts/analysis.schema.json   fixtures/analysis_example.json
scripts/validate_contracts.py   src/taxonomy.py
src/gen/ api/ web/ eval/ tests/ data/        (planned; currently empty placeholders)
```
Requirements and scope: `docs/SPEC.md`.

## Setup / run (future)
_Not available yet._ The only runnable check today:
```bash
pip install pyyaml jsonschema pytest
python scripts/validate_contracts.py
pytest tests/test_contracts.py -q
```
Install/run instructions for the pipeline, API and dashboard will be added by T-12 once implemented.

## Contributing (agents)
Read `AGENTS.md` first.
