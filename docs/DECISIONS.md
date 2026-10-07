# DECISIONS (append-only — never edit old entries; add new ones at the end)

Format: `D-NNN | date | status | decision | why | consequence`

**D-001 | 2026-10-07 | Accepted | Shared flaw taxonomy in `config/flaw_types.yaml`.** One source for flaw IDs, features, levels, injection parameters, detection bounds, templates, rubric. *Why:* generator, detector, evaluator, explainer, scorer, frontend must not drift. *Consequence:* no hard-coded second taxonomy anywhere; changes need a decision entry.

**D-002 | 2026-10-07 | Accepted | Shared analysis schema `docs/contracts/analysis.schema.json`.** *Why:* backend and frontend decouple. *Consequence:* backend output must validate; schema enums mirror the taxonomy (checked by `scripts/validate_contracts.py`).

**D-003 | 2026-10-07 | Accepted | Fixture-first frontend.** `fixtures/analysis_example.json` lets the dashboard be built before the backend exists. *Consequence:* fixture must stay schema-valid; contract changes update it.

**D-004 | 2026-10-07 | Accepted | Clear directory/task ownership.** One owner per file, with small named exceptions (STATUS own-section, TASKS own-line, DECISIONS append-only, requirements append-only). *Why:* parallel agents without merge conflicts.

**D-005 | 2026-10-07 | Accepted | Deterministic acoustic evidence before optional LLM wording.** Numbers/timestamps/severity/scores come from code; an LLM may only reword. System works without any LLM.

**D-006 | 2026-10-07 | Accepted | Small, high-quality contrastive dataset over a large one.** Matches the official "quality over quantity" consideration and the 30% data-engineering weight.

**D-007 | 2026-10-07 | Accepted | Synthetic controlled flaws for exact ground truth + a few real human flawed recordings for realism/validation.**

**D-008 | 2026-10-07 | Accepted | Transcript/word-index alignment is the primary comparison mechanism; audio DTW is not core.** Timeline-changing flaws must update ground-truth timestamps.

**D-009 | 2026-10-07 | Accepted | Speaker normalization:** pitch in semitones re speaker median; loudness in dB re recording median; rate/pauses as deltas vs reference on the same words.

**D-010 | 2026-10-07 | Accepted | No authentication, database, or cloud complexity.** Local FastAPI + static dashboard; offline-capable demo.

**D-011 | 2026-10-07 | Accepted | Git-based handoff between coding agents.** Task branches, merge (not rebase), no force pushes, STATUS/TASKS updated and pushed at the end of every session (see AGENTS.md §5).

**D-012 | 2026-10-07 | Proposed (needs human approval) | Baseline for arbitrary uploads = a reference recording of the same transcript** (picked from the dataset library or uploaded); reference-free scoring is out of scope. 

**D-013 | 2026-10-07 | Accepted | Level 1 flaws are "bonus"** (not required to be detected); levels 2–5 must be (`evaluation.must_detect_min_level`).
