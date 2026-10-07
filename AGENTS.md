# AGENTS.md — Track C: Contrastive Speech Analytics & Temporal Flaw Grounding

This file is **authoritative** for every coding agent (Antigravity, Codex, Claude, Gemini, Copilot, ...).

## 0. Read order (always)
1. `AGENTS.md` (this file)
2. `docs/STATUS.md` — current state + handoff notes
3. `docs/TASKS.md` — find **your assigned task**; read its allowed files + acceptance criteria
4. Contracts you touch: `config/flaw_types.yaml`, `docs/contracts/analysis.schema.json`, `fixtures/analysis_example.json`
5. As needed: `docs/SPEC.md`, `docs/ARCHITECTURE.md`, `docs/DECISIONS.md`

## 1. Project in five lines
Hackathon (one day). Given a participant speech recording + transcript + a good reference recording of the same text, find **where** delivery deviates (start/end timestamps), **what** the flaw is (taxonomy), **how bad** (severity 1–5), explain it with **measured acoustic evidence**, score it with a **reproducible rubric**, and show it in a dashboard. Official problem statement terminology is preserved in `docs/SPEC.md`.

## 2. Non-negotiable rules
- Follow the shared contracts. **Flaw IDs, feature IDs, rubric component IDs, and templates come only from `config/flaw_types.yaml`.** Never hard-code a second taxonomy.
- Backend output must validate against `docs/contracts/analysis.schema.json`. The frontend consumes only that shape (develop against `fixtures/analysis_example.json`).
- **Do not casually redesign architecture.** If a contract must change: stop, add a proposal to `docs/DECISIONS.md` (append-only), update schema + taxonomy + fixture together, run `python scripts/validate_contracts.py`, and get human approval (record it in the entry).
- **Respect file ownership** (section 4). Do not edit files outside your task's allowed list unless strictly necessary; if you must, make the smallest possible change and say so in your report.
- Keep it simple. One-day hackathon: no auth, no database, no cloud infra, no new heavy dependencies without need. Prefer a working, reproducible demo over elegance.
- **Never commit secrets/API keys/tokens.** Use environment variables; `.env` is git-ignored.
- **Prefer deterministic mathematical/acoustic evidence over LLM judgment.** All numbers (deltas, timestamps, severities, scores) come from the analysis code. If an LLM is used at all, it may only *reword* an explanation built from the computed evidence; it must never invent measurements. The system must work with no LLM/API key.
- Be reproducible: fix random seeds, pin versions you add to `requirements.txt` (append-only, one dependency per line with a `# T-xx` comment), no network calls at analysis time.
- Add/update tests for meaningful implementation work and run them before declaring done.
- Do not claim something is implemented/tested unless you ran it.

## 3. Shared contracts (single sources of truth)
| Contract | Purpose |
|---|---|
| `config/flaw_types.yaml` | Flaw taxonomy, feature registry, severity levels, injection parameters, ground-truth rules, detection bounds, explanation templates, rubric scoring. Used by generator, detector, evaluator, explainer, scorer, frontend. |
| `docs/contracts/analysis.schema.json` | Shape of the backend analysis output. |
| `fixtures/analysis_example.json` | Valid example; frontend develops against it before backend exists. |
| `src/taxonomy.py` | Tiny loader/helper for the taxonomy. Use it; don't re-parse YAML ad hoc. |

Check contracts any time with: `python scripts/validate_contracts.py` (needs `pyyaml jsonschema`).

## 4. Ownership map
| Path | Owner task | Notes |
|---|---|---|
| `src/gen/`, `data/`, `docs/DATASET.md`, `tests/test_gen*.py` | T-04 | dataset + flaw injection + ground truth |
| `src/audio.py`, `src/align.py`, `tests/test_align*.py`, `tests/test_audio*.py` | T-02 | loading/preprocessing + forced alignment |
| `src/features.py`, `tests/test_features*.py` | T-03 | acoustic features + speaker normalization |
| `src/detect.py`, `tests/test_detect*.py` | T-05 | detection + temporal grounding |
| `src/explain.py`, `tests/test_explain*.py` | T-06 | explanations (templates; optional LLM rewording) |
| `src/score.py`, `tests/test_score*.py` | T-07 | rubric scoring |
| `eval/`, `tests/test_eval*.py` | T-08 | evaluation harness |
| `api/`, `tests/test_api*.py` | T-09 | backend API |
| `web/` | T-10 | frontend |
| `src/pipeline.py`, `scripts/run_*`, `tests/test_pipeline*.py`, `tests/test_e2e*.py` | T-11 | integration/demo |
| `README.md` (after Phase 0), `docs/TECHNICAL_REPORT.md`, packaging files | T-12 | docs/packaging |
| `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `.github/`, `config/`, `docs/contracts/`, `fixtures/`, `docs/SPEC.md`, `docs/ARCHITECTURE.md`, `scripts/validate_contracts.py`, `src/taxonomy.py`, `src/__init__.py`, `tests/test_contracts.py` | T-01 (contracts) | change only via the contract-change rule in section 2 |

Shared **append-only / own-section-only** files (exceptions to single ownership):
- `docs/STATUS.md` — edit **only your task's subsection** (+ the "Last updated" line).
- `docs/TASKS.md` — edit **only your task's `Status:` line**.
- `docs/DECISIONS.md` — append new entries at the end; never edit old ones.
- `requirements.txt` — append your dependency lines only.

## 5. Git / GitHub operating rules (agents are responsible for sync)
Several agents share one public repo. The base branch is `main`. One branch per task: `task/T-XX-<slug>` (e.g. `task/T-02-alignment`). Commit messages: `T-02: <what changed>`.

### BEFORE WORK
```bash
git status --short --branch          # 1. must understand current state
git fetch origin --prune             # 2. get latest remote state
git checkout main && git pull --ff-only origin main      # 3. update base
git checkout -b task/T-XX-<slug>     # 4. new task branch ...
#    ... or, if the branch already exists on origin:
#    git checkout task/T-XX-<slug> && git pull --ff-only origin task/T-XX-<slug> && git merge origin/main
git branch --show-current            # confirm branch == your task; re-read your TASKS.md entry
```
If the working tree has uncommitted changes you did not make: **stop and report**; do not stash, reset, or discard them.

### DURING WORK
5. Commit small, logical checkpoints (`git add <specific paths>`; avoid blind `git add -A`; check `git status` for out-of-scope files first).
6. Never overwrite another agent's work. Merge conflicts: keep both sides' intent; never resolve wholesale with `-X ours/theirs` in files you don't own. A conflict in a contract file → stop and report.
7. **Forbidden unless the human explicitly instructs:** `git push --force` / `--force-with-lease`, `git reset --hard`, `git clean -fd`, `git checkout -- .`, `git restore .` on others' changes, `git rebase` of pushed branches, `git branch -D`, history rewriting, deleting others' files/branches.
   Sync using **merge** (`git merge origin/main`), never rebase of pushed commits.

### BEFORE STOPPING (every session, even if unfinished)
8. Run relevant tests/checks (`python scripts/validate_contracts.py`, your task's `pytest` command).
9. Commit all intended changes.
10. Sync + push:
```bash
git fetch origin
git merge origin/main                 # resolve conflicts if any, re-run tests
git push -u origin task/T-XX-<slug>   # if rejected: fetch, merge, retry. Never force.
```
   If (and only if) the task is **complete and tests pass**, integrate to `main`:
```bash
git checkout main && git pull --ff-only origin main
git merge --no-ff task/T-XX-<slug> -m "Merge T-XX: <title>"
<re-run tests>
git push origin main                  # if rejected: git pull --no-rebase origin main, re-test, push (max 3 tries, then report)
```
11. Update `docs/STATUS.md` (your subsection) and your `TASKS.md` status line, commit, and push (to the branch, and to `main` if you merged). If push is impossible (auth/network), say so plainly — never claim a push that did not happen.
12. Final report to the human must include: **final commit hash** (`git rev-parse --short HEAD`), **branch** (`git branch --show-current`), **push status** (output of `git status -sb` after push; merged to main? yes/no), **tests run + results**, **files changed outside scope (if any)**, **next step**.

## 6. Testing & definition of done
A task is done when: acceptance criteria in `docs/TASKS.md` are met; its tests pass; `python scripts/validate_contracts.py` passes; outputs (if any) validate against the schema; STATUS/TASKS updated; work is pushed. Use `pytest -q`. Tests must not need network or GPU unless the task says so; use tiny synthetic audio for unit tests.

## 7. Style
Python 3.10+, type hints on public functions, small functions, docstring stating inputs/outputs/units (seconds, dB, semitones). Time is always **seconds (float)**; word indices are **0-based** and refer to the transcript word list. Pitch/energy comparisons are always in speaker-normalized terms (semitones relative to speaker median; dB relative to recording median).
