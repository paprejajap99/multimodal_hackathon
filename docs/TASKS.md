# TASKS

Edit only **your task's `Status:` line** here (statuses: `Todo`, `In progress`, `Blocked`, `Review`, `Done`). Append notes in `docs/STATUS.md` (your subsection). Branch per task: `task/T-XX-<slug>`.

**Allocation (intended):** ANTIGRAVITY = T-02, T-03, T-04, T-09, T-10 · CODEX = T-05, T-06, T-07, T-08, T-11 · CHATGPT/CLAUDE WEB = T-01 (done), T-12 drafting, plans/reviews/debugging advice.

**Dependency order:** `T-01 → {T-02, T-04*, T-07, T-10, T-09(fixture mode)} → T-03 → T-05 → {T-06, T-08} → T-11 → T-12`. (*T-04 can start with its own alignment of the good audio before T-02 lands.) Parallel from day start: T-02, T-04, T-07, T-10, T-09, T-06 (template part).

**Shared-file exceptions (why they're not single-owner):** `requirements.txt` (append-only, any task), `docs/STATUS.md` (own subsection), `docs/TASKS.md` (own status line), `docs/DECISIONS.md` (append-only). All other files have exactly one owner.

Common verification: `python scripts/validate_contracts.py` must keep passing in every task.

---
### T-01 Foundation / contracts
- **Owner:** CLAUDE WEB (planning) · **Status:** Done (Phase 0; awaiting human push) · **Depends on:** —
- **Allowed files:** `AGENTS.md CLAUDE.md GEMINI.md .github/ config/ docs/{SPEC,ARCHITECTURE,TASKS,STATUS,DECISIONS}.md docs/contracts/ fixtures/ scripts/validate_contracts.py src/__init__.py src/taxonomy.py tests/test_contracts.py README.md .gitignore`
- **Acceptance:** all contracts exist; fixture validates; flaw IDs consistent everywhere.
- **Verify:** `pip install pyyaml jsonschema pytest && python scripts/validate_contracts.py && pytest tests/test_contracts.py -q`

### T-02 Audio & forced alignment
- **Owner:** ANTIGRAVITY · **Status:** Done · **Depends on:** T-01
- **Allowed files:** `src/audio.py src/align.py tests/test_audio*.py tests/test_align*.py` (+ `requirements.txt` append)
- **Acceptance:** `load_audio` (mono/16 kHz); `align(audio, sr, transcript)` returns Word list (`index,text,start_s,end_s`), monotonic, one entry per transcript word, deterministic; works offline once models are downloaded; documents chosen aligner (e.g. torchaudio MMS forced alignment / WhisperX / MFA — pick the one that installs reliably). Alignment verified by eye on ≥1 real clip (note in STATUS).
- **Verify:** `pytest tests/test_audio*.py tests/test_align*.py -q`

### T-03 Feature extraction & speaker normalization
- **Owner:** ANTIGRAVITY · **Status:** Done · **Depends on:** T-01 (T-02 for real alignments; develop with synthetic alignments)
- **Allowed files:** `src/features.py tests/test_features*.py` (+ `requirements.txt`)
- **Acceptance:** produces for any (audio, words): frame series `f0_st_rel` (semitones re speaker median, null if unvoiced), `rms_db_rel`, windowed `speech_rate_wps`; region stats for `speech_rate_wps`, `f0_std_st`, `rms_db_rel`, `pause_duration_s`; also MFCC + spectrum (FFT) arrays available; exports series in the schema `track` format (`start_s,hop_s,values`); tests on synthetic tones/noise (e.g. gain −6 dB ⇒ rms_db_rel shifts as expected; flat vs varying pitch ⇒ f0_std_st).
- **Verify:** `pytest tests/test_features*.py -q`

### T-04 Dataset & flaw generator (+ ground truth)
- **Owner:** ANTIGRAVITY · **Status:** Todo · **Depends on:** T-01 (baseline alignment from T-02 or own)
- **Allowed files:** `src/gen/ data/ docs/DATASET.md tests/test_gen*.py` (+ `requirements.txt`)
- **Acceptance:** source 2–3 good recordings + transcripts; generator reads `config/flaw_types.yaml` (via `src/taxonomy.py`) and creates 6 flaws × 5 levels per source (+ a few mixed files); **timeline-changing flaws recompute word times and labels**; each file has a `.gt.json` per ARCHITECTURE §5; `manifest.csv`; seeded/reproducible (`python -m src.gen.build_dataset --seed 7`); tests check GT region bounds vs actual audio (e.g. inserted silence length, stretched duration); `docs/DATASET.md` documents sources, licenses, method; 2–3 real human flawed recordings hand-labeled.
- **Verify:** `pytest tests/test_gen*.py -q`

### T-05 Detector + temporal grounding
- **Owner:** CODEX · **Status:** Todo · **Depends on:** T-01, T-03 (T-02 for real runs)
- **Allowed files:** `src/detect.py tests/test_detect*.py`
- **Acceptance:** implements ARCHITECTURE detection sketch; thresholds/bounds only from taxonomy; outputs regions with start/end (participant timeline), reference span, flaw type, severity, confidence, transcript range, **evidence items (primary first)**; direction check against `expected_direction`; deterministic; tests with synthetic feature bundles for all six flaw types incl. no-flaw case.
- **Verify:** `pytest tests/test_detect*.py -q`

### T-06 Explanation
- **Owner:** CODEX · **Status:** Todo · **Depends on:** T-01 (integrates with T-05 output)
- **Allowed files:** `src/explain.py tests/test_explain*.py`
- **Acceptance:** `explain(region)` builds `summary/cause/impact/suggestion` from taxonomy templates + region evidence; numbers in text equal evidence values; works with no LLM/API key; optional LLM rewording behind an env flag, must preserve all numbers (verified by test); `method` field set honestly.
- **Verify:** `pytest tests/test_explain*.py -q`

### T-07 Rubric scoring
- **Owner:** CODEX · **Status:** Todo · **Depends on:** T-01
- **Allowed files:** `src/score.py tests/test_score*.py`
- **Acceptance:** implements `scoring` section of taxonomy exactly; reproduces `overall_score` and `rubric_scores` of `fixtures/analysis_example.json` from its `flaw_regions` (test); clamps 0–100; monotone in severity.
- **Verify:** `pytest tests/test_score*.py -q`

### T-08 Evaluation harness
- **Owner:** CODEX · **Status:** Todo · **Depends on:** T-01, T-04 (GT), T-05 (predictions)
- **Allowed files:** `eval/ tests/test_eval*.py`
- **Acceptance:** CLI `python -m eval.run --data data --out eval/results.json` reports per-flaw/per-level precision, recall (IoU ≥ taxonomy threshold), type accuracy, boundary error, severity MAE, score-vs-level monotonicity; separate real-human section; unit tests on hand-made GT/pred pairs.
- **Verify:** `pytest tests/test_eval*.py -q`

### T-09 Backend API
- **Owner:** ANTIGRAVITY · **Status:** Todo · **Depends on:** T-01 (fixture mode); T-11 for live mode
- **Allowed files:** `api/ tests/test_api*.py` (+ `requirements.txt`)
- **Acceptance:** FastAPI: `GET /health`, `GET /fixture`, `POST /analyze` (multipart: participant audio, transcript text/file, reference id or file); serves `web/` static; imports `src.pipeline.analyze` lazily and **falls back to fixture** (with a `warnings` entry) if unavailable; validates responses against the schema in tests; file-size limits; no auth/DB.
- **Verify:** `pytest tests/test_api*.py -q`; `uvicorn api.main:app --port 8000`

### T-10 Frontend dashboard
- **Owner:** ANTIGRAVITY · **Status:** Todo · **Depends on:** T-01 (fixture); T-09 for live
- **Allowed files:** `web/`
- **Acceptance:** works fully from `fixtures/analysis_example.json` (load via fetch of `/fixture` or bundled copy under `web/`); shows overall + rubric scores; transcript with highlighted flaw regions colored by flaw type (names from taxonomy); reference-vs-participant overlay charts for the three series with region shading; clicking a region shows timestamps, snippet, evidence table, explanation; upload form posts to `/analyze`; handles `null` series values; works offline (vendor chart lib locally or pinned CDN with local fallback). Keep simple (static SPA is fine).
- **Verify:** manual: open the page with fixture; screenshot noted in STATUS.

### T-11 Integration & demo
- **Owner:** CODEX · **Status:** Todo · **Depends on:** T-02, T-03, T-05, T-06, T-07, T-09 (T-10, T-04 for demo data)
- **Allowed files:** `src/pipeline.py scripts/run_* tests/test_pipeline*.py tests/test_e2e*.py`
- **Acceptance:** `analyze()` runs the full chain and its output validates against the schema; end-to-end run on ≥1 synthetic flawed file from T-04 finds the injected regions; `scripts/run_demo.*` starts API + dashboard in one command; bug fixes in other owners' files are done by **handing back to the owner or making a minimal patch recorded in STATUS**.
- **Verify:** `pytest tests/test_pipeline*.py tests/test_e2e*.py -q`

### T-12 Packaging & documentation
- **Owner:** CHATGPT/CLAUDE WEB (drafts) + CODEX (repro check) · **Status:** Todo · **Depends on:** T-11 (drafts can start earlier)
- **Allowed files:** `README.md docs/TECHNICAL_REPORT.md requirements.txt(consolidate) docs/VIDEO_SCRIPT.md Dockerfile(optional)`
- **Acceptance:** README has real setup/run instructions verified on a clean checkout; technical report ≤ 6 pages (dataset, rubric, architecture, scoring); dataset link/location; video script; README claims only implemented features.
- **Verify:** fresh clone → follow README → demo runs.
