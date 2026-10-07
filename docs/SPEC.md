# SPEC — Track C: Contrastive Speech Analytics & Temporal Flaw Grounding

Source of truth for requirements: the official Track C problem statement (Multimodal AI Hackathon 2026). This file restates it in implementation terms. Where we make a choice that the statement does not mandate, it is labelled **[our decision]**.

**Challenge (official):** Develop a speech evaluation system by constructing a custom contrastive dataset of "ideal" vs. "flawed" speeches, to generate reproducible custom evaluative rubric-based scores and actionable delivery feedback via an interactive dashboard.

---
## A. Official / required requirements

### A1. Custom Contrastive Dataset & Stress Testing
- **1a "Good" baseline:** source and transcribe recordings of highly effective public speakers (politicians, champion debaters, TED speakers, ...).
- **1b "Bad" spectrum:** synthesize, self-record, or modify audio of the **exact same transcripts** with intentionally injected delivery flaws, forming a **gradient from egregiously wrong pacing/tone to near-perfect delivery**.
- **1c Labeling:** accurately label and **temporally bound** paired good/bad recordings for training and testing.

### A2. Advanced Feature Extraction & Forced Alignment
- **2a** Perform **forced alignment** mapping the transcript to audio timestamps.
- **2b** Apply feature extraction (e.g. FFT, MFCCs, pitch, F0 contours, speech rate, pause intervals, vocal clarity) to capture the acoustic footprint. Treat speech as a time-series signal.

### A3. Temporal Grounding & Causal Flaw Extraction
- **3a Temporal grounding:** isolate exact **start and end timestamps** of regions where delivery significantly deviates from baseline acoustic expectations.
- **3b Causal explainability:** translate the raw mathematical delta into a structured, human-readable causal explanation for each flaw region.

### A4. Frontend / Visualization & Dashboarding
- Interactive frontend that **accepts audio and transcript uploads**.
- Render a **time-series overlay comparing baseline and participant features**, highlighting flaw regions alongside their causal explanations.

### A5. Expected deliverables
1. GitHub link with all code and documentation.
2. Custom dataset: well-documented paired good/bad audio, transcripts, alignment labels (in GitHub, or a public Google Drive link in the README).
3. Interactive dashboard: functional prototype showing upload, feature extraction processing, visual/textual representation.
4. Technical documentation (max. 6 pages): dataset construction, evaluation rubrics, model architecture, scoring methodology.
5. 3–10 minute YouTube video: dataset collection, stress testing across speech qualities, dashboard catching specific deviations.

### A6. Technical considerations (official)
- **Data quality over quantity:** a cleanly aligned contrastive spectrum beats a large noisy corpus.
- **Speaker agnostic:** normalize pitch and energy for biological differences between speakers.
- Flaw regions need enough **temporal precision** to support causal explanation.
- **Reproducible** across repeated runs and deployment environments.

### A7. Evaluation criteria (official weights)
Causal Explainability & Temporal Grounding 25% · Data Engineering & Stress Testing 30% · Feature Extraction 20% · Visualization & Dashboard 15% · Reproducibility & Code Quality 10%.

---
## B. MVP scope for our one-day implementation

### B1. Principles [our decisions]
- Data quality > size. Controlled synthetic flaws give **exact ground truth**; a few real human flawed recordings add realism/validation.
- Find **exact flaw regions**, not just a whole-speech score.
- Every explanation is tied to **measured acoustic evidence** (numbers come from code, never from an LLM).
- Demo must be reliable: one command to run, works offline, no auth/database/cloud.

### B2. Flaw taxonomy (`config/flaw_types.yaml`, six types, five levels)
`pace_fast`, `pace_slow`, `monotone`, `low_volume`, `short_pauses`, `long_pauses`. Level 1 = very slight (near-perfect delivery) … level 5 = severe. Level 0 = untouched good reference. Levels 2–5 must be detected by the system; level 1 is "bonus" (taxonomy: `evaluation.must_detect_min_level`).

### B3. Dataset target [our decision, adjustable]
- 2–3 source speeches/passages, each ~30–90 s, public-domain or permissively licensed, with transcripts. Fallback if a suitable expert recording is unavailable: carefully self-recorded or high-quality TTS "good" read (documented as such).
- Per source: 1 good reference + 6 flaws × 5 levels single-flaw synthetic files (≈30) + 2–3 mixed multi-flaw demo files.
- 2–3 real human flawed recordings (self-recorded, deliberately flawed) with hand-labeled regions, for validation only.
- Each flawed file ships with ground-truth labels (`*.gt.json`, see ARCHITECTURE §5). Dataset documented in `docs/DATASET.md` (T-04).

### B4. Analysis pipeline
Inputs: participant audio + transcript + a **reference recording of the same transcript** (selected from the dataset library or uploaded). Steps: preprocessing → forced alignment of both recordings → feature extraction (F0 contour, energy, speech rate, pauses; MFCC/spectral features computed and stored but not used for MVP detection) → speaker normalization → **word-index-aligned comparison** against the reference → region detection with severity → structured evidence → template explanation → rubric score → analysis JSON (`docs/contracts/analysis.schema.json`) → dashboard.

### B5. Scoring
Rubric from taxonomy: Pacing (30%), Pausing (25%), Vocal expressiveness (25%), Volume & projection (20%). Deterministic penalties by severity level; overall 0–100. Same input ⇒ same output.

### B6. Dashboard
Upload audio + transcript (+ choose/upload reference) → run analysis → show: overall + rubric scores; participant waveform/transcript with flaw regions highlighted; time-series overlay (reference vs participant) for speech rate, relative pitch, relative loudness; per-region card with timestamps, transcript snippet, evidence table, causal explanation. Must also load `fixtures/analysis_example.json` directly (fixture mode).

### B7. Evaluation harness
Against synthetic ground truth: precision/recall at temporal IoU ≥ 0.5, flaw-type accuracy, severity error (MAE / rank correlation across the gradient), score monotonicity vs severity level, and boundary error (seconds). Real human recordings reported separately.

### B8. Deliverable mapping
Repo + README (T-12), dataset + `docs/DATASET.md` (T-04), dashboard (T-10/T-09/T-11), ≤6-page technical doc `docs/TECHNICAL_REPORT.md` (T-12), video script/shot list (T-12; recording done by humans).

---
## C. Stretch goals / explicitly out of scope
**Out of scope (MVP):** authentication/user accounts, databases, cloud deployment/autoscaling, real-time/streaming analysis, ASR (transcript is provided), reference-free scoring (no reference recording), multilingual support, speaker identification/diarization, scoring of textual *content* quality (we score delivery only), training large models, mobile apps.

**Stretch (only if MVP is done & stable):** extra flaw types (e.g. filler words, mispronunciation), spectral "vocal clarity" metric in detection, LLM *rewording* of template explanations (must keep the numbers unchanged), audio DTW as a secondary check, waveform playback synced to regions, PDF export of the report, Dockerfile / public hosting.

## Known limitations to state honestly in docs
- Loudness is measured relative to the recording's own median, so a flaw spanning the **whole** recording is not detectable by that feature; synthetic `low_volume` flaws cover a span.
- Comparison assumes the participant read the same transcript as the reference; large omissions/insertions degrade word-aligned comparison.
