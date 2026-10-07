# ARCHITECTURE

One-day build: a Python backend library (`src/`) + thin FastAPI wrapper (`api/`) + a static single-page dashboard (`web/`). No database, no auth, no cloud. Everything runs locally with one command.

## 1. Analysis pipeline (runtime)
```
Audio + transcript (+ reference audio of same transcript)
 → preprocessing (mono, 16 kHz, trim, peak/loudness sanity)         src/audio.py
 → forced alignment (word start/end)  for reference AND participant  src/align.py
 → acoustic features (F0, energy, rate, pauses, MFCC/spectral)       src/features.py
 → speaker normalization (semitones re speaker median; dB re median) src/features.py
 → reference comparison (same word index ↔ same word)               src/detect.py
 → temporal flaw detection + grounding (start/end, type, severity)   src/detect.py
 → structured evidence (feature, ref value, participant value, Δ)    src/detect.py
 → explanation (taxonomy templates; optional LLM reword)             src/explain.py
 → rubric scoring                                                    src/score.py
 → analysis JSON (validates against analysis.schema.json)            src/pipeline.py
 → dashboard                                                         web/
```

## 2. Dataset pipeline (offline, build-time)
```
good/reference speech (+ transcript, alignment)
 → controlled flaw injection (flaw_types.yaml: method + level value)  src/gen/
 → flawed audio
 → updated alignment/timestamps (recomputed if the timeline changed)  src/gen/
 → ground-truth labels (*.gt.json)                                    src/gen/
 → evaluation (detector output vs ground truth)                       eval/
```

## 3. Key architectural decisions
1. **Same transcript ⇒ word-level comparison.** Reference and participant are both force-aligned; word *i* in one is word *i* in the other. Features are compared over the same word ranges. Audio DTW is **not** the core method (stretch only).
2. **Timeline changes update ground truth.** `pace_fast`, `pace_slow`, `short_pauses`, `long_pauses` change the audio timeline (`changes_timeline: true` in the taxonomy). The generator MUST recompute every word time and every label after such a transform (e.g. an inserted 1.7 s silence shifts all later words by 1.7 s). Labels are always stored on the **flawed** timeline plus the matching reference timeline span.
3. **Speaker-agnostic features.** Pitch = semitones relative to the speaker's own median F0; energy = dB relative to the recording's median speech-frame RMS; rate/pauses are compared as ratios/differences to the reference on the same words. No raw Hz/dBFS comparisons across speakers.
4. **Evidence is computed, explanation is wording.** Detection produces numbers; `explain.py` formats them with the taxonomy templates. An LLM, if enabled, may only reword the text and must keep the numbers; the system runs without it.
5. **Frontend develops against a fixture.** `fixtures/analysis_example.json` is a schema-valid response; the dashboard must work from it with no backend.
6. **Determinism.** Fixed seeds, pinned dependencies, no network at analysis time.

## 3b. Flaw → feature → timeline map (summary of the taxonomy; the YAML is authoritative)
| flaw_type | primary feature | region | changes timeline | rubric component |
|---|---|---|---|---|
| `pace_fast` | `speech_rate_wps` ↑ | words | yes | pacing |
| `pace_slow` | `speech_rate_wps` ↓ | words | yes | pacing |
| `monotone` | `f0_std_st` ↓ | words | no | expressiveness |
| `low_volume` | `rms_db_rel` ↓ | words | no | volume |
| `short_pauses` | `pause_duration_s` ↓ | gap | yes | pausing |
| `long_pauses` | `pause_duration_s` ↑ | gap | yes | pausing |

## 4. Contract relationships
```
                    config/flaw_types.yaml  (what flaws/features/levels/templates exist)
                      │        │         │            │
        defines IDs   │        │         │            └── explanation templates → src/explain.py
                      ▼        ▼         ▼
      src/gen (inject+GT)   src/detect (bounds)   src/score (components, penalties)
                      │        │         │
                      │        ▼         ▼
                      │   analysis JSON  ◄── must validate against docs/contracts/analysis.schema.json
                      │        │                      (enums mirror the taxonomy; checked by scripts/validate_contracts.py)
                      ▼        ▼
     data/**/*.gt.json ─►  eval/ (compare predicted regions vs GT)       web/ (renders analysis JSON; fixture first)
```
- **flaw_types.yaml** = vocabulary + parameters. **analysis.schema.json** = shape of backend output. **Backend** produces schema-valid JSON using taxonomy IDs. **Frontend** only reads that JSON (+ taxonomy display names). **Dataset ground truth** uses the same flaw IDs/levels, so **evaluation** can compare predictions and labels directly.

## 5. Dataset ground-truth record (proposed; T-04 finalizes, must keep these field names)
```
data/
  sources/<speech_id>/{good.wav, transcript.txt, good.alignment.json}
  flawed/<speech_id>/<flaw_id>_L<level>.wav  + .gt.json     (synthetic, single flaw)
  flawed/<speech_id>/mixed_<n>.wav           + .gt.json     (synthetic, multi flaw)
  real/<speech_id>/<speaker>_<n>.wav         + .gt.json     (hand-labeled)
  manifest.csv                                              (one row per audio file)
```
```json
{
  "recording_id": "gettysburg_pace_fast_L3", "speech_id": "gettysburg", "reference_id": "gettysburg_good",
  "kind": "synthetic", "seed": 7, "overall_level": 3,
  "words": [{"index": 0, "text": "Four", "start_s": 0.35, "end_s": 0.62}],
  "regions": [{
    "flaw_type": "pace_fast", "level": 3, "injected_value": 1.32,
    "start_s": 1.13, "end_s": 3.68, "word_start_index": 2, "word_end_index": 10,
    "reference_start_s": 1.13, "reference_end_s": 4.1
  }]
}
```
`kind` ∈ `reference | synthetic | real_human`. `words` is the alignment of the **flawed** audio (same shape as `words.participant` in the analysis schema).

## 6. Module interfaces (proposed; owners may refine, outputs must satisfy the schema)
| Module | Signature (sketch) |
|---|---|
| `src/audio.py` | `load_audio(path, sr=16000) -> (np.ndarray, int)` |
| `src/align.py` | `align(audio, sr, transcript: str) -> list[Word]` (Word = `{index,text,start_s,end_s}`) |
| `src/features.py` | `extract(audio, sr, words) -> FeatureBundle` (frame series + normalization stats); `region_value(bundle, feature_id, start_s, end_s)` |
| `src/detect.py` | `detect(ref: FeatureBundle, par: FeatureBundle) -> list[Region]` (regions incl. evidence, severity, confidence) |
| `src/explain.py` | `explain(region) -> explanation dict` |
| `src/score.py` | `score(regions) -> (overall_score, rubric_scores)` |
| `src/pipeline.py` | `analyze(participant_audio, transcript, reference_audio, reference_transcript=None) -> dict` (schema-valid) |
| `api/` | `POST /analyze` (multipart) → analysis JSON; `GET /fixture` → fixture; `GET /health`; serves `web/` statically |

### Detection sketch (MVP)
Slide a window of ~4–6 words over the transcript; per window compute each taxonomy feature for reference and participant; compute the taxonomy metric (`abs_delta_pct` / `abs_delta`); flag windows ≥ `severity_bounds[0]` in the expected direction; merge adjacent flags of the same flaw type; snap boundaries to word boundaries; severity = `level_for_metric`. Pause flaws are checked per word boundary. Confidence is a documented deterministic function of metric margin over threshold.

## 7. Run modes
- **Fixture mode** (`web/` + static file, or `GET /fixture`): no backend analysis.
- **Live mode:** `api/` calls `src.pipeline.analyze`; `api/` falls back to fixture mode if the pipeline import fails (so the demo never shows a blank page).

## 8. Repository layout
```
AGENTS.md CLAUDE.md GEMINI.md .github/copilot-instructions.md
config/flaw_types.yaml
docs/{SPEC,ARCHITECTURE,TASKS,STATUS,DECISIONS}.md  docs/contracts/analysis.schema.json
fixtures/analysis_example.json
scripts/validate_contracts.py
src/{taxonomy.py, gen/, audio.py, align.py, features.py, detect.py, explain.py, score.py, pipeline.py}   (all but taxonomy.py: not yet implemented)
api/  web/  eval/  tests/  data/
```
