"""tests/fixtures/make_fixtures.py — generate synthetic test audio fixtures.

Run this script to regenerate the fixture WAV files:
    python tests/fixtures/make_fixtures.py

The output files are committed to git (they are tiny: < 100 KB each).
They contain synthetic pure-tone segments separated by silence — not real speech.
They exist solely to supply valid audio to unit tests without requiring network
access, GPU, or any real audio dataset.
"""

from __future__ import annotations

import struct
import wave
from pathlib import Path

import numpy as np

FIXTURES_DIR = Path(__file__).parent

# ---------------------------------------------------------------------------
# Fixture 1: sample_8words.wav
# ---------------------------------------------------------------------------
# 8 "words" modelled as 440 Hz sine tone segments separated by short silences.
# Word durations and transcript designed to match the RuledAligner mock contract.
#
# Layout (all times in seconds, SR=16000):
#   0.00–0.05  leading silence
#   0.05–0.35  word 0 "Four"      (0.30 s)
#   0.35–0.45  inter-word gap
#   0.45–0.75  word 1 "score"     (0.30 s)
#   0.75–0.85  inter-word gap
#   0.85–1.15  word 2 "and"       (0.30 s)
#   1.15–1.25  inter-word gap
#   1.25–1.55  word 3 "seven"     (0.30 s)
#   1.55–1.65  inter-word gap
#   1.65–1.95  word 4 "years"     (0.30 s)
#   1.95–2.05  inter-word gap
#   2.05–2.35  word 5 "ago"       (0.30 s)
#   2.35–2.45  inter-word gap
#   2.45–2.75  word 6 "our"       (0.30 s)
#   2.75–2.85  inter-word gap
#   2.85–3.15  word 7 "fathers"   (0.30 s)
#   3.15–3.20  trailing silence
# Total duration: 3.20 s
SAMPLE_TRANSCRIPT_8 = "Four score and seven years ago our fathers"

SR = 16_000
TONE_HZ = 440.0
WORD_DUR_S = 0.30
GAP_DUR_S = 0.10
LEADING_S = 0.05
TRAILING_S = 0.05


def make_tone(duration_s: float, freq_hz: float, sr: int) -> np.ndarray:
    """Generate a sine-wave tone segment, float32, amplitude 0.5."""
    t = np.linspace(0.0, duration_s, int(duration_s * sr), endpoint=False)
    return (0.5 * np.sin(2 * np.pi * freq_hz * t)).astype(np.float32)


def make_silence(duration_s: float, sr: int) -> np.ndarray:
    return np.zeros(int(duration_s * sr), dtype=np.float32)


def float32_to_int16(data: np.ndarray) -> np.ndarray:
    """Convert float32 [-1, 1] to int16."""
    clamped = np.clip(data, -1.0, 1.0)
    return (clamped * 32767).astype(np.int16)


def build_sample_8words() -> np.ndarray:
    """Build 8 tone-segments interleaved with silences."""
    segments: list[np.ndarray] = [make_silence(LEADING_S, SR)]
    n_words = len(SAMPLE_TRANSCRIPT_8.split())
    for i in range(n_words):
        segments.append(make_tone(WORD_DUR_S, TONE_HZ, SR))
        if i < n_words - 1:
            segments.append(make_silence(GAP_DUR_S, SR))
    segments.append(make_silence(TRAILING_S, SR))
    return np.concatenate(segments)


def write_wav(path: Path, data: np.ndarray, sr: int) -> None:
    """Write float32 data as 16-bit PCM WAV."""
    data_int16 = float32_to_int16(data)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sr)
        wf.writeframes(data_int16.tobytes())


# ---------------------------------------------------------------------------
# Fixture 2: sample_stereo.wav — stereo version to test mono downmix
# ---------------------------------------------------------------------------

def build_sample_stereo(mono: np.ndarray) -> np.ndarray:
    """Left channel = mono, right channel = 0.5 * mono → mean = 0.75 * mono."""
    left = mono
    right = 0.5 * mono
    # interleaved stereo for WAV
    stereo = np.stack([left, right], axis=1)  # (N, 2)
    return stereo


def write_stereo_wav(path: Path, data: np.ndarray, sr: int) -> None:
    """Write 2-channel float32 data as 16-bit PCM WAV."""
    int16_data = float32_to_int16(data.flatten())
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(sr)
        wf.writeframes(int16_data.tobytes())


# ---------------------------------------------------------------------------
# Fixture 3: sample_8khz.wav — 8 kHz file to test resampling
# ---------------------------------------------------------------------------

def build_sample_8khz() -> np.ndarray:
    """Build a short mono tone at 8 kHz sample rate."""
    sr8 = 8_000
    t = np.linspace(0.0, 1.0, sr8, endpoint=False)
    return (0.5 * np.sin(2 * np.pi * 440.0 * t)).astype(np.float32)


def write_8khz_wav(path: Path) -> None:
    sr8 = 8_000
    data = build_sample_8khz()
    data_int16 = float32_to_int16(data)
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sr8)
        wf.writeframes(data_int16.tobytes())


# ---------------------------------------------------------------------------
# Fixture 4: empty.wav — zero-sample file to test AudioLoadError
# ---------------------------------------------------------------------------

def write_empty_wav(path: Path) -> None:
    with wave.open(str(path), "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(b"")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    FIXTURES_DIR.mkdir(parents=True, exist_ok=True)

    # 1. 8-word mono WAV
    mono_data = build_sample_8words()
    wav_path = FIXTURES_DIR / "sample_8words.wav"
    write_wav(wav_path, mono_data, SR)
    duration = len(mono_data) / SR
    print(f"✓ {wav_path.name}  {len(mono_data)} samples  {duration:.2f}s  {wav_path.stat().st_size} bytes")

    # 2. Stereo WAV
    stereo_data = build_sample_stereo(mono_data)
    stereo_path = FIXTURES_DIR / "sample_stereo.wav"
    write_stereo_wav(stereo_path, stereo_data, SR)
    print(f"✓ {stereo_path.name}  {stereo_path.stat().st_size} bytes")

    # 3. 8 kHz WAV
    wav8k_path = FIXTURES_DIR / "sample_8khz.wav"
    write_8khz_wav(wav8k_path)
    print(f"✓ {wav8k_path.name}  {wav8k_path.stat().st_size} bytes")

    # 4. Empty WAV
    empty_path = FIXTURES_DIR / "sample_empty.wav"
    write_empty_wav(empty_path)
    print(f"✓ {empty_path.name}  {empty_path.stat().st_size} bytes")

    # 5. Transcript text file
    transcript_path = FIXTURES_DIR / "sample_transcript.txt"
    transcript_path.write_text(SAMPLE_TRANSCRIPT_8, encoding="utf-8")
    print(f"✓ {transcript_path.name}")

    print("\nAll fixtures generated successfully.")


if __name__ == "__main__":
    main()
