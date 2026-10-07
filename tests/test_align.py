"""tests/test_align.py — Unit tests for src/align.py

All tests use the MockAligner backend (backend='mock') for full offline/no-GPU
operation. The MockAligner is a test stub; these tests verify the alignment
*contract* (types, count, monotonicity, indices) — not acoustic accuracy.

Additional tests verify WhisperXAligner raises AlignerNotAvailableError when
whisperx is not installed, and that backend='auto' silently falls back to mock.

Owner: T-02
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.align import (
    AlignmentError,
    AlignerNotAvailableError,
    MockAligner,
    WhisperXAligner,
    Word,
    align,
    get_aligner,
)
from src.audio import load_audio

FIXTURES = Path(__file__).parent / "fixtures"
MONO_WAV = FIXTURES / "sample_8words.wav"
TRANSCRIPT_8 = "Four score and seven years ago our fathers"
TOKENS_8 = TRANSCRIPT_8.split()

SR = 16_000


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_fixture() -> tuple[np.ndarray, int]:
    return load_audio(MONO_WAV, sr=SR)


def _make_short_audio(duration_s: float = 0.5, sr: int = SR) -> np.ndarray:
    """Synthetic silence array of given duration."""
    return np.zeros(int(duration_s * sr), dtype=np.float32)


# ---------------------------------------------------------------------------
# Contract: output structure
# ---------------------------------------------------------------------------


def test_align_returns_list():
    audio, sr = _load_fixture()
    result = align(audio, sr, TRANSCRIPT_8, backend="mock")
    assert isinstance(result, list)


def test_align_word_count_matches_transcript():
    audio, sr = _load_fixture()
    words = align(audio, sr, TRANSCRIPT_8, backend="mock")
    assert len(words) == len(TOKENS_8), (
        f"Expected {len(TOKENS_8)} words, got {len(words)}"
    )


def test_align_texts_match_transcript_tokens():
    audio, sr = _load_fixture()
    words = align(audio, sr, TRANSCRIPT_8, backend="mock")
    for i, (word, token) in enumerate(zip(words, TOKENS_8)):
        assert word["text"] == token, (
            f"words[{i}].text={word['text']!r} != token={token!r}"
        )


def test_align_indices_are_sequential_zero_based():
    audio, sr = _load_fixture()
    words = align(audio, sr, TRANSCRIPT_8, backend="mock")
    for i, word in enumerate(words):
        assert word["index"] == i, f"words[{i}].index={word['index']}, expected {i}"


def test_align_times_are_positive():
    audio, sr = _load_fixture()
    words = align(audio, sr, TRANSCRIPT_8, backend="mock")
    for w in words:
        assert w["start_s"] >= 0.0, f"start_s < 0: {w}"
        assert w["end_s"] > 0.0, f"end_s <= 0: {w}"


def test_align_end_after_start():
    audio, sr = _load_fixture()
    words = align(audio, sr, TRANSCRIPT_8, backend="mock")
    for w in words:
        assert w["end_s"] > w["start_s"], (
            f"word '{w['text']}': end_s {w['end_s']} not > start_s {w['start_s']}"
        )


# ---------------------------------------------------------------------------
# Contract: monotonicity
# ---------------------------------------------------------------------------


def test_align_monotonic():
    """words[i].end_s <= words[i+1].start_s for all i."""
    audio, sr = _load_fixture()
    words = align(audio, sr, TRANSCRIPT_8, backend="mock")
    for i in range(len(words) - 1):
        assert words[i]["end_s"] <= words[i + 1]["start_s"] + 1e-6, (
            f"Monotonicity violated at i={i}: "
            f"words[{i}].end_s={words[i]['end_s']:.4f} > "
            f"words[{i+1}].start_s={words[i+1]['start_s']:.4f}"
        )


def test_align_within_audio_duration():
    """Last word must end at or before audio duration + small tolerance."""
    audio, sr = _load_fixture()
    words = align(audio, sr, TRANSCRIPT_8, backend="mock")
    audio_duration = len(audio) / sr
    epsilon = 0.02  # 20 ms tolerance for rounding
    assert words[-1]["end_s"] <= audio_duration + epsilon, (
        f"Last word end_s {words[-1]['end_s']:.4f} > audio duration {audio_duration:.4f}"
    )


# ---------------------------------------------------------------------------
# Contract: TypedDict keys
# ---------------------------------------------------------------------------


def test_align_word_has_required_keys():
    audio, sr = _load_fixture()
    words = align(audio, sr, TRANSCRIPT_8, backend="mock")
    required = {"index", "text", "start_s", "end_s"}
    for w in words:
        assert required.issubset(w.keys()), (
            f"Word missing keys: {required - w.keys()}, word={w}"
        )


def test_align_word_types():
    audio, sr = _load_fixture()
    words = align(audio, sr, TRANSCRIPT_8, backend="mock")
    for w in words:
        assert isinstance(w["index"], int)
        assert isinstance(w["text"], str)
        assert isinstance(w["start_s"], float)
        assert isinstance(w["end_s"], float)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


def test_align_empty_transcript_raises():
    audio, sr = _load_fixture()
    with pytest.raises(AlignmentError, match="empty"):
        align(audio, sr, "", backend="mock")


def test_align_whitespace_only_transcript_raises():
    audio, sr = _load_fixture()
    with pytest.raises(AlignmentError, match="empty"):
        align(audio, sr, "   \t\n  ", backend="mock")


def test_align_zero_length_audio_raises():
    audio = np.zeros(0, dtype=np.float32)
    with pytest.raises(AlignmentError):
        align(audio, SR, TRANSCRIPT_8, backend="mock")


def test_align_too_short_audio_raises():
    """Audio of 0.01 s is too short to align any words."""
    audio = _make_short_audio(duration_s=0.01)
    with pytest.raises(AlignmentError):
        align(audio, SR, TRANSCRIPT_8, backend="mock")


def test_align_2d_audio_raises():
    audio_2d = np.zeros((2, 1000), dtype=np.float32)
    with pytest.raises(AlignmentError):
        align(audio_2d, SR, TRANSCRIPT_8, backend="mock")


def test_align_unknown_backend_raises():
    audio, sr = _load_fixture()
    with pytest.raises(ValueError, match="Unknown alignment backend"):
        align(audio, sr, TRANSCRIPT_8, backend="not_a_backend")


# ---------------------------------------------------------------------------
# Mock aligner specifically
# ---------------------------------------------------------------------------


def test_mock_aligner_works_with_single_word():
    audio, sr = _load_fixture()
    words = align(audio, sr, "hello", backend="mock")
    assert len(words) == 1
    assert words[0]["text"] == "hello"
    assert words[0]["index"] == 0


def test_mock_aligner_longer_words_get_longer_duration():
    """Words with more characters should get longer duration (proportional model)."""
    audio, sr = _load_fixture()
    # "fathers" (7 chars) vs "a" (1 char)
    words = align(audio, sr, "a fathers", backend="mock")
    assert words[1]["end_s"] - words[1]["start_s"] > words[0]["end_s"] - words[0]["start_s"]


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_align_deterministic_mock():
    """MockAligner: same inputs → identical output on repeated calls."""
    audio, sr = _load_fixture()
    words1 = align(audio, sr, TRANSCRIPT_8, backend="mock")
    words2 = align(audio, sr, TRANSCRIPT_8, backend="mock")
    assert words1 == words2, "MockAligner is not deterministic"


# ---------------------------------------------------------------------------
# WhisperX not available path
# ---------------------------------------------------------------------------


def test_whisperx_aligner_raises_when_not_installed():
    """WhisperXAligner must raise AlignerNotAvailableError if whisperx not importable."""
    # Temporarily hide whisperx from imports
    import importlib

    original = sys.modules.get("whisperx", None)
    sys.modules["whisperx"] = None  # type: ignore[assignment]
    try:
        with pytest.raises(AlignerNotAvailableError, match="[Ww]hisper"):
            WhisperXAligner()
    finally:
        if original is None:
            sys.modules.pop("whisperx", None)
        else:
            sys.modules["whisperx"] = original


def test_get_aligner_whisperx_raises_when_not_installed():
    original = sys.modules.get("whisperx", None)
    sys.modules["whisperx"] = None  # type: ignore[assignment]
    try:
        with pytest.raises(AlignerNotAvailableError):
            get_aligner("whisperx")
    finally:
        if original is None:
            sys.modules.pop("whisperx", None)
        else:
            sys.modules["whisperx"] = original


def test_auto_backend_falls_back_to_mock_when_whisperx_absent():
    """backend='auto' must succeed (returns mock words) and emit a RuntimeWarning."""
    audio, sr = _load_fixture()
    original = sys.modules.get("whisperx", None)
    sys.modules["whisperx"] = None  # type: ignore[assignment]
    try:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            words = align(audio, sr, TRANSCRIPT_8, backend="auto")
        assert len(words) == len(TOKENS_8)
        # At least one RuntimeWarning about fallback
        runtime_warnings = [x for x in w if issubclass(x.category, RuntimeWarning)]
        assert runtime_warnings, "Expected a RuntimeWarning about mock fallback"
    finally:
        if original is None:
            sys.modules.pop("whisperx", None)
        else:
            sys.modules["whisperx"] = original


# ---------------------------------------------------------------------------
# MockAligner is honest about its limitations
# ---------------------------------------------------------------------------


def test_mock_aligner_method_name_indicates_testonly():
    """ALIGNMENT_METHOD must clearly label the mock as test-only."""
    aligner = MockAligner()
    method = aligner.ALIGNMENT_METHOD
    assert "TEST" in method.upper() or "MOCK" in method.upper(), (
        f"MockAligner.ALIGNMENT_METHOD should contain 'MOCK' or 'TEST', got: {method!r}"
    )


def test_whisperx_aligner_method_name_is_informative():
    """WhisperXAligner.ALIGNMENT_METHOD should name the real method."""
    # Check class attribute without instantiating (avoids import side-effects)
    method = WhisperXAligner.ALIGNMENT_METHOD
    assert "whisperx" in method.lower() or "wav2vec" in method.lower(), (
        f"WhisperXAligner.ALIGNMENT_METHOD should reference real method, got: {method!r}"
    )
