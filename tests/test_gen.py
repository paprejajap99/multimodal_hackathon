import pytest
import numpy as np
import pyworld
from src.gen import apply_flaw
from src.align import Word
from src.taxonomy import flaw_ids, get_flaw

@pytest.fixture
def dummy_audio_and_words():
    sr = 16000
    audio = np.zeros(sr * 10, dtype=np.float32)
    words = [
        Word(index=0, text="hello", start_s=0.5, end_s=1.0),
        Word(index=1, text="world", start_s=1.5, end_s=2.0),
        Word(index=2, text="foo", start_s=2.5, end_s=3.0),
        Word(index=3, text="bar", start_s=3.5, end_s=4.0),
        Word(index=4, text="baz", start_s=4.5, end_s=5.0),
    ]

    # Give each word a distinct voiced pitch and leave deterministic gaps.
    frequencies = [180.0, 220.0, 260.0, 300.0, 340.0]
    for word, frequency in zip(words, frequencies):
        start = int(word["start_s"] * sr)
        end = int(word["end_s"] * sr)
        samples = np.arange(end - start) / sr
        audio[start:end] = 0.2 * np.sin(2 * np.pi * frequency * samples)
    return audio, sr, words


def _rms(audio):
    return float(np.sqrt(np.mean(np.square(audio))))


def _f0_std(audio, sr):
    f0, _ = pyworld.dio(audio.astype(np.float64), sr)
    voiced = f0[f0 > 0]
    return float(np.std(voiced))

@pytest.mark.parametrize("flaw_id", flaw_ids())
def test_apply_flaws(dummy_audio_and_words, flaw_id):
    audio, sr, words = dummy_audio_and_words
    level = 5
    
    start_idx = 1
    end_idx = 3 # "world" to "bar" inclusive, or gap between 1 and 2 if pause
    if flaw_id in ["short_pauses", "long_pauses"]:
        end_idx = start_idx + 1
        
    flawed_audio, new_words, gt = apply_flaw(
        audio, sr, words,
        flaw_id=flaw_id,
        level=level,
        word_start_idx=start_idx,
        word_end_idx=end_idx,
        seed=42
    )
    
    assert len(flawed_audio) > 0
    assert "start_s" in gt
    assert "end_s" in gt
    assert "level" in gt
    assert gt["flaw_type"] == flaw_id
    
    # Check bounds
    if flaw_id in ["pace_fast", "pace_slow"]:
        assert flawed_audio.shape[0] != audio.shape[0]
        # check that end word shifted (word index 4)
        assert new_words[4]["start_s"] != words[4]["start_s"]
    elif flaw_id in ["short_pauses", "long_pauses"]:
        assert flawed_audio.shape[0] != audio.shape[0]
        # check shift starting at word 2 (which is end_idx)
        assert new_words[2]["start_s"] != words[2]["start_s"]
        # word 1 should be unchanged
        assert new_words[1]["start_s"] == words[1]["start_s"]
    else:
        # low_volume, monotone
        assert flawed_audio.shape == audio.shape
        assert new_words[4]["start_s"] == words[4]["start_s"]

def test_invalid_level(dummy_audio_and_words):
    audio, sr, words = dummy_audio_and_words
    with pytest.raises(ValueError):
        apply_flaw(audio, sr, words, "pace_fast", 6, 1, 3)

def test_reproducibility(dummy_audio_and_words):
    audio, sr, words = dummy_audio_and_words
    a1, w1, gt1 = apply_flaw(audio, sr, words, "pace_fast", 3, 1, 3, seed=42)
    a2, w2, gt2 = apply_flaw(audio, sr, words, "pace_fast", 3, 1, 3, seed=42)
    assert np.allclose(a1, a2)


@pytest.mark.parametrize(
    ("flaw_id", "expected_direction"),
    [
        ("pace_fast", "increase"),
        ("pace_slow", "decrease"),
    ],
)
def test_pace_changes_speech_rate(dummy_audio_and_words, flaw_id, expected_direction):
    audio, sr, words = dummy_audio_and_words
    flawed_audio, new_words, gt = apply_flaw(audio, sr, words, flaw_id, 5, 1, 3)

    reference_duration = words[3]["end_s"] - words[1]["start_s"]
    participant_duration = new_words[3]["end_s"] - new_words[1]["start_s"]
    reference_rate = 3 / reference_duration
    participant_rate = 3 / participant_duration

    assert (participant_rate > reference_rate) == (expected_direction == "increase")
    assert gt["end_s"] == round(new_words[3]["end_s"], 4)
    assert len(flawed_audio) != len(audio)


def test_monotone_reduces_f0_variation(dummy_audio_and_words):
    audio, sr, words = dummy_audio_and_words
    flawed_audio, _, gt = apply_flaw(audio, sr, words, "monotone", 5, 0, 4)

    start = int(gt["start_s"] * sr)
    end = int(gt["end_s"] * sr)
    original_std = _f0_std(audio[start:end], sr)
    flawed_std = _f0_std(flawed_audio[start:end], sr)

    assert flawed_std < original_std


def test_low_volume_reduces_rms(dummy_audio_and_words):
    audio, sr, words = dummy_audio_and_words
    flawed_audio, _, gt = apply_flaw(audio, sr, words, "low_volume", 5, 1, 3)

    start = int(gt["start_s"] * sr)
    end = int(gt["end_s"] * sr)
    assert _rms(flawed_audio[start:end]) < _rms(audio[start:end])


@pytest.mark.parametrize("flaw_id", ["short_pauses", "long_pauses"])
def test_pause_flaws_change_pause_duration(dummy_audio_and_words, flaw_id):
    audio, sr, words = dummy_audio_and_words
    _, new_words, gt = apply_flaw(audio, sr, words, flaw_id, 5, 1, 2)

    reference_pause = words[2]["start_s"] - words[1]["end_s"]
    participant_pause = new_words[2]["start_s"] - new_words[1]["end_s"]
    expected_direction = get_flaw(flaw_id)["expected_direction"]

    assert (participant_pause > reference_pause) == (expected_direction == "increase")
    assert gt["start_s"] == round(new_words[1]["end_s"], 4)
    assert gt["end_s"] == round(new_words[2]["start_s"], 4)
