import pytest
import numpy as np
from src.gen import apply_flaw
from src.align import Word
from src.taxonomy import flaw_ids

@pytest.fixture
def dummy_audio_and_words():
    sr = 16000
    audio = np.random.randn(sr * 10).astype(np.float32) * 0.1 # 10 seconds of "speech"
    words = [
        Word(index=0, text="hello", start_s=0.5, end_s=1.0),
        Word(index=1, text="world", start_s=1.5, end_s=2.0),
        Word(index=2, text="foo", start_s=2.5, end_s=3.0),
        Word(index=3, text="bar", start_s=3.5, end_s=4.0),
        Word(index=4, text="baz", start_s=4.5, end_s=5.0),
    ]
    return audio, sr, words

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
