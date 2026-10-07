import numpy as np
import pytest
from src.features import FeatureExtractor

def test_feature_extractor_basic():
    # 3 seconds of audio at 16000 Hz
    sr = 16000
    t = np.linspace(0, 1, sr, endpoint=False)
    
    # Tone 1: 440 Hz, amplitude 0.5
    tone1 = 0.5 * np.sin(2 * np.pi * 440 * t)
    # Silence: 1.0 sec
    silence = np.zeros(sr)
    # Tone 2: 880 Hz (1 octave higher), amplitude 0.25 (half amp, approx -6dB)
    tone2 = 0.25 * np.sin(2 * np.pi * 880 * t)
    
    audio = np.concatenate([tone1, silence, tone2]).astype(np.float32)
    
    words = [
        {"index": 0, "text": "hello", "start_s": 0.0, "end_s": 1.0},
        {"index": 1, "text": "world", "start_s": 2.0, "end_s": 3.0},
    ]
    
    fe = FeatureExtractor(audio, sr, words)
    
    # Test tracking sizes
    f0_series = fe.get_series("f0_st_rel")
    rms_series = fe.get_series("rms_db_rel")
    wps_series = fe.get_series("speech_rate_wps")
    
    assert f0_series["hop_s"] > 0
    assert len(f0_series["values"]) == len(rms_series["values"])
    assert len(rms_series["values"]) == len(wps_series["values"])
    
    # Test region stats for Tone 1
    reg_1 = fe.get_region_stats(0, 0)
    assert reg_1["speech_rate_wps"] == 1.0  # 1 word / 1 second
    assert "f0_std_st" in reg_1
    assert "rms_db_rel" in reg_1
    
    # Test region stats for Tone 2
    reg_2 = fe.get_region_stats(1, 1)
    assert reg_2["speech_rate_wps"] == 1.0
    
    # Test loudness decrease between tone 1 and tone 2
    # half amplitude translates to roughly -6.02 dB
    db_diff = reg_2["rms_db_rel"] - reg_1["rms_db_rel"]
    assert -7.0 < db_diff < -5.0
    
    # Test pitch increase between tone 1 and tone 2
    # tone 2 is 880 Hz, tone 1 is 440 Hz. 12 semitones diff.
    # We don't have separate f0 region stats except std, but we know median.
    
    # Test pause duration
    # gap between word 0 (ends at 1.0) and word 1 (starts at 2.0) is 1.0s
    gap_stats = fe.get_region_stats(0, 1)
    assert gap_stats["pause_duration_s"] == 1.0
    # The speech rate for region 0..1 (2 words over 3.0 sec)
    assert pytest.approx(gap_stats["speech_rate_wps"], 0.01) == 2.0 / 3.0

def test_feature_extractor_flat_vs_varying_pitch():
    sr = 16000
    t = np.linspace(0, 2, 2 * sr, endpoint=False)
    # Flat tone
    flat = 0.5 * np.sin(2 * np.pi * 440 * t)
    
    # Varying tone (chirp from 440 Hz to 880 Hz)
    f_inst = np.linspace(440, 880, len(t))
    phase = 2 * np.pi * np.cumsum(f_inst) / sr
    varying = 0.5 * np.sin(phase)
    
    words_flat = [{"index": 0, "text": "flat", "start_s": 0.0, "end_s": 2.0}]
    fe_flat = FeatureExtractor(flat.astype(np.float32), sr, words_flat)
    reg_flat = fe_flat.get_region_stats(0, 0)
    
    words_var = [{"index": 0, "text": "varying", "start_s": 0.0, "end_s": 2.0}]
    fe_var = FeatureExtractor(varying.astype(np.float32), sr, words_var)
    reg_var = fe_var.get_region_stats(0, 0)
    
    # Flat pitch should have near zero pitch variance
    assert reg_flat["f0_std_st"] < 1.0
    
    # Varying pitch should have large pitch variance
    assert reg_var["f0_std_st"] > reg_flat["f0_std_st"]
