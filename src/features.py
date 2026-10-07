"""src/features.py — Acoustic feature extraction & speaker normalization.

Owner: T-03
"""
from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import librosa

from src.align import Word

class FeatureExtractor:
    """Extracts acoustic features normalized to the speaker/recording."""

    def __init__(self, audio: np.ndarray, sr: int, words: list[Word]) -> None:
        """
        Compute frame-level and word/region-level acoustic features.
        
        Parameters
        ----------
        audio : np.ndarray
            Mono audio waveform.
        sr : int
            Sample rate in Hz.
        words : list[Word]
            Forced alignment words (from src.align).
        """
        if audio.size == 0:
            raise ValueError("Empty audio array.")
            
        self.audio = audio
        self.sr = sr
        self.words = sorted(words, key=lambda w: w['index'])
        self.hop_length = 512
        self.hop_s = self.hop_length / float(sr)
        
        self.duration_s = audio.size / sr
        
        # 1. Pitch (F0) extraction using librosa.pyin
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # fmin/fmax cover most human pitches
            f0, voiced_flag, _ = librosa.pyin(
                y=audio,
                fmin=50,
                fmax=500,
                sr=sr,
                frame_length=2048,
                hop_length=self.hop_length,
                fill_na=np.nan
            )
        
        self.f0 = np.asarray(f0)
        self.voiced_flag = np.asarray(voiced_flag, dtype=bool)

        # Normalize F0 -> f0_st_rel
        voiced_f0s = self.f0[self.voiced_flag]
        if len(voiced_f0s) > 0:
            self.median_f0 = float(np.median(voiced_f0s))
            # 12*log2(f0 / median) -> semitones relative to median
            st_rel = 12.0 * np.log2(np.maximum(self.f0, 1e-8) / self.median_f0)
            self.f0_st_rel = np.where(self.voiced_flag, st_rel, np.nan)
        else:
            self.median_f0 = 100.0
            self.f0_st_rel = np.full_like(self.f0, np.nan)
            
        # 2. Loudness (RMS)
        rms = librosa.feature.rms(
            y=audio, 
            frame_length=2048, 
            hop_length=self.hop_length, 
            center=True
        )[0]
        
        rms_db = 20 * np.log10(np.maximum(rms, 1e-10))
        
        # "dB minus median speech-frame dB of the same recording"
        if np.any(self.voiced_flag):
            self.median_speech_rms_db = float(np.median(rms_db[self.voiced_flag]))
        elif len(rms_db) > 0:
            self.median_speech_rms_db = float(np.median(rms_db))
        else:
            self.median_speech_rms_db = -50.0
            
        self.rms_db_rel = rms_db - self.median_speech_rms_db

        # 3. Speech Rate (windowed words per second)
        num_frames = len(self.rms_db_rel)
        wps_series = np.zeros(num_frames, dtype=float)
        
        for w in self.words:
            start_s = w['start_s']
            end_s = w['end_s']
            dur = end_s - start_s
            rate = 1.0 / dur if dur > 0 else 0.0
            
            start_frame = max(0, int(round(start_s / self.hop_s)))
            end_frame = min(num_frames, int(round(end_s / self.hop_s)))
            if start_frame < num_frames:
                wps_series[start_frame:end_frame] = rate
                
        self.speech_rate_wps_series = wps_series

        # 4. Additional arrays
        self.mfcc = librosa.feature.mfcc(y=audio, sr=sr, hop_length=self.hop_length, n_mfcc=13)
        self.fft = np.abs(librosa.stft(audio, hop_length=self.hop_length))

    def get_series(self, feature_id: str) -> dict[str, Any]:
        """
        Return the series in schema track format.
        { "start_s": float, "hop_s": float, "values": list[float | None] }
        """
        if feature_id == "f0_st_rel":
            vals = [float(v) if not np.isnan(v) else None for v in self.f0_st_rel]
        elif feature_id == "rms_db_rel":
            vals = [float(v) for v in self.rms_db_rel]
        elif feature_id == "speech_rate_wps":
            vals = [float(v) for v in self.speech_rate_wps_series]
        else:
            raise ValueError(f"Unsupported series feature: {feature_id}")

        return {
            "start_s": 0.0,
            "hop_s": float(self.hop_s),
            "values": vals
        }

    def get_region_stats(self, word_start_idx: int, word_end_idx: int) -> dict[str, float]:
        """
        Compute scalar statistics over a region (inclusive of word_end_idx).
        """
        if not self.words:
            return {
                "speech_rate_wps": 0.0,
                "f0_std_st": 0.0,
                "rms_db_rel": 0.0,
                "pause_duration_s": 0.0,
            }
            
        start_w = self.words[word_start_idx]
        end_w = self.words[word_end_idx]
        start_s = start_w['start_s']
        end_s = end_w['end_s']

        # 1. pause_duration_s
        if word_end_idx == word_start_idx + 1:
            gap = end_w['start_s'] - start_w['end_s']
            pause_duration_s = max(0.0, float(gap))
        else:
            pause_duration_s = 0.0
            
        # 2. speech_rate_wps
        duration_s = end_s - start_s
        num_words = word_end_idx - word_start_idx + 1
        wps = float(num_words / duration_s) if duration_s > 0 else 0.0

        # Frame indices
        start_frame = max(0, int(round(start_s / self.hop_s)))
        end_frame = min(len(self.rms_db_rel), int(round(end_s / self.hop_s)))
        if end_frame == start_frame:
            end_frame += 1
            
        num_frames = len(self.rms_db_rel)
        end_frame = min(end_frame, num_frames)
        start_frame = min(start_frame, end_frame - 1)
        if start_frame < 0:
            start_frame = 0

        # 3. f0_std_st
        f0_region = self.f0_st_rel[start_frame:end_frame]
        voiced_region = f0_region[~np.isnan(f0_region)]
        if len(voiced_region) > 1:
            f0_std_st = float(np.std(voiced_region))
        else:
            f0_std_st = 0.0
            
        # 4. rms_db_rel
        rms_region = self.rms_db_rel[start_frame:end_frame]
        reg_voiced_flags = self.voiced_flag[start_frame:end_frame]
        if np.any(reg_voiced_flags):
            region_rms = np.median(rms_region[reg_voiced_flags])
        elif len(rms_region) > 0:
            region_rms = np.median(rms_region)
        else:
            region_rms = 0.0
            
        return {
            "speech_rate_wps": wps,
            "f0_std_st": f0_std_st,
            "rms_db_rel": float(region_rms),
            "pause_duration_s": pause_duration_s,
        }
