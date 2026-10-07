import copy
import logging
from typing import Any, Tuple, List, Dict

import numpy as np
import pyworld
import librosa

from src.taxonomy import get_flaw
from src.align import Word

logger = logging.getLogger(__name__)

def apply_flaw(
    audio: np.ndarray,
    sr: int,
    words: List[Word],
    flaw_id: str,
    level: int,
    word_start_idx: int,
    word_end_idx: int,
    seed: int = 42
) -> Tuple[np.ndarray, List[Word], Dict[str, Any]]:
    """
    Applies a controlled flaw to the given audio and alignment.

    word_start_idx, word_end_idx are inclusive word indices in the transcript.
    For gap flaws (short_pauses, long_pauses), it means the gap is between word_start_idx and word_start_idx+1 (word_end_idx is often assumed to be word_start_idx+1).

    Returns:
        (flawed_audio, new_words, ground_truth)
    """
    if level < 1 or level > 5:
        raise ValueError(f"Level must be 1-5, got {level}")

    flaw = get_flaw(flaw_id)
    values = flaw["injection"]["values"]
    injected_value = values[level - 1]
    expected_direction = flaw["expected_direction"]

    new_words = copy.deepcopy(words)
    flawed_audio = audio.copy()
    
    ref_start_word = words[word_start_idx]
    ref_end_word = words[word_end_idx]

    part_start_time = 0.0
    part_end_time = 0.0
    duration_diff = 0.0
    
    np.random.seed(seed)
    
    if flaw_id in ["pace_fast", "pace_slow"]:
        ref_start_s = ref_start_word["start_s"]
        ref_end_s = ref_end_word["end_s"]
        part_start_time = ref_start_s
        part_end_time = ref_end_s
        
        start_sample = int(ref_start_s * sr)
        end_sample = int(ref_end_s * sr)
        
        pre = audio[:start_sample]
        mid = audio[start_sample:end_sample]
        pos = audio[end_sample:]
        
        rate = injected_value # speed_factor
        if len(mid) > 0:
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                stretched = librosa.effects.time_stretch(mid, rate=rate)
        else:
            stretched = mid
            
        flawed_audio = np.concatenate([pre, stretched, pos])
        
        new_mid_dur = len(stretched) / sr
        old_mid_dur = ref_end_s - ref_start_s
        duration_diff = new_mid_dur - old_mid_dur
        
        # update overlapping and future words
        for i, w in enumerate(new_words):
            if i >= word_start_idx and i <= word_end_idx:
                # scale proportionally inside region
                if old_mid_dur > 0:
                    rel_start = (w["start_s"] - ref_start_s) / old_mid_dur
                    rel_end = (w["end_s"] - ref_start_s) / old_mid_dur
                    w["start_s"] = ref_start_s + rel_start * new_mid_dur
                    w["end_s"] = ref_start_s + rel_end * new_mid_dur
                else:
                    pass
            elif i > word_end_idx:
                w["start_s"] += duration_diff
                w["end_s"] += duration_diff

        part_end_time = ref_start_s + new_mid_dur

    elif flaw_id == "monotone":
        ref_start_s = ref_start_word["start_s"]
        ref_end_s = ref_end_word["end_s"]
        part_start_time = ref_start_s
        part_end_time = ref_end_s
        
        start_sample = int(ref_start_s * sr)
        end_sample = int(ref_end_s * sr)
        
        pre = audio[:start_sample]
        mid = audio[start_sample:end_sample].astype(np.float64)
        pos = audio[end_sample:]
        
        f0_range_retained = injected_value
        
        if len(mid) > 0:
            f0, t_pw = pyworld.dio(mid, sr)
            valid_f0 = f0[f0 > 0]
            if len(valid_f0) > 0:
                f0 = pyworld.stonemask(mid, f0, t_pw, sr)
                sp = pyworld.cheaptrick(mid, f0, t_pw, sr)
                ap = pyworld.d4c(mid, f0, t_pw, sr)
                
                voiced_f0 = f0[f0 > 0]
                if len(voiced_f0) > 0:
                    median_f0 = np.median(voiced_f0)
                    voiced_mask = f0 > 0
                    f0_new = f0.copy()
                    f0_new[voiced_mask] = median_f0 + (f0[voiced_mask] - median_f0) * f0_range_retained
                else:
                    f0_new = f0
                synth = pyworld.synthesize(f0_new, sp, ap, sr).astype(np.float32)
            else:
                synth = mid.astype(np.float32)
            
            # length match
            orig_len = len(mid)
            if len(synth) > orig_len:
                synth = synth[:orig_len]
            elif len(synth) < orig_len:
                synth = np.pad(synth, (0, orig_len - len(synth)))
        else:
            synth = mid.astype(np.float32)
            
        flawed_audio = np.concatenate([pre, synth, pos])
        
    elif flaw_id == "low_volume":
        ref_start_s = ref_start_word["start_s"]
        ref_end_s = ref_end_word["end_s"]
        part_start_time = ref_start_s
        part_end_time = ref_end_s
        
        start_sample = int(ref_start_s * sr)
        end_sample = int(ref_end_s * sr)
        
        gain_db = injected_value
        factor = 10 ** (gain_db / 20.0)
        
        pre = audio[:start_sample]
        mid = audio[start_sample:end_sample] * factor
        pos = audio[end_sample:]
        
        flawed_audio = np.concatenate([pre, mid, pos])

    elif flaw_id == "short_pauses":
        # gap is after word_start_idx and before word_start_idx + 1
        if word_end_idx <= word_start_idx:
            word_end_idx = word_start_idx + 1 # default to next word if not provided properly
        if word_end_idx >= len(words):
            raise ValueError("Cannot shorten pause after the last word as it's not a gap between words.")
            
        ref_start_s = words[word_start_idx]["end_s"]
        ref_end_s = words[word_end_idx]["start_s"]
        
        start_sample = int(ref_start_s * sr)
        end_sample = int(ref_end_s * sr)
        
        gap_audio = audio[start_sample:end_sample]
        gap_dur = ref_end_s - ref_start_s
        
        pause_scale = injected_value
        keep_len = int(len(gap_audio) * pause_scale)
        drop_len = len(gap_audio) - keep_len
        
        if drop_len > 0:
            # Drop from middle
            drop_start = (len(gap_audio) - drop_len) // 2
            keep_audio = np.concatenate([gap_audio[:drop_start], gap_audio[drop_start + drop_len:]])
        else:
            keep_audio = gap_audio
            
        pre = audio[:start_sample]
        pos = audio[end_sample:]
        
        flawed_audio = np.concatenate([pre, keep_audio, pos])
        
        new_mid_dur = len(keep_audio) / sr
        duration_diff = new_mid_dur - gap_dur
        
        # update all subsequent words
        for i, w in enumerate(new_words):
            if i >= word_end_idx:
                w["start_s"] += duration_diff
                w["end_s"] += duration_diff
                
        part_start_time = ref_start_s
        part_end_time = ref_start_s + new_mid_dur

    elif flaw_id == "long_pauses":
        # gap is between word_start_idx and word_end_idx
        if word_end_idx <= word_start_idx:
            word_end_idx = word_start_idx + 1 # default
            
        ref_start_s = words[word_start_idx]["end_s"]
        ref_end_s = words[word_end_idx]["start_s"]
        
        start_sample = int(ref_start_s * sr)
        # We just insert silence exactly at ref_start_s
        pre = audio[:start_sample]
        pos = audio[start_sample:]
        
        added_pause_s = injected_value
        insert_samples = int(added_pause_s * sr)
        actual_added_pause_s = insert_samples / sr
        silence = np.zeros(insert_samples, dtype=audio.dtype)
        
        flawed_audio = np.concatenate([pre, silence, pos])
        duration_diff = actual_added_pause_s
        
        part_start_time = ref_start_s
        
        # shift subsequent words
        for i, w in enumerate(new_words):
            # start shifting from next word (word_end_idx)
            if i >= word_end_idx:
                w["start_s"] += duration_diff
                w["end_s"] += duration_diff

        # The participant pause includes the original gap plus inserted silence.
        part_end_time = new_words[word_end_idx]["start_s"]
                
    else:
        raise ValueError(f"Unsupported flaw {flaw_id}")
        
    # Build Ground Truth
    gt = {
        "flaw_type": flaw_id,
        "level": level,
        "start_s": round(part_start_time, 4),
        "end_s": round(part_end_time, 4),
        "word_start_index": word_start_idx,
        "word_end_index": word_end_idx,
        "reference_start_s": round(words[word_start_idx]["start_s"] if flaw_id not in ["short_pauses", "long_pauses"] else words[word_start_idx]["end_s"], 4),
        "reference_end_s": round(words[word_end_idx]["end_s"] if flaw_id not in ["short_pauses", "long_pauses"] else words[word_end_idx]["start_s"], 4),
        "injected_value": injected_value
    }

    # Ensure precision is matched roughly
    for w in new_words:
        w["start_s"] = round(w["start_s"], 4)
        w["end_s"] = round(w["end_s"], 4)

    return flawed_audio, new_words, gt
