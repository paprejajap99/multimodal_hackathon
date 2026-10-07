"""src/audio.py — Audio loading and preprocessing for Track C pipeline.

Public API
----------
load_audio(path, sr=16_000) -> (np.ndarray, int)
    Load an audio file, convert to mono float32, resample to *sr* Hz.
    Returns (waveform_1d, sample_rate_hz). Time is always in **seconds**;
    the waveform is normalised to peak ≤ 1.0 (float32, values in [-1, 1]).

Exceptions
----------
AudioError          Base exception for this module.
AudioLoadError      Cannot open / decode the file.
AudioFormatError    File format not supported.

Units
-----
- Sample rate: Hz (int)
- Duration: seconds (float)
- Amplitude: linear float32 in [-1, 1]

Owner: T-02
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Union

import numpy as np

PathLike = Union[str, Path]


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AudioError(Exception):
    """Base exception for src.audio errors."""


class AudioLoadError(AudioError):
    """Raised when an audio file cannot be opened or decoded."""


class AudioFormatError(AudioError):
    """Raised when the audio format/codec is not supported."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _to_mono(waveform: np.ndarray) -> np.ndarray:
    """Convert any shape (samples,) or (channels, samples) waveform to mono 1D."""
    if waveform.ndim == 1:
        return waveform.astype(np.float32)
    if waveform.ndim == 2:
        # (channels, samples) → mean across channels
        return waveform.mean(axis=0).astype(np.float32)
    raise AudioFormatError(f"Unexpected waveform ndim={waveform.ndim}; expected 1 or 2.")


def _resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample waveform using scipy.signal.resample_poly (high-quality, no network)."""
    if orig_sr == target_sr:
        return waveform
    try:
        from math import gcd

        from scipy.signal import resample_poly  # type: ignore[import-untyped]

        g = gcd(orig_sr, target_sr)
        up, down = target_sr // g, orig_sr // g
        return resample_poly(waveform, up, down).astype(np.float32)
    except ImportError as exc:
        raise AudioLoadError(
            "scipy is required for resampling: pip install scipy"
        ) from exc


def _load_with_soundfile(path: Path) -> tuple[np.ndarray, int]:
    """Primary loader: soundfile supports WAV, FLAC, OGG, AIFF, etc."""
    import soundfile as sf  # type: ignore[import-untyped]

    try:
        data, sr = sf.read(str(path), dtype="float32", always_2d=False)
    except sf.SoundFileError as exc:
        raise AudioLoadError(f"soundfile could not read '{path}': {exc}") from exc
    # soundfile returns (samples,) for mono or (samples, channels) for multi-channel
    # Transpose to (channels, samples) convention used in _to_mono
    if data.ndim == 2:
        data = data.T
    return data, int(sr)


def _load_with_scipy(path: Path) -> tuple[np.ndarray, int]:
    """Fallback loader: scipy.io.wavfile for basic WAV files only."""
    from scipy.io import wavfile  # type: ignore[import-untyped]

    try:
        sr, data = wavfile.read(str(path))
    except Exception as exc:
        raise AudioLoadError(f"scipy.io.wavfile could not read '{path}': {exc}") from exc

    # wavfile returns int types; normalise to float32
    if np.issubdtype(data.dtype, np.integer):
        info = np.iinfo(data.dtype)
        data = data.astype(np.float32) / max(abs(info.min), info.max)
    else:
        data = data.astype(np.float32)

    # scipy returns (samples,) mono or (samples, channels) stereo
    if data.ndim == 2:
        data = data.T  # → (channels, samples)
    return data, int(sr)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_audio(path: PathLike, sr: int = 16_000) -> tuple[np.ndarray, int]:
    """Load an audio file, convert to mono float32, resample to *sr* Hz.

    Parameters
    ----------
    path : str or Path
        Path to the audio file (WAV, FLAC, OGG, AIFF supported via soundfile;
        WAV-only via scipy fallback).
    sr : int
        Target sample rate in Hz. Default: 16 000 Hz (required by the pipeline).

    Returns
    -------
    waveform : np.ndarray
        1-D float32 array of audio samples, peak-normalised if > 1.0.
    sample_rate : int
        The target sample rate *sr* (always equal to the *sr* argument).

    Raises
    ------
    AudioLoadError
        File does not exist, cannot be decoded, or the audio is empty.
    AudioFormatError
        The channel layout is not supported.
    """
    if sr <= 0:
        raise ValueError(f"sr must be positive, got {sr!r}")

    p = Path(path)
    if not p.exists():
        raise AudioLoadError(f"Audio file not found: '{p}'")
    if not p.is_file():
        raise AudioLoadError(f"Path is not a file: '{p}'")
    if p.stat().st_size == 0:
        raise AudioLoadError(f"Audio file is empty (0 bytes): '{p}'")

    # --- Load raw data ---
    raw_waveform: np.ndarray | None = None
    orig_sr: int | None = None

    try:
        raw_waveform, orig_sr = _load_with_soundfile(p)
    except (AudioLoadError, ImportError):
        warnings.warn(
            "soundfile not available or failed; falling back to scipy.io.wavfile "
            "(WAV only). Install soundfile for full format support.",
            RuntimeWarning,
            stacklevel=2,
        )
        raw_waveform, orig_sr = _load_with_scipy(p)

    # --- Mono downmix ---
    mono = _to_mono(raw_waveform)

    # --- Empty audio guard ---
    if mono.size == 0:
        raise AudioLoadError(f"Audio file contains no samples: '{p}'")

    # --- Resample ---
    resampled = _resample(mono, orig_sr, sr)

    # --- Empty after resample guard ---
    if resampled.size == 0:
        raise AudioLoadError(f"Resampling produced empty output for: '{p}'")

    # --- Peak normalise (soft: only if peak > 1.0, preserving quiet audio) ---
    peak = float(np.max(np.abs(resampled)))
    if peak > 1.0:
        resampled = resampled / peak

    return resampled, sr
