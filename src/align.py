"""src/align.py — Word-level forced alignment for Track C pipeline.

Public API
----------
align(audio, sr, transcript, backend="auto") -> list[Word]
    Force-align *transcript* to *audio*, returning one Word per whitespace token.

get_aligner(backend) -> BaseAligner
    Return the aligner instance for the named backend.

Word (TypedDict)
    {index: int, text: str, start_s: float, end_s: float}

ALIGNMENT_BACKENDS
    Dict of available backend names → short description.

Exceptions
----------
AlignmentError              Invalid input or alignment failed.
AlignerNotAvailableError    Requested backend not installed / not importable.

Contract (ARCHITECTURE §6 + analysis.schema.json $defs.word)
------------------------------------------------------------
- One Word per whitespace-split token of the transcript (no merging / splitting).
- word.index == position in token list (0-based).
- word.text == transcript token exactly.
- Monotonic: word[i].end_s <= word[i+1].start_s  (gap ≥ 0 allowed).
- All times in **seconds** (float).
- Deterministic per backend (same input → same output).

Backends
--------
whisperx   Real forced alignment via WhisperX + a wav2vec2 phoneme model.
           Requires: pip install whisperx  (GPU optional but recommended).
           Downloads model on first run; works offline thereafter.
           This is the PRODUCTION backend for real/demo speech.

mock       Deterministic duration-proportional model — for UNIT TESTS ONLY.
           WARNING: Does NOT perform real forced alignment. Word boundaries
           are estimated from character lengths, not acoustic evidence.
           NEVER use this backend to evaluate real speech quality.

auto       Tries whisperx first; if unavailable falls back to mock with a
           warning. Auto mode is for test/development convenience only —
           production pipelines should specify 'whisperx' explicitly.

Owner: T-02
"""

from __future__ import annotations

import abc
import re
import warnings
from typing import TypedDict

import numpy as np

# ---------------------------------------------------------------------------
# Shared types
# ---------------------------------------------------------------------------


class Word(TypedDict):
    """One aligned word token.

    Fields
    ------
    index : int
        0-based position in the transcript token list.
    text : str
        Exact transcript token (may include punctuation, e.g. "war,").
    start_s : float
        Start time of the word in seconds on the respective audio timeline.
    end_s : float
        End time of the word in seconds on the respective audio timeline.
        Invariant: end_s > start_s.
    """

    index: int
    text: str
    start_s: float
    end_s: float


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------


class AlignmentError(Exception):
    """Invalid input to align() or the aligner failed to produce output."""


class AlignerNotAvailableError(AlignmentError):
    """The requested alignment backend is not installed or importable."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_WORD_SEP = re.compile(r"\s+")


def _tokenise(transcript: str) -> list[str]:
    """Split transcript on whitespace; return empty list for empty/whitespace-only input."""
    stripped = transcript.strip()
    if not stripped:
        return []
    return [t for t in _WORD_SEP.split(stripped) if t]


def _validate_inputs(audio: np.ndarray, sr: int, tokens: list[str]) -> None:
    if len(tokens) == 0:
        raise AlignmentError("transcript is empty or contains only whitespace.")
    if audio.ndim != 1:
        raise AlignmentError(
            f"audio must be a 1-D numpy array, got ndim={audio.ndim}."
        )
    if audio.size == 0:
        raise AlignmentError("audio array is empty (0 samples).")
    if sr <= 0:
        raise AlignmentError(f"sr must be positive, got {sr!r}.")
    duration_s = audio.size / sr
    if duration_s < 0.05:
        raise AlignmentError(
            f"Audio too short ({duration_s:.3f} s) to align {len(tokens)} words."
        )


def _assert_monotonic(words: list[Word]) -> None:
    """Raise AlignmentError if words violate the monotonicity contract."""
    for i in range(len(words) - 1):
        if words[i]["end_s"] > words[i + 1]["start_s"] + 1e-6:
            raise AlignmentError(
                f"Monotonicity violation: word[{i}].end_s={words[i]['end_s']:.4f} > "
                f"word[{i+1}].start_s={words[i+1]['start_s']:.4f}"
            )


# ---------------------------------------------------------------------------
# Aligner base class
# ---------------------------------------------------------------------------


class BaseAligner(abc.ABC):
    """Abstract base for all alignment backends."""

    #: Short identifier used in metadata.alignment_method field.
    ALIGNMENT_METHOD: str

    @abc.abstractmethod
    def align(self, audio: np.ndarray, sr: int, transcript: str) -> list[Word]:
        """Return word-level timestamps.

        Parameters
        ----------
        audio : np.ndarray  shape (N,), float32, peak ≤ 1.0
            Mono waveform as returned by src.audio.load_audio.
        sr : int
            Sample rate of *audio* in Hz (typically 16 000).
        transcript : str
            The spoken text, whitespace-tokenised. Must not be empty.

        Returns
        -------
        list[Word]
            One entry per whitespace token of *transcript*, monotonic,
            all times in seconds.
        """


# ---------------------------------------------------------------------------
# Backend 1: WhisperX (PRODUCTION — real forced alignment)
# ---------------------------------------------------------------------------


class WhisperXAligner(BaseAligner):
    """Real word-level forced aligner using WhisperX.

    Uses WhisperX's forced alignment pipeline (wav2vec2 phoneme model from
    Meta MMS / torchaudio) to produce accurate word-level timestamps.

    Requirements
    ------------
    pip install whisperx      # pulls torch, transformers, etc.
    # GPU optional; CPU inference is much slower (~10-30× real-time).

    Model download
    --------------
    The phoneme alignment model (~1 GB) is downloaded automatically on first
    use and cached locally (HuggingFace cache). Subsequent runs are offline.

    Parameters
    ----------
    model_size : str
        Whisper ASR model size for language detection / transcription
        (not used for alignment itself when transcript is provided).
        Default: "base" (fastest; alignment quality is independent of this).
    language : str
        BCP-47 language code.  Default: "en".
    device : str or None
        "cuda" | "cpu" | None (auto-detect).
    compute_type : str
        quantisation type passed to WhisperX ("float32" on CPU, "float16" on GPU).
    """

    ALIGNMENT_METHOD = "whisperx_wav2vec2_mms"

    def __init__(
        self,
        model_size: str = "base",
        language: str = "en",
        device: str | None = None,
        compute_type: str | None = None,
    ) -> None:
        self._model_size = model_size
        self._language = language
        self._device = device
        self._compute_type = compute_type
        self._align_model = None
        self._align_metadata = None
        self._check_available()

    @staticmethod
    def _check_available() -> None:
        try:
            import whisperx  # noqa: F401
        except ImportError as exc:
            raise AlignerNotAvailableError(
                "WhisperX is not installed. Install it with:\n"
                "  pip install whisperx\n"
                "Note: WhisperX requires PyTorch and will download ~1 GB of models "
                "on first run. A GPU is recommended but not required."
            ) from exc

    def _load_model(self) -> None:
        """Lazy-load the alignment model (downloads on first call)."""
        if self._align_model is not None:
            return
        import torch  # type: ignore[import-untyped]
        import whisperx  # type: ignore[import-untyped]

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        compute_type = self._compute_type or ("float16" if device == "cuda" else "float32")
        self._device = device
        self._compute_type = compute_type

        self._align_model, self._align_metadata = whisperx.load_align_model(
            language_code=self._language, device=device
        )

    def align(self, audio: np.ndarray, sr: int, transcript: str) -> list[Word]:
        """Force-align transcript to audio using WhisperX wav2vec2 model.

        The transcript is provided externally (no ASR step); WhisperX's
        forced aligner maps phonemes → audio frames → word boundaries.

        Parameters
        ----------
        audio : np.ndarray  shape (N,), float32
        sr : int            sample rate, should be 16 000 Hz
        transcript : str    the spoken text

        Returns
        -------
        list[Word]  one entry per whitespace token, monotonic timestamps.
        """
        import whisperx  # type: ignore[import-untyped]

        tokens = _tokenise(transcript)
        _validate_inputs(audio, sr, tokens)
        self._load_model()

        # WhisperX alignment expects a list of segments with word-level boxes.
        # We feed the transcript as a single segment covering the whole audio.
        duration_s = audio.size / sr
        whisperx_segments = [
            {
                "start": 0.0,
                "end": duration_s,
                "text": transcript.strip(),
                "words": [{"word": t} for t in tokens],
            }
        ]

        try:
            result = whisperx.align(
                whisperx_segments,
                self._align_model,
                self._align_metadata,
                audio,
                self._device,
                return_char_alignments=False,
            )
        except Exception as exc:
            raise AlignmentError(
                f"WhisperX alignment failed: {exc}"
            ) from exc

        # --- Parse result ---
        aligned_words: list[dict] = []
        for seg in result.get("segments", []):
            aligned_words.extend(seg.get("words", []))

        if len(aligned_words) == 0:
            raise AlignmentError(
                "WhisperX returned no aligned words. "
                "Check that the transcript matches the audio content."
            )

        # Some words may be missing 'start'/'end' if WhisperX couldn't place them.
        # Fill gaps with linear interpolation so the contract is always satisfied.
        word_list = self._fill_gaps(aligned_words, tokens, duration_s)

        _assert_monotonic(word_list)
        return word_list

    @staticmethod
    def _fill_gaps(
        aligned_words: list[dict],
        tokens: list[str],
        duration_s: float,
    ) -> list[Word]:
        """Build a Word list, filling in missing timestamps by linear interpolation."""
        # Map text → (start, end) from alignment results
        timed: list[tuple[float | None, float | None]] = []
        for aw in aligned_words:
            timed.append((aw.get("start"), aw.get("end")))

        # If count mismatch (WhisperX may silently merge/skip tokens), fall through
        if len(timed) != len(tokens):
            # Best-effort: zip to min length, then extrapolate
            warnings.warn(
                f"WhisperX returned {len(timed)} aligned words for {len(tokens)} tokens; "
                "filling missing words with interpolated timestamps.",
                RuntimeWarning,
                stacklevel=3,
            )
            # Pad with Nones
            while len(timed) < len(tokens):
                timed.append((None, None))
            timed = timed[: len(tokens)]

        # Compute a fallback grid for missing entries
        known_ends = [e for _, e in timed if e is not None]
        last_known = max(known_ends) if known_ends else duration_s
        grid_step = last_known / max(len(tokens), 1)

        words: list[Word] = []
        prev_end = 0.0
        for i, (token, (start, end)) in enumerate(zip(tokens, timed)):
            if start is None:
                start = prev_end
            if end is None:
                end = start + grid_step
            end = max(end, start + 0.01)  # ensure end > start
            words.append(Word(index=i, text=token, start_s=round(start, 4), end_s=round(end, 4)))
            prev_end = end

        return words


# ---------------------------------------------------------------------------
# Backend 2: MockAligner (TEST-ONLY — not real forced alignment)
# ---------------------------------------------------------------------------


class MockAligner(BaseAligner):
    """Deterministic word-boundary estimator — FOR UNIT TESTS ONLY.

    .. warning::
        This backend does NOT perform real forced alignment.
        Word timestamps are estimated purely from character lengths and a
        constant speaking-rate model. It produces structurally-valid output
        (correct types, counts, monotonicity) suitable for testing downstream
        components, but the timestamps bear no relation to the actual acoustic
        content of the audio.

        NEVER use this backend for real speech evaluation, scoring, or any
        result that will be presented to a user.

    Algorithm
    ---------
    1. Estimate total speech duration = audio duration − leading & trailing
       silences (heuristic: 0.2 s each).
    2. Assign each token a duration proportional to its character count,
       with a minimum of 0.05 s per word.
    3. Place a fixed inter-word gap of GAP_S seconds between words.
    4. All calculations are deterministic given audio.size and sr.
    """

    ALIGNMENT_METHOD = "mock_proportional_TESTONLY"

    #: Fixed silence assumed before first word and after last word (seconds).
    LEADING_SILENCE_S: float = 0.2
    TRAILING_SILENCE_S: float = 0.1

    #: Fixed gap between consecutive words (seconds).
    GAP_S: float = 0.05

    #: Minimum duration per word (seconds).
    MIN_WORD_S: float = 0.05

    def align(self, audio: np.ndarray, sr: int, transcript: str) -> list[Word]:
        """Return structurally-valid but acoustically-meaningless word timestamps.

        Parameters
        ----------
        audio : np.ndarray  shape (N,), float32
        sr : int
        transcript : str

        Returns
        -------
        list[Word]  — valid contract shape; timestamps are NOT acoustically accurate.
        """
        tokens = _tokenise(transcript)
        _validate_inputs(audio, sr, tokens)

        total_duration_s = audio.size / sr
        n = len(tokens)

        # Available speech time (after accounting for silences and gaps)
        available_s = max(
            total_duration_s - self.LEADING_SILENCE_S - self.TRAILING_SILENCE_S - (n - 1) * self.GAP_S,
            n * self.MIN_WORD_S,
        )

        # Token durations proportional to character count
        char_counts = np.array([max(len(t), 1) for t in tokens], dtype=np.float64)
        durations = (char_counts / char_counts.sum()) * available_s
        durations = np.maximum(durations, self.MIN_WORD_S)

        # Build word list
        words: list[Word] = []
        cursor = self.LEADING_SILENCE_S
        for i, (token, dur) in enumerate(zip(tokens, durations)):
            start = round(cursor, 4)
            end = round(cursor + float(dur), 4)
            words.append(Word(index=i, text=token, start_s=start, end_s=end))
            cursor = end + self.GAP_S

        _assert_monotonic(words)
        return words


# ---------------------------------------------------------------------------
# Backend registry & dispatcher
# ---------------------------------------------------------------------------

ALIGNMENT_BACKENDS: dict[str, str] = {
    "whisperx": "Real forced alignment via WhisperX + wav2vec2 MMS model (PRODUCTION).",
    "mock": "Deterministic character-proportional mock — UNIT TESTS ONLY; not real alignment.",
    "auto": "Try whisperx; fall back to mock with a warning (development/test use only).",
}


def get_aligner(backend: str) -> BaseAligner:
    """Return an aligner instance for the named backend.

    Parameters
    ----------
    backend : str
        One of 'whisperx', 'mock', or 'auto'.

    Returns
    -------
    BaseAligner instance.

    Raises
    ------
    AlignerNotAvailableError
        If backend='whisperx' and WhisperX is not installed.
    ValueError
        If backend name is not recognised.
    """
    if backend == "whisperx":
        return WhisperXAligner()
    if backend == "mock":
        return MockAligner()
    if backend == "auto":
        try:
            return WhisperXAligner()
        except AlignerNotAvailableError:
            warnings.warn(
                "WhisperX is not available; falling back to MockAligner. "
                "MockAligner does NOT perform real forced alignment — "
                "install whisperx for production use.",
                RuntimeWarning,
                stacklevel=2,
            )
            return MockAligner()
    raise ValueError(
        f"Unknown alignment backend: {backend!r}. "
        f"Choose from: {list(ALIGNMENT_BACKENDS)}"
    )


def align(
    audio: np.ndarray,
    sr: int,
    transcript: str,
    backend: str = "auto",
) -> list[Word]:
    """Force-align *transcript* to *audio*, returning one Word per whitespace token.

    Parameters
    ----------
    audio : np.ndarray  shape (N,), float32, peak ≤ 1.0
        Mono waveform as returned by src.audio.load_audio.
    sr : int
        Sample rate in Hz (typically 16 000).
    transcript : str
        The full spoken text. Must not be empty.
    backend : str
        Alignment backend: 'whisperx' (production), 'mock' (test-only), or
        'auto' (tries whisperx, falls back to mock with a warning).

    Returns
    -------
    list[Word]
        One Word per whitespace token of *transcript*. Monotonic timestamps.
        word.index == position in token list (0-based).

    Raises
    ------
    AlignmentError
        Empty transcript, audio too short, or alignment failure.
    AlignerNotAvailableError
        backend='whisperx' requested but not installed.
    ValueError
        Unknown backend name.

    Notes
    -----
    - The 'mock' backend is a test stub; its timestamps do not reflect real audio.
    - For real speech analysis, always use backend='whisperx'.
    - Alignment is deterministic per backend (same inputs → same output).
    """
    aligner = get_aligner(backend)
    return aligner.align(audio, sr, transcript)
