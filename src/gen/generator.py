"""Build small, reproducible contrastive datasets from aligned reference audio."""

from __future__ import annotations

import csv
import json
import re
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from src.align import Word, align
from src.audio import load_audio
from src.taxonomy import flaw_ids as taxonomy_flaw_ids, get_flaw

from .flaw_generator import apply_flaw


MANIFEST_COLUMNS = [
	"sample_id",
	"reference_id",
	"source_file",
	"flaw_type",
	"level",
	"seed",
	"word_start_index",
	"word_end_index",
	"reference_start_s",
	"reference_end_s",
	"participant_start_s",
	"participant_end_s",
	"audio_path",
	"ground_truth_path",
]


@dataclass(frozen=True)
class GenerationConfig:
	"""Configuration for one deterministic synthetic dataset build."""

	flaw_ids: Sequence[str] = field(default_factory=taxonomy_flaw_ids)
	levels: Sequence[int] = (1, 2, 3, 4, 5)
	word_start_idx: int = 0
	word_end_idx: int | None = None
	word_ranges: Mapping[str, tuple[int, int]] = field(default_factory=dict)
	seed: int = 7
	alignment_backend: str = "mock"
	source_id: str | None = None
	reference_id: str | None = None


def _as_config(config: GenerationConfig | Mapping[str, Any] | None) -> GenerationConfig:
	if config is None:
		return GenerationConfig()
	if isinstance(config, GenerationConfig):
		return config
	if not isinstance(config, Mapping):
		raise TypeError("config must be a GenerationConfig or mapping")
	values = dict(config)
	if "flaw_types" in values and "flaw_ids" not in values:
		values["flaw_ids"] = values.pop("flaw_types")
	return GenerationConfig(**values)


def _read_transcript(transcript: str | Path | None) -> str | None:
	if transcript is None:
		return None
	path = Path(transcript)
	if path.is_file():
		return path.read_text(encoding="utf-8")
	if isinstance(transcript, Path):
		raise ValueError(f"Transcript file not found: '{path}'")
	return transcript


def _read_alignment(alignment: Sequence[Word] | str | Path) -> list[Word]:
	if isinstance(alignment, (str, Path)):
		path = Path(alignment)
		if not path.is_file():
			raise ValueError(f"Alignment file not found: '{path}'")
		payload = json.loads(path.read_text(encoding="utf-8"))
		alignment = payload.get("words") if isinstance(payload, dict) else payload
	if not isinstance(alignment, Sequence) or isinstance(alignment, (str, bytes)):
		raise ValueError("alignment must be a sequence of word records")
	return [dict(word) for word in alignment]


def _validate_alignment(words: Sequence[Mapping[str, Any]], duration_s: float) -> list[Word]:
	if not words:
		raise ValueError("alignment must contain at least one word")
	validated: list[Word] = []
	for expected_index, word in enumerate(words):
		required = {"index", "text", "start_s", "end_s"}
		if not required.issubset(word):
			raise ValueError(f"alignment word {expected_index} is missing required fields")
		if word["index"] != expected_index:
			raise ValueError("alignment word indices must be contiguous and 0-based")
		try:
			start_s = float(word["start_s"])
			end_s = float(word["end_s"])
		except (TypeError, ValueError) as exc:
			raise ValueError(f"alignment word {expected_index} has invalid timestamps") from exc
		if not str(word["text"]).strip() or start_s < 0 or end_s <= start_s:
			raise ValueError(f"alignment word {expected_index} has invalid word bounds")
		if end_s > duration_s + 1e-4:
			raise ValueError(f"alignment word {expected_index} extends beyond audio duration")
		if validated and start_s < validated[-1]["end_s"] - 1e-4:
			raise ValueError("alignment word timestamps must be monotonic")
		validated.append(
			Word(
				index=expected_index,
				text=str(word["text"]),
				start_s=round(start_s, 4),
				end_s=round(end_s, 4),
			)
		)
	return validated


def _validate_config(config: GenerationConfig) -> list[str]:
	requested = list(config.flaw_ids)
	if not requested:
		raise ValueError("config.flaw_ids must contain at least one flaw ID")
	if len(set(requested)) != len(requested):
		raise ValueError("config.flaw_ids must not contain duplicates")
	known = set(taxonomy_flaw_ids())
	invalid = [flaw_id for flaw_id in requested if flaw_id not in known]
	if invalid:
		raise ValueError(f"invalid flaw ID(s): {', '.join(invalid)}")
	levels = list(config.levels)
	if not levels:
		raise ValueError("config.levels must contain at least one severity level")
	if any(not isinstance(level, int) or isinstance(level, bool) or not 1 <= level <= 5 for level in levels):
		raise ValueError("severity levels must be integers from 1 through 5")
	if len(set(levels)) != len(levels):
		raise ValueError("config.levels must not contain duplicates")
	if not isinstance(config.seed, int) or isinstance(config.seed, bool):
		raise ValueError("config.seed must be an integer")
	if (
		not isinstance(config.word_start_idx, int)
		or isinstance(config.word_start_idx, bool)
		or (
			config.word_end_idx is not None
			and (
				not isinstance(config.word_end_idx, int)
				or isinstance(config.word_end_idx, bool)
			)
		)
	):
		raise ValueError("word range indices must be integers")
	return requested


def _range_for(config: GenerationConfig, flaw_id: str, word_count: int) -> tuple[int, int]:
	if flaw_id in config.word_ranges:
		bounds = config.word_ranges[flaw_id]
		if not isinstance(bounds, (tuple, list)) or len(bounds) != 2:
			raise ValueError(f"word range for {flaw_id} must contain start and end indices")
		start_idx, end_idx = bounds
	else:
		start_idx = config.word_start_idx
		end_idx = config.word_end_idx
		if end_idx is None:
			end_idx = start_idx + 1 if get_flaw(flaw_id)["region_type"] == "gap" else word_count - 1
	if not isinstance(start_idx, int) or not isinstance(end_idx, int):
		raise ValueError(f"word range for {flaw_id} must use integer indices")
	if not 0 <= start_idx < end_idx < word_count:
		raise ValueError(
			f"invalid word range for {flaw_id}: ({start_idx}, {end_idx}); "
			f"expected 0 <= start < end < {word_count}"
		)
	if get_flaw(flaw_id)["region_type"] == "gap" and end_idx != start_idx + 1:
		raise ValueError(f"word range for {flaw_id} must identify adjacent words")
	return start_idx, end_idx


def _safe_id(value: str) -> str:
	return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "reference"


def _write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
	samples = np.clip(np.asarray(audio, dtype=np.float32), -1.0, 1.0)
	pcm = (samples * 32767.0).astype(np.int16)
	with wave.open(str(path), "wb") as wav_file:
		wav_file.setnchannels(1)
		wav_file.setsampwidth(2)
		wav_file.setframerate(sr)
		wav_file.writeframes(pcm.tobytes())


def _sample_seed(base_seed: int, flaw_id: str, level: int) -> int:
	taxonomy_index = taxonomy_flaw_ids().index(flaw_id) + 1
	return base_seed + taxonomy_index * 100 + level


def generate_dataset(
	reference_audio: str | Path,
	output_dir: str | Path,
	transcript: str | Path | None = None,
	alignment: Sequence[Word] | str | Path | None = None,
	config: GenerationConfig | Mapping[str, Any] | None = None,
) -> Path:
	"""Generate synthetic flawed audio, GT JSON, and a CSV manifest.

	The returned path is ``output_dir / "manifest.csv"``. All timestamps are
	seconds, and GT ``start_s``/``end_s`` values are on each participant's
	updated timeline.
	"""
	generation_config = _as_config(config)
	requested_flaws = _validate_config(generation_config)
	reference_path = Path(reference_audio)
	if not reference_path.is_file():
		raise ValueError(f"reference audio file not found: '{reference_path}'")
	output_path = Path(output_dir)
	output_path.mkdir(parents=True, exist_ok=True)
	flawed_dir = output_path / "flawed"
	flawed_dir.mkdir(parents=True, exist_ok=True)

	audio, sr = load_audio(reference_path)
	if alignment is None:
		transcript_text = _read_transcript(transcript)
		if transcript_text is None:
			raise ValueError("provide transcript when alignment is not supplied")
		words = align(audio, sr, transcript_text, backend=generation_config.alignment_backend)
	else:
		words = _read_alignment(alignment)
	reference_words = _validate_alignment(words, len(audio) / sr)

	source_id = _safe_id(generation_config.source_id or reference_path.stem)
	reference_id = _safe_id(generation_config.reference_id or f"{source_id}_good")
	rows: list[dict[str, Any]] = []
	for flaw_id in requested_flaws:
		start_idx, end_idx = _range_for(generation_config, flaw_id, len(reference_words))
		for level in generation_config.levels:
			sample_seed = _sample_seed(generation_config.seed, flaw_id, level)
			sample_id = f"{source_id}_{flaw_id}_L{level}_W{start_idx}-{end_idx}"
			flawed_audio, participant_words, region = apply_flaw(
				audio,
				sr,
				reference_words,
				flaw_id,
				level,
				start_idx,
				end_idx,
				seed=sample_seed,
			)
			audio_rel = Path("flawed") / f"{sample_id}.wav"
			gt_rel = Path("flawed") / f"{sample_id}.gt.json"
			_write_wav(output_path / audio_rel, flawed_audio, sr)
			region = dict(region)
			if get_flaw(flaw_id)["region_type"] == "gap":
				participant_start_s = participant_words[start_idx]["end_s"]
				participant_end_s = participant_words[end_idx]["start_s"]
			else:
				participant_start_s = participant_words[start_idx]["start_s"]
				participant_end_s = participant_words[end_idx]["end_s"]
			region["start_s"] = round(participant_start_s, 4)
			region["end_s"] = round(participant_end_s, 4)
			region["participant_start_s"] = region["start_s"]
			region["participant_end_s"] = region["end_s"]
			gt = {
				"recording_id": sample_id,
				"speech_id": source_id,
				"reference_id": reference_id,
				"kind": "synthetic",
				"seed": sample_seed,
				"overall_level": level,
				"source_file": str(reference_path),
				"participant_audio": audio_rel.as_posix(),
				"words": participant_words,
				"regions": [region],
			}
			(output_path / gt_rel).write_text(json.dumps(gt, indent=2) + "\n", encoding="utf-8")
			rows.append(
				{
					"sample_id": sample_id,
					"reference_id": reference_id,
					"source_file": str(reference_path),
					"flaw_type": flaw_id,
					"level": level,
					"seed": sample_seed,
					"word_start_index": region["word_start_index"],
					"word_end_index": region["word_end_index"],
					"reference_start_s": region["reference_start_s"],
					"reference_end_s": region["reference_end_s"],
					"participant_start_s": region["participant_start_s"],
					"participant_end_s": region["participant_end_s"],
					"audio_path": audio_rel.as_posix(),
					"ground_truth_path": gt_rel.as_posix(),
				}
			)

	manifest_path = output_path / "manifest.csv"
	with manifest_path.open("w", newline="", encoding="utf-8") as manifest_file:
		writer = csv.DictWriter(manifest_file, fieldnames=MANIFEST_COLUMNS)
		writer.writeheader()
		writer.writerows(rows)
	return manifest_path


build_dataset = generate_dataset


__all__ = ["GenerationConfig", "MANIFEST_COLUMNS", "build_dataset", "generate_dataset"]
