import csv
import json
from pathlib import Path

import pytest

from src.audio import load_audio
from src.align import align
from src.gen import GenerationConfig, generate_dataset
from src.taxonomy import flaw_ids


FIXTURES = Path(__file__).parent / "fixtures"
REFERENCE_AUDIO = FIXTURES / "sample_8words.wav"
TRANSCRIPT = FIXTURES / "sample_transcript.txt"


def _all_flaw_ranges():
    return {
        "pace_fast": (0, 4),
        "pace_slow": (0, 4),
        "monotone": (0, 4),
        "low_volume": (0, 4),
        "short_pauses": (3, 4),
        "long_pauses": (3, 4),
    }


def _config():
    return GenerationConfig(
        flaw_ids=flaw_ids(),
        levels=[1],
        word_ranges=_all_flaw_ranges(),
        seed=19,
        source_id="tiny_fixture",
    )


def test_generate_dataset_writes_expected_samples_and_manifest(tmp_path):
    manifest_path = generate_dataset(
        REFERENCE_AUDIO,
        tmp_path,
        transcript=TRANSCRIPT,
        config=_config(),
    )

    with manifest_path.open(newline="", encoding="utf-8") as manifest_file:
        rows = list(csv.DictReader(manifest_file))

    assert len(rows) == 6
    required = {
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
    }
    assert required.issubset(rows[0])
    assert {row["flaw_type"] for row in rows} == set(flaw_ids())

    for row in rows:
        audio_path = tmp_path / row["audio_path"]
        gt_path = tmp_path / row["ground_truth_path"]
        assert audio_path.is_file()
        assert gt_path.is_file()
        gt = json.loads(gt_path.read_text(encoding="utf-8"))
        region = gt["regions"][0]
        assert gt["recording_id"] == row["sample_id"]
        assert gt["reference_id"] == row["reference_id"]
        assert gt["source_file"] == row["source_file"]
        assert region["flaw_type"] == row["flaw_type"]
        assert region["level"] == int(row["level"])
        assert region["word_start_index"] == int(row["word_start_index"])
        assert region["word_end_index"] == int(row["word_end_index"])
        assert region["start_s"] == float(row["participant_start_s"])
        assert region["end_s"] == float(row["participant_end_s"])
        assert region["reference_start_s"] == float(row["reference_start_s"])
        assert region["reference_end_s"] == float(row["reference_end_s"])


def test_generation_is_reproducible(tmp_path):
    first = tmp_path / "first"
    second = tmp_path / "second"
    config = GenerationConfig(
        flaw_ids=["pace_fast", "long_pauses"],
        levels=[2, 5],
        word_ranges={"pace_fast": (0, 4), "long_pauses": (3, 4)},
        seed=23,
        source_id="repeatable",
    )

    first_manifest = generate_dataset(REFERENCE_AUDIO, first, TRANSCRIPT, config=config)
    second_manifest = generate_dataset(REFERENCE_AUDIO, second, TRANSCRIPT, config=config)

    assert first_manifest.read_bytes() == second_manifest.read_bytes()
    for relative_path in [
        Path("flawed/repeatable_pace_fast_L2_W0-4.wav"),
        Path("flawed/repeatable_pace_fast_L2_W0-4.gt.json"),
        Path("flawed/repeatable_long_pauses_L5_W3-4.wav"),
    ]:
        assert (first / relative_path).read_bytes() == (second / relative_path).read_bytes()


def test_generation_accepts_supplied_alignment(tmp_path):
    audio, sr = load_audio(REFERENCE_AUDIO)
    transcript = TRANSCRIPT.read_text(encoding="utf-8")
    words = align(audio, sr, transcript, backend="mock")

    manifest = generate_dataset(
        REFERENCE_AUDIO,
        tmp_path,
        alignment=words,
        config=GenerationConfig(
            flaw_ids=["long_pauses"],
            levels=[3],
            word_ranges={"long_pauses": (2, 3)},
        ),
    )
    row = next(csv.DictReader(manifest.open(newline="", encoding="utf-8")))
    assert row["word_start_index"] == "2"
    assert row["word_end_index"] == "3"


@pytest.mark.parametrize(
    ("config", "message"),
    [
        ({"flaw_ids": ["not_a_flaw"]}, "invalid flaw ID"),
        ({"flaw_ids": ["pace_fast"], "levels": [0]}, "severity levels"),
        ({"flaw_ids": ["pace_fast"], "word_ranges": {"pace_fast": (3, 99)}}, "invalid word range"),
    ],
)
def test_invalid_configuration_fails_clearly(tmp_path, config, message):
    with pytest.raises(ValueError, match=message):
        generate_dataset(REFERENCE_AUDIO, tmp_path, TRANSCRIPT, config=config)


def test_missing_reference_and_invalid_alignment_fail_clearly(tmp_path):
    with pytest.raises(ValueError, match="reference audio file not found"):
        generate_dataset(tmp_path / "missing.wav", tmp_path / "out", transcript=TRANSCRIPT)

    with pytest.raises(ValueError, match="monotonic"):
        generate_dataset(
            REFERENCE_AUDIO,
            tmp_path / "out",
            alignment=[
                {"index": 0, "text": "one", "start_s": 0.1, "end_s": 0.3},
                {"index": 1, "text": "two", "start_s": 0.2, "end_s": 0.4},
            ],
            config={"flaw_ids": ["pace_fast"], "levels": [1], "word_ranges": {"pace_fast": (0, 1)}},
        )