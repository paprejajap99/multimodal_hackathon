"""Command-line wrapper for the reproducible T-04 dataset builder."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .generator import GenerationConfig, generate_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate contrastive flaw audio, ground truth, and a manifest."
    )
    parser.add_argument(
        "--reference-audio",
        "--reference",
        "--input-audio",
        dest="reference_audio",
        required=True,
        type=Path,
        help="reference/good audio file",
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--transcript", type=Path, help="transcript text file")
    source.add_argument("--alignment", type=Path, help="alignment JSON file")
    parser.add_argument("--output-dir", required=True, type=Path, help="dataset output directory")
    parser.add_argument("--seed", type=int, default=7, help="base deterministic seed (default: 7)")
    parser.add_argument("--flaw-id", action="append", dest="flaw_ids", help="flaw ID (repeatable)")
    parser.add_argument("--level", action="append", type=int, dest="levels", help="severity level (repeatable)")
    parser.add_argument("--word-start", type=int, default=0, help="inclusive affected word index")
    parser.add_argument("--word-end", type=int, help="inclusive affected word index")
    parser.add_argument("--alignment-backend", default="mock", choices=("mock", "auto", "whisperx"))
    parser.add_argument("--source-id", help="stable source identifier")
    parser.add_argument("--reference-id", help="stable reference identifier")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse CLI arguments, delegate generation, and print the manifest path."""
    args = _parser().parse_args(argv)
    if args.alignment is not None:
        transcript = None
        alignment = args.alignment
    else:
        transcript = args.transcript
        alignment = None
    config = GenerationConfig(
        flaw_ids=args.flaw_ids or GenerationConfig().flaw_ids,
        levels=args.levels or GenerationConfig().levels,
        word_start_idx=args.word_start,
        word_end_idx=args.word_end,
        seed=args.seed,
        alignment_backend=args.alignment_backend,
        source_id=args.source_id,
        reference_id=args.reference_id,
    )
    try:
        manifest_path = generate_dataset(
            reference_audio=args.reference_audio,
            output_dir=args.output_dir,
            transcript=transcript,
            alignment=alignment,
            config=config,
        )
    except (OSError, TypeError, ValueError) as exc:
        _parser().error(str(exc))
    print(manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())