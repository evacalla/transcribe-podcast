from __future__ import annotations

import argparse
import json
import sys

from transcribe_podcast.config import load_config
from transcribe_podcast.processor import ProcessingResult, process_file
from transcribe_podcast.transcriber import discover_files


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="transcribe-podcast",
        description="Transcribe and summarise MP3 podcasts using Whisper and OpenRouter.",
    )
    parser.add_argument(
        "--input-dir",
        dest="input_dir",
        metavar="PATH",
        help="Directory to scan for .mp3 files (default: ./input)",
    )
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        metavar="PATH",
        help="Directory to write summary .md files (default: ./output)",
    )
    parser.add_argument(
        "--whisper-model",
        dest="whisper_model",
        metavar="MODEL",
        help="Whisper model size: tiny, base, small, medium, large (default: base)",
    )
    parser.add_argument(
        "--language",
        dest="language",
        metavar="LANG",
        help="Language code for transcription, e.g. 'es', 'en' (default: auto-detect)",
    )
    parser.add_argument(
        "--fp16",
        dest="fp16",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Use fp16 precision for Whisper. --no-fp16 forces fp32 (default: auto)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON to stdout",
    )
    return parser


def _print(msg: str, silent: bool) -> None:
    if not silent:
        print(msg)


def main() -> None:
    args = build_parser().parse_args()
    config = load_config(args)
    silent = config.json_output

    # Discover files
    if not config.input_dir.exists():
        print(
            f"ERROR: Input directory '{config.input_dir}' does not exist.",
            file=sys.stderr,
        )
        sys.exit(1)

    files = discover_files(config.input_dir)

    if not silent:
        print("[INIT] Starting podcast transcription batch")
        print(f"[CONFIG] Input: {config.input_dir}")
        print(f"[CONFIG] Output: {config.output_dir}")
        print(f"[CONFIG] LLM Model: {config.model}")
        print(f"[CONFIG] Whisper Model: {config.whisper_model}")

    if not files:
        _print(f"No .mp3 files found in {config.input_dir}.", silent)
        sys.exit(0)

    total = len(files)
    results: list[ProcessingResult] = []

    for i, podcast_file in enumerate(files, start=1):
        _print(f"[{i}/{total}] Processing: {podcast_file.path.name}", silent)

        result = process_file(podcast_file, config)
        results.append(result)

        if result.status == "success":
            assert result.transcription is not None
            assert result.summary is not None
            if result.transcription.is_long:
                _print("      Long episode detected — splitting into chunks", silent)
            _print(f"      Saved: {result.summary.output_path}", silent)
        else:
            _print(f"      ERROR: {result.error_msg}", silent)

    succeeded = sum(1 for r in results if r.status == "success")
    failed = total - succeeded

    if silent:
        # JSON output mode
        output = {
            "total": total,
            "succeeded": succeeded,
            "failed": failed,
            "results": [
                (
                    {
                        "file": r.file.path.name,
                        "status": "success",
                        "output": str(r.summary.output_path),
                        "chunked": r.summary.chunked,
                    }
                    if r.status == "success"
                    else {
                        "file": r.file.path.name,
                        "status": "error",
                        "error": r.error_msg,
                    }
                )
                for r in results
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\n[DONE] Batch complete: {succeeded} succeeded, {failed} failed.")

    sys.exit(0 if failed == 0 else 2)
