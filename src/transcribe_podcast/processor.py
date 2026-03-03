from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from transcribe_podcast.config import AppConfig
from transcribe_podcast.transcriber import PodcastFile, Transcription, transcribe


@dataclass
class ProcessingResult:
    file: PodcastFile
    status: str  # "success" | "error"
    output_path: Path | None = field(default=None)
    transcription: Transcription | None = field(default=None)
    error_msg: str | None = field(default=None)


def _write_raw_transcription(transcription: Transcription, output_path: Path) -> None:
    """Write raw transcription text to a markdown file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(transcription.text, encoding="utf-8")


def process_file(podcast_file: PodcastFile, config: AppConfig) -> ProcessingResult:
    """Transcribe podcast and save raw text output."""
    try:
        t0 = time.perf_counter()
        transcription = transcribe(
            podcast_file,
            config.whisper_model,
            config.language,
            config.fp16,
        )
        elapsed = time.perf_counter() - t0

        duration_min = transcription.duration_s / 60
        print(f"      [DURATION] Episode duration: {duration_min:.1f} minutes")
        print(f"      [TIME] Processed in {elapsed:.1f}s")

        output_path = config.output_dir / (podcast_file.stem + ".md")
        _write_raw_transcription(transcription, output_path)

        return ProcessingResult(
            file=podcast_file,
            status="success",
            output_path=output_path,
            transcription=transcription,
        )
    except Exception as exc:
        print(f"      [ERROR] {exc}")
        return ProcessingResult(
            file=podcast_file,
            status="error",
            error_msg=str(exc),
        )
