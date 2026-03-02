from __future__ import annotations

import time
from dataclasses import dataclass, field

from transcribe_podcast.config import AppConfig
from transcribe_podcast.summarizer import Summary, summarise, write_summary
from transcribe_podcast.transcriber import PodcastFile, Transcription, transcribe


@dataclass
class ProcessingResult:
    file: PodcastFile
    status: str  # "success" | "error"
    summary: Summary | None = field(default=None)
    transcription: Transcription | None = field(default=None)
    error_msg: str | None = field(default=None)


def process_file(podcast_file: PodcastFile, config: AppConfig) -> ProcessingResult:
    """Orchestrate transcription → summarisation → file write for one podcast."""
    try:
        print(f"      [START] Processing file: {podcast_file.path.name}")
        print(f"      [MODEL] Using LLM model: {config.model}")
        print(f"      [WHISPER] Using Whisper model: {config.whisper_model}")

        transcription = transcribe(podcast_file, config.whisper_model, config.language, config.fp16)
        duration_min = transcription.duration_s / 60
        print(f"      [DURATION] Episode duration: {duration_min:.1f} minutes")

        print("      [SUMMARY] Generating summary...")
        summary_text, chunked = summarise(transcription, config)
        print(f"      [SUMMARY] Summary generated (chunked: {chunked})")

        summary = Summary(
            title=podcast_file.stem,
            content=summary_text,
            output_path=config.output_dir / (podcast_file.stem + ".md"),
            chunked=chunked,
        )
        write_summary(summary)
        print(f"      [DONE] Summary saved to: {summary.output_path}")

        # Delay based on episode duration (1 second per minute of audio)
        delay_seconds = int(duration_min)
        if delay_seconds > 0:
            print(f"      [DELAY] Waiting {delay_seconds} seconds ({duration_min:.1f} min)...")
            time.sleep(delay_seconds)

        return ProcessingResult(
            file=podcast_file,
            status="success",
            summary=summary,
            transcription=transcription,
        )
    except Exception as exc:
        print(f"      [ERROR] {exc}")
        return ProcessingResult(
            file=podcast_file,
            status="error",
            error_msg=str(exc),
        )
