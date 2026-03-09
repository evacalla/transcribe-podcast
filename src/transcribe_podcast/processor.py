from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

from transcribe_podcast.config import AppConfig
from transcribe_podcast.summarizer import Summary, summarise, write_summary
from transcribe_podcast.transcriber import PodcastFile, Transcription, transcribe


@dataclass
class ProcessingResult:
    file: PodcastFile
    status: str  # "success" | "error"
    output_path: Path | None = field(default=None)
    transcription: Transcription | None = field(default=None)
    error_msg: str | None = field(default=None)


def process_file(podcast_file: PodcastFile, config: AppConfig) -> ProcessingResult:
    """Transcribe podcast, generate summary (optional), and save to output."""
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
        print(f"      [TIME] Transcribed in {elapsed:.1f}s")

        output_path = config.output_dir / (podcast_file.stem + ".md")

        if config.no_summary:
            # Skip LLM summarization, write transcription only
            content = f"# {podcast_file.stem}\n\n{transcription.text}"
            output_path.write_text(content, encoding="utf-8")
            chunked = False
        else:
            # Generate summary
            summary_content, chunked = summarise(transcription, config)
            if chunked:
                print("      Long episode detected, used chunked summarisation")

            summary = Summary(
                title=podcast_file.stem,
                content=summary_content,
                output_path=output_path,
                chunked=chunked,
            )
            write_summary(summary)

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
