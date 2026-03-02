from __future__ import annotations

from dataclasses import dataclass, field

from transcribe_podcast.config import AppConfig
from transcribe_podcast.summarizer import Summary, write_summary
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
        transcription = transcribe(podcast_file, config.whisper_model, config.language, config.fp16)
        summary = Summary(
            title=podcast_file.stem,
            content=transcription.text,
            output_path=config.output_dir / (podcast_file.stem + ".md"),
            chunked=False,
        )
        write_summary(summary)
        return ProcessingResult(
            file=podcast_file,
            status="success",
            summary=summary,
            transcription=transcription,
        )
    except Exception as exc:
        return ProcessingResult(
            file=podcast_file,
            status="error",
            error_msg=str(exc),
        )
