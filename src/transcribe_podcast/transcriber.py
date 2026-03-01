from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class PodcastFile:
    path: Path

    @property
    def stem(self) -> str:
        return self.path.stem


@dataclass
class Transcription:
    source: PodcastFile
    text: str
    duration_s: float

    @property
    def is_long(self) -> bool:
        return self.duration_s >= 3600


def discover_files(input_dir: Path) -> list[PodcastFile]:
    """Return all .mp3 files in input_dir sorted by name.

    Raises SystemExit(1) if input_dir does not exist.
    """
    if not input_dir.exists():
        print(f"ERROR: Input directory '{input_dir}' does not exist.", file=sys.stderr)
        sys.exit(1)

    files = sorted(input_dir.glob("*.mp3")) + sorted(input_dir.glob("*.MP3"))
    # Deduplicate while preserving order (case-insensitive filesystems may return both)
    seen: set[Path] = set()
    unique: list[PodcastFile] = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(PodcastFile(path=f))
    return unique


def transcribe(podcast_file: PodcastFile, whisper_model: str) -> Transcription:
    """Transcribe an MP3 file using a local Whisper model."""
    import whisper  # imported here to allow tests to mock easily

    print(f"      Loading Whisper model '{whisper_model}'...")
    model = whisper.load_model(whisper_model)
    print("      Transcribing audio...")
    result = model.transcribe(str(podcast_file.path))

    text: str = result["text"]
    segments = result.get("segments", [])
    duration_s: float = segments[-1]["end"] if segments else 0.0

    return Transcription(source=podcast_file, text=text, duration_s=duration_s)
