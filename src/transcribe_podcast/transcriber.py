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


def transcribe(podcast_file: PodcastFile, whisper_model: str, language: str | None = None, fp16: bool | None = None) -> Transcription:
    """Transcribe an MP3 file using a local Whisper model."""
    import os
    import ssl
    import tempfile
    import imageio_ffmpeg  # bundled ffmpeg binary, no system install needed
    import whisper  # imported here to allow tests to mock easily

    # imageio_ffmpeg bundles ffmpeg but with a versioned name (e.g. ffmpeg-macos-aarch64-v7.1).
    # Whisper looks for "ffmpeg" specifically, so we create a symlink with that name in a
    # temporary directory and add it to PATH for the duration of the call.
    ffmpeg_exe = Path(imageio_ffmpeg.get_ffmpeg_exe())
    tmp_bin = Path(tempfile.mkdtemp())
    ffmpeg_link = tmp_bin / "ffmpeg"
    if not ffmpeg_link.exists():
        ffmpeg_link.symlink_to(ffmpeg_exe)
    os.environ["PATH"] = str(tmp_bin) + os.pathsep + os.environ.get("PATH", "")

    import torch

    if torch.backends.mps.is_available():
        device = "mps"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    print(f"      Loading Whisper model '{whisper_model}' on {device}...")
    # Some corporate networks use SSL inspection proxies with self-signed certs.
    # Disable verification only for the model download; whisper uses urllib internally.
    _orig_ctx = ssl._create_default_https_context
    ssl._create_default_https_context = ssl._create_unverified_context
    try:
        model = whisper.load_model(whisper_model, device=device)
    finally:
        ssl._create_default_https_context = _orig_ctx
    print("      Transcribing audio...")
    use_fp16 = fp16 if fp16 is not None else (device != "cpu")
    result = model.transcribe(str(podcast_file.path), fp16=use_fp16, language=language)

    text: str = result["text"]
    segments = result.get("segments", [])
    duration_s: float = segments[-1]["end"] if segments else 0.0

    return Transcription(source=podcast_file, text=text, duration_s=duration_s)
