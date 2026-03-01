from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

VALID_WHISPER_MODELS = {"tiny", "base", "small", "medium", "large"}


@dataclass
class AppConfig:
    api_key: str
    model: str
    whisper_model: str
    input_dir: Path
    output_dir: Path
    json_output: bool


def load_config(args) -> AppConfig:
    """Load and validate configuration from .env and CLI arguments.

    Precedence: CLI flag > environment variable > built-in default.
    Raises SystemExit(1) on missing required credentials or invalid values.
    """
    load_dotenv()

    api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY is not set. Add it to your .env file.", file=sys.stderr)
        sys.exit(1)

    model = os.getenv("OPENROUTER_MODEL", "").strip()
    if not model:
        print("ERROR: OPENROUTER_MODEL is not set. Add it to your .env file.", file=sys.stderr)
        sys.exit(1)

    # whisper_model: CLI flag > env var > default
    whisper_model = getattr(args, "whisper_model", None) or os.getenv("WHISPER_MODEL", "") or "base"
    if whisper_model not in VALID_WHISPER_MODELS:
        print(
            f"ERROR: Invalid whisper model '{whisper_model}'. "
            f"Choose one of: {', '.join(sorted(VALID_WHISPER_MODELS))}",
            file=sys.stderr,
        )
        sys.exit(1)

    # input_dir: CLI flag > env var > default
    input_dir_str = getattr(args, "input_dir", None) or os.getenv("INPUT_DIR", "") or "./input"
    input_dir = Path(input_dir_str).resolve()

    # output_dir: CLI flag > env var > default
    output_dir_str = getattr(args, "output_dir", None) or os.getenv("OUTPUT_DIR", "") or "./output"
    output_dir = Path(output_dir_str).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    json_output = bool(getattr(args, "json", False))

    return AppConfig(
        api_key=api_key,
        model=model,
        whisper_model=whisper_model,
        input_dir=input_dir,
        output_dir=output_dir,
        json_output=json_output,
    )
