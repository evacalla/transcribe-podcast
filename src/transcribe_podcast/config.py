from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

VALID_WHISPER_MODELS = {"tiny", "base", "small", "medium", "large"}


@dataclass
class AppConfig:
    whisper_model: str
    language: str | None
    fp16: bool | None  # None = auto (fp16 on GPU, fp32 on CPU)
    input_dir: Path
    output_dir: Path
    json_output: bool
    api_key: str
    model: str
    no_summary: bool


def load_config(args) -> AppConfig:
    """Load and validate configuration from .env and CLI arguments.

    Precedence: CLI flag > environment variable > built-in default.
    Raises SystemExit(1) on invalid values.
    """
    load_dotenv()

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

    # language: CLI flag > env var > None (auto-detect)
    language = getattr(args, "language", None) or os.getenv("WHISPER_LANGUAGE") or None

    # fp16: CLI flag > env var > None (auto: fp16 on GPU, fp32 on CPU)
    fp16_arg = getattr(args, "fp16", None)
    fp16_env = os.getenv("WHISPER_FP16", "").strip().lower()
    if fp16_arg is not None:
        fp16: bool | None = fp16_arg
    elif fp16_env in ("true", "1", "yes"):
        fp16 = True
    elif fp16_env in ("false", "0", "no"):
        fp16 = False
    else:
        fp16 = None

    json_output = bool(getattr(args, "json", False))

    no_summary = bool(getattr(args, "no_summary", False))

    api_key = getattr(args, "api_key", None) or os.getenv("OPENROUTER_API_KEY", "")
    model = getattr(args, "model", None) or os.getenv("OPENROUTER_MODEL", "")

    if not no_summary:
        if not api_key:
            print("ERROR: OPENROUTER_API_KEY is required (or use --no-summary).", file=sys.stderr)
            sys.exit(1)
        if not model:
            print("ERROR: OPENROUTER_MODEL is required (or use --no-summary).", file=sys.stderr)
            sys.exit(1)

    return AppConfig(
        whisper_model=whisper_model,
        language=language,
        fp16=fp16,
        input_dir=input_dir,
        output_dir=output_dir,
        json_output=json_output,
        api_key=api_key,
        model=model,
        no_summary=no_summary,
    )
