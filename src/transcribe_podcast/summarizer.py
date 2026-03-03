from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from openai import OpenAI

from transcribe_podcast.config import AppConfig
from transcribe_podcast.transcriber import Transcription


@dataclass
class Summary:
    title: str
    content: str
    output_path: Path
    chunked: bool


def write_summary(summary: Summary) -> None:
    """Write summary to disk as a Markdown file."""
    if summary.output_path.exists():
        print(f"WARNING: Overwriting existing file {summary.output_path}", file=sys.stderr)
    summary.output_path.write_text(
        f"# {summary.title}\n\n{summary.content}\n",
        encoding="utf-8",
    )


def _build_client(config: AppConfig) -> OpenAI:
    return OpenAI(
        api_key=config.api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def _chat(client: OpenAI, model: str, prompt: str) -> str:
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
    )
    return response.choices[0].message.content


def _split_text(text: str, chunk_size: int = 4000, overlap: int = 300) -> list[str]:
    """Split text into chunks respecting paragraph and sentence boundaries."""
    separators = ["\n\n", "\n", ". ", " "]

    def _split(text: str, sep_idx: int) -> list[str]:
        if len(text) <= chunk_size or sep_idx >= len(separators):
            return [text]
        sep = separators[sep_idx]
        parts = text.split(sep)
        chunks: list[str] = []
        current = ""
        for part in parts:
            candidate = current + (sep if current else "") + part
            if len(candidate) > chunk_size and current:
                chunks.append(current)
                overlap_text = current[-overlap:] if len(current) > overlap else current
                current = overlap_text + (sep if overlap_text else "") + part
            else:
                current = candidate
        if current:
            chunks.append(current)
        return chunks

    return _split(text, 0)


def summarise(transcription: Transcription, config: AppConfig) -> tuple[str, bool]:
    """Summarise the transcription, choosing single-pass or map-reduce as needed."""
    if transcription.is_long:
        return _summarise_long(transcription, config), True
    return _summarise_short(transcription, config), False


def _summarise_short(transcription: Transcription, config: AppConfig) -> str:
    """Single LLM call for episodes under 1 hour."""
    client = _build_client(config)
    prompt = (
        "You are an expert podcast summariser. "
        "Read the following transcript and write a clear, concise summary in flowing prose. "
        "Capture the main topics, key insights, and conclusions.\n\n"
        f"TRANSCRIPT:\n{transcription.text}"
    )
    print("      Summarising...")
    return _chat(client, config.model, prompt)


def _summarise_long(transcription: Transcription, config: AppConfig) -> str:
    """Map-reduce summarisation for episodes over 1 hour."""
    chunks = _split_text(transcription.text)
    print(f"      Summarising {len(chunks)} chunks (map-reduce)...")
    client = _build_client(config)

    map_prompt = (
        "You are an expert podcast summariser. Read the following excerpt from a podcast "
        "transcript and write a concise summary of the key points discussed in this section. "
        "Focus on main topics, insights, and conclusions from this excerpt only.\n\n"
        "EXCERPT:\n{context}"
    )

    partial_summaries = []
    for i, chunk in enumerate(chunks, 1):
        print(f"      Chunk {i}/{len(chunks)}...")
        partial_summaries.append(_chat(client, config.model, map_prompt.format(context=chunk)))

    print("      Combining chunks...")
    combined_text = "\n\n".join(partial_summaries)
    reduce_prompt = (
        "You are an expert podcast summariser. The following are summaries of different "
        "sections of a podcast episode. Read them and write a single coherent summary "
        "in flowing prose. Capture the main topics, key insights, and conclusions from "
        "the entire episode.\n\n"
        f"PARTIAL SUMMARIES:\n{combined_text}"
    )
    return _chat(client, config.model, reduce_prompt)
