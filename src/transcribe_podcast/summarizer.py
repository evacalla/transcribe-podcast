from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI

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
        print(
            f"WARNING: Overwriting existing file {summary.output_path}",
            file=sys.stderr,
        )
    summary.output_path.write_text(
        f"# {summary.title}\n\n{summary.content}\n",
        encoding="utf-8",
    )


def build_llm(config: AppConfig) -> ChatOpenAI:
    return ChatOpenAI(
        model=config.model,
        api_key=config.api_key,
        base_url="https://openrouter.ai/api/v1",
    )


def summarise(transcription: Transcription, config: AppConfig) -> tuple[str, bool]:
    """Summarise the transcription, choosing single-pass or map-reduce as needed.

    Returns (summary_text, chunked).
    """
    if transcription.is_long:
        return _summarise_long(transcription, config), True
    return _summarise_short(transcription, config), False


def _summarise_short(transcription: Transcription, config: AppConfig) -> str:
    """Single LLM call for episodes under 1 hour."""
    llm = build_llm(config)
    prompt = (
        "You are an expert podcast summariser. "
        "Read the following transcript and write a clear, concise summary in flowing prose. "
        "Capture the main topics, key insights, and conclusions.\n\n"
        f"TRANSCRIPT:\n{transcription.text}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


def _summarise_long(transcription: Transcription, config: AppConfig) -> str:
    """Map-reduce summarisation for episodes over 1 hour."""
    from langchain.chains.summarize import load_summarize_chain
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=300)
    chunks = splitter.split_text(transcription.text)
    docs = [Document(page_content=chunk) for chunk in chunks]

    print(f"      Splitting into {len(docs)} chunks for map-reduce...")
    llm = build_llm(config)
    chain = load_summarize_chain(llm, chain_type="map_reduce")
    result = chain.invoke({"input_documents": docs})
    return result["output_text"]
