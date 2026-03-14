"""
Semantic sliding window chunker for academic papers.

Chunks are produced within section boundaries — a chunk never spans two
sections. This preserves the logical structure of a paper and prevents
the retriever from surfacing a chunk that mixes Introduction text with
Methods text.

Tokenisation uses tiktoken (cl100k_base) for accurate token counting that
is consistent with OpenAI embedding models. For HuggingFace models the
counts are approximate but close enough for chunking purposes.

Usage:
    chunker = Chunker()
    chunks = chunker.chunk_document(parsed_doc)
"""

from __future__ import annotations

import hashlib
import re
from typing import Iterator

import tiktoken

from config import settings
from ingestion.pdf_parser import ParsedDocument, Section
from logger import get_logger
from models import Chunk

log = get_logger(__name__)

# Use cl100k_base — compatible with OpenAI embedding and GPT models.
# For local models this slightly overcounts but is consistent and fast.
_TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _token_count(text: str) -> int:
    return len(_TOKENIZER.encode(text, disallowed_special=()))


def _chunk_id(paper_id: str, section_title: str, index: int) -> str:
    """Stable, URL-safe chunk identifier."""
    slug = re.sub(r"[^a-z0-9]+", "_", section_title.lower()).strip("_")
    return f"{paper_id}__{slug}__{index:04d}"


def _sentence_split(text: str) -> list[str]:
    """
    Split text into sentences while preserving whitespace context.
    Uses a simple regex that handles common academic abbreviations
    (e.g. "et al.", "Fig.", "Eq.") without false splits.
    """
    # Protect common abbreviations from sentence splitting
    protected = re.sub(
        r"\b(et al|fig|eq|sec|cf|e\.g|i\.e|vs|approx|ref|др)\.",
        lambda m: m.group(0).replace(".", "<DOT>"),
        text,
        flags=re.IGNORECASE,
    )
    # Split on sentence-ending punctuation followed by whitespace + capital
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z\[\(])", protected)
    # Restore protected dots
    return [s.replace("<DOT>", ".") for s in sentences if s.strip()]


class Chunker:
    """
    Sliding window chunker that operates within section boundaries.

    Parameters (read from settings, overridable for testing):
        chunk_size    — target chunk size in tokens (default 512)
        chunk_overlap — overlap between consecutive chunks in tokens (default 128)
    """

    def __init__(
        self,
        chunk_size: int | None = None,
        chunk_overlap: int | None = None,
    ) -> None:
        self.chunk_size = chunk_size or settings.chunk_size
        self.chunk_overlap = chunk_overlap or settings.chunk_overlap

        if self.chunk_overlap >= self.chunk_size:
            raise ValueError(
                f"chunk_overlap ({self.chunk_overlap}) must be less than "
                f"chunk_size ({self.chunk_size})"
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def chunk_document(self, doc: ParsedDocument) -> list[Chunk]:
        """
        Chunk a fully parsed document into Chunk objects ready for embedding.
        Skips the references section — embedding reference strings adds noise.
        """
        skip_sections = {"references", "bibliography"}
        chunks: list[Chunk] = []

        for section in doc.sections:
            if section.title.lower() in skip_sections:
                continue
            section_chunks = list(self._chunk_section(section, doc))
            chunks.extend(section_chunks)

        log.info(
            "Chunked %s into %d chunks across %d sections",
            doc.paper_id, len(chunks), len(doc.sections),
        )
        return chunks

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _chunk_section(
        self, section: Section, doc: ParsedDocument
    ) -> Iterator[Chunk]:
        """
        Yield Chunk objects for a single section using a sliding window
        over sentences.

        Strategy:
        1. Split section text into sentences.
        2. Greedily accumulate sentences until the token budget is reached.
        3. When a window is full, emit a Chunk and step forward by
           (chunk_size - chunk_overlap) tokens worth of sentences.
        """
        sentences = _sentence_split(section.text)
        if not sentences:
            return

        # Pre-compute token counts once per sentence
        token_counts = [_token_count(s) for s in sentences]

        window_start = 0   # index into sentences[]
        chunk_index = 0

        while window_start < len(sentences):
            window_tokens = 0
            window_end = window_start

            # Expand window until we hit chunk_size
            while window_end < len(sentences):
                next_tokens = token_counts[window_end]
                # A single sentence longer than chunk_size gets its own chunk
                if window_tokens + next_tokens > self.chunk_size and window_end > window_start:
                    break
                window_tokens += next_tokens
                window_end += 1

            chunk_text = " ".join(sentences[window_start:window_end]).strip()
            if chunk_text:
                yield Chunk(
                    chunk_id=_chunk_id(doc.paper_id, section.title, chunk_index),
                    paper_id=doc.paper_id,
                    text=chunk_text,
                    section=section.title,
                    page=section.page_start,
                    chunk_index=chunk_index,
                    title=doc.title,
                    authors=list(doc.authors),
                    year=doc.year,
                    doi=doc.doi,
                    arxiv_id=doc.arxiv_id,
                )
                chunk_index += 1

            # Step forward by (chunk_size - chunk_overlap) tokens
            step_tokens = 0
            step_end = window_start
            target_step = self.chunk_size - self.chunk_overlap
            while step_end < window_end:
                step_tokens += token_counts[step_end]
                step_end += 1
                if step_tokens >= target_step:
                    break

            # Always advance by at least one sentence to avoid infinite loops
            window_start = max(window_start + 1, step_end)

    def chunk_text(
        self,
        text: str,
        paper_id: str = "unknown",
        section: str = "unknown",
        **metadata,
    ) -> list[Chunk]:
        """
        Convenience method — chunk a raw string without a ParsedDocument.
        Useful for testing and one-off ingestion.
        """
        fake_section = Section(
            title=section,
            text=text,
            page_start=0,
            page_end=0,
            section_index=0,
        )
        fake_doc = ParsedDocument(paper_id=paper_id, file_path="")
        fake_doc.sections = [fake_section]
        for k, v in metadata.items():
            if hasattr(fake_doc, k):
                setattr(fake_doc, k, v)
        return list(self._chunk_section(fake_section, fake_doc))


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(description="Chunk a parsed PDF and print stats.")
    ap.add_argument("--input", required=True, help="Path to PDF file")
    ap.add_argument("--paper-id", default=None)
    ap.add_argument(
        "--chunk-size", type=int, default=settings.chunk_size
    )
    ap.add_argument(
        "--chunk-overlap", type=int, default=settings.chunk_overlap
    )
    args = ap.parse_args()

    from logger import configure_logging
    configure_logging()

    from ingestion.pdf_parser import PDFParser

    doc = PDFParser().parse(args.input, paper_id=args.paper_id)
    chunks = Chunker(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
    ).chunk_document(doc)

    print(f"Total chunks : {len(chunks)}")
    print(f"Avg tokens   : {sum(_token_count(c.text) for c in chunks) // len(chunks)}")
    print()
    for c in chunks[:5]:
        print(f"  {c.chunk_id}")
        print(f"  Section : {c.section}")
        print(f"  Tokens  : {_token_count(c.text)}")
        print(f"  Preview : {c.text[:100]!r}")
        print()
