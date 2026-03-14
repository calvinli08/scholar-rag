"""
Shared data models for ScholarRAG.
These are the core types that flow through the entire pipeline.
Import from here — never redefine these in individual modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Chunk:
    """A single text chunk produced by the ingestion pipeline."""

    # Identity
    chunk_id: str                    # Unique ID: f"{paper_id}_{section}_{index}"
    paper_id: str                    # Stable identifier for the source paper (e.g. arXiv ID)

    # Content
    text: str                        # Raw chunk text
    embedding: Optional[list[float]] = None  # Set after embedding pass

    # Provenance metadata
    section: str = ""                # e.g. "Introduction", "Methods", "Abstract"
    page: int = 0                    # 0-indexed page number in the original PDF
    chunk_index: int = 0             # Position of this chunk within its section

    # Paper-level metadata
    title: str = ""
    authors: list[str] = field(default_factory=list)
    year: Optional[int] = None
    doi: str = ""
    arxiv_id: str = ""

    def __repr__(self) -> str:
        preview = self.text[:80].replace("\n", " ")
        return f"Chunk({self.chunk_id!r}, page={self.page}, text={preview!r}...)"


@dataclass
class RetrievedChunk:
    """A Chunk augmented with retrieval scores, returned by the retrieval pipeline."""

    chunk: Chunk
    score: float = 0.0              # Final score after RRF / reranking
    dense_rank: Optional[int] = None
    sparse_rank: Optional[int] = None
    rerank_score: Optional[float] = None

    @property
    def chunk_id(self) -> str:
        return self.chunk.chunk_id

    @property
    def text(self) -> str:
        return self.chunk.text


@dataclass
class QueryResult:
    """Final output of the full RAG pipeline for a single query."""

    query: str
    answer: str
    sources: list[RetrievedChunk] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens
