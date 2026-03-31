"""
Indexer — persists chunks to pgvector and builds the BM25 index.

Two indexes are maintained:
  1. pgvector (Postgres) — dense vector index for ANN search
  2. BM25 (in-memory, serialised to disk) — sparse keyword index

Both are written in a single pass over the chunks list.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Optional

import psycopg
from pgvector.psycopg import register_vector
from rank_bm25 import BM25Okapi

from config import settings
from logger import get_logger
from data_models.models import Chunk

log = get_logger(__name__)

_BM25_CACHE_PATH = Path(".bm25_index.pkl")


# ---------------------------------------------------------------------------
# BM25 index wrapper
# ---------------------------------------------------------------------------

class BM25Index:
    """
    Thin wrapper around rank_bm25.BM25Okapi that stores the original
    chunk_ids alongside the index so retrieval results map back to chunks.

    The index is serialised to disk after each write so it survives restarts.
    On startup it is loaded from disk if it exists, otherwise rebuilt from
    the database.
    """

    def __init__(self) -> None:
        self._index: Optional[BM25Okapi] = None
        self._chunk_ids: list[str] = []
        self._corpus: list[list[str]] = []  # tokenised documents

    @property
    def is_ready(self) -> bool:
        return self._index is not None and len(self._chunk_ids) > 0

    def add(self, chunks: list[Chunk]) -> None:
        """Add chunks to the in-memory corpus and rebuild the index."""
        for chunk in chunks:
            tokens = self._tokenize(chunk.text)
            self._corpus.append(tokens)
            self._chunk_ids.append(chunk.chunk_id)

        self._index = BM25Okapi(self._corpus)
        log.debug("BM25 index rebuilt: %d documents", len(self._corpus))

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        """
        Return (chunk_id, score) pairs for the top-k results.
        """
        if not self.is_ready:
            raise RuntimeError("BM25 index is empty. Run indexer.index() first.")

        tokens = self._tokenize(query)
        scores = self._index.get_scores(tokens)

        # argsort descending — numpy isn't imported to keep deps light
        ranked = sorted(
            enumerate(scores), key=lambda x: x[1], reverse=True
        )[:top_k]

        return [(self._chunk_ids[i], float(score)) for i, score in ranked]

    def remove_paper(self, paper_id: str) -> None:
        """Remove all chunks belonging to paper_id and rebuild."""
        pairs = [
            (cid, tokens)
            for cid, tokens in zip(self._chunk_ids, self._corpus)
            if not cid.startswith(paper_id)
        ]
        if not pairs:
            self._chunk_ids = []
            self._corpus = []
            self._index = None
            return
        self._chunk_ids, self._corpus = zip(*pairs)  # type: ignore[assignment]
        self._chunk_ids = list(self._chunk_ids)
        self._corpus = list(self._corpus)
        self._index = BM25Okapi(self._corpus)

    def save(self, path: Path = _BM25_CACHE_PATH) -> None:
        with open(path, "wb") as f:
            pickle.dump(
                {"chunk_ids": self._chunk_ids, "corpus": self._corpus}, f
            )
        log.debug("BM25 index saved to %s", path)

    def load(self, path: Path = _BM25_CACHE_PATH) -> bool:
        """Return True if successfully loaded, False if file not found."""
        if not path.exists():
            return False
        with open(path, "rb") as f:
            data = pickle.load(f)
        self._chunk_ids = data["chunk_ids"]
        self._corpus = data["corpus"]
        self._index = BM25Okapi(self._corpus)
        log.info("BM25 index loaded from %s (%d docs)", path, len(self._chunk_ids))
        return True

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        """Simple whitespace + lowercase tokenisation — consistent with BM25Okapi."""
        return text.lower().split()


# ---------------------------------------------------------------------------
# Indexer
# ---------------------------------------------------------------------------

_UPSERT = """
INSERT INTO {table}
    (chunk_id, paper_id, text, embedding, section, page, chunk_index,
     title, authors, year, doi, arxiv_id)
VALUES
    (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (chunk_id) DO UPDATE SET
    text        = EXCLUDED.text,
    embedding   = EXCLUDED.embedding,
    section     = EXCLUDED.section,
    page        = EXCLUDED.page,
    chunk_index = EXCLUDED.chunk_index,
    title       = EXCLUDED.title,
    authors     = EXCLUDED.authors,
    year        = EXCLUDED.year,
    doi         = EXCLUDED.doi,
    arxiv_id    = EXCLUDED.arxiv_id;
"""


class Indexer:
    """
    Writes embedded chunks to pgvector and the BM25 index.

    Instantiating this class opens a Postgres connection and ensures
    the schema exists. Use as a context manager for automatic cleanup:

        with Indexer() as idx:
            idx.index(chunks)
    """

    def __init__(self) -> None:
        self._conn = psycopg.connect(settings.database_url, autocommit=False)
        register_vector(self._conn)
        self._table = settings.pgvector_table
        self._dim = settings.embed_dim
        self._bm25 = BM25Index()
        self._bm25.load()  # load from disk if available

    def __enter__(self) -> "Indexer":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Commit on success, rollback on error."""
        if exc_type is not None:
            self._conn.rollback()
            log.warning("Transaction rolled back due to error: %s", exc_val)
        else:
            self._conn.commit()
            log.debug("Transaction committed.")
        self.close()

    def close(self) -> None:
        try:
            self._conn.close()
        except Exception:
            pass

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index(self, chunks: list[Chunk]) -> None:
        """
        Write chunks to pgvector and update the BM25 index.
        Chunks without embeddings are skipped with a warning.
        """
        ready = [c for c in chunks if c.embedding is not None]
        skipped = len(chunks) - len(ready)
        if skipped:
            log.warning(
                "%d chunks have no embedding and will be skipped.", skipped
            )

        self._write_pgvector(ready)
        self._bm25.add(ready)
        self._bm25.save()
        log.info("Indexed %d chunks.", len(ready))

    def delete_paper(self, paper_id: str) -> int:
        """
        Remove all chunks for a paper from both indexes.
        Returns the number of rows deleted from pgvector.
        """
        with self._conn.cursor() as cur:
            cur.execute(
                f"DELETE FROM {self._table} WHERE paper_id = %s", (paper_id,)
            )
            deleted = cur.rowcount
        self._bm25.remove_paper(paper_id)
        self._bm25.save()
        log.info("Deleted %d chunks for paper %r.", deleted, paper_id)
        return deleted

    def paper_exists(self, paper_id: str) -> bool:
        """Return True if any chunks for this paper_id are already indexed."""
        with self._conn.cursor() as cur:
            cur.execute(
                f"SELECT 1 FROM {self._table} WHERE paper_id = %s LIMIT 1",
                (paper_id,),
            )
            return cur.fetchone() is not None

    def save_paper_record(self, doc, file_path: str, chunk_count: int) -> None:
        """
        Insert or update the paper record in the papers table.
        This is part of the same transaction as chunk indexing.
        """
        from ingestion.pdf_parser import ParsedDocument  # Avoid circular import

        sql = """
            INSERT INTO papers (paper_id, title, authors, year, doi, arxiv_id, file_path, chunk_count)
            VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (paper_id) DO UPDATE SET
                title       = EXCLUDED.title,
                authors     = EXCLUDED.authors,
                year        = EXCLUDED.year,
                doi         = EXCLUDED.doi,
                arxiv_id    = EXCLUDED.arxiv_id,
                file_path   = EXCLUDED.file_path,
                chunk_count = EXCLUDED.chunk_count,
                ingested_at = now();
        """
        with self._conn.cursor() as cur:
            cur.execute(sql, (
                doc.paper_id,
                doc.title,
                json.dumps(doc.authors),
                doc.year,
                doc.doi,
                doc.arxiv_id,
                file_path,
                chunk_count,
            ))
        log.debug("Paper record saved for %s", doc.paper_id)

    def chunk_count(self) -> int:
        """Total number of chunks in pgvector."""
        with self._conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {self._table}")
            return cur.fetchone()[0]

    def get_bm25_index(self) -> BM25Index:
        """Return the BM25 index for use by the sparse retriever."""
        return self._bm25

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _write_pgvector(self, chunks: list[Chunk]) -> None:
        sql = _UPSERT.format(table=self._table)
        rows = [
            (
                c.chunk_id,
                c.paper_id,
                c.text,
                c.embedding,
                c.section,
                c.page,
                c.chunk_index,
                c.title,
                json.dumps(c.authors),
                c.year,
                c.doi,
                c.arxiv_id,
            )
            for c in chunks
        ]
        with self._conn.cursor() as cur:
            cur.executemany(sql, rows)
        log.debug("pgvector: upserted %d rows into %r", len(rows), self._table)