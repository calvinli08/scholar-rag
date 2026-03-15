"""
Embedding client for ScholarRAG.

Supports four backends controlled by EMBED_BACKEND:
  hf      — HuggingFace Sentence Transformers (local, default)
  ollama  — Ollama embedding endpoint (local, OpenAI-compatible)
  openai  — OpenAI Embeddings API (hosted)
  cohere  — Cohere Embed API (hosted)

All backends share the same interface: embed(texts) → list[list[float]].
Batch size, retry logic, and progress logging are handled uniformly
regardless of backend.

Usage:
    embedder = get_embedder()
    chunks_with_embeddings = embedder.embed_chunks(chunks)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Protocol

from config import EmbedBackend, settings
from logger import get_logger
from models import Chunk

log = get_logger(__name__)

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseEmbedder(ABC):
    """Common interface all embedding backends must implement."""

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of texts.
        Returns a list of float vectors, one per input text.
        """
        ...

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension."""
        ...

    def embed_chunks(self, chunks: list[Chunk]) -> list[Chunk]:
        """
        Embed a list of Chunk objects in batches.
        Mutates each chunk in-place, setting chunk.embedding.
        Returns the same list for chaining.
        """
        total = len(chunks)
        log.info("Embedding %d chunks (batch_size=%d)", total, settings.embed_batch_size)

        for batch_start in range(0, total, settings.embed_batch_size):
            batch = chunks[batch_start : batch_start + settings.embed_batch_size]
            texts = [c.text for c in batch]

            embeddings = self._embed_with_retry(texts)

            for chunk, vector in zip(batch, embeddings):
                chunk.embedding = vector

            log.debug(
                "Embedded batch %d–%d / %d",
                batch_start + 1,
                min(batch_start + settings.embed_batch_size, total),
                total,
            )

        return chunks

    def _embed_with_retry(
        self,
        texts: list[str],
        max_retries: int = 3,
        backoff: float = 2.0,
    ) -> list[list[float]]:
        """Retry wrapper around embed() with exponential backoff."""
        for attempt in range(1, max_retries + 1):
            try:
                return self.embed(texts)
            except Exception as exc:
                if attempt == max_retries:
                    raise
                wait = backoff ** attempt
                log.warning(
                    "Embedding attempt %d/%d failed (%s). Retrying in %.1fs.",
                    attempt, max_retries, exc, wait,
                )
                time.sleep(wait)
        raise RuntimeError("Unreachable")  # satisfy type checker


# ---------------------------------------------------------------------------
# HuggingFace (local)
# ---------------------------------------------------------------------------

class HFEmbedder(BaseEmbedder):
    """
    Local embedding via HuggingFace Sentence Transformers.
    Any model on the HuggingFace hub that is compatible with
    SentenceTransformer() will work — set HF_EMBED_MODEL in .env.
    """

    def __init__(self) -> None:
        # Defer heavy import so other backends don't pay the torch load cost
        from sentence_transformers import SentenceTransformer

        log.info("Loading HF embedding model: %s", settings.hf_embed_model)
        self._model = SentenceTransformer(settings.hf_embed_model)
        self._dim: int = self._model.get_sentence_embedding_dimension()
        log.info("HF embedder ready (dim=%d)", self._dim)

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=settings.embed_batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        return vectors.tolist()

    @property
    def dimension(self) -> int:
        return self._dim


# ---------------------------------------------------------------------------
# Ollama (local)
# ---------------------------------------------------------------------------

class OllamaEmbedder(BaseEmbedder):
    """
    Local embedding via Ollama's OpenAI-compatible /api/embeddings endpoint.
    Requires Ollama running locally and the model pulled:
      ollama pull nomic-embed-text
    """

    def __init__(self) -> None:
        import httpx

        self._client = httpx.Client(base_url=settings.ollama_embed_url, timeout=60)
        self._model = settings.ollama_embed_model
        self._dim = settings.embed_dim  # Must be set correctly in .env for Ollama models
        log.info(
            "Ollama embedder ready (model=%s, url=%s)",
            self._model, settings.ollama_embed_url,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for text in texts:
            resp = self._client.post(
                "/api/embeddings",
                json={"model": self._model, "prompt": text},
            )
            resp.raise_for_status()
            vectors.append(resp.json()["embedding"])
        return vectors

    @property
    def dimension(self) -> int:
        return self._dim


# ---------------------------------------------------------------------------
# OpenAI (hosted)
# ---------------------------------------------------------------------------

class OpenAIEmbedder(BaseEmbedder):
    """
    Hosted embedding via OpenAI Embeddings API.
    Requires OPENAI_API_KEY in .env.
    """

    def __init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_embed_model
        # Dimensions for known OpenAI models
        _dims = {
            "text-embedding-3-large": 3072,
            "text-embedding-3-small": 1536,
            "text-embedding-ada-002": 1536,
        }
        self._dim = _dims.get(self._model, settings.embed_dim)
        log.info("OpenAI embedder ready (model=%s, dim=%d)", self._model, self._dim)

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            input=texts,
            model=self._model,
        )
        return [item.embedding for item in response.data]

    @property
    def dimension(self) -> int:
        return self._dim


# ---------------------------------------------------------------------------
# Cohere (hosted)
# ---------------------------------------------------------------------------

class CohereEmbedder(BaseEmbedder):
    """
    Hosted embedding via Cohere Embed API.
    Requires COHERE_API_KEY in .env.
    """

    def __init__(self) -> None:
        import cohere

        self._client = cohere.Client(api_key=settings.cohere_api_key)
        self._model = settings.cohere_embed_model
        self._dim = settings.embed_dim
        log.info("Cohere embedder ready (model=%s)", self._model)

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embed(
            texts=texts,
            model=self._model,
            input_type="search_document",
        )
        return [list(v) for v in response.embeddings]

    @property
    def dimension(self) -> int:
        return self._dim


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_embedder() -> BaseEmbedder:
    """
    Return the configured embedding backend.
    Call once at startup and reuse the instance — model loading is expensive.
    """
    backend = settings.embed_backend
    log.info("Initialising embedder: backend=%s", backend)

    if backend == EmbedBackend.HF:
        return HFEmbedder()
    if backend == EmbedBackend.OLLAMA:
        return OllamaEmbedder()
    if backend == EmbedBackend.OPENAI:
        return OpenAIEmbedder()
    if backend == EmbedBackend.COHERE:
        return CohereEmbedder()

    raise ValueError(f"Unknown EMBED_BACKEND: {backend!r}")



