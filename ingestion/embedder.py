"""
Embedding client for ScholarRAG.

Supports three backends controlled by MODEL_BACKEND:
  qwen    — Qwen models hosted on vLLM server (local, default)
  openai  — OpenAI Embeddings API (hosted)
  gemini  — Google Gemini Embedding API (hosted)

Note: The Qwen backend requires Qwen models to be hosted on vLLM server.
The vLLM inference engine provides the OpenAI-compatible API layer,
but only Qwen family models are supported for ScholarRAG.

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

from config import ModelBackend, settings
from logger import get_logger
from data_models.models import Chunk

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
# Qwen on vLLM (local)
# ---------------------------------------------------------------------------

class QWENEmbedder(BaseEmbedder):
    """
    Local embedding via Qwen models hosted on vLLM server.
    Requires vLLM server running locally with a Qwen pooling model loaded.
    Uses OpenAI-compatible API: /v1/embeddings
    See: https://docs.vllm.ai/en/stable/models/pooling_models/
    """

    def __init__(self) -> None:
        import httpx

        self._client = httpx.Client(base_url=settings.qwen_url, timeout=120)
        self._model = settings.qwen_embed_model
        self._dim = settings.embed_dim
        log.info(
            "Qwen embedder ready (model=%s, url=%s)",
            self._model, settings.qwen_url,
        )

    def embed(self, texts: list[str]) -> list[list[float]]:
        # Qwen on vLLM OpenAI-compatible embeddings API
        resp = self._client.post(
            "/v1/embeddings",
            json={
                "model": self._model,
                "input": texts,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        # vLLM returns {"data": [{"embedding": [...], "index": 0}, ...]}
        return [item["embedding"] for item in data["data"]]

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
    Supports text-embedding-3-large and text-embedding-3-small.
    """

    def __init__(self) -> None:
        from openai import OpenAI

        self._client = OpenAI(api_key=settings.openai_api_key)
        self._model = settings.openai_embed_model
        self._dim = settings.openai_embed_dim
        log.info("OpenAI embedder ready (model=%s, dim=%d)", self._model, self._dim)

    def embed(self, texts: list[str]) -> list[list[float]]:
        response = self._client.embeddings.create(
            input=texts,
            model=self._model,
            dimensions=self._dim,
        )
        return [item.embedding for item in response.data]

    @property
    def dimension(self) -> int:
        return self._dim


# ---------------------------------------------------------------------------
# Gemini (hosted)
# ---------------------------------------------------------------------------

class GeminiEmbedder(BaseEmbedder):
    """
    Hosted embedding via Google Gemini Embedding API.
    Requires GEMINI_API_KEY in .env.
    Supports gemini-embedding-001 and gemini-embedding-002.
    """

    def __init__(self) -> None:
        from google import genai

        self._client = genai.Client(api_key=settings.gemini_api_key)
        self._model = settings.gemini_embed_model
        # 1536 dimensions preserves compatibility between gemini-embedding-001 and gemini-embedding-002
        self._dim = 1536
        log.info("Gemini embedder ready (model=%s, dim=%d)", self._model, self._dim)

    def embed(self, texts: list[str]) -> list[list[float]]:
        # 1536 dimensions preserves compatibility between gemini-embedding-001 and gemini-embedding-002
        result = self._client.models.embed_content(
            model=self._model,
            contents=texts,
            config={"output_dimensionality": 1536},
        )
        return [emb.values for emb in result.embeddings]

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
    backend = settings.model_backend
    log.info("Initialising embedder: backend=%s", backend)

    if backend == ModelBackend.QWEN:
        return QWENEmbedder()
    if backend == ModelBackend.OPENAI:
        return OpenAIEmbedder()
    if backend == ModelBackend.GEMINI:
        return GeminiEmbedder()

    raise ValueError(f"Unknown MODEL_BACKEND: {backend!r}")



