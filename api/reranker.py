"""
Reranking module for retrieved chunks.
Supports multiple backends: vLLM (local), OpenAI, Gemini API.
"""

from __future__ import annotations

from functools import lru_cache

from config import settings, ModelBackend
from logger import get_logger
from data_models.models import RetrievedChunk

log = get_logger(__name__)

async def rerank_chunks(
    query: str,
    chunks: list[RetrievedChunk],
    top_k: int | None = None,
) -> list[RetrievedChunk]:
    """
    Rerank retrieved chunks using a cross-encoder model.

    Args:
        query: Original query string
        chunks: List of RetrievedChunk to rerank
        top_k: Number of chunks to keep after reranking (default: from settings)

    Returns:
        List of RetrievedChunk sorted by reranker score
    """
    if not chunks:
        return []

    backend = settings.model_backend

    if backend == ModelBackend.VLLM:
        import httpx

        # vLLM rerank API (OpenAI-compatible)
        response = httpx.post(
            f"{settings.vllm_url}/v1/rerank",
            json={
                "model": settings.vllm_reranker_model,
                "query": query,
                "documents": [chunk.text for chunk in chunks],
                "top_n": len(chunks),
            },
            timeout=60,
        )
        response.raise_for_status()
        data = response.json()

        # vLLM returns {"results": [{"index": 0, "relevance_score": 0.9}, ...]}
        scores = [0.0] * len(chunks)
        for result in data["results"]:
            scores[result["index"]] = result["relevance_score"]

    elif backend == ModelBackend.OPENAI:
        from openai import OpenAI
        import numpy as np

        client = OpenAI(api_key=settings.openai_api_key)

        # Get embeddings for query and all documents
        # OpenAI embeddings are normalized, so cosine similarity = dot product
        query_result = client.embeddings.create(
            model=settings.openai_reranker_model,
            input=query,
            dimensions=settings.openai_embed_dim,
        )
        query_embedding = np.array(query_result.data[0].embedding)

        # Embed all documents in batch
        doc_texts = [chunk.text for chunk in chunks]
        doc_result = client.embeddings.create(
            model=settings.openai_reranker_model,
            input=doc_texts,
            dimensions=settings.openai_embed_dim,
        )
        doc_embeddings = [np.array(emb.embedding) for emb in doc_result.data]

        # Calculate cosine similarity between query and each document
        # Since OpenAI embeddings are normalized, dot product equals cosine similarity
        scores = []
        for doc_emb in doc_embeddings:
            similarity = float(np.dot(query_embedding, doc_emb))
            scores.append(similarity)

    elif backend == ModelBackend.GEMINI:
        from google import genai
        import numpy as np

        client = genai.Client(api_key=settings.gemini_api_key)

        # Get embeddings for query and all documents
        # 1536 dimensions preserves compatibility between gemini-embedding-001 and gemini-embedding-002
        query_result = client.models.embed_content(
            model=settings.gemini_embed_model,
            contents=query,
            config={"output_dimensionality": 1536},
        )
        query_embedding = np.array(query_result.embeddings[0].values)

        # Embed all documents in batch
        doc_texts = [chunk.text for chunk in chunks]
        doc_result = client.models.embed_content(
            model=settings.gemini_embed_model,
            contents=doc_texts,
            config={"output_dimensionality": 1536},
        )
        doc_embeddings = [np.array(emb.values) for emb in doc_result.embeddings]

        # Calculate cosine similarity between query and each document
        scores = []
        for doc_emb in doc_embeddings:
            # Cosine similarity via dot product (embeddings are normalized)
            similarity = float(np.dot(query_embedding, doc_emb))
            scores.append(similarity)

    else:
        raise ValueError(f"Unknown reranker backend: {backend}")

    # Update chunks with reranker scores
    for chunk, score in zip(chunks, scores):
        chunk.rerank_score = score
        chunk.score = score  # Use reranker score as final score

    # Sort by reranker score descending
    chunks.sort(key=lambda x: x.rerank_score or 0.0, reverse=True)

    # Keep only top_k if specified
    if top_k:
        chunks = chunks[:top_k]
    else:
        chunks = chunks[:settings.reranker_top_k]

    log.debug("Reranked %d chunks, keeping top %d", len(chunks), len(chunks))
    return chunks
