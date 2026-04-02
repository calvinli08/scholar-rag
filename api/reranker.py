"""
Reranking module for retrieved chunks.
Supports multiple backends: vLLM (local), Cohere API.
"""

from __future__ import annotations

from functools import lru_cache

from config import settings, RerankerBackend
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

    if settings.reranker_backend == RerankerBackend.NONE:
        return chunks[:top_k or settings.reranker_top_k]

    backend = settings.reranker_backend

    if backend == RerankerBackend.VLLM:
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

    elif backend == RerankerBackend.COHERE:
        import cohere

        reranker = cohere.ClientV2(api_key=settings.cohere_api_key)

        response = reranker.rerank(
            model=settings.cohere_reranker_model,
            query=query,
            documents=[chunk.text for chunk in chunks],
            top_n=len(chunks),
        )

        # Cohere returns results already sorted, extract scores
        scores = [0.0] * len(chunks)
        for result in response.results:
            scores[result.index] = result.relevance_score

    elif backend == RerankerBackend.GEMINI:
        from google import genai
        import numpy as np

        client = genai.Client(api_key=settings.gemini_api_key)

        # Get embeddings for query and all documents
        # Use SEMANTIC_SIMILARITY task type for optimal reranking
        query_result = client.models.embed_content(
            model=settings.gemini_embed_model,
            contents=query,
        )
        query_embedding = np.array(query_result.embeddings[0].values)

        # Embed all documents in batch
        doc_texts = [chunk.text for chunk in chunks]
        doc_result = client.models.embed_content(
            model=settings.gemini_embed_model,
            contents=doc_texts,
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
