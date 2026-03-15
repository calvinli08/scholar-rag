"""
Reranking module for retrieved chunks.
Supports multiple backends: HuggingFace cross-encoder (local), Cohere API.
"""

from __future__ import annotations

from functools import lru_cache

from config import settings, RerankerBackend
from logger import get_logger
from models import RetrievedChunk

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _get_reranker():
    """Get the configured reranking model."""
    backend = settings.reranker_backend
    
    if backend == RerankerBackend.HF:
        from sentence_transformers import CrossEncoder
        log.info("Loading HF reranker model: %s", settings.hf_reranker_model)
        return CrossEncoder(settings.hf_reranker_model)
    
    elif backend == RerankerBackend.COHERE:
        import cohere
        log.info("Using Cohere reranker backend")
        return cohere.Client(api_key=settings.cohere_api_key)
    
    elif backend == RerankerBackend.NONE:
        return None
    
    else:
        raise ValueError(f"Unknown reranker backend: {backend}")


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
    
    reranker = _get_reranker()
    backend = settings.reranker_backend
    
    # Prepare pairs for cross-encoder
    pairs = [(query, chunk.text) for chunk in chunks]
    
    if backend == RerankerBackend.HF:
        scores = reranker.predict(pairs).tolist()
        
    elif backend == RerankerBackend.COHERE:
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
