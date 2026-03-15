"""
Query embedding module for retrieval.
Supports multiple backends: HuggingFace (local), Ollama, OpenAI, Cohere.
"""

from __future__ import annotations

from functools import lru_cache

from config import settings, EmbedBackend
from logger import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=1)
def _get_embedder():
    """Get the configured embedding model."""
    backend = settings.embed_backend
    
    if backend == EmbedBackend.HF:
        from sentence_transformers import SentenceTransformer
        log.info("Loading HF embedding model: %s", settings.hf_embed_model)
        return SentenceTransformer(settings.hf_embed_model)
    
    elif backend == EmbedBackend.OLLAMA:
        log.info("Using Ollama embedding backend")
        return "ollama"
    
    elif backend == EmbedBackend.OPENAI:
        from openai import OpenAI
        log.info("Using OpenAI embedding backend")
        return OpenAI(api_key=settings.openai_api_key)
    
    elif backend == EmbedBackend.COHERE:
        import cohere
        log.info("Using Cohere embedding backend")
        return cohere.Client(api_key=settings.cohere_api_key)
    
    else:
        raise ValueError(f"Unknown embed backend: {backend}")


async def embed_query(query: str) -> list[float]:
    """
    Embed a query string using the configured backend.
    
    Args:
        query: Query text to embed
        
    Returns:
        List of floats representing the embedding vector
    """
    embedder = _get_embedder()
    backend = settings.embed_backend
    
    if backend == EmbedBackend.HF:
        embedding = embedder.encode(
            query,
            convert_to_numpy=True,
            normalize_embeddings=True,
        ).tolist()
        
    elif backend == EmbedBackend.OLLAMA:
        import httpx
        response = await httpx.AsyncClient().post(
            f"{settings.ollama_embed_url}/api/embeddings",
            json={
                "model": settings.ollama_embed_model,
                "prompt": query,
            },
        )
        response.raise_for_status()
        embedding = response.json()["embedding"]
        
    elif backend == EmbedBackend.OPENAI:
        response = embedder.embeddings.create(
            model=settings.openai_embed_model,
            input=query,
        )
        embedding = response.data[0].embedding
        
    elif backend == EmbedBackend.COHERE:
        response = embedder.embed(
            texts=[query],
            model=settings.cohere_embed_model,
        )
        embedding = response.embeddings[0]
        
    else:
        raise ValueError(f"Unknown embed backend: {backend}")
    
    return embedding
