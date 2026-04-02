"""
Query embedding module for retrieval.
Supports multiple backends: vLLM (local), Ollama, OpenAI, Cohere.
"""

from __future__ import annotations

from config import settings, EmbedBackend
from logger import get_logger

log = get_logger(__name__)

async def embed_query(query: str) -> list[float]:
    """
    Embed a query string using the configured backend.

    Args:
        query: Query text to embed

    Returns:
        List of floats representing the embedding vector
    """
    backend = settings.embed_backend

    if backend == EmbedBackend.VLLM:
        import httpx

        # vLLM OpenAI-compatible embeddings API
        response = await httpx.AsyncClient().post(
            f"{settings.vllm_url}/v1/embeddings",
            json={
                "model": settings.vllm_embed_model,
                "input": query,
            },
        )

        response.raise_for_status()
        data = response.json()
        embedding = data["data"][0]["embedding"]

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
        from openai import OpenAI

        embedder = OpenAI(api_key=settings.openai_api_key)

        response = embedder.embeddings.create(
            model=settings.openai_embed_model,
            input=query,
        )

        embedding = response.data[0].embedding

    elif backend == EmbedBackend.COHERE:
        import cohere

        embedder = cohere.ClientV2(api_key=settings.cohere_api_key)

        response = embedder.embed(
            texts=[query],
            model=settings.cohere_embed_model,
            input_type="search_query",
        )

        embedding = response.embeddings.float[0]

    elif backend == EmbedBackend.GEMINI:
        from google import genai

        client = genai.Client(api_key=settings.gemini_api_key)

        result = client.models.embed_content(
            model=settings.gemini_embed_model,
            contents=query,
        )

        embedding = result.embeddings[0].values

    else:
        raise ValueError(f"Unknown embed backend: {backend}")

    return embedding
