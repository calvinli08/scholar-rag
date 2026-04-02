"""
Query embedding module for retrieval.
Supports multiple backends: Qwen on vLLM (local), OpenAI, Gemini.
Note: Qwen models must be hosted on vLLM server to work with ScholarRAG.
"""

from __future__ import annotations

from config import settings, ModelBackend
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
    backend = settings.model_backend

    if backend == ModelBackend.QWEN:
        import httpx

        # Qwen embedder hosted on vLLM server (OpenAI-compatible embeddings API)
        response = await httpx.AsyncClient().post(
            f"{settings.qwen_url}/v1/embeddings",
            json={
                "model": settings.qwen_embed_model,
                "input": query,
            },
        )

        response.raise_for_status()
        data = response.json()
        embedding = data["data"][0]["embedding"]

    elif backend == ModelBackend.OPENAI:
        from openai import OpenAI

        embedder = OpenAI(api_key=settings.openai_api_key)

        response = embedder.embeddings.create(
            model=settings.openai_embed_model,
            input=query,
            dimensions=settings.openai_embed_dim,
        )

        embedding = response.data[0].embedding

    elif backend == ModelBackend.GEMINI:
        from google import genai

        client = genai.Client(api_key=settings.gemini_api_key)

        # 1536 dimensions preserves compatibility between gemini-embedding-001 and gemini-embedding-002
        result = client.models.embed_content(
            model=settings.gemini_embed_model,
            contents=query,
            config={"output_dimensionality": 1536},
        )

        embedding = result.embeddings[0].values

    else:
        raise ValueError(f"Unknown embed backend: {backend}")

    return embedding
