"""
Central configuration for ScholarRAG.
All settings are loaded from environment variables (see .env.example).
Import `settings` anywhere in the codebase — never read os.environ directly.
"""

from enum import Enum
from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelBackend(str, Enum):
    """Unified model backend for all components (embedding, reranking, LLM, evaluation)."""
    VLLM = "vllm"          # vLLM server (local, high-throughput)
    OPENAI = "openai"      # OpenAI API (hosted)
    GEMINI = "gemini"      # Google Gemini API (hosted)


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # -- Unified Model Backend --
    model_backend: ModelBackend = ModelBackend.VLLM

    # -- Embedding --
    vllm_embed_model: str = "intfloat/e5-mistral-7b-instruct"
    openai_embed_model: str = "text-embedding-3-large"
    # 1024 dimensions preserves compatibility between text-embedding-3-large and text-embedding-3-small
    openai_embed_dim: int = 1024
    # Gemini embedding model: gemini-embedding-001 or gemini-embedding-002
    gemini_embed_model: str = "gemini-embedding-001"
    # 1536 dimensions preserves compatibility between gemini-embedding-001 and gemini-embedding-002
    gemini_embed_dim: int = 1536
    embed_batch_size: int = 64
    embed_dim: int = Field(
        default=1024,
        description="Vector dimension — must match the chosen model output size.",
    )

    # -- Reranker --
    vllm_reranker_model: str = "BAAI/bge-reranker-v2-minimum"
    openai_reranker_model: str = "text-embedding-3-large"
    # Gemini uses embedding models for reranking: gemini-embedding-001 or gemini-embedding-002
    gemini_reranker_model: str = "gemini-embedding-001"
    reranker_top_k: int = Field(default=8, description="Chunks kept after reranking.")
    reranker_initial_k: int = Field(
        default=50, description="Candidates fetched before reranking."
    )

    # -- LLM --
    vllm_url: str = "http://localhost:8000"
    vllm_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    openai_model: str = "gpt-4o-mini"
    gemini_model: str = "gemini-2.5-flash-lite"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.1

    # -- API keys (only required for hosted backends) --
    openai_api_key: str = ""
    gemini_api_key: str = ""

    # -- Vector store --
    database_url: str = "postgresql://localhost:5432/scholarrag"
    pgvector_table: str = "chunks"

    # -- Retrieval --
    bm25_top_k: int = 50
    dense_top_k: int = 50
    rrf_k: int = Field(
        default=60,
        description="RRF constant. Higher values reduce impact of rank differences.",
    )

    # -- Chunking --
    chunk_size: int = 512       # tokens
    chunk_overlap: int = 128    # tokens

    # -- App --
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    log_level: str = "info"

    # -- Ingestion --
    upload_dir: str = "./uploads"

    # -- Agentic RAG & Evaluation --
    use_hyde: bool = True
    grounding_threshold: float = 0.9
    max_retries: int = 2

    # -- Evaluation --
    vllm_eval_url: str = "http://localhost:8000"
    vllm_eval_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    openai_eval_model: str = "gpt-4o-mini"
    gemini_eval_model: str = "gemini-2.5-flash-lite"

    # -- Eval --
    eval_test_set_path: str = "eval/test_set.json"
    eval_output_path: str = "eval/results.json"

    # ---------------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------------

    @model_validator(mode="after")
    def check_api_keys(self) -> "Settings":
        if self.model_backend == ModelBackend.OPENAI and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when MODEL_BACKEND=openai")
        if self.model_backend == ModelBackend.GEMINI and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when MODEL_BACKEND=gemini")

        return self

    @model_validator(mode="after")
    def check_embedding_models(self) -> "Settings":
        # Validate OpenAI embedding model
        allowed_openai_models = {"text-embedding-3-large", "text-embedding-3-small"}
        if self.openai_embed_model not in allowed_openai_models:
            raise ValueError(
                f"OPENAI_EMBED_MODEL must be one of {allowed_openai_models}, "
                f"got '{self.openai_embed_model}'"
            )

        # Validate Gemini embedding model
        allowed_gemini_models = {"gemini-embedding-001", "gemini-embedding-002"}
        if self.gemini_embed_model not in allowed_gemini_models:
            raise ValueError(
                f"GEMINI_EMBED_MODEL must be one of {allowed_gemini_models}, "
                f"got '{self.gemini_embed_model}'"
            )

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance. Call this everywhere instead of instantiating directly."""
    return Settings()


# Convenience alias — `from config import settings`
settings = get_settings()
