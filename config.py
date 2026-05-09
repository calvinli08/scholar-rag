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
    """Unified model backend for all components (embedding, reranking, LLM, evaluation).
    
    Note: QWEN backend requires Qwen models to be hosted on vLLM server.
    The vLLM inference engine provides the OpenAI-compatible API layer,
    but only Qwen family models are supported for ScholarRAG.
    """
    QWEN = "qwen"          # Qwen models hosted on vLLM server (local, high-throughput)
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
    # Default: Qwen models hosted on vLLM server
    model_backend: ModelBackend = ModelBackend.QWEN

    # -- Embedding --
    # Qwen3-Embedding models: Qwen/Qwen3-Embedding-0.6B, Qwen/Qwen3-Embedding-4B, Qwen/Qwen3-Embedding-8B
    # Note: These Qwen models must be hosted on vLLM server to work with ScholarRAG
    qwen_embed_model: str = "Qwen/Qwen3-Embedding-0.6B"
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
    # Qwen3-Reranker models: Qwen/Qwen3-Reranker-0.6B, Qwen/Qwen3-Reranker-4B, Qwen/Qwen3-Reranker-8B
    # Note: These Qwen models must be hosted on vLLM server to work with ScholarRAG
    qwen_reranker_model: str = "Qwen/Qwen3-Reranker-0.6B"
    openai_reranker_model: str = "text-embedding-3-large"
    # Gemini uses embedding models for reranking: gemini-embedding-001 or gemini-embedding-002
    gemini_reranker_model: str = "gemini-embedding-001"
    reranker_top_k: int = Field(default=8, description="Chunks kept after reranking.")
    reranker_initial_k: int = Field(
        default=50, description="Candidates fetched before reranking."
    )

    # -- LLM --
    # Qwen models must be hosted on vLLM server for ScholarRAG
    # Qwen3.5 models: Qwen/Qwen3.5-397B-A17B, Qwen/Qwen3.5-397B-A17B-FP8
    # Qwen3 models: Qwen/Qwen3-8B, Qwen/Qwen3-8B-Instruct
    # Qwen2.5 models: Qwen/Qwen2.5-7B-Instruct, Qwen/Qwen2.5-3B-Instruct
    qwen_url: str = "http://localhost:8000"
    qwen_model: str = "Qwen/Qwen3.5-397B-A17B-FP8"
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
    debug: bool = True

    # -- Ingestion --
    upload_dir: str = "./uploads"

    # -- Agentic RAG & Evaluation --
    use_hyde: bool = True
    grounding_threshold: float = 0.9
    max_retries: int = 2

    # -- Eval --
    eval_test_set_path: str = "eval/test_set.json"
    eval_output_path: str = "eval/results.json"

    # LangFuse
    langfuse_private_key: str = ""
    langfuse_public_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"

    # Celery
    celery_broker_url: str = "redis://redis:6379/0"
    celery_result_backend: str = "redis://redis:6379/0"

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
    def check_models(self) -> "Settings":
        if self.model_backend == ModelBackend.OPENAI:
            # Validate OpenAI embedding model
            allowed_openai_models = {"text-embedding-3-large", "text-embedding-3-small"}
            if self.openai_embed_model not in allowed_openai_models:
                raise ValueError(
                    f"OPENAI_EMBED_MODEL must be one of {allowed_openai_models}, "
                    f"got '{self.openai_embed_model}'"
                )
        elif self.model_backend == ModelBackend.GEMINI:
            # Validate Gemini embedding model
            allowed_gemini_models = {"gemini-embedding-001", "gemini-embedding-002"}
            if self.gemini_embed_model not in allowed_gemini_models:
                raise ValueError(
                    f"GEMINI_EMBED_MODEL must be one of {allowed_gemini_models}, "
                    f"got '{self.gemini_embed_model}'"
                )
        elif self.model_backend == ModelBackend.QWEN:
            # Validate Qwen models (hosted on vLLM server)
            # Allowed Qwen3-Embedding models

            ALLOWED_QWEN_EMBED_MODELS = {
                "Qwen/Qwen3-Embedding-0.6B",
                "Qwen/Qwen3-Embedding-4B",
                "Qwen/Qwen3-Embedding-8B",
            }

            if self.qwen_embed_model not in ALLOWED_QWEN_EMBED_MODELS:
                raise ValueError(
                    f"QWEN_EMBED_MODEL must be a Qwen3-Embedding model, "
                    f"one of {ALLOWED_QWEN_EMBED_MODELS}, got '{self.qwen_embed_model}'"
                )

            # Allowed Qwen3-Reranker models
            ALLOWED_QWEN_RERANKER_MODELS = {
                "Qwen/Qwen3-Reranker-0.6B",
                "Qwen/Qwen3-Reranker-4B",
                "Qwen/Qwen3-Reranker-8B",
            }

            if self.qwen_reranker_model not in ALLOWED_QWEN_RERANKER_MODELS:
                raise ValueError(
                    f"QWEN_RERANKER_MODEL must be a Qwen3-Reranker model, "
                    f"one of {ALLOWED_QWEN_RERANKER_MODELS}, got '{self.qwen_reranker_model}'"
                )

            # Allowed Qwen LLM models (Instruct variants for chat/completion)
            # Qwen2.5-Instruct: https://huggingface.co/collections/Qwen/qwen25
            # Qwen3/3.5: https://huggingface.co/collections/Qwen/qwen35
            ALLOWED_QWEN_LLM_MODELS = {
                # Qwen2.5-Instruct
                "Qwen/Qwen2.5-0.5B-Instruct",
                "Qwen/Qwen2.5-1.5B-Instruct",
                "Qwen/Qwen2.5-3B-Instruct",
                "Qwen/Qwen2.5-7B-Instruct",
                "Qwen/Qwen2.5-14B-Instruct",
                "Qwen/Qwen2.5-32B-Instruct",
                "Qwen/Qwen2.5-72B-Instruct",
                # Qwen3
                "Qwen/Qwen3-8B",
                "Qwen/Qwen3-8B-Instruct",
                "Qwen/Qwen3-14B",
                "Qwen/Qwen3-14B-Instruct",
                "Qwen/Qwen3-32B",
                "Qwen/Qwen3-32B-Instruct",
                # Qwen3.5
                "Qwen/Qwen3.5-0.8B",
                "Qwen/Qwen3.5-2B",
                "Qwen/Qwen3.5-4B",
                "Qwen/Qwen3.5-9B",
                "Qwen/Qwen3.5-27B",
                "Qwen/Qwen3.5-35B-A3B",
                "Qwen/Qwen3.5-122B-A10B",
                "Qwen/Qwen3.5-397B-A17B",
                # FP8 quantized variants (recommended for production)
                "Qwen/Qwen3.5-397B-A17B-FP8",
                "Qwen/Qwen3.5-122B-A10B-FP8",
                "Qwen/Qwen3.5-35B-A3B-FP8",
                "Qwen/Qwen3.5-27B-FP8",
            }

            if self.qwen_model not in ALLOWED_QWEN_LLM_MODELS:
                raise ValueError(
                    f"QWEN_MODEL must be a Qwen Instruct model, "
                    f"one of {ALLOWED_QWEN_LLM_MODELS}, got '{self.qwen_model}'"
                )

        return self

@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance. Call this everywhere instead of instantiating directly."""
    return Settings()


# Convenience alias — `from config import settings`
settings = get_settings()
