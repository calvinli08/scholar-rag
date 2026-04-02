"""
Central configuration for ScholarRAG.
All settings are loaded from environment variables (see .env.example).
Import `settings` anywhere in the codebase — never read os.environ directly.
"""

from enum import Enum
from functools import lru_cache

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class EmbedBackend(str, Enum):
    VLLM = "vllm"        # vLLM pooling models (local, default)
    OLLAMA = "ollama"    # Ollama embedding endpoint (local)
    OPENAI = "openai"    # OpenAI Embeddings API (hosted)
    COHERE = "cohere"    # Cohere Embed API (hosted)
    GEMINI = "gemini"    # Google Gemini Embedding API (hosted)


class RerankerBackend(str, Enum):
    VLLM = "vllm"        # vLLM pooling models (local, default)
    COHERE = "cohere"    # Cohere Rerank API (hosted)
    GEMINI = "gemini"    # Google Gemini API (hosted)
    NONE = "none"        # Skip reranking (useful for ablation)


class LLMBackend(str, Enum):
    OLLAMA = "ollama"      # Ollama (local, recommended)
    VLLM = "vllm"          # vLLM server (local, high-throughput)
    OPENAI = "openai"      # OpenAI API (hosted fallback)
    COHERE = "cohere"      # Cohere Generate API (hosted fallback)
    GEMINI = "gemini"      # Google Gemini API (hosted)


class EvaluationBackend(str, Enum):
    VLLM = "vllm"          # vLLM server (local, high-throughput)
    OLLAMA = "ollama"      # Ollama (local, recommended)
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

    # -- Embedding --
    embed_backend: EmbedBackend = EmbedBackend.VLLM
    vllm_embed_model: str = "intfloat/e5-mistral-7b-instruct"
    ollama_embed_model: str = "nomic-embed-text"
    ollama_embed_url: str = "http://localhost:11434"
    openai_embed_model: str = "text-embedding-3-large"
    cohere_embed_model: str = "embed-english-v3.0"
    gemini_embed_model: str = "gemini-embedding-001"
    embed_batch_size: int = 64
    embed_dim: int = Field(
        default=1024,
        description="Vector dimension — must match the chosen model output size.",
    )

    # -- Reranker --
    reranker_backend: RerankerBackend = RerankerBackend.VLLM
    vllm_reranker_model: str = "BAAI/bge-reranker-v2-minimum"
    cohere_reranker_model: str = "rerank-v4.0-fast"
    # Gemini uses embedding models for reranking
    gemini_reranker_model: str = "gemini-embedding-001"
    reranker_top_k: int = Field(default=8, description="Chunks kept after reranking.")
    reranker_initial_k: int = Field(
        default=50, description="Candidates fetched before reranking."
    )

    # -- LLM --
    llm_backend: LLMBackend = LLMBackend.OLLAMA
    ollama_model: str = "llama3"
    ollama_llm_url: str = "http://localhost:11434"
    vllm_url: str = "http://localhost:8000"
    vllm_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    openai_model: str = "gpt-4o-mini"
    cohere_model: str = "command-a-03-2025"
    gemini_model: str = "gemini-1.5-flash"
    llm_max_tokens: int = 1024
    llm_temperature: float = 0.1

    # -- API keys (only required for hosted backends) --
    openai_api_key: str = ""
    cohere_api_key: str = ""
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

    # -- Evaluation LLM --
    eval_backend: EvaluationBackend = EvaluationBackend.OLLAMA
    ollama_eval_model: str = "llama3"
    ollama_eval_url: str = "http://localhost:11434"
    vllm_eval_url: str = "http://localhost:8000"
    vllm_eval_model: str = "mistralai/Mistral-7B-Instruct-v0.3"
    openai_eval_model: str = "gpt-4o-mini"
    gemini_eval_model: str = "gemini-1.5-flash"

    # -- Eval --
    eval_test_set_path: str = "eval/test_set.json"
    eval_output_path: str = "eval/results.json"

    # ---------------------------------------------------------------------------
    # Validation
    # ---------------------------------------------------------------------------

    @model_validator(mode="after")
    def check_api_keys(self) -> "Settings":
        hosted_embed = self.embed_backend in (EmbedBackend.OPENAI, EmbedBackend.COHERE, EmbedBackend.GEMINI)
        hosted_rerank = self.reranker_backend in (RerankerBackend.COHERE, RerankerBackend.GEMINI)
        hosted_llm = self.llm_backend in (LLMBackend.OPENAI, LLMBackend.COHERE, LLMBackend.GEMINI)
        hosted_eval = self.eval_backend in (EvaluationBackend.OPENAI, EvaluationBackend.GEMINI)

        if hosted_embed and self.embed_backend == EmbedBackend.OPENAI and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when EMBED_BACKEND=openai")
        if hosted_embed and self.embed_backend == EmbedBackend.COHERE and not self.cohere_api_key:
            raise ValueError("COHERE_API_KEY is required when EMBED_BACKEND=cohere")
        if hosted_embed and self.embed_backend == EmbedBackend.GEMINI and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when EMBED_BACKEND=gemini")
        if hosted_rerank and self.reranker_backend == RerankerBackend.COHERE and not self.cohere_api_key:
            raise ValueError("COHERE_API_KEY is required when RERANKER_BACKEND=cohere")
        if hosted_rerank and self.reranker_backend == RerankerBackend.GEMINI and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when RERANKER_BACKEND=gemini")
        if hosted_llm and self.llm_backend == LLMBackend.OPENAI and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when LLM_BACKEND=openai")
        if hosted_llm and self.llm_backend == LLMBackend.COHERE and not self.cohere_api_key:
            raise ValueError("COHERE_API_KEY is required when LLM_BACKEND=cohere")
        if hosted_llm and self.llm_backend == LLMBackend.GEMINI and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when LLM_BACKEND=gemini")
        if hosted_eval and self.eval_backend == EvaluationBackend.OPENAI and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY is required when EVAL_BACKEND=openai")
        if hosted_eval and self.eval_backend == EvaluationBackend.GEMINI and not self.gemini_api_key:
            raise ValueError("GEMINI_API_KEY is required when EVAL_BACKEND=gemini")

        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached Settings instance. Call this everywhere instead of instantiating directly."""
    return Settings()


# Convenience alias — `from config import settings`
settings = get_settings()
