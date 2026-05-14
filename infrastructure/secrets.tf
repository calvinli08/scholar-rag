# Auto-generated from .env.example file


resource "kubernetes_secret" "app_secrets" {
    metadata {
       name      = "scholarrag-env"
       namespace = kubernetes_namespace.scholarrag.metadata[0].name
    }
    type = "Opaque"
}
variable "MODEL_BACKEND" {
  type      = string
  sensitive = true
  default   = "qwen"
}

variable "QWEN_URL" {
  type      = string
  sensitive = true
  default   = "http://localhost:8000"
}

variable "QWEN_MODEL" {
  type      = string
  sensitive = true
  default   = "Qwen/Qwen2.5-7B-Instruct"
}

variable "QWEN_EMBED_MODEL" {
  type      = string
  sensitive = true
  default   = "Qwen/Qwen2.5-Embed-7B"
}

variable "QWEN_RERANKER_MODEL" {
  type      = string
  sensitive = true
  default   = "Qwen/Qwen2.5-Reranker-7B"
}

variable "QWEN_EVAL_MODEL" {
  type      = string
  sensitive = true
  default   = "Qwen/Qwen2.5-7B-Instruct"
}

variable "OPENAI_MODEL" {
  type      = string
  sensitive = true
  default   = "gpt-4o-mini"
}

variable "OPENAI_EMBED_MODEL" {
  type      = string
  sensitive = true
  default   = "text-embedding-3-large"
}

variable "OPENAI_EVAL_MODEL" {
  type      = string
  sensitive = true
  default   = "gpt-4o-mini"
}

variable "OPENAI_RERANKER_MODEL" {
  type      = string
  sensitive = true
  default   = "text-embedding-3-large"
}

variable "GEMINI_MODEL" {
  type      = string
  sensitive = true
  default   = "gemini-2.5-flash-lite"
}

variable "GEMINI_EMBED_MODEL" {
  type      = string
  sensitive = true
  default   = "gemini-embedding-001"
}

variable "GEMINI_RERANKER_MODEL" {
  type      = string
  sensitive = true
  default   = "gemini-embedding-001"
}

variable "GEMINI_EVAL_MODEL" {
  type      = string
  sensitive = true
  default   = "gemini-2.5-flash-lite"
}

variable "OPENAI_API_KEY" {
  type      = string
  sensitive = true
  default   = ""
}

variable "GEMINI_API_KEY" {
  type      = string
  sensitive = true
  default   = ""
}

variable "LLM_MAX_TOKENS" {
  type      = number
  sensitive = true
  default   = 1024
}

variable "LLM_TEMPERATURE" {
  type      = string
  sensitive = true
  default   = "0.1"
}

variable "EMBED_BATCH_SIZE" {
  type      = number
  sensitive = true
  default   = 64
}

variable "RERANKER_TOP_K" {
  type      = number
  sensitive = true
  default   = 8
}

variable "RERANKER_INITIAL_K" {
  type      = number
  sensitive = true
  default   = 50
}

variable "DATABASE_URL" {
  type      = string
  sensitive = true
  default   = "postgresql://localhost:5432/scholarrag"
}

variable "PGVECTOR_TABLE" {
  type      = string
  sensitive = true
  default   = "chunks"
}

variable "BM25_TOP_K" {
  type      = number
  sensitive = true
  default   = 50
}

variable "DENSE_TOP_K" {
  type      = number
  sensitive = true
  default   = 50
}

variable "RRF_K" {
  type      = number
  sensitive = true
  default   = 60
}

variable "CHUNK_SIZE" {
  type      = number
  sensitive = true
  default   = 512
}

variable "CHUNK_OVERLAP" {
  type      = number
  sensitive = true
  default   = 128
}

variable "APP_HOST" {
  type      = string
  sensitive = true
  default   = "0.0.0.0"
}

variable "APP_PORT" {
  type      = number
  sensitive = true
  default   = 8000
}

variable "LOG_LEVEL" {
  type      = string
  sensitive = true
  default   = "info"
}

variable "DEBUG" {
  type      = bool
  sensitive = true
  default   = true
}

variable "FILE_UPLOAD_PROVIDER" {
  type      = string
  sensitive = true
  default   = "local"
}

variable "DOWNLOAD_DIR" {
  type      = string
  sensitive = true
  default   = "downloads"
}

variable "S3_ENDPOINT" {
  type      = string
  sensitive = true
  default   = "http://localhost:9000"
}

variable "S3_ACCESS_KEY" {
  type      = string
  sensitive = true
  default   = "minioadmin"
}

variable "S3_SECRET_KEY" {
  type      = string
  sensitive = true
  default   = "minioadmin"
}

variable "S3_BUCKET" {
  type      = string
  sensitive = true
  default   = "scholar-rag"
}

variable "S3_REGION" {
  type      = string
  sensitive = true
  default   = "us-east-1"
}

variable "USE_HYDE" {
  type      = bool
  sensitive = true
  default   = true
}

variable "GROUNDING_THRESHOLD" {
  type      = string
  sensitive = true
  default   = "0.9"
}

variable "MAX_RETRIES" {
  type      = number
  sensitive = true
  default   = 2
}

variable "EVAL_TEST_SET_PATH" {
  type      = string
  sensitive = true
  default   = "eval/test_set.json"
}

variable "EVAL_OUTPUT_PATH" {
  type      = string
  sensitive = true
  default   = "eval/results.json"
}

variable "LANGFUSE_SECRET_KEY" {
  type      = string
  sensitive = true
  default   = ""
}

variable "LANGFUSE_PUBLIC_KEY" {
  type      = string
  sensitive = true
  default   = ""
}

variable "LANGFUSE_BASE_URL" {
  type      = string
  sensitive = true
  default   = "https://cloud.langfuse.com"
}

variable "CELERY_BROKER_URL" {
  type      = string
  sensitive = true
  default   = "redis://redis:6379/0"
}

variable "CELERY_RESULT_BACKEND" {
  type      = string
  sensitive = true
  default   = "redis://redis:6379/0"
}
