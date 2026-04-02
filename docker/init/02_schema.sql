-- 02_schema.sql
-- Creates the chunks table and all supporting indexes.
-- Three embedding columns are provided for different providers.
-- If you change the model and dimension, drop and recreate this table.

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT        PRIMARY KEY,
    paper_id    TEXT        NOT NULL,
    text        TEXT        NOT NULL,

    -- Vector columns: dimension must match your embedding model output
    embedding_openai  vector(1024),
    embedding_gemini  vector(1536),
    embedding_qwen    vector(1024),

    -- Provenance
    section     TEXT,
    page        INTEGER,
    chunk_index INTEGER,

    -- Paper-level metadata
    title       TEXT,
    authors     JSONB        DEFAULT '[]'::jsonb,
    year        INTEGER,
    doi         TEXT,
    arxiv_id    TEXT,

    created_at  TIMESTAMPTZ  DEFAULT now(),
    updated_at  TIMESTAMPTZ  DEFAULT now()
);

-- ── Indexes ──────────────────────────────────────────────────────────────────

-- Lookup all chunks for a paper (used by delete_paper and paper_exists)
CREATE INDEX IF NOT EXISTS chunks_paper_id_idx
    ON chunks (paper_id);

-- ANN vector search using HNSW (one index per embedding provider)
-- m=16, ef_construction=64 is a good starting point for up to ~1M vectors.
-- Increase ef_construction (e.g. 128) for better recall at the cost of
-- slower index builds. Increase m (e.g. 32) for higher accuracy at the
-- cost of more memory.
CREATE INDEX IF NOT EXISTS chunks_embedding_openai_hnsw_idx
    ON chunks
    USING hnsw (embedding_openai vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS chunks_embedding_gemini_hnsw_idx
    ON chunks
    USING hnsw (embedding_gemini vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

CREATE INDEX IF NOT EXISTS chunks_embedding_qwen_hnsw_idx
    ON chunks
    USING hnsw (embedding_qwen vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- BM25 full-text search index using pg_search
-- This provides true BM25 scoring for sparse retrieval
CREATE INDEX IF NOT EXISTS chunks_text_bm25_idx
    ON chunks
    USING bm25 (chunk_id, text)
    WITH (key_field = 'chunk_id');

-- Trigram similarity fallback (useful for fuzzy text search, optional)
CREATE INDEX IF NOT EXISTS chunks_text_trgm_idx
    ON chunks
    USING gin (text gin_trgm_ops);

-- ── Auto-update updated_at ───────────────────────────────────────────────────
CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = now();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE TRIGGER chunks_updated_at
    BEFORE UPDATE ON chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at();

-- ── Papers metadata table ────────────────────────────────────────────────────
-- Tracks which papers have been ingested, for idempotent re-ingestion checks.
CREATE TABLE IF NOT EXISTS papers (
    paper_id    TEXT        PRIMARY KEY,
    title       TEXT,
    authors     JSONB        DEFAULT '[]'::jsonb,
    year        INTEGER,
    doi         TEXT,
    arxiv_id    TEXT,
    file_path   TEXT,
    chunk_count INTEGER      DEFAULT 0,
    ingested_at TIMESTAMPTZ  DEFAULT now()
);