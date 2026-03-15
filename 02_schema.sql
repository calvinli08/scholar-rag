-- 02_schema.sql
-- Creates the chunks table and all supporting indexes.
-- The embedding dimension here must match EMBED_DIM in your .env (default 1024).
-- If you change the model and dimension, drop and recreate this table.

-- ── Chunks table ─────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chunks (
    chunk_id    TEXT        PRIMARY KEY,
    paper_id    TEXT        NOT NULL,
    text        TEXT        NOT NULL,

    -- Vector column: dimension must match your embedding model output
    -- Default: 1024 (bge-large-en-v1.5, nomic-embed-text-v1.5)
    -- OpenAI text-embedding-3-large: 3072
    -- OpenAI text-embedding-3-small / ada-002: 1536
    embedding   vector(1024),

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

-- ANN vector search using HNSW
-- m=16, ef_construction=64 is a good starting point for up to ~1M vectors.
-- Increase ef_construction (e.g. 128) for better recall at the cost of
-- slower index builds. Increase m (e.g. 32) for higher accuracy at the
-- cost of more memory.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw_idx
    ON chunks
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- BM25 full-text search index using pg_search
-- This provides true BM25 scoring for sparse retrieval
CREATE INDEX IF NOT EXISTS chunks_text_bm25_idx
    ON chunks
    USING bm25 (text)
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
