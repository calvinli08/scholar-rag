-- 01_extensions.sql
-- Runs once on first container start.
-- Enables pgvector and any other extensions ScholarRAG needs.

-- pgvector: dense vector similarity search
CREATE EXTENSION IF NOT EXISTS vector;

-- pg_search: true BM25 full-text search extension
CREATE EXTENSION IF NOT EXISTS pg_search;

-- pg_trgm: trigram similarity (useful for fuzzy text search, optional)
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- uuid-ossp: UUID generation (useful for future chunk ID schemes)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
