"""
Hybrid retrieval module combining dense (vector) and sparse (BM25) search.
Implements Reciprocal Rank Fusion (RRF) for merging results.
"""

from __future__ import annotations

from db import get_pool
from logger import get_logger
from data_models.models import Chunk, RetrievedChunk
from config import settings

log = get_logger(__name__)


async def retrieve_chunks(
    query: str,
    top_k: int = 5,
    use_reranker: bool = True,
) -> list[RetrievedChunk]:
    """
    Retrieve chunks using hybrid search (dense + BM25).
    
    Args:
        query: Search query string
        top_k: Number of chunks to return after reranking
        use_reranker: Whether to apply cross-encoder reranking
        
    Returns:
        List of RetrievedChunk objects sorted by final score
    """
    # Get dense and sparse retrievals in parallel
    dense_results = await _dense_retrieve(query, k=settings.reranker_initial_k if use_reranker else top_k)
    sparse_results = await _sparse_retrieve(query, k=settings.reranker_initial_k if use_reranker else top_k)
    
    # Merge using RRF
    merged = _reciprocal_rank_fusion(dense_results, sparse_results, k=settings.rrf_k)
    
    # Apply reranking if enabled
    if use_reranker and settings.reranker_backend != "none":
        from api.reranker import rerank_chunks
        merged = await rerank_chunks(query, merged, top_k=top_k)
    
    return merged[:top_k]


async def _dense_retrieve(query: str, k: int = 50) -> list[RetrievedChunk]:
    """
    Retrieve chunks using vector similarity search.
    
    Args:
        query: Query string to embed and search
        k: Number of results to return
        
    Returns:
        List of RetrievedChunk with dense_rank set
    """
    from api.embedder import embed_query
    
    # Embed the query
    query_embedding = await embed_query(query)
    
    pool = get_pool()
    results = []
    
    with pool.connection() as conn:
        with conn.cursor() as cur:
            # Use pgvector cosine distance (<=>)
            cur.execute("""
                SELECT chunk_id, paper_id, text, section, page, chunk_index,
                       title, authors, year, doi, arxiv_id,
                       1 - (embedding <=> %s::vector) AS similarity
                FROM chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s
            """, (query_embedding, query_embedding, k))
            
            rows = cur.fetchall()
            
            for row in rows:
                chunk = Chunk(
                    chunk_id=row[0],
                    paper_id=row[1],
                    text=row[2],
                    section=row[3],
                    page=row[4],
                    chunk_index=row[5],
                    title=row[6],
                    authors=row[7] or [],
                    year=row[8],
                    doi=row[9] or "",
                    arxiv_id=row[10] or "",
                )
                retrieved = RetrievedChunk(
                    chunk=chunk,
                    score=row[11],  # similarity score
                    dense_rank=len(results) + 1,
                )
                results.append(retrieved)
    
    log.debug("Dense retrieval returned %d chunks", len(results))
    return results


async def _sparse_retrieve(query: str, k: int = 50) -> list[RetrievedChunk]:
    """
    Retrieve chunks using BM25 full-text search via pg_search extension.
    
    Args:
        query: Query string for text search
        k: Number of results to return
        
    Returns:
        List of RetrievedChunk with sparse_rank set
    """
    pool = get_pool()
    results = []
    
    with pool.connection() as conn:
        with conn.cursor() as cur:
            # Use pg_search BM25 for true BM25 scoring
            # The paradedb.score() function returns the BM25 score
            cur.execute("""
                SELECT chunk_id, paper_id, text, section, page, chunk_index,
                       title, authors, year, doi, arxiv_id,
                       paradedb.score(id => chunk_id) AS bm25_score
                FROM chunks
                WHERE text @@@ %s
                ORDER BY bm25_score DESC
                LIMIT %s
            """, (query, k))
            
            rows = cur.fetchall()
            
            for row in rows:
                chunk = Chunk(
                    chunk_id=row[0],
                    paper_id=row[1],
                    text=row[2],
                    section=row[3],
                    page=row[4],
                    chunk_index=row[5],
                    title=row[6],
                    authors=row[7] or [],
                    year=row[8],
                    doi=row[9] or "",
                    arxiv_id=row[10] or "",
                )
                retrieved = RetrievedChunk(
                    chunk=chunk,
                    score=row[11],  # BM25 score from pg_search
                    sparse_rank=len(results) + 1,
                )
                results.append(retrieved)
    
    log.debug("Sparse retrieval (BM25 via pg_search) returned %d chunks", len(results))
    return results


def _reciprocal_rank_fusion(
    dense: list[RetrievedChunk],
    sparse: list[RetrievedChunk],
    k: int = 60,
) -> list[RetrievedChunk]:
    """
    Combine dense and sparse results using Reciprocal Rank Fusion.
    
    RRF formula: score = sum(1 / (k + rank)) for each ranking
    
    Args:
        dense: Results from vector search (with dense_rank set)
        sparse: Results from BM25 search (with sparse_rank set)
        k: RRF constant (higher = less impact from rank differences)
        
    Returns:
        Combined list sorted by RRF score
    """
    # Build lookup tables
    dense_lookup = {r.chunk_id: r for r in dense}
    sparse_lookup = {r.chunk_id: r for r in sparse}
    
    # All unique chunk IDs
    all_ids = set(dense_lookup.keys()) | set(sparse_lookup.keys())
    
    fused = []
    for chunk_id in all_ids:
        dense_result = dense_lookup.get(chunk_id)
        sparse_result = sparse_lookup.get(chunk_id)
        
        # Get ranks (use max + 1 if not present in one list)
        dense_rank = dense_result.dense_rank if dense_result else len(dense) + 1
        sparse_rank = sparse_result.sparse_rank if sparse_result else len(sparse) + 1
        
        # Calculate RRF score
        rrf_score = (1.0 / (k + dense_rank)) + (1.0 / (k + sparse_rank))
        
        # Use the chunk from whichever source has it
        chunk_obj = dense_result.chunk if dense_result else sparse_result.chunk
        
        retrieved = RetrievedChunk(
            chunk=chunk_obj,
            score=rrf_score,
            dense_rank=dense_result.dense_rank if dense_result else None,
            sparse_rank=sparse_result.sparse_rank if sparse_result else None,
        )
        fused.append(retrieved)
    
    # Sort by RRF score descending
    fused.sort(key=lambda x: x.score, reverse=True)
    
    log.debug("RRF fusion produced %d chunks", len(fused))
    return fused
