"""
Query and retrieval API for ScholarRAG.
Provides FastAPI endpoints for semantic search and Q&A.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.retriever import retrieve_chunks
from api.generator import generate_answer
from logger import get_logger
from models import QueryResult, RetrievedChunk

log = get_logger(__name__)

app = FastAPI(
    title="ScholarRAG API",
    description="Academic paper retrieval and Q&A system",
    version="0.1.0",
)


class QueryRequest(BaseModel):
    """Request model for query endpoint."""
    
    query: str = Field(..., description="Natural language query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of chunks to retrieve")
    use_reranker: bool = Field(default=True, description="Whether to use reranking")
    use_hyde: bool = Field(default=True, description="Whether to use HyDE expansion")


class QueryResponse(BaseModel):
    """Response model for query endpoint."""
    
    query: str
    answer: str | None = None
    sources: list[dict] = Field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    
    @classmethod
    def from_result(cls, result: QueryResult) -> "QueryResponse":
        """Convert QueryResult to response format."""
        return cls(
            query=result.query,
            answer=result.answer,
            sources=[
                {
                    "chunk_id": src.chunk_id,
                    "text": src.text,
                    "score": src.score,
                    "title": src.chunk.title,
                    "section": src.chunk.section,
                }
                for src in result.sources
            ],
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            total_tokens=result.total_tokens,
        )


class SearchRequest(BaseModel):
    """Request model for search-only endpoint."""
    
    query: str = Field(..., description="Search query")
    top_k: int = Field(default=5, ge=1, le=20, description="Number of results")
    use_reranker: bool = Field(default=True, description="Whether to use reranking")


class SearchResponse(BaseModel):
    """Response model for search endpoint."""
    
    query: str
    results: list[dict] = Field(default_factory=list)
    
    @classmethod
    def from_chunks(cls, query: str, chunks: list[RetrievedChunk]) -> "SearchResponse":
        """Convert retrieved chunks to response format."""
        return cls(
            query=query,
            results=[
                {
                    "chunk_id": chunk.chunk_id,
                    "text": chunk.text,
                    "score": chunk.score,
                    "title": chunk.chunk.title,
                    "section": chunk.chunk.section,
                    "paper_id": chunk.chunk.paper_id,
                }
                for chunk in chunks
            ],
        )


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}


@app.post("/search", response_model=SearchResponse)
async def search(request: SearchRequest):
    """
    Search for relevant paper chunks without generating an answer.
    Returns ranked chunks with scores.
    """
    try:
        chunks = await retrieve_chunks(
            query=request.query,
            top_k=request.top_k,
            use_reranker=request.use_reranker,
        )
        return SearchResponse.from_chunks(request.query, chunks)
    except Exception as e:
        log.error("Search failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Full RAG pipeline: retrieve chunks and generate an answer.
    Returns answer with cited sources.
    """
    try:
        # Retrieve relevant chunks
        chunks = await retrieve_chunks(
            query=request.query,
            top_k=request.top_k,
            use_reranker=request.use_reranker,
        )
        
        # Generate answer using retrieved context
        result = await generate_answer(
            query=request.query,
            chunks=chunks,
            use_hyde=request.use_hyde,
        )
        
        return QueryResponse.from_result(result)
    except Exception as e:
        log.error("Query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
