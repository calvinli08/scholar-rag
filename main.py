"""
Query and retrieval API for ScholarRAG.
Provides FastAPI endpoints for semantic search and Q&A.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel, Field
import os
import uuid
from pathlib import Path
from celery.result import AsyncResult
from celery_app import celery_app
from ingestion.tasks import ingest_paper_task
from api.retriever import retrieve_chunks
from api.rag import run_rag_workflow
from logger import get_logger, configure_logging
from data_models.models import QueryResult, RetrievedChunk
from db import get_pool

configure_logging()

log = get_logger(__name__)

# Directory for uploaded files
UPLOAD_DIR = Path(os.getenv("UPLOAD_DIR", "./uploads"))
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Track ingestion jobs
ingestion_jobs: dict[str, dict] = {}

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


@app.post("/ingest/upload")
async def upload_paper(file: UploadFile = File(...)):
    """
    Upload a PDF paper for ingestion.
    Returns a job ID to track ingestion progress.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    task_id = str(uuid.uuid4())
    
    # Save uploaded file
    file_path = UPLOAD_DIR / f"{task_id}_{file.filename}"
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        result = ingest_paper_task.apply_async(
            kwargs={"file_path": str(file_path), "paper_id": task_id}, 
            task_id=task_id,
            queue="ingestion"
        )
        
        return {"job_id": task_id, "status": result.status.lower()}
    except Exception as e:
        log.error("Upload failed: %s", e)

        if file_path.exists():
            file_path.unlink()

        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/api/papers")
async def list_papers():
    """
    List all uploaded papers with their metadata and status.
    Returns array of paper objects.
    """
    try:
        pool = get_pool()
        with pool.connection() as conn:
            with conn.cursor() as cur:
                # Query the papers table with chunk counts
                query = """
                SELECT 
                    p.paper_id,
                    p.title AS filename,
                    p.ingested_at AS uploaded_at,
                    p.chunk_count,
                    COALESCE(SUM(LENGTH(c.text) / 4), 0) AS token_count,
                    jsonb_build_object(
                        'title', p.title,
                        'authors', p.authors,
                        'year', p.year,
                        'doi', p.doi,
                        'arxiv_id', p.arxiv_id
                    ) AS metadata
                FROM papers p
                LEFT JOIN chunks c ON p.paper_id = c.paper_id
                GROUP BY p.paper_id, p.title, p.ingested_at, p.chunk_count, p.authors, p.year, p.doi, p.arxiv_id
                ORDER BY p.ingested_at DESC
                """
                
                cur.execute(query)
                rows = cur.fetchall()
                
                papers = []
                for row in rows:
                    papers.append({
                        "paper_id": row[0],
                        "filename": row[1] or "Unknown",
                        "status": "completed",  # All papers in DB are completed
                        "uploaded_at": row[2].isoformat() if row[2] else "N/A",
                        "chunk_count": row[3] or 0,
                        "token_count": row[4] or 0,
                        "metadata": row[5] or {}
                    })
        
        return {"papers": papers}
    except Exception as e:
        log.error("Failed to list papers: %s", e)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ingestion/status")
async def get_all_ingestion_status():
    """
    Get the status of all pending ingestion jobs.
    Returns array of job objects.
    """
    try:
        inspector = celery_app.control.inspect()

        active_tasks = inspector.active()

        jobs = []
        for worker, tasks in active_tasks.items():
            for task in tasks:
                if task["name"] == "ingestion.ingest_paper_task":
                    jobs.append({
                        "job_id": task["id"],
                        "filename": task["args"][0],
                        "status": task["status"].lower(),
                        "started_at": task["time_start"],
                    })
        
        return {"jobs": jobs}
    except Exception as e:
        log.error("Failed to get ingestion status: %s", e)

        raise HTTPException(status_code=500, detail="Failed to get ingestion status")


@app.get("/ingest/status/{job_id}")
async def get_ingestion_status(job_id: str):
    """
    Get the status of an ingestion job.
    Returns job status: pending, processing, completed, or failed.
    """
    result = AsyncResult(job_id, app=celery_app)
    
    task_args = {}
    if isinstance(result.info, dict):
        file_name = result.info.get("file_path", "Unknown")
    
    if not task_args.get("file_path"):
        inspector = celery_app.control.inspect()

        query = inspector.query_task(job_id)

        if query:
            for worker_tasks in query.values():
                if job_id in worker_tasks:
                    file_name = worker_tasks[job_id][2].get("file_path", "Unknown")
    
    return {
        "job_id": job_id,
        "status": result.state.lower(),
        "file_name": file_name
    }


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


@app.post("/query")
async def query(request: QueryRequest):
    """
    Full RAG pipeline with agentic grounding evaluation.
    Uses LangGraph workflow with HyDE, retrieval, generation, and DeepEval grounding check.
    Returns answer with cited sources and grounding score.
    """
    try:
        # Run the agentic RAG workflow
        result = await run_rag_workflow(
            query=request.query,
            top_k=request.top_k,
            use_reranker=request.use_reranker
        )

        return {
            "query": request.query,
            "answer": result["answer"],
            "sources": result["sources"],
            "metadata": {
                "grounding_score": result["grounding_score"],
                "retries": result["retries"]
            }
        }
    except Exception as e:
        import traceback

        log.error("Query failed: %s\n%s", e, traceback.format_exc())

        raise HTTPException(status_code=500, detail="Error generating response")
