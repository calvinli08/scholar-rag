"""
Query and retrieval API for ScholarRAG.
Provides FastAPI endpoints for semantic search and Q&A.
"""

from fastapi import FastAPI, HTTPException, UploadFile, File, BackgroundTasks
from pydantic import BaseModel, Field
import os
import uuid
from pathlib import Path

from api.retriever import retrieve_chunks
from api.rag import run_rag_workflow
from ingestion.pipeline import ingest as ingest_paper
from logger import get_logger
from data_models.models import QueryResult, RetrievedChunk

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
async def upload_paper(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """
    Upload a PDF paper for ingestion.
    Returns a job ID to track ingestion progress.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")
    
    # Generate unique job ID
    job_id = str(uuid.uuid4())
    
    # Save uploaded file
    file_path = UPLOAD_DIR / f"{job_id}_{file.filename}"
    try:
        content = await file.read()
        with open(file_path, "wb") as f:
            f.write(content)
        
        # Initialize job tracking
        ingestion_jobs[job_id] = {
            "status": "pending",
            "filename": file.filename,
            "path": str(file_path),
            "error": None,
        }
        
        # Start ingestion in background
        background_tasks.add_task(process_ingestion, job_id, file_path)
        
        return {"job_id": job_id, "status": "pending"}
    except Exception as e:
        log.error("Upload failed: %s", e)
        if file_path.exists():
            file_path.unlink()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


async def process_ingestion(job_id: str, file_path: Path):
    """Process paper ingestion in the background."""
    try:
        ingestion_jobs[job_id]["status"] = "processing"
        log.info("Starting ingestion for job %s: %s", job_id, file_path)
        
        # Run ingestion pipeline
        ingest_paper(str(file_path))
        
        ingestion_jobs[job_id]["status"] = "completed"
        log.info("Ingestion completed for job %s", job_id)
    except Exception as e:
        log.error("Ingestion failed for job %s: %s", job_id, e)
        ingestion_jobs[job_id]["status"] = "failed"
        ingestion_jobs[job_id]["error"] = str(e)


@app.get("/ingest/status/{job_id}")
async def get_ingestion_status(job_id: str):
    """
    Get the status of an ingestion job.
    Returns job status: pending, processing, completed, or failed.
    """
    if job_id not in ingestion_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    
    job = ingestion_jobs[job_id]
    return {
        "job_id": job_id,
        "status": job["status"],
        "filename": job["filename"],
        "error": job["error"],
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
        result = run_rag_workflow(request.query)
        
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
        log.error("Query failed: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
