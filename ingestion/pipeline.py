"""
Ingestion pipeline for ScholarRAG.

Orchestrates the full ingestion flow: parse → chunk → embed → index.
This is the single entry point for ingesting papers via the FastAPI backend.

Usage:
    from ingestion.pipeline import ingest
    paper_id = ingest("path/to/paper.pdf")
"""

from __future__ import annotations

from pathlib import Path
import traceback
from typing import Optional

from logger import get_logger, configure_logging
from data_models.models import Chunk

configure_logging()

log = get_logger(__name__)


def ingest(file_path: str | Path, paper_id: Optional[str] = None) -> str:
    """
    Ingest a single PDF paper into the retrieval index.
    
    This function orchestrates the complete ingestion pipeline:
    1. Parse the PDF to extract structured content
    2. Chunk the content into semantically coherent segments
    3. Embed each chunk using the configured embedding backend
    4. Index chunks into pgvector and BM25
    
    Args:
        file_path: Path to the PDF file to ingest
        paper_id: Optional paper identifier. If not provided, 
                  uses the filename stem.
    
    Returns:
        The paper_id of the ingested paper.
    
    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        Exception: Any error during parsing, chunking, embedding, or indexing
    """
    try:
        from ingestion.pdf_parser import PDFParser
        from ingestion.chunker import Chunker
        from ingestion.embedder import get_embedder
        from ingestion.indexer import Indexer
        
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF not found: {file_path}")
        
        # Use filename stem as paper_id if not provided
        paper_id = paper_id or file_path.stem
        
        log.info("Starting ingestion pipeline for %s (id=%s)", file_path.name, paper_id)
        
        # Step 1: Parse PDF
        log.debug("Parsing PDF...")
        parser = PDFParser()
        doc = parser.parse(file_path, paper_id=paper_id)
        log.debug("Parsed %d sections from %s", len(doc.sections), paper_id)
        
        # Step 2: Chunk document
        log.debug("Chunking document...")
        chunker = Chunker()
        chunks: list[Chunk] = chunker.chunk_document(doc)
        log.debug("Created %d chunks from %s", len(chunks), paper_id)
        
        # Step 3: Embed chunks
        log.debug("Embedding chunks...")
        embedder = get_embedder()
        embedder.embed_chunks(chunks)
        log.debug("Embedded %d chunks for %s", len(chunks), paper_id)
        
        # Step 4: Index chunks
        log.debug("Indexing chunks...")
        with Indexer() as indexer:
            # Check if paper already exists
            if indexer.paper_exists(paper_id):
                log.warning("Paper %s already exists in index. Deleting and re-indexing.", paper_id)
                indexer.delete_paper(paper_id)
            
            indexer.index(chunks)
            log.info("Successfully indexed %d chunks for %s", len(chunks), paper_id)
        
        log.info("Ingestion complete for %s (id=%s)", file_path.name, paper_id)
        
        return paper_id
    except Exception as e:
        log.error("Ingestion failed for %s (id=%s): %s", file_path.name, paper_id, traceback.format_exc())

        raise e