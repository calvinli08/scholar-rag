"""
Celery tasks for ScholarRAG ingestion pipeline.

This module defines the core ingestion logic as a Celery task, allowing
it to be processed asynchronously by Celery workers.
"""

from __future__ import annotations

from pathlib import Path
import traceback
from typing import Optional

from celery_app import celery_app
from logger import get_logger
from data_models.models import Chunk

from s3_client import s3_client
from config import settings

# Import the actual ingestion pipeline components
from ingestion.pdf_parser import PDFParser
from ingestion.chunker import Chunker
from ingestion.embedder import get_embedder
from ingestion.indexer import Indexer

log = get_logger(__name__)


@celery_app.task(bind=True, name="ingestion.ingest_paper_task")
def ingest_paper_task(self, file_path: str, paper_id: str) -> dict:
    """
    Celery task to ingest a single PDF paper into the retrieval index.

    This function orchestrates the complete ingestion pipeline:
    1. Parse the PDF to extract structured content
    2. Chunk the content into semantically coherent segments
    3. Embed each chunk using the configured embedding backend
    4. Index chunks into pgvector and BM25

    Args:
        file_path: Path to the PDF file to ingest
        paper_id: Paper identifier.

    Returns:
        The paper_id of the ingested paper.

    Raises:
        FileNotFoundError: If the PDF file doesn't exist
        Exception: Any error during parsing, chunking, embedding, or indexing
    """
    try:
        download_dir_path = Path(settings.download_dir)
        download_dir_path.mkdir(exist_ok=True)

        local_filepath = f"{settings.download_dir}/{file_path}.pdf"

        s3_client.download_file(settings.s3_bucket, file_path, local_filepath)

        file_path_obj = Path(local_filepath)
        if not file_path_obj.exists():
            raise FileNotFoundError(f"PDF not found: {file_path_obj}")

        log.info("Downloaded %s from storage bucket", file_path)

        # Store arguments in task metadata so they are accessible via AsyncResult.info
        self.update_state(
            state='STARTED',
            meta={'file_path': file_path, 'paper_id': paper_id}
        )

        log.info("Starting ingestion pipeline for %s (id=%s)", file_path_obj.name, paper_id)

        # Step 1: Parse PDF
        log.debug("Parsing PDF...")
        parser = PDFParser()
        doc = parser.parse(file_path_obj, paper_id=paper_id)
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

        # Step 4: Index chunks and save paper record in a single transaction
        log.debug("Indexing chunks...")
        with Indexer() as indexer:
            # Check if paper already exists
            if indexer.paper_exists(paper_id):
                log.warning("Paper %s already exists in index. Deleting and re-indexing.", paper_id)
                indexer.delete_paper(paper_id)

            indexer.index(chunks)
            log.debug("Successfully indexed %d chunks for %s", len(chunks), paper_id)

            # Save paper record to the papers table
            indexer.save_paper_record(doc, str(file_path_obj), len(chunks))
            log.info("Successfully indexed %d chunks and saved paper record for %s", len(chunks), paper_id)

        log.info("Ingestion complete for %s (id=%s)", file_path_obj.name, paper_id)

        file_path_obj.unlink(missing_ok=True)

        return {
            "paper_id": paper_id,
            "file_path": file_path
        }
    except Exception as e:
        log.error("Ingestion failed for %s (id=%s): %s", file_path, paper_id, traceback.format_exc())
        
        raise