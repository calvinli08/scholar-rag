"""
Test suite for the ingestion pipeline.

This module tests the `ingest` function in `ingestion.pipeline`, covering:
- Successful ingestion flows
- Error handling (missing files, empty content)
- Idempotency (re-indexing logic)
- Interaction with dependencies (parser, chunker, embedder, indexer)
"""

import pytest
import uuid
from unittest.mock import patch, MagicMock, call

from ingestion.pipeline import ingest


class TestIngestPipeline:
    """Test cases for the ingest pipeline."""

    @pytest.fixture
    def mock_parse_pdf(self):
        """Mock for pdf_parser.parse_pdf."""
        with patch('ingestion.pipeline.parse_pdf') as mock:
            yield mock

    @pytest.fixture
    def mock_chunk_text(self):
        """Mock for chunker.chunk_text."""
        with patch('ingestion.pipeline.chunk_text') as mock:
            yield mock

    @pytest.fixture
    def mock_get_embedder(self):
        """Mock for get_embedder factory."""
        with patch('ingestion.pipeline.get_embedder') as mock:
            # Setup default embedder behavior
            mock_embedder = MagicMock()
            mock_embedder.embed_documents.return_value = [[0.1] * 1536]
            mock.return_value = mock_embedder
            yield mock

    @pytest.fixture
    def mock_indexer_cls(self):
        """Mock for Indexer class."""
        with patch('ingestion.pipeline.Indexer') as mock:
            mock_instance = MagicMock()
            mock.return_value = mock_instance
            yield mock_instance

    def test_ingest_success_generates_id(
        self, mock_parse_pdf, mock_chunk_text, mock_get_embedder, mock_indexer_cls
    ):
        """
        Objective: Verify successful ingestion generates a UUID when none provided.
        Importance: Ensures the system can auto-assign unique IDs to new papers.
        """
        # Arrange
        mock_parse_pdf.return_value = {
            "text": "Sample content for testing",
            "metadata": {"title": "Test Paper"}
        }
        mock_chunk_text.return_value = [
            {"content": "Chunk 1", "metadata": {"page": 1}},
            {"content": "Chunk 2", "metadata": {"page": 1}}
        ]

        # Act
        result_id = ingest("dummy_path.pdf", paper_id=None)

        # Assert
        assert result_id is not None
        assert isinstance(result_id, str)
        try:
            uuid.UUID(result_id, version=4)
        except ValueError:
            pytest.fail("Generated ID is not a valid UUID")

        mock_parse_pdf.assert_called_once_with("dummy_path.pdf")
        mock_indexer_cls.assert_called_once()
        mock_indexer_cls.return_value.index_chunks.assert_called_once()

    def test_ingest_uses_provided_id(
        self, mock_parse_pdf, mock_chunk_text, mock_get_embedder, mock_indexer_cls
    ):
        """
        Objective: Verify ingestion uses user-provided paper_id.
        Importance: Allows external systems to manage paper identifiers.
        """
        custom_id = "my-custom-paper-123"
        
        # Arrange
        mock_parse_pdf.return_value = {"text": "Content", "metadata": {}}
        mock_chunk_text.return_value = [{"content": "Chunk 1", "metadata": {}}]

        # Act
        result_id = ingest("dummy_path.pdf", paper_id=custom_id)

        # Assert
        assert result_id == custom_id
        
        # Verify delete and index were called with our ID
        mock_indexer_cls.return_value.delete_paper.assert_called_once_with(custom_id)
        mock_indexer_cls.return_value.index_papers.assert_called_once()
        
        # Check that the paper data passed to index_papers contains our ID
        call_args = mock_indexer_cls.return_value.index_papers.call_args[0][0]
        assert call_args["paper_id"] == custom_id

    def test_ingest_file_not_found(self):
        """
        Objective: Handle non-existent file path.
        Importance: Provides clear error feedback when input file is missing.
        """
        # Arrange
        nonexistent_path = "/non/existent/path.pdf"

        # Act & Assert
        with pytest.raises(FileNotFoundError):
            ingest(nonexistent_path)

    def test_ingest_empty_text_handling(
        self, mock_parse_pdf, mock_indexer_cls
    ):
        """
        Objective: Handle PDF with no extractable text.
        Importance: Prevents crashes on empty or image-only PDFs.
        """
        # Arrange
        mock_parse_pdf.return_value = {"text": "", "metadata": {"title": "Empty"}}
        # chunk_text should return empty list for empty text
        with patch('ingestion.pipeline.chunk_text', return_value=[]):
            with patch('ingestion.pipeline.get_embedder') as mock_get_emb:
                mock_embedder = MagicMock()
                mock_embedder.embed_documents.return_value = []
                mock_get_emb.return_value = mock_embedder

                # Act
                result_id = ingest("dummy.pdf")

                # Assert
                assert result_id is not None
                # Verify index_chunks was called with empty list
                mock_indexer_cls.return_value.index_chunks.assert_called_once()
                call_args = mock_indexer_cls.return_value.index_chunks.call_args[0][0]
                assert len(call_args) == 0

    def test_ingest_reindex_cleanup_order(
        self, mock_parse_pdf, mock_chunk_text, mock_get_embedder, mock_indexer_cls
    ):
        """
        Objective: Verify idempotency - old chunks are removed BEFORE new ones inserted.
        Importance: Critical for data integrity; prevents duplicate search results on re-upload.
        """
        paper_id = "test-reindex-001"
        
        # Arrange
        mock_parse_pdf.return_value = {"text": "New Content", "metadata": {}}
        mock_chunk_text.return_value = [{"content": "New Chunk", "metadata": {}}]

        # Act
        ingest("dummy.pdf", paper_id=paper_id)

        # Assert: Verify call order
        method_calls = [call_arg[0] for call_arg in mock_indexer_cls.method_calls]
        method_names = [name[0] for name in method_calls]

        assert 'delete_paper' in method_names, "delete_paper was not called"
        assert 'index_chunks' in method_names, "index_chunks was not called"
        
        delete_index = method_names.index('delete_paper')
        insert_index = method_names.index('index_chunks')
        
        assert delete_index < insert_index, (
            f"delete_paper (index {delete_index}) must be called before "
            f"index_chunks (index {insert_index})"
        )

    def test_ingest_propagates_parse_error(
        self, mock_parse_pdf
    ):
        """
        Objective: Ensure parsing errors are propagated to the caller.
        Importance: Allows the API layer to catch and report specific ingestion failures.
        """
        # Arrange
        mock_parse_pdf.side_effect = Exception("PDF parsing failed")

        # Act & Assert
        with pytest.raises(Exception, match="PDF parsing failed"):
            ingest("corrupted.pdf")

    def test_ingest_embedding_call(
        self, mock_parse_pdf, mock_chunk_text, mock_get_embedder, mock_indexer_cls
    ):
        """
        Objective: Verify embeddings are generated for all chunks.
        Importance: Ensures vector search capability is maintained.
        """
        # Arrange
        chunks = [
            {"content": "Chunk A", "metadata": {}},
            {"content": "Chunk B", "metadata": {}},
            {"content": "Chunk C", "metadata": {}}
        ]
        mock_parse_pdf.return_value = {"text": "Content", "metadata": {}}
        mock_chunk_text.return_value = chunks
        
        mock_embedder = MagicMock()
        # Return 3 vectors of dimension 1536
        mock_embedder.embed_documents.return_value = [[0.1] * 1536 for _ in range(3)]
        mock_get_embedder.return_value = mock_embedder

        # Act
        ingest("dummy.pdf")

        # Assert
        # Extract texts from chunks to verify what was sent to embedder
        expected_texts = [c["content"] for c in chunks]
        mock_embedder.embed_documents.assert_called_once_with(expected_texts)
        
        # Verify chunks passed to indexer have embeddings attached
        index_call_args = mock_indexer_cls.return_value.index_chunks.call_args[0][0]
        for chunk in index_call_args:
            assert "embedding" in chunk
            assert len(chunk["embedding"]) == 1536
