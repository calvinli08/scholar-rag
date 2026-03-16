"""
Comprehensive test suite for the ingestion module.

This module combines tests for:
1. Low-level components (chunker, parser helpers)
2. High-level pipeline (ingest function)
"""

from __future__ import annotations

import pytest
import uuid
from unittest.mock import patch, MagicMock, call

from ingestion.chunker import Chunker, _chunk_id, _sentence_split, _token_count
from ingestion.pdf_parser import ParsedDocument, Section, _clean_heading, _is_heading
from ingestion.pipeline import ingest


# =============================================================================
# CHUNKER TESTS
# =============================================================================

class TestSentenceSplit:
    def test_basic_split(self):
        text = "First sentence. Second sentence. Third sentence."
        parts = _sentence_split(text)
        assert len(parts) == 3

    def test_preserves_abbreviations(self):
        text = "As shown in Fig. 3, the results improve. See Eq. 4 for details."
        parts = _sentence_split(text)
        # Should NOT split on "Fig." or "Eq."
        assert len(parts) == 2

    def test_et_al_not_split(self):
        text = "As noted by Smith et al. (2023), accuracy improved. This was replicated."
        parts = _sentence_split(text)
        assert len(parts) == 2


class TestTokenCount:
    def test_non_zero(self):
        assert _token_count("hello world") > 0

    def test_longer_text_more_tokens(self):
        short = _token_count("short text")
        long = _token_count("this is a much longer piece of text with many more words")
        assert long > short


class TestChunkId:
    def test_format(self):
        cid = _chunk_id("paper123", "Introduction", 0)
        assert "paper123" in cid
        assert "introduction" in cid
        assert "0000" in cid

    def test_special_chars_stripped(self):
        cid = _chunk_id("paper", "Related Work & Background", 1)
        assert " " not in cid
        assert "&" not in cid


class TestChunker:
    def _make_doc(self, text: str, paper_id: str = "test_paper") -> ParsedDocument:
        doc = ParsedDocument(paper_id=paper_id, file_path="")
        doc.sections = [
            Section(title="Introduction", text=text, page_start=0, page_end=1, section_index=0)
        ]
        return doc

    def test_produces_chunks(self):
        text = " ".join(["This is a sentence about deep learning."] * 40)
        chunker = Chunker(chunk_size=64, chunk_overlap=16)
        doc = self._make_doc(text)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) > 1

    def test_chunk_size_respected(self):
        text = " ".join(["Word"] * 2000)
        chunker = Chunker(chunk_size=100, chunk_overlap=20)
        doc = self._make_doc(text)
        chunks = chunker.chunk_document(doc)
        for c in chunks[:-1]:  # last chunk may be smaller
            assert _token_count(c.text) <= 120  # small tolerance

    def test_overlap_creates_shared_content(self):
        sentences = [f"Sentence number {i} contains some words." for i in range(30)]
        text = " ".join(sentences)
        chunker = Chunker(chunk_size=60, chunk_overlap=20)
        doc = self._make_doc(text)
        chunks = chunker.chunk_document(doc)
        assert len(chunks) >= 2
        # Overlap: end of chunk N and start of chunk N+1 should share tokens
        end_of_first = chunks[0].text.split()[-5:]
        start_of_second = chunks[1].text.split()[:10]
        overlap = set(end_of_first) & set(start_of_second)
        assert len(overlap) > 0

    def test_references_skipped(self):
        doc = ParsedDocument(paper_id="p", file_path="")
        doc.sections = [
            Section("Introduction", "Some intro text here.", 0, 0, 0),
            Section("References", "[1] Smith et al. 2020", 5, 6, 1),
        ]
        chunker = Chunker()
        chunks = chunker.chunk_document(doc)
        assert all(c.section != "References" for c in chunks)

    def test_metadata_propagated(self):
        doc = ParsedDocument(paper_id="myid", file_path="")
        doc.title = "My Paper"
        doc.authors = ["Alice", "Bob"]
        doc.year = 2023
        doc.sections = [
            Section("Abstract", "We propose a new method.", 0, 0, 0)
        ]
        chunker = Chunker()
        chunks = chunker.chunk_document(doc)
        assert chunks[0].paper_id == "myid"
        assert chunks[0].title == "My Paper"
        assert chunks[0].authors == ["Alice", "Bob"]
        assert chunks[0].year == 2023

    def test_invalid_overlap_raises(self):
        with pytest.raises(ValueError):
            Chunker(chunk_size=100, chunk_overlap=100)

    def test_chunk_text_convenience(self):
        chunker = Chunker(chunk_size=50, chunk_overlap=10)
        chunks = chunker.chunk_text("Hello world. " * 50, paper_id="test")
        assert len(chunks) > 0
        assert all(c.paper_id == "test" for c in chunks)


# =============================================================================
# PARSER HELPER TESTS
# =============================================================================

class TestHeadingDetection:
    def test_known_heading(self):
        assert _is_heading("Abstract", 12.0, 11.0) is True
        assert _is_heading("Introduction", 12.0, 11.0) is True

    def test_numbered_heading(self):
        assert _is_heading("1 Introduction", 12.0, 11.0) is True
        assert _is_heading("2.1 Related Work", 12.0, 11.0) is True

    def test_large_font_is_heading(self):
        assert _is_heading("Some Section Title", 16.0, 11.0) is True

    def test_body_text_not_heading(self):
        body = (
            "In this paper we propose a novel approach to retrieval-augmented "
            "generation that combines dense and sparse retrieval signals."
        )
        assert _is_heading(body, 11.0, 11.0) is False

    def test_clean_heading_strips_numbering(self):
        assert _clean_heading("1 Introduction") == "Introduction"
        assert _clean_heading("2.1 Related Work") == "Related Work"
        assert _clean_heading("III. Methods") == "Methods"


# =============================================================================
# PIPELINE INTEGRATION TESTS
# =============================================================================

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
