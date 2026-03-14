"""Unit tests for the ingestion pipeline."""

from __future__ import annotations

import pytest

from ingestion.chunker import Chunker, _chunk_id, _sentence_split, _token_count
from ingestion.pdf_parser import ParsedDocument, Section, _clean_heading, _is_heading


# ---------------------------------------------------------------------------
# Chunker
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Parser helpers
# ---------------------------------------------------------------------------

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
