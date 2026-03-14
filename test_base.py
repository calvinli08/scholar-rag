"""Smoke tests — verify config loads and models instantiate correctly."""

from models import Chunk, QueryResult, RetrievedChunk


def test_settings_load():
    from config import get_settings
    s = get_settings()
    assert s.chunk_size == 512
    assert s.chunk_overlap == 128
    assert s.reranker_top_k < s.reranker_initial_k


def test_chunk_repr(sample_chunk: Chunk):
    r = repr(sample_chunk)
    assert "arxiv_1706.03762" in r


def test_retrieved_chunk_passthrough(sample_retrieved_chunk: RetrievedChunk):
    assert sample_retrieved_chunk.chunk_id == "arxiv_1706.03762_introduction_0"
    assert sample_retrieved_chunk.text.startswith("The dominant")


def test_query_result_token_total():
    qr = QueryResult(
        query="What is attention?",
        answer="Attention is a mechanism...",
        prompt_tokens=200,
        completion_tokens=80,
    )
    assert qr.total_tokens == 280
