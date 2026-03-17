"""Shared pytest fixtures for ScholarRAG tests."""

import pytest

from data_models.models import Chunk, RetrievedChunk


@pytest.fixture
def sample_chunk() -> Chunk:
    return Chunk(
        chunk_id="arxiv_1706.03762_introduction_0",
        paper_id="arxiv_1706.03762",
        text=(
            "The dominant sequence transduction models are based on complex recurrent "
            "or convolutional neural networks that include an encoder and a decoder."
        ),
        section="Introduction",
        page=1,
        chunk_index=0,
        title="Attention Is All You Need",
        authors=["Vaswani, A.", "Shazeer, N."],
        year=2017,
        arxiv_id="1706.03762",
    )


@pytest.fixture
def sample_retrieved_chunk(sample_chunk: Chunk) -> RetrievedChunk:
    return RetrievedChunk(
        chunk=sample_chunk,
        score=0.91,
        dense_rank=1,
        sparse_rank=3,
        rerank_score=0.91,
    )
