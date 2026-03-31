# 📚 ScholarRAG — Retrieval-Augmented Generation for Academic Research

> Ask questions across research papers. Get cited, grounded answers.  

---

## Why This Exists

LLMs hallucinate. Academic research demands precision. ScholarRAG bridges the gap by grounding every answer in retrieved source material — with inline citations traceable to the original paper, section, and page.

ScholarRAG is designed to handle real challenges in academic RAG:
- **Dense academic language** that defeats naive keyword search
- **Long papers** that require smarter chunking than fixed token windows
- **Multi-paper queries** that demand cross-document reasoning
- **Evaluation** so you know when your retrieval actually works

---

## Architecture

```
Ingestion Pipeline
┌──────────────────────────────────────────────────────────────┐
│  PDF / arXiv → Parser → Semantic Chunker → Embeddings        │
│                           └──────────────→ BM25 Index        │
└──────────────────────────────────────────────────────────────┘

Query & Retrieval Pipeline
┌──────────────────────────────────────────────────────────────┐
│  User Query → HyDE Expansion → Hybrid Retriever (RRF)        │
│                                └──→ Reranker → Context       │
└──────────────────────────────────────────────────────────────┘

Generation
┌──────────────────────────────────────────────────────────────┐
│  Context → Prompt Builder → LLM → Grounded Response          │
│                     ↳ Eval: Faithfulness · Recall · MRR      │
└──────────────────────────────────────────────────────────────┘
```

See [`docs/architecture.md`](docs/architecture.md) for component-level detail.

---

## Technical Highlights

### 🔍 Hybrid Retrieval (Dense + Sparse)
Dense vector search alone misses exact terminology — critical in scientific literature. ScholarRAG combines:
- **Dense retrieval** via a configurable embedding backend (see below), vectors stored in pgvector
- **BM25 keyword search** via `rank_bm25` as a complementary sparse signal
- **Reciprocal Rank Fusion (RRF)** to merge ranked lists without score normalization headaches

**Embedding backends** — switch via `EMBED_BACKEND`:
- **Local:** vLLM pooling models — runs on GPU, OpenAI-compatible API, no API key needed
- **Hosted:** `text-embedding-3-large` (OpenAI) or `embed-english-v3.0` (Cohere) for maximum retrieval quality

### 📄 Semantic Chunking for Research Papers
Fixed-size chunking destroys paper structure. Instead:
- Parse PDFs with `PyMuPDF` + `pdfplumber` to extract sections, abstracts, figure captions, and references separately
- Apply **sliding window chunking** (512 tokens, 128 overlap) within sections
- Store **metadata per chunk**: paper ID, section, page, authors, year, DOI

### 🔄 HyDE Query Expansion
Short natural language queries fail against dense academic text. [HyDE (Hypothetical Document Embeddings)](https://arxiv.org/abs/2212.10496) generates a hypothetical answer first, embeds *that*, and uses it to retrieve — dramatically improving recall on technical queries.

### 🏆 Cross-Encoder Reranking
After initial retrieval (top-50), a reranker rescores chunk-query pairs for precision. Top-8 chunks form the context window.

**Reranker backends** — switch via `RERANKER_BACKEND`:
- **Local:** vLLM pooling models with reranking — runs on GPU, OpenAI-compatible API, no API key needed
- **Hosted:** Cohere Rerank API — best out-of-the-box quality, easy to swap in for production

### 🖥️ Local LLM Support
ScholarRAG runs fully offline. The LLM backend is provider-agnostic:
- **Ollama** — run `llama3`, `mistral`, `phi3`, or any GGUF model locally via a drop-in OpenAI-compatible endpoint
- **vLLM** — production-grade local inference with PagedAttention for higher throughput
- **Hosted fallback** — OpenAI / Cohere APIs still supported via the same interface

Switch backends with a single env variable: `LLM_BACKEND=ollama|vllm|openai|cohere`

### 📊 Evaluation Built In
RAG without eval is guesswork. ScholarRAG ships with:
- **Faithfulness** — does the answer contradict any retrieved chunk? (LLM-as-judge)
- **Answer Recall** — does the answer cover the ground truth key points?
- **MRR / Hit@k** — is the relevant chunk in the top-k retrieved?
- **Context Precision** — are retrieved chunks actually relevant to the question?

Eval harness uses [RAGAS](https://github.com/explodinggradients/ragas) + a curated test set of 50 paper-question-answer triples.

---

## Quickstart

```bash
# Clone and install
git clone https://github.com/yourusername/scholar-rag
cd scholar-rag
pip install -r requirements.txt

# Set credentials
cp .env.example .env
# Choose your backends (all default to local if unset):
#
#   EMBED_BACKEND=vllm          (local vLLM, default)
#   EMBED_BACKEND=ollama        (local via Ollama)
#   EMBED_BACKEND=openai        (hosted)
#   EMBED_BACKEND=cohere        (hosted)
#
#   RERANKER_BACKEND=vllm       (local vLLM, default)
#   RERANKER_BACKEND=cohere     (hosted, higher quality)
#
#   LLM_BACKEND=ollama          (local, recommended)
#   LLM_BACKEND=vllm            (local, high-throughput)
#   LLM_BACKEND=openai          (hosted fallback)
#   LLM_BACKEND=cohere          (hosted fallback)

# Fully local — zero API keys required (vLLM for embeddings/reranking, Ollama for LLM)
export EMBED_BACKEND=vllm
export VLLM_EMBED_MODEL=intfloat/e5-mistral-7b-instruct
export RERANKER_BACKEND=vllm
export VLLM_RERANKER_MODEL=BAAI/bge-reranker-v2-minimum
export LLM_BACKEND=ollama
export OLLAMA_MODEL=llama3
ollama pull llama3

# Hosted embeddings + reranker (higher quality, needs API keys)
export EMBED_BACKEND=openai
export OPENAI_API_KEY=sk-...
export RERANKER_BACKEND=cohere
export COHERE_API_KEY=...

# Ingest a paper
python -m ingestion.pdf_parser --input papers/attention_is_all_you_need.pdf
python -m ingestion.embedder --run-all

# Run the API
uvicorn api.main:app --reload

# Run the Streamlit demo
streamlit run frontend/app.py

# Run evals
python eval/run_eval.py --test-set eval/test_set.json
```

---

## Design Decisions & Tradeoffs

**Why support local embeddings?**
vLLM pooling models give you access to high-quality embedding models that run entirely offline with high throughput. For a project ingesting potentially sensitive research, keeping embeddings local removes a meaningful data exposure risk — and it eliminates per-token API costs during the ingestion phase entirely.

**Why two reranker options (local vs hosted)?**
vLLM reranking runs on GPU with no API key, which is the right default for development and private use. Cohere Rerank is the upgrade path for production — best-in-class quality without the overhead of running your own inference.

**Why support local LLMs at all?**  
Two reasons: zero inference cost during development, no data leaving your machine (important for proprietary research).

**Why HyDE over multi-query expansion?**  
HyDE generates one high-quality hypothetical document. Multi-query generates N queries. For academic search, one good embedding outperforms averaging N noisy ones — at lower cost.

**Why not ColBERT (late interaction)?**  
ColBERT is genuinely better for this domain. Excluded from v1 due to infrastructure cost and index complexity. Tracked in `docs/future_work.md`.
