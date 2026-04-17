# 📚 ScholarRAG — Retrieval-Augmented Generation for Academic Papers

> Ask questions across research papers. Get cited, grounded answers.  

ScholarRAG is designed to handle real challenges in academic RAG:
- **Dense academic language** that defeats naive keyword search
- **Long papers** that require smarter chunking than fixed token windows
- **Multi-paper queries** that demand cross-document reasoning
- **Evaluation** so you know when your retrieval actually works

Supports OpenAI, Gemini, and Qwen (via vLLM).

---

## Architecture

```
Ingestion Pipeline
┌──────────────────────────────────────────────────────────────┐
│  PDF → Parser → Semantic Chunker → Embeddings                │
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

See [`docs/architecture.html`](docs/architecture.html) for component-level detail.

---

## Technical Highlights

### 🔍 Hybrid Retrieval (Dense + Sparse)
Dense vector search alone misses exact terminology — critical in scientific literature. ScholarRAG combines:
- **Dense retrieval** via a configurable embedding backend (see below), vectors stored in pgvector
- **BM25 keyword search** via `rank_bm25` as a complementary sparse signal
- **Reciprocal Rank Fusion (RRF)** to merge ranked lists without score normalization headaches

### 📄 Semantic Chunking for Research Papers
Fixed-size chunking destroys paper structure. Instead:
- Parse PDFs with `PyMuPDF` + `pdfplumber` to extract sections, abstracts, figure captions, and references separately
- Apply **sliding window chunking** (512 tokens, 128 overlap) within sections
- Store **metadata per chunk**: paper ID, section, page, authors, year, DOI

### 🔄 HyDE Query Expansion
Short natural language queries fail against dense academic text. [HyDE (Hypothetical Document Embeddings)](https://arxiv.org/abs/2212.10496) generates a hypothetical answer first, embeds *that*, and uses it to retrieve — dramatically improving recall on technical queries.

### 🏆 Cross-Encoder Reranking
After initial retrieval (top-50), a reranker rescores chunk-query pairs for precision. Top-8 chunks form the context window.

### Multi-Model Support

Switch providers via the `MODEL_BACKEND` environment variable

**Embedding backends**
- **Qwen:** Qwen embedding models hosted on vLLM server, runs on GPU, OpenAI-compatible API, no API key needed
- **Gemini:** `gemini-embedding-*` via Gemini API
- **OpenAI:** `text-embedding-3-*` via OpenAI API

**Reranker backends**
- **Qwen:** Qwen reranker models hosted on vLLM server, runs on GPU, OpenAI-compatible API, no API key needed
- **Gemini:** Embeds text using Gemini embedding API, and then reranks text based on cosine similarity to the input query
- **OpenAI:** Embeds text using OpenAI embedding API, and then reranks text based on cosine similarity to the input query

**Text generation backends**
- **Qwen:** Qwen text generation models hosted on vLLM server, runs on GPU, OpenAI-compatible API, no API key needed
- **Gemini:** Gemini text generation models (2.5, 3.0, etc) available via API
- **OpenAI:** OpenAI text generation models (GPT-4o, GPT-4.1, GPT-5, etc) available via API

Switch backends with a single env variable: `LLM_BACKEND=vllm||openai|gemini`

### 📊 Evaluation Built In
RAG without eval is guesswork. ScholarRAG ships with DeepEval to measure:
- **Faithfulness** — does the answer contradict any retrieved chunk? (LLM-as-judge)
- **Answer Recall** — does the answer cover the ground truth key points?
- **MRR / Hit@k** — is the relevant chunk in the top-k retrieved?
- **Context Precision** — are retrieved chunks actually relevant to the question?

---

## Quickstart

### OpenAI & Gemini Quickstart

To run ScholarRAG with OpenAI or Gemini models, simply update the `.env` file with your preferred embedding and text-generation models, along with your API key.

### Qwen on vLLM Quickstart

1. Install vLLM [instructions](https://docs.vllm.ai/en/stable/getting_started/installation/)
2. Update `.env` values with your preferred Qwen models. You can find text generation, embedding, and reranker models on [HuggingFace](https://huggingface.co/Qwen).
3. Start three distinct vLLM instances, one for each of the text generation, embedding, and reranker models. Ensure that each model has sufficient context length, GPU memory allocation, and its own HTTP port. You will need to change model sizes or GPUs if there is insufficient memory.

---

## Design Decisions & Tradeoffs

**Why support local embeddings?**
Qwen embedding models hosted on vLLM server give you access to high-quality embedding models that run entirely offline with high throughput. For a project ingesting potentially sensitive research, keeping embeddings local removes a meaningful data exposure risk — and it eliminates per-token API costs during the ingestion phase entirely.

**Why two reranker options (local vs hosted)?**
Qwen reranker models hosted on vLLM server run on GPU with no API key, which is the right default for development and private use. Hosted reranking is the upgrade path for production — best-in-class quality without the overhead of running your own inference.

**Why support local LLMs at all?**  
Two reasons: zero inference cost during development, no data leaving your machine (important for proprietary research).

**Why HyDE over multi-query expansion?**  
HyDE generates one high-quality hypothetical document. Multi-query generates N queries. For academic search, one good embedding outperforms averaging N noisy ones — at lower cost.
