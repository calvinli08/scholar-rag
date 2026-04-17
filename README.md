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

**Embedding backends** — switch via `EMBED_BACKEND`:
- **Local:** Qwen on vLLM — Qwen embedding models hosted on vLLM server, runs on GPU, OpenAI-compatible API, no API key needed
- **Hosted:** `text-embedding-3-*` (OpenAI) or `gemini-embedding-*` (Google) for maximum retrieval quality

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
- **Local:** Qwen on vLLM — Qwen reranker models hosted on vLLM server, runs on GPU, OpenAI-compatible API, no API key needed
- **Hosted:** Embeds text using Gemini and OpenAI embedding models, and then reranks text based on cosine similarity to input query

### 🖥️ Local LLM Support
ScholarRAG runs fully offline. The LLM backend is provider-agnostic:
- **Qwen on vLLM** — Qwen models hosted on vLLM server for production-grade local inference with PagedAttention for higher throughput
- **Hosted fallback** — OpenAI / Gemini APIs still supported via the same interface

Switch backends with a single env variable: `LLM_BACKEND=vllm||openai|gemini`

### 📊 Evaluation Built In
RAG without eval is guesswork. ScholarRAG ships with:
- **Faithfulness** — does the answer contradict any retrieved chunk? (LLM-as-judge)
- **Answer Recall** — does the answer cover the ground truth key points?
- **MRR / Hit@k** — is the relevant chunk in the top-k retrieved?
- **Context Precision** — are retrieved chunks actually relevant to the question?

Eval harness uses [RAGAS](https://github.com/explodinggradients/ragas) + a curated test set of 50 paper-question-answer triples.

---

## Supported Models

### OpenAI Models

#### OpenAI Embedding Models
| Model | Dimensions | Description |
|-------|------------|-------------|
| `text-embedding-3-large` | 1024 (default) | Best quality, recommended for production |
| `text-embedding-3-small` | 1024 | Faster, lower cost alternative |

#### OpenAI Text Generation Models
| Model       | Description                                                |
| ----------- | ---------------------------------------------------------- |
| GPT-5       | OpenAI’s flagship reasoning model for complex tasks.       |
| GPT-4o      | General-purpose multimodal model for text and image input. |
| GPT-4o mini | Smaller, lower-cost multimodal model.                      |
| GPT-4.1     | Strong general text-generation model.                      |
| o3          | Reasoning-focused model for harder prompts.                |

### Gemini Models

#### Gemini Embedding Models
| Model | Dimensions | Description |
|-------|------------|-------------|
| `gemini-embedding-001` | 1536 (default) | First-gen Gemini embeddings |
| `gemini-embedding-002` | 1536 | Improved Gemini embeddings |

### Gemini Text Generation Models
| Model                 | Description                                                                |
| --------------------- | -------------------------------------------------------------------------- |
| Gemini 2.5 Pro        | High-capability model for complex reasoning, coding, and multimodal tasks. |
| Gemini 2.5 Flash      | Fast, balanced model for strong performance and lower latency.             |
| Gemini 2.5 Flash-Lite | Cost-effective, high-throughput model optimized for efficiency.            |
| Gemini 3.1 Pro        | Latest reasoning-first model for complex agentic workflows and coding.     |
| Gemini 3.1 Flash      | Powerful agentic and coding model with strong multimodal understanding.    |
| Gemini 3.1 Flash-Lite | Most cost-efficient option for low-latency, high-volume use cases.         |

### Qwen Models (hosted on vLLM)
ScholarRAG exclusively supports Qwen family models when running locally on vLLM. This ensures optimal compatibility and performance.

**Embedding Models:**
| Model | Dimensions | Description |
|-------|------------|-------------|
| `Qwen/Qwen3-Embedding-0.6B` | 1024 (default) | Lightweight, fast inference |
| `Qwen/Qwen3-Embedding-4B` | 1024 | Balanced quality/speed |
| `Qwen/Qwen3-Embedding-8B` | 1024 | Highest quality embedding |

**Reranker Models:**
| Model | Description |
|-------|-------------|
| `Qwen/Qwen3-Reranker-0.6B` | Lightweight, fast reranking (default) |
| `Qwen/Qwen3-Reranker-4B` | Balanced quality/speed |
| `Qwen/Qwen3-Reranker-8B` | Highest quality reranking |

**LLM Models (Instruct variants for chat/completion):**
| Model | Description |
|-------|-------------|
| `Qwen/Qwen2.5-7B-Instruct` | Entry-level Qwen2.5 |
| `Qwen/Qwen2.5-14B-Instruct` | Mid-range Qwen2.5 |
| `Qwen/Qwen2.5-32B-Instruct` | High-quality Qwen2.5 |
| `Qwen/Qwen2.5-72B-Instruct` | Premium Qwen2.5 |
| `Qwen/Qwen3-8B-Instruct` | Entry-level Qwen3 |
| `Qwen/Qwen3-14B-Instruct` | Mid-range Qwen3 |
| `Qwen/Qwen3-32B-Instruct` | High-quality Qwen3 |
| `Qwen/Qwen3.5-9B` | Entry-level Qwen3.5 |
| `Qwen/Qwen3.5-27B-FP8` | Quantized for efficiency |
| `Qwen/Qwen3.5-397B-A17B-FP8` | Flagship model (default) |

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
