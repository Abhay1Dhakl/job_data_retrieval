# Job Data RAG Pipeline — Project Documentation

**Project:** Job Data Retrieval (RAG Pipeline)
**Version:** 1.0
**Date:** March 2026
**Author:** Abhay Dhakal

---

## Table of Contents

1. [High-Level Architecture & Engineering Decisions](#1-high-level-architecture--engineering-decisions)
2. [Setup & Installation Instructions](#2-setup--installation-instructions)
3. [Example Usage — Requests & Expected Responses](#3-example-usage--requests--expected-responses)
4. [Assumptions Made During Development](#4-assumptions-made-during-development)
5. [Drawbacks & Future Enhancements](#5-drawbacks--future-enhancements)

---

## 1. High-Level Architecture & Engineering Decisions

### 1.1 System Overview

The system is a **production-ready Retrieval-Augmented Generation (RAG) pipeline** built on top of the LF Jobs dataset. Given a natural-language query (e.g., *"senior data engineer in remote"*), the system retrieves the most relevant job listings from a vector store, optionally reranks them, and uses a Large Language Model (LLM) to synthesize a concise, human-readable answer.

```
                     ┌──────────────┐
                     │   Next.js UI │  (port 3000)
                     └──────┬───────┘
                            │ HTTP
                     ┌──────▼───────┐
                     │  FastAPI API │  (port 8000)
                     └──────┬───────┘
               ┌────────────┼────────────────┐
               │            │                │
        ┌──────▼──────┐ ┌───▼──────┐  ┌─────▼──────┐
        │  Embeddings  │ │  Redis   │  │ PostgreSQL │
        │ (e5-large-v2)│ │  Cache   │  │  (logs)    │
        └──────┬───────┘ └──────────┘  └────────────┘
               │
     ┌─────────▼──────────┐
     │   Pinecone (Vector) │
     │   + BM25 (optional) │
     └─────────┬───────────┘
               │
     ┌─────────▼──────────┐
     │   Cross-Encoder     │
     │   Reranker (opt.)   │
     └─────────┬───────────┘
               │
     ┌─────────▼──────────┐
     │  OpenAI-Compatible  │
     │       LLM           │
     └────────────────────┘
```

### 1.2 Pipeline Stages

| Stage | Component | File |
|-------|-----------|------|
| Data Ingestion & Chunking | CSV reader + HTML cleaner | `backend/scripts/build_index.py` |
| Embedding | Sentence-Transformers `EmbeddingModel` | `backend/app/rag/embeddings/` |
| Vector Storage | Pinecone `PineconeVectorStore` | `backend/app/rag/retrieval/` |
| Lexical Index | BM25 `BM25Index` (pickle) | `backend/app/rag/retrieval/` |
| Retrieval | `Retriever` (vector + optional hybrid) | `backend/app/rag/retrieval/` |
| Reranking | `CrossEncoderReranker` (optional) | `backend/app/rag/retrieval/` |
| Generation | `OpenAICompatibleClient` + prompt | `backend/app/rag/llm/`, `backend/app/rag/prompts/` |
| Orchestration | `RagPipeline.run()` | `backend/app/rag/pipeline.py` |
| API Layer | FastAPI route `POST /api/query` | `backend/app/api/routes.py` |
| Caching | Redis TTL cache | `backend/app/core/cache.py` |
| Configuration | Pydantic `Settings` | `backend/app/core/config.py` |

### 1.3 Engineering Decisions & Rationale

#### 1.3.1 Local Embeddings — `intfloat/e5-large-v2` (1024-dim)

**Decision:** Use a locally-hosted Sentence-Transformers model instead of a paid embedding API.

**Reasoning:**
- Eliminates per-token costs and external API latency for embedding inference.
- `e5-large-v2` is state-of-the-art on MTEB benchmarks for retrieval tasks, achieving strong semantic similarity without the overhead of GPT embeddings.
- The model requires specific prefixes: `query:` for search queries and `passage:` for indexed documents — this is enforced in the `EmbeddingModel` wrapper to prevent silent ranking degradation.
- The model is pre-downloaded at Docker image build time via `ARG EMBEDDING_MODEL`, ensuring cold-start reliability in production.

#### 1.3.2 Pinecone as the Vector Store

**Decision:** Use Pinecone (managed cloud) over self-hosted solutions like Qdrant, Weaviate, or pgvector.

**Reasoning:**
- Pinecone provides a fully managed, horizontally scalable vector index with low-latency approximate nearest-neighbor (ANN) search.
- No operational overhead (no need to manage HNSW index tuning or RAM provisioning).
- Supports metadata filtering (by `company`, `location`, `level`, `category`, `publication_date`) natively at query time, which is applied as `PreFilterOptions` before ANN search, avoiding post-hoc filtering on large result sets.
- The index dimension (1024) must match the embedding model; this is enforced at pipeline build time.
- Cosine similarity metric is used as it normalizes for vector magnitude, making it appropriate for semantic similarity tasks.

#### 1.3.3 Optional BM25 Hybrid Retrieval

**Decision:** Implement an optional BM25 lexical index that blends with vector scores using configurable alpha weighting.

**Reasoning:**
- Pure dense vector retrieval can miss exact keyword matches (e.g., a user searching for a specific programming language or certification).
- BM25 (Best Matching 25) adds lexical precision by scoring based on term frequency and inverse document frequency.
- Scores from both indexes are independently min-max normalized before blending: `hybrid_score = alpha × vector_score + (1 − alpha) × bm25_score`.
- The `hybrid_alpha` defaults to `0.35`, slightly weighting BM25, giving it a supportive rather than dominant role.
- The BM25 index is serialized as `bm25.pkl` and loaded at startup, keeping it fast and dependency-free in production.
- Hybrid retrieval is opt-in (`USE_HYBRID=false` by default) to keep default deployments simple.

#### 1.3.4 Optional Cross-Encoder Reranking

**Decision:** Add a second-pass reranker (`cross-encoder/ms-marco-MiniLM-L-6-v2`) after retrieval.

**Reasoning:**
- Vector and BM25 retrievers use bi-encoder or term-based scoring, which are computationally efficient but do not model the full query-document interaction.
- A cross-encoder sees the query and each candidate jointly, producing a more accurate relevance score at the cost of higher latency (O(n) inference calls).
- By running the cross-encoder only on the top-K retrieved candidates (not the full corpus), the cost remains bounded.
- Using a small MiniLM cross-encoder keeps latency low while providing meaningful ranking improvement.
- Reranking is enabled by default in `.env.example` but can be disabled per request via `use_rerank: false`.

#### 1.3.5 OpenAI-Compatible LLM Interface

**Decision:** Design the LLM client against the OpenAI `chat/completions` API contract rather than any specific provider SDK.

**Reasoning:**
- Decouples the application from any single provider — the same code works with OpenAI, Azure OpenAI, Groq, Mistral, Ollama (local), or any other OpenAI-compatible endpoint.
- Configured entirely via environment variables: `LLM_BASE_URL`, `LLM_API_KEY`, `LLM_MODEL`.
- If `LLM_API_KEY` is not set, the pipeline gracefully falls back to returning retrieval-only results with a clear message, so the service is still functional without an LLM.

#### 1.3.6 Pydantic Settings for Configuration

**Decision:** Use `pydantic-settings` (`BaseSettings`) for all application configuration.

**Reasoning:**
- Provides type-safe, validated configuration loaded from environment variables and `.env` files.
- Single source of truth — `Settings` is cached via `@lru_cache` and injected via FastAPI's dependency injection system.
- Eliminates the need for manual `os.environ.get()` calls scattered throughout the codebase.
- Supports layered configuration: `.env.example` provides safe defaults; `.env` overrides for local development; Docker environment variables override for production.

#### 1.3.7 Docker Compose for Multi-Service Orchestration

**Decision:** Use Docker Compose to orchestrate the API, frontend, Redis, and PostgreSQL services.

**Reasoning:**
- Mirrors a production topology with a single `docker compose up --build` command.
- Redis caches repeated queries with a 5-minute TTL (`CACHE_TTL_SECONDS=300`), significantly reducing LLM API costs and latency for common queries.
- PostgreSQL is included for future persistence of query logs and analytics.
- Service dependencies (`api` depends on `redis` and `postgres`; `frontend` depends on `api`) are explicitly declared.

#### 1.3.8 Post-Retrieval Filtering

**Decision:** Support lightweight post-retrieval filters (`PostFilterOptions`) in addition to Pinecone pre-filters.

**Reasoning:**
- Some filters are difficult or expensive to express in Pinecone's metadata filter syntax (e.g., minimum word count of a job description snippet, or tag exclusion logic).
- Post-filters run in Python after retrieval, allowing arbitrary filtering logic without index changes.
- Supports `min_words`, `max_words`, `include_tags`, and `exclude_tags` as configurable options.

---

## 2. Setup & Installation Instructions

### 2.1 Prerequisites

| Requirement | Version | Notes |
|-------------|---------|-------|
| Python | ≥ 3.11 | Required |
| `uv` | latest | Fast Python package manager |
| Node.js | ≥ 18 | For Next.js frontend (optional) |
| Docker + Docker Compose | latest | For containerized deployment |
| Pinecone account | — | Free tier available at pinecone.io |
| OpenAI API key | — | Optional; needed for LLM-generated answers |

### 2.2 Local Development Setup

#### Step 1 — Clone the repository and navigate to the project

```bash
git clone <repository-url>
cd job_data_retrival
```

#### Step 2 — Install `uv` (if not already installed)

```bash
# macOS / Linux
brew install uv
# or
pipx install uv
```

#### Step 3 — Create a virtual environment and install backend dependencies

```bash
uv venv
source .venv/bin/activate
uv pip install -e backend
```

> Or using the Makefile shortcut:
> ```bash
> make setup
> ```

#### Step 4 — Configure environment variables

```bash
cp .env.example .env
```

Open `.env` and set the required values:

| Variable | Required | Description |
|----------|----------|-------------|
| `PINECONE_API_KEY` | **Yes** | Your Pinecone API key |
| `PINECONE_INDEX` | **Yes** | Name of the Pinecone index (e.g., `job-retrieval`) |
| `LLM_API_KEY` | Optional | OpenAI (or compatible) API key |
| `LLM_BASE_URL` | Optional | Override to use alternative providers |
| `LLM_MODEL` | Optional | Default: `gpt-4o-mini` |
| `USE_HYBRID` | Optional | Enable BM25 hybrid search (`true`/`false`) |
| `RERANK_MODEL` | Optional | Cross-encoder model name, or leave empty to disable |
| `DATA_PATH` | Optional | Path to the LF Jobs CSV (default: `./data/lf_jobs.csv`) |

#### Step 5 — Add the dataset

Place the LF Jobs CSV file at:

```
data/lf_jobs.csv
```

The CSV is expected to contain columns including: `job_title`, `company`, `location`, `level`, `category`, `publication_date`, and `description`.

#### Step 6 — Build the vector index

```bash
PYTHONPATH=backend python backend/scripts/build_index.py
# or
make build-index
```

This script:
1. Reads and cleans job descriptions (strips HTML tags)
2. Chunks long descriptions into overlapping windows
3. Generates embeddings using `e5-large-v2`
4. Upserts all vectors and metadata into Pinecone
5. (If `USE_HYBRID=true`) Builds and saves the BM25 index to `storage/bm25.pkl`

#### Step 7 — Run the API server

```bash
PYTHONPATH=backend uvicorn app.main:app --reload
# or
make api
```

The API will be available at `http://localhost:8000`.
Interactive API docs: `http://localhost:8000/docs` (Swagger UI).

### 2.3 Docker Deployment

#### Build and start all services

```bash
docker compose up --build
# or
make docker-up
```

This starts:
- `api` on port **8000**
- `frontend` (Next.js) on port **3000**
- `redis` on port **6379**
- `postgres` on port **5432**

#### Build the vector index inside Docker

```bash
docker compose run --rm api python backend/scripts/build_index.py
# or
make docker-build-index
```

#### Stop all services

```bash
docker compose down
# or
make docker-down
```

> **Note:** The API Docker image pre-downloads the embedding model and reranker at build time. If you change `EMBEDDING_MODEL` or `RERANK_MODEL`, you must rebuild the image with `docker compose build`.

---

## 3. Example Usage — Requests & Expected Responses

The service exposes a single primary endpoint:

### `POST /api/query`

**Base URL:** `http://localhost:8000`

### 3.1 Basic Query

**Request:**

```bash
curl -X POST http://localhost:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "senior data engineer in remote",
    "top_k": 5
  }'
```

**Expected Response:**

```json
{
  "answer": "Based on the job listings, several Senior Data Engineer positions are available remotely. These roles typically require expertise in Python, SQL, and distributed data systems like Spark or Kafka. Companies such as Acme Corp offer competitive salaries and flexible working arrangements for experienced engineers looking to work remotely.",
  "hits": [
    {
      "id": "LF0123-0",
      "score": 0.82,
      "job_title": "Senior Data Engineer",
      "company": "Acme Corp",
      "location": "Remote",
      "level": "Senior Level",
      "snippet": "Build and optimize large-scale data pipelines using Python and Apache Spark. Collaborate with data scientists to productionize machine learning models..."
    },
    {
      "id": "LF0456-1",
      "score": 0.79,
      "job_title": "Lead Data Engineer",
      "company": "DataStream Inc",
      "location": "Remote (US)",
      "level": "Senior Level",
      "snippet": "Design and maintain streaming data infrastructure. Experience with Kafka, dbt, and cloud data warehouses required..."
    }
  ]
}
```

### 3.2 Query with Hybrid Search Enabled

**Request:**

```bash
curl -X POST http://localhost:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "mid level product manager in NYC",
    "top_k": 5,
    "use_hybrid": true
  }'
```

**Expected Response:**

```json
{
  "answer": "There are several mid-level Product Manager opportunities in New York City. These roles generally require 3-5 years of product management experience and a background in agile methodologies.",
  "hits": [
    {
      "id": "LF0456-1",
      "score": 0.79,
      "job_title": "Product Manager",
      "company": "Example Inc",
      "location": "New York, NY",
      "level": "Mid Level",
      "snippet": "Own the product roadmap and work cross-functionally with engineering and design..."
    }
  ]
}
```

### 3.3 Query with Pre-Filters (Metadata Filters)

Narrow results by company, location, level, or category **before** vector search.

**Request:**

```bash
curl -X POST http://localhost:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "machine learning engineer",
    "top_k": 3,
    "use_hybrid": false,
    "use_rerank": true,
    "filters": {
      "level": "Senior Level",
      "location": ["Remote", "San Francisco, CA"]
    }
  }'
```

**Expected Response:**

```json
{
  "answer": "Senior Machine Learning Engineer roles in Remote and San Francisco focus on building and scaling ML systems for production...",
  "hits": [
    {
      "id": "LF0789-0",
      "score": 0.91,
      "job_title": "Senior ML Engineer",
      "company": "TechCo",
      "location": "Remote",
      "level": "Senior Level",
      "snippet": "Lead the design of MLOps infrastructure. Experience with PyTorch, Kubernetes, and A/B testing required..."
    }
  ]
}
```

### 3.4 Query with Post-Filters

Apply filters **after** retrieval — e.g., filter by snippet length or tags.

**Request:**

```bash
curl -X POST http://localhost:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{
    "query": "frontend developer React",
    "top_k": 5,
    "post_filters": {
      "min_words": 30,
      "exclude_tags": ["contract", "part-time"]
    }
  }'
```

### 3.5 Retrieval-Only Mode (No LLM)

If `LLM_API_KEY` is not configured, the API still functions and returns job hits with a fallback message:

**Expected Response:**

```json
{
  "answer": "LLM not configured. Showing top matching jobs based on retrieval. Set LLM_API_KEY to enable generated answers.",
  "hits": [ ... ]
}
```

### 3.6 Request Schema Reference

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `query` | `string` | required | Natural-language job search query (min 3 chars) |
| `top_k` | `int` | 5 | Number of results to return (1–20) |
| `use_hybrid` | `bool` | `env USE_HYBRID` | Enable BM25 + vector hybrid retrieval |
| `use_rerank` | `bool` | `true` if model set | Apply cross-encoder reranking |
| `filters` | `object` | none | Pre-retrieval metadata filters (see below) |
| `post_filters` | `object` | none | Post-retrieval filters applied in Python |

**`filters` fields:** `company`, `location`, `level`, `category` (all accept a string or list of strings), `publication_date_from`, `publication_date_to` (ISO 8601 datetime).

**`post_filters` fields:** `min_words`, `max_words` (integers), `include_tags`, `exclude_tags` (string or list of strings).

---

## 4. Assumptions Made During Development

1. **LF Jobs Dataset Schema**: The CSV dataset follows the LF Jobs schema with columns for `job_title`, `company`, `location`, `level`, `category`, `publication_date`, and `description`. Any deviation in column names would require updating the ingestion script.

2. **HTML in Job Descriptions**: Job descriptions may contain raw HTML markup (e.g., `<p>`, `<ul>`, `<li>` tags). The ingestion pipeline always strips HTML before chunking and embedding. This is assumed to be safe and lossless for retrieval quality.

3. **OpenAI-Compatible LLM Provider**: The LLM integration assumes the configured endpoint follows the OpenAI `POST /v1/chat/completions` API contract. The prompt is a single-turn `user` message containing job context. No conversation history or tool-calling is assumed.

4. **Pinecone Index Dimension Consistency**: The Pinecone index dimension must exactly match the embedding model's output dimension (1024 for `intfloat/e5-large-v2`). If the embedding model is changed, the Pinecone index must be deleted and recreated.

5. **E5-Large Prefix Convention**: The `e5-large-v2` model requires `query:` prefix for search queries and `passage:` prefix for documents to achieve optimal performance. Violating this convention would silently degrade retrieval quality. This is enforced in the `EmbeddingModel` wrapper.

6. **Static LLM Prompt**: The prompt template is static and hardcoded in `backend/app/rag/prompts/`. It is assumed to be sufficient for summarizing job listings. No dynamic prompt selection or intent detection is implemented.

7. **BM25 Is Pre-Built**: The BM25 index is built offline and saved as `storage/bm25.pkl`. It is assumed the dataset does not change dynamically at runtime. If new jobs are added, the index must be rebuilt.

8. **Redis for Caching Only**: Redis is used exclusively as a query-level TTL cache. It is assumed that identical query strings + parameters within the TTL window are safe to return cached results for (i.e., the underlying index has not changed).

9. **PostgreSQL Is Reserved for Future Use**: PostgreSQL is included in Docker Compose for query logging and analytics but is not actively used in the current application logic. It is assumed this will be wired up in a future iteration.

10. **Single-Region Pinecone Deployment**: The Pinecone index is deployed to a single cloud/region (`aws / us-east-1` by default). Multi-region or global replication is not accounted for.

---

## 5. Drawbacks & Future Enhancements

### 5.1 Current Drawbacks

| Area | Drawback |
|------|----------|
| **Dataset Loading** | No automatic dataset download or version management. The user must manually place `lf_jobs.csv` in the `data/` directory. |
| **Static Prompt** | The LLM prompt is a fixed template. It does not adapt to query intent (e.g., "compare these jobs" vs. "find jobs matching X"). |
| **Embedding Throughput** | Index building processes documents sequentially in batches of 16. For large datasets (>100k records), this can be slow without distributed processing. |
| **No Streaming** | The API response is a single JSON payload. Long LLM responses block until generation is complete; no streaming chunks are returned. |
| **No Authentication** | The API endpoint has no authentication or rate limiting, making it unsuitable for public deployment without an API gateway in front. |
| **BM25 Staleness** | The BM25 index is a static snapshot. Any new job postings added after index build are invisible to hybrid search. |
| **No Evaluation Harness** | There is no automated retrieval or generation quality evaluation. Retrieval recall, NDCG, and LLM answer faithfulness are not measured. |
| **PostgreSQL Unused** | PostgreSQL is configured but not integrated into the application code, adding unnecessary container overhead. |

### 5.2 Future Enhancements

#### Retrieval & Indexing
- **Incremental indexing**: Support upsert of new job records into Pinecone without a full re-index, and update the BM25 index incrementally.
- **Automatic dataset loading**: Add a CLI script or data loader to download and version the LF Jobs dataset automatically.
- **Embedding batching improvements**: Parallelize embedding generation using multiple CPU/GPU workers or an embedding microservice.
- **Metadata enrichment**: Extract and index additional metadata fields (salary range, required skills, remote type) for richer filtering.

#### Retrieval Quality
- **Query expansion**: Automatically expand queries using synonyms or an LLM to improve recall (e.g., "ML engineer" → "machine learning / AI engineer").
- **Late interaction models**: Evaluate ColBERT-style models for token-level interaction scoring within Pinecone's late-interaction index.
- **Feedback loop**: Collect click-through and usefulness signals to retrain or fine-tune the embedding model on the job domain.

#### Generation & API
- **Streaming responses**: Implement Server-Sent Events (SSE) to stream LLM tokens as they are generated.
- **Intent detection**: Classify query intent (e.g., explore, compare, apply) and select an appropriate prompt template.
- **Structured extraction**: Use function-calling or guided generation to produce structured output (salary, requirements, apply link) alongside the summary.
- **Multi-turn conversation**: Support follow-up questions that reference previous query context (e.g., "show me more from the second company").

#### Observability & Production Readiness
- **Distributed tracing**: Integrate OpenTelemetry with a tracing backend (Jaeger, Honeycomb) covering the full request path from embedding to LLM.
- **Latency metrics**: Export per-stage timing (retrieval, rerank, LLM) as Prometheus metrics with Grafana dashboards.
- **Evaluation harness**: Build an offline evaluation pipeline using RAGAS or a custom annotated query set to measure Recall@K, NDCG, and answer faithfulness.
- **Authentication & rate limiting**: Add API key authentication and per-key rate limiting via a middleware layer or API gateway.
- **Activate PostgreSQL**: Use PostgreSQL to log all queries, latency, and hit IDs for analytics, A/B testing different retrieval strategies, and auditing.
- **Horizontal scaling**: Containerize the API as a stateless service and deploy behind a load balancer with autoscaling based on query latency metrics.

---

*Document last updated: March 2026*
