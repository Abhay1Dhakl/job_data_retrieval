# Job Data RAG Pipeline

A production-ready Retrieval-Augmented Generation (RAG) service for job data. It ingests the LF Jobs dataset, builds a vector index, and exposes a FastAPI endpoint that returns relevant job listings and a concise answer.

## Features
- Clean + chunk job descriptions (HTML-safe)
- Local embeddings via Sentence-Transformers (`intfloat/e5-large-v2`, 1024-dim)
- Vector search backed by Pinecone
- Optional hybrid retrieval with BM25
- Optional cross-encoder reranking
- OpenAI-compatible LLM integration

## Command Options To Run Project

---

### Option 1 — Makefile (Convenience commands)

> Fastest way to run everything. Requires `uv` (local) or Docker (for Docker commands).

**Step 1 — Configure `.env`**

```bash
cp .env.example .env
# Open .env and set:
#   PINECONE_API_KEY=your_key
#   LLM_API_KEY=your_key   (optional — needed for LLM-generated answers)
```

**Step 2 — Local development**

```bash
make setup              # create venv + install backend deps
make build-index        # build Pinecone + BM25 index
make api                # start API on :8000 with hot-reload
```

**Step 2 (alt) — Docker Compose**

```bash
make docker-up          # build images + start all services (API, UI, Redis, Postgres)
make docker-build-index # run index builder inside Docker
make docker-down        # stop and remove all containers
```

---

### Option 2 — Manual Setup (step-by-step)

**Step 1 — Install `uv`**

```bash
brew install uv          # macOS
# or: pipx install uv
```

**Step 2 — Create virtual environment and install dependencies**

```bash
uv venv
source .venv/bin/activate
uv pip install -e backend
```

**Step 3 — Configure `.env`**

```bash
cp .env.example .env
# Open .env and set:
#   PINECONE_API_KEY=your_key
#   LLM_API_KEY=your_key   (optional — needed for LLM-generated answers)
```

Docker Compose reads `.env.example` by default; `.env` overrides it when present.
If you change `EMBEDDING_MODEL`, rebuild the Docker image.

**Step 4 — Place the dataset**

```
data/lf_jobs.csv   ← put your LF Jobs CSV here
```

**Step 5 — Build the vector index**

```bash
PYTHONPATH=backend python backend/scripts/build_index.py
```

**Step 6 — Start the API**

```bash
PYTHONPATH=backend uvicorn app.main:app --reload
```


## Query API
`POST /api/query`

Example:

```bash
curl -X POST http://localhost:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "senior data engineer in remote", "top_k": 5}'
```

Example response shape:

```json
{
  "answer": "Short summary...",
  "hits": [
    {
      "id": "LF0123-0",
      "score": 0.82,
      "job_title": "Senior Data Engineer",
      "company": "Acme",
      "location": "Remote",
      "level": "Senior Level",
      "snippet": "Build and optimize data pipelines..."
    }
  ]
}
```

## Notes
- Hybrid search requires `bm25.pkl`, created by `backend/scripts/build_index.py`.
- Reranking is enabled by default via `RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2` in `.env.example`.
- LLM responses require `LLM_API_KEY`. If missing, the API returns retrieval-only results.
- Pinecone index configuration is controlled via `PINECONE_*` env vars in `.env`.
- Pinecone index dimension must match your embedding dimension (1024 for `intfloat/e5-large-v2`).
- `intfloat/e5-large-v2` performs best when inputs are prefixed with `query:` (for searches) and `passage:` (for documents).

## Project Structure
- `backend/` Python API + RAG pipeline
- `frontend/` Next.js UI
- `docker/` Dockerfiles
- `docs/` documentation report
- `data/` dataset (not committed)
- `storage/` vector/BM25 indexes (not committed)
