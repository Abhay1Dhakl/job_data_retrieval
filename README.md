# Job Data RAG Pipeline

A production-ready Retrieval-Augmented Generation (RAG) service for job data. It ingests the LF Jobs dataset, builds a vector index, and exposes a FastAPI endpoint that returns relevant job listings and a concise answer.

## Features
- Clean + chunk job descriptions (HTML-safe)
- Local embeddings via Sentence-Transformers (`intfloat/e5-large-v2`, 1024-dim)
- Vector search backed by Pinecone
- Optional hybrid retrieval with BM25
- Optional cross-encoder reranking
- OpenAI-compatible LLM integration

## Setup
0. Install `uv` if needed (e.g., `pipx install uv` or `brew install uv`).
1. Create a virtual environment and install dependencies (uv):

```bash
uv venv
source .venv/bin/activate
uv pip install -e backend
```

2. Place the dataset CSV at `data/lf_jobs.csv` or change `DATA_PATH` in `.env`.

3. Configure environment variables (Local 1024-dim embeddings + Pinecone):

```bash
cp .env.example .env
```

Docker Compose will read `.env.example` by default; `.env` overrides it when present.

4. Build the index:

```bash
PYTHONPATH=backend python backend/scripts/build_index.py
```

5. Run the API:

```bash
PYTHONPATH=backend uvicorn app.main:app --reload
```

### Convenience (Makefile)
```bash
make setup
make build-index
make api
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
- Reranking requires setting `RERANK_MODEL` in `.env` (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`).
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
