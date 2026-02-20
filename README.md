# Job Data RAG Pipeline

A production-ready Retrieval-Augmented Generation (RAG) service for job data. It ingests the LF Jobs dataset, builds a vector index, and exposes a FastAPI endpoint that returns relevant job listings and a concise answer.

## Features
- Clean + chunk job descriptions (HTML-safe)
- Gemini embeddings
- Vector search backed by Pinecone
- Optional hybrid retrieval with BM25
- Optional cross-encoder reranking
- OpenAI-compatible LLM integration

## Setup
1. Create a virtual environment and install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Place the dataset CSV at `data/lf_jobs.csv` or change `DATA_PATH` in `.env`.

3. Configure environment variables (Gemini + Pinecone):

```bash
cp .env.example .env
```

4. Build the index:

```bash
python scripts/build_index.py
```

5. Run the API:

```bash
uvicorn app.main:app --reload
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
- Hybrid search requires `bm25.pkl`, created by `scripts/build_index.py`.
- Reranking requires setting `RERANK_MODEL` in `.env` (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`).
- LLM responses require `LLM_API_KEY`. If missing, the API returns retrieval-only results.
- Pinecone index configuration is controlled via `PINECONE_*` env vars in `.env`.

## Project Structure
- `app/` application code
- `scripts/` data ingestion and index builds
- `docs/` documentation report
- `data/` dataset (not committed)
- `storage/` vector/BM25 indexes (not committed)
