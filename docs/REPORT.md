# RAG Pipeline Report

## 1) Architecture Overview
The system is a standard RAG stack:
- **Ingestion**: `scripts/build_index.py` reads the LF Jobs CSV, cleans HTML, and chunks descriptions.
- **Embedding**: `app/rag/embeddings.py` uses a Hugging Face sentence-transformer to embed each chunk.
- **Vector Store**: `app/rag/vector_store.py` stores embeddings and metadata in Chroma for similarity search.
- **Retriever**: `app/rag/retriever.py` runs vector search and optionally combines it with BM25 scores for hybrid retrieval.
- **Reranker (optional)**: `app/rag/reranker.py` uses a cross-encoder model for improved ranking.
- **LLM**: `app/rag/llm.py` calls an OpenAI-compatible endpoint to synthesize a concise answer.
- **API**: `app/api/routes.py` exposes `POST /api/query`.

## 2) Engineering Decisions
- **Chroma for vector storage**: Persistent, easy to operate, fast for small-to-medium datasets.
- **Sentence-Transformers embeddings**: Strong baseline with minimal operational overhead.
- **Hybrid retrieval**: BM25 adds lexical precision; combined scoring uses min-max normalization.
- **OpenAI-compatible LLM**: Keeps provider flexible (OpenAI, Azure, or other compatible endpoints).
- **Config via Pydantic Settings**: Centralized, typed configuration with `.env` support.

## 3) Setup & Installation
1. Install dependencies: `pip install -r requirements.txt`
2. Add dataset CSV at `data/lf_jobs.csv` or set `DATA_PATH`.
3. Configure `.env` from `.env.example` and set `LLM_API_KEY` if using LLM responses.
4. Build indexes: `python scripts/build_index.py`
5. Start API: `uvicorn app.main:app --reload`

## 4) Example Usage
Request:
```bash
curl -X POST http://localhost:8000/api/query \
  -H 'Content-Type: application/json' \
  -d '{"query": "mid level product manager in NYC", "top_k": 5, "use_hybrid": true}'
```

Expected response (shape):
```json
{
  "answer": "Short answer...",
  "hits": [
    {
      "id": "LF0456-1",
      "score": 0.79,
      "job_title": "Product Manager",
      "company": "Example Inc",
      "location": "New York, NY",
      "level": "Mid Level",
      "snippet": "..."
    }
  ]
}
```

## 5) Assumptions
- Dataset columns follow the LF Jobs schema from the assignment.
- Job descriptions may be HTML; they are cleaned before chunking.
- LLM provider supports OpenAI-compatible `chat/completions` API.

## 6) Drawbacks & Future Enhancements
- No automatic dataset download; can be added as a script or data loader.
- LLM prompt is static; future work could add intent detection or structured extraction.
- For larger datasets, distributed vector stores (Pinecone, Weaviate) would scale better.
- Add monitoring (latency, recall metrics), tracing, and evaluation harness.
