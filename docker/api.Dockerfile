# syntax=docker/dockerfile:1.7
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend

WORKDIR /app

ARG EMBEDDING_MODEL=intfloat/e5-large-v2
ARG RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2
ENV HF_HOME=/app/.cache/huggingface

COPY backend/pyproject.toml /app/backend/pyproject.toml
RUN pip install --no-cache-dir --upgrade pip \
&& pip install --no-cache-dir uv \
&& uv pip install --system /app/backend

RUN python - <<PY
from sentence_transformers import SentenceTransformer
SentenceTransformer("${EMBEDDING_MODEL}")
PY

RUN python - <<PY
from sentence_transformers import CrossEncoder
CrossEncoder("${RERANK_MODEL}")
PY

COPY backend/app /app/backend/app
COPY backend/scripts /app/backend/scripts
COPY docs ./docs
COPY README.md ./

RUN mkdir -p /app/data /app/storage

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
