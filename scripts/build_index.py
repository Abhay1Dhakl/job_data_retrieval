from __future__ import annotations

import argparse
import os
import pickle
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
from tqdm import tqdm

from app.core.config import get_settings
from app.rag.embeddings import EmbeddingModel
from app.rag.preprocess import chunk_text, clean_html
from app.rag.vector_store import PineconeVectorStore


@dataclass
class JobRecord:
    job_id: str
    job_category: str
    job_title: str
    company: str
    publication_date: str
    location: str
    level: str
    tags: str
    description: str


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    return df


def load_jobs(path: str) -> List[JobRecord]:
    df = _normalize_columns(pd.read_csv(path))
    records: List[JobRecord] = []
    for _, row in df.iterrows():
        description = clean_html(str(row.get("Job Description", "")))
        records.append(
            JobRecord(
                job_id=str(row.get("ID", "")),
                job_category=str(row.get("Job Category", "")),
                job_title=str(row.get("Job Title", "")),
                company=str(row.get("Company Name", "")),
                publication_date=str(row.get("Publication Date", "")),
                location=str(row.get("Job Location", "")),
                level=str(row.get("Job Level", "")),
                tags=str(row.get("Tags", "")),
                description=description,
            )
        )
    return records


def build_index(data_path: str, vector_dir: str, index_name: str) -> None:
    os.makedirs(vector_dir, exist_ok=True)
    settings = get_settings()
    embedder = EmbeddingModel(
        settings.embedding_model,
        settings.gemini_api_key,
        settings.embedding_batch_size,
    )
    vector_store = PineconeVectorStore(
        api_key=settings.pinecone_api_key,
        index_name=index_name,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
        metric=settings.pinecone_metric,
        dimension=embedder.dimension(),
    )

    ids: List[str] = []
    documents: List[str] = []
    metadatas: List[Dict[str, str]] = []

    jobs = load_jobs(data_path)
    for job in tqdm(jobs, desc="Chunking jobs"):
        if not job.description:
            continue
        chunks = chunk_text(job.description)
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{job.job_id}-{idx}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append(
                {
                    "job_id": job.job_id,
                    "job_title": job.job_title,
                    "company": job.company,
                    "location": job.location,
                    "level": job.level,
                    "category": job.job_category,
                    "tags": job.tags,
                    "publication_date": job.publication_date,
                }
            )

    for i in tqdm(range(0, len(documents), settings.embedding_batch_size), desc="Embedding"):
        batch_docs = documents[i : i + settings.embedding_batch_size]
        batch_ids = ids[i : i + settings.embedding_batch_size]
        batch_meta = metadatas[i : i + settings.embedding_batch_size]
        embeddings = embedder.embed(batch_docs)
        vector_store.upsert(batch_ids, embeddings, batch_docs, batch_meta)

    bm25_path = os.path.join(vector_dir, "bm25.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump({"ids": ids, "texts": documents, "metadatas": metadatas}, f)

    print(f"Indexed {len(ids)} chunks into {index_name}.")
    print(f"BM25 index saved to {bm25_path}.")


def main() -> None:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build vector and BM25 indexes.")
    parser.add_argument("--data", default=settings.data_path, help="Path to CSV dataset")
    parser.add_argument("--vector-dir", default=settings.vector_dir, help="Vector store directory")
    parser.add_argument("--index", default=settings.pinecone_index, help="Pinecone index name")
    args = parser.parse_args()

    build_index(args.data, args.vector_dir, args.index)


if __name__ == "__main__":
    main()
