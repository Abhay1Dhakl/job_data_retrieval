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
from app.rag.retrieval import PineconeVectorStore


@dataclass
class JobRecord:
    """Structured job record parsed from the CSV dataset."""

    job_id: str
    job_category: str
    job_title: str
    company: str
    publication_date: str
    location: str
    level: str
    tags: str
    description: str


def _parse_publication_ts(value: str) -> int | None:
    """Parse a publication date string into a UTC epoch timestamp.

    Args:
        value: Raw publication date string.
    Returns:
        Epoch seconds if parsing succeeds; otherwise None.
    """
    if not value:
        return None
    parsed = pd.to_datetime(value, errors="coerce", utc=True)
    if pd.isna(parsed):
        return None
    return int(parsed.value // 1_000_000_000)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from column names.

    Args:
        df: Input dataframe.
    Returns:
        A copy of the dataframe with normalized column names.
    """
    df = df.copy()
    df.columns = [col.strip() for col in df.columns]
    return df


def load_jobs(path: str) -> List[JobRecord]:
    """Load job records from a CSV file.

    Args:
        path: Path to the CSV dataset.
    Returns:
        A list of JobRecord entries.
    """
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
    """Build Pinecone and BM25 indexes from job data.

    Args:
        data_path: Path to the CSV dataset.
        vector_dir: Directory for vector/BM25 artifacts.
        index_name: Name of the Pinecone index to use.
    """
    os.makedirs(vector_dir, exist_ok=True)
    settings = get_settings()
    embedder = EmbeddingModel(
        settings.embedding_model,
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
        publication_ts = _parse_publication_ts(job.publication_date)
        chunks = chunk_text(job.description)
        for idx, chunk in enumerate(chunks):
            chunk_id = f"{job.job_id}-{idx}"
            ids.append(chunk_id)
            documents.append(chunk)
            metadata = {
                "job_id": job.job_id,
                "job_title": job.job_title,
                "company": job.company,
                "location": job.location,
                "level": job.level,
                "category": job.job_category,
                "tags": job.tags,
                "publication_date": job.publication_date,
            }
            if publication_ts is not None:
                metadata["publication_ts"] = publication_ts
            metadatas.append(metadata)

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
    """CLI entry point for building the indexes."""
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Build vector and BM25 indexes.")
    parser.add_argument("--data", default=settings.data_path, help="Path to CSV dataset")
    parser.add_argument("--vector-dir", default=settings.vector_dir, help="Vector store directory")
    parser.add_argument("--index", default=settings.pinecone_index, help="Pinecone index name")
    args = parser.parse_args()

    build_index(args.data, args.vector_dir, args.index)


if __name__ == "__main__":
    main()
