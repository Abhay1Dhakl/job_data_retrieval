from __future__ import annotations

import pickle
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
from rank_bm25 import BM25Okapi

from app.rag.embeddings import EmbeddingModel
from app.rag.retrieval.vector_store import PineconeVectorStore


_TOKEN_RE = re.compile(r"\b\w+\b")


def tokenize(text: str) -> List[str]:
    """Tokenize text into lowercase word tokens.

    Args:
        text: Input text.
    Returns:
        A list of word tokens.
    """
    return _TOKEN_RE.findall(text.lower())


@dataclass
class RetrievedChunk:
    """Represents a retrieved document chunk with metadata and score."""

    id: str
    text: str
    metadata: Dict[str, Any]
    score: float


class BM25Index:
    """BM25 index wrapper for lexical retrieval."""

    def __init__(self, ids: List[str], texts: List[str], metadatas: List[Dict[str, Any]]):
        """Initialize a BM25 index from documents and metadata.

        Args:
            ids: Document IDs.
            texts: Document texts.
            metadatas: Document metadata entries.
        """
        self.ids = ids
        self.texts = texts
        self.metadatas = metadatas
        self._tokens = [tokenize(text) for text in texts]
        self._bm25 = BM25Okapi(self._tokens)

    @classmethod
    def load(cls, path: str) -> "BM25Index":
        """Load a serialized BM25 index from disk.

        Args:
            path: Path to the pickled BM25 index.
        Returns:
            A BM25Index instance.
        """
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["ids"], data["texts"], data["metadatas"])

    def query(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """Query the BM25 index and return top-scoring chunks.

        Args:
            query: Query string.
            top_k: Number of results to return.
        Returns:
            A list of retrieved chunks sorted by BM25 score.
        """
        tokens = tokenize(query)
        scores = self._bm25.get_scores(tokens)
        ranked = np.argsort(scores)[::-1][:top_k]
        return [
            RetrievedChunk(
                id=self.ids[i],
                text=self.texts[i],
                metadata=self.metadatas[i],
                score=float(scores[i]),
            )
            for i in ranked
        ]


class Retriever:
    """Hybrid retriever combining vector and BM25 search."""

    def __init__(
        self,
        vector_store: PineconeVectorStore,
        embedding_model: EmbeddingModel,
        top_k: int = 5,
        bm25_index: Optional[BM25Index] = None,
        hybrid_alpha: float = 0.35,
    ) -> None:
        """Initialize the retriever.

        Args:
            vector_store: Vector store used for semantic search.
            embedding_model: Embedding model for query encoding.
            top_k: Default number of results to return.
            bm25_index: Optional BM25 index for hybrid retrieval.
            hybrid_alpha: Weight for BM25 scores in hybrid mode.
        """
        self.vector_store = vector_store
        self.embedding_model = embedding_model
        self.top_k = top_k
        self.bm25_index = bm25_index
        self.hybrid_alpha = hybrid_alpha

    def retrieve(self, query: str, use_hybrid: bool = False) -> List[RetrievedChunk]:
        """Retrieve relevant chunks for a query.

        Args:
            query: Query string.
            use_hybrid: Whether to combine vector and BM25 results.
        Returns:
            A list of retrieved chunks.
        """
        vector_results = self._vector_search(query, self.top_k)
        if not use_hybrid or not self.bm25_index:
            return vector_results

        bm25_results = self.bm25_index.query(query, self.top_k)
        return self._merge_results(vector_results, bm25_results)

    def _vector_search(self, query: str, top_k: int) -> List[RetrievedChunk]:
        """Run vector search against the vector store.

        Args:
            query: Query string.
            top_k: Number of results to return.
        Returns:
            A list of retrieved chunks from vector search.
        """
        query_embedding = self.embedding_model.embed_query([query])
        results = self.vector_store.query(query_embedding, n_results=top_k)
        if not results:
            return []
        return [
            RetrievedChunk(
                id=item["id"],
                text=item["document"],
                metadata=item["metadata"],
                score=float(item["score"]),
            )
            for item in results[0]
        ]

    def _merge_results(
        self,
        vector_results: List[RetrievedChunk],
        bm25_results: List[RetrievedChunk],
    ) -> List[RetrievedChunk]:
        """Merge vector and BM25 results using normalized scores.

        Args:
            vector_results: Results from vector search.
            bm25_results: Results from BM25 search.
        Returns:
            A combined, score-normalized result list.
        """
        vector_scores = self._normalize([r.score for r in vector_results])
        bm25_scores = self._normalize([r.score for r in bm25_results])

        combined: Dict[str, RetrievedChunk] = {}
        for idx, result in enumerate(vector_results):
            combined[result.id] = RetrievedChunk(
                id=result.id,
                text=result.text,
                metadata=result.metadata,
                score=(1.0 - self.hybrid_alpha) * vector_scores[idx],
            )

        for idx, result in enumerate(bm25_results):
            score = self.hybrid_alpha * bm25_scores[idx]
            if result.id in combined:
                combined[result.id].score += score
            else:
                combined[result.id] = RetrievedChunk(
                    id=result.id,
                    text=result.text,
                    metadata=result.metadata,
                    score=score,
                )

        ranked = sorted(combined.values(), key=lambda r: r.score, reverse=True)
        return ranked[: self.top_k]

    @staticmethod
    def _normalize(scores: Iterable[float]) -> List[float]:
        """Normalize a list of scores to the [0, 1] range.

        Args:
            scores: Iterable of scores.
        Returns:
            A list of normalized scores.
        """
        scores = list(scores)
        if not scores:
            return []
        min_val, max_val = min(scores), max(scores)
        if max_val - min_val < 1e-8:
            return [1.0 for _ in scores]
        return [(s - min_val) / (max_val - min_val) for s in scores]
