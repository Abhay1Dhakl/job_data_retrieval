from __future__ import annotations

from typing import List, Optional

from sentence_transformers import CrossEncoder

from app.rag.retrieval.retriever import RetrievedChunk


class CrossEncoderReranker:
    """Rerank retrieved chunks using a cross-encoder model."""

    def __init__(self, model_name: str) -> None:
        """Initialize the cross-encoder reranker.

        Args:
            model_name: Name or path of the cross-encoder model.
        """
        self.model_name = model_name
        self._model = CrossEncoder(model_name)

    def rerank(self, query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
        """Score and rerank chunks by relevance to the query.

        Args:
            query: Query text.
            chunks: Retrieved chunks to score.
        Returns:
            A list of chunks sorted by descending relevance.
        """
        if not chunks:
            return []
        pairs = [(query, chunk.text) for chunk in chunks]
        scores = self._model.predict(pairs)
        reranked = [
            RetrievedChunk(
                id=chunk.id,
                text=chunk.text,
                metadata=chunk.metadata,
                score=float(score),
            )
            for chunk, score in zip(chunks, scores)
        ]
        return sorted(reranked, key=lambda c: c.score, reverse=True)


def build_reranker(model_name: Optional[str]) -> Optional[CrossEncoderReranker]:
    """Build a reranker when a model name is configured.

    Args:
        model_name: Reranker model name or None.
    Returns:
        A CrossEncoderReranker instance, or None if not configured.
    """
    if not model_name:
        return None
    return CrossEncoderReranker(model_name)
