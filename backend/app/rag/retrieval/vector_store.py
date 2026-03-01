from __future__ import annotations

from typing import Any, Dict, List, Optional

from pinecone import Pinecone, ServerlessSpec


class PineconeVectorStore:
    """Pinecone-backed vector store abstraction."""

    def __init__(
        self,
        api_key: Optional[str],
        index_name: str,
        cloud: str,
        region: str,
        metric: str = "cosine",
        dimension: Optional[int] = None,
    ) -> None:
        """Initialize the Pinecone vector store.

        Args:
            api_key: Pinecone API key.
            index_name: Name of the Pinecone index.
            cloud: Pinecone cloud provider.
            region: Pinecone region.
            metric: Similarity metric (e.g., "cosine").
            dimension: Embedding dimension (required if creating the index).
        """
        if not api_key:
            raise RuntimeError("PINECONE_API_KEY is not configured")
        if not index_name:
            raise RuntimeError("PINECONE_INDEX is not configured")
        self._pc = Pinecone(api_key=api_key)
        self._index_name = index_name
        self._metric = metric
        self._ensure_index(index_name, cloud, region, dimension)
        self._index = self._pc.Index(index_name)

    def _list_index_names(self) -> List[str]:
        """Return existing index names from the Pinecone client."""
        indexes = self._pc.list_indexes()
        if isinstance(indexes, dict):
            if "indexes" in indexes and isinstance(indexes["indexes"], list):
                return [item.get("name", "") for item in indexes["indexes"] if isinstance(item, dict)]
            return list(indexes.keys())
        if hasattr(indexes, "names"):
            return list(indexes.names())
        if isinstance(indexes, list):
            names: List[str] = []
            for item in indexes:
                if isinstance(item, str):
                    names.append(item)
                elif isinstance(item, dict):
                    names.append(item.get("name", ""))
                else:
                    name = getattr(item, "name", "")
                    if name:
                        names.append(name)
            return names
        return []

    def _ensure_index(
        self,
        index_name: str,
        cloud: str,
        region: str,
        dimension: Optional[int],
    ) -> None:
        """Ensure the index exists, creating it if needed.

        Args:
            index_name: Name of the index to ensure.
            cloud: Cloud provider.
            region: Cloud region.
            dimension: Embedding dimension for index creation.
        """
        existing = set(self._list_index_names())
        if index_name in existing:
            return
        if dimension is None:
            raise RuntimeError(
                "Pinecone index does not exist and no embedding dimension was provided."
            )
        self._pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=self._metric,
            spec=ServerlessSpec(cloud=cloud, region=region),
        )

    def upsert(
        self,
        ids: List[str],
        embeddings: List[List[float]],
        documents: List[str],
        metadatas: List[Dict[str, Any]],
    ) -> None:
        """Upsert embeddings and metadata into the index.

        Args:
            ids: Vector IDs.
            embeddings: Embedding vectors.
            documents: Raw document text associated with embeddings.
            metadatas: Metadata dicts aligned with the documents.
        """
        if not ids:
            return
        vectors = []
        for idx, vector_id in enumerate(ids):
            metadata = dict(metadatas[idx]) if idx < len(metadatas) else {}
            metadata["document"] = documents[idx] if idx < len(documents) else ""
            vectors.append((vector_id, embeddings[idx], metadata))
        self._index.upsert(vectors=vectors)

    def query(
        self,
        query_embeddings: List[List[float]],
        n_results: int,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[List[Dict[str, Any]]]:
        """Query the index for nearest neighbors.

        Args:
            query_embeddings: Query vectors.
            n_results: Number of results per query.
            metadata_filter: Optional metadata filter for the search.
        Returns:
            A list of result lists with id, document, metadata, and score.
        """
        if not query_embeddings:
            return []
        hits: List[List[Dict[str, Any]]] = []
        for embedding in query_embeddings:
            query_kwargs: Dict[str, Any] = {
                "vector": embedding,
                "top_k": n_results,
                "include_metadata": True,
            }
            if metadata_filter:
                query_kwargs["filter"] = metadata_filter
            response = self._index.query(**query_kwargs)
            row: List[Dict[str, Any]] = []
            if isinstance(response, dict):
                matches = response.get("matches", [])
            else:
                matches = getattr(response, "matches", [])
            for match in matches:
                if not isinstance(match, dict):
                    match = match.to_dict() if hasattr(match, "to_dict") else {"id": getattr(match, "id", "")}
                metadata = match.get("metadata") or {}
                row.append(
                    {
                        "id": match.get("id", ""),
                        "document": metadata.get("document", ""),
                        "metadata": metadata,
                        "score": float(match.get("score", 0.0)),
                    }
                )
            hits.append(row)
        return hits

    def count(self) -> int:
        """Return the number of vectors in the index.

        Returns:
            Total vector count.
        """
        stats = self._index.describe_index_stats()
        if isinstance(stats, dict):
            return int(stats.get("total_vector_count", 0))
        return int(getattr(stats, "total_vector_count", 0))
