from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """Wrapper around SentenceTransformer with optional E5-style prefixes."""

    def __init__(self, model_name: str, batch_size: int = 64) -> None:
        """Initialize the embedding model and configuration.

        Args:
            model_name: The SentenceTransformer model name or path.
            batch_size: The batch size used during encoding.
        """
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name)
        self._use_e5_prefix = "e5" in model_name.lower()

    def _apply_prefix(self, texts: List[str], prefix: str) -> List[str]:
        """Apply the appropriate prefix to the texts based on the model's requirements.
        
        Args:
            texts(list): A list of input texts to be embedded.
            prefix: The prefix to apply (e.g., "query:" or "passage:")
        Returns:
        
            A list of texts with the appropriate prefixes applied.
        """
        if not self._use_e5_prefix:
            return texts
        prefixed: List[str] = []
        for text in texts:
            stripped = text.strip()
            lowered = stripped.lower()
            if lowered.startswith("query:") or lowered.startswith("passage:"):
                prefixed.append(stripped)
            else:
                prefixed.append(f"{prefix} {stripped}")
        return prefixed

    def _encode(self, texts: List[str]) -> List[List[float]]:
        """Encode texts into normalized embedding vectors.

        Args:
            texts: A list of input texts to be embedded.
        Returns:
            A list of embedding vectors, one per input text.
        """
        embeddings = self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if isinstance(embeddings, np.ndarray):
            return embeddings.tolist()
        return [emb.tolist() for emb in embeddings]

    def embed(self, texts: List[str]) -> List[List[float]]:
        """Embed general passages with model-specific prefixes.

        Args:
            texts: A list of passage texts to be embedded.
        Returns:
            A list of embedding vectors for the passages.
        """
        if not texts:
            return []
        texts = self._apply_prefix(texts, "passage:")
        return self._encode(texts)

    def embed_query(self, texts: List[str]) -> List[List[float]]:
        """Embed query texts with model-specific prefixes.

        Args:
            texts: A list of query texts to be embedded.
        Returns:
            A list of embedding vectors for the queries.
        """
        if not texts:
            return []
        texts = self._apply_prefix(texts, "query:")
        return self._encode(texts)

    def dimension(self) -> int:
        """Return the embedding dimension for the configured model.

        Returns:
            The dimensionality of the embeddings produced by this model.
        """
        probe = self.embed(["dimension probe"])[0]
        return len(probe)
