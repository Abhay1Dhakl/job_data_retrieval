from __future__ import annotations

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name: str, batch_size: int = 64) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = SentenceTransformer(model_name)
        self._use_e5_prefix = "e5" in model_name.lower()

    def _apply_prefix(self, texts: List[str], prefix: str) -> List[str]:
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
        if not texts:
            return []
        texts = self._apply_prefix(texts, "passage:")
        return self._encode(texts)

    def embed_query(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        texts = self._apply_prefix(texts, "query:")
        return self._encode(texts)

    def dimension(self) -> int:
        probe = self.embed(["dimension probe"])[0]
        return len(probe)
