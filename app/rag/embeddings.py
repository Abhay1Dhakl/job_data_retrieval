from __future__ import annotations

from typing import List, Optional

import google.generativeai as genai


class EmbeddingModel:
    def __init__(self, model_name: str, api_key: Optional[str], batch_size: int = 64) -> None:
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY is not configured")
        self.model_name = model_name
        self.batch_size = batch_size
        self._dimension: Optional[int] = None
        genai.configure(api_key=api_key)

    def embed(self, texts: List[str], task_type: Optional[str] = None) -> List[List[float]]:
        if not texts:
            return []
        resolved_task_type = task_type or "retrieval_document"
        embeddings: List[List[float]] = []
        for text in texts:
            response = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type=resolved_task_type,
            )
            vector = None
            if isinstance(response, dict):
                vector = response.get("embedding") or response.get("values")
            else:
                vector = getattr(response, "embedding", None) or getattr(response, "values", None)
                if vector is None:
                    embedding_obj = getattr(response, "embedding", None)
                    if embedding_obj is not None and hasattr(embedding_obj, "values"):
                        vector = embedding_obj.values
            if vector is None:
                raise RuntimeError("Gemini embedding response missing embedding vector")
            embeddings.append(list(vector))
        return embeddings

    def embed_query(self, texts: List[str]) -> List[List[float]]:
        return self.embed(texts, task_type="retrieval_query")

    def dimension(self) -> int:
        if self._dimension is None:
            probe = self.embed(["dimension probe"])[0]
            self._dimension = len(probe)
        return self._dimension
