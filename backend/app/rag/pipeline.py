from __future__ import annotations

import logging
import time
from typing import Any, Dict, List, Optional

from app.core.config import Settings
from app.rag.embeddings import EmbeddingModel
from app.rag.llm import OpenAICompatibleClient
from app.rag.prompts import build_prompt
from app.rag.retrieval import BM25Index, CrossEncoderReranker, RetrievedChunk, Retriever, build_reranker
from app.rag.retrieval import PineconeVectorStore


class RagPipeline:
    """Orchestrates retrieval, optional reranking, and generation."""

    def __init__(
        self,
        retriever: Retriever,
        llm: OpenAICompatibleClient,
        reranker: Optional[CrossEncoderReranker] = None,
    ) -> None:
        """Initialize the pipeline components.

        Args:
            retriever: Retriever instance for fetching relevant chunks.
            llm: LLM client used to generate answers.
            reranker: Optional reranker for refining retrieval results.
        """
        self.retriever = retriever
        self.llm = llm
        self.reranker = reranker

    def run(
        self,
        query: str,
        top_k: int,
        use_hybrid: bool,
        use_rerank: bool,
        filters: Optional[Dict[str, Any]] = None,
        post_filters: Optional[Dict[str, Any]] = None,
    ) -> tuple[str, List[RetrievedChunk]]:
        """Run retrieval (and optional reranking) then generate an answer.

        Args:
            query: User query string.
            top_k: Number of results to return.
            use_hybrid: Whether to blend vector and BM25 results.
            use_rerank: Whether to apply reranking.
            filters: Optional metadata filters applied during retrieval.
            post_filters: Optional filters applied after retrieval.
        Returns:
            A tuple of (answer, retrieved chunks).
        """
        logger = logging.getLogger(__name__)
        start = time.perf_counter()

        retrieval_start = time.perf_counter()
        results = self.retriever.retrieve(
            query,
            use_hybrid=use_hybrid,
            top_k=top_k,
            filters=filters,
        )
        retrieval_ms = (time.perf_counter() - retrieval_start) * 1000.0

        post_start = time.perf_counter()
        results = self._apply_post_filters(results, post_filters)
        post_ms = (time.perf_counter() - post_start) * 1000.0

        rerank_ms = 0.0
        if results and use_rerank and self.reranker:
            rerank_start = time.perf_counter()
            results = self.reranker.rerank(query, results)
            rerank_ms = (time.perf_counter() - rerank_start) * 1000.0

        results = results[:top_k]

        prompt_start = time.perf_counter()
        prompt = build_prompt(query, results)
        prompt_ms = (time.perf_counter() - prompt_start) * 1000.0

        llm_start = time.perf_counter()
        answer = self._safe_generate(prompt)
        llm_ms = (time.perf_counter() - llm_start) * 1000.0
        total_ms = (time.perf_counter() - start) * 1000.0

        logger.info(
            "rag_timing total_ms=%.1f retrieval_ms=%.1f post_ms=%.1f rerank_ms=%.1f prompt_ms=%.1f llm_ms=%.1f results=%d use_hybrid=%s use_rerank=%s",
            total_ms,
            retrieval_ms,
            post_ms,
            rerank_ms,
            prompt_ms,
            llm_ms,
            len(results),
            use_hybrid,
            use_rerank,
        )
        return answer, results

    def _safe_generate(self, prompt: str) -> str:
        """Generate with a safe fallback if the LLM call fails.

        Args:
            prompt: Prompt passed to the LLM.
        Returns:
            The generated answer, or a fallback message if generation fails.
        """
        try:
            return self.llm.generate(prompt)
        except Exception:
            return (
                "LLM not configured. Showing top matching jobs based on retrieval. "
                "Set LLM_API_KEY to enable generated answers."
            )

    @staticmethod
    def _apply_post_filters(
        results: List[RetrievedChunk],
        post_filters: Optional[Dict[str, Any]],
    ) -> List[RetrievedChunk]:
        """Apply lightweight post-filters to retrieved chunks.

        Args:
            results: Retrieved chunks to filter.
            post_filters: Post-filter configuration.
        Returns:
            Filtered list of retrieved chunks.
        """
        if not results or not post_filters:
            return results

        min_words = post_filters.get("min_words")
        max_words = post_filters.get("max_words")
        include_tags = post_filters.get("include_tags")
        exclude_tags = post_filters.get("exclude_tags")

        include_set = {tag.strip().lower() for tag in include_tags or [] if str(tag).strip()}
        exclude_set = {tag.strip().lower() for tag in exclude_tags or [] if str(tag).strip()}

        filtered: List[RetrievedChunk] = []
        for chunk in results:
            words = chunk.text.split()
            word_count = len(words)
            if min_words and word_count < min_words:
                continue
            if max_words and word_count > max_words:
                continue

            if include_set or exclude_set:
                raw_tags = str(chunk.metadata.get("tags", ""))
                tag_list = [t.strip().lower() for t in raw_tags.replace(";", ",").split(",") if t.strip()]
                tag_set = set(tag_list)

                if include_set and not (include_set & tag_set):
                    continue
                if exclude_set and (exclude_set & tag_set):
                    continue

            filtered.append(chunk)

        return filtered


def build_pipeline(settings: Settings) -> RagPipeline:
    """Construct a RagPipeline based on application settings.

    Args:
        settings: Application settings.
    Returns:
        A configured RagPipeline instance.
    """
    embedding_model = EmbeddingModel(
        model_name=settings.embedding_model,
        batch_size=settings.embedding_batch_size,
    )
    vector_store = PineconeVectorStore(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
        metric=settings.pinecone_metric,
        dimension=embedding_model.dimension(),
    )

    bm25_index = None
    if settings.use_hybrid:
        try:
            bm25_index = BM25Index.load(f"{settings.vector_dir}/bm25.pkl")
        except FileNotFoundError:
            bm25_index = None

    retriever = Retriever(
        vector_store=vector_store,
        embedding_model=embedding_model,
        top_k=settings.top_k,
        bm25_index=bm25_index,
        hybrid_alpha=settings.hybrid_alpha,
    )

    llm = OpenAICompatibleClient(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=settings.llm_temperature,
        max_tokens=settings.llm_max_tokens,
    )

    reranker = build_reranker(settings.rerank_model)
    return RagPipeline(retriever=retriever, llm=llm, reranker=reranker)
