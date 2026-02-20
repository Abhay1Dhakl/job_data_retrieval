from __future__ import annotations

from typing import List, Optional

from app.core.config import Settings
from app.rag.embeddings import EmbeddingModel
from app.rag.llm import OpenAICompatibleClient
from app.rag.prompts import build_prompt
from app.rag.reranker import CrossEncoderReranker, build_reranker
from app.rag.retriever import BM25Index, RetrievedChunk, Retriever
from app.rag.vector_store import PineconeVectorStore


class RagPipeline:
    def __init__(
        self,
        retriever: Retriever,
        llm: OpenAICompatibleClient,
        reranker: Optional[CrossEncoderReranker] = None,
    ) -> None:
        self.retriever = retriever
        self.llm = llm
        self.reranker = reranker

    def run(
        self,
        query: str,
        top_k: int,
        use_hybrid: bool,
        use_rerank: bool,
    ) -> tuple[str, List[RetrievedChunk]]:
        results = self.retriever.retrieve(query, use_hybrid=use_hybrid)
        if results and use_rerank and self.reranker:
            results = self.reranker.rerank(query, results)[:top_k]
        else:
            results = results[:top_k]

        prompt = build_prompt(query, results)
        answer = self._safe_generate(prompt)
        return answer, results

    def _safe_generate(self, prompt: str) -> str:
        try:
            return self.llm.generate(prompt)
        except Exception:
            return (
                "LLM not configured. Showing top matching jobs based on retrieval. "
                "Set LLM_API_KEY to enable generated answers."
            )


def build_pipeline(settings: Settings) -> RagPipeline:
    vector_store = PineconeVectorStore(
        api_key=settings.pinecone_api_key,
        index_name=settings.pinecone_index,
        cloud=settings.pinecone_cloud,
        region=settings.pinecone_region,
        metric=settings.pinecone_metric,
    )
    embedding_model = EmbeddingModel(
        model_name=settings.embedding_model,
        api_key=settings.gemini_api_key,
        batch_size=settings.embedding_batch_size,
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
