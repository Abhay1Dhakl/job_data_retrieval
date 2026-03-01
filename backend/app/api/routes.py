from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from functools import lru_cache

from fastapi import APIRouter, Depends

from app.core.cache import get_cache
from app.core.config import Settings, get_settings
from app.rag.pipeline import RagPipeline, build_pipeline
from app.rag.schemas import JobHit, PreFilterOptions, QueryRequest, QueryResponse

router = APIRouter()


@lru_cache
def get_pipeline() -> RagPipeline:
    """Create and cache a configured RAG pipeline instance.

    Returns:
        A configured RagPipeline.
    """
    settings = get_settings()
    return build_pipeline(settings)


def _to_hit(chunk) -> JobHit:
    """Convert a retrieved chunk into an API response hit.

    Args:
        chunk: A retrieved chunk with metadata and score.
    Returns:
        A JobHit formatted for API responses.
    """
    meta = chunk.metadata
    snippet = chunk.text[:240] + ("..." if len(chunk.text) > 240 else "")
    return JobHit(
        id=chunk.id,
        score=chunk.score,
        job_title=str(meta.get("job_title", "")),
        company=str(meta.get("company", "")),
        location=str(meta.get("location", "")),
        level=str(meta.get("level", "")),
        snippet=snippet,
    )


def _cache_key(payload: QueryRequest, top_k: int, use_hybrid: bool, use_rerank: bool) -> str:
    """Build a stable cache key for a query request.

    Args:
        payload: The query request payload.
        top_k: The number of results to return.
        use_hybrid: Whether hybrid retrieval is enabled.
        use_rerank: Whether reranking is enabled.
    Returns:
        A deterministic cache key string.
    """
    filters_payload = (
        payload.filters.model_dump(exclude_none=True, mode="json") if payload.filters else None
    )
    post_filters_payload = (
        payload.post_filters.model_dump(exclude_none=True, mode="json") if payload.post_filters else None
    )
    blob = json.dumps(
        {
            "query": payload.query,
            "top_k": top_k,
            "use_hybrid": use_hybrid,
            "use_rerank": use_rerank,
            "filters": filters_payload,
            "post_filters": post_filters_payload,
        },
        sort_keys=True,
    )
    return f"query:{hashlib.sha256(blob.encode('utf-8')).hexdigest()}"


def _to_epoch_seconds(value: datetime) -> int:
    """Convert a datetime to UTC epoch seconds.

    Args:
        value: Datetime to convert.
    Returns:
        Epoch seconds (UTC).
    """
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return int(value.timestamp())


def _build_pinecone_filter(filters: PreFilterOptions | None) -> dict | None:
    """Build a Pinecone metadata filter from request options.

    Args:
        filters: Pre-filter options from the request.
    Returns:
        A Pinecone-compatible filter dict, or None when not set.
    """
    if not filters:
        return None

    result: dict = {}

    if filters.company:
        result["company"] = {"$in": [str(v) for v in filters.company]}
    if filters.location:
        result["location"] = {"$in": [str(v) for v in filters.location]}
    if filters.level:
        result["level"] = {"$in": [str(v) for v in filters.level]}
    if filters.category:
        result["category"] = {"$in": [str(v) for v in filters.category]}

    if filters.publication_date_from or filters.publication_date_to:
        ts_filter: dict = {}
        if filters.publication_date_from:
            ts_filter["$gte"] = _to_epoch_seconds(filters.publication_date_from)
        if filters.publication_date_to:
            ts_filter["$lte"] = _to_epoch_seconds(filters.publication_date_to)
        if ts_filter:
            result["publication_ts"] = ts_filter

    return result or None


@router.post("/api/query", response_model=QueryResponse)
def query_jobs(
    payload: QueryRequest,
    settings: Settings = Depends(get_settings),
    pipeline: RagPipeline = Depends(get_pipeline),
) -> QueryResponse:
    """Query the RAG pipeline and return a formatted response.

    Args:
        payload: The incoming query payload.
        settings: Application settings dependency.
        pipeline: RAG pipeline dependency.
    Returns:
        A response containing the generated answer and job hits.
    """
    top_k = payload.top_k or settings.top_k
    use_hybrid = payload.use_hybrid if payload.use_hybrid is not None else settings.use_hybrid
    use_rerank = payload.use_rerank if payload.use_rerank is not None else bool(settings.rerank_model)

    cache = get_cache(settings)
    cache_key = None
    if cache and settings.cache_ttl_seconds > 0:
        cache_key = _cache_key(payload, top_k, use_hybrid, use_rerank)
        try:
            cached = cache.get(cache_key)
        except Exception:
            cached = None
        if cached:
            try:
                return QueryResponse.model_validate_json(cached)
            except Exception:
                pass

    pre_filters = _build_pinecone_filter(payload.filters)
    post_filters = payload.post_filters.model_dump(exclude_none=True) if payload.post_filters else None

    answer, results = pipeline.run(
        query=payload.query,
        top_k=top_k,
        use_hybrid=use_hybrid,
        use_rerank=use_rerank,
        filters=pre_filters,
        post_filters=post_filters,
    )

    hits = [_to_hit(chunk) for chunk in results]
    response = QueryResponse(answer=answer, hits=hits)

    if cache and cache_key and settings.cache_ttl_seconds > 0:
        try:
            cache.setex(cache_key, settings.cache_ttl_seconds, response.model_dump_json())
        except Exception:
            pass

    return response
