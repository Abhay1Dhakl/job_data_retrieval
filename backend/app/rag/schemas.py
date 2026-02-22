from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """Request payload for job search queries."""

    query: str = Field(..., min_length=3)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    use_hybrid: Optional[bool] = Field(default=None)
    use_rerank: Optional[bool] = Field(default=None)


class JobHit(BaseModel):
    """Job hit returned by the search endpoint."""

    id: str
    score: float
    job_title: str
    company: str
    location: str
    level: str
    snippet: str


class QueryResponse(BaseModel):
    """Response payload containing answer and job hits."""

    answer: str
    hits: List[JobHit]
