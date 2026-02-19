from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=3)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    use_hybrid: Optional[bool] = Field(default=None)
    use_rerank: Optional[bool] = Field(default=None)


class JobHit(BaseModel):
    id: str
    score: float
    job_title: str
    company: str
    location: str
    level: str
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    hits: List[JobHit]
