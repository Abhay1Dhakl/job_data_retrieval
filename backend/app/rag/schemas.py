from __future__ import annotations

from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field
from pydantic import field_validator


class PreFilterOptions(BaseModel):
    """Metadata filters applied before vector search."""

    company: Optional[List[str] | str] = Field(default=None)
    location: Optional[List[str] | str] = Field(default=None)
    level: Optional[List[str] | str] = Field(default=None)
    category: Optional[List[str] | str] = Field(default=None)
    publication_date_from: Optional[datetime] = Field(default=None)
    publication_date_to: Optional[datetime] = Field(default=None)

    @field_validator("company", "location", "level", "category", mode="before")
    @classmethod
    def _coerce_list(cls, value):
        """Coerce single values into lists for filter fields."""
        if value is None:
            return None
        if isinstance(value, str):
            return [value]
        return list(value)


class PostFilterOptions(BaseModel):
    """Lightweight filters applied after retrieval."""

    min_words: Optional[int] = Field(default=None, ge=1)
    max_words: Optional[int] = Field(default=None, ge=1)
    include_tags: Optional[List[str] | str] = Field(default=None)
    exclude_tags: Optional[List[str] | str] = Field(default=None)

    @field_validator("include_tags", "exclude_tags", mode="before")
    @classmethod
    def _coerce_list(cls, value):
        """Coerce single tag values into lists."""
        if value is None:
            return None
        if isinstance(value, str):
            return [value]
        return list(value)


class QueryRequest(BaseModel):
    """Request payload for job search queries."""

    query: str = Field(..., min_length=3)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)
    use_hybrid: Optional[bool] = Field(default=None)
    use_rerank: Optional[bool] = Field(default=None)
    filters: Optional[PreFilterOptions] = Field(default=None)
    post_filters: Optional[PostFilterOptions] = Field(default=None)


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
