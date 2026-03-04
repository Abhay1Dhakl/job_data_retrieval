import pytest
from pydantic import ValidationError

from app.rag.schemas import PostFilterOptions, PreFilterOptions, QueryRequest


def test_prefilter_coerces_scalar_to_list():
    options = PreFilterOptions(company="Acme")
    assert options.company == ["Acme"]


def test_postfilter_coerces_tags():
    options = PostFilterOptions(include_tags="python")
    assert options.include_tags == ["python"]


def test_query_request_requires_min_length():
    with pytest.raises(ValidationError):
        QueryRequest(query="hi")
