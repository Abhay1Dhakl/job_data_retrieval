from datetime import datetime, timezone

from fastapi.testclient import TestClient

from app.api import routes
from app.core.config import Settings
from app.main import app
from app.rag.retrieval import RetrievedChunk
from app.rag.schemas import PreFilterOptions


class DummyPipeline:
    def __init__(self) -> None:
        self.calls = []

    def run(self, query, top_k, use_hybrid, use_rerank, filters, post_filters):
        self.calls.append(
            {
                "query": query,
                "top_k": top_k,
                "use_hybrid": use_hybrid,
                "use_rerank": use_rerank,
                "filters": filters,
                "post_filters": post_filters,
            }
        )
        chunk = RetrievedChunk(
            id="X1",
            text="Job description",
            metadata={
                "job_title": "Engineer",
                "company": "Acme",
                "location": "Remote",
                "level": "Senior",
            },
            score=0.99,
        )
        return "dummy answer", [chunk]


def test_query_route_uses_settings_defaults():
    pipeline = DummyPipeline()
    settings = Settings(redis_url=None, cache_ttl_seconds=0, top_k=7, use_hybrid=True, rerank_model=None)

    app.dependency_overrides[routes.get_pipeline] = lambda: pipeline
    app.dependency_overrides[routes.get_settings] = lambda: settings

    client = TestClient(app)
    response = client.post("/api/query", json={"query": "data engineer"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["answer"] == "dummy answer"
    assert payload["hits"][0]["job_title"] == "Engineer"

    assert pipeline.calls[0]["top_k"] == 7
    assert pipeline.calls[0]["use_hybrid"] is True
    assert pipeline.calls[0]["use_rerank"] is False

    app.dependency_overrides.clear()


def test_build_pinecone_filter_from_prefilters():
    ts = datetime(2024, 1, 2, tzinfo=timezone.utc)
    filters = PreFilterOptions(
        company="Acme",
        location=["Remote"],
        publication_date_from=ts,
    )

    pinecone_filter = routes._build_pinecone_filter(filters)
    assert pinecone_filter["company"]["$in"] == ["Acme"]
    assert pinecone_filter["location"]["$in"] == ["Remote"]
    assert pinecone_filter["publication_ts"]["$gte"] == int(ts.timestamp())
