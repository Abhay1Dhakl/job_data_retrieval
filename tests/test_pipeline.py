from app.rag.pipeline import RagPipeline


class DummyRetriever:
    def retrieve(self, query, use_hybrid=False, top_k=None, filters=None):
        return []


class DummyLLM:
    def generate(self, prompt: str) -> str:
        raise RuntimeError("boom")


def test_pipeline_safe_generate_fallback():
    pipeline = RagPipeline(DummyRetriever(), DummyLLM())
    answer, results = pipeline.run(
        query="data engineer",
        top_k=5,
        use_hybrid=False,
        use_rerank=False,
        filters=None,
        post_filters=None,
    )

    assert "LLM not configured" in answer
    assert results == []
