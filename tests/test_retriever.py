import pytest

from app.rag.retrieval.retriever import BM25Index, RetrievedChunk, Retriever, tokenize


def test_tokenize_lowercases_and_splits():
    assert tokenize("Hello, WORLD!") == ["hello", "world"]


def test_normalize_scales_range():
    normalized = Retriever._normalize([2.0, 4.0])
    assert normalized == pytest.approx([0.0, 1.0])


def test_filter_results_applies_in_and_ranges():
    results = [
        RetrievedChunk(id="1", text="a", metadata={"company": "Acme", "publication_ts": 100}, score=0.1),
        RetrievedChunk(id="2", text="b", metadata={"company": "Beta", "publication_ts": 200}, score=0.2),
    ]

    filtered = Retriever._filter_results(results, {"company": {"$in": ["Acme"]}})
    assert [chunk.id for chunk in filtered] == ["1"]

    filtered = Retriever._filter_results(results, {"publication_ts": {"$gte": 150}})
    assert [chunk.id for chunk in filtered] == ["2"]


def test_merge_results_ranks_combined_scores():
    retriever = Retriever(vector_store=object(), embedding_model=object(), hybrid_alpha=0.3)
    vector_results = [
        RetrievedChunk(id="A", text="a", metadata={}, score=0.9),
        RetrievedChunk(id="B", text="b", metadata={}, score=0.2),
    ]
    bm25_results = [
        RetrievedChunk(id="B", text="b", metadata={}, score=5.0),
        RetrievedChunk(id="C", text="c", metadata={}, score=4.0),
    ]

    merged = retriever._merge_results(vector_results, bm25_results, top_k=3)
    assert [chunk.id for chunk in merged] == ["A", "B", "C"]


def test_bm25_index_returns_best_match():
    index = BM25Index(ids=["1", "2"], texts=["python data", "sales manager"], metadatas=[{}, {}])
    results = index.query("python", top_k=2)
    ids = {result.id for result in results}
    assert ids == {"1", "2"}
    assert "1" in ids
    assert all(isinstance(result.score, float) for result in results)
