from app.rag.prompts import build_prompt
from app.rag.retrieval import RetrievedChunk


def test_build_prompt_includes_context_and_format():
    chunk = RetrievedChunk(
        id="X1",
        text="This is a role for data engineering.",
        metadata={
            "job_title": "Data Engineer",
            "company": "Acme",
            "location": "Remote",
            "level": "Senior",
        },
        score=0.9,
    )

    prompt = build_prompt("data engineer", [chunk])
    assert "Query: data engineer" in prompt
    assert "Data Engineer" in prompt
    assert "Acme" in prompt
    assert "Remote" in prompt
    assert "Level: Senior" in prompt
    assert "SUMMARY:" in prompt
    assert "JOBS:" in prompt
