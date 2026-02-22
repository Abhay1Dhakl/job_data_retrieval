from __future__ import annotations

from typing import List

from app.rag.retriever import RetrievedChunk


def build_prompt(query: str, chunks: List[RetrievedChunk]) -> str:
    """Build the LLM prompt from a query and retrieved chunks.

    Args:
        query: User query string.
        chunks: Retrieved chunks to include as context.
    Returns:
        A formatted prompt string for the LLM.
    """
    context_blocks = []
    for idx, chunk in enumerate(chunks, start=1):
        meta = chunk.metadata
        header = (
            f"[{idx}] {meta.get('job_title', 'Unknown Role')}"
            f" at {meta.get('company', 'Unknown Company')}"
            f" | {meta.get('location', 'N/A')}"
            f" | Level: {meta.get('level', 'N/A')}"
        )
        context_blocks.append(f"{header}\n{chunk.text}")

    context = "\n\n".join(context_blocks) if context_blocks else "No context found."

    return (
        "You are an expert job assistant. Use ONLY the provided context. "
        "Do not invent companies, roles, or locations. If the context is insufficient, say so.\n\n"
        f"Query: {query}\n\n"
        f"Context:\n{context}\n\n"
        "Respond ONLY in this exact format:\n"
        "SUMMARY: <2-4 sentences>\n"
        "JOBS:\n"
        "- <Job Title> | <Company> | <Location> | <1-sentence reason>\n"
        "- <Job Title> | <Company> | <Location> | <1-sentence reason>\n"
        "If there are no relevant jobs in context, return SUMMARY and then JOBS: with no bullets."
    )
