from __future__ import annotations

import re
from typing import Iterable, List

from bs4 import BeautifulSoup


_WHITESPACE_RE = re.compile(r"\s+")


def clean_html(raw_html: str) -> str:
    """Strip HTML markup and normalize whitespace.

    Args:
        raw_html: Raw HTML string.
    Returns:
        Cleaned text with normalized whitespace.
    """
    if not raw_html:
        return ""
    soup = BeautifulSoup(raw_html, "html.parser")
    text = soup.get_text(" ")
    return normalize_whitespace(text)


def normalize_whitespace(text: str) -> str:
    """Normalize whitespace to single spaces and trim ends.

    Args:
        text: Input text.
    Returns:
        Text with collapsed whitespace.
    """
    return _WHITESPACE_RE.sub(" ", text).strip()


def chunk_text(text: str, max_chars: int = 1200, overlap: int = 200) -> List[str]:
    """Split text into overlapping character chunks.

    Args:
        text: Input text to chunk.
        max_chars: Maximum characters per chunk.
        overlap: Number of characters to overlap between chunks.
    Returns:
        A list of chunked text strings.
    """
    text = normalize_whitespace(text)
    if not text:
        return []

    if max_chars <= 0:
        return [text]

    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(len(text), start + max_chars)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


def batch_chunk_text(texts: Iterable[str], max_chars: int, overlap: int) -> List[List[str]]:
    """Chunk multiple texts with shared parameters.

    Args:
        texts: Iterable of input texts.
        max_chars: Maximum characters per chunk.
        overlap: Number of characters to overlap between chunks.
    Returns:
        A list of chunk lists, aligned with the input texts.
    """
    return [chunk_text(text, max_chars=max_chars, overlap=overlap) for text in texts]
