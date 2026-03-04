from app.rag.preprocess.text import batch_chunk_text, chunk_text, clean_html, normalize_whitespace


def test_clean_html_strips_tags_and_normalizes():
    raw = "<p>Hello</p>\n<p>World</p>"
    assert clean_html(raw) == "Hello World"


def test_normalize_whitespace_collapses_runs():
    assert normalize_whitespace(" a \n  b\tc ") == "a b c"


def test_chunk_text_overlaps():
    text = "abcdefghijklmnopqrstuvwxyz"
    chunks = chunk_text(text, max_chars=10, overlap=3)
    assert chunks[0] == "abcdefghij"
    assert chunks[1].startswith("hij")
    assert chunks[-1].endswith("z")


def test_batch_chunk_text_aligns_results():
    texts = ["alpha beta", "gamma delta epsilon"]
    chunks = batch_chunk_text(texts, max_chars=8, overlap=2)
    assert len(chunks) == 2
    assert all(isinstance(item, list) for item in chunks)
