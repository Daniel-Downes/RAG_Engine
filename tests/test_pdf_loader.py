from itertools import pairwise

import pymupdf

from backend.ingestion import load_pdfs, split_pages


def _write_pdf(path, text, title):
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    document.set_metadata({"title": title, "author": "Test Author"})
    document.save(path)
    document.close()


def test_load_pdfs_extracts_each_page_and_metadata(tmp_path):
    nested_dir = tmp_path / "nested"
    nested_dir.mkdir()
    _write_pdf(nested_dir / "notes.pdf", "A useful page", "Notes")

    results = load_pdfs(tmp_path)

    assert len(results) == 1
    assert results[0]["text"] == "A useful page"
    assert results[0]["metadata"]["title"] == "Notes"
    assert results[0]["metadata"]["author"] == "Test Author"
    assert results[0]["metadata"]["source"] == "nested/notes.pdf"
    assert results[0]["metadata"]["file_name"] == "notes.pdf"
    assert results[0]["metadata"]["page_number"] == 1
    assert results[0]["metadata"]["total_pages"] == 1


def test_load_pdfs_returns_empty_list_for_empty_directory(tmp_path):
    assert load_pdfs(tmp_path) == []


def test_load_pdfs_removes_repeated_edge_lines_and_page_number_text(tmp_path):
    pdf_path = tmp_path / "report.pdf"
    document = pymupdf.open()
    for page_number in range(1, 5):
        page = document.new_page()
        page.insert_text((72, 40), "Repeated running header")
        page.insert_text((72, 120), f"Main body for page {page_number}")
        page.insert_text((72, 800), "Repeated proceedings footer")
        page.insert_text((300, 800), str(page_number))
    document.save(pdf_path)
    document.close()

    pages = load_pdfs(tmp_path)

    assert len(pages) == 4
    for page_number, page in enumerate(pages, start=1):
        assert "Repeated running header" not in page["text"]
        assert "Repeated proceedings footer" not in page["text"]
        assert f"Main body for page {page_number}" in page["text"]
        assert page["metadata"]["page_number"] == page_number
        assert page["metadata"]["total_pages"] == 4


def test_split_pages_respects_token_limit_and_preserves_metadata():
    import tiktoken

    page = {
        "text": " ".join(f"term{index}" for index in range(500)),
        "metadata": {"source": "guide.pdf", "page_number": 3},
    }

    chunks = split_pages([page], chunk_size=80, chunk_overlap=10)
    encoding = tiktoken.get_encoding("cl100k_base")

    assert len(chunks) > 1
    assert all(len(encoding.encode(chunk["text"])) <= 80 for chunk in chunks)
    for left, right in pairwise(chunks):
        shared_text = next(
            left["text"][-size:]
            for size in range(min(len(left["text"]), len(right["text"])), 0, -1)
            if right["text"].startswith(left["text"][-size:])
        )
        assert len(encoding.encode(shared_text)) == 10
    assert [chunk["metadata"]["chunk_index"] for chunk in chunks] == list(
        range(len(chunks))
    )
    assert all(chunk["metadata"]["source"] == "guide.pdf" for chunk in chunks)
    assert all(chunk["metadata"]["page_number"] == 3 for chunk in chunks)


def test_split_pages_keeps_fitting_fenced_code_block_together():
    code_block = "```python\ndef answer():\n    return 42\n```"
    page = {
        "text": f"A short introduction.\n\n{code_block}\n\nA short conclusion.",
        "metadata": {"source": "code.pdf", "page_number": 1},
    }

    chunks = split_pages([page], chunk_size=80, chunk_overlap=5)

    assert any(code_block in chunk["text"] for chunk in chunks)