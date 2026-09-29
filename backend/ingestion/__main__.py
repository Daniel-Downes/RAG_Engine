from .pdf_loader import DEFAULT_INPUT_DIR, load_pdfs
from .text_splitter import split_pages


def main() -> None:
    pages = load_pdfs()
    chunks = split_pages(pages)
    print(
        f"Extracted {len(pages)} pages into {len(chunks)} chunks "
        f"from PDFs in {DEFAULT_INPUT_DIR}"
    )


if __name__ == "__main__":
    main()