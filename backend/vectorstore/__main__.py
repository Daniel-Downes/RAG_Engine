from .indexer import index_raw_documents


def main() -> None:
    count = index_raw_documents()
    print(f"Indexed {count} chunks into Qdrant")


if __name__ == "__main__":
    main()