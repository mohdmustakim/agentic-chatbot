
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader

from app.rag.vector_store import get_vector_store


DOCUMENTS_DIR = Path(__file__).resolve().parents[2] / "data" / "documents"


def load_documents():
    documents = []

    for file_path in DOCUMENTS_DIR.rglob("*"):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()

        try:
            if suffix == ".pdf":
                loader = PyPDFLoader(str(file_path))
                documents.extend(loader.load())

            elif suffix == ".txt":
                loader = TextLoader(
                    str(file_path),
                    encoding="utf-8",
                )
                documents.extend(loader.load())

        except Exception as exc:
            print(f"[Ingestion] Failed to load {file_path.name}: {exc}")

    return documents


def ingest_documents():
    DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

    documents = load_documents()

    if not documents:
        print(f"No PDF/TXT documents found in {DOCUMENTS_DIR}")
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120,
    )

    chunks = splitter.split_documents(documents)

    if not chunks:
        print("No text chunks were generated.")
        return 0

    store = get_vector_store()
    store.add_documents(chunks)

    print(f"Ingested {len(chunks)} chunks from {len(documents)} document pages/files.")
    return len(chunks)


if __name__ == "__main__":
    ingest_documents()
