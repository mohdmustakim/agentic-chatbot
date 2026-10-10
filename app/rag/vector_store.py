
from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_DIR = PROJECT_ROOT / "data" / "chroma_db"

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

_embeddings = None
_vector_store = None


def get_embeddings():
    global _embeddings

    if _embeddings is None:
        _embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

    return _embeddings


def get_vector_store():
    global _vector_store

    if _vector_store is None:
        CHROMA_DIR.mkdir(parents=True, exist_ok=True)

        _vector_store = Chroma(
            collection_name="agentic_chatbot_docs",
            embedding_function=get_embeddings(),
            persist_directory=str(CHROMA_DIR),
        )

    return _vector_store
