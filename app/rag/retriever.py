
from app.rag.vector_store import get_vector_store


def retrieve_context(query: str, k: int = 4) -> str:
    store = get_vector_store()

    results = store.similarity_search(query, k=k)

    if not results:
        return ""

    context_parts = []

    for index, document in enumerate(results, start=1):
        source = document.metadata.get("source", "Unknown source")
        page = document.metadata.get("page")

        location = f"{source}, page {page + 1}" if isinstance(page, int) else source

        context_parts.append(
            f"[Source {index}: {location}]\n{document.page_content}"
        )

    return "\n\n".join(context_parts)
