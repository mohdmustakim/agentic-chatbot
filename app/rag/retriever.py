
import re
from pathlib import Path

from app.rag.vector_store import get_vector_store


def retrieve_context(query: str, k: int = 4) -> str:
    store = get_vector_store()

    # Over-fetch so duplicate chunks do not crowd distinct sources out.
    results = store.similarity_search(query, k=max(k * 3, k))

    if not results:
        return ""

    context_parts = []
    seen_content = set()

    for document in results:
        content = document.page_content.strip()
        normalized_content = re.sub(r"\s+", " ", content).casefold()
        if not normalized_content or normalized_content in seen_content:
            continue
        seen_content.add(normalized_content)

        source = document.metadata.get("source", "Unknown source")
        source = Path(source).name if source != "Unknown source" else source
        page = document.metadata.get("page")

        location = f"{source}, page {page + 1}" if isinstance(page, int) else source

        context_parts.append(
            f"[Source {len(context_parts) + 1}: {location}]\n{content}"
        )
        if len(context_parts) == k:
            break

    return "\n\n".join(context_parts)
