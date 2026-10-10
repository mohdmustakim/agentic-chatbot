
from app.rag.retriever import retrieve_context


def main():
    query = input("Ask something about your documents: ").strip()

    if not query:
        print("Please enter a question.")
        return

    try:
        context = retrieve_context(query, k=4)

        if not context:
            print("No documents found in the vector store.")
            return

        print("\nRetrieved context:\n")
        print(context)

    except Exception as exc:
        print(f"Retrieval failed: {exc}")


if __name__ == "__main__":
    main()
