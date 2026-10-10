
from app.agents.supervisor import supervisor_node


if __name__ == "__main__":
    test_queries = [
        "Write a Python function to reverse a string",
        "Explain RAG architecture",
        "Create a roadmap for an AI chatbot",
        "Hi, how are you?",
    ]

    for query in test_queries:
        result = supervisor_node({
            "query": query,
            "intent": "",
            "draft": "",
            "review": "",
            "answer": "",
        })

        print(f"Query: {query}")
        print(f"Selected agent: {result['intent']}")
        print("-" * 40)
