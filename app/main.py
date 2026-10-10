
from uuid import uuid4

from dotenv import load_dotenv

load_dotenv()

from app.graph.workflow import app_graph


def main():
    thread_id = str(uuid4())
    history = []

    print("Agentic Chatbot | Type 'exit' to quit")
    print(f"Conversation ID: {thread_id}")
    print("Type 'new' to start a new conversation.")

    while True:
        query = input("\nYou: ").strip()

        if query.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        if query.lower() == "new":
            thread_id = str(uuid4())
            history = []
            print(f"\nNew conversation started: {thread_id}")
            continue

        if not query:
            continue

        try:
            result = app_graph.invoke(
                {
                    "query": query,
                    "original_query": query,
                    "intent": "",
                    "draft": "",
                    "review": "",
                    "answer": "",
                    "retry_count": 0,
                    "history": history.copy(),
                },
                config={
                    "configurable": {
                        "thread_id": thread_id,
                    }
                },
            )

            answer = result.get("answer", "").strip()

            print(f"\nRoute: {result.get('intent')}")
            print("\nAnswer:\n")
            print(answer)

            # Save the completed conversation turn.
            history.append({
                "role": "user",
                "content": query,
            })
            history.append({
                "role": "assistant",
                "content": answer,
            })

            # Keep only the latest 10 messages.
            history = history[-10:]

        except Exception as exc:
            print(f"\nError: {exc}")


if __name__ == "__main__":
    main()
