from langchain_core.documents import Document

from app.agents.supervisor import supervisor_node
from app.agents import reviewer_agent
from app.rag import retriever


def test_cancellation_policy_routes_to_research():
    result = supervisor_node({"query": "What is the cancellation policy?"})

    assert result["intent"] == "research"


def test_retrieval_deduplicates_chunks_and_keeps_distinct_sources(monkeypatch):
    duplicate_text = "IRCTC cancellation rules and refund details."

    class FakeVectorStore:
        def __init__(self):
            self.requested_k = None

        def similarity_search(self, query, k):
            self.requested_k = k
            return [
                Document(
                    page_content=duplicate_text,
                    metadata={"source": "/old/path/policy.pdf", "page": 0},
                ),
                Document(
                    page_content=f"  {duplicate_text}  ",
                    metadata={"source": "/current/path/policy.pdf", "page": 0},
                ),
                Document(
                    page_content="A separate refund rule.",
                    metadata={"source": "/current/path/policy.pdf", "page": 1},
                ),
            ]

    store = FakeVectorStore()
    monkeypatch.setattr(retriever, "get_vector_store", lambda: store)

    context = retriever.retrieve_context("cancellation rules", k=2)

    assert store.requested_k == 6
    assert context.count("[Source ") == 2
    assert context.count(duplicate_text) == 1
    assert "Source 1: policy.pdf, page 1" in context
    assert "/current/path" not in context


def test_reviewer_receives_retrieved_context(monkeypatch):
    expected_context = "[Source 1: policy.pdf, page 1]\nRefund rules"
    captured_payload = {}

    def fake_generate(payload):
        captured_payload.update(payload)
        return {"answer": "PASS: The answer follows the document."}

    monkeypatch.setattr(
        reviewer_agent, "generate_reviewer_answer", fake_generate
    )

    result = reviewer_agent.reviewer_node({
        "original_query": "What is the refund rule?",
        "draft": "According to the policy, refunds are available.",
        "retrieved_context": expected_context,
    })

    assert result["review"].startswith("PASS")
    assert expected_context in captured_payload["messages"][1]["content"]
