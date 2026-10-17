import pytest

from app.agents import supervisor
from app.agents.supervisor import (
    keyword_intent,
    parse_intent,
    supervisor_node,
)


def fake_router(answer=None, error=None):
    def _generate(payload):
        if error:
            raise error
        return {"answer": answer}

    return _generate


@pytest.fixture
def llm_router(monkeypatch):
    monkeypatch.setattr(supervisor, "USE_LLM_ROUTER", True)


@pytest.mark.parametrize("raw, expected", [
    ("coding", "coding"),
    ("  Research.\n", "research"),
    ("<think>maybe coding?</think>planning", "planning"),
    ("general", "general"),
    ("I don't know", None),
    ("", None),
])
def test_parse_intent(raw, expected):
    assert parse_intent(raw) == expected


def test_llm_label_is_used(llm_router, monkeypatch):
    monkeypatch.setattr(
        supervisor, "generate_router_answer", fake_router("research")
    )

    result = supervisor_node({"query": "Explain RAG architecture"})

    assert result == {"intent": "research", "routing_method": "llm"}


def test_unknown_llm_output_falls_back_to_keywords(llm_router, monkeypatch):
    monkeypatch.setattr(
        supervisor, "generate_router_answer", fake_router("banana")
    )

    result = supervisor_node({"query": "Write a python function"})

    assert result == {"intent": "coding", "routing_method": "keyword"}


def test_llm_failure_falls_back_to_keywords(llm_router, monkeypatch):
    monkeypatch.setattr(
        supervisor,
        "generate_router_answer",
        fake_router(error=ConnectionError("ollama down")),
    )

    result = supervisor_node({"query": "What is the cancellation policy?"})

    assert result == {"intent": "research", "routing_method": "keyword"}


def test_router_receives_recent_history(llm_router, monkeypatch):
    seen = {}

    def _generate(payload):
        seen["messages"] = payload["messages"]
        return {"answer": "coding"}

    monkeypatch.setattr(supervisor, "generate_router_answer", _generate)

    supervisor_node({
        "query": "make it faster",
        "history": [
            {"role": "user", "content": "write a sort function"},
            {"role": "assistant", "content": "def sort(xs): ..."},
        ],
    })

    roles = [m["role"] for m in seen["messages"]]
    assert roles == ["system", "user", "assistant", "user"]
    assert seen["messages"][-1]["content"] == "make it faster"


def test_flag_off_never_calls_llm(monkeypatch):
    def boom(payload):
        raise AssertionError("LLM should not be called")

    monkeypatch.setattr(supervisor, "generate_router_answer", boom)

    result = supervisor_node({"query": "Hi, how are you?"})

    assert result == {"intent": "general", "routing_method": "keyword"}


@pytest.mark.parametrize("query, expected", [
    ("Write a Python function to reverse a string", "coding"),
    ("Create a roadmap for an AI chatbot", "planning"),
    ("What is the refund for a cancelled train?", "research"),
    ("Hi, how are you?", "general"),
    # Regression: "class" must not match inside "classification".
    ("What is the difference between classification and regression?", "general"),
    # Suffixes still work.
    ("I keep getting errors when debugging", "coding"),
])
def test_keyword_fallback(query, expected):
    assert keyword_intent(query) == expected
