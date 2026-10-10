
from typing import Literal

from app.graph.state import ChatState


def route_query(
    state: ChatState,
) -> Literal["coding", "research", "planning", "general"]:
    intent = state.get("intent", "general")

    if intent not in {"coding", "research", "planning", "general"}:
        return "general"

    return intent


def route_review(
    state: ChatState,
) -> Literal["retry", "finish"]:
    review = state.get("review", "").strip().upper()
    retry_count = state.get("retry_count", 0)

    if review.startswith("REVISE") and retry_count < 1:
        return "retry"

    return "finish"


def prepare_retry(state: ChatState) -> dict:
    original_query = state.get("original_query") or state["query"]
    feedback = state.get("review", "")

    return {
        "query": (
            f"{original_query}\n\n"
            f"Reviewer feedback:\n{feedback}\n\n"
            "Please improve your previous answer using this feedback. "
            "Answer the original question directly."
        ),
        "retry_count": state.get("retry_count", 0) + 1,
        "review": "",
        "draft": "",
        "answer": "",
    }
