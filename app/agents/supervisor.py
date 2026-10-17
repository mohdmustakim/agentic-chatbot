"""Supervisor: routes a query to one of the specialist agents.

Primary path: a small LLM (ROUTER_MODEL) classifies the query.
Fallback path: word-boundary keyword matching, used when the LLM is
disabled (USE_LLM_ROUTER=false), unreachable, or returns an unknown label.
"""

import re

from langsmith import traceable

from app.agents.ollama_metrics import generate_ollama_answer
from app.config import (
    ROUTER_MODEL,
    ROUTER_NUM_PREDICT,
    ROUTER_TEMPERATURE,
    ROUTER_TIMEOUT,
    USE_LLM_ROUTER,
)
from app.graph.state import ChatState
from app.prompts.supervisor import SUPERVISOR_PROMPT


VALID_INTENTS = ("coding", "research", "planning", "general")

CODING_KEYWORDS = {
    "python", "code", "coding", "programming", "debug",
    "bug", "function", "script", "syntax", "exception",
    "error", "class", "api", "implement", "refactor",
}

PLANNING_KEYWORDS = {
    "roadmap", "plan", "planning", "steps", "milestone",
    "strategy", "schedule", "breakdown", "project plan",
}

RAILWAY_KEYWORDS = {
    "irctc", "train", "railway", "pnr", "tdr",
    "confirmed ticket", "ticket cancellation",
    "cancellation", "cancel", "cancellation charge",
    "refund", "departure",
}

HISTORY_TURNS = 4          # recent messages given to the router for follow-ups
HISTORY_CHARS = 300        # per-message truncation


def _matches(query: str, keywords: set[str]) -> bool:
    """Whole-word match, tolerating simple suffixes (s, es, ing, ed).

    Plain substring matching misfires: "class" matches "classification".
    """
    return any(
        re.search(rf"\b{re.escape(word)}(?:s|es|ing|ed)?\b", query)
        for word in keywords
    )


def keyword_intent(query: str) -> str:
    query = query.lower()

    # Prioritize specific multi-word phrases
    if _matches(query, PLANNING_KEYWORDS):
        return "planning"

    if _matches(query, CODING_KEYWORDS):
        return "coding"

    if _matches(query, RAILWAY_KEYWORDS):
        return "research"

    return "general"


def parse_intent(text: str) -> str | None:
    """Extract a valid label from raw model output, or None."""
    # Drop any reasoning block a thinking model may emit.
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    match = re.search(
        r"\b(" + "|".join(VALID_INTENTS) + r")\b", text.lower()
    )
    return match.group(1) if match else None


@traceable(name="Ollama Router Generation", run_type="llm")
def generate_router_answer(payload: dict) -> dict:
    return generate_ollama_answer(payload, timeout=ROUTER_TIMEOUT)


def _router_messages(state: ChatState) -> list[dict]:
    messages = [{"role": "system", "content": SUPERVISOR_PROMPT}]

    for item in (state.get("history") or [])[-HISTORY_TURNS:]:
        if item.get("role") in {"user", "assistant"}:
            content = item.get("content", "").strip()[:HISTORY_CHARS]
            if content:
                messages.append({"role": item["role"], "content": content})

    messages.append({"role": "user", "content": state["query"]})
    return messages


def llm_intent(state: ChatState) -> str | None:
    payload = {
        "model": ROUTER_MODEL,
        "messages": _router_messages(state),
        "stream": True,
        "think": False,
        "options": {
            "temperature": ROUTER_TEMPERATURE,
            "num_predict": ROUTER_NUM_PREDICT,
        },
    }

    try:
        result = generate_router_answer(payload)
    except Exception as exc:  # Ollama down, timeout, bad response...
        print(f"[Supervisor] LLM router failed ({exc}); using keywords.")
        return None

    intent = parse_intent(result.get("answer", ""))

    if intent is None:
        print(
            "[Supervisor] Unrecognised router output "
            f"{result.get('answer')!r}; using keywords."
        )

    return intent


@traceable(name="Supervisor", run_type="chain")
def supervisor_node(state: ChatState) -> dict:
    intent = llm_intent(state) if USE_LLM_ROUTER else None
    method = "llm"

    if intent is None:
        intent = keyword_intent(state["query"])
        method = "keyword"

    print(f"[Supervisor] Route: {intent} ({method})")

    return {"intent": intent, "routing_method": method}
