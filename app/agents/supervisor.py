
from app.graph.state import ChatState


CODING_KEYWORDS = {
    "python", "code", "coding", "programming", "debug",
    "bug", "function", "script", "syntax", "exception",
    "error", "class", "api", "implement", "refactor",
}

PLANNING_KEYWORDS = {
    "roadmap", "plan", "planning", "steps", "milestone",
    "strategy", "schedule", "breakdown", "project plan",
}

railway_keywords = [
    "irctc", "train", "railway", "pnr", "tdr",
    "confirmed ticket", "ticket cancellation",
    "cancellation charge", "refund", "departure",
]


def supervisor_node(state: ChatState) -> dict:
    query = state["query"].lower()

    # Prioritize specific multi-word phrases
    if any(word in query for word in PLANNING_KEYWORDS):
        intent = "planning"

    elif any(word in query for word in CODING_KEYWORDS):
        intent = "coding"

    elif any(word in query for word in railway_keywords):
        intent = "research"


    else:
        intent = "general"

    print(f"[Supervisor] Route: {intent}")

    return {"intent": intent}
