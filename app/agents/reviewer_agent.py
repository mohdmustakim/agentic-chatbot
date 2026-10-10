"""Reviewer agent using shared Ollama metrics and LangSmith tracing."""

from langsmith import traceable

from app.config import (
    AGENT_MODEL,
    AGENT_TEMPERATURE,
    REVIEWER_NUM_PREDICT,
)
from app.agents.ollama_metrics import generate_ollama_answer


REVIEWER_PROMPT = """
You are a strict but helpful AI response reviewer.

Review the generated answer against the original user query.

Evaluate:
1. Relevance to the original query.
2. Factual correctness.
3. Completeness.
4. Clarity and formatting.
5. Whether the answer follows the user's instructions.

Return exactly one of these formats:

PASS: <short explanation>

REVISE: <specific improvements required>

Rules:
- Use PASS when the answer is correct, relevant, and good enough.
- Use REVISE when a meaningful improvement is necessary.
- Do not request changes merely for stylistic preferences.
- Do not invent errors or facts.
- Keep your review concise.
- Return only the PASS or REVISE decision and its explanation.
"""


@traceable(name="Ollama Reviewer Generation", run_type="llm")
def generate_reviewer_answer(payload: dict) -> dict:
    """Generate the review through the shared streaming/metrics helper."""
    return generate_ollama_answer(payload)


@traceable(name="Reviewer Agent", run_type="chain")
def reviewer_node(state: dict) -> dict:
    original_query = state.get("original_query") or state.get("query", "")
    draft = state.get("draft", "").strip()

    if not draft:
        return {"review": "REVISE: The generated answer is empty."}

    payload = {
        "model": AGENT_MODEL,
        "messages": [
            {"role": "system", "content": REVIEWER_PROMPT},
            {
                "role": "user",
                "content": (
                    f"Original user query:\\n{original_query}\\n\\n"
                    f"Generated answer to review:\\n{draft}\\n\\n"
                    "Review this answer using the required format."
                ),
            },
        ],
        "stream": True,
        "think": False,
        "options": {
            "temperature": AGENT_TEMPERATURE,
            "num_predict": REVIEWER_NUM_PREDICT,
        },
    }

    try:
        result = generate_reviewer_answer(payload)
        review = result["answer"].strip()

        if not review:
            raise ValueError("Reviewer returned an empty response.")

        normalized_review = review.upper()
        if normalized_review.startswith("PASS"):
            pass
        elif normalized_review.startswith("REVISE"):
            pass
        else:
            # Fail-safe: unknown formats do not trigger a retry.
            review = (
                "PASS: Reviewer returned an unrecognized format; "
                "no retry was requested."
            )

        print(f"\\n[Reviewer] {review}")
        print(f"[Reviewer] Model: {result.get('model')}")
        print(f"[Reviewer] Tokens: {result.get('total_tokens')}")
        print(f"[Reviewer] Latency: {result.get('latency_seconds')}s")

        return {
            "review": review,
            "reviewer_model": result.get("model"),
            "reviewer_input_tokens": result.get("input_tokens"),
            "reviewer_output_tokens": result.get("output_tokens"),
            "reviewer_total_tokens": result.get("total_tokens"),
            "reviewer_latency_seconds": result.get("latency_seconds"),
            "reviewer_first_token_seconds": result.get("first_token_seconds"),
            "reviewer_cost_usd_low": result.get("total_cost_usd_low"),
            "reviewer_cost_usd_high": result.get("total_cost_usd_high"),
        }

    except (ValueError, KeyError) as exc:
        # Preserve the generated draft if review parsing/response fails.
        print(f"\\n[Reviewer Error] {exc}")
        return {
            "review": "PASS: Review unavailable; preserving generated answer."
        }
