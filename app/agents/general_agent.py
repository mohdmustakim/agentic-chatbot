
from langsmith import traceable

from app.config import AGENT_MODEL, AGENT_NUM_PREDICT, AGENT_TEMPERATURE
from app.agents.ollama_metrics import generate_ollama_answer


GENERAL_PROMPT = """
You are a helpful general-purpose AI assistant.

Answer clearly, accurately, and naturally.
Use conversation history to understand follow-up questions.
Do not invent facts or claim to perform actions you did not perform.
Use Markdown formatting when useful.
"""


@traceable(name="Ollama General Generation", run_type="llm")
def generate_general_answer(payload: dict) -> dict:
    """Use the shared Ollama generation and metrics helper."""
    return generate_ollama_answer(payload)


@traceable(name="General Agent", run_type="chain")
def general_node(state: dict) -> dict:
    messages = [
        {
            "role": "system",
            "content": GENERAL_PROMPT,
        }
    ]

    for item in state.get("history", []):
        if item.get("role") in {"user", "assistant"}:
            content = item.get("content", "").strip()

            if content:
                messages.append({
                    "role": item["role"],
                    "content": content,
                })

    messages.append({
        "role": "user",
        "content": state.get("query", ""),
    })

    payload = {
        "model": AGENT_MODEL,
        "messages": messages,
        "stream": True,
        "think": False,
        "options": {
            "temperature": AGENT_TEMPERATURE,
            "num_predict": AGENT_NUM_PREDICT,
        },
    }

    result = generate_general_answer(payload)
    answer = result["answer"]

    if not answer:
        raise ValueError("General agent returned an empty answer.")

    for label, key in [
        ("Model", "model"),
        ("Input tokens", "input_tokens"),
        ("Output tokens", "output_tokens"),
        ("Total tokens", "total_tokens"),
        ("Total latency", "latency_seconds"),
        ("First content latency", "first_token_seconds"),
        ("Generation seconds", "generation_seconds"),
        ("Estimated cost low (USD)", "total_cost_usd_low"),
        ("Estimated cost high (USD)", "total_cost_usd_high"),
    ]:
        print(f"[LLM] {label}: {result.get(key)}")

    return {
        "draft": answer,
        "answer": answer,
        "ollama_model": result["model"],
        "ollama_input_tokens": result["input_tokens"],
        "ollama_output_tokens": result["output_tokens"],
        "ollama_total_tokens": result["total_tokens"],
        "ollama_latency_seconds": result["latency_seconds"],
        "ollama_first_token_seconds": result["first_token_seconds"],
        "ollama_generation_seconds": result["generation_seconds"],
        "ollama_cost_usd_low": result["total_cost_usd_low"],
        "ollama_cost_usd_high": result["total_cost_usd_high"],
        "ollama_done": result["done"],
    }
