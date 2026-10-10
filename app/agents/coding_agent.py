
from langsmith import traceable

from app.config import AGENT_MODEL, AGENT_NUM_PREDICT, AGENT_TEMPERATURE
from app.prompts.coding import CODING_PROMPT
from app.agents.ollama_metrics import generate_ollama_answer


@traceable(name="Ollama Coding Generation", run_type="llm")
def generate_coding_answer(payload: dict) -> dict:
    return generate_ollama_answer(payload)


@traceable(name="Coding Agent", run_type="chain")
def coding_node(state: dict) -> dict:
    messages = [
        {"role": "system", "content": CODING_PROMPT}
    ]

    for item in state.get("history", [])[-10:]:
        role = item.get("role")
        content = item.get("content", "").strip()

        if role in {"user", "assistant"} and content:
            messages.append({
                "role": role,
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

    result = generate_coding_answer(payload)
    answer = result["answer"]

    if not answer:
        raise ValueError("Coding agent returned an empty answer.")

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
        "ollama_prompt_eval_seconds": result["prompt_eval_seconds"],
        "ollama_load_seconds": result["load_seconds"],
        "ollama_cost_usd_low": result["total_cost_usd_low"],
        "ollama_cost_usd_high": result["total_cost_usd_high"],
        "ollama_done": result["done"],
    }
