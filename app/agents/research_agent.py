
from langsmith import traceable

from app.config import AGENT_MODEL, AGENT_TEMPERATURE, RESEARCH_NUM_PREDICT
from app.prompts.research import RESEARCH_PROMPT
from app.rag.retriever import retrieve_context
from app.agents.ollama_metrics import generate_ollama_answer


@traceable(name="RAG Retrieve Context", run_type="retriever")
def traced_retrieve_context(query: str) -> str:
    return retrieve_context(query, k=4)


@traceable(name="Ollama Research Generation", run_type="llm")
def generate_research_answer(payload: dict) -> dict:
    return generate_ollama_answer(payload)


@traceable(name="Research Agent", run_type="chain")
def research_node(state: dict) -> dict:
    query = state.get("query", "").strip()

    if not query:
        raise ValueError("Research query cannot be empty.")

    context = traced_retrieve_context(query)

    if context:
        rag_instructions = f"""
Use the following retrieved document context to answer the user's question.

Rules:
- Prioritize information from the provided context.
- Do not invent policy details or unsupported facts.
- If the context does not contain enough information, clearly say so.
- You may explain the retrieved information in simple language.
- Source citations will be added in a later step.

Retrieved context:
{context}
"""
    else:
        rag_instructions = """
No relevant document context was retrieved.

Tell the user that sufficient information was not found in the
available documents. Do not invent document-specific facts.
"""

    messages = [{
        "role": "system",
        "content": RESEARCH_PROMPT + "\n\n" + rag_instructions,
    }]

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
        "content": query,
    })

    payload = {
        "model": AGENT_MODEL,
        "messages": messages,
        "stream": True,
        "think": False,
        "options": {
            "temperature": AGENT_TEMPERATURE,
            "num_predict": RESEARCH_NUM_PREDICT,
        },
    }

    result = generate_research_answer(payload)
    answer = result["answer"]

    if not answer:
        raise ValueError("Research agent returned an empty answer.")

    print(f"[RAG] Retrieved document context: {'Yes' if context else 'No'}")

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
