
from typing import TypedDict


class ChatState(TypedDict):
    query: str
    original_query: str
    intent: str
    draft: str
    review: str
    answer: str
    retrieved_context: str
    retry_count: int
    history: list[dict]
    ollama_model: str
    ollama_input_tokens: int | None
    ollama_output_tokens: int | None
    ollama_total_tokens: int | None
    ollama_latency_seconds: float
    ollama_done: bool | None
