
"""Shared Ollama usage, latency, and reference-pricing helpers."""

import json
import time
from datetime import datetime, timezone
from typing import Any

import requests
from langsmith import get_current_run_tree

from app.config import MODEL_PRICING, OLLAMA_BASE_URL, OLLAMA_TIMEOUT


def _pricing_key(model: str) -> str:
    name = (model or "").strip().lower()

    if name in MODEL_PRICING:
        return name

    if name.startswith("qwen3.5:4b"):
        return "qwen3.5:4b"

    if name.startswith("qwen3.5:latest"):
        return "qwen3.5:latest"

    return name


def _valid_token_count(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def calculate_reference_cost(
    model: str,
    input_tokens: int | None,
    output_tokens: int | None,
) -> dict[str, Any]:
    """Calculate low/high external-provider-equivalent cost in USD."""

    rates = MODEL_PRICING.get(_pricing_key(model))

    valid_counts = (
        _valid_token_count(input_tokens)
        and _valid_token_count(output_tokens)
    )

    result = {
        "pricing_available": rates is not None and valid_counts,
        "input_cost_usd_low": None,
        "input_cost_usd_high": None,
        "output_cost_usd_low": None,
        "output_cost_usd_high": None,
        "total_cost_usd_low": None,
        "total_cost_usd_high": None,
    }

    if rates is None or not valid_counts:
        return result

    input_low = input_tokens * rates["input_low"] / 1_000_000
    input_high = input_tokens * rates["input_high"] / 1_000_000
    output_low = output_tokens * rates["output_low"] / 1_000_000
    output_high = output_tokens * rates["output_high"] / 1_000_000

    result.update({
        "input_cost_usd_low": input_low,
        "input_cost_usd_high": input_high,
        "output_cost_usd_low": output_low,
        "output_cost_usd_high": output_high,
        "total_cost_usd_low": input_low + output_low,
        "total_cost_usd_high": input_high + output_high,
    })

    return result


def generate_ollama_answer(
    payload: dict,
    timeout: int = OLLAMA_TIMEOUT,
) -> dict[str, Any]:
    """Generate a streaming Ollama response and capture usage and timing."""

    start_time = time.perf_counter()
    first_content_seconds = None
    content_parts = []
    final_data = {}

    streaming_payload = {**payload, "stream": True}

    with requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=streaming_payload,
        timeout=timeout,
        stream=True,
    ) as response:
        response.raise_for_status()

        for line in response.iter_lines():
            if not line:
                continue

            if isinstance(line, bytes):
                line = line.decode("utf-8")

            chunk = json.loads(line)
            content = chunk.get("message", {}).get("content", "")

            if content:
                if first_content_seconds is None:
                    first_content_seconds = round(
                        time.perf_counter() - start_time, 4
                    )

                    # Record the first generated content event in LangSmith.
                    run = get_current_run_tree()
                    if run is not None:
                        run.add_event({
                            "name": "new_token",
                            "time": datetime.now(
                                timezone.utc
                            ).isoformat(),
                        })

                content_parts.append(content)

            if chunk.get("done"):
                final_data = chunk

    latency_seconds = round(time.perf_counter() - start_time, 3)

    model = final_data.get("model") or payload["model"]
    input_tokens = final_data.get("prompt_eval_count")
    output_tokens = final_data.get("eval_count")

    valid_counts = (
        _valid_token_count(input_tokens)
        and _valid_token_count(output_tokens)
    )

    total_tokens = (
        input_tokens + output_tokens if valid_counts else None
    )

    def seconds_from_ns(field: str) -> float | None:
        value = final_data.get(field)

        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return round(value / 1_000_000_000, 4)

        return None

    # Calculate reference pricing estimates.
    cost = calculate_reference_cost(
        model,
        input_tokens,
        output_tokens,
    )

    input_cost_low = cost["input_cost_usd_low"]
    input_cost_high = cost["input_cost_usd_high"]
    output_cost_low = cost["output_cost_usd_low"]
    output_cost_high = cost["output_cost_usd_high"]

    # Use midpoint estimates for LangSmith's single cost values.
    input_cost = (
        (input_cost_low + input_cost_high) / 2
        if input_cost_low is not None and input_cost_high is not None
        else None
    )

    output_cost = (
        (output_cost_low + output_cost_high) / 2
        if output_cost_low is not None and output_cost_high is not None
        else None
    )

    usage_metadata = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }

    if input_cost is not None and output_cost is not None:
        usage_metadata.update({
            "input_cost": input_cost,
            "output_cost": output_cost,
            "total_cost": input_cost + output_cost,
        })

    # Attach usage and model metadata to the active LangSmith run.
    run = get_current_run_tree()

    if run is not None:
        run.set(
            metadata={
                "ls_provider": "ollama",
                "ls_model_name": model,
            },
            usage_metadata=usage_metadata,
        )

    return {
        "answer": "".join(content_parts).strip(),
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "usage_metadata": usage_metadata,
        "latency_seconds": latency_seconds,
        "first_token_seconds": first_content_seconds,
        "generation_seconds": seconds_from_ns("eval_duration"),
        "prompt_eval_seconds": seconds_from_ns(
            "prompt_eval_duration"
        ),
        "load_seconds": seconds_from_ns("load_duration"),
        "ollama_total_seconds": seconds_from_ns("total_duration"),
        "done": final_data.get("done"),
        **cost,
    }
