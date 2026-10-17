
"""Centralized configuration loaded from environment variables and .env."""

import os
from dotenv import load_dotenv

# Load environment variables from the project-root .env.
load_dotenv()


def env_int(name: str, default: int) -> int:
    return int(os.getenv(name, str(default)))


def env_float(name: str, default: float) -> float:
    return float(os.getenv(name, str(default)))


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


# Ollama connection and model selection
OLLAMA_BASE_URL = os.getenv(
    "OLLAMA_BASE_URL", "http://localhost:11434"
).rstrip("/")
OLLAMA_TIMEOUT = env_int("OLLAMA_TIMEOUT", 180)

ROUTER_MODEL = os.getenv("ROUTER_MODEL", "qwen3.5:4b")
AGENT_MODEL = os.getenv("AGENT_MODEL", "qwen3.5:latest")


# Generation settings
ROUTER_TEMPERATURE = env_float("ROUTER_TEMPERATURE", 0.0)
AGENT_TEMPERATURE = env_float("AGENT_TEMPERATURE", 0.0)

ROUTER_NUM_PREDICT = env_int("ROUTER_NUM_PREDICT", 512)
ROUTER_TIMEOUT = env_int("ROUTER_TIMEOUT", 30)
USE_LLM_ROUTER = env_bool("USE_LLM_ROUTER", True)
AGENT_NUM_PREDICT = env_int("AGENT_NUM_PREDICT", 512)
RESEARCH_NUM_PREDICT = env_int("RESEARCH_NUM_PREDICT", 1200)
REVIEWER_NUM_PREDICT = env_int("REVIEWER_NUM_PREDICT", 128)

DEFAULT_CONTEXT_LENGTH = env_int("DEFAULT_CONTEXT_LENGTH", 8192)
MAX_RETRIES = env_int("MAX_RETRIES", 1)


# LangSmith tracing
LANGSMITH_TRACING = env_bool("LANGSMITH_TRACING", True)
LANGSMITH_API_KEY = os.getenv("LANGSMITH_API_KEY", "").strip()
LANGSMITH_PROJECT = os.getenv(
    "LANGSMITH_PROJECT", "agentic-chatbot-local"
)
LANGSMITH_ENDPOINT = os.getenv(
    "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
)


# Reference pricing: USD per 1 million tokens.
# These are external-provider estimates, not local Ollama charges.
MODEL_PRICING = {
    "qwen3.5:latest": {
        "input_low": env_float("QWEN35_LATEST_INPUT_PRICE_LOW", 0.08),
        "input_high": env_float("QWEN35_LATEST_INPUT_PRICE_HIGH", 0.24),
        "output_low": env_float("QWEN35_LATEST_OUTPUT_PRICE_LOW", 0.75),
        "output_high": env_float("QWEN35_LATEST_OUTPUT_PRICE_HIGH", 1.80),
    },
    "qwen3.5:4b": {
        "input_low": env_float("QWEN35_4B_INPUT_PRICE_LOW", 0.17),
        "input_high": env_float("QWEN35_4B_INPUT_PRICE_HIGH", 0.17),
        "output_low": env_float("QWEN35_4B_OUTPUT_PRICE_LOW", 0.25),
        "output_high": env_float("QWEN35_4B_OUTPUT_PRICE_HIGH", 0.25),
    },
}
