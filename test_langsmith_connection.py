
from langsmith import traceable
from app.config import (
    LANGSMITH_TRACING,
    LANGSMITH_API_KEY,
    LANGSMITH_PROJECT,
    LANGSMITH_ENDPOINT,
)

print("Tracing enabled:", LANGSMITH_TRACING)
print("API key configured:", bool(LANGSMITH_API_KEY))
print("Project:", LANGSMITH_PROJECT)
print("Endpoint:", LANGSMITH_ENDPOINT)


@traceable(name="LangSmith Connection Test")
def test_trace():
    return {"status": "LangSmith test successful"}


if __name__ == "__main__":
    result = test_trace()
    print("Test result:", result)
