
from app.evaluation.metrics import (
    routing_accuracy,
    keyword_coverage,
    latency_summary,
    pass_rate,
)

sample_results = [
    {
        "expected_intent": "coding",
        "actual_intent": "coding",
        "latency_seconds": 2.5,
        "review_decision": "PASS",
    },
    {
        "expected_intent": "research",
        "actual_intent": "general",
        "latency_seconds": 3.5,
        "review_decision": "REVISE",
    },
]

print("Routing accuracy:", routing_accuracy(sample_results))
print(
    "Keyword coverage:",
    keyword_coverage("RAG retrieves relevant context", ["RAG", "context"]),
)
print("Latency:", latency_summary(sample_results))
print("Reviewer pass rate:", pass_rate(sample_results))
