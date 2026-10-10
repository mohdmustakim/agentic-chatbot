
from statistics import mean


def routing_accuracy(results: list[dict]) -> float:
    """Fraction of queries routed to the expected agent."""
    if not results:
        return 0.0

    correct = sum(
        result.get("actual_intent") == result.get("expected_intent")
        for result in results
    )

    return correct / len(results)


def keyword_coverage(
    answer: str,
    expected_keywords: list[str],
) -> float:
    """Fraction of expected keywords found in an answer."""
    if not expected_keywords:
        return 1.0

    answer = answer.lower()

    matched = sum(
        keyword.lower() in answer
        for keyword in expected_keywords
    )

    return matched / len(expected_keywords)


def latency_summary(results: list[dict]) -> dict:
    """Summarize response latency in seconds."""
    latencies = [
        result["latency_seconds"]
        for result in results
        if result.get("latency_seconds") is not None
    ]

    if not latencies:
        return {
            "count": 0,
            "average_seconds": 0.0,
            "max_seconds": 0.0,
        }

    return {
        "count": len(latencies),
        "average_seconds": round(mean(latencies), 3),
        "max_seconds": round(max(latencies), 3),
    }


def pass_rate(results: list[dict]) -> float:
    """Fraction of results whose reviewer decision is PASS."""
    if not results:
        return 0.0

    passed = sum(
        result.get("review_decision") == "PASS"
        for result in results
    )

    return passed / len(results)
