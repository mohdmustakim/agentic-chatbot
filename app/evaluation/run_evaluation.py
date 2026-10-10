
import json
import time
from pathlib import Path
from uuid import uuid4

from app.graph.workflow import app_graph
from app.evaluation.dataset import EVALUATION_DATASET
from app.evaluation.metrics import (
    routing_accuracy,
    keyword_coverage,
    latency_summary,
    pass_rate,
)


def run_evaluation():
    results = []

    print("\n========== AGENTIC CHATBOT EVALUATION ==========\n")

    for test_case in EVALUATION_DATASET:
        test_id = test_case["id"]
        query = test_case["query"]

        print(f"\nRunning test: {test_id}")
        print(f"Query: {query}")

        initial_state = {
            "query": query,
            "original_query": query,
            "intent": "",
            "draft": "",
            "review": "",
            "answer": "",
            "retry_count": 0,
            "history": [],
        }

        config = {
            "configurable": {
                "thread_id": f"eval-{test_id}-{uuid4().hex[:8]}"
            }
        }

        start_time = time.perf_counter()

        try:
            result = app_graph.invoke(
                initial_state,
                config=config,
            )

            latency = time.perf_counter() - start_time
            answer = result.get("answer", "").strip()
            review = result.get("review", "").strip()

            decision = review.split(":", 1)[0].strip().upper()

            if decision not in {"PASS", "REVISE"}:
                decision = "UNKNOWN"

            actual_intent = result.get("intent", "unknown")

            coverage = keyword_coverage(
                answer,
                test_case.get("expected_keywords", []),
            )

            record = {
                "test_id": test_id,
                "query": query,
                "expected_intent": test_case["expected_intent"],
                "actual_intent": actual_intent,
                "routing_correct": (
                    actual_intent == test_case["expected_intent"]
                ),
                "keyword_coverage": round(coverage, 3),
                "latency_seconds": round(latency, 3),
                "review_decision": decision,
                "retry_count": result.get("retry_count", 0),
                "rag_required": test_case.get("rag_required", False),
                "answer": answer,
                "review": review,
                "error": None,
            }

        except Exception as exc:
            latency = time.perf_counter() - start_time

            record = {
                "test_id": test_id,
                "query": query,
                "expected_intent": test_case["expected_intent"],
                "actual_intent": "error",
                "routing_correct": False,
                "keyword_coverage": 0.0,
                "latency_seconds": round(latency, 3),
                "review_decision": "ERROR",
                "retry_count": 0,
                "rag_required": test_case.get("rag_required", False),
                "answer": "",
                "review": "",
                "error": str(exc),
            }

        results.append(record)

        print(f"Actual route: {record['actual_intent']}")
        print(f"Routing correct: {record['routing_correct']}")
        print(f"Keyword coverage: {record['keyword_coverage']}")
        print(f"Reviewer: {record['review_decision']}")
        print(f"Latency: {record['latency_seconds']} seconds")

        if record["error"]:
            print(f"Error: {record['error']}")

    print("\n========== EVALUATION SUMMARY ==========\n")

    print(
        "Routing accuracy:",
        f"{routing_accuracy(results) * 100:.1f}%",
    )
    print(
        "Average keyword coverage:",
        f"{sum(r['keyword_coverage'] for r in results) / len(results) * 100:.1f}%"
        if results else "0.0%",
    )
    print("Latency:", latency_summary(results))
    print("Reviewer pass rate:", f"{pass_rate(results) * 100:.1f}%")
    print(
        "Failed tests:",
        sum(1 for r in results if r["error"]),
    )

    output_path = Path("evaluation_results.json")
    output_path.write_text(
        json.dumps(results, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"\nDetailed results saved to: {output_path.resolve()}")


if __name__ == "__main__":
    run_evaluation()
