
EVALUATION_DATASET = [
    {
        "id": "coding_01",
        "query": "Write a Python function to reverse a string",
        "expected_intent": "coding",
        "expected_keywords": ["def", "return"],
        "rag_required": False,
    },
    {
        "id": "coding_02",
        "query": "Explain how to debug a Python exception",
        "expected_intent": "coding",
        "expected_keywords": ["error", "debug"],
        "rag_required": False,
    },
    {
        "id": "research_01",
        "query": "Explain RAG architecture in simple words",
        "expected_intent": "research",
        "expected_keywords": ["retrieval", "context"],
        "rag_required": False,
    },
    {
        "id": "research_02",
        "query": "What is the difference between classification and regression?",
        "expected_intent": "research",
        "expected_keywords": ["classification", "regression"],
        "rag_required": False,
    },
    {
        "id": "planning_01",
        "query": "Create a roadmap for building an AI chatbot",
        "expected_intent": "planning",
        "expected_keywords": ["phase", "testing"],
        "rag_required": False,
    },
    {
        "id": "planning_02",
        "query": "Plan a project to deploy a machine learning model",
        "expected_intent": "planning",
        "expected_keywords": ["deployment", "testing"],
        "rag_required": False,
    },
    {
        "id": "general_01",
        "query": "Hi, how are you?",
        "expected_intent": "general",
        "expected_keywords": ["help"],
        "rag_required": False,
    },
    {
        "id": "general_02",
        "query": "Tell me a fun fact",
        "expected_intent": "general",
        "expected_keywords": [],
        "rag_required": False,
    },
    {
        "id": "rag_01",
        "query": (
            "According to the IRCTC cancellation policy document, "
            "what is the cancellation charge for a confirmed 3A ticket "
            "cancelled 24 hours before departure?"
        ),
        "expected_intent": "research",
        "expected_keywords": ["25%", "48 hours", "12 hours"],
        "rag_required": True,
    },
    {
        "id": "rag_02",
        "query": (
            "According to the uploaded train cancellation document, "
            "what happens if a train is cancelled completely?"
        ),
        "expected_intent": "research",
        "expected_keywords": ["refund"],
        "rag_required": True,
    },
]
