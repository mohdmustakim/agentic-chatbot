import pytest

from app.agents import supervisor


@pytest.fixture(autouse=True)
def no_llm_router(monkeypatch):
    """Keep tests offline: the supervisor uses keyword routing by default.

    Tests that exercise the LLM router re-enable it and mock the model call.
    """
    monkeypatch.setattr(supervisor, "USE_LLM_ROUTER", False)
