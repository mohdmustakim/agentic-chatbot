
from app.memory.checkpointer import checkpointer
from langgraph.graph import StateGraph, START, END


from app.graph.state import ChatState
from app.graph.routing import (
    route_query,
    route_review,
    prepare_retry,
)
from app.agents.supervisor import supervisor_node
from app.agents.coding_agent import coding_node
from app.agents.research_agent import research_node
from app.agents.planning_agent import planning_node
from app.agents.general_agent import general_node
from app.agents.reviewer_agent import reviewer_node


def build_graph():
    graph = StateGraph(ChatState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("coding", coding_node)
    graph.add_node("research", research_node)
    graph.add_node("planning", planning_node)
    graph.add_node("general", general_node)
    graph.add_node("reviewer", reviewer_node)
    graph.add_node("prepare_retry", prepare_retry)

    graph.add_edge(START, "supervisor")

    graph.add_conditional_edges(
        "supervisor",
        route_query,
        {
            "coding": "coding",
            "research": "research",
            "planning": "planning",
            "general": "general",
        },
    )

    for agent in ["coding", "research", "planning", "general"]:
        graph.add_edge(agent, "reviewer")

    graph.add_conditional_edges(
        "reviewer",
        route_review,
        {
            "retry": "prepare_retry",
            "finish": END,
        },
    )

    graph.add_conditional_edges(
        "prepare_retry",
        route_query,
        {
            "coding": "coding",
            "research": "research",
            "planning": "planning",
            "general": "general",
        },
    )

    return graph.compile(checkpointer=checkpointer)


app_graph = build_graph()
