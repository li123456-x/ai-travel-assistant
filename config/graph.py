from langgraph.graph import END, START, StateGraph

from agents.attraction import attraction_node
from agents.hotel import hotel_node
from agents.planner import planner_node
from agents.reviewer import reviewer_node
from agents.supervisor import supervisor_node
from agents.transport import transport_node
from config.human_reviewer import human_review_node
from config.route import route_human_review, route_reviewer, route_supervisor
from config.state import TravelState
from harness.runtime import run_with_harness


def attraction_harness_node(state):
    return run_with_harness(
        agent_name="attraction",
        agent_func=attraction_node,
        state=state,
        max_retries=1,
    )


def hotel_harness_node(state):
    return run_with_harness(
        agent_name="hotel",
        agent_func=hotel_node,
        state=state,
        max_retries=1,
    )


def transport_harness_node(state):
    return run_with_harness(
        agent_name="transport",
        agent_func=transport_node,
        state=state,
        max_retries=1,
    )


def planner_harness_node(state):
    return run_with_harness(
        agent_name="planner",
        agent_func=planner_node,
        state=state,
        max_retries=1,
    )


def reviewer_harness_node(state):
    return run_with_harness(
        agent_name="reviewer",
        agent_func=reviewer_node,
        state=state,
        max_retries=1,
    )


builder = StateGraph(TravelState)

builder.add_node("supervisor", supervisor_node)
builder.add_node("attraction", attraction_harness_node)
builder.add_node("hotel", hotel_harness_node)
builder.add_node("transport", transport_harness_node)
builder.add_node("planner", planner_harness_node)
builder.add_node("reviewer", reviewer_harness_node)
builder.add_node("human_review", human_review_node)

builder.add_edge(START, "supervisor")

builder.add_conditional_edges(
    "supervisor",
    route_supervisor,
    {
        "attraction": "attraction",
        "hotel": "hotel",
        "transport": "transport",
        "planner": "planner",
        "end": END,
    },
)

builder.add_edge("attraction", "supervisor")
builder.add_edge("hotel", "supervisor")
builder.add_edge("transport", "supervisor")
builder.add_edge("planner", "reviewer")

builder.add_conditional_edges(
    "reviewer",
    route_reviewer,
    {
        "pass": "human_review",
        "reject": "planner",
    },
)

builder.add_conditional_edges(
    "human_review",
    route_human_review,
    {
        "approve": END,
        "reject": "planner",
    },
)
