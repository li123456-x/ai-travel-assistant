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
from config.revision import validate_node_update
from harness.runtime import run_with_harness


def _run_checked_agent(agent_name, agent_func, state):
    def checked_agent(agent_state):
        return validate_node_update(agent_name, agent_func(agent_state), state=agent_state)

    return run_with_harness(
        agent_name=agent_name,
        agent_func=checked_agent,
        state=state,
        max_retries=1,
    )


def attraction_harness_node(state):
    return _run_checked_agent("attraction", attraction_node, state)


def hotel_harness_node(state):
    return _run_checked_agent("hotel", hotel_node, state)


def transport_harness_node(state):
    return _run_checked_agent("transport", transport_node, state)


def planner_harness_node(state):
    return _run_checked_agent("planner", planner_node, state)


def reviewer_harness_node(state):
    return _run_checked_agent("reviewer", reviewer_node, state)


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
        "reject": "supervisor",
    },
)
