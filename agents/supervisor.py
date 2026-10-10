from config.state import TravelState
from config.revision import INVALIDATION_RULES, RESULT_FIELDS, validate_node_update


def supervisor_node(state: TravelState):
    next_agent = "planner"
    update = {}
    for agent_name in ("attraction", "hotel", "transport"):
        if state.get(f"{agent_name}_status", "pending") != "completed":
            next_agent = agent_name
            break
        try:
            validate_node_update(agent_name, state)
        except ValueError:
            next_agent = agent_name
            break
    if next_agent in INVALIDATION_RULES:
        # If a previously completed upstream result is unusable, its dependents
        # cannot remain completed merely because their old text is nonempty.
        for affected in INVALIDATION_RULES[next_agent]:
            if state.get(f"{affected}_status") == "completed":
                update[f"{affected}_status"] = "dirty"
                for field in RESULT_FIELDS[affected]:
                    update[field] = [] if field == "attraction_names" else ""
    print("\n========== Supervisor ==========")
    print("下一步：", next_agent)
    return {
        **update,
        "next_agent": next_agent
    }
