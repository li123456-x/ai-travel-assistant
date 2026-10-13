from config.state import TravelState
from config.revision import INVALIDATION_RULES, RESULT_FIELDS, validate_node_update
from config.trip_policy import requires_hotel, skipped_hotel_update


def supervisor_node(state: TravelState):
    next_agent = "planner"
    needs_hotel = requires_hotel(state)
    update = {} if needs_hotel else skipped_hotel_update()
    effective_state = {**state, **update}
    if not needs_hotel and state.get("hotel_name") and state.get("transport_status") == "completed":
        update.update({"transport_status": "dirty", "transport_result": "", "transport_segments": []})
        effective_state.update(update)
    agents = ("attraction", "hotel", "transport") if needs_hotel else ("attraction", "transport")
    for agent_name in agents:
        if effective_state.get(f"{agent_name}_status", "pending") != "completed":
            next_agent = agent_name
            break
        try:
            validate_node_update(agent_name, effective_state, state=effective_state)
        except ValueError:
            next_agent = agent_name
            break
    if next_agent in INVALIDATION_RULES:
        # If a previously completed upstream result is unusable, its dependents
        # cannot remain completed merely because their old text is nonempty.
        for affected in INVALIDATION_RULES[next_agent]:
            if affected == "hotel" and not needs_hotel:
                continue
            if state.get(f"{affected}_status") == "completed":
                update[f"{affected}_status"] = "dirty"
                for field in RESULT_FIELDS[affected]:
                    update[field] = [] if field in {"attraction_names", "transport_segments"} else ""
    print("\n========== Supervisor ==========")
    print("下一步：", next_agent)
    return {
        **update,
        "next_agent": next_agent
    }
