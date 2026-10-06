from config.state import TravelState


def supervisor_node(state: TravelState):
    if not state["attractions_result"]:
        next_agent = "attraction"
    elif not state["hotel_result"]:
        next_agent = "hotel"
    elif not state["transport_result"]:
        next_agent = "transport"
    else:
        next_agent = "planner"
    print("\n========== Supervisor ==========")
    print("下一步：", next_agent)
    return {
        "next_agent": next_agent
    }