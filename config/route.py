from config.state import TravelState


def route_supervisor(state: TravelState):
    return state["next_agent"]


def route_reviewer(state: TravelState):
    if state.get("review_passed", False):
        return "pass"

    review_retry_count = state.get("review_retry_count", 0)
    max_review_retries = state.get("max_review_retries", 2)

    if review_retry_count >= max_review_retries:
        return "pass"

    return "reject"


def route_human_review(state: TravelState):
    if state.get("human_review_passed", False):
        return "approve"

    return "reject"
