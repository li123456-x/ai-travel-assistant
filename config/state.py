from typing import TypedDict


class TravelState(TypedDict):
    session_id: int | None
    thread_id: str
    trigger_reason: str
    agent_run_id: int | None

    user_request: str
    origin: str
    destination: str
    departure_date: str
    return_date: str
    days: int
    people: int
    budget: float
    preferences: list[str]
    constraints: list[str]

    attractions_result: str
    attraction_names: list[str]
    hotel_result: str
    hotel_name: str
    transport_result: str
    planner_result: str

    review_result: str
    review_passed: bool
    review_retry_count: int
    max_review_retries: int

    human_review_result: str
    human_review_passed: bool
    human_revision_count: int
    max_human_revisions: int

    next_agent: str
