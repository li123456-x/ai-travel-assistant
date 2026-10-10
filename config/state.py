from typing import Literal, TypedDict


class TravelState(TypedDict):
    session_id: int | None
    thread_id: str
    trigger_reason: str
    agent_run_id: int | None
    # 节点状态：pending、dirty、completed
    attraction_status: Literal["pending", "dirty", "completed"]
    hotel_status: Literal["pending", "dirty", "completed"]
    transport_status: Literal["pending", "dirty", "completed"]

    # 每类节点分别保存自己的最新修改要求
    attraction_revision_request: str
    hotel_revision_request: str
    transport_revision_request: str
    planner_revision_request: str

    # 具体查询和筛选条件
    attraction_excluded_names: list[str]
    hotel_search_keyword: str
    hotel_excluded_names: list[str]
    city_transport_mode: Literal["transit", "driving", "walking"]
    intercity_transport_mode: Literal["train", "flight"]

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
