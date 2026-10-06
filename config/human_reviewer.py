from langgraph.types import interrupt

from config.state import TravelState


def human_review_node(state: TravelState):
    print("\n========== 人工审核 ==========")
    print("\n当前旅游方案：")
    print(state["planner_result"])
    print("\nAI审核结果：")
    print(state["review_result"])

    human_decision = interrupt(
        {
            "type": "human_review",
            "message": "请审核当前旅游方案",
            "planner_result": state["planner_result"],
            "review_result": state["review_result"],
        }
    )

    print("\n========== 收到人工审核结果 ==========")
    print(human_decision)

    approved = bool(human_decision["approved"])
    comment = human_decision.get("comment", "")

    result = {
        "human_review_passed": approved,
        "human_review_result": comment,
    }

    if approved:
        result["trigger_reason"] = "human_review_pass"
        return result

    result["trigger_reason"] = "human_review_reject"
    result["human_revision_count"] = (
        state.get("human_revision_count", 0) + 1
    )

    return result
