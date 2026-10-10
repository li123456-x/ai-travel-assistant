from langgraph.types import interrupt

from config.state import TravelState
from config.revision import build_revision_update


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

    return build_revision_update(state, human_decision)
