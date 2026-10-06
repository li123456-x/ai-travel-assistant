import time
from pydantic import BaseModel, Field
from config.model import model
from config.state import TravelState
from database.repository import record_execution_step_safe

class ReviewDecision(BaseModel):
    review_passed: bool = Field(
        description="旅游方案是否通过审核"
    )
    review_result: str = Field(
        description="简洁指出方案关键问题和必要修改建议"
    )

reviewer_model = model.with_structured_output(ReviewDecision)

def reviewer_node(state: TravelState):
    total_start = time.time()
    agent_run_id = state.get("agent_run_id")
    prompt = f"""
    你是一个专业的旅游方案审核 Agent。
    请审核下面的旅游方案。
    ================ 用户需求 ================
    {state["user_request"]}
    目的地：{state["destination"]}
    天数：{state["days"]}
    人数：{state["people"]}
    预算：{state["budget"]}
    用户偏好：{state["preferences"]}
    用户限制：{state["constraints"]}
    ================ 当前旅游方案 ================
    {state["planner_result"]}
    ================ 审核要求 ================
    请检查：
    1. 是否符合用户需求
    2. 是否符合旅游天数
    3. 时间安排是否合理
    4. 景点数量是否合理
    5. 交通是否合理
    6. 是否存在明显往返
    7. 酒店位置是否合理
    8. 是否可能超出预算
    9. 是否符合用户偏好
    10. 是否违反用户限制
    11. 是否过于疲劳
    12. 去返程时间是否合理
    13. 市内交通是否与高德结果明显冲突
    14. 返程安排是否预留足够到站时间
    只检查影响方案实际可执行性的关键问题。
    如果没有重大问题，直接通过。
    审核结果控制在 300 字以内，不要重复完整旅游方案。
"""
    llm_start = time.time()
    try:
        decision = reviewer_model.invoke(prompt)
        llm_duration = time.time() - llm_start
        record_execution_step_safe(
            agent_run_id=agent_run_id,
            step_type="llm",
            step_name="reviewer_llm",
            duration=llm_duration,
            success=True,
        )
    except Exception as e:
        llm_duration = time.time() - llm_start
        record_execution_step_safe(
            agent_run_id=agent_run_id,
            step_type="llm",
            step_name="reviewer_llm",
            duration=llm_duration,
            success=False,
            error_message=str(e),
        )
        raise
    print(f"[Reviewer] LLM耗时：{time.time() - llm_start:.2f}s")
    print(f"[Reviewer] 总耗时：{time.time() - total_start:.2f}s")
    print("\n========== AI Reviewer ==========")
    print("审核是否通过：", decision.review_passed)
    print(decision.review_result)
    result = {
        "review_result": decision.review_result,
        "review_passed": decision.review_passed,
    }
    if decision.review_passed:
        result["trigger_reason"] = "ai_review_pass"
        return result
    retry_count = state.get("review_retry_count", 0) + 1
    result["review_retry_count"] = retry_count
    result["trigger_reason"] = "ai_review_reject"
    return result
