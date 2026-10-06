from pydantic import BaseModel, Field
import time
from config.model import model
from database.repository import record_execution_step_safe
from tools.attraction import search_attractions

class AttractionDecision(BaseModel):
    attraction_names: list[str] = Field(
        description="最终推荐的景点名称列表，只填写景点正式名称"
    )
    summary: str = Field(
        description="完整的景点推荐结果"
    )
attraction_output_model = model.with_structured_output(
    AttractionDecision
)

def attraction_node(state):
    total_start = time.time()
    agent_run_id = state.get("agent_run_id")
    destination = state["destination"]
    preferences = state.get("preferences",[])
    tool_start = time.time()
    try:
        query = (
            f"{destination} "
            f"{preferences} "
            f"旅游景点推荐 "
            f"门票 开放时间 预约 游玩时间"
        )
        tool_result = (
            search_attractions.invoke(
                {
                    "query": query
                }
            )
        )
        tool_duration = (
            time.time()- tool_start
        )
        record_execution_step_safe(
            agent_run_id=agent_run_id,
            step_type="tool",
            step_name="tavily_attraction",
            duration=tool_duration,
            success=True
        )
    except Exception as e:
        tool_duration = (
            time.time()
            - tool_start
        )
        record_execution_step_safe(
            agent_run_id=agent_run_id,
            step_type="tool",
            step_name="tavily_attraction",
            duration=tool_duration,
            success=False,
            error_message=str(e)
        )
        raise
    print(
        f"[Attraction] Tavily耗时："
        f"{tool_duration:.2f}s"
    )

    prompt = f"""
    你是一名专业旅游景点推荐 Agent。
    目的地：{destination}
    用户偏好：{preferences}
    检索结果：{tool_result}
    请根据真实检索结果推荐合适景点。
    要求：
    1. 只推荐检索结果中有依据的景点。
    2. 考虑用户偏好。
    3. 给出简洁推荐理由。
    4. 输出最终景点名称列表。
"""
    llm_start = time.time()
    try:
        decision = (
            attraction_output_model.invoke(prompt)
        )
        llm_duration = (
            time.time()- llm_start
        )
        record_execution_step_safe(
            agent_run_id=agent_run_id,
            step_type="llm",
            step_name="attraction_llm",
            duration=llm_duration,
            success=True
        )
    except Exception as e:
        llm_duration = (
            time.time()- llm_start
        )
        record_execution_step_safe(
            agent_run_id=agent_run_id,
            step_type="llm",
            step_name="attraction_llm",
            duration=llm_duration,
            success=False,
            error_message=str(e)
        )
        raise
    print(
        f"[Attraction] LLM耗时："
        f"{llm_duration:.2f}s"
    )
    print(
        f"[Attraction] 总耗时："
        f"{time.time() - total_start:.2f}s"
    )
    # print("\n========== Attraction Agent ==========")
    # print("最终选择景点：")
    # print(decision.attraction_names)
    # print("\n景点推荐说明：")
    # print(decision.summary)
    return {
        "attractions_result":decision.summary,
        "attraction_names":decision.attraction_names
    }
