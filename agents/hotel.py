import time
from pydantic import BaseModel, Field
from config.model import model
from config.state import TravelState
from database.repository import record_execution_step_safe
from tools.hotel import search_hotels

class HotelDecision(BaseModel):
    hotel_name: str = Field(
        description="最终最推荐的一家酒店正式名称"
    )
    summary: str = Field(
        description="完整酒店推荐分析"
    )

hotel_output_model = model.with_structured_output(
    HotelDecision
)

def hotel_node(state: TravelState):
    total_start = time.time()
    agent_run_id = state.get("agent_run_id")
    tool_start = time.time()
    try:
        tool_result = search_hotels.invoke(
            {
                "city": state["destination"],
                "keyword": "酒店",
            }
        )
        tool_duration = time.time() - tool_start
        record_execution_step_safe(
            agent_run_id=agent_run_id,
            step_type="tool",
            step_name="amap_hotel",
            duration=tool_duration,
            success=True,
        )
    except Exception as e:
        tool_duration = time.time() - tool_start
        record_execution_step_safe(
            agent_run_id=agent_run_id,
            step_type="tool",
            step_name="amap_hotel",
            duration=tool_duration,
            success=False,
            error_message=str(e),
        )
        raise
    print(
        f"[Hotel] search_hotels Tool耗时："
        f"{tool_duration:.2f}s"
    )

    prompt = f"""
    你是一个专业的旅游住宿 Agent。
    请根据用户需求、景点信息以及真实酒店查询结果，推荐合适的住宿区域和酒店。
    ================ 用户信息 ================
    目的地：{state["destination"]}
    旅游天数：{state["days"]}
    人数：{state["people"]}
    总预算：{state["budget"]}
    用户偏好：{state["preferences"]}
    用户限制：{state["constraints"]}
    ================ 景点信息 ================
    {state["attractions_result"]}
    核心景点：
    {state.get("attraction_names", [])}
    ================ 高德酒店 POI 查询结果 ================
    {tool_result}
    ================ 任务 ================
    请从真实工具结果中选择 2-4 个合理住宿方案。
    重点考虑：
    1. 靠近主要景点
    2. 靠近地铁或公共交通
    3. 避免每天大量往返
    4. 符合整体预算
    5. 符合用户限制
    6. 酒店位置与主要景点之间是否方便
    重要要求：
    1. 不要编造工具结果中不存在的酒店
    2. 不要编造不存在的地址、评分或电话
    3. cost 只能称为“参考消费”，不能称为“实时房价”
    4. 工具没有返回价格或评分时不要自行虚构
    5. 优先选择交通方便、位置合理的酒店
    最终给出：
    - 推荐酒店名称
    - 酒店所在区域
    - 地址
    - 评分（如工具提供）
    - 参考消费（如工具提供）
    - 推荐理由
    - 与主要景点的位置关系
    """
    llm_start = time.time()
    try:
        decision = hotel_output_model.invoke(prompt)
        llm_duration = time.time() - llm_start
        record_execution_step_safe(
            agent_run_id=agent_run_id,
            step_type="llm",
            step_name="hotel_llm",
            duration=llm_duration,
            success=True,
        )
    except Exception as e:
        llm_duration = time.time() - llm_start
        record_execution_step_safe(
            agent_run_id=agent_run_id,
            step_type="llm",
            step_name="hotel_llm",
            duration=llm_duration,
            success=False,
            error_message=str(e),
        )
        raise
    print(
        f"[Hotel] LLM耗时："
        f"{llm_duration:.2f}s"
    )
    print(
        f"[Hotel] 总耗时："
        f"{time.time() - total_start:.2f}s"
    )
    #
    # print(
    #     "\n========== Hotel Agent =========="
    # )
    # print(decision.summary)
    return {
        "hotel_result": decision.summary,
        "hotel_name": decision.hotel_name,
    }
