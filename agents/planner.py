import time
from config.model import model
from config.state import TravelState
from config.revision import build_revision_context
from database.repository import record_execution_step_safe

def _build_revision_instruction(state: TravelState) -> str:
    instructions = [
        "以当前最新景点、酒店和交通结果为准；旧行程仅供参考。",
        "住宿必须使用当前 hotel_name 指定的酒店完整名称，不得自行替换。",
        "不得安排排除名单中的景点，交通方式必须遵循当前实际查询结果。",
        build_revision_context(state),
    ]
    if state.get("trigger_reason") == "ai_review_reject":
        instructions.append("同时修正以下 AI 审核问题：\n" + state.get("review_result", ""))
    return "\n".join(instructions)

def planner_node(state: TravelState):
    total_start = time.time()
    agent_run_id = state.get("agent_run_id")
    revision_instruction = _build_revision_instruction(state)
    prompt = f"""
    你是一个专业的旅游行程规划 Agent。
    你的任务是根据用户需求，以及景点、酒店、交通三个专业 Agent 提供的信息，
    制定一份完整、合理、可执行的旅游行程。
    ================ 用户需求 ================
    用户需求：{state["user_request"]}
    出发地：{state["origin"]}
    目的地：{state["destination"]}
    出发日期：{state["departure_date"]}
    返程日期：{state["return_date"]}
    旅游天数：{state["days"]} 天
    人数：{state["people"]} 人
    预算：{state["budget"]} 元
    用户偏好：{state["preferences"]}
    用户限制：{state["constraints"]}
    ================ 景点 Agent 结果 ================
    {state["attractions_result"]}
    核心景点：
    {state.get("attraction_names", [])}
    ================ 酒店 Agent 结果 ================
    {state["hotel_result"]}
    最终酒店：
    {state.get("hotel_name", "")}
    ================ 交通 Agent 结果 ================
    {state["transport_result"]}
    ================ 当前旅游方案 ================
    {state.get("planner_result", "")}
    ================ 修改要求 ================
    {revision_instruction}
    ================ Planner 任务 ================
    请生成一份完整旅游方案。
    要求：
    1. 按照旅游天数进行安排
    2. 每天安排合理数量的景点
    3. 不要安排过于紧凑的行程
    4. 尽量将距离较近的景点安排在同一天
    5. 合理安排酒店与景点之间的交通
    6. 考虑用户预算、偏好和限制
    7. 给出每天的大致时间安排和交通方式
    8. 城际交通优先遵循 Transport Agent 的 Tavily 查询结果
    9. 市内交通优先遵循 Transport Agent 的高德查询结果
    10. 不得自行改变真实交通数据
    11. 不得编造工具结果中不存在的酒店、景点、车次或航班
"""
    llm_start = time.time()
    try:
        result = model.invoke(prompt)
        if not isinstance(result.content, str) or not result.content.strip():
            raise ValueError("行程生成结果为空")
        llm_duration = time.time() - llm_start
        record_execution_step_safe(
            agent_run_id=agent_run_id,
            step_type="llm",
            step_name="planner_llm",
            duration=llm_duration,
            success=True,
        )
    except Exception as e:
        llm_duration = time.time() - llm_start
        record_execution_step_safe(
            agent_run_id=agent_run_id,
            step_type="llm",
            step_name="planner_llm",
            duration=llm_duration,
            success=False,
            error_message=str(e),
        )
        raise
    print(f"[Planner] LLM耗时：{time.time() - llm_start:.2f}s")
    print(f"[Planner] 总耗时：{time.time() - total_start:.2f}s")
    # print("\n========== Planner ==========")
    # print(result.content)
    return {
        "planner_result": result.content,
    }
