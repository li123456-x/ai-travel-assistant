from pydantic import BaseModel, Field
import time
import json
import unicodedata
from config.model import model
from database.repository import record_execution_step_safe
from tools.attraction import search_attractions

class AttractionDecision(BaseModel):
    attraction_names: list[str] = Field(
        min_length=1,
        description="最终推荐的景点名称列表，只填写景点正式名称"
    )
    summary: str = Field(
        min_length=1,
        description="完整的景点推荐结果"
    )
attraction_output_model = model.with_structured_output(
    AttractionDecision
)


def _normalize_attraction_name(name: str) -> str:
    """用于去重和排除的名称键；不替代地点真实性或同名地点校验。"""
    return "".join(unicodedata.normalize("NFKC", name).split()).casefold()


def _validate_attraction_decision(decision, excluded_names: list[str]) -> None:
    if not isinstance(decision.attraction_names, list) or not decision.attraction_names:
        raise ValueError("景点名称列表不能为空")
    if not isinstance(decision.summary, str) or not decision.summary.strip():
        raise ValueError("景点推荐说明不能为空")

    excluded = {
        _normalize_attraction_name(name)
        for name in excluded_names
        if isinstance(name, str) and name.strip()
    }
    selected = []
    seen = set()
    for name in decision.attraction_names:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("景点名称不能是空白字符串")
        name = name.strip()
        normalized = _normalize_attraction_name(name)
        if normalized in excluded:
            raise ValueError(f"模型仍选择了用户排除的景点：{name}")
        if normalized not in seen:
            selected.append(name)
            seen.add(normalized)

    decision.attraction_names = selected
    decision.summary = decision.summary.strip()


def attraction_node(state):
    total_start = time.time()
    agent_run_id = state.get("agent_run_id")
    destination = state["destination"]
    preferences = state.get("preferences",[])
    revision_request = state.get("attraction_revision_request", "")
    excluded_names = state.get("attraction_excluded_names", [])
    tool_start = time.time()
    try:
        query = (
            f"{destination} "
            f"{preferences} "
            f"{state.get('days', 1)}天 "
            f"{state.get('constraints', [])} "
            f"{revision_request} "
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
        try:
            search_results = json.loads(tool_result)
        except (TypeError, json.JSONDecodeError) as exc:
            raise ValueError("景点查询未返回有效检索资料，不能继续推荐") from exc
        if not isinstance(search_results, list) or not search_results:
            raise ValueError("景点查询结果为空，请调整需求后重试")
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
    旅游天数：{state.get("days", 1)}
    出行人数：{state.get("people", 1)}
    总预算：{state.get("budget", 0)}
    用户限制：{state.get("constraints", [])}
    最新景点修改要求：{revision_request}
    禁止推荐的景点：{excluded_names}
    检索结果：{tool_result}
    请根据真实检索结果推荐合适景点。
    要求：
    1. 只推荐检索结果中有依据的景点。
    2. 考虑用户偏好。
    3. 给出简洁推荐理由。
    4. 输出最终景点名称列表。
    5. 根据旅游天数和游览时长推荐合理数量，不要为了填满行程过量推荐。
    6. 满足最新景点修改要求，不得选择禁止推荐的景点。
    7. 检索资料不足时说明不确定性，不要自行编造票价和开放时间。
"""
    llm_start = time.time()
    try:
        decision = (
            attraction_output_model.invoke(prompt)
        )
        _validate_attraction_decision(decision, excluded_names)
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
