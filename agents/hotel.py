import time
import json
import unicodedata
from pydantic import BaseModel, Field
from config.model import model
from config.state import TravelState
from database.repository import record_execution_step_safe
from tools.hotel import search_hotels

class HotelDecision(BaseModel):
    hotel_name: str = Field(
        min_length=1,
        description="最终最推荐的一家酒店正式名称"
    )
    summary: str = Field(
        min_length=1,
        description="完整酒店推荐分析"
    )

hotel_output_model = model.with_structured_output(
    HotelDecision
)


def _normalize_hotel_name(name: str) -> str:
    return "".join(unicodedata.normalize("NFKC", name).split()).casefold()


def _filter_hotel_candidates(
    tool_result: str, keyword: str, excluded_names: list[str]
) -> list[dict]:
    try:
        candidates = json.loads(tool_result)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("酒店工具未返回有效候选数据，不能继续推荐") from exc
    if not isinstance(candidates, list):
        raise ValueError("酒店候选数据必须是列表")

    excluded = {
        _normalize_hotel_name(name)
        for name in excluded_names
        if isinstance(name, str) and name.strip()
    }
    keyword_key = _normalize_hotel_name(keyword)
    generic_keyword = keyword_key in {"酒店", "hotel", "hotels"}
    valid_candidates = []
    seen = set()
    for item in candidates:
        if not isinstance(item, dict):
            continue
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        name = name.strip()
        name_key = _normalize_hotel_name(name)
        if name_key in excluded or name_key in seen:
            continue
        if not generic_keyword and keyword_key not in name_key:
            continue
        valid_candidates.append({**item, "name": name})
        seen.add(name_key)

    if not valid_candidates:
        raise ValueError(
            "没有符合酒店关键词且未被排除的真实候选，请调整关键词或筛选条件"
        )
    return valid_candidates


def _validate_hotel_decision(decision, candidates: list[dict]) -> None:
    if not isinstance(decision.hotel_name, str) or not decision.hotel_name.strip():
        raise ValueError("酒店名称不能为空")
    if not isinstance(decision.summary, str) or not decision.summary.strip():
        raise ValueError("酒店推荐说明不能为空")
    candidate_names = {
        _normalize_hotel_name(item["name"]): item["name"]
        for item in candidates
    }
    selected = candidate_names.get(_normalize_hotel_name(decision.hotel_name))
    if selected is None:
        raise ValueError("模型选择的酒店不在真实查询候选中")
    # 将状态中的名称恢复为工具返回的正式名称，避免后续地理编码使用改写名称。
    decision.hotel_name = selected
    decision.summary = decision.summary.strip()


def hotel_node(state: TravelState):
    total_start = time.time()
    agent_run_id = state.get("agent_run_id")
    keyword = state.get("hotel_search_keyword", "酒店").strip() or "酒店"
    excluded_names = state.get("hotel_excluded_names", [])
    tool_start = time.time()
    try:
        tool_result = search_hotels.invoke(
            {
                "city": state["destination"],
                "keyword": keyword,
            }
        )
        candidates = _filter_hotel_candidates(tool_result, keyword, excluded_names)
        tool_result = json.dumps(candidates, ensure_ascii=False, indent=2)
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
    ================ 最新住宿修改要求 ================
    {state.get("hotel_revision_request", "")}
    查询关键词：{keyword}
    不再选择的酒店：{excluded_names}
    ================ 景点信息 ================
    {state["attractions_result"]}
    核心景点：
    {state.get("attraction_names", [])}
    ================ 高德酒店 POI 查询结果 ================
    {tool_result}
    ================ 任务 ================
    请从真实工具结果中选择最多 2-4 个合理住宿方案；候选不足时按实际数量推荐。
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
    6. 最推荐的一家酒店必须使用候选中的完整正式名称，不得缩写或改写
    7. 遵守最新住宿修改要求，所有推荐酒店都必须来自当前候选
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
        _validate_hotel_decision(decision, candidates)
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
