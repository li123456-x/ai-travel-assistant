import json
import time
from concurrent.futures import ThreadPoolExecutor
from config.model import model
from database.repository import record_execution_step_safe
from tools.intercity_transport import search_intercity_transport
from tools.transport import resolve_locations, search_transport_route_by_coords


def _validate_route_result(result, mode: str) -> None:
    """工具可能返回错误字符串或成功但无路线的数据；两者都不能交给模型补全。"""
    if _looks_like_route_error(result):
        raise ValueError(result)
    try:
        data = json.loads(result) if isinstance(result, str) else result
    except (TypeError, json.JSONDecodeError) as e:
        raise ValueError("高德路线结果不是有效 JSON") from e
    if not isinstance(data, dict) or str(data.get("status")) != "1":
        raise ValueError("高德路线查询未成功返回")
    route = data.get("route")
    route_key = "transits" if mode == "transit" else "paths"
    paths = route.get(route_key) if isinstance(route, dict) else None
    if (
        not isinstance(paths, list)
        or not paths
        or any(not isinstance(path, dict) or not path for path in paths)
    ):
        raise ValueError(
            f"高德没有可用的 {mode} 路线，请调整景点或交通方式"
        )


def _validate_intercity_result(result) -> None:
    if not isinstance(result, str) or not result.strip():
        raise ValueError("城际交通查询结果为空")
    if result.strip().startswith(("暂未查询到", "城际交通查询失败")):
        raise ValueError(result)


def _looks_like_route_error(
    result,
) -> bool:
    """识别 Tool 以字符串返回的失败结果。"""
    if not isinstance(result, str):
        return False
    error_markers = (
        "交通查询失败",
        "交通查询请求失败",
        "路线规划失败",
        "地点查询失败",
        "找不到地点",
        "路线查询失败",
    )
    return any(
        marker in result
        for marker in error_markers
    )


def _build_route_tasks(
    hotel_name: str,
    attraction_names: list[str],
) -> list[
    tuple[str, str, str]
]:
    """
    构造：
    酒店→首个景点
    景点→景点
    最后景点→酒店
    """
    if (
        not hotel_name
        or not attraction_names
    ):
        return []
    tasks = [
        (
            hotel_name,
            attraction_names[0],
            "酒店 → 第一个景点",
        )
    ]
    for i in range(
        len(attraction_names) - 1
    ):
        tasks.append(
            (
                attraction_names[i],
                attraction_names[i + 1],
                "景点 → 景点",
            )
        )
    tasks.append(
        (
            attraction_names[-1],
            hotel_name,
            "最后一个景点 → 酒店",
        )
    )
    return tasks

def transport_node(state):
    total_start = time.time()
    agent_run_id = state.get(
        "agent_run_id"
    )
    origin = state["origin"]
    destination = state["destination"]
    departure_date = state[
        "departure_date"
    ]
    return_date = state[
        "return_date"
    ]
    hotel_name = state.get(
        "hotel_name",
        "",
    )
    attraction_names = state.get(
        "attraction_names",
        [],
    )
    city_mode = state.get("city_transport_mode", "transit")
    intercity_mode = state.get("intercity_transport_mode", "train")
    if city_mode not in {"transit", "driving", "walking"}:
        raise ValueError("不支持的市内交通方式")
    if intercity_mode not in {"train", "flight"}:
        raise ValueError("不支持的城际交通方式")
    if not isinstance(hotel_name, str) or not hotel_name.strip():
        raise ValueError("交通规划缺少有效酒店名称，请先重新选择酒店")
    if (
        not isinstance(attraction_names, list)
        or not attraction_names
        or any(not isinstance(name, str) or not name.strip()
               for name in attraction_names)
    ):
        raise ValueError("交通规划缺少有效景点列表，请先重新选择景点")
    hotel_name = hotel_name.strip()
    attraction_names = [name.strip() for name in attraction_names]
    print(
        "[Transport] hotel_name =",
        hotel_name,
    )
    print(
        "[Transport] attraction_names =",
        attraction_names,
    )

    intercity_start = time.time()
    try:
        with ThreadPoolExecutor(
            max_workers=2
        ) as executor:
            outbound_future = executor.submit(
                search_intercity_transport.invoke,
                {
                    "origin": origin,
                    "destination": destination,
                    "date": departure_date,
                    "transport_mode": intercity_mode,
                },
            )
            return_future = executor.submit(
                search_intercity_transport.invoke,
                {
                    "origin": destination,
                    "destination": origin,
                    "date": return_date,
                    "transport_mode": intercity_mode,
                },
            )
            outbound_result = (
                outbound_future.result()
            )
            return_result = (
                return_future.result()
            )
        _validate_intercity_result(outbound_result)
        _validate_intercity_result(return_result)
        intercity_duration = (
            time.time()- intercity_start
        )
        record_execution_step_safe(
            agent_run_id=agent_run_id,
            step_type="tool",
            step_name="intercity_tavily",
            duration=intercity_duration,
            success=True,
        )
    except Exception as e:
        intercity_duration = (
            time.time()- intercity_start
        )
        record_execution_step_safe(
            agent_run_id=agent_run_id,
            step_type="tool",
            step_name="intercity_tavily",
            duration=intercity_duration,
            success=False,
            error_message=str(e),
        )
        raise
    print(
        "[Transport] 城际去程+返程查询耗时："
        f"{intercity_duration:.2f}s"
    )

    route_tasks = _build_route_tasks(
        hotel_name=hotel_name,
        attraction_names=attraction_names,
    )
    city_route_results = []
    geocoding_start = time.time()
    places = []
    for (
        start_place,
        end_place,
        _
    ) in route_tasks:
        places.extend(
            [
                start_place,
                end_place,
            ]
        )
    try:
        locations, geocoding_errors = resolve_locations(
            places=places,
            city=destination,
        )
        # 即使工具没有明确报告错误，也要保证所有端点都具备查询所需数据。
        geocoding_errors = dict(geocoding_errors)
        for place in dict.fromkeys(places):
            info = locations.get(place)
            if (
                not isinstance(info, dict)
                or not isinstance(info.get("location"), str)
                or not info["location"].strip()
                or not str(info.get("citycode") or "").strip()
            ):
                geocoding_errors.setdefault(place, "地点缺少有效坐标或 citycode")
    except Exception as e:
        record_execution_step_safe(
            agent_run_id=agent_run_id,
            step_type="tool",
            step_name="amap_geocoding",
            duration=time.time() - geocoding_start,
            success=False,
            error_message=str(e),
        )
        raise
    geocoding_duration = (
        time.time()
        - geocoding_start
    )
    record_execution_step_safe(
        agent_run_id=agent_run_id,
        step_type="tool",
        step_name="amap_geocoding",
        duration=geocoding_duration,
        success=not geocoding_errors,
        error_message=(
            "\n".join(
                f"{place}：{error}"
                for place, error
                in geocoding_errors.items()
            )
            if geocoding_errors
            else None
        ),
    )
    print(
        "[Transport] 高德地点解析耗时："
        f"{geocoding_duration:.2f}s"
    )
    if geocoding_errors:
        raise ValueError(
            "市内交通无法规划："
            + "; ".join(f"{place}：{error}" for place, error in geocoding_errors.items())
        )
    city_route_start = time.time()
    city_route_success = True
    city_route_errors = []
    valid_tasks = []
    for (
        start_place,
        end_place,
        route_type,
    ) in route_tasks:
        start_info = locations.get(
            start_place
        )
        end_info = locations.get(
            end_place
        )
        if (
            not start_info
            or not end_info
        ):
            city_route_success = False
            error_text = (
                f"{start_place} → "
                f"{end_place}："
                "起点或终点坐标解析失败"
            )
            city_route_errors.append(
                error_text
            )
            city_route_results.append(
                f"""
【{route_type}】
{start_place} → {end_place}
高德查询结果：
{error_text}
"""
            )
            continue
        valid_tasks.append(
            (
                start_place,
                end_place,
                route_type,
                start_info,
                end_info,
            )
        )
    if valid_tasks:
        with ThreadPoolExecutor(
            max_workers=2
        ) as executor:
            future_map = {}
            for (
                start_place,
                end_place,
                route_type,
                start_info,
                end_info,
            ) in valid_tasks:
                future = executor.submit(
                    search_transport_route_by_coords.invoke,
                    {
                        "origin_location":
                            start_info["location"],
                        "destination_location":
                            end_info["location"],
                        "origin_citycode":
                            start_info["citycode"],
                        "destination_citycode":
                            end_info["citycode"],
                        "mode": city_mode,
                    },
                )
                future_map[future] = (
                    start_place,
                    end_place,
                    route_type,
                )
            for (
                future,
                info,
            ) in future_map.items():
                (
                    start_place,
                    end_place,
                    route_type,
                ) = info
                try:
                    route_result = (
                        future.result()
                    )
                    _validate_route_result(route_result, city_mode)
                except Exception as e:
                    city_route_success = False
                    error_text = (
                        f"{start_place} → "
                        f"{end_place}：{e}"
                    )
                    city_route_errors.append(
                        error_text
                    )
                    route_result = (
                        f"路线查询失败：{e}"
                    )
                city_route_results.append(
                    f"""
【{route_type}】
{start_place} → {end_place}
高德查询结果：
{route_result}
"""
                )

    elif not route_tasks:
        city_route_results.append(
            "未获取到有效 hotel_name "
            "或 attraction_names，"
            "因此未执行高德市内路线查询。"
        )
    city_route_duration = (
        time.time()- city_route_start
    )
    record_execution_step_safe(
        agent_run_id=agent_run_id,
        step_type="tool",
        step_name="amap_city_route",
        duration=city_route_duration,
        success=(
            city_route_success
            and bool(route_tasks)
        ),
        error_message=(
            "\n".join(
                city_route_errors
            )
            if city_route_errors
            else None
        ),
    )
    print(
        "[Transport] 高德市内路线查询耗时："
        f"{city_route_duration:.2f}s"
    )
    if not city_route_success or not route_tasks:
        raise ValueError(
            "市内路线查询未完成，不能生成可靠交通方案："
            + "; ".join(city_route_errors)
        )

    prompt = f"""
    你是一名专业的旅游交通规划 Agent。
    请根据：
    1. Tavily 城际交通检索结果
    2. 高德地图真实市内路线结果
    3. 用户需求
    制定一份完整、简洁、可执行的交通方案。
    ================ 用户信息 ================
    出发地：{origin}
    目的地：{destination}
    出发日期：{departure_date}
    返程日期：{return_date}
    人数：{state["people"]}
    总预算：{state["budget"]}
    用户限制：{state["constraints"]}
    ================ 最新交通要求 ================
    {state.get("transport_revision_request", "")}
    市内方式：{city_mode}（transit=公共交通，driving=自驾，walking=步行）
    去程及返程方式：{intercity_mode}（train=铁路，flight=飞机）
    ================ 酒店 ================
    {hotel_name}
    ================ 景点 ================
    {attraction_names}
    ================ 去程城际交通 ================
    {outbound_result}
    ================ 返程城际交通 ================
    {return_result}
    ================ 高德市内交通 ================
    {"".join(city_route_results)}
    ================ 任务 ================
    一、去程城际交通
    - 推荐交通方式
    - 推荐出发/到达时间
    - 参考车次或航班（仅限检索结果中存在的信息）
    - 参考票价与耗时
    - 推荐理由
    二、{destination}市内交通
    必须依据高德查询结果，按“起点 → 终点”说明：
    - 推荐交通方式
    - 预计耗时
    - 距离或换乘信息
    - 按当前查询的交通方式解释，不得自行换成未查询的方式
    
    三、返程城际交通
    - 推荐返程时间和交通方式
    - 参考车次或航班（仅限检索结果中存在的信息）
    - 参考票价与预计耗时
    - 推荐理由
    ================ 规则 ================
    1. 市内路线不得脱离高德结果自行编造。
    2. 城际交通不得编造 Tavily 结果中不存在的具体车次或航班。
    3. 无法确认具体班次时，只推荐合理时间段。
    4. 不得虚构实时票价和余票。
    5. 考虑用户限制，尽量减少明显往返和过度疲劳。
    6. 不要重复介绍景点和酒店，输出尽量简洁。
    7. 必须遵守当前指定的市内及城际交通方式；证据不足时说明限制，不得用旧方案或其他方式替代。
    8. Tavily 是参考信息检索，不是实时票务接口；不要声称已验证库存或预订成功。
    
    最后注明：
    实际班次、票价、余票及临时调整以官方渠道为准。
"""

    llm_start = time.time()
    try:
        result = model.invoke(
            prompt
        )
        if not isinstance(result.content, str) or not result.content.strip():
            raise ValueError("交通模型返回了空说明")
        llm_duration = (
            time.time()- llm_start
        )
        record_execution_step_safe(
            agent_run_id=agent_run_id,
            step_type="llm",
            step_name="transport_llm",
            duration=llm_duration,
            success=True,
        )
    except Exception as e:
        llm_duration = (
            time.time()- llm_start
        )
        record_execution_step_safe(
            agent_run_id=agent_run_id,
            step_type="llm",
            step_name="transport_llm",
            duration=llm_duration,
            success=False,
            error_message=str(e),
        )
        raise
    print(
        "[Transport] LLM耗时："
        f"{llm_duration:.2f}s"
    )
    print(
        "[Transport] 总耗时："
        f"{time.time() - total_start:.2f}s"
    )
    # print(
    #     "\n========== Transport Agent =========="
    # )
    #
    # print(
    #     result.content
    # )
    return {
        "transport_result":result.content
    }
