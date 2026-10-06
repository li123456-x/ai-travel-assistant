import time
from concurrent.futures import ThreadPoolExecutor
from config.model import model
from database.repository import record_execution_step_safe
from tools.intercity_transport import search_intercity_transport
from tools.transport import resolve_locations,search_transport_route_by_coords


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
                },
            )
            return_future = executor.submit(
                search_intercity_transport.invoke,
                {
                    "origin": destination,
                    "destination": origin,
                    "date": return_date,
                },
            )
            outbound_result = (
                outbound_future.result()
            )
            return_result = (
                return_future.result()
            )
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
    locations, geocoding_errors = (
        resolve_locations(
            places=places,
            city=destination,
        )
    )
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
                        "mode": "transit",
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
                    if _looks_like_route_error(
                        route_result
                    ):
                        city_route_success = (
                            False
                        )
                        city_route_errors.append(
                            f"{start_place} → "
                            f"{end_place}："
                            f"{route_result}"
                        )
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
    - 是否适合步行/公交/地铁/出租车
    
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
    
    最后注明：
    实际班次、票价、余票及临时调整以官方渠道为准。
"""

    llm_start = time.time()
    try:
        result = model.invoke(
            prompt
        )
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
