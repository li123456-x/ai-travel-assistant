import json
import math
import time
from concurrent.futures import ThreadPoolExecutor

from config.model import model
from config.trip_policy import requires_hotel, walking_policy
from database.repository import record_execution_step_safe
from tools.intercity_transport import search_intercity_transport
from tools.transport import resolve_locations, search_transport_route_by_coords


class NoAvailableRouteError(ValueError):
    """The API succeeded and returned a well-formed, empty route list."""


def _looks_like_route_error(result) -> bool:
    return isinstance(result, str) and result.strip().startswith((
        "交通查询失败", "交通查询请求失败", "路线规划失败", "地点查询失败",
        "找不到地点", "路线查询失败",
    ))


def _validate_route_result(result, mode: str) -> dict:
    if _looks_like_route_error(result):
        raise ValueError(result)
    try:
        data = json.loads(result) if isinstance(result, str) else result
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError("高德路线结果不是有效 JSON") from exc
    if not isinstance(data, dict) or str(data.get("status")) != "1":
        raise ValueError("高德路线查询未成功返回")
    route = data.get("route")
    key = "transits" if mode == "transit" else "paths"
    paths = route.get(key) if isinstance(route, dict) else None
    if not isinstance(paths, list):
        raise ValueError(f"高德路线响应缺少有效 {key} 列表")
    if not paths:
        raise NoAvailableRouteError(f"高德没有可用的 {mode} 路线")
    if any(not isinstance(path, dict) or not path for path in paths):
        raise ValueError("高德路线方案格式无效")
    return data


def _validate_intercity_result(result) -> None:
    if not isinstance(result, str) or not result.strip():
        raise ValueError("城际交通查询结果为空")
    if result.strip().startswith(("暂未查询到", "城际交通查询失败")):
        raise ValueError(result)


def _build_route_tasks(hotel_name: str, attraction_names: list[str], *, needs_hotel=True):
    if not attraction_names or (needs_hotel and not hotel_name):
        return []
    tasks = []
    if needs_hotel:
        tasks.append((hotel_name, attraction_names[0], "酒店 → 第一个景点"))
    tasks.extend((start, end, "景点 → 景点") for start, end in zip(
        attraction_names, attraction_names[1:]
    ))
    if needs_hotel:
        tasks.append((attraction_names[-1], hotel_name, "最后一个景点 → 酒店"))
    return tasks


def _route_metrics(data: dict, mode: str) -> tuple[float | None, float | None]:
    path = data["route"]["transits" if mode == "transit" else "paths"][0]
    cost = path.get("cost")
    duration = cost.get("duration") if isinstance(cost, dict) else None
    if duration is None:
        duration = path.get("duration")
    values = []
    for raw in (path.get("distance"), duration):
        if raw is None:
            values.append(None)
            continue
        if isinstance(raw, bool):
            raise ValueError("高德路线距离或耗时无效")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("高德路线距离或耗时无效") from exc
        if not math.isfinite(value) or value < 0:
            raise ValueError("高德路线距离或耗时无效")
        values.append(value)
    return tuple(values)


def _query_city_segment(start, end, route_type, start_info, end_info, mode, policy, agent_run_id):
    def query(query_mode):
        started = time.time()
        try:
            raw = search_transport_route_by_coords.invoke({
                "origin_location": start_info["location"],
                "destination_location": end_info["location"],
                "origin_citycode": start_info["citycode"],
                "destination_citycode": end_info["citycode"],
                "mode": query_mode,
            })
            data = _validate_route_result(raw, query_mode)
        except Exception as exc:
            record_execution_step_safe(
                agent_run_id=agent_run_id, step_type="tool",
                step_name=f"amap_segment_{query_mode}", duration=time.time() - started,
                success=False, error_message=f"{start} → {end}：{exc}",
            )
            raise
        record_execution_step_safe(
            agent_run_id=agent_run_id, step_type="tool",
            step_name=f"amap_segment_{query_mode}", duration=time.time() - started,
            success=True,
        )
        return data

    actual_mode = mode
    fallback_reason = None
    try:
        data = query(mode)
    except NoAvailableRouteError:
        if mode != "transit":
            raise
        if not policy["allow_walking_fallback"]:
            raise ValueError("没有可用公共交通路线，且未允许本路段使用步行兜底")
        data = query("walking")
        actual_mode = "walking"
        fallback_reason = "高德公共交通查询成功但无方案，本路段另行查询并采用步行路线"
    distance, duration = _route_metrics(data, actual_mode)
    if fallback_reason:
        if distance is None or duration is None:
            raise ValueError("步行路线缺少真实距离或耗时，不能验证步行限制")
        if distance > policy["max_walking_distance_m"] or duration > policy["max_walking_duration_s"]:
            raise ValueError(
                f"步行兜底超出单段限制：实际 {distance:g} 米、{duration / 60:g} 分钟；"
                f"上限 {policy['max_walking_distance_m']:g} 米、"
                f"{policy['max_walking_duration_s'] / 60:g} 分钟，请调整景点或交通方式"
            )
    return {
        "origin": start, "destination": end, "route_type": route_type,
        "requested_mode": mode, "actual_mode": actual_mode,
        "distance_m": distance, "duration_s": duration,
        "fallback_reason": fallback_reason, "route_data": data,
    }


def transport_node(state):
    total_start = time.time()
    agent_run_id = state.get("agent_run_id")
    origin, destination = state["origin"], state["destination"]
    departure_date, return_date = state["departure_date"], state["return_date"]
    needs_hotel = requires_hotel(state)
    hotel_name = state.get("hotel_name", "") if needs_hotel else ""
    attractions = state.get("attraction_names", [])
    city_mode = state.get("city_transport_mode", "transit")
    intercity_mode = state.get("intercity_transport_mode", "train")
    if city_mode not in {"transit", "driving", "walking"}:
        raise ValueError("不支持的市内交通方式")
    if intercity_mode not in {"train", "flight"}:
        raise ValueError("不支持的城际交通方式")
    policy = walking_policy(state)
    if needs_hotel and (not isinstance(hotel_name, str) or not hotel_name.strip()):
        raise ValueError("交通规划缺少有效酒店名称，请先重新选择酒店")
    if not isinstance(attractions, list) or not attractions or any(
        not isinstance(name, str) or not name.strip() for name in attractions
    ):
        raise ValueError("交通规划缺少有效景点列表，请先重新选择景点")
    hotel_name = hotel_name.strip()
    attractions = [name.strip() for name in attractions]
    print(f"[Transport] hotel_name = {hotel_name or '一日游，不使用酒店'}")
    print(f"[Transport] attraction_names = {attractions}")

    started = time.time()
    try:
        with ThreadPoolExecutor(max_workers=2) as executor:
            outbound_future = executor.submit(search_intercity_transport.invoke, {
                "origin": origin, "destination": destination, "date": departure_date,
                "transport_mode": intercity_mode,
            })
            return_future = executor.submit(search_intercity_transport.invoke, {
                "origin": destination, "destination": origin, "date": return_date,
                "transport_mode": intercity_mode,
            })
            outbound_result, return_result = outbound_future.result(), return_future.result()
        _validate_intercity_result(outbound_result)
        _validate_intercity_result(return_result)
    except Exception as exc:
        record_execution_step_safe(
            agent_run_id=agent_run_id, step_type="tool", step_name="intercity_tavily",
            duration=time.time() - started, success=False, error_message=str(exc),
        )
        raise
    record_execution_step_safe(
        agent_run_id=agent_run_id, step_type="tool", step_name="intercity_tavily",
        duration=time.time() - started, success=True,
    )
    print(f"[Transport] 城际去程+返程查询耗时：{time.time() - started:.2f}s")

    tasks = _build_route_tasks(hotel_name, attractions, needs_hotel=needs_hotel)
    segments = []
    if tasks:
        places = list(dict.fromkeys(place for start, end, _ in tasks for place in (start, end)))
        started = time.time()
        try:
            locations, errors = resolve_locations(places=places, city=destination)
            errors = dict(errors)
            for place in places:
                info = locations.get(place)
                if not isinstance(info, dict) or not isinstance(info.get("location"), str) or not (
                    info["location"].strip() and str(info.get("citycode") or "").strip()
                ):
                    errors.setdefault(place, "地点缺少有效坐标或 citycode")
            if errors:
                raise ValueError("市内交通无法规划：" + "; ".join(
                    f"{place}：{reason}" for place, reason in errors.items()
                ))
        except Exception as exc:
            record_execution_step_safe(
                agent_run_id=agent_run_id, step_type="tool", step_name="amap_geocoding",
                duration=time.time() - started, success=False, error_message=str(exc),
            )
            raise
        record_execution_step_safe(
            agent_run_id=agent_run_id, step_type="tool", step_name="amap_geocoding",
            duration=time.time() - started, success=True,
        )
        print(f"[Transport] 高德地点解析耗时：{time.time() - started:.2f}s")

        started = time.time()
        route_errors = []
        with ThreadPoolExecutor(max_workers=2) as executor:
            futures = [(start, end, executor.submit(
                _query_city_segment, start, end, kind, locations[start], locations[end],
                city_mode, policy, agent_run_id,
            )) for start, end, kind in tasks]
            for start, end, future in futures:
                try:
                    segments.append(future.result())
                except Exception as exc:
                    route_errors.append(f"{start} → {end}：{exc}")
        fallback_segments = [segment for segment in segments if segment["fallback_reason"]]
        walking_distance = sum(segment["distance_m"] for segment in fallback_segments)
        walking_duration = sum(segment["duration_s"] for segment in fallback_segments)
        if walking_distance > policy["max_total_walking_distance_m"] or (
            walking_duration > policy["max_total_walking_duration_s"]
        ):
            route_errors.append(
                f"步行兜底累计 {walking_distance:g} 米、{walking_duration / 60:g} 分钟，"
                "超过整趟行程的步行兜底上限，请减少景点或调整交通方式"
            )
        record_execution_step_safe(
            agent_run_id=agent_run_id, step_type="tool", step_name="amap_city_route",
            duration=time.time() - started, success=not route_errors,
            error_message="; ".join(route_errors) if route_errors else None,
        )
        print(f"[Transport] 高德市内路线查询耗时：{time.time() - started:.2f}s")
        if route_errors:
            raise ValueError("市内路线查询未完成，不能生成可靠交通方案：" + "; ".join(route_errors))
    city_evidence = json.dumps(segments, ensure_ascii=False, indent=2) if segments else (
        "一日游仅一个景点，无景点之间的市内路段，因此不查询酒店或虚构市内路线。"
    )
    prompt = f"""
你是旅游交通规划 Agent，只能依据下面的真实查询证据生成方案。
出发地：{origin}；目的地：{destination}
出发日期：{departure_date}；返程日期：{return_date}
人数：{state['people']}；总预算：{state['budget']}；用户限制：{state['constraints']}
最新交通要求：{state.get('transport_revision_request', '')}
市内方式：{city_mode}（transit=公共交通优先，driving=自驾，walking=步行）
去程及返程方式：{intercity_mode}（train=铁路，flight=飞机）
住宿：{hotel_name if needs_hotel else '一日游，不安排酒店'}
景点顺序：{attractions}
去程城际交通参考：{outbound_result}
返程城际交通参考：{return_result}
高德逐段市内交通：{city_evidence}
输出去程、市内各段、返程的交通方案。
规则：
1. 每段按 actual_mode 说明实际方式；允许已另行查询且符合限制的步行兜底，不要求所有段统一方式。
2. 步行兜底须说明该段公交无方案，并使用真实 distance_m、duration_s；缺失值明确未知，不能估算成接口结果。
3. 不得编造车次、航班、实时票价、余票或未查询的出租车/公交方案。
4. 一日游不得添加酒店或酒店往返路线；首末到站接驳没有证据时明确尚未核查。
5. 参考路线不保证未来日期实时路况或运营状况；实际班次、价格、余票以官方渠道为准。
"""
    started = time.time()
    try:
        result = model.invoke(prompt)
        if not isinstance(result.content, str) or not result.content.strip():
            raise ValueError("交通模型返回了空说明")
    except Exception as exc:
        record_execution_step_safe(
            agent_run_id=agent_run_id, step_type="llm", step_name="transport_llm",
            duration=time.time() - started, success=False, error_message=str(exc),
        )
        raise
    record_execution_step_safe(
        agent_run_id=agent_run_id, step_type="llm", step_name="transport_llm",
        duration=time.time() - started, success=True,
    )
    print(f"[Transport] 总耗时：{time.time() - total_start:.2f}s")
    compact_segments = [{key: value for key, value in segment.items() if key != "route_data"}
                        for segment in segments]
    return {"transport_result": result.content, "transport_segments": compact_segments}
