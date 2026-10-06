import json
import os
import threading
import time
from functools import lru_cache

import requests
from dotenv import load_dotenv
from langchain.tools import tool


load_dotenv()

AMAP_KEY = os.getenv("AMAP_KEY")

_LOCATION_INTERVAL = 0.5
_location_lock = threading.Lock()
_last_location_request = 0.0

_ROUTE_INTERVAL = 0.5
_route_lock = threading.Lock()
_last_route_request = 0.0


def _wait_for_location_request():
    global _last_location_request

    with _location_lock:
        now = time.time()
        wait_time = _LOCATION_INTERVAL - (now - _last_location_request)
        if wait_time > 0:
            time.sleep(wait_time)
        _last_location_request = time.time()


def _wait_for_route_request():
    global _last_route_request

    with _route_lock:
        now = time.time()
        wait_time = _ROUTE_INTERVAL - (now - _last_route_request)
        if wait_time > 0:
            time.sleep(wait_time)
        _last_route_request = time.time()


@lru_cache(maxsize=256)
def get_location_info(place: str, city: str) -> dict:
    if not AMAP_KEY:
        raise ValueError("地点查询失败：未配置 AMAP_KEY")

    _wait_for_location_request()

    url = "https://restapi.amap.com/v5/place/text"
    params = {
        "key": AMAP_KEY,
        "keywords": place,
        "region": city,
        "city_limit": "true",
        "page_size": 1,
        "page_num": 1,
    }

    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    if data.get("status") != "1":
        raise ValueError(f"地点查询失败：{data.get('info', '未知错误')}")

    pois = data.get("pois", [])
    if not pois:
        raise ValueError(f"找不到地点：{place}")

    poi = pois[0]
    location = poi.get("location")
    citycode = str(poi.get("citycode", "")).strip()

    if not location:
        raise ValueError(f"地点查询失败：{place} 缺少坐标")

    if not citycode:
        raise ValueError(f"地点查询失败：{place} 缺少 citycode")

    return {
        "location": location,
        "citycode": citycode,
    }


def get_location(place: str, city: str) -> str:
    return get_location_info(place, city)["location"]


def resolve_locations(
    places: list[str],
    city: str,
) -> tuple[dict[str, dict], dict[str, str]]:
    locations = {}
    errors = {}

    unique_places = list(
        dict.fromkeys(
            place.strip()
            for place in places
            if place and place.strip()
        )
    )

    for place in unique_places:
        try:
            locations[place] = get_location_info(place, city)
        except Exception as e:
            errors[place] = str(e)

    return locations, errors


def _request_route(
    origin_location: str,
    destination_location: str,
    origin_citycode: str,
    destination_citycode: str,
    mode: str = "transit",
) -> str:
    if not AMAP_KEY:
        return "交通查询失败：未配置 AMAP_KEY"

    if mode == "driving":
        url = "https://restapi.amap.com/v5/direction/driving"
        params = {
            "key": AMAP_KEY,
            "origin": origin_location,
            "destination": destination_location,
        }
    elif mode == "walking":
        url = "https://restapi.amap.com/v5/direction/walking"
        params = {
            "key": AMAP_KEY,
            "origin": origin_location,
            "destination": destination_location,
        }
    else:
        url = "https://restapi.amap.com/v5/direction/transit/integrated"
        params = {
            "key": AMAP_KEY,
            "origin": origin_location,
            "destination": destination_location,
            "city1": origin_citycode,
            "city2": destination_citycode,
            "strategy": 0,
        }

    _wait_for_route_request()

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        return f"交通查询请求失败：{e}"

    if data.get("status") != "1":
        return f"路线规划失败：{data.get('info', '未知错误')}"

    return json.dumps(data, ensure_ascii=False, indent=2)


@tool
def search_transport_route_by_coords(
    origin_location: str,
    destination_location: str,
    origin_citycode: str,
    destination_citycode: str,
    mode: str = "transit",
) -> str:
    """
    基于已经解析好的坐标和 citycode 查询路线。
    """
    return _request_route(
        origin_location=origin_location,
        destination_location=destination_location,
        origin_citycode=origin_citycode,
        destination_citycode=destination_citycode,
        mode=mode,
    )


@tool
def search_transport_route(
    origin: str,
    destination: str,
    city: str,
    mode: str = "transit",
) -> str:
    """
    兼容旧调用方式：输入地点名称后直接查询路线。
    """
    try:
        origin_info = get_location_info(origin, city)
        destination_info = get_location_info(destination, city)
    except Exception as e:
        return str(e)

    return _request_route(
        origin_location=origin_info["location"],
        destination_location=destination_info["location"],
        origin_citycode=origin_info["citycode"],
        destination_citycode=destination_info["citycode"],
        mode=mode,
    )
