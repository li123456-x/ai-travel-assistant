import json
import os

import requests
from dotenv import load_dotenv
from langchain.tools import tool


load_dotenv()

AMAP_KEY = os.getenv("AMAP_KEY")


@tool
def search_hotels(city: str, keyword: str = "酒店") -> str:
    """
    查询指定城市中的酒店 POI 信息。
    """
    if not AMAP_KEY:
        return "酒店查询失败：未配置 AMAP_KEY"

    url = "https://restapi.amap.com/v5/place/text"
    params = {
        "key": AMAP_KEY,
        "keywords": keyword,
        "region": city,
        "city_limit": "true",
        "show_fields": "business",
        "page_size": 10,
        "page_num": 1,
    }

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
    except requests.RequestException as e:
        return f"酒店查询请求失败：{e}"

    if data.get("status") != "1":
        return f"酒店查询失败：{data.get('info', '未知错误')}"

    pois = data.get("pois", [])
    if not pois:
        return "[]"

    results = []
    for poi in pois:
        business = poi.get("business", {}) or {}
        results.append(
            {
                "name": poi.get("name"),
                "address": poi.get("address"),
                "district": poi.get("adname"),
                "location": poi.get("location"),
                "rating": business.get("rating"),
                "cost": business.get("cost"),
                "tel": business.get("tel"),
            }
        )

    return json.dumps(results, ensure_ascii=False, indent=2)
