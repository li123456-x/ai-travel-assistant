import os

from dotenv import load_dotenv
from langchain_core.tools import tool
from tavily import TavilyClient


load_dotenv()


def _get_tavily_client():
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("城际交通查询失败：未配置 TAVILY_API_KEY")
    return TavilyClient(api_key=api_key)


@tool
def search_intercity_transport(
    origin: str,
    destination: str,
    date: str,
    transport_mode: str = "train",
) -> str:
    """
    按 train（铁路）或 flight（飞机）检索城际交通参考信息，不提供实时库存或订票。
    """
    keywords = {
        "train": "高铁 动车 火车 车次 出发时间 到达时间 参考票价",
        "flight": "飞机 航班 机场 起飞时间 到达时间 参考票价",
    }
    if transport_mode not in keywords:
        raise ValueError("不支持的城际交通方式，只支持 train 或 flight")

    query = (
        f"{date} {origin} 到 {destination} "
        f"{keywords[transport_mode]}"
    )

    result = _get_tavily_client().search(
        query=query,
        max_results=6,
    )

    results = result.get("results", [])
    if not isinstance(results, list):
        raise ValueError("城际交通查询返回格式错误")
    results = [
        item for item in results
        if isinstance(item, dict)
        and (str(item.get("title") or "").strip()
             or str(item.get("content") or "").strip())
    ]
    if not results:
        raise ValueError(
            f"暂未查询到 {date} {origin} 到 {destination} 的"
            f"{transport_mode} 交通参考信息，请调整日期或交通方式。"
        )

    text_list = []
    for index, item in enumerate(results, start=1):
        title = item.get("title", "")
        content = item.get("content", "")
        url = item.get("url", "")
        text_list.append(
            f"""
查询结果{index}
标题：{title}
内容：{content}
来源：{url}
"""
        )

    return "\n".join(text_list)
