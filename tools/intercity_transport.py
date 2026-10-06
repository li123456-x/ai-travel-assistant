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
) -> str:
    """
    查询两个城市之间的高铁、动车、火车等城际交通信息。
    """
    query = (
        f"{date} {origin} 到 {destination} "
        "高铁 动车 火车 车次 出发时间 到达时间 票价"
    )

    result = _get_tavily_client().search(
        query=query,
        max_results=6,
    )

    results = result.get("results", [])
    if not results:
        return f"暂未查询到 {date} {origin} 到 {destination} 的城际交通信息。"

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
