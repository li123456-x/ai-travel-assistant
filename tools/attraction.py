import json
import os

from dotenv import load_dotenv
from langchain.tools import tool
from tavily import TavilyClient


load_dotenv()


def _get_tavily_client():
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("景点查询失败：未配置 TAVILY_API_KEY")
    return TavilyClient(api_key=api_key)


@tool
def search_attractions(query: str) -> str:
    """
    搜索旅游景点相关的实时信息。
    """
    result = _get_tavily_client().search(
        query=query,
        max_results=5,
    )

    return json.dumps(
        result.get("results", []),
        ensure_ascii=False,
        indent=2,
    )
