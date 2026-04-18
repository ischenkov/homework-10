"""Load LangChain tools from a remote SearchMCP (FastMCP streamable HTTP) — lesson 9 pattern."""

from __future__ import annotations

from datetime import timedelta

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient


def _normalize_mcp_url(url: str) -> str:
    u = url.rstrip("/")
    if not u.endswith("/mcp"):
        u = f"{u}/mcp"
    return u


async def mcp_tools_to_langchain(search_mcp_url: str) -> list[BaseTool]:
    """Connect to SearchMCP and return LangChain tools (web_search, read_url, knowledge_search)."""
    url = _normalize_mcp_url(search_mcp_url)
    client = MultiServerMCPClient(
        {
            "search": {
                "url": url,
                "transport": "http",
                "timeout": timedelta(minutes=10),
                "sse_read_timeout": timedelta(minutes=15),
            },
        }
    )
    return await client.get_tools(server_name="search")
