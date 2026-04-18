"""SearchMCP: web_search, read_url, knowledge_search + knowledge-base-stats resource."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastmcp import FastMCP

from config import Settings
from tool_impl import (
    impl_knowledge_search,
    impl_read_url,
    impl_web_search,
    knowledge_base_stats,
)

settings = Settings()
mcp = FastMCP(name="SearchMCP")


@mcp.tool
def web_search(query: str) -> str:
    """Search the internet for information. Returns results with title, url, and snippet."""
    return impl_web_search(query, settings)


@mcp.tool
def read_url(url: str) -> str:
    """Fetch and extract the main text content from a webpage."""
    return impl_read_url(url, settings)


@mcp.tool
def knowledge_search(query: str) -> str:
    """Search the local knowledge base (index built via ingest.py)."""
    return impl_knowledge_search(query, settings)


@mcp.resource("resource://knowledge-base-stats")
def knowledge_base_stats_resource() -> str:
    """Chunk count in index and last index update time."""
    n, updated = knowledge_base_stats(settings)
    return json.dumps(
        {"chunk_count": n, "last_index_update": updated},
        indent=2,
    )


if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=settings.search_mcp_port)
