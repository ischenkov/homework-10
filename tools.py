from __future__ import annotations

from langchain_core.tools import tool

from config import Settings
from tool_impl import (
    impl_knowledge_search,
    impl_read_url,
    impl_web_search,
    write_report_to_disk,
)

settings = Settings()


@tool
def web_search(query: str) -> str:
    """Search the internet for information. Returns results with title, url, and snippet.
    Use this to find relevant sources before reading full pages."""
    return impl_web_search(query, settings)


@tool
def read_url(url: str) -> str:
    """Fetch and extract the main text content from a webpage. Use when you need full content
    from a URL found via web_search. Returns truncated text if the page is very long."""
    return impl_read_url(url, settings)


@tool
def knowledge_search(query: str) -> str:
    """Search the local knowledge base (built from data/ via ingest.py). REQUIRED first step for any research or factual question: call this before web_search with a query that matches the user's topic. Not reading data/ directly — only the embedded index in index/."""
    return impl_knowledge_search(query, settings)


@tool
def save_report(filename: str, content: str) -> str:
    """Save a Markdown report under the project output folder (no HITL). Supervisor uses ReportMCP + middleware instead."""
    return write_report_to_disk(filename, content, settings)


@tool
def write_report(filename: str, content: str) -> str:
    """Save a Markdown report without HITL (legacy / scripts)."""
    return write_report_to_disk(filename, content, settings)
