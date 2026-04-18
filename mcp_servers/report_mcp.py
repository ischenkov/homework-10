"""ReportMCP: save_report + output-dir resource."""

from __future__ import annotations

import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from fastmcp import FastMCP

from config import Settings
from tool_impl import list_saved_reports, write_report_to_disk

settings = Settings()
mcp = FastMCP(name="ReportMCP")


@mcp.tool
def save_report(filename: str, content: str) -> str:
    """Save a Markdown report under the configured output directory (no HITL; approval is on the Supervisor)."""
    return write_report_to_disk(filename, content, settings)


@mcp.resource("resource://output-dir")
def output_dir_resource() -> str:
    """Resolved output directory path and list of saved .md reports."""
    path, files = list_saved_reports(settings)
    return json.dumps({"output_dir": path, "reports": files}, indent=2)


if __name__ == "__main__":
    mcp.run(transport="http", host="127.0.0.1", port=settings.report_mcp_port)
