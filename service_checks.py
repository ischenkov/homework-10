"""Quick TCP checks for MCP/ACP backends before running the supervisor REPL."""

from __future__ import annotations

import socket
from urllib.parse import urlparse

from config import Settings


def _host_port(url: str, default_port: int) -> tuple[str, int]:
    u = urlparse(url)
    host = u.hostname or "127.0.0.1"
    port = u.port if u.port is not None else default_port
    return host, port


def _tcp_open(host: str, port: int, timeout: float = 1.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def check_backend_services(settings: Settings) -> list[tuple[str, bool, str]]:
    """(label, ok, start_command) for each required service."""
    h1, p1 = _host_port(settings.search_mcp_url, settings.search_mcp_port)
    h2, p2 = _host_port(settings.report_mcp_url, settings.report_mcp_port)
    h3, p3 = _host_port(settings.acp_base_url, settings.acp_port)
    return [
        (
            f"SearchMCP {h1}:{p1}",
            _tcp_open(h1, p1),
            "python mcp_servers/search_mcp.py",
        ),
        (
            f"ReportMCP {h2}:{p2}",
            _tcp_open(h2, p2),
            "python mcp_servers/report_mcp.py",
        ),
        (
            f"ACP server {h3}:{p3}",
            _tcp_open(h3, p3),
            "python acp_server.py",
        ),
    ]


def format_startup_service_report(settings: Settings) -> str:
    lines = []
    all_ok = True
    for label, ok, cmd in check_backend_services(settings):
        status = "OK" if ok else "NOT RUNNING"
        lines.append(f"  [{status}] {label}")
        if not ok:
            all_ok = False
            lines.append(f"         → start: {cmd}")
    if not all_ok:
        lines.append("")
        lines.append("Run each missing command in a separate terminal (from the project root), then try again.")
    return "\n".join(lines)
