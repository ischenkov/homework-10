"""Pytest fixtures: project path, live-backend gate, eval judge model."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# Settings() is imported by many modules and requires OPENAI_API_KEY at import time.
if not os.getenv("OPENAI_API_KEY", "").strip():
    os.environ["OPENAI_API_KEY"] = "sk-placeholder-for-pytest-collection-only"


def pytest_configure(config):
    os.environ.setdefault("DEEPEVAL_EVAL_MODEL", os.getenv("DEEPEVAL_EVAL_MODEL", "gpt-4o-mini"))


@pytest.fixture(scope="session")
def project_root() -> Path:
    return _ROOT


@pytest.fixture(scope="session")
def live_backends_ok() -> bool:
    """True when SearchMCP, ReportMCP, and ACP ports respond (TCP)."""
    try:
        from config import Settings
        from service_checks import check_backend_services

        return all(ok for _label, ok, _cmd in check_backend_services(Settings()))
    except Exception:
        return False


def _real_openai_key() -> bool:
    k = os.getenv("OPENAI_API_KEY", "").strip()
    return bool(k) and "placeholder" not in k.lower()


@pytest.fixture
def require_live_backends(live_backends_ok):
    if not _real_openai_key():
        pytest.skip(
            "Set a real OPENAI_API_KEY for live MCP/agent tests and DeepEval judge calls"
        )
    if not live_backends_ok:
        pytest.skip(
            "MCP/ACP backends not reachable. Start: python mcp_servers/search_mcp.py, "
            "python mcp_servers/report_mcp.py, python acp_server.py"
        )
