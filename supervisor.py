"""Supervisor: local LangChain agent with ACP delegation tools, ReportMCP save_report, HITL middleware."""

from __future__ import annotations

import sys

try:
    from fastmcp import Client as McpClient
except ModuleNotFoundError as e:
    print(
        "Error: missing package for MCP/ACP (e.g. acp-sdk, fastmcp).\n"
        "This homework needs Python 3.11 or newer.\n"
        "  python3.11 -m venv .venv\n"
        "  .venv/bin/pip install -r requirements.txt\n"
        f"Import detail: {e.name!r} not found.\n"
        "Your current interpreter may be 3.9 or 3.10 — check: python --version"
    )
    sys.exit(1)
from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from acp_http import acp_run_sync
from config import SUPERVISOR_SYSTEM_PROMPT, Settings
from text_sanitize import strip_surrogates

settings = Settings()

llm = ChatOpenAI(
    model=settings.model_name,
    temperature=0.5,
    api_key=settings.api_key.get_secret_value(),
)


def _format_acp_output(output) -> str:
    chunks: list[str] = []
    for msg in output:
        parts = getattr(msg, "parts", None) or []
        for p in parts:
            c = getattr(p, "content", None)
            if c:
                chunks.append(str(c))
    return "\n".join(chunks) if chunks else str(output)


def _connection_err(e: Exception) -> bool:
    err = str(e).lower()
    return any(
        x in err
        for x in ("connection", "connect", "refused", "all connection attempts failed")
    )


async def _run_acp_agent_async(agent_name: str, user_text: str) -> str:
    try:
        run = await acp_run_sync(settings.acp_base_url, agent_name, user_text)
        return _format_acp_output(run.output)
    except Exception as e:
        if _connection_err(e):
            raise RuntimeError(
                f"Cannot reach ACP at {settings.acp_base_url}. "
                "Start SearchMCP, ReportMCP, then: python acp_server.py"
            ) from e
        raise


@tool
async def delegate_to_planner(request: str) -> str:
    """Delegate to the remote planner agent (ACP). Returns structured plan JSON text."""
    return await _run_acp_agent_async("planner", request.strip())


@tool
async def delegate_to_researcher(instructions: str, round_number: int = 1) -> str:
    """Delegate to the remote researcher (ACP). Use round_number 1–3 (initial research plus up to two revisions)."""
    payload = f"ROUND:{int(round_number)}\n\n{instructions.strip()}"
    return await _run_acp_agent_async("researcher", payload)


@tool
async def delegate_to_critic(findings: str) -> str:
    """Delegate to the remote critic (ACP). Returns CritiqueResult JSON text."""
    return await _run_acp_agent_async("critic", findings.strip())


@tool
async def save_report(filename: str, content: str) -> str:
    """Save a Markdown report via ReportMCP (human approval is enforced by HumanInTheLoopMiddleware)."""
    try:
        url = settings.report_mcp_url
        if not url.rstrip("/").endswith("/mcp"):
            url = f"{url.rstrip('/')}/mcp"
        async with McpClient(url) as client:
            res = await client.call_tool(
                "save_report",
                {
                    "filename": strip_surrogates(filename),
                    "content": strip_surrogates(content),
                },
            )
            if getattr(res, "data", None) is not None:
                return str(res.data)
            if getattr(res, "content", None):
                return str(res.content)
            return str(res)
    except Exception as e:
        if _connection_err(e):
            raise RuntimeError(
                f"Cannot reach ReportMCP at {settings.report_mcp_url}. "
                "Start: python mcp_servers/report_mcp.py"
            ) from e
        raise


supervisor = create_agent(
    llm,
    tools=[delegate_to_planner, delegate_to_researcher, delegate_to_critic, save_report],
    system_prompt=SUPERVISOR_SYSTEM_PROMPT,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"save_report": True},
            description_prefix="ACTION REQUIRES APPROVAL (save_report)",
        ),
    ],
    checkpointer=InMemorySaver(),
    name="supervisor",
)
