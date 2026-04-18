
from __future__ import annotations

import json
import sys
from collections.abc import AsyncGenerator
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from acp_sdk.models import Message, MessagePart
from acp_sdk.server import Context, RunYield, RunYieldResume, Server

from agent_factory import create_agent
from agents._llm import chat_model
from agents.critic import critic_task_message
from agents.planner import planner_task_message
from agents.research import (
    MAX_RESEARCH_ROUNDS,
    extract_findings_text,
    parse_research_round_payload,
    research_task_message,
)
from config import CRITIC_SYSTEM_PROMPT, PLANNER_SYSTEM_PROMPT, RESEARCH_SYSTEM_PROMPT, Settings
from mcp_utils import mcp_tools_to_langchain
from schemas import CritiqueResult, ResearchPlan
from text_sanitize import strip_surrogates, strip_surrogates_deep

settings = Settings()
server = Server()


def _acp_messages_to_text(messages: list[Message]) -> str:
    parts: list[str] = []
    for m in messages:
        for p in m.parts:
            if p.content:
                parts.append(str(p.content))
    return strip_surrogates("\n".join(parts))


@server.agent()
async def planner(
    input: list[Message], context: Context
) -> AsyncGenerator[RunYield, RunYieldResume]:
    """Structured research plan (ResearchPlan) using SearchMCP tools."""
    text = _acp_messages_to_text(input)
    tools = await mcp_tools_to_langchain(settings.search_mcp_url)
    graph = create_agent(
        chat_model(),
        tools,
        system_prompt=PLANNER_SYSTEM_PROMPT,
        response_format=ResearchPlan,
        checkpointer=None,
        name="planner",
    )
    # MCP tools from langchain-mcp-adapters are async-only; must use ainvoke (not invoke in executor).
    result = await graph.ainvoke(
        {"messages": [planner_task_message(text)]},
        config={"recursion_limit": settings.subagent_recursion_limit},
    )
    structured = result.get("structured_response")
    if isinstance(structured, ResearchPlan):
        out = json.dumps(
            strip_surrogates_deep(structured.model_dump(mode="json")),
            indent=2,
            ensure_ascii=False,
        )
    elif structured is not None:
        out = strip_surrogates(str(structured))
    else:
        out = "Error: planner did not return a structured plan."
    yield Message(parts=[MessagePart(content=strip_surrogates(out), content_type="text/plain")])


@server.agent()
async def researcher(
    input: list[Message], context: Context
) -> AsyncGenerator[RunYield, RunYieldResume]:
    """Research findings via SearchMCP; honors ROUND:n prefix for max rounds."""
    text = _acp_messages_to_text(input)
    rnd, body = parse_research_round_payload(text)
    if rnd > MAX_RESEARCH_ROUNDS:
        msg = (
            "Research round limit reached (3 rounds: initial plus two revisions). "
            "Do not call research again; use current findings and proceed to "
            "save_report with the best possible markdown, or critique once more if needed."
        )
        yield Message(parts=[MessagePart(content=strip_surrogates(msg), content_type="text/plain")])
        return
    tools = await mcp_tools_to_langchain(settings.search_mcp_url)
    graph = create_agent(
        chat_model(),
        tools,
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        response_format=None,
        checkpointer=None,
        name="researcher",
    )
    result = await graph.ainvoke(
        {"messages": [research_task_message(body)]},
        config={"recursion_limit": settings.subagent_recursion_limit},
    )
    messages = result.get("messages") or []
    out = extract_findings_text(messages)
    yield Message(parts=[MessagePart(content=strip_surrogates(out), content_type="text/plain")])


@server.agent()
async def critic(
    input: list[Message], context: Context
) -> AsyncGenerator[RunYield, RunYieldResume]:
    """Structured critique (CritiqueResult) using SearchMCP tools."""
    text = _acp_messages_to_text(input)
    tools = await mcp_tools_to_langchain(settings.search_mcp_url)
    graph = create_agent(
        chat_model(),
        tools,
        system_prompt=CRITIC_SYSTEM_PROMPT,
        response_format=CritiqueResult,
        checkpointer=None,
        name="critic",
    )
    result = await graph.ainvoke(
        {"messages": [critic_task_message(text)]},
        config={"recursion_limit": settings.subagent_recursion_limit},
    )
    structured = result.get("structured_response")
    if isinstance(structured, CritiqueResult):
        out = json.dumps(
            strip_surrogates_deep(structured.model_dump(mode="json")),
            indent=2,
            ensure_ascii=False,
        )
    elif structured is not None:
        out = json.dumps(
            strip_surrogates_deep(structured)
            if isinstance(structured, dict)
            else strip_surrogates(str(structured)),
            indent=2,
            default=str,
            ensure_ascii=False,
        )
    else:
        out = json.dumps(
            {
                "verdict": "REVISE",
                "is_fresh": False,
                "is_complete": False,
                "is_well_structured": False,
                "strengths": [],
                "gaps": ["Critic did not return structured output"],
                "revision_requests": ["Re-run research with clearer sources"],
            }
        )
    yield Message(parts=[MessagePart(content=strip_surrogates(out), content_type="text/plain")])


if __name__ == "__main__":
    server.run(host="127.0.0.1", port=settings.acp_port)
