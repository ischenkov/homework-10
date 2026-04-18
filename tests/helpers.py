"""In-process agent runners and utilities for DeepEval tests (mirrors acp_server agents)."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command

from langchain_openai import ChatOpenAI

from agent_factory import create_agent
from agents.critic import critic_task_message
from agents.planner import planner_task_message
from agents.research import extract_findings_text, research_task_message
from config import CRITIC_SYSTEM_PROMPT, PLANNER_SYSTEM_PROMPT, RESEARCH_SYSTEM_PROMPT, Settings
from mcp_utils import mcp_tools_to_langchain
from schemas import CritiqueResult, ResearchPlan
from text_sanitize import strip_surrogates_deep

_PROJECT_ROOT = Path(__file__).resolve().parents[1]

EVAL_JUDGE_MODEL = os.getenv("DEEPEVAL_EVAL_MODEL", "gpt-4o-mini")


def _chat_model():
    s = Settings()
    return ChatOpenAI(
        model=s.model_name,
        temperature=0.7,
        api_key=s.api_key.get_secret_value(),
    )

_ALLOWED_CATEGORIES = frozenset({"happy_path", "edge_cases", "failure_cases"})


def load_golden_dataset(path: Path | None = None) -> list[dict[str, Any]]:
    p = path or (_PROJECT_ROOT / "tests" / "golden_dataset.json")
    data = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("golden_dataset.json must be a JSON array")
    for i, row in enumerate(data):
        for key in ("input", "expected_output", "category"):
            if key not in row:
                raise ValueError(f"golden row {i} missing {key!r}")
        if row["category"] not in _ALLOWED_CATEGORIES:
            raise ValueError(f"golden row {i} invalid category: {row['category']!r}")
    return data


def _stream_updates_payload(chunk: Any) -> dict[str, Any]:
    if isinstance(chunk, dict) and chunk.get("type") == "updates" and isinstance(
        chunk.get("data"), dict
    ):
        return chunk["data"]
    return chunk if isinstance(chunk, dict) else {}


def _extract_interrupt_list(chunk: dict) -> list | None:
    payload = _stream_updates_payload(chunk)
    if "__interrupt__" in payload:
        return payload["__interrupt__"]
    for _node, node_output in payload.items():
        if isinstance(node_output, dict) and "__interrupt__" in node_output:
            return node_output["__interrupt__"]
    return None


def tool_messages_to_retrieval_context(messages: list) -> list[str]:
    chunks: list[str] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            content = m.content or ""
            if isinstance(content, list):
                content = " ".join(
                    c.get("text", "") if isinstance(c, dict) else str(c)
                    for c in content
                )
            body = content.strip()
            if body:
                chunks.append(body[:16000])
    return chunks if chunks else ["(no tool outputs)"]


def messages_to_tool_calls(messages: list) -> list:
    from deepeval.test_case import ToolCall

    out: list[ToolCall] = []
    for m in messages:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls or []:
                if isinstance(tc, dict):
                    name = tc.get("name") or ""
                    args = tc.get("args") or {}
                else:
                    name = getattr(tc, "name", "") or ""
                    args = getattr(tc, "args", None) or {}
                if not isinstance(args, dict):
                    args = {}
                out.append(ToolCall(name=name, input_parameters=args))
    return out


async def run_planner(user_request: str) -> tuple[str, list]:
    s = Settings()
    tools = await mcp_tools_to_langchain(s.search_mcp_url)
    graph = create_agent(
        _chat_model(),
        tools,
        system_prompt=PLANNER_SYSTEM_PROMPT,
        response_format=ResearchPlan,
        checkpointer=None,
        name="planner",
    )
    result = await graph.ainvoke(
        {"messages": [planner_task_message(user_request)]},
        config={"recursion_limit": s.subagent_recursion_limit},
    )
    structured = result.get("structured_response")
    if isinstance(structured, ResearchPlan):
        out = json.dumps(
            strip_surrogates_deep(structured.model_dump(mode="json")),
            indent=2,
            ensure_ascii=False,
        )
    elif structured is not None:
        out = strip_surrogates_deep(str(structured))
    else:
        out = "Error: planner did not return a structured plan."
    return out, list(result.get("messages") or [])


async def run_researcher(instructions: str) -> tuple[str, list]:
    s = Settings()
    tools = await mcp_tools_to_langchain(s.search_mcp_url)
    graph = create_agent(
        _chat_model(),
        tools,
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        response_format=None,
        checkpointer=None,
        name="researcher",
    )
    result = await graph.ainvoke(
        {"messages": [research_task_message(instructions)]},
        config={"recursion_limit": s.subagent_recursion_limit},
    )
    messages = list(result.get("messages") or [])
    return extract_findings_text(messages), messages


async def run_critic(findings: str) -> tuple[str, list]:
    s = Settings()
    tools = await mcp_tools_to_langchain(s.search_mcp_url)
    graph = create_agent(
        _chat_model(),
        tools,
        system_prompt=CRITIC_SYSTEM_PROMPT,
        response_format=CritiqueResult,
        checkpointer=None,
        name="critic",
    )
    result = await graph.ainvoke(
        {"messages": [critic_task_message(findings)]},
        config={"recursion_limit": s.subagent_recursion_limit},
    )
    structured = result.get("structured_response")
    if isinstance(structured, CritiqueResult):
        out = json.dumps(
            strip_surrogates_deep(structured.model_dump(mode="json")),
            indent=2,
            ensure_ascii=False,
        )
    elif structured is not None:
        out = json.dumps(structured, default=str, ensure_ascii=False)
    else:
        out = json.dumps(
            {
                "verdict": "REVISE",
                "gaps": ["Critic did not return structured output"],
                "revision_requests": ["Re-run research"],
            },
            ensure_ascii=False,
        )
    return out, list(result.get("messages") or [])


async def run_supervisor_auto_approve(user_input: str) -> str:
    """Run production supervisor; auto-approve save_report HITL interrupts.

    Returns the save_report content (the actual research report) when available,
    falling back to the last non-tool assistant message.
    """
    from supervisor import supervisor

    s = Settings()
    thread_id = str(uuid.uuid4())
    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": s.agent_recursion_limit,
    }
    payload: Any = {"messages": [HumanMessage(content=user_input)]}
    last_text = ""
    save_report_content = ""
    stream_kwargs: dict[str, Any] = {"stream_mode": "updates", "version": "v2"}
    while True:
        got_interrupt = False
        try:
            stream_iter = supervisor.astream(payload, config=config, **stream_kwargs)
        except TypeError:
            stream_kwargs.pop("version", None)
            stream_iter = supervisor.astream(payload, config=config, **stream_kwargs)
        async for chunk in stream_iter:
            if not isinstance(chunk, dict):
                continue
            intrs = _extract_interrupt_list(chunk)
            if intrs:
                payload = Command(resume={"decisions": [{"type": "approve"}]})
                got_interrupt = True
                break
            inner = _stream_updates_payload(chunk)
            for _node, node_output in inner.items():
                if str(_node).startswith("__"):
                    continue
                if not isinstance(node_output, dict):
                    continue
                for msg in node_output.get("messages", []) or []:
                    if isinstance(msg, AIMessage) and (msg.content or "").strip():
                        if not (msg.tool_calls or []):
                            last_text = str(msg.content).strip()
                    # Capture the report content from save_report tool call args
                    if isinstance(msg, AIMessage):
                        for tc in msg.tool_calls or []:
                            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
                            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", {})
                            if name == "save_report" and isinstance(args, dict):
                                content = args.get("content", "")
                                if content:
                                    save_report_content = content
        if not got_interrupt:
            break
    return save_report_content or last_text
