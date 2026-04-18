"""Tool correctness: planner search tools, researcher tools, supervisor save_report."""

from __future__ import annotations

import asyncio
import json

import pytest
from deepeval import assert_test
from deepeval.metrics import ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from tests.helpers import EVAL_JUDGE_MODEL, run_planner, run_researcher, messages_to_tool_calls


def test_planner_uses_search_tools(require_live_backends):
    user = "What is BM25 in two sentences? Use tools briefly then plan indexing tradeoffs."
    _plan, messages = asyncio.run(run_planner(user))
    called = messages_to_tool_calls(messages)
    names = {t.name for t in called}
    assert "web_search" in names or "knowledge_search" in names
    expected: list[ToolCall] = []
    if "web_search" in names:
        expected.append(ToolCall(name="web_search"))
    if "knowledge_search" in names:
        expected.append(ToolCall(name="knowledge_search"))
    metric = ToolCorrectnessMetric(threshold=0.4, model=EVAL_JUDGE_MODEL)
    assert_test(
        test_case=LLMTestCase(
            input=user,
            tools_called=called,
            expected_tools=expected,
        ),
        metrics=[metric],
    )


def test_researcher_uses_tools_per_plan(require_live_backends):
    user = "One-paragraph definition of vector embeddings for search."
    plan_json, _ = asyncio.run(run_planner(user))
    instructions = f"User:\n{user}\n\nPlan JSON:\n{plan_json}\n\nExecute with tools."
    _findings, messages = asyncio.run(run_researcher(instructions))
    called = messages_to_tool_calls(messages)
    names = {t.name for t in called}
    assert names & {"web_search", "knowledge_search", "read_url"}
    metric = ToolCorrectnessMetric(threshold=0.4, model=EVAL_JUDGE_MODEL)
    try:
        plan = json.loads(plan_json)
    except json.JSONDecodeError:
        plan = {}
    expected: list[ToolCall] = []
    for s in plan.get("sources_to_check") or []:
        sl = str(s).lower()
        if "knowledge" in sl or "local" in sl:
            expected.append(ToolCall(name="knowledge_search"))
        if "web" in sl:
            expected.append(ToolCall(name="web_search"))
    if not expected:
        expected = [ToolCall(name="web_search")]
    for t in expected:
        assert t.name in names, f"plan implied {t.name} but it was not called"
    assert_test(
        test_case=LLMTestCase(
            input=instructions[:2000],
            tools_called=called,
            expected_tools=expected,
        ),
        metrics=[metric],
    )


def test_supervisor_calls_save_report_after_workflow(require_live_backends):
    """Expected save_report once critique path completes (synthetic trace, names match supervisor tools)."""
    metric = ToolCorrectnessMetric(threshold=0.45, model=EVAL_JUDGE_MODEL)
    user = "After APPROVE, persist the final markdown report via save_report."
    called = [
        ToolCall(name="delegate_to_planner", input_parameters={"request": "topic"}),
        ToolCall(
            name="delegate_to_researcher",
            input_parameters={"instructions": "plan text", "round_number": 1},
        ),
        ToolCall(name="delegate_to_critic", input_parameters={"findings": "text"}),
        ToolCall(
            name="save_report",
            input_parameters={"filename": "report.md", "content": "# Summary\n"},
        ),
    ]
    assert_test(
        test_case=LLMTestCase(
            input=user,
            tools_called=called,
            expected_tools=[ToolCall(name="save_report")],
        ),
        metrics=[metric],
    )
