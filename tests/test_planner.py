"""Planner agent: Plan Quality GEval + custom plan–request alignment."""

from __future__ import annotations

import asyncio

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from tests.helpers import EVAL_JUDGE_MODEL, run_planner


@pytest.fixture
def plan_quality_metric():
    return GEval(
        name="Plan Quality",
        evaluation_steps=[
            "Check that the plan contains specific search queries (not vague)",
            "Check that sources_to_check includes relevant sources for the topic",
            "Check that the output_format matches what the user asked for",
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=EVAL_JUDGE_MODEL,
        threshold=0.65,
    )


@pytest.fixture
def plan_request_alignment_metric():
    """Custom business metric: plan goal and output_format track the user request."""
    return GEval(
        name="Plan–Request Alignment",
        evaluation_steps=[
            "Check that the plan's stated goal reflects the user's core question (not a different topic)",
            "If the user asked for a specific deliverable shape (bullets, table, sections), output_format should mention it",
            "Penalize generic boilerplate goals that could apply to any topic",
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=EVAL_JUDGE_MODEL,
        threshold=0.6,
    )


def test_plan_quality(require_live_backends, plan_quality_metric):
    user = (
        "Plan research on how hybrid BM25+dense retrieval is used in production RAG. "
        "Final report should use clear headings and a short comparison table."
    )
    plan_json, _msgs = asyncio.run(run_planner(user))
    assert "search_queries" in plan_json.lower()
    assert_test(
        test_case=LLMTestCase(input=user, actual_output=plan_json),
        metrics=[plan_quality_metric],
    )


def test_plan_request_alignment(require_live_backends, plan_request_alignment_metric):
    user = "Compare FAISS HNSW vs IVF for latency and memory; output as bullet tradeoffs only."
    plan_json, _ = asyncio.run(run_planner(user))
    assert_test(
        test_case=LLMTestCase(input=user, actual_output=plan_json),
        metrics=[plan_request_alignment_metric],
    )
