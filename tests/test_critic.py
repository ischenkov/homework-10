"""Critic agent: critique quality for APPROVE vs REVISE-style outputs."""

from __future__ import annotations

import asyncio
import json

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from tests.helpers import EVAL_JUDGE_MODEL, run_critic, run_planner, run_researcher


@pytest.fixture
def critique_quality_metric():
    return GEval(
        name="Critique Quality",
        evaluation_steps=[
            "Check that the critique identifies specific issues, not vague complaints",
            "Check that revision_requests are actionable (researcher can act on them)",
            "If verdict is APPROVE, gaps list should be empty or contain only minor items",
            "If verdict is REVISE, there must be at least one revision_request",
        ],
        evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT],
        model=EVAL_JUDGE_MODEL,
        threshold=0.6,
    )


def _run_pipeline(user: str) -> str:
    plan_json, _ = asyncio.run(run_planner(user))
    instructions = f"User:\n{user}\n\nPlan:\n{plan_json}\n\nExecute and summarize."
    findings, _ = asyncio.run(run_researcher(instructions))
    critique_json, _ = asyncio.run(run_critic(findings))
    return critique_json


def test_critique_on_strong_brief(require_live_backends, critique_quality_metric):
    user = "Define RAG in 3 bullet points for executives; keep each bullet under 25 words."
    critique_json = _run_pipeline(user)
    data = json.loads(critique_json)
    assert "verdict" in data
    assert_test(
        test_case=LLMTestCase(input=user, actual_output=critique_json),
        metrics=[critique_quality_metric],
    )


def test_critique_on_weak_brief(require_live_backends, critique_quality_metric):
    user = "List the exact GDP of every country in 2030 as definitive fact."
    critique_json = _run_pipeline(user)
    assert "verdict" in json.loads(critique_json)
    assert_test(
        test_case=LLMTestCase(input=user, actual_output=critique_json),
        metrics=[critique_quality_metric],
    )
