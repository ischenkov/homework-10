"""Research agent: groundedness vs tool retrieval context."""

from __future__ import annotations

import asyncio
import json

import pytest
from deepeval import assert_test
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from tests.helpers import EVAL_JUDGE_MODEL, run_planner, run_researcher, tool_messages_to_retrieval_context


@pytest.fixture
def groundedness_metric():
    return GEval(
        name="Groundedness",
        evaluation_steps=[
            "Extract every factual claim from 'actual output'",
            "For each claim, check if it can be directly supported by 'retrieval context'",
            "Claims not present in retrieval context count as ungrounded, even if true",
            "Score = number of grounded claims / total claims (if no factual claims, score 1.0)",
        ],
        evaluation_params=[
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.RETRIEVAL_CONTEXT,
        ],
        model=EVAL_JUDGE_MODEL,
        threshold=0.55,
    )


def test_research_groundedness(require_live_backends, groundedness_metric):
    user = "What is sentence-window retrieval in RAG? One short paragraph with definition only."
    plan_json, _ = asyncio.run(run_planner(user))
    try:
        json.loads(plan_json)
    except json.JSONDecodeError:
        pytest.fail("planner did not return valid JSON")
    instructions = (
        f"User request:\n{user}\n\nResearch plan (JSON):\n{plan_json}\n\n"
        "Execute the plan and answer succinctly."
    )
    findings, messages = asyncio.run(run_researcher(instructions))
    ctx = tool_messages_to_retrieval_context(messages)
    assert_test(
        test_case=LLMTestCase(
            input=user,
            actual_output=findings,
            retrieval_context=ctx,
        ),
        metrics=[groundedness_metric],
    )
