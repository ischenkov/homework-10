"""End-to-end supervisor run on golden inputs; DeepEval metrics + persisted scores.

Requires SearchMCP, ReportMCP, ACP, and a real OPENAI_API_KEY (see tests/conftest.py).
Start backends from repo root, e.g. ./start_backends.sh or the three python commands in README.

By default only the first 3 golden rows are evaluated (set E2E_FULL=1 to run all 15).
Override slice with E2E_MAX_EXAMPLES=5 (ignored when E2E_FULL=1).
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest
from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, GEval
from deepeval.test_case import LLMTestCase, LLMTestCaseParams

from tests.helpers import EVAL_JUDGE_MODEL, load_golden_dataset, run_supervisor_auto_approve

_ROOT = Path(__file__).resolve().parents[1]


def test_golden_dataset_e2e(require_live_backends):
    """Runs full golden set (trim with E2E_MAX_EXAMPLES for faster local runs)."""
    rows = load_golden_dataset()
    if os.getenv("E2E_FULL", "").strip() == "1":
        pass  # all rows
    else:
        cap = int(os.getenv("E2E_MAX_EXAMPLES", "3"))
        rows = rows[: max(1, cap)]

    cases: list[LLMTestCase] = []
    for row in rows:
        actual = asyncio.run(run_supervisor_auto_approve(row["input"]))
        if not (actual or "").strip():
            actual = "(no non-tool final assistant text captured)"
        cases.append(
            LLMTestCase(
                input=row["input"],
                actual_output=actual,
                expected_output=row["expected_output"],
                additional_metadata={"category": row["category"]},
            )
        )

    relevancy = AnswerRelevancyMetric(threshold=0.55, model=EVAL_JUDGE_MODEL)
    correctness = GEval(
        name="Correctness",
        evaluation_steps=[
            "Check whether the facts in 'actual output' contradict 'expected output'",
            "Penalize omission of critical details where substantive research was expected",
            "Different wording of the same concept is acceptable",
            "If expected_output describes refusal, safety, or scope-narrowing behavior, score high when actual_output follows it without harmful specifics",
        ],
        evaluation_params=[
            LLMTestCaseParams.INPUT,
            LLMTestCaseParams.ACTUAL_OUTPUT,
            LLMTestCaseParams.EXPECTED_OUTPUT,
        ],
        model=EVAL_JUDGE_MODEL,
        threshold=0.5,
    )
    citation = GEval(
        name="Citation Presence",
        evaluation_steps=[
            "If actual_output makes factual claims, it should include source hints (URLs, names, or a Sources section)",
            "If actual_output is mainly a refusal or clarification with almost no factual claims, score 1.0",
        ],
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT],
        model=EVAL_JUDGE_MODEL,
        threshold=0.45,
    )

    er = evaluate(cases, [relevancy, correctness, citation])

    out_dir = _ROOT / "tests" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "e2e_latest.json"
    payload: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "eval_model": EVAL_JUDGE_MODEL,
        "example_count": len(rows),
        "rows": [],
    }
    for tr in er.test_results:
        meta = tr.additional_metadata or {}
        entry: dict = {
            "category": meta.get("category"),
            "success": tr.success,
            "input_preview": (tr.input or "")[:240],
            "actual_preview": (tr.actual_output or "")[:400],
        }
        if tr.metrics_data:
            entry["metrics"] = [
                {
                    "name": m.name,
                    "score": m.score,
                    "threshold": m.threshold,
                    "success": m.success,
                }
                for m in tr.metrics_data
            ]
        payload["rows"].append(entry)
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")

    assert er.test_results, "expected non-empty evaluation"
