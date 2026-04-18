"""Planner: prompts and user message builder for ACP / LangGraph."""

from __future__ import annotations

from langchain_core.messages import HumanMessage


def planner_task_message(request: str) -> HumanMessage:
    return HumanMessage(
        content=(
            "Create a research plan for this request:\n\n"
            f"{request}\n\n"
            "Use tools if needed, then output the structured plan."
        )
    )
