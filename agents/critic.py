"""Critic: user message builder for ACP / LangGraph."""

from __future__ import annotations

from langchain_core.messages import HumanMessage


def critic_task_message(findings: str) -> HumanMessage:
    return HumanMessage(
        content=(
            "Critique the following research findings. "
            "Use tools to verify key claims where needed.\n\n"
            f"{findings}"
        )
    )
