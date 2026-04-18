"""Researcher: round limits, payload parsing, and findings extraction."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage

MAX_RESEARCH_ROUNDS = 3


def parse_research_round_payload(text: str) -> tuple[int, str]:
    """Parse 'ROUND:n\\n\\nbody' from supervisor; default round 1."""
    t = text.strip()
    if not t.upper().startswith("ROUND:"):
        return 1, t
    first, _, rest = t.partition("\n")
    try:
        n = int(first.split(":", 1)[1].strip())
    except (ValueError, IndexError):
        n = 1
    return max(1, n), rest.strip()


def research_task_message(instructions: str) -> HumanMessage:
    return HumanMessage(content=instructions)


def extract_findings_text(messages: list) -> str:
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and (msg.content or "").strip():
            if not msg.tool_calls:
                return str(msg.content).strip()
    parts = []
    for msg in messages:
        role = type(msg).__name__
        body = getattr(msg, "content", "") or ""
        if body.strip():
            parts.append(f"[{role}]\n{body.strip()[:4000]}")
    return "\n\n".join(parts) if parts else "No findings produced."


def reset_research_budget() -> None:
    """No-op: research rounds are passed explicitly via ACP payload."""
