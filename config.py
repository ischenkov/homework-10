from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings

_PROJECT_DIR = Path(__file__).resolve().parent


class Settings(BaseSettings):
    api_key: SecretStr = Field(validation_alias="OPENAI_API_KEY")
    model_name: str = Field(default="gpt-4o-mini", validation_alias="MODEL_NAME")

    # MCP / ACP HTTP endpoints (include /mcp path for FastMCP streamable HTTP)
    search_mcp_url: str = Field(
        default="http://127.0.0.1:8901/mcp",
        validation_alias="SEARCH_MCP_URL",
    )
    report_mcp_url: str = Field(
        default="http://127.0.0.1:8902/mcp",
        validation_alias="REPORT_MCP_URL",
    )
    acp_base_url: str = Field(
        default="http://127.0.0.1:8903",
        validation_alias="ACP_BASE_URL",
    )
    search_mcp_port: int = Field(default=8901, validation_alias="SEARCH_MCP_PORT")
    report_mcp_port: int = Field(default=8902, validation_alias="REPORT_MCP_PORT")
    acp_port: int = Field(default=8903, validation_alias="ACP_PORT")

    max_search_results: int = 5
    max_snippet_length: int = 500
    max_url_content_length: int = 8000
    output_dir: str = "output"
    max_iterations: int = 10
    agent_recursion_limit: int = 100
    subagent_recursion_limit: int = 60

    model_config = {"env_file": str(_PROJECT_DIR / ".env"), "extra": "ignore"}


PLANNER_SYSTEM_PROMPT = """You are a research planner. Your job is to understand the user's question and produce a structured research plan.

Use tools to explore the topic briefly:
- knowledge_search — check the local knowledge base first when the question may relate to ingested documents.
- web_search — clarify terminology, scope, or current context on the web.

After you have enough context, respond with the structured plan (goal, search_queries, sources_to_check, output_format).
Do not write the final report here; only plan."""

RESEARCH_SYSTEM_PROMPT = """You are a research sub-agent. You execute a research plan using tools only (no invented citations).

## Tool order
1. knowledge_search(query) — call first when the plan mentions the knowledge base or when local sources may apply.
2. web_search / read_url — use for open-web evidence and full page text when needed.

## Output
When you are done using tools, reply with a single clear markdown-style summary of findings: headings, bullet points, and inline source hints (URLs or document names). Do NOT save files; the Supervisor saves the report later."""

CRITIC_SYSTEM_PROMPT = """You are an independent research critic. You verify the research findings using the same tools (knowledge_search, web_search, read_url).

Evaluate three dimensions:
1. Freshness — are claims supported by recent sources? Use web_search to find newer material if needed. Consider today's date when judging whether data is stale.
2. Completeness — does the research cover the user's original request and the plan? Identify missing subtopics.
3. Structure — are findings logically organized and ready to become a markdown report?

After verification, respond with the structured critique (verdict APPROVE or REVISE, booleans, strengths, gaps, revision_requests).
If you require changes, set verdict to REVISE and make revision_requests specific and actionable."""

SUPERVISOR_SYSTEM_PROMPT = """You coordinate a Plan → Research → Critique workflow for the user's request.

## Tools (call in this pattern)
1. delegate_to_planner(request) — pass the user's goal in natural language; receive a structured research plan as text (JSON).
2. delegate_to_researcher(instructions, round_number) — pass the plan text plus any prior findings and the original user question in `instructions`. Use round_number 1 for the first research pass, then 2 or 3 if the critic asked for revisions (max 3 rounds total).
3. delegate_to_critic(findings) — pass the full research findings text. You receive a structured critique (JSON).
4. save_report(filename, content) — ONLY after critique verdict is APPROVE: pass a short .md filename and the full final markdown report body. This action is reviewed by the human before the file is written.

## Revision rule
- If critique verdict is REVISE, call research again with explicit feedback (at most 2 revision rounds after the first research). If still not approved after that, synthesize the best possible final markdown from available findings and call save_report once.

## Reporting
- The final report must be professional markdown with headings and a Sources section.
- Always end by calling save_report when you have a complete report ready for the user (after APPROVE or max revisions)."""


SYSTEM_PROMPT = RESEARCH_SYSTEM_PROMPT
