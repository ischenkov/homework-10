import asyncio
import json
import os
import sys
import uuid
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings("ignore", message="urllib3 v2 only supports OpenSSL")

try:
    from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
except ImportError:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    venv_python = os.path.join(script_dir, ".venv", "bin", "python")
    if os.path.exists(venv_python):
        os.execv(venv_python, [venv_python] + sys.argv)
    print("Error: Dependencies not found. Install with: pip install -r requirements.txt")
    print("Then run: .venv/bin/python main.py  or  ./run.sh")
    sys.exit(1)

from langgraph.types import Command

from agents import reset_research_budget
from config import Settings
from service_checks import format_startup_service_report
from supervisor import supervisor

settings = Settings()
_PROJECT_ROOT = Path(__file__).resolve().parent


def _print_connection_help() -> None:
    print()
    print("Connection failed — supervisor needs MCP + ACP running. Check:")
    print(format_startup_service_report(settings))


def _print_startup_hints():
    index_dir = _PROJECT_ROOT / "index"
    chunks = index_dir / "chunks.pkl"
    if not chunks.is_file():
        print(
            "Note: Knowledge index missing. TXT/PDF in data/ are not searchable until you run:\n"
            "  python ingest.py\n"
        )
    print("Backend services (must be running before you chat):")
    print(format_startup_service_report(settings))
    out = (
        _PROJECT_ROOT / settings.output_dir
        if not Path(settings.output_dir).is_absolute()
        else Path(settings.output_dir)
    )
    out = out.resolve()
    print(f"Reports go to: {out}/")
    print()


def _stream_updates_payload(chunk: Any) -> dict[str, Any]:
    """LangGraph astream(..., version='v2') wraps updates in {type, ns, data}; v1 yields the inner dict."""
    if isinstance(chunk, dict) and chunk.get("type") == "updates" and isinstance(
        chunk.get("data"), dict
    ):
        return chunk["data"]
    return chunk if isinstance(chunk, dict) else {}


def _print_stream_chunk(chunk: dict) -> None:
    payload = _stream_updates_payload(chunk)
    for node_name, node_output in payload.items():
        if node_name == "__interrupt__" or str(node_name).startswith("__"):
            continue
        messages = node_output.get("messages", []) if isinstance(node_output, dict) else []
        for msg in messages:
            if isinstance(msg, AIMessage):
                tool_calls = getattr(msg, "tool_calls", None) or []
                if msg.content and str(msg.content).strip():
                    if tool_calls:
                        print(f"  Thought: {str(msg.content).strip()}")
                    else:
                        print(f"  {str(msg.content).strip()}")
                for tc in tool_calls:
                    name = tc.get("name", "?") if isinstance(tc, dict) else getattr(tc, "name", "?")
                    args = tc.get("args", {}) if isinstance(tc, dict) else getattr(tc, "args", {}) or {}
                    parts = []
                    for k, v in args.items():
                        st = f'"{v}"' if isinstance(v, str) else str(v)
                        if isinstance(v, str) and len(st) > 60:
                            st = st[:57] + '..."'
                        parts.append(st)
                    arg_str = ", ".join(parts)
                    print(f"  → {name}({arg_str})")
            elif isinstance(msg, ToolMessage):
                body = (msg.content or "").strip()
                tname = getattr(msg, "name", None) or ""
                if len(body) > 500:
                    body = body[:500] + "..."
                print(f"  ← {tname or 'tool'}: {body}")


def _extract_interrupt_list(chunk: dict) -> list | None:
    payload = _stream_updates_payload(chunk)
    if "__interrupt__" in payload:
        return payload["__interrupt__"]
    for _node, node_output in payload.items():
        if isinstance(node_output, dict) and "__interrupt__" in node_output:
            return node_output["__interrupt__"]
    return None


def _hitl_tool_args(raw: Any) -> dict[str, Any]:
    """Normalize tool call args (dict or JSON string) for HITL preview."""
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _hitl_data_from_interrupt(intrs: list) -> dict | None:
    if not intrs:
        return None
    first = intrs[0]
    v = first.value if hasattr(first, "value") else first
    if not isinstance(v, dict):
        return None
    if "filename" in v and "content" in v:
        return {"filename": v["filename"], "content": v["content"]}
    action_requests = v.get("action_requests") or []
    if action_requests and isinstance(action_requests[0], dict):
        ar0 = action_requests[0]
        # HumanInTheLoopMiddleware uses key "args" (not "arguments").
        raw_args = ar0.get("args")
        if raw_args is None:
            raw_args = ar0.get("arguments")
        args = _hitl_tool_args(raw_args)
        fn = args.get("filename")
        co = args.get("content")
        if fn is not None and co is not None:
            return {"filename": fn, "content": co}
    return None


def _hitl_banner(data: dict) -> None:
    fn = data.get("filename", "?")
    content = data.get("content", "") or ""
    preview = content[:2000] + ("..." if len(content) > 2000 else "")
    print()
    print("  " + "=" * 60)
    print("  ACTION REQUIRES APPROVAL (save_report)")
    print("  " + "=" * 60)
    print(f"    File:  {fn}")
    print("    Preview:")
    for line in preview.splitlines()[:40]:
        print(f"      {line}")
    if preview.count("\n") >= 40 or len(content) > len(preview):
        print("      ...")
    print()


def _prompt_hitl(filename: str, content: str) -> dict:
    while True:
        choice = input("  approve / edit / reject: ").strip().lower()
        if choice == "approve":
            return {"decisions": [{"type": "approve"}]}
        if choice == "reject":
            reason = input("  Reason (optional): ").strip()
            return {
                "decisions": [
                    {"type": "reject", "message": reason or "User rejected save."}
                ]
            }
        if choice == "edit":
            fb = input("  Your feedback: ").strip()
            new_body = (content + "\n\n### User feedback (pre-save)\n" + fb).strip()
            return {
                "decisions": [
                    {
                        "type": "edit",
                        "edited_action": {
                            "name": "save_report",
                            "args": {
                                "filename": filename,
                                "content": new_body,
                            },
                        },
                    }
                ]
            }
        print("  Please enter approve, edit, or reject.")


async def _astream_supervisor(payload: Any, config: dict[str, Any]):
    """Async stream so async @tool handlers run via ainvoke (sync stream would call invoke only)."""
    kwargs: dict[str, Any] = {
        "stream_mode": "updates",
        "version": "v2",
    }
    try:
        async for chunk in supervisor.astream(payload, config=config, **kwargs):
            yield chunk
    except TypeError:
        kwargs.pop("version", None)
        async for chunk in supervisor.astream(payload, config=config, **kwargs):
            yield chunk


async def _run_one_user_turn(user_input: str) -> None:
    thread_id = str(uuid.uuid4())
    config: dict[str, Any] = {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": settings.agent_recursion_limit,
    }
    payload: Any = {"messages": [HumanMessage(content=user_input)]}

    print("\nAgent:")
    while True:
        got_interrupt = False
        async for chunk in _astream_supervisor(payload, config):
            if not isinstance(chunk, dict):
                continue
            intrs = _extract_interrupt_list(chunk)
            if intrs:
                data = _hitl_data_from_interrupt(intrs)
                if data:
                    _hitl_banner(data)
                    resume_dict = _prompt_hitl(
                        str(data.get("filename", "report.md")),
                        str(data.get("content", "")),
                    )
                    payload = Command(resume=resume_dict)
                    got_interrupt = True
                    break
            _print_stream_chunk(chunk)

        if not got_interrupt:
            break


def main():
    print("Multi-Agent Research Supervisor (type 'exit' to quit)")
    print("-" * 40)
    _print_startup_hints()

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        reset_research_budget()

        try:
            asyncio.run(_run_one_user_turn(user_input))
        except Exception as e:
            err = str(e).lower()
            if any(
                x in err
                for x in (
                    "connection",
                    "connect",
                    "refused",
                    "all connection attempts failed",
                    "name or service not known",
                )
            ):
                _print_connection_help()
            print(f"\nAgent error: {e}")


if __name__ == "__main__":
    main()
