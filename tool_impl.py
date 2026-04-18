"""Pure implementations of search/report helpers (no LangChain @tool, no HITL). Used by MCP servers and optional local wrappers."""

from __future__ import annotations

import pickle
from pathlib import Path

import trafilatura
from ddgs import DDGS

from config import Settings
from retriever import search_knowledge_base
from text_sanitize import strip_surrogates

_PROJECT_ROOT = Path(__file__).resolve().parent


def resolved_output_dir(settings: Settings) -> Path:
    p = Path(settings.output_dir)
    if p.is_absolute():
        return p.resolve()
    return (_PROJECT_ROOT / p).resolve()


def format_search_results(results: list[dict]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        if "error" in r:
            return r["error"]
        lines.append(f"{i}. {r.get('title', '')}\n   URL: {r.get('url', '')}\n   {r.get('snippet', '')}")
    return "\n\n".join(lines) if lines else "No results found."


def impl_web_search(query: str, settings: Settings) -> str:
    try:
        results = list(DDGS().text(query, max_results=settings.max_search_results))
        formatted = []
        for r in results:
            title = r.get("title", "")
            href = r.get("href", "")
            body = r.get("body", "") or ""
            max_snip = settings.max_snippet_length
            if len(body) > max_snip:
                body = body[:max_snip] + "..."
            formatted.append({"title": title, "url": href, "snippet": body})
        return format_search_results(formatted)
    except Exception as e:
        return f"Search failed: {str(e)}"


def impl_read_url(url: str, settings: Settings) -> str:
    try:
        downloaded = trafilatura.fetch_url(url)
        if downloaded is None:
            return (
                "Error: Could not fetch or parse this URL. It may be invalid, blocked, "
                "or not contain extractable text."
            )
        text = trafilatura.extract(downloaded)
        if not text or not text.strip():
            return "Error: No text content could be extracted from this page."
        max_len = settings.max_url_content_length
        if len(text) > max_len:
            text = text[:max_len] + "\n\n[Content truncated for context length.]"
        return strip_surrogates(text)
    except Exception as e:
        return f"Error: {str(e)}"


def impl_knowledge_search(query: str, settings: Settings) -> str:
    try:
        docs = search_knowledge_base(query, top_k=5)
        if not docs:
            return (
                "No documents found in the knowledge base. Run 'python ingest.py' to load documents "
                "from ./data/ first, or the query may not match any ingested content."
            )
        lines = []
        for doc in docs:
            source = doc.metadata.get("source", "Unknown")
            page = doc.metadata.get("page", "")
            ref = f"[{source}]"
            if page is not None and page != "":
                ref += f" Page {page}"
            content = doc.page_content[:800] + ("..." if len(doc.page_content) > 800 else "")
            lines.append(f"   - {ref}\n{content}")
        return f"[{len(docs)} documents found]\n\n" + "\n\n---\n\n".join(lines)
    except Exception as e:
        return (
            f"Knowledge base search failed: {str(e)}. Ensure you've run 'python ingest.py' "
            "to build the index."
        )


def safe_report_basename(filename: str) -> str | None:
    name = (filename or "").strip()
    if not name:
        return None
    if ".." in name or "/" in name or "\\" in name:
        return None
    p = Path(name)
    if p.is_absolute():
        return None
    if len(p.parts) != 1:
        return None
    base = p.name
    if not base or base in (".", ".."):
        return None
    return base


def write_report_to_disk(filename: str, content: str, settings: Settings) -> str:
    try:
        safe = safe_report_basename(filename)
        if safe is None:
            return (
                "Error: invalid filename. Use a single file name only "
                "(e.g. report.md), without paths or '..'."
            )
        if not safe.endswith(".md"):
            safe = safe + ".md"
        out_dir = resolved_output_dir(settings)
        out_dir.mkdir(parents=True, exist_ok=True)
        path = (out_dir / safe).resolve()
        if not path.is_relative_to(out_dir):
            return "Error: invalid report path."
        path.write_text(strip_surrogates(content), encoding="utf-8")
        return f"Report saved successfully to {path.absolute()}"
    except Exception as e:
        return f"Error saving report: {str(e)}"


def knowledge_base_stats(settings: Settings) -> tuple[int, str | None]:
    """Chunk count and ISO mtime of index/chunks.pkl if present."""
    chunks_path = _PROJECT_ROOT / "index" / "chunks.pkl"
    if not chunks_path.is_file():
        return 0, None
    try:
        mtime = chunks_path.stat().st_mtime
        from datetime import datetime

        updated = datetime.fromtimestamp(mtime).isoformat(timespec="seconds")
    except OSError:
        updated = None
    try:
        with open(chunks_path, "rb") as f:
            chunks = pickle.load(f)
        n = len(chunks) if isinstance(chunks, list) else 0
    except Exception:
        n = 0
    return n, updated


def list_saved_reports(settings: Settings) -> tuple[str, list[str]]:
    """Resolved output dir and basenames of *.md files."""
    out = resolved_output_dir(settings)
    if not out.is_dir():
        return str(out), []
    names = sorted(p.name for p in out.iterdir() if p.is_file() and p.suffix.lower() == ".md")
    return str(out.resolve()), names
