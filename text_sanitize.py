"""Remove lone UTF-16 surrogate code units so UTF-8 / JSON serialization never fails."""

from __future__ import annotations

from typing import Any


def strip_surrogates(s: str) -> str:
    if not s:
        return s
    return "".join(c for c in s if not (0xD800 <= ord(c) <= 0xDFFF))


def strip_surrogates_deep(obj: Any) -> Any:
    if isinstance(obj, str):
        return strip_surrogates(obj)
    if isinstance(obj, dict):
        return {
            strip_surrogates(k) if isinstance(k, str) else k: strip_surrogates_deep(v)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [strip_surrogates_deep(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(strip_surrogates_deep(v) for v in obj)
    return obj
