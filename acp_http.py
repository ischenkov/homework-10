
from __future__ import annotations

import httpx

from acp_sdk.client.utils import input_to_messages
from acp_sdk.models import Message, MessagePart, Run, RunMode
from acp_sdk.models.schemas import RunCreateRequest, RunCreateResponse

from text_sanitize import strip_surrogates_deep


async def acp_run_sync(
    base_url: str,
    agent_name: str,
    user_text: str,
    *,
    timeout_s: float = 600.0,
) -> Run:
    messages = input_to_messages(
        Message(parts=[MessagePart(content=user_text, content_type="text/plain")])
    )
    req = RunCreateRequest(
        agent_name=agent_name,
        input=messages,
        mode=RunMode.SYNC,
    )
    url = f"{base_url.rstrip('/')}/runs"
    # Scraped/LLM text may contain lone surrogates; json= / UTF-8 would raise otherwise.
    payload = strip_surrogates_deep(req.model_dump(mode="json"))
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s)) as client:
        response = await client.post(url, json=payload)
        response.raise_for_status()
        run = RunCreateResponse.model_validate(response.json())
    run.raise_for_status()
    return run
