
from __future__ import annotations

import warnings
from typing import Any, Optional, Sequence, Union

from langchain.agents import create_agent as lc_create_agent
from langchain_core.tools import BaseTool

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=r"^Pydantic serializer warnings:",
)


def create_agent(
    model: Any,
    tools: Sequence[Union[BaseTool, Any]],
    *,
    system_prompt: str,
    response_format: Any = None,
    checkpointer: Any = None,
    name: Optional[str] = None,
) -> Any:
    """Same role as `langchain.agents.create_agent` / legacy `create_react_agent` from homework README."""
    return lc_create_agent(
        model,
        tools=list(tools),
        system_prompt=system_prompt,
        response_format=response_format,
        checkpointer=checkpointer,
        name=name,
    )
