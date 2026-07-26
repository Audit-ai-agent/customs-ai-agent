from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from customs_ai_agent.runtime import (
        AgentRuntime,
        AgentRuntimeError,
        RuntimeConfig,
        RuntimeConfigurationError,
        RuntimeConflictError,
        RuntimeExecutionError,
        RuntimeInputError,
    )


def __getattr__(name: str) -> Any:
    """Keep isolated OCR workers independent from service/control-plane imports."""

    if name not in __all__:
        raise AttributeError(name)
    from customs_ai_agent import runtime

    return getattr(runtime, name)


__all__ = [
    "AgentRuntime",
    "AgentRuntimeError",
    "RuntimeConfig",
    "RuntimeConfigurationError",
    "RuntimeConflictError",
    "RuntimeExecutionError",
    "RuntimeInputError",
]
