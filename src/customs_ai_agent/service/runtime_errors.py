"""Small error vocabulary shared by the two local Agent profiles.

The local service needs only these content-free categories to map profile
failures to HTTP responses.
"""

from __future__ import annotations


class AgentRuntimeError(RuntimeError):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class RuntimeConfigurationError(AgentRuntimeError):
    pass


class RuntimeInputError(AgentRuntimeError):
    pass


class RuntimeExecutionError(AgentRuntimeError):
    pass


__all__ = [
    "AgentRuntimeError",
    "RuntimeConfigurationError",
    "RuntimeExecutionError",
    "RuntimeInputError",
]
