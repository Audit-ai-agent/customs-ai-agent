"""Redacted structured service logs."""

from __future__ import annotations

import json
import logging
from typing import Any


_ALLOWED = frozenset({"error_code", "request_id", "retryable", "status_code"})


def safe_log(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    payload: dict[str, Any] = {"event": event}
    payload.update(
        {
            key: value
            for key, value in fields.items()
            if key in _ALLOWED and isinstance(value, (str, int, float, bool))
        }
    )
    logger.log(level, json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True))


__all__ = ["safe_log"]
