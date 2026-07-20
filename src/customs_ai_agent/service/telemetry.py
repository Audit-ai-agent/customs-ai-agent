from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass, field
from typing import Any


_ALLOWED_LOG_FIELDS = frozenset(
    {
        "duration_ms",
        "error_code",
        "event",
        "inflight",
        "request_id",
        "retryable",
        "status_code",
    }
)


def safe_log(logger: logging.Logger, level: int, event: str, **fields: Any) -> None:
    payload: dict[str, Any] = {"event": event}
    for key, value in fields.items():
        if key in _ALLOWED_LOG_FIELDS and isinstance(value, (str, int, float, bool)):
            payload[key] = value
    logger.log(
        level,
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"), sort_keys=True),
    )


@dataclass(slots=True)
class ServiceMetrics:
    """Process-local, label-free metrics that cannot retain document identifiers."""

    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    review_requests_total: int = 0
    review_completed_total: int = 0
    review_failed_total: int = 0
    review_replayed_total: int = 0
    review_rejected_total: int = 0
    review_timeouts_total: int = 0
    replay_cache_evictions_total: int = 0
    replay_cache_oversize_total: int = 0
    replay_cache_entries: int = 0
    replay_cache_result_bytes: int = 0
    inflight: int = 0
    duration_ms_sum: int = 0
    duration_ms_max: int = 0
    model_calls_total: int = 0

    def increment(self, name: str, amount: int = 1) -> None:
        if name not in {
            "review_requests_total",
            "review_completed_total",
            "review_failed_total",
            "review_replayed_total",
            "review_rejected_total",
            "review_timeouts_total",
            "replay_cache_evictions_total",
            "replay_cache_oversize_total",
            "model_calls_total",
        }:
            raise ValueError("unsupported service metric")
        with self._lock:
            setattr(self, name, getattr(self, name) + amount)

    def set_replay_cache(self, *, entries: int, result_bytes: int) -> None:
        with self._lock:
            self.replay_cache_entries = max(0, entries)
            self.replay_cache_result_bytes = max(0, result_bytes)

    def begin(self) -> int:
        with self._lock:
            self.inflight += 1
            return self.inflight

    def finish(self, duration_ms: int) -> int:
        with self._lock:
            self.inflight = max(0, self.inflight - 1)
            self.duration_ms_sum += max(0, duration_ms)
            self.duration_ms_max = max(self.duration_ms_max, max(0, duration_ms))
            return self.inflight

    def snapshot(self) -> dict[str, int]:
        with self._lock:
            return {
                "duration_ms_max": self.duration_ms_max,
                "duration_ms_sum": self.duration_ms_sum,
                "inflight": self.inflight,
                "model_calls_total": self.model_calls_total,
                "replay_cache_entries": self.replay_cache_entries,
                "replay_cache_evictions_total": self.replay_cache_evictions_total,
                "replay_cache_oversize_total": self.replay_cache_oversize_total,
                "replay_cache_result_bytes": self.replay_cache_result_bytes,
                "review_completed_total": self.review_completed_total,
                "review_failed_total": self.review_failed_total,
                "review_rejected_total": self.review_rejected_total,
                "review_replayed_total": self.review_replayed_total,
                "review_requests_total": self.review_requests_total,
                "review_timeouts_total": self.review_timeouts_total,
            }


__all__ = ["ServiceMetrics", "safe_log"]
