"""In-memory live progress for the synchronous review call.

The review workflow runs in a worker thread while the event loop stays free,
so the Backend can poll a snapshot of stage progress during the call. This is
display-only state: it never feeds the review result, is bounded in size, and
losing it (restart) merely degrades the caller to "no live progress".
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from typing import Any

STAGE_ORDER: tuple[str, ...] = (
    "classify_documents",
    "extract_facts",
    "normalize_facts",
    "match_documents",
    "validate_documents",
    "reextract_documents",
    "retrieve_evidence",
    "generate_result",
)

_MAX_TRACKED_RUNS = 200


class ReviewProgressRegistry:
    """Thread-safe per-run stage progress, evicting the oldest runs."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._runs: OrderedDict[str, dict[str, Any]] = OrderedDict()

    def begin(self, review_run_id: str) -> None:
        with self._lock:
            self._runs.pop(review_run_id, None)
            self._runs[review_run_id] = {
                "status": "running",
                "stages": {stage: "pending" for stage in STAGE_ORDER},
            }
            while len(self._runs) > _MAX_TRACKED_RUNS:
                self._runs.popitem(last=False)

    def update(self, review_run_id: str, stage: str, status: str) -> None:
        with self._lock:
            entry = self._runs.get(review_run_id)
            if entry is None or stage not in entry["stages"]:
                return
            entry["stages"][stage] = status

    def finish(self, review_run_id: str, status: str = "completed") -> None:
        with self._lock:
            entry = self._runs.get(review_run_id)
            if entry is None:
                return
            entry["status"] = status
            if status == "completed":
                for stage, stage_status in entry["stages"].items():
                    if stage_status == "running":
                        entry["stages"][stage] = "completed"

    def snapshot(self, review_run_id: str) -> dict[str, Any] | None:
        with self._lock:
            entry = self._runs.get(review_run_id)
            if entry is None:
                return None
            return {
                "review_run_id": review_run_id,
                "status": entry["status"],
                "stages": [
                    {"stage": stage, "status": entry["stages"][stage]} for stage in STAGE_ORDER
                ],
            }


PROGRESS = ReviewProgressRegistry()
