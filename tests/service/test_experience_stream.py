from __future__ import annotations

import json
import time
from typing import Any, Callable

from fastapi.testclient import TestClient

from customs_ai_agent.service.app import create_app
from tests.service.conftest import FakeResult, FakeRuntime, settings_with


class ObservedRuntime(FakeRuntime):
    def review_experience_stream_json(
        self,
        payload: bytes,
        callback: Callable[[dict[str, Any]], None],
    ) -> FakeResult:
        self.calls += 1
        self.payloads.append(payload)
        callback(
            {
                "stage_code": "request_validation",
                "state": "started",
                "duration_ms": None,
                "safe_counts": {},
            }
        )
        callback(
            {
                "stage_code": "request_validation",
                "state": "completed",
                "duration_ms": 1,
                "safe_counts": {},
            }
        )
        callback(
            {
                "stage_code": "document_analysis",
                "state": "started",
                "duration_ms": None,
                "safe_counts": {},
            }
        )
        time.sleep(0.01)
        callback(
            {
                "stage_code": "document_analysis",
                "state": "completed",
                "duration_ms": 10,
                "safe_counts": {"documents": 1},
            }
        )
        callback(
            {
                "stage_code": "completed",
                "state": "completed",
                "duration_ms": None,
                "safe_counts": {"documents": 1},
            }
        )
        return FakeResult("run-service-1")


def test_review_stream_forwards_only_runtime_observed_stages_and_replays_them(
    service_settings,
    valid_payload,
    auth_headers,
) -> None:
    runtime = ObservedRuntime()
    settings = settings_with(
        service_settings,
        enhanced_agent_experience=True,
    )
    app = create_app(
        settings,
        runtime_factory=lambda _settings: runtime,
        readiness_probe_factory=lambda _settings: lambda: object(),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        responses = []
        for _attempt in range(2):
            with client.stream(
                "POST",
                "/internal/agent/v3/reviews/stream",
                headers=auth_headers,
                json=valid_payload,
            ) as response:
                assert response.status_code == 200
                assert response.headers["content-type"].startswith("application/x-ndjson")
                responses.append([json.loads(line) for line in response.iter_lines()])

    assert runtime.calls == 1
    for lines in responses:
        stage_lines = [line for line in lines if line["type"] == "stage_event"]
        assert [line["event"]["stage_code"] for line in stage_lines] == [
            "request_validation",
            "request_validation",
            "document_analysis",
            "document_analysis",
            "completed",
        ]
        assert [line["event"]["sequence"] for line in stage_lines] == list(range(len(stage_lines)))
        assert lines[-1]["type"] == "result"
        assert not any(
            line["event"]["duration_ms"] == 0
            for line in stage_lines
            if line["event"]["state"] == "completed"
        )
        active: set[str] = set()
        for line in stage_lines:
            event = line["event"]
            if event["stage_code"] == "completed":
                continue
            if event["state"] == "started":
                active.add(event["stage_code"])
                continue
            assert event["stage_code"] in active
            active.remove(event["stage_code"])
        assert not active
