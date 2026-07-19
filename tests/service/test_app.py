from __future__ import annotations

import json
from typing import Any

import pytest
from fastapi.testclient import TestClient


def test_live_is_downstream_independent_and_ready_is_dependency_aware(
    client: TestClient,
) -> None:
    live = client.get("/health/live")
    ready = client.get("/health/ready")
    assert live.status_code == 200
    assert live.json() == {"status": "alive"}
    assert ready.status_code == 200
    assert ready.json() == {"status": "ready"}
    assert "no-store" in live.headers["cache-control"]


def test_unready_dependency_is_generic_503(
    service_settings: Any,
    fake_runtime: Any,
) -> None:
    from customs_ai_agent.service.app import create_app

    def unavailable() -> object:
        raise RuntimeError("private dependency details")

    app = create_app(
        service_settings,
        runtime_factory=lambda _settings: fake_runtime,
        readiness_probe_factory=lambda _settings: unavailable,
    )
    with TestClient(app, raise_server_exceptions=False) as local_client:
        response = local_client.get("/health/ready")
    assert response.status_code == 503
    assert response.json()["error"]["code"] == "dependency_unready"
    assert "private dependency details" not in response.text


def test_metadata_and_metrics_are_authenticated_and_safe(
    client: TestClient,
    auth_headers: dict[str, str],
) -> None:
    assert client.get("/internal/agent/v1/metadata").status_code == 401
    metadata = client.get("/internal/agent/v1/metadata", headers=auth_headers)
    assert metadata.status_code == 200
    payload = metadata.json()
    assert payload["capabilities"] == ["text"]
    assert payload["service_api_version"] == "internal-agent-service/v1"
    encoded = metadata.text.casefold()
    assert "bearer" not in encoded
    assert "api_key" not in encoded

    metrics = client.get("/internal/agent/v1/metrics", headers=auth_headers)
    assert metrics.status_code == 200
    assert set(metrics.json()) == {
        "duration_ms_max",
        "duration_ms_sum",
        "inflight",
        "model_calls_total",
        "review_completed_total",
        "review_failed_total",
        "review_rejected_total",
        "review_replayed_total",
        "review_requests_total",
        "review_timeouts_total",
    }


@pytest.mark.parametrize(
    ("headers", "body", "status", "code"),
    [
        (
            {"Content-Type": "text/plain"},
            b"{}",
            415,
            "unsupported_media_type",
        ),
        (
            {"Content-Encoding": "gzip", "Content-Type": "application/json"},
            b"{}",
            415,
            "content_encoding_unsupported",
        ),
        (
            {"Content-Type": "application/json"},
            b'{"case_id":"a","case_id":"b"}',
            422,
            "request_invalid",
        ),
    ],
)
def test_transport_rejects_unsafe_wire_before_runtime(
    client: TestClient,
    auth_headers: dict[str, str],
    fake_runtime: Any,
    headers: dict[str, str],
    body: bytes,
    status: int,
    code: str,
) -> None:
    response = client.post(
        "/internal/agent/v1/reviews",
        headers={**auth_headers, **headers},
        content=body,
    )
    assert response.status_code == status
    assert response.json()["error"]["code"] == code
    assert fake_runtime.calls == 0


def test_transport_rejects_oversized_body_before_runtime(
    client: TestClient,
    auth_headers: dict[str, str],
    fake_runtime: Any,
) -> None:
    response = client.post(
        "/internal/agent/v1/reviews",
        headers={**auth_headers, "Content-Type": "application/json"},
        content=b"{" + (b" " * (2 * 1024 * 1024)) + b"}",
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"
    assert fake_runtime.calls == 0


def test_chunked_json_body_is_bounded_and_accepted(
    client: TestClient,
    auth_headers: dict[str, str],
    valid_payload: dict[str, Any],
    fake_runtime: Any,
) -> None:
    encoded = json.dumps(valid_payload, separators=(",", ":")).encode()

    def chunks() -> Any:
        yield encoded[:17]
        yield encoded[17:]

    response = client.post(
        "/internal/agent/v1/reviews",
        headers={**auth_headers, "Content-Type": "application/json"},
        content=chunks(),
    )
    assert response.status_code == 200
    assert fake_runtime.calls == 1


@pytest.mark.parametrize("visual_field", ["image_pages", "pdf_pages"])
def test_visual_keys_are_rejected_before_runtime(
    client: TestClient,
    auth_headers: dict[str, str],
    valid_payload: dict[str, Any],
    fake_runtime: Any,
    visual_field: str,
) -> None:
    valid_payload["documents"][0][visual_field] = None
    response = client.post(
        "/internal/agent/v1/reviews",
        headers=auth_headers,
        json=valid_payload,
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "visual_input_not_supported"
    assert fake_runtime.calls == 0


@pytest.mark.parametrize("missing", ["review_run_id", "review_as_of"])
def test_reproducibility_fields_are_service_required(
    client: TestClient,
    auth_headers: dict[str, str],
    valid_payload: dict[str, Any],
    fake_runtime: Any,
    missing: str,
) -> None:
    valid_payload.pop(missing)
    response = client.post(
        "/internal/agent/v1/reviews",
        headers=auth_headers,
        json=valid_payload,
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "required_field_missing"
    assert fake_runtime.calls == 0


def test_success_and_process_local_canonical_replay(
    client: TestClient,
    auth_headers: dict[str, str],
    valid_payload: dict[str, Any],
    fake_runtime: Any,
) -> None:
    first = client.post(
        "/internal/agent/v1/reviews",
        headers={**auth_headers, "Content-Type": "application/json"},
        content=json.dumps(valid_payload, separators=(",", ":")).encode(),
    )
    second = client.post(
        "/internal/agent/v1/reviews",
        headers={**auth_headers, "Content-Type": "application/json"},
        content=json.dumps(valid_payload, indent=2, sort_keys=True).encode(),
    )
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json()
    assert first.headers["x-idempotent-replay"] == "false"
    assert second.headers["x-idempotent-replay"] == "true"
    assert fake_runtime.calls == 1

    metrics = client.get("/internal/agent/v1/metrics", headers=auth_headers).json()
    assert metrics["review_requests_total"] == 2
    assert metrics["review_replayed_total"] == 1
    assert metrics["review_completed_total"] == 1
    assert metrics["model_calls_total"] == 1


def test_same_run_id_with_different_request_is_conflict(
    client: TestClient,
    auth_headers: dict[str, str],
    valid_payload: dict[str, Any],
    fake_runtime: Any,
) -> None:
    first = client.post("/internal/agent/v1/reviews", headers=auth_headers, json=valid_payload)
    valid_payload["documents"][0]["content_text"] += "\nChanged"
    conflict = client.post(
        "/internal/agent/v1/reviews",
        headers=auth_headers,
        json=valid_payload,
    )
    assert first.status_code == 200
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "idempotency_conflict"
    assert fake_runtime.calls == 1
