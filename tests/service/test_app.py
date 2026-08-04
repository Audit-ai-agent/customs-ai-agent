from __future__ import annotations

import json
from datetime import date
from types import SimpleNamespace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr


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
    assert "x-agent-execution-state" not in response.headers
    assert "private dependency details" not in response.text


def test_unreleased_packaged_data_lock_keeps_service_live_but_never_calls_claude(
    monkeypatch: pytest.MonkeyPatch,
    service_settings: Any,
    auth_headers: dict[str, str],
    valid_payload: dict[str, Any],
) -> None:
    from customs_ai_agent.service.app import create_app

    settings = type(service_settings)(
        bearer_token=service_settings.bearer_token,
        evidence_base_url=service_settings.evidence_base_url,
        evidence_allowed_origins=service_settings.evidence_allowed_origins,
        evidence_service_token=SecretStr("EvidenceBearer_abcdefghijklmnopqrstuvwxyz0123456789"),
        environment="test",
        readiness_ttl_seconds=0.1,
    )
    monkeypatch.setattr(
        "customs_ai_agent.runtime.ClaudeDocumentAnalyzer.from_settings",
        lambda *_args, **_kwargs: pytest.fail("Claude client must not be constructed"),
    )
    monkeypatch.setattr(
        "customs_ai_agent.adapters.retrieval.urlopen",
        lambda *_args, **_kwargs: pytest.fail("Data network must not run while lock is unreleased"),
    )
    app = create_app(settings)
    with TestClient(app, raise_server_exceptions=False) as local_client:
        assert local_client.get("/health/live").status_code == 200
        assert local_client.get("/health/ready").status_code == 503
        review = local_client.post(
            "/internal/agent/v1/reviews",
            headers=auth_headers,
            json=valid_payload,
        )
    assert review.status_code == 503
    assert review.json()["error"]["code"] == "dependency_unready"
    assert review.headers["x-agent-execution-state"] == "not-started"


def test_all_execution_endpoints_mark_only_admission_dependency_rejection_not_started(
    monkeypatch: pytest.MonkeyPatch,
    service_settings: Any,
    fake_runtime: Any,
    auth_headers: dict[str, str],
    valid_payload: dict[str, Any],
) -> None:
    import customs_ai_agent.service.app as app_module

    def unavailable() -> object:
        raise RuntimeError("dependency not ready")

    monkeypatch.setattr(
        app_module,
        "inspect_v3_auxiliary_request",
        lambda *_args, **_kwargs: SimpleNamespace(
            replay_id="admission-test",
            canonical_sha256="a" * 64,
        ),
    )
    app = app_module.create_app(
        service_settings,
        runtime_factory=lambda _settings: fake_runtime,
        readiness_probe_factory=lambda _settings: unavailable,
    )
    requests = (
        ("/internal/agent/v1/reviews", valid_payload),
        ("/internal/agent/v2/reviews", valid_payload),
        ("/internal/agent/v2/revisions", {}),
        ("/internal/agent/v2/answers", {}),
    )
    with TestClient(app, raise_server_exceptions=False) as local_client:
        responses = [
            local_client.post(path, headers=auth_headers, json=payload)
            for path, payload in requests
        ]

    assert fake_runtime.calls == 0
    for response in responses:
        assert response.status_code == 503
        assert response.json()["error"]["code"] == "dependency_unready"
        assert response.headers["x-agent-execution-state"] == "not-started"
        assert response.json()["error"]["request_id"] == response.headers["x-request-id"]


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

    metadata_v2 = client.get("/internal/agent/v2/metadata", headers=auth_headers)
    assert metadata_v2.status_code == 200
    metadata_v2_payload = metadata_v2.json()
    assert metadata_v2_payload["transport_limits"] == {
        "max_body_bytes": 2 * 1024 * 1024,
        "max_documents": 6,
        "max_text_characters": 500_000,
        "text_only": True,
        "empty_visual_placeholders_accepted": True,
    }
    agentic = metadata_v2_payload["agentic_limits"]
    assert agentic == {
        "max_cycles": 2,
        "max_tool_calls": 6,
        "ocr_enabled": False,
        "planner_mode": "shadow",
        "active_execution_authorized_by_release": False,
        "active_tool_handlers_registered": True,
        "trusted_assertion_verifier_configured": False,
        "effective_execution_mode": "shadow",
    }
    from customs_ai_agent.schemas.v3 import RuntimeReleaseIdentityV1

    runtime_release = RuntimeReleaseIdentityV1.model_validate(
        metadata_v2_payload["runtime_release"]
    )
    assert runtime_release.release_profile_state == "unreleased"
    assert runtime_release.evidence_runtime_v3_sha256 is None
    assert runtime_release.kill_switch_active is True
    assert metadata_v2_payload["automatic_completion_release"]["production_admission"] == {
        "admitted": False,
        "reason_code": "evidence_runtime_release_unreleased",
    }
    assert metadata_v2_payload["evidence_provider_contract"] == {
        "contract": "evidence-runtime-release-provider-lock",
        "schema_version": "1.0",
        "provider_release": "evidence-api-v1.2.0",
        "provider_revision": "b13e8bf7238a61eb185efcf2e9cd768ad5a908e2",
        "provider_build_sha256": (
            "1591ebafb01aadc8d9c5e7fc91950e3bf5c7d44f88a7ff9e4065b7d57e2ed1b7"
        ),
        "provider_source_manifest_sha256": (
            "ddf271e0e0f8b3628917146d6f1a83bedb8e17d9dc98de30f50e12e0c8b1d8e0"
        ),
        "retrieval_schema_sha256": (
            "531b59ba4e86f78cef20a4fca052996efd8b88c48853ac12113ca76e69cdd69a"
        ),
        "readiness_schema_sha256": (
            "0a0b50fef627ecbbad954af5878ccb3630ecc51f618132f8cfc674c324467a69"
        ),
    }
    assert metadata_v2_payload["evidence_runtime_release"] == {"release_state": "unreleased"}
    assert metadata_v2_payload["evidence_runtime_release_v3"] == {"release_state": "unreleased"}
    assert metadata_v2_payload["schemas"]["service_openapi_sha256"] == (
        "c7b4f174e80e1594c092bc9e29d23fed87ca254e5968177eccca6d0219625859"
    )

    metrics = client.get("/internal/agent/v1/metrics", headers=auth_headers)
    assert metrics.status_code == 200
    assert set(metrics.json()) == {
        "duration_ms_max",
        "duration_ms_sum",
        "inflight",
        "model_calls_total",
        "replay_cache_entries",
        "replay_cache_evictions_total",
        "replay_cache_oversize_total",
        "replay_cache_result_bytes",
        "review_completed_total",
        "review_failed_total",
        "review_rejected_total",
        "review_replayed_total",
        "review_requests_total",
        "review_timeouts_total",
    }


def test_metadata_released_shapes_expose_full_nested_base_and_v3_identities(
    monkeypatch: pytest.MonkeyPatch,
    service_settings: Any,
) -> None:
    import customs_ai_agent.service.metadata as metadata_module
    from customs_ai_agent.contracts.evidence_runtime_release import (
        EvidenceRuntimeReleaseIdentity,
        EvidenceRuntimeReleaseLock,
    )
    from customs_ai_agent.contracts.evidence_v3 import (
        EvidenceRuntimeReleaseIdentityV3,
        EvidenceRuntimeReleaseLockV3,
    )

    provider = metadata_module.load_pinned_evidence_provider_runtime_identity()
    base = EvidenceRuntimeReleaseIdentity(
        contract="evidence-runtime-release",
        schema_version="2.0",
        retrieval_schema_version="1.0",
        provider_release=provider.provider_release,
        provider_revision=provider.provider_revision,
        provider_build_sha256=provider.provider_build_sha256,
        retrieval_schema_sha256=provider.retrieval_schema_sha256,
        provider_source_manifest_sha256=provider.provider_source_manifest_sha256,
        knowledge_generation_id="4" * 64,
        knowledge_generation_attestation_sha256="5" * 64,
        knowledge_source_manifest_sha256="6" * 64,
        model_manifest_sha256="7" * 64,
        index_config_sha256="8" * 64,
        evaluation_report_sha256="9" * 64,
        seal_attestation_sha256="a" * 64,
        knowledge_snapshot_id="knowledge-release-test",
        supported_review_as_of=date(2026, 7, 22),
        runtime_release_sha256="b" * 64,
    )
    v3 = EvidenceRuntimeReleaseIdentityV3(
        contract="evidence-runtime-release",
        schema_version="3.0",
        retrieval_schema_version="3.0",
        base_runtime_release=base,
        retrieval_schema_sha256=(
            "d20de13f4cc8db14a2cd3213e063916fd046922ae32687ac939dbb4019e485d6"
        ),
        readiness_schema_sha256=(
            "3488e98b9899ce6cbb375fad4120360a2d4a77d82117ef56f098514547a7d882"
        ),
        requirement_policy_schema_sha256=(
            "b1e9e5918d03c247d8503af861ee647451b0c2f378c1a9d2cc380f23ca26c226"
        ),
        requirement_policy_release="requirement-policy-v1",
        requirement_policy_manifest_sha256="c" * 64,
        coverage_matrix_sha256="d" * 64,
        requirement_policy_approval_sha256="e" * 64,
        reviewer_trust_store_sha256="f" * 64,
        runtime_release_v3_sha256="0" * 64,
    )
    monkeypatch.setattr(
        metadata_module,
        "load_evidence_runtime_release_lock",
        lambda: EvidenceRuntimeReleaseLock(release_state="released", identity=base),
    )
    monkeypatch.setattr(
        metadata_module,
        "load_evidence_runtime_release_lock_v3",
        lambda: EvidenceRuntimeReleaseLockV3(
            release_state="released",
            provider_runtime_identity=provider,
            identity=v3,
        ),
    )

    payload = metadata_module.build_metadata(service_settings)

    assert payload["evidence_runtime_release"] == {
        "release_state": "released",
        "identity": base.to_wire(),
    }
    assert payload["evidence_runtime_release_v3"] == {
        "release_state": "released",
        "identity": v3.to_wire(),
    }


def test_metadata_rejects_cross_lock_base_and_v3_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    service_settings: Any,
) -> None:
    import customs_ai_agent.service.metadata as metadata_module
    from customs_ai_agent.service.settings import ServiceConfigurationError

    monkeypatch.setattr(
        metadata_module,
        "load_evidence_runtime_release_lock",
        lambda: SimpleNamespace(identity=object()),
    )
    monkeypatch.setattr(
        metadata_module,
        "load_evidence_runtime_release_lock_v3",
        lambda: SimpleNamespace(
            identity=SimpleNamespace(base_runtime_release=object()),
        ),
    )

    with pytest.raises(
        ServiceConfigurationError,
        match="customs_agent_evidence_release_lock_mismatch",
    ):
        metadata_module.build_metadata(service_settings)


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
    assert "x-agent-execution-state" not in response.headers
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


def test_v3_accepts_backend_text_dump_with_only_empty_visual_defaults(
    client: TestClient,
    auth_headers: dict[str, str],
    valid_payload: dict[str, Any],
    fake_runtime: Any,
) -> None:
    valid_payload["documents"][0]["image_pages"] = []
    valid_payload["documents"][0]["pdf_pages"] = None
    response = client.post(
        "/internal/agent/v2/reviews",
        headers=auth_headers,
        json=valid_payload,
    )
    assert response.status_code == 200
    assert "x-agent-execution-state" not in response.headers
    assert fake_runtime.calls == 1


def test_runtime_dependency_failure_after_submit_never_claims_not_started(
    client: TestClient,
    auth_headers: dict[str, str],
    valid_payload: dict[str, Any],
    fake_runtime: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from customs_ai_agent.runtime import RuntimeConfigurationError

    calls = 0

    def fail_after_submit(_payload: bytes) -> object:
        nonlocal calls
        calls += 1
        raise RuntimeConfigurationError(
            "runtime_dependency_unready",
            "The runtime failed after admission.",
        )

    monkeypatch.setattr(fake_runtime, "review_v3_json", fail_after_submit)
    response = client.post(
        "/internal/agent/v2/reviews",
        headers=auth_headers,
        json=valid_payload,
    )

    assert calls == 1
    assert response.status_code == 503
    assert response.json()["error"] == {
        "code": "dependency_unready",
        "message": "A required service dependency is not ready.",
        "request_id": response.headers["x-request-id"],
        "retryable": True,
    }
    assert "x-agent-execution-state" not in response.headers


def test_service_http_error_rejects_invalid_not_started_claims() -> None:
    from customs_ai_agent.service.errors import ServiceHttpError

    with pytest.raises(ValueError, match="reserved"):
        ServiceHttpError(
            "dependency_unready",
            503,
            retryable=True,
            execution_state="started",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="reserved"):
        ServiceHttpError(
            "review_execution_failed",
            502,
            retryable=True,
            execution_state="not-started",
        )
    with pytest.raises(ValueError, match="reserved"):
        ServiceHttpError(
            "dependency_unready",
            503,
            retryable=False,
            execution_state="not-started",
        )


def test_v3_rejects_nonempty_backend_visual_default_before_runtime(
    client: TestClient,
    auth_headers: dict[str, str],
    valid_payload: dict[str, Any],
    fake_runtime: Any,
) -> None:
    valid_payload["documents"][0]["image_pages"] = [{}]
    valid_payload["documents"][0]["pdf_pages"] = None
    response = client.post(
        "/internal/agent/v2/reviews",
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
