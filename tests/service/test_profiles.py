from __future__ import annotations

import json
from dataclasses import replace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from customs_ai_agent.service.app import create_app
from customs_ai_agent.service.profiles import (
    ProfileEvidenceRetriever,
    ProfileRuntime,
    SERVICE_MODE_HEADER,
)
from customs_ai_agent.service.runtime_errors import RuntimeConfigurationError
from customs_ai_agent.service.settings import ServiceSettings


_TOKEN = "Abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJK"


def _settings(**changes: Any) -> ServiceSettings:
    return replace(
        ServiceSettings(
            bearer_token=SecretStr(_TOKEN),
            evidence_base_url="http://127.0.0.1:18083",
            evidence_service_token=SecretStr(_TOKEN),
        ),
        **changes,
    )


def test_default_profile_is_local_development() -> None:
    settings = ServiceSettings.from_environment(
        {
            "CUSTOMS_AGENT_SERVICE_BEARER_TOKEN": _TOKEN,
            "EVIDENCE_SERVICE_BASE_URL": "http://127.0.0.1:18083",
            "EVIDENCE_SERVICE_TOKEN": _TOKEN,
            "EVIDENCE_SERVICE_TOKEN_HEADER": "Authorization",
        }
    )
    assert settings.service_mode == "local-development"
    assert settings.port == 8090


def test_backend_profile_loader_mode_is_accepted() -> None:
    settings = ServiceSettings.from_environment(
        {
            "CUSTOMS_AGENT_SERVICE_BEARER_TOKEN": _TOKEN,
            "EVIDENCE_SERVICE_BASE_URL": "http://127.0.0.1:18083",
            "EVIDENCE_SERVICE_TOKEN": _TOKEN,
            "EVIDENCE_SERVICE_TOKEN_HEADER": "Authorization",
            "CUSTOMS_AGENT_SERVICE_MODE": "functional-uat",
        }
    )
    assert settings.service_mode == "functional-uat"


def test_profile_rejects_non_loopback_evidence() -> None:
    with pytest.raises(RuntimeConfigurationError) as raised:
        ProfileEvidenceRetriever(replace(_settings(), evidence_base_url="https://example.invalid"))
    assert raised.value.code == "evidence_base_url_invalid"


def test_review_uses_one_readiness_then_one_retrieve_and_stays_human_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def data_request(
        _self: ProfileEvidenceRetriever,
        path: str,
        *,
        method: str,
        payload: dict[str, Any] | None = None,
        review_run_id: str | None = None,
    ) -> dict[str, Any]:
        assert review_run_id == "run-profile-v3"
        calls.append((method, path))
        if path == ProfileEvidenceRetriever.readiness_path:
            assert method == "GET"
            assert payload is None
            return {"status": "ready"}
        assert path == ProfileEvidenceRetriever.retrieve_path
        assert method == "POST"
        assert payload is not None
        return {
            "schema_version": "3.0",
            "request_id": payload["request_id"],
            "status": "ok",
            "knowledge_snapshot_id": "dev-fixture",
            "results": [],
        }

    monkeypatch.setattr(ProfileEvidenceRetriever, "_request", data_request)
    runtime = ProfileRuntime(_settings())
    result = runtime.review_v3_json(
        json.dumps(
            {
                "case_id": "case-profile-v3",
                "documents": [
                    {
                        "document_id": "doc-profile-v3",
                        "filename": "invoice.txt",
                        "content_text": "COMMERCIAL INVOICE\nInvoice No: INV-003",
                    }
                ],
                "review_run_id": "run-profile-v3",
                "review_as_of": "2026-08-05",
                "trade_type": "import",
                "jurisdiction": "KR",
            }
        ).encode()
    )
    assert result.schema_version == "3.0"
    assert result.status == "needs_review"
    assert result.requires_human_review is True
    assert result.engine is not None and result.engine.kind == "deterministic"
    assert any(action.action_code == "human_review_required" for action in result.required_actions)
    assert calls == [
        ("GET", ProfileEvidenceRetriever.readiness_path),
        ("POST", ProfileEvidenceRetriever.retrieve_path),
    ]


def test_internal_api_requires_matching_mode_header() -> None:
    class Runtime:
        def review_v3_json(self, _payload: bytes) -> object:
            raise AssertionError("review is not part of this test")

        def close(self) -> None:
            return None

    app = create_app(
        _settings(),
        runtime_factory=lambda _settings: Runtime(),  # type: ignore[arg-type]
        readiness_probe_factory=lambda _settings: lambda: object(),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        missing = client.get(
            "/internal/agent/v2/metadata", headers={"Authorization": f"Bearer {_TOKEN}"}
        )
        accepted = client.get(
            "/internal/agent/v2/metadata",
            headers={
                "Authorization": f"Bearer {_TOKEN}",
                SERVICE_MODE_HEADER: "local-development",
            },
        )
    assert missing.status_code == 403
    assert missing.json()["error"]["code"] == "service_mode_mismatch"
    assert accepted.status_code == 200
    assert accepted.json()["human_review_only"] is True


def test_http_review_keeps_human_only_v3_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ProfileEvidenceRetriever,
        "retrieve",
        lambda _self, request: {
            "schema_version": "3.0",
            "request_id": request["request_id"],
            "status": "ok",
            "knowledge_snapshot_id": "dev-fixture",
            "results": [],
        },
    )
    app = create_app(_settings())
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/internal/agent/v2/reviews",
            headers={
                "Authorization": f"Bearer {_TOKEN}",
                SERVICE_MODE_HEADER: "local-development",
            },
            json={
                "case_id": "case-profile-http",
                "documents": [
                    {
                        "document_id": "doc-profile-http",
                        "filename": "invoice.txt",
                        "content_text": "COMMERCIAL INVOICE\nInvoice No: INV-004",
                    }
                ],
                "review_run_id": "run-profile-http",
                "review_as_of": "2026-08-05",
                "trade_type": "import",
                "jurisdiction": "KR",
            },
        )
    assert response.status_code == 200
    assert response.json()["status"] == "needs_review"
    assert response.json()["requires_human_review"] is True
