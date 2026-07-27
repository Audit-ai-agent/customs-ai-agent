from __future__ import annotations

import time
from dataclasses import replace
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from customs_ai_agent.service.app import create_app
from customs_ai_agent.service.settings import ServiceSettings


TEST_TOKEN = "Abcdefghijklmnopqrstuvwxyz0123456789-._~+/=XYZ"


class FakeResult:
    def __init__(self, run_id: str = "run-service-1") -> None:
        self.run_id = run_id

    def model_dump(self, **_options: Any) -> dict[str, Any]:
        return {
            "model_calls": [{"request_id": "req_test00000001"}],
            "requires_human_review": True,
            "review_run_id": self.run_id,
            "schema_version": "2.0",
            "status": "needs_review",
        }


class FakeRuntime:
    def __init__(self, *, delay_seconds: float = 0.0) -> None:
        self.calls = 0
        self.close_calls = 0
        self.delay_seconds = delay_seconds
        self.payloads: list[bytes] = []

    def review_json(self, payload: bytes) -> FakeResult:
        self.calls += 1
        self.payloads.append(payload)
        if self.delay_seconds:
            time.sleep(self.delay_seconds)
        return FakeResult()

    def review_v3_json(self, payload: bytes) -> FakeResult:
        return self.review_json(payload)

    def revision_v3_json(self, payload: bytes) -> FakeResult:
        return self.review_json(payload)

    def answer_v3_json(self, payload: bytes) -> FakeResult:
        return self.review_json(payload)

    def assert_uat_capability_current(self, *, require_active: bool = False) -> object:
        del require_active
        return object()

    def close(self) -> None:
        self.close_calls += 1


@pytest.fixture
def service_settings() -> ServiceSettings:
    return ServiceSettings(
        bearer_token=SecretStr(TEST_TOKEN),
        evidence_base_url="http://127.0.0.1:8010",
        evidence_allowed_origins=frozenset({"http://127.0.0.1:8010"}),
        environment="test",
        readiness_ttl_seconds=0.1,
    )


@pytest.fixture
def valid_payload() -> dict[str, Any]:
    return {
        "case_id": "case-service-1",
        "documents": [
            {
                "content_text": "COMMERCIAL INVOICE\nInvoice No: INV-001",
                "document_id": "invoice-1",
                "filename": "invoice.txt",
            }
        ],
        "llm_mode": "llm",
        "review_as_of": "2026-07-10",
        "review_run_id": "run-service-1",
    }


@pytest.fixture
def fake_runtime() -> FakeRuntime:
    return FakeRuntime()


@pytest.fixture
def client(
    service_settings: ServiceSettings,
    fake_runtime: FakeRuntime,
) -> TestClient:
    app = create_app(
        service_settings,
        runtime_factory=lambda _settings: fake_runtime,
        readiness_probe_factory=lambda _settings: lambda: object(),
    )
    with TestClient(app, raise_server_exceptions=False) as active:
        yield active


@pytest.fixture
def auth_headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {TEST_TOKEN}"}


def settings_with(settings: ServiceSettings, **changes: Any) -> ServiceSettings:
    return replace(settings, **changes)
