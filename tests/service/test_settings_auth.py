from __future__ import annotations

import hashlib
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from customs_ai_agent.service.auth import bearer_authorized
from customs_ai_agent.service.settings import ServiceConfigurationError, ServiceSettings
from tests.service.conftest import TEST_TOKEN


def test_service_settings_require_high_entropy_bearer() -> None:
    with pytest.raises(ServiceConfigurationError, match="service_bearer_token_weak"):
        ServiceSettings(
            bearer_token=SecretStr("short"),
            evidence_base_url="http://127.0.0.1:8010",
        )
    with pytest.raises(ServiceConfigurationError, match="service_bearer_token_weak"):
        ServiceSettings(
            bearer_token=SecretStr("a" * 64),
            evidence_base_url="http://127.0.0.1:8010",
        )


def test_production_settings_require_exact_build_identity() -> None:
    source = {
        "ANTHROPIC_API_KEY": "not-read-by-service-settings",
        "CUSTOMS_AGENT_ENVIRONMENT": "production",
        "CUSTOMS_AGENT_SERVICE_BEARER_TOKEN": TEST_TOKEN,
        "EVIDENCE_SERVICE_BASE_URL": "https://evidence.example.test",
        "EVIDENCE_SERVICE_ALLOWED_ORIGINS": "https://evidence.example.test",
    }
    with pytest.raises(ServiceConfigurationError, match="customs_agent_build_sha_invalid"):
        ServiceSettings.from_environment(source)
    source["CUSTOMS_AGENT_BUILD_SHA"] = "1" * 40
    source["CUSTOMS_AGENT_WHEEL_SHA256"] = "2" * 64
    settings = ServiceSettings.from_environment(source)
    assert settings.build_sha == "1" * 40
    assert settings.wheel_sha256 == "2" * 64


def test_bearer_comparison_hashes_both_sides_before_constant_time_compare(
    monkeypatch: pytest.MonkeyPatch,
    service_settings: ServiceSettings,
) -> None:
    observed: list[tuple[bytes, bytes]] = []

    def compare(left: bytes, right: bytes) -> bool:
        observed.append((left, right))
        return left == right

    monkeypatch.setattr("customs_ai_agent.service.auth.hmac.compare_digest", compare)
    assert bearer_authorized(f"Bearer {TEST_TOKEN}", service_settings) is True
    assert bearer_authorized("Bearer wrong", service_settings) is False
    assert all(len(left) == len(right) == hashlib.sha256().digest_size for left, right in observed)


@pytest.mark.parametrize(
    ("headers", "expected_status", "expected_code"),
    [
        ({}, 401, "authentication_required"),
        ({"Authorization": "Bearer wrong"}, 403, "authentication_invalid"),
        (
            {"Authorization": (f"Bearer {TEST_TOKEN}, Bearer {TEST_TOKEN}")},
            403,
            "authentication_invalid",
        ),
    ],
)
def test_review_authentication_precedes_body_processing(
    client: TestClient,
    fake_runtime: Any,
    headers: dict[str, str],
    expected_status: int,
    expected_code: str,
) -> None:
    response = client.post(
        "/internal/agent/v1/reviews",
        headers=headers,
        content=b"private-document-text",
    )
    assert response.status_code == expected_status
    assert response.json()["error"]["code"] == expected_code
    assert fake_runtime.calls == 0
    assert "private-document-text" not in response.text
