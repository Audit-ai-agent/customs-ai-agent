from __future__ import annotations

import hashlib
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from customs_ai_agent.build_info import BuildManifest
from customs_ai_agent.service.auth import bearer_authorized
from customs_ai_agent.service.metadata import build_metadata
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
        "EVIDENCE_SERVICE_TOKEN": "EvidenceBearer_abcdefghijklmnopqrstuvwxyz0123456789",
    }
    with pytest.raises(ServiceConfigurationError, match="customs_agent_build_sha_invalid"):
        ServiceSettings.from_environment(source)
    source["CUSTOMS_AGENT_BUILD_SHA"] = "1" * 40
    source["CUSTOMS_AGENT_WHEEL_SHA256"] = "2" * 64
    settings = ServiceSettings.from_environment(source)
    assert settings.build_sha == "1" * 40
    assert settings.wheel_sha256 == "2" * 64
    assert settings.evidence_token_header == "Authorization"


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("CUSTOMS_AGENT_MAX_DOCUMENTS", "5"),
        ("CUSTOMS_AGENT_MAX_TEXT_CHARACTERS", "499999"),
        ("CUSTOMS_AGENT_MAX_BODY_BYTES", str(2 * 1024 * 1024 - 1)),
    ],
)
def test_production_transport_limits_are_contract_fixed(name: str, value: str) -> None:
    source = {
        "CUSTOMS_AGENT_ENVIRONMENT": "production",
        "CUSTOMS_AGENT_BUILD_SHA": "1" * 40,
        "CUSTOMS_AGENT_WHEEL_SHA256": "2" * 64,
        "CUSTOMS_AGENT_SERVICE_BEARER_TOKEN": TEST_TOKEN,
        "EVIDENCE_SERVICE_BASE_URL": "https://evidence.example.test",
        "EVIDENCE_SERVICE_ALLOWED_ORIGINS": "https://evidence.example.test",
        "EVIDENCE_SERVICE_TOKEN": "EvidenceBearer_abcdefghijklmnopqrstuvwxyz0123456789",
        name: value,
    }
    with pytest.raises(ServiceConfigurationError, match="production_transport_limits_mismatch"):
        ServiceSettings.from_environment(source)


def test_production_settings_require_evidence_bearer_auth() -> None:
    source = {
        "CUSTOMS_AGENT_ENVIRONMENT": "production",
        "CUSTOMS_AGENT_BUILD_SHA": "1" * 40,
        "CUSTOMS_AGENT_WHEEL_SHA256": "2" * 64,
        "CUSTOMS_AGENT_SERVICE_BEARER_TOKEN": TEST_TOKEN,
        "EVIDENCE_SERVICE_BASE_URL": "https://evidence.example.test",
        "EVIDENCE_SERVICE_ALLOWED_ORIGINS": "https://evidence.example.test",
        "EVIDENCE_SERVICE_TOKEN": "EvidenceBearer_abcdefghijklmnopqrstuvwxyz0123456789",
        "EVIDENCE_SERVICE_TOKEN_HEADER": "X-Demo-Token",
    }
    with pytest.raises(
        ServiceConfigurationError,
        match="production_evidence_auth_must_use_bearer",
    ):
        ServiceSettings.from_environment(source)

    source["EVIDENCE_SERVICE_TOKEN_HEADER"] = "Authorization"
    source.pop("EVIDENCE_SERVICE_TOKEN")
    with pytest.raises(ServiceConfigurationError, match="evidence_service_token_missing"):
        ServiceSettings.from_environment(source)


@pytest.mark.parametrize(
    ("name", "value", "error"),
    [
        (
            "CUSTOMS_AGENT_REPLAY_MAX_BYTES",
            str(256 * 1024 * 1024 + 1),
            "customs_agent_replay_max_bytes_invalid",
        ),
        (
            "CUSTOMS_AGENT_REPLAY_MAX_ENTRIES",
            "10001",
            "customs_agent_replay_max_entries_invalid",
        ),
    ],
)
def test_replay_cache_limits_are_bounded_for_one_gib_container(
    name: str,
    value: str,
    error: str,
) -> None:
    source = {
        "CUSTOMS_AGENT_SERVICE_BEARER_TOKEN": TEST_TOKEN,
        "EVIDENCE_SERVICE_BASE_URL": "http://127.0.0.1:8010",
        name: value,
    }
    with pytest.raises(ServiceConfigurationError, match=error):
        ServiceSettings.from_environment(source)


def test_readiness_timeout_can_cover_a_bounded_slow_dependency() -> None:
    source = {
        "CUSTOMS_AGENT_SERVICE_BEARER_TOKEN": TEST_TOKEN,
        "EVIDENCE_SERVICE_BASE_URL": "http://127.0.0.1:8010",
        "CUSTOMS_AGENT_READINESS_TIMEOUT_SECONDS": "90",
    }

    assert ServiceSettings.from_environment(source).readiness_timeout_seconds == 90
    source["CUSTOMS_AGENT_READINESS_TIMEOUT_SECONDS"] = "121"
    with pytest.raises(
        ServiceConfigurationError,
        match="customs_agent_readiness_timeout_seconds_invalid",
    ):
        ServiceSettings.from_environment(source)


@pytest.mark.parametrize(
    ("enhanced", "adaptive"),
    [
        (True, False),
        (False, True),
        (True, True),
    ],
)
@pytest.mark.parametrize(
    "host",
    [
        "0.0.0.0",
        "localhost",
        "::1",
        "127.0.0.2",
    ],
)
def test_local_uat_enhanced_features_require_literal_ipv4_loopback_bind(
    enhanced: bool,
    adaptive: bool,
    host: str,
) -> None:
    source = {
        "CUSTOMS_AGENT_SERVICE_BEARER_TOKEN": TEST_TOKEN,
        "EVIDENCE_SERVICE_BASE_URL": "http://127.0.0.1:8010",
        "CUSTOMS_AGENT_HOST": host,
        "CUSTOMS_ENHANCED_AGENT_EXPERIENCE": str(enhanced),
        "CUSTOMS_UAT_ADAPTIVE_ACTIVE": str(adaptive),
    }

    with pytest.raises(
        ServiceConfigurationError,
        match="local_uat_enhanced_bind_host_not_loopback",
    ):
        ServiceSettings.from_environment(source)


def test_literal_ipv4_loopback_accepts_local_uat_enhanced_features() -> None:
    source = {
        "CUSTOMS_AGENT_SERVICE_BEARER_TOKEN": TEST_TOKEN,
        "EVIDENCE_SERVICE_BASE_URL": "http://127.0.0.1:8010",
        "CUSTOMS_AGENT_HOST": "127.0.0.1",
        "CUSTOMS_ENHANCED_AGENT_EXPERIENCE": "true",
        "CUSTOMS_UAT_ADAPTIVE_ACTIVE": "true",
    }

    settings = ServiceSettings.from_environment(source)

    assert settings.host == "127.0.0.1"
    assert settings.enhanced_agent_experience is True
    assert settings.uat_adaptive_active is True


@pytest.mark.parametrize("host", ["0.0.0.0", "localhost", "::1"])
def test_non_loopback_bind_remains_compatible_when_uat_features_are_disabled(
    host: str,
) -> None:
    source = {
        "CUSTOMS_AGENT_SERVICE_BEARER_TOKEN": TEST_TOKEN,
        "EVIDENCE_SERVICE_BASE_URL": "http://127.0.0.1:8010",
        "CUSTOMS_AGENT_HOST": host,
        "CUSTOMS_ENHANCED_AGENT_EXPERIENCE": "false",
        "CUSTOMS_UAT_ADAPTIVE_ACTIVE": "false",
    }

    settings = ServiceSettings.from_environment(source)

    assert settings.host == host
    assert settings.enhanced_agent_experience is False
    assert settings.uat_adaptive_active is False


def test_metadata_rejects_build_sha_not_bound_to_packaged_manifest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = ServiceSettings(
        bearer_token=SecretStr(TEST_TOKEN),
        evidence_base_url="https://evidence.example.test",
        evidence_allowed_origins=frozenset({"https://evidence.example.test"}),
        evidence_service_token=SecretStr("EvidenceBearer_abcdefghijklmnopqrstuvwxyz0123456789"),
        environment="production",
        build_sha="1" * 40,
        wheel_sha256="2" * 64,
    )
    monkeypatch.setattr(
        "customs_ai_agent.service.metadata.load_packaged_build_manifest",
        lambda **_options: BuildManifest(
            git_sha="3" * 40,
            source_snapshot_sha256="4" * 64,
        ),
    )
    with pytest.raises(ServiceConfigurationError, match="customs_agent_build_sha_mismatch"):
        build_metadata(settings)


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
