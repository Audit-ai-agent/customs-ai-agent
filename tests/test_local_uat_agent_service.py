from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

import customs_ai_agent.adapters.retrieval_v3 as production_retrieval_v3
from customs_ai_agent.runtime import RuntimeExecutionError
from customs_ai_agent.service.settings import ServiceSettings
from tests.test_evidence_v3_consumer import _identity
from tests.test_v3_conformance_runner import _result
from tools.agent_execution_inspector.evidence_v3_conformance import (
    LoopbackEvidenceV3ConformanceRetriever,
    LoopbackEvidenceV3DiagnosticRetriever,
)
from tools.local_uat_agent_service.app import (
    LOCAL_UAT_ACKNOWLEDGEMENT,
    LocalUatConfigurationError,
    LocalUatDiagnosticRuntime,
    _evidence_timeout_seconds,
    _readiness_policy,
    LocalUatManifest,
    _compose_local_uat_app,
    create_local_uat_app,
    load_local_uat_manifest,
    local_uat_service_settings,
)


SERVICE_TOKEN = "Abcdefghijklmnopqrstuvwxyz0123456789-._~+/=XYZ"
EVIDENCE_TOKEN = "EvidenceBearer_abcdefghijklmnopqrstuvwxyz0123456789"


def _manifest_payload(**updates: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema_version": "customs-agent-local-uat-manifest/v1",
        "acknowledgement": LOCAL_UAT_ACKNOWLEDGEMENT,
        "diagnostic_only": True,
        "text_only": True,
        "mock_fallback_allowed": False,
        "auto_completion_allowed": False,
        "evidence_mode": "synthetic-v3",
        "evidence_base_url": "http://127.0.0.1:18091",
        "evidence_runtime_identity": _identity().to_wire(),
    }
    payload.update(updates)
    return payload


def _load_manifest(tmp_path: Path, **updates: Any) -> LocalUatManifest:
    path = tmp_path / "uat-manifest.json"
    path.write_text(json.dumps(_manifest_payload(**updates)), encoding="utf-8")
    return load_local_uat_manifest(path)


def _settings(manifest: LocalUatManifest, *, environment: str = "test") -> ServiceSettings:
    return ServiceSettings(
        bearer_token=SecretStr(SERVICE_TOKEN),
        evidence_base_url=manifest.evidence_base_url,
        evidence_allowed_origins=frozenset({manifest.evidence_base_url}),
        evidence_service_token=SecretStr(EVIDENCE_TOKEN),
        environment=environment,
        build_sha="1" * 40 if environment == "production" else "unknown",
        wheel_sha256="2" * 64 if environment == "production" else "unknown",
        host="127.0.0.1",
        max_concurrency=1,
        readiness_ttl_seconds=0.1,
    )


class _Dump:
    def __init__(self, payload: dict[str, Any]) -> None:
        self.payload = payload

    def model_dump(self, **_kwargs: Any) -> dict[str, Any]:
        return self.payload


class _RouteRuntime:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def review_json(self, _payload: bytes) -> _Dump:
        raise AssertionError("V1 must not be routed")

    def review_v3_json(self, _payload: bytes) -> _Dump:
        self.calls.append("review")
        return _Dump(
            {
                "schema_version": "3.0",
                "model_calls": [],
                "assurance_level": "diagnostic",
                "review_disposition": "human_review_required",
            }
        )

    def revision_v3_json(self, _payload: bytes) -> _Dump:
        self.calls.append("revision")
        return _Dump({"operation": "revision", "diagnostic_only": True})

    def answer_v3_json(self, _payload: bytes) -> _Dump:
        self.calls.append("answer")
        return _Dump({"operation": "answer", "diagnostic_only": True})

    def close(self) -> None:
        return None


def _app(
    manifest: LocalUatManifest,
    runtime: Any,
    *,
    probe: Any = lambda: object(),
) -> Any:
    return _compose_local_uat_app(
        manifest=manifest,
        settings=_settings(manifest),
        acknowledge_local_uat=True,
        runtime=runtime,
        readiness_probe=probe,
        model_id="claude-sonnet-5",
    )


def test_manifest_requires_explicit_non_release_policy_and_literal_loopback(
    tmp_path: Path,
) -> None:
    manifest = _load_manifest(tmp_path)
    assert manifest.evidence_mode == "synthetic-v3"
    assert manifest.evidence_base_url == "http://127.0.0.1:18091"
    assert len(manifest.manifest_sha256) == 64

    for updates, code in (
        ({"acknowledgement": "yes"}, "uat_manifest_policy_invalid"),
        ({"mock_fallback_allowed": True}, "uat_manifest_policy_invalid"),
        ({"auto_completion_allowed": True}, "uat_manifest_policy_invalid"),
        ({"evidence_mode": "preserved-3law"}, "uat_manifest_policy_invalid"),
        ({"evidence_mode": "production-v3"}, "uat_manifest_policy_invalid"),
        ({"evidence_base_url": "http://localhost:18091"}, "not_loopback"),
        ({"evidence_base_url": "https://example.test"}, "not_loopback"),
    ):
        path = tmp_path / f"invalid-{len(code)}-{len(json.dumps(updates))}.json"
        path.write_text(json.dumps(_manifest_payload(**updates)), encoding="utf-8")
        with pytest.raises(LocalUatConfigurationError, match=code):
            load_local_uat_manifest(path)


@pytest.mark.parametrize(
    ("evidence_mode", "profile_kind"),
    [
        ("synthetic-v3", "synthetic_contract_control"),
        (
            "preserved-3law-v3-diagnostic",
            "preserved_three_law_rag_diagnostic",
        ),
    ],
)
def test_manifest_accepts_only_explicit_diagnostic_profiles_and_exposes_safe_metadata(
    tmp_path: Path,
    evidence_mode: str,
    profile_kind: str,
) -> None:
    manifest = _load_manifest(tmp_path, evidence_mode=evidence_mode)

    assert manifest.evidence_mode == evidence_mode
    assert manifest.evidence_runtime_identity == _identity()
    profile = manifest.safe_metadata()["evidence_profile"]
    assert profile == {
        "profile_id": evidence_mode,
        "profile_kind": profile_kind,
        "transport": "evidence_http_v3",
        "identity_binding": "exact_readiness_and_response_headers",
        "direct_database_access": False,
        "mock_fallback_allowed": False,
        "release_eligible": False,
        "provider_execution_corroboration": "backend_harness_required",
        "rag_quality_status": "not_attested_by_agent",
        "result_provenance_paths": [
            "knowledge_snapshot_id",
            "rag_evidence[].source_record_id",
            "rag_evidence[].source_uri",
            "rag_evidence[].version_hash",
            "rag_evidence[].retrieval_methods",
            "rag_evidence[].score",
            "rag_evidence[].raw_scores",
        ],
    }


def test_actual_store_profile_has_a_bounded_longer_http_budget() -> None:
    assert _evidence_timeout_seconds("synthetic-v3") == 30.0
    assert _evidence_timeout_seconds("preserved-3law-v3-diagnostic") == 75.0
    assert _readiness_policy("synthetic-v3") == (5.0, 5.0)
    assert _readiness_policy("preserved-3law-v3-diagnostic") == (30.0, 30.0)


def test_diagnostic_runtime_structurally_forces_human_review() -> None:
    _request, result = _result()

    class Core:
        def review_v3_json(self, _payload: bytes) -> Any:
            return result

        def close(self) -> None:
            return None

    runtime = LocalUatDiagnosticRuntime(
        Core(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        model_id="claude-sonnet-5",
    )
    observed = runtime.review_v3_json(b"{}")

    assert observed.assurance_level == "diagnostic"
    assert observed.review_disposition == "human_review_required"
    assert observed.status == result.status
    assert observed.requires_human_review is result.requires_human_review
    assert observed.report.overall_status == result.report.overall_status
    assert observed.completion_eligibility.eligible is False
    assert "local_uat_non_release" in observed.completion_eligibility.unmet_reason_codes
    assert observed.result_sha256 != result.result_sha256


def test_preserved_three_law_diagnostic_keeps_safe_rag_provenance_unchanged(
    tmp_path: Path,
) -> None:
    manifest = _load_manifest(
        tmp_path,
        evidence_mode="preserved-3law-v3-diagnostic",
    )
    _request, result = _result()

    class Core:
        def review_v3_json(self, _payload: bytes) -> Any:
            return result

        def close(self) -> None:
            return None

    runtime = LocalUatDiagnosticRuntime(
        Core(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        model_id="claude-sonnet-5",
    )
    observed = runtime.review_v3_json(b"{}")

    assert manifest.evidence_mode == "preserved-3law-v3-diagnostic"
    assert observed.knowledge_snapshot_id == result.knowledge_snapshot_id
    assert observed.rag_evidence == result.rag_evidence
    assert observed.report.rag_evidence == result.report.rag_evidence
    assert observed.rag_evidence
    evidence = observed.rag_evidence[0]
    assert evidence.source_record_id
    assert evidence.source_uri
    assert len(evidence.version_hash) == 64
    assert evidence.retrieval_methods
    assert evidence.score is not None
    assert evidence.raw_scores
    assert observed.assurance_level == "diagnostic"
    assert observed.review_disposition == "human_review_required"
    assert observed.completion_eligibility.eligible is False


def test_diagnostic_runtime_refuses_to_reframe_a_release_approved_narrative() -> None:
    _request, result = _result()
    release_approved = result.model_copy(
        update={
            "completion_eligibility": result.completion_eligibility.model_copy(
                update={
                    "eligible": True,
                    "profile_id": "release-approved-test",
                    "unmet_reason_codes": [],
                }
            ),
            "review_disposition": "auto_review_complete",
        }
    )

    class Core:
        def review_v3_json(self, _payload: bytes) -> Any:
            return release_approved

        def close(self) -> None:
            return None

    runtime = LocalUatDiagnosticRuntime(
        Core(),  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        model_id="claude-sonnet-5",
    )
    with pytest.raises(RuntimeExecutionError) as captured:
        runtime.review_v3_json(b"{}")
    assert captured.value.code == "uat_release_result_forbidden"


def test_diagnostic_retriever_never_reads_or_patches_production_identity_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_loader() -> Any:
        pytest.fail("Production identity loader must remain outside LOCAL UAT")

    monkeypatch.setattr(
        production_retrieval_v3,
        "load_released_evidence_runtime_identity_v3",
        forbidden_loader,
    )
    retriever = LoopbackEvidenceV3ConformanceRetriever(
        "http://127.0.0.1:18091",
        conformance_identity=_identity(),
        service_token=EVIDENCE_TOKEN,
        acknowledge_non_release=True,
    )
    assert retriever.expected_identity == _identity()
    assert production_retrieval_v3.load_released_evidence_runtime_identity_v3 is forbidden_loader


def test_preserved_three_law_retriever_uses_the_same_no_production_lock_http_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden_loader() -> Any:
        pytest.fail("Diagnostic profile must not load a Production identity")

    monkeypatch.setattr(
        production_retrieval_v3,
        "load_released_evidence_runtime_identity_v3",
        forbidden_loader,
    )
    retriever = LoopbackEvidenceV3DiagnosticRetriever(
        "http://127.0.0.1:18091",
        conformance_identity=_identity(),
        service_token=EVIDENCE_TOKEN,
        acknowledge_non_release=True,
    )

    assert retriever.expected_identity == _identity()
    assert retriever.url == "http://127.0.0.1:18091/internal/rag/v3/retrieve"
    assert production_retrieval_v3.load_released_evidence_runtime_identity_v3 is forbidden_loader


def test_local_env_cannot_smuggle_production_or_concurrent_mode(tmp_path: Path) -> None:
    manifest = _load_manifest(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            (
                f"CUSTOMS_AGENT_SERVICE_BEARER_TOKEN={SERVICE_TOKEN}",
                f"EVIDENCE_SERVICE_TOKEN={EVIDENCE_TOKEN}",
                "CUSTOMS_AGENT_ENVIRONMENT=production",
                "CUSTOMS_AGENT_MAX_CONCURRENCY=2",
            )
        ),
        encoding="utf-8",
    )
    with pytest.raises(LocalUatConfigurationError, match="uat_env_boundary_conflict"):
        local_uat_service_settings(manifest, env_file)


def test_uat_health_metadata_and_v2_routes_are_explicit_and_safe(tmp_path: Path) -> None:
    manifest = _load_manifest(tmp_path)
    runtime = _RouteRuntime()
    app = _app(manifest, runtime)
    auth = {"Authorization": f"Bearer {SERVICE_TOKEN}"}
    review_payload = {
        "case_id": "case-uat-1",
        "documents": [
            {
                "document_id": "invoice-1",
                "filename": "invoice.txt",
                "content_text": "COMMERCIAL INVOICE\nInvoice No: SECRET-DOCUMENT-VALUE",
            }
        ],
        "llm_mode": "llm",
        "review_as_of": "2026-07-19",
        "review_run_id": "run-uat-1",
    }
    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/health/live").status_code == 200
        production_ready = client.get("/health/ready")
        assert production_ready.status_code == 503
        assert production_ready.json()["error"]["code"] == (
            "production_readiness_unavailable_in_local_uat"
        )
        uat_ready = client.get("/health/uat-ready")
        assert uat_ready.status_code == 200
        assert uat_ready.json()["evidence_mode"] == "synthetic-v3"
        assert uat_ready.json()["auto_completion_allowed"] is False

        assert client.get("/internal/agent/v1/metadata", headers=auth).status_code == 404
        assert client.get("/internal/agent/v2/metadata").status_code == 401
        metadata = client.get("/internal/agent/v2/metadata", headers=auth)
        assert metadata.status_code == 200
        metadata_payload = metadata.json()
        assert metadata_payload["status"] == "ready"
        assert metadata_payload["service_api_version"] == "internal-agent-service/v2"
        assert metadata_payload["service_mode"] == "local_user_uat_diagnostic"
        assert metadata_payload["assurance_level"] == "diagnostic"
        assert metadata_payload["transport_limits"]["text_only"] is True
        assert metadata_payload["agentic_limits"]["effective_execution_mode"] == "shadow"
        assert not {
            "diagnostic_only",
            "text_only",
            "mock_fallback_allowed",
            "auto_completion_allowed",
            "evidence_mode",
            "planner_mode",
        }.intersection(metadata_payload)
        assert metadata_payload["uat"]["evidence_mode"] == "synthetic-v3"
        assert metadata_payload["uat"]["evidence_profile"]["profile_kind"] == (
            "synthetic_contract_control"
        )
        assert metadata_payload["uat"]["evidence_profile"]["direct_database_access"] is False
        assert metadata_payload["uat"]["evidence_profile"]["rag_quality_status"] == (
            "not_attested_by_agent"
        )
        assert metadata_payload["uat"]["mock_fallback_allowed"] is False
        assert metadata_payload["automatic_completion_release"]["uat_forced_disabled"] is True

        review = client.post("/internal/agent/v2/reviews", headers=auth, json=review_payload)
        assert review.status_code == 200
        assert review.json()["assurance_level"] == "diagnostic"
        revision = client.post(
            "/internal/agent/v2/revisions",
            headers=auth,
            json={
                "parent_request": {"documents": [{"content_text": "invoice"}]},
                "revised_request": {
                    "review_run_id": "child-uat-1",
                    "documents": [{"content_text": "invoice corrected"}],
                },
            },
        )
        assert revision.status_code == 200
        answer = client.post(
            "/internal/agent/v2/answers",
            headers=auth,
            json={"question_id": "question_" + "a" * 64},
        )
        assert answer.status_code == 200

    assert runtime.calls == ["review", "revision", "answer"]
    encoded = metadata.text
    for secret in (
        SERVICE_TOKEN,
        EVIDENCE_TOKEN,
        "SECRET-DOCUMENT-VALUE",
        "ANTHROPIC_API_KEY",
    ):
        assert secret not in encoded


def test_preserved_three_law_mode_is_visible_but_stays_non_release(
    tmp_path: Path,
) -> None:
    manifest = _load_manifest(
        tmp_path,
        evidence_mode="preserved-3law-v3-diagnostic",
    )
    app = _app(manifest, _RouteRuntime())
    auth = {"Authorization": f"Bearer {SERVICE_TOKEN}"}

    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/health/ready").status_code == 503
        ready = client.get("/health/uat-ready")
        metadata = client.get("/internal/agent/v2/metadata", headers=auth)

    assert ready.status_code == 200
    assert ready.json()["evidence_mode"] == "preserved-3law-v3-diagnostic"
    payload = metadata.json()
    assert payload["assurance_level"] == "diagnostic"
    assert payload["uat"]["evidence_mode"] == "preserved-3law-v3-diagnostic"
    profile = payload["uat"]["evidence_profile"]
    assert profile["profile_kind"] == "preserved_three_law_rag_diagnostic"
    assert profile["provider_execution_corroboration"] == "backend_harness_required"
    assert profile["rag_quality_status"] == "not_attested_by_agent"
    assert profile["release_eligible"] is False
    assert payload["automatic_completion_release"]["uat_forced_disabled"] is True


def test_uat_errors_redact_runtime_and_secret_detail(tmp_path: Path) -> None:
    manifest = _load_manifest(tmp_path)

    class ExplodingRuntime(_RouteRuntime):
        def review_v3_json(self, _payload: bytes) -> _Dump:
            raise RuntimeExecutionError(
                "private_failure",
                f"{SERVICE_TOKEN}:{EVIDENCE_TOKEN}:raw-document",
            )

    app = _app(manifest, ExplodingRuntime())
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/internal/agent/v2/reviews",
            headers={"Authorization": f"Bearer {SERVICE_TOKEN}"},
            json={
                "case_id": "case-uat-2",
                "documents": [
                    {
                        "document_id": "invoice-2",
                        "filename": "invoice.txt",
                        "content_text": "private source text",
                    }
                ],
                "llm_mode": "llm",
                "review_as_of": "2026-07-19",
                "review_run_id": "run-uat-2",
            },
        )
    assert response.status_code == 502
    assert response.json()["error"]["code"] == "review_execution_failed"
    assert SERVICE_TOKEN not in response.text
    assert EVIDENCE_TOKEN not in response.text
    assert "raw-document" not in response.text


def test_uat_cannot_run_with_production_settings_or_without_acknowledgement(
    tmp_path: Path,
) -> None:
    manifest = _load_manifest(tmp_path)
    runtime = _RouteRuntime()
    assert "_runtime_for_tests" not in inspect.signature(create_local_uat_app).parameters
    with pytest.raises(LocalUatConfigurationError, match="uat_service_boundary_invalid"):
        _compose_local_uat_app(
            manifest=manifest,
            settings=_settings(manifest, environment="production"),
            acknowledge_local_uat=True,
            runtime=runtime,
            readiness_probe=lambda: object(),
            model_id="claude-sonnet-5",
        )
    with pytest.raises(LocalUatConfigurationError, match="uat_acknowledgement_required"):
        _compose_local_uat_app(
            manifest=manifest,
            settings=_settings(manifest),
            acknowledge_local_uat=False,  # type: ignore[arg-type]
            runtime=runtime,
            readiness_probe=lambda: object(),
            model_id="claude-sonnet-5",
        )
