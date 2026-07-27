from __future__ import annotations

import base64
import hashlib
import inspect
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from fastapi.testclient import TestClient
from pydantic import SecretStr

import customs_ai_agent.adapters.retrieval_v3 as production_retrieval_v3
import customs_ai_agent.assertion_trust as assertion_trust
from customs_ai_agent.runtime import RuntimeConfigurationError, RuntimeExecutionError
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

_PROFILE_POLICY_IDENTITIES: dict[str, dict[str, str]] = {
    "synthetic-v3": {
        "requirement_policy_release": "test-only-conformance.no-production-policy.v1",
        "requirement_policy_manifest_sha256": (
            "882d18df864cf19149589705c27e9d94b8441e80c8d6bef397a7507cf67a9c90"
        ),
        "coverage_matrix_sha256": (
            "a47fd7e683b7e0a270c5d4b76365034b2f24a0b4f2a52ba1c69421480c1f5263"
        ),
    },
    "preserved-3law-v3-diagnostic": {
        "requirement_policy_release": "test-only-conformance.no-production-policy.v1",
        "requirement_policy_manifest_sha256": (
            "54aa31a0baed9a8eeeea57f9c436a78b8d687e244538491a46e34af46238cf7a"
        ),
        "coverage_matrix_sha256": (
            "a47fd7e683b7e0a270c5d4b76365034b2f24a0b4f2a52ba1c69421480c1f5263"
        ),
    },
    "preserved-3law-v3-scenario-eval": {
        "requirement_policy_release": "test-only-three-law-scenario-eval-human-only.v1",
        "requirement_policy_manifest_sha256": (
            "2467226fe64c84f8fdc0b41a8ab44ebaba04e6ba7280cd3e4bd56bd2e9c38c95"
        ),
        "coverage_matrix_sha256": (
            "21012116b4fddf0ae0e0bb12343558aca472373480d32d535e6b59875800c444"
        ),
        "reviewer_trust_store_sha256": (
            "7864482157b701b44c457bd35c7721b31ffc31564ea43d83948aa27e08233637"
        ),
    },
}


def _rehash_v3_identity(payload: dict[str, Any]) -> dict[str, Any]:
    result = dict(payload)
    result.pop("runtime_release_v3_sha256", None)
    canonical = json.dumps(
        result,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    result["runtime_release_v3_sha256"] = hashlib.sha256(canonical).hexdigest()
    return result


def _profile_identity(profile: str) -> dict[str, Any]:
    payload = _identity().to_wire()
    payload.update(_PROFILE_POLICY_IDENTITIES[profile])
    return _rehash_v3_identity(payload)


def _manifest_payload(**updates: Any) -> dict[str, Any]:
    requested_mode = updates.get("evidence_mode", "synthetic-v3")
    identity_profile = (
        requested_mode if requested_mode in _PROFILE_POLICY_IDENTITIES else "synthetic-v3"
    )
    payload: dict[str, Any] = {
        "schema_version": "customs-agent-local-uat-manifest/v2",
        "acknowledgement": LOCAL_UAT_ACKNOWLEDGEMENT,
        "diagnostic_only": True,
        "text_only": True,
        "mock_fallback_allowed": False,
        "auto_completion_allowed": False,
        "evidence_mode": "synthetic-v3",
        "evidence_base_url": "http://127.0.0.1:18091",
        "evidence_runtime_identity": _profile_identity(identity_profile),
        "assertion_fixture_mode": "DISABLED",
        "assertion_trust_store_path": None,
        "assertion_trust_store_sha256": None,
    }
    payload.update(updates)
    return payload


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _write_uat_assertion_trust_store(
    tmp_path: Path,
    *,
    valid_from: str = "2026-01-01",
    valid_until: str = "2027-12-31",
) -> tuple[Path, dict[str, Any], Ed25519PrivateKey]:
    private = Ed25519PrivateKey.generate()
    asserted_by = "uat-approver"
    authority_id = (
        "backend:principal-sha256:" + hashlib.sha256(asserted_by.encode("utf-8")).hexdigest()
    )
    payload: dict[str, Any] = {
        "contract": "customs-ai-agent-assertion-trust-store",
        "schema_version": "1.0",
        "release_state": "active",
        "trust_domain": "customs-ai-agent/supporting-document-verification/v1",
        "keys": [
            {
                "authority_id": authority_id,
                "authority_release": "backend-human-decision-policy/v1",
                "method": "human_approver",
                "key_id": "local-uat-approver-key-v1",
                "public_key_base64url": _b64(
                    private.public_key().public_bytes(
                        encoding=serialization.Encoding.Raw,
                        format=serialization.PublicFormat.Raw,
                    )
                ),
                "valid_from": valid_from,
                "valid_until": valid_until,
                "status": "active",
            }
        ],
    }
    payload["trust_store_sha256"] = assertion_trust._sha256(payload)
    path = tmp_path / "_secrets" / "uat" / "assertion-trust-store.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path, payload, private


def _load_manifest(tmp_path: Path, **updates: Any) -> LocalUatManifest:
    path = tmp_path / "uat-manifest.json"
    payload = _manifest_payload(**updates)
    if payload["evidence_mode"] == "preserved-3law-v3-scenario-eval" and not {
        "assertion_fixture_mode",
        "assertion_trust_store_path",
        "assertion_trust_store_sha256",
    }.intersection(updates):
        trust_path, trust_store, _private = _write_uat_assertion_trust_store(tmp_path)
        payload.update(
            {
                "assertion_fixture_mode": "LOOPBACK_TEST_ONLY_HUMAN_APPROVER",
                "assertion_trust_store_path": str(trust_path.resolve()),
                "assertion_trust_store_sha256": trust_store["trust_store_sha256"],
            }
        )
    path.write_text(json.dumps(payload), encoding="utf-8")
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
    ("evidence_mode", "profile_kind", "policy_mode"),
    [
        ("synthetic-v3", "synthetic_contract_control", "SYNTHETIC_FIXTURE"),
        (
            "preserved-3law-v3-diagnostic",
            "preserved_three_law_rag_diagnostic",
            "ACTUAL_CODES_HUMAN_ONLY",
        ),
        (
            "preserved-3law-v3-scenario-eval",
            "preserved_three_law_rag_scenario_evaluation",
            "SEALED_TEST_ONLY_ACTUAL_CCCT_HUMAN_ONLY",
        ),
    ],
)
def test_manifest_accepts_only_explicit_diagnostic_profiles_and_exposes_safe_metadata(
    tmp_path: Path,
    evidence_mode: str,
    profile_kind: str,
    policy_mode: str,
) -> None:
    manifest = _load_manifest(tmp_path, evidence_mode=evidence_mode)

    assert manifest.evidence_mode == evidence_mode
    assert manifest.evidence_runtime_identity.to_wire() == _profile_identity(evidence_mode)
    safe_metadata = manifest.safe_metadata()
    assert safe_metadata["evidence_policy_mode"] == policy_mode
    profile = safe_metadata["evidence_profile"]
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


@pytest.mark.parametrize(
    ("evidence_mode", "identity_profile"),
    [
        ("synthetic-v3", "preserved-3law-v3-diagnostic"),
        ("synthetic-v3", "preserved-3law-v3-scenario-eval"),
        ("preserved-3law-v3-diagnostic", "synthetic-v3"),
        ("preserved-3law-v3-diagnostic", "preserved-3law-v3-scenario-eval"),
        ("preserved-3law-v3-scenario-eval", "synthetic-v3"),
        ("preserved-3law-v3-scenario-eval", "preserved-3law-v3-diagnostic"),
    ],
)
def test_manifest_rejects_a_structurally_valid_cross_profile_policy_identity(
    tmp_path: Path,
    evidence_mode: str,
    identity_profile: str,
) -> None:
    with pytest.raises(
        LocalUatConfigurationError,
        match="uat_evidence_policy_identity_mismatch",
    ):
        _load_manifest(
            tmp_path,
            evidence_mode=evidence_mode,
            evidence_runtime_identity=_profile_identity(identity_profile),
        )


@pytest.mark.parametrize(
    "field",
    [
        "requirement_policy_release",
        "requirement_policy_manifest_sha256",
        "coverage_matrix_sha256",
        "reviewer_trust_store_sha256",
    ],
)
def test_scenario_eval_manifest_rejects_a_rehashed_policy_identity_mismatch(
    tmp_path: Path,
    field: str,
) -> None:
    identity = _profile_identity("preserved-3law-v3-scenario-eval")
    identity[field] = "f" * 64 if field != "requirement_policy_release" else "old-policy.v1"
    identity = _rehash_v3_identity(identity)

    with pytest.raises(
        LocalUatConfigurationError,
        match="uat_evidence_policy_identity_mismatch",
    ):
        _load_manifest(
            tmp_path,
            evidence_mode="preserved-3law-v3-scenario-eval",
            evidence_runtime_identity=identity,
        )


def test_scenario_eval_requires_exact_external_test_only_assertion_trust(
    tmp_path: Path,
) -> None:
    trust_path, trust_store, _private = _write_uat_assertion_trust_store(tmp_path)
    common = {
        "evidence_mode": "preserved-3law-v3-scenario-eval",
        "assertion_fixture_mode": "LOOPBACK_TEST_ONLY_HUMAN_APPROVER",
        "assertion_trust_store_path": str(trust_path.resolve()),
        "assertion_trust_store_sha256": trust_store["trust_store_sha256"],
    }
    manifest = _load_manifest(tmp_path, **common)
    assert manifest.assertion_trust_store is not None
    assert manifest.assertion_trust_store.trust_store_sha256 == (trust_store["trust_store_sha256"])
    assert manifest.safe_metadata()["trusted_assertion_verifier_configured"] is True
    assert "assertion_trust_store_path" not in manifest.safe_metadata()

    for mutation in (
        {"assertion_fixture_mode": "DISABLED"},
        {"assertion_trust_store_sha256": "0" * 64},
        {"assertion_trust_store_path": None},
    ):
        with pytest.raises(
            LocalUatConfigurationError,
            match="uat_assertion",
        ):
            _load_manifest(tmp_path, **{**common, **mutation})


def test_scenario_eval_rejects_wrong_approver_authority_even_with_a_valid_digest(
    tmp_path: Path,
) -> None:
    trust_path, trust_store, _private = _write_uat_assertion_trust_store(tmp_path)
    trust_store["keys"][0]["authority_id"] = (
        "backend:principal-sha256:" + hashlib.sha256(b"not-uat-approver").hexdigest()
    )
    trust_store["trust_store_sha256"] = assertion_trust._sha256(
        {key: value for key, value in trust_store.items() if key != "trust_store_sha256"}
    )
    trust_path.write_text(json.dumps(trust_store), encoding="utf-8")

    with pytest.raises(
        LocalUatConfigurationError,
        match="uat_assertion_trust_store_policy_invalid",
    ):
        _load_manifest(
            tmp_path,
            evidence_mode="preserved-3law-v3-scenario-eval",
            assertion_fixture_mode="LOOPBACK_TEST_ONLY_HUMAN_APPROVER",
            assertion_trust_store_path=str(trust_path.resolve()),
            assertion_trust_store_sha256=trust_store["trust_store_sha256"],
        )


def test_manifest_v1_is_rejected_after_assertion_binding_became_mandatory(
    tmp_path: Path,
) -> None:
    with pytest.raises(
        LocalUatConfigurationError,
        match="uat_manifest_policy_invalid",
    ):
        _load_manifest(tmp_path, schema_version="customs-agent-local-uat-manifest/v1")


def test_scenario_eval_rejects_stale_or_non_secret_assertion_trust_paths(
    tmp_path: Path,
) -> None:
    stale_path, stale_store, _private = _write_uat_assertion_trust_store(
        tmp_path,
        valid_from="2025-01-01",
        valid_until="2025-12-31",
    )
    with pytest.raises(
        LocalUatConfigurationError,
        match="uat_assertion_trust_store_policy_invalid",
    ):
        _load_manifest(
            tmp_path,
            evidence_mode="preserved-3law-v3-scenario-eval",
            assertion_fixture_mode="LOOPBACK_TEST_ONLY_HUMAN_APPROVER",
            assertion_trust_store_path=str(stale_path.resolve()),
            assertion_trust_store_sha256=stale_store["trust_store_sha256"],
        )

    valid_path, valid_store, _private = _write_uat_assertion_trust_store(tmp_path)
    outside = tmp_path / "public-trust-store.json"
    outside.write_bytes(valid_path.read_bytes())
    with pytest.raises(
        LocalUatConfigurationError,
        match="uat_assertion_trust_store_path_invalid",
    ):
        _load_manifest(
            tmp_path,
            evidence_mode="preserved-3law-v3-scenario-eval",
            assertion_fixture_mode="LOOPBACK_TEST_ONLY_HUMAN_APPROVER",
            assertion_trust_store_path=str(outside.resolve()),
            assertion_trust_store_sha256=valid_store["trust_store_sha256"],
        )


def test_non_scenario_profiles_reject_assertion_fixture_activation(tmp_path: Path) -> None:
    trust_path, trust_store, _private = _write_uat_assertion_trust_store(tmp_path)
    for evidence_mode in ("synthetic-v3", "preserved-3law-v3-diagnostic"):
        with pytest.raises(
            LocalUatConfigurationError,
            match="uat_assertion_fixture_policy_invalid",
        ):
            _load_manifest(
                tmp_path,
                evidence_mode=evidence_mode,
                assertion_fixture_mode="LOOPBACK_TEST_ONLY_HUMAN_APPROVER",
                assertion_trust_store_path=str(trust_path.resolve()),
                assertion_trust_store_sha256=trust_store["trust_store_sha256"],
            )


def test_actual_store_profile_has_a_bounded_longer_http_budget() -> None:
    assert _evidence_timeout_seconds("synthetic-v3") == 30.0
    assert _evidence_timeout_seconds("preserved-3law-v3-diagnostic") == 195.0
    assert _evidence_timeout_seconds("preserved-3law-v3-scenario-eval") == 195.0
    assert _readiness_policy("synthetic-v3") == (5.0, 5.0)
    assert _readiness_policy("preserved-3law-v3-diagnostic") == (60.0, 90.0)
    assert _readiness_policy("preserved-3law-v3-scenario-eval") == (60.0, 90.0)


@pytest.mark.parametrize(
    "evidence_mode",
    [
        "preserved-3law-v3-diagnostic",
        "preserved-3law-v3-scenario-eval",
    ],
)
def test_actual_store_profile_rejects_service_budget_not_larger_than_http_budget(
    tmp_path: Path,
    evidence_mode: str,
) -> None:
    manifest = _load_manifest(
        tmp_path,
        evidence_mode=evidence_mode,
    )
    env_file = tmp_path / ".env"
    env_file.write_text(
        "\n".join(
            (
                "CUSTOMS_AGENT_SERVICE_BEARER_TOKEN=Abcdefghijklmnopqrstuvwxyz0123456789-._~+/=XYZ",
                "CUSTOMS_AGENT_RUN_TIMEOUT_SECONDS=195",
            )
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(
        LocalUatConfigurationError,
        match="uat_service_boundary_invalid",
    ):
        local_uat_service_settings(manifest, env_file)


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
        assert uat_ready.json()["evidence_policy_mode"] == "SYNTHETIC_FIXTURE"
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
        assert metadata_payload["uat"]["evidence_policy_mode"] == "SYNTHETIC_FIXTURE"
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


def test_enhanced_uat_health_rejects_expired_capability_before_evidence_probe(
    tmp_path: Path,
) -> None:
    manifest = replace(
        _load_manifest(
            tmp_path,
            evidence_mode="preserved-3law-v3-scenario-eval",
        ),
        enhanced_experience_enabled=True,
    )
    readiness_calls = 0

    def probe() -> object:
        nonlocal readiness_calls
        readiness_calls += 1
        return object()

    class ExpiredRuntime(_RouteRuntime):
        def assert_uat_capability_current(self, *, require_active: bool = False) -> None:
            assert require_active is False
            raise RuntimeConfigurationError(
                "uat_capability_invalid_or_expired",
                "expired test capability",
            )

    app = _app(manifest, ExpiredRuntime(), probe=probe)
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/health/uat-ready")

    assert response.status_code == 503
    assert response.json()["error"]["code"] == "uat_capability_invalid_or_expired"
    assert readiness_calls == 0


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


def test_scenario_eval_mode_uses_actual_store_boundary_and_sealed_test_policy(
    tmp_path: Path,
) -> None:
    manifest = _load_manifest(
        tmp_path,
        evidence_mode="preserved-3law-v3-scenario-eval",
    )
    app = _app(manifest, _RouteRuntime())
    auth = {"Authorization": f"Bearer {SERVICE_TOKEN}"}

    with TestClient(app, raise_server_exceptions=False) as client:
        assert client.get("/health/ready").status_code == 503
        ready = client.get("/health/uat-ready")
        metadata = client.get("/internal/agent/v2/metadata", headers=auth)

    assert ready.status_code == 200
    assert ready.json()["evidence_mode"] == "preserved-3law-v3-scenario-eval"
    assert ready.json()["evidence_policy_mode"] == "SEALED_TEST_ONLY_ACTUAL_CCCT_HUMAN_ONLY"
    assert ready.json()["assertion_fixture_mode"] == "LOOPBACK_TEST_ONLY_HUMAN_APPROVER"
    assert ready.json()["trusted_assertion_verifier_configured"] is True
    assert len(ready.json()["assertion_trust_store_sha256"]) == 64
    payload = metadata.json()
    assert payload["assurance_level"] == "diagnostic"
    assert payload["uat"]["evidence_mode"] == "preserved-3law-v3-scenario-eval"
    assert payload["uat"]["evidence_policy_mode"] == "SEALED_TEST_ONLY_ACTUAL_CCCT_HUMAN_ONLY"
    assert payload["uat"]["trusted_assertion_verifier_configured"] is True
    assert "assertion_trust_store_path" not in payload["uat"]
    profile = payload["uat"]["evidence_profile"]
    assert profile["profile_kind"] == "preserved_three_law_rag_scenario_evaluation"
    assert profile["direct_database_access"] is False
    assert profile["mock_fallback_allowed"] is False
    assert profile["release_eligible"] is False
    assert payload["automatic_completion_release"]["uat_forced_disabled"] is True


def test_scenario_eval_review_identity_uses_the_injected_active_trust_store(
    tmp_path: Path,
) -> None:
    from customs_ai_agent.agentic import runtime_release_identity
    from customs_ai_agent.assertion_trust import SupportingDocumentAssertionVerifier
    from customs_ai_agent.runtime import AgentRuntime

    manifest = _load_manifest(
        tmp_path,
        evidence_mode="preserved-3law-v3-scenario-eval",
    )
    assert manifest.assertion_trust_store is not None
    runtime = object.__new__(AgentRuntime)
    runtime._assertion_verifier = SupportingDocumentAssertionVerifier(
        manifest.assertion_trust_store
    )

    injected = runtime._runtime_assertion_trust_store()
    identity = runtime_release_identity(
        model_id="claude-sonnet-5-20260722",
        evidence_snapshot_id="snapshot-three-law",
        evidence_runtime_v3_sha256="e" * 64,
        assertion_trust_store=injected,
    )

    assert injected == manifest.assertion_trust_store
    assert identity.assertion_trust_store_sha256 == (
        manifest.assertion_trust_store.trust_store_sha256
    )
    assert identity.assertion_trust_store_state == "active"


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
