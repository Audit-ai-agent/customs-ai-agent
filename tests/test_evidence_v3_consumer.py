from __future__ import annotations

import hashlib
import json
import threading
from contextlib import AbstractContextManager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

import customs_ai_agent.adapters.retrieval as retrieval_transport
import customs_ai_agent.adapters.retrieval_v3 as retrieval_v3
import customs_ai_agent.contracts.evidence_v3 as contract_v3
from customs_ai_agent.adapters.retrieval import RetrievalError
from customs_ai_agent.adapters.retrieval_v3 import ProductionHttpRegulationRetrieverV3
from customs_ai_agent.contracts.evidence_runtime_release import (
    EvidenceRuntimeReleaseIdentity,
    canonical_json_bytes,
    load_pinned_evidence_provider_runtime_identity,
)
from customs_ai_agent.contracts.evidence_v3 import (
    EvidenceRuntimeReleaseIdentityV3,
    EvidenceV3ContractError,
    EvidenceV3UnreleasedError,
    load_evidence_runtime_release_lock_v3,
    load_released_evidence_runtime_identity_v3,
    packaged_evidence_v3_schema_hashes,
)
from customs_ai_agent.schemas.models import EvidenceRetrievalRequest, stable_evidence_id
from tools.agent_execution_inspector.evidence_v3_conformance import (
    LoopbackEvidenceV3ConformanceRetriever,
)


def _base_identity() -> EvidenceRuntimeReleaseIdentity:
    payload: dict[str, Any] = {
        "contract": "evidence-runtime-release",
        "schema_version": "2.0",
        "retrieval_schema_version": "1.0",
        "provider_release": "evidence-api-v1.2.0",
        "provider_revision": "1" * 40,
        "provider_build_sha256": "2" * 64,
        "retrieval_schema_sha256": (
            "531b59ba4e86f78cef20a4fca052996efd8b88c48853ac12113ca76e69cdd69a"
        ),
        "provider_source_manifest_sha256": "3" * 64,
        "knowledge_generation_id": "4" * 64,
        "knowledge_generation_attestation_sha256": "5" * 64,
        "knowledge_source_manifest_sha256": "6" * 64,
        "model_manifest_sha256": "7" * 64,
        "index_config_sha256": "8" * 64,
        "evaluation_report_sha256": "9" * 64,
        "seal_attestation_sha256": "a" * 64,
        "knowledge_snapshot_id": "knowledge-2026-07-19",
        "supported_review_as_of": "2026-07-19",
    }
    payload["runtime_release_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return EvidenceRuntimeReleaseIdentity.from_mapping(payload)


def _identity() -> EvidenceRuntimeReleaseIdentityV3:
    hashes = packaged_evidence_v3_schema_hashes()
    payload: dict[str, Any] = {
        "contract": "evidence-runtime-release",
        "schema_version": "3.0",
        "retrieval_schema_version": "3.0",
        "base_runtime_release": _base_identity().to_wire(),
        "retrieval_schema_sha256": hashes["rag_retrieval_v3.schema.json"],
        "readiness_schema_sha256": hashes["readiness_v3.schema.json"],
        "requirement_policy_schema_sha256": hashes["requirement_policy_v1.schema.json"],
        "requirement_policy_release": "requirement-policy-v1",
        "requirement_policy_manifest_sha256": "b" * 64,
        "coverage_matrix_sha256": "c" * 64,
        "requirement_policy_approval_sha256": "d" * 64,
        "reviewer_trust_store_sha256": "e" * 64,
    }
    payload["runtime_release_v3_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return EvidenceRuntimeReleaseIdentityV3.from_mapping(payload)


def _request() -> EvidenceRetrievalRequest:
    return EvidenceRetrievalRequest(
        request_id="request-v3-1",
        case_id="case-v3-1",
        review_run_id="run-v3-1",
        review_as_of="2026-07-19",
        jurisdiction="KR",
        trade_type="import",
        items=[
            {
                "item_id": "item-v3-1",
                "description": "Frozen mackerel",
                "hs_code": "0303540000",
                "origin_country": "NO",
                "destination_country": "KR",
            }
        ],
    )


def _readiness(identity: EvidenceRuntimeReleaseIdentityV3) -> bytes:
    return json.dumps(
        {"status": "ready", "evidence_runtime_release": identity.to_wire()},
        separators=(",", ":"),
    ).encode()


def _response_payload(
    request: EvidenceRetrievalRequest,
    identity: EvidenceRuntimeReleaseIdentityV3,
    *,
    with_requirement: bool = False,
) -> bytes:
    evidence: list[dict[str, Any]] = []
    if with_requirement:
        source_record_id = "requirement:0303540000:import:1"
        version_hash = "f" * 64
        evidence.append(
            {
                "evidence_id": stable_evidence_id(source_record_id, version_hash),
                "source_record_id": source_record_id,
                "source_type": "customs_requirement",
                "item_id": request.items[0].item_id,
                "title": "Synthetic requirement",
                "text": "A synthetic confirmation is required.",
                "law_id": "law-test",
                "law_name": "Synthetic Act",
                "article_no": "1",
                "source_uri": "https://example.test/requirements/1",
                "valid_from": "2026-01-01",
                "valid_to": None,
                "version_hash": version_hash,
                "matched_hs_code": "0303540000",
                "trade_type": "import",
                "required_document_code": "REQ_CERT",
                "required_document_name": "Synthetic confirmation",
                "agencies": ["Synthetic agency"],
                "retrieval_methods": ["exact_hs"],
                "eligibility": "finding",
                "score": 1.0,
                "raw_scores": {"exact_hs": 1.0},
                "requirement_verification": {
                    "required_document_code": "REQ_CERT",
                    "verification_policy": "external_verification_required",
                    "policy_source_sha256": "1" * 64,
                    "policy_release": identity.requirement_policy_release,
                    "approved": True,
                },
            }
        )
    return json.dumps(
        {
            "schema_version": "3.0",
            "request_id": request.request_id,
            "status": "ok",
            "knowledge_snapshot_id": identity.base_runtime_release.knowledge_snapshot_id,
            "results": [
                {
                    "item_id": request.items[0].item_id,
                    "hs_resolution": "exact",
                    "resolved_hsk10": "0303540000",
                    "candidate_hsk10s": ["0303540000"],
                    "coverage": "complete",
                    "requirements_found": with_requirement,
                    "evidence": evidence,
                    "errors": [],
                }
            ],
            "errors": [],
        },
        separators=(",", ":"),
    ).encode()


class _Response(AbstractContextManager["_Response"]):
    def __init__(self, body: bytes, headers: dict[str, str] | None = None) -> None:
        self.body = body
        self.headers = headers or {}
        self.read_calls = 0
        self._offset = 0

    def __enter__(self) -> "_Response":
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, limit: int) -> bytes:
        self.read_calls += 1
        chunk = self.body[self._offset : self._offset + limit]
        self._offset += len(chunk)
        return chunk

    def getheader(self, name: str) -> str | None:
        target = name.casefold()
        values = [value for key, value in self.headers.items() if key.casefold() == target]
        return values[0] if len(values) == 1 else None


def _released_retriever(
    monkeypatch: pytest.MonkeyPatch,
    identity: EvidenceRuntimeReleaseIdentityV3,
) -> ProductionHttpRegulationRetrieverV3:
    monkeypatch.setattr(
        retrieval_v3, "load_released_evidence_runtime_identity_v3", lambda: identity
    )
    return ProductionHttpRegulationRetrieverV3(
        "http://127.0.0.1:8010",
        service_token="A" * 40,
        allowed_origins={"http://127.0.0.1:8010"},
        max_retries=0,
    )


def test_packaged_v3_contract_hashes_are_provider_frozen_and_release_is_unreleased() -> None:
    assert packaged_evidence_v3_schema_hashes() == {
        "rag_retrieval_v3.schema.json": (
            "d20de13f4cc8db14a2cd3213e063916fd046922ae32687ac939dbb4019e485d6"
        ),
        "readiness_v3.schema.json": (
            "3488e98b9899ce6cbb375fad4120360a2d4a77d82117ef56f098514547a7d882"
        ),
        "requirement_policy_v1.schema.json": (
            "b1e9e5918d03c247d8503af861ee647451b0c2f378c1a9d2cc380f23ca26c226"
        ),
    }
    provider = load_pinned_evidence_provider_runtime_identity()
    assert provider.to_wire() == {
        "contract": "evidence-runtime-release-provider-lock",
        "schema_version": "1.0",
        "provider_release": "evidence-api-v1.2.0",
        "provider_revision": "14cde864425eee615a2e3f6799f6f2719448aab9",
        "provider_build_sha256": (
            "d7d64cf3df464cfc934ca0f347c1b1c461916fe12671eeeaf59c1445872486d3"
        ),
        "provider_source_manifest_sha256": (
            "dd8ceb68c2da83d1fdb82f3d00f4f1137a559fcb577efdf86b6f87e52d84db1c"
        ),
        "retrieval_schema_sha256": (
            "531b59ba4e86f78cef20a4fca052996efd8b88c48853ac12113ca76e69cdd69a"
        ),
        "readiness_schema_sha256": (
            "0a0b50fef627ecbbad954af5878ccb3630ecc51f618132f8cfc674c324467a69"
        ),
    }
    lock = load_evidence_runtime_release_lock_v3()
    assert lock.release_state == "unreleased"
    assert lock.provider_runtime_identity == provider
    assert lock.identity is None
    with pytest.raises(EvidenceV3UnreleasedError):
        load_released_evidence_runtime_identity_v3()


@pytest.mark.parametrize(
    "malformed",
    [
        b'{"release_state":"released","release_state":"unreleased","identity":null}',
        b'{"release_state":"released","identity":NaN}',
    ],
)
def test_v3_release_lock_rejects_duplicate_keys_and_nonfinite_json(
    monkeypatch: pytest.MonkeyPatch,
    malformed: bytes,
) -> None:
    original = contract_v3._resource

    def resource(name: str) -> bytes:
        return malformed if name == "runtime-release-v3.lock.json" else original(name)

    monkeypatch.setattr(contract_v3, "_resource", resource)
    contract_v3.load_released_evidence_runtime_identity_v3.cache_clear()
    with pytest.raises(EvidenceV3ContractError):
        contract_v3.load_released_evidence_runtime_identity_v3()
    contract_v3.load_released_evidence_runtime_identity_v3.cache_clear()


def test_v3_http_consumer_sends_and_verifies_all_seventeen_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    request = _request()
    retriever = _released_retriever(monkeypatch, identity)
    responses = [
        _Response(_readiness(identity)),
        _Response(
            _response_payload(request, identity, with_requirement=True),
            {**identity.binding_headers(), "X-Request-ID": request.request_id},
        ),
    ]
    captured: list[Any] = []

    def fake_urlopen(http_request: Any, timeout: float) -> _Response:
        assert timeout > 0
        captured.append(http_request)
        return responses.pop(0)

    monkeypatch.setattr(retrieval_transport, "urlopen", fake_urlopen)
    response = retriever.retrieve(request)
    assert response.status == "ok"
    assert len(identity.binding_headers()) == 17
    sent = {name.casefold(): value for name, value in captured[1].header_items()}
    for name, expected in identity.binding_headers().items():
        assert sent[name.casefold()] == expected
    assert sent["x-request-id"] == request.request_id
    assert retriever.policy_bindings(request.request_id) == {
        "REQ_CERT": {
            "required_document_code": "REQ_CERT",
            "verification_policy": "external_verification_required",
            "policy_source_sha256": "1" * 64,
            "policy_release": identity.requirement_policy_release,
            "approved": True,
        }
    }


def test_v3_post_identity_mismatch_discards_body_and_prior_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    request = _request()
    retriever = _released_retriever(monkeypatch, identity)
    bad_headers = {**identity.binding_headers(), "X-Request-ID": request.request_id}
    bad_headers["X-Evidence-Runtime-V3-SHA256"] = "0" * 64
    post = _Response(_response_payload(request, identity), bad_headers)
    responses = [_Response(_readiness(identity)), post]

    def fake_urlopen(_request: Any, timeout: float) -> _Response:
        assert timeout > 0
        return responses.pop(0)

    monkeypatch.setattr(retrieval_transport, "urlopen", fake_urlopen)
    with pytest.raises(RetrievalError, match="identity"):
        retriever.retrieve(request)
    assert post.read_calls == 0
    assert retriever.policy_bindings(request.request_id) == {}


def test_non_release_factory_executes_real_loopback_get_and_post() -> None:
    identity = _identity()
    request = _request()
    observed: dict[str, Any] = {"methods": [], "headers": {}, "body": b""}

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, _format: str, *_args: object) -> None:
            return None

        def do_GET(self) -> None:  # noqa: N802 - stdlib HTTP handler API
            observed["methods"].append(("GET", self.path))
            body = _readiness(identity)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_POST(self) -> None:  # noqa: N802 - stdlib HTTP handler API
            observed["methods"].append(("POST", self.path))
            length = int(self.headers.get("Content-Length", "0"))
            observed["body"] = self.rfile.read(length)
            observed["headers"] = {key.casefold(): value for key, value in self.headers.items()}
            body = _response_payload(request, identity, with_requirement=True)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            for name, value in identity.binding_headers().items():
                self.send_header(name, value)
            self.send_header("X-Request-ID", request.request_id)
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}"
        retriever = LoopbackEvidenceV3ConformanceRetriever(
            base_url,
            conformance_identity=identity,
            service_token="A" * 40,
            acknowledge_non_release=True,
            timeout_seconds=10,
        )
        response = retriever.retrieve(request)
        assert retriever.successful_readiness_get_count == 1
        assert retriever.successful_retrieval_post_count == 1
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)

    assert response.status == "ok"
    assert observed["methods"] == [
        ("GET", "/internal/rag/v3/health/ready"),
        ("POST", "/internal/rag/v3/retrieve"),
    ]
    sent_body = json.loads(observed["body"])
    assert sent_body["schema_version"] == "3.0"
    assert sent_body["request_id"] == request.request_id
    for name, expected in identity.binding_headers().items():
        assert observed["headers"][name.casefold()] == expected


@pytest.mark.parametrize(
    "base_url",
    [
        "http://localhost:8010",
        "http://127.0.0.2:8010",
        "http://example.test:8010",
    ],
)
def test_non_release_conformance_rejects_non_literal_loopback_hosts(base_url: str) -> None:
    with pytest.raises(ValueError, match="loopback-only"):
        LoopbackEvidenceV3ConformanceRetriever(
            base_url,
            conformance_identity=_identity(),
            service_token="A" * 40,
            acknowledge_non_release=True,
        )
