from __future__ import annotations

import hashlib
import json
import threading
import time
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any
from urllib.error import HTTPError

import pytest

import customs_ai_agent.adapters.retrieval as retrieval_module
import customs_ai_agent.contracts.evidence_runtime_release as release_module
from customs_ai_agent.adapters.retrieval import (
    EVIDENCE_EVALUATION_SHA256_HEADER,
    EVIDENCE_GENERATION_ATTESTATION_SHA256_HEADER,
    EVIDENCE_KNOWLEDGE_GENERATION_HEADER,
    EVIDENCE_PROVIDER_BUILD_SHA256_HEADER,
    EVIDENCE_PROVIDER_RELEASE_HEADER,
    EVIDENCE_PROVIDER_REVISION_HEADER,
    EVIDENCE_RUNTIME_RELEASE_SHA256_HEADER,
    EVIDENCE_SCHEMA_SHA256_HEADER,
    ProductionHttpRegulationRetriever,
    RetrievalError,
    fetch_evidence_runtime_readiness,
)
from customs_ai_agent.contracts.evidence_runtime_release import (
    EvidenceRuntimeReleaseIdentity,
    EvidenceRuntimeReleaseLockError,
    EvidenceRuntimeReleaseUnreleasedError,
    canonical_json_bytes,
    load_evidence_runtime_release_lock,
    load_pinned_evidence_provider_runtime_identity,
    load_released_evidence_runtime_identity,
)
from customs_ai_agent.schemas.models import (
    EvidenceRetrievalRequest,
    EvidenceRetrievalResponse,
    RetrievalItemResult,
)
from tools.build_evidence_runtime_release_lock import (
    ReleaseLockBuildError,
    build_released_lock,
)


def _identity_payload(**changes: str) -> dict[str, str]:
    payload = {
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
    payload.update(changes)
    payload["runtime_release_sha256"] = hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
    return payload


def _identity(**changes: str) -> EvidenceRuntimeReleaseIdentity:
    return EvidenceRuntimeReleaseIdentity.from_mapping(_identity_payload(**changes))


def _readiness(identity: EvidenceRuntimeReleaseIdentity) -> bytes:
    return json.dumps(
        {
            "status": "ready",
            "evidence_runtime_release": identity.to_wire(),
        },
        separators=(",", ":"),
    ).encode()


def _request() -> EvidenceRetrievalRequest:
    return EvidenceRetrievalRequest(
        request_id="request-v2-1",
        case_id="case-v2-1",
        review_run_id="run-v2-1",
        review_as_of="2026-07-19",
        jurisdiction="KR",
        trade_type="import",
        items=[
            {
                "item_id": "item-v2-1",
                "description": "Machine screw",
                "hs_code": "7318151000",
                "origin_country": "NO",
                "destination_country": "KR",
            }
        ],
    )


def _response_payload(request: EvidenceRetrievalRequest, snapshot: str) -> bytes:
    response = EvidenceRetrievalResponse(
        request_id=request.request_id,
        status="ok",
        knowledge_snapshot_id=snapshot,
        results=[
            RetrievalItemResult(
                item_id=request.items[0].item_id,
                hs_resolution="exact",
                resolved_hsk10="7318151000",
                candidate_hsk10s=["7318151000"],
                coverage="complete",
                requirements_found=False,
                evidence=[],
                errors=[],
            )
        ],
        errors=[],
    )
    return response.model_dump_json().encode()


def _headers(identity: EvidenceRuntimeReleaseIdentity) -> dict[str, str]:
    return {
        EVIDENCE_SCHEMA_SHA256_HEADER: identity.retrieval_schema_sha256,
        EVIDENCE_PROVIDER_RELEASE_HEADER: identity.provider_release,
        EVIDENCE_PROVIDER_REVISION_HEADER: identity.provider_revision,
        EVIDENCE_PROVIDER_BUILD_SHA256_HEADER: identity.provider_build_sha256,
        EVIDENCE_KNOWLEDGE_GENERATION_HEADER: identity.knowledge_generation_id,
        EVIDENCE_GENERATION_ATTESTATION_SHA256_HEADER: (
            identity.knowledge_generation_attestation_sha256
        ),
        EVIDENCE_EVALUATION_SHA256_HEADER: identity.evaluation_report_sha256,
        EVIDENCE_RUNTIME_RELEASE_SHA256_HEADER: identity.runtime_release_sha256,
    }


class _Response(AbstractContextManager["_Response"]):
    def __init__(self, body: bytes, headers: dict[str, str] | None = None) -> None:
        self.body = body
        self.headers = headers or {}
        self.read_calls = 0
        self._offset = 0

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def read(self, limit: int) -> bytes:
        self.read_calls += 1
        chunk = self.body[self._offset : self._offset + limit]
        self._offset += len(chunk)
        return chunk

    def getheader(self, name: str) -> str | None:
        return self.headers.get(name)


class _DuplicateIdentityHeaders:
    def __init__(self, values: dict[str, str], duplicate: str) -> None:
        self.values = values
        self.duplicate = duplicate.casefold()

    def get_all(self, name: str) -> list[str] | None:
        value = self.values.get(name)
        if value is None:
            return None
        return [value, value] if name.casefold() == self.duplicate else [value]


class _SlowDripResponse(AbstractContextManager["_SlowDripResponse"]):
    def __init__(
        self,
        body: bytes,
        *,
        headers: dict[str, str] | None = None,
        interval_seconds: float = 0.01,
    ) -> None:
        self.body = body
        self.headers = headers or {}
        self.interval_seconds = interval_seconds
        self.closed = threading.Event()
        self._offset = 0

    def __enter__(self) -> _SlowDripResponse:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()

    def close(self) -> None:
        self.closed.set()

    def read(self, limit: int) -> bytes:
        if self.closed.wait(self.interval_seconds):
            return b""
        if self._offset >= len(self.body):
            return b""
        chunk = self.body[self._offset : self._offset + min(limit, 1)]
        self._offset += len(chunk)
        return chunk

    def getheader(self, name: str) -> str | None:
        return self.headers.get(name)


def test_packaged_production_lock_is_explicitly_unreleased() -> None:
    lock = load_evidence_runtime_release_lock()
    assert lock.release_state == "unreleased"
    assert lock.identity is None
    with pytest.raises(EvidenceRuntimeReleaseUnreleasedError):
        load_released_evidence_runtime_identity()


def test_missing_packaged_lock_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    original = release_module._read_resource_bytes

    def missing(name: str) -> bytes:
        if name == "runtime-release-v2.lock.json":
            raise FileNotFoundError
        return original(name)

    monkeypatch.setattr(release_module, "_read_resource_bytes", missing)
    release_module.load_evidence_runtime_release_lock.cache_clear()
    with pytest.raises(EvidenceRuntimeReleaseLockError):
        release_module.load_evidence_runtime_release_lock()
    release_module.load_evidence_runtime_release_lock.cache_clear()


def test_checkout_and_packaged_v2_release_resources_are_byte_identical() -> None:
    root = Path(__file__).parents[1]
    for name in (
        "runtime-release-v2.lock.json",
        "runtime-release-v2.schema.json",
    ):
        assert (root / "contracts" / "evidence" / name).read_bytes() == (
            root / "src" / "customs_ai_agent" / "contracts" / "evidence" / name
        ).read_bytes()
    provider_lock = (root / "contracts" / "evidence" / "provider-runtime-lock.json").read_bytes()
    assert (
        provider_lock
        == (
            root
            / "src"
            / "customs_ai_agent"
            / "contracts"
            / "evidence"
            / "provider-runtime-lock.json"
        ).read_bytes()
    )
    assert (
        provider_lock
        == (
            root / "tests" / "fixtures" / "evidence_runtime_v2" / "provider-runtime-lock.json"
        ).read_bytes()
    )
    assert hashlib.sha256(provider_lock).hexdigest() == (
        "e3e16ef56e2e2ce7106b7a5dce2eb3b3fb2618e7658f97742c76bbf47de99a5e"
    )


def test_packaged_provider_runtime_pin_is_exact_and_tamper_evident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = load_pinned_evidence_provider_runtime_identity()
    assert identity.provider_revision == "c9214d17343cec21a575ffde979f045169c7a341"
    assert identity.provider_build_sha256 == (
        "16da9abf8b528815ffa3363cdb9424533edc78bb82f417dc25a56a31966bfeaa"
    )
    assert identity.provider_source_manifest_sha256 == (
        "0c99c24ff4bc51d3f458bbb093666952c36a2942f4967141e0e35bcb06ad5158"
    )

    original = release_module._read_resource_bytes

    def tampered(name: str) -> bytes:
        raw = original(name)
        if name == "provider-runtime-lock.json":
            return raw.replace(b"89fe32ed", b"09fe32ed", 1)
        return raw

    monkeypatch.setattr(release_module, "_read_resource_bytes", tampered)
    release_module.load_pinned_evidence_provider_runtime_identity.cache_clear()
    with pytest.raises(EvidenceRuntimeReleaseLockError):
        release_module.load_pinned_evidence_provider_runtime_identity()
    release_module.load_pinned_evidence_provider_runtime_identity.cache_clear()


def test_malformed_packaged_lock_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    original = release_module._read_resource_bytes

    def malformed(name: str) -> bytes:
        if name == "runtime-release-v2.lock.json":
            return b'{"release_state":"released","identity":null}'
        return original(name)

    monkeypatch.setattr(release_module, "_read_resource_bytes", malformed)
    release_module.load_evidence_runtime_release_lock.cache_clear()
    with pytest.raises(EvidenceRuntimeReleaseLockError):
        release_module.load_evidence_runtime_release_lock()
    release_module.load_evidence_runtime_release_lock.cache_clear()


def test_duplicate_packaged_lock_key_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = release_module._read_resource_bytes
    valid = original("runtime-release-v2.lock.json")

    def duplicate(name: str) -> bytes:
        if name == "runtime-release-v2.lock.json":
            return valid.replace(
                b'"release_state": "unreleased",',
                b'"release_state": "released", "release_state": "unreleased",',
            )
        return original(name)

    monkeypatch.setattr(release_module, "_read_resource_bytes", duplicate)
    release_module.load_evidence_runtime_release_lock.cache_clear()
    with pytest.raises(EvidenceRuntimeReleaseLockError):
        release_module.load_evidence_runtime_release_lock()
    release_module.load_evidence_runtime_release_lock.cache_clear()


@pytest.mark.parametrize(
    ("field", "invalid"),
    [
        ("contract", "other"),
        ("schema_version", "1.0"),
        ("retrieval_schema_version", "2.0"),
        ("provider_release", "evidence-api-v1.1.0"),
        ("provider_revision", "f" * 39),
        ("provider_build_sha256", "g" * 64),
        ("retrieval_schema_sha256", "g" * 64),
        ("provider_source_manifest_sha256", "g" * 64),
        ("knowledge_generation_id", "generation-name"),
        ("knowledge_generation_attestation_sha256", "g" * 64),
        ("knowledge_source_manifest_sha256", "g" * 64),
        ("model_manifest_sha256", "g" * 64),
        ("index_config_sha256", "g" * 64),
        ("evaluation_report_sha256", "g" * 64),
        ("seal_attestation_sha256", "g" * 64),
        ("knowledge_snapshot_id", ""),
        ("supported_review_as_of", "2026-7-19"),
    ],
)
def test_every_runtime_identity_field_is_validated(field: str, invalid: str) -> None:
    payload = _identity_payload()
    payload[field] = invalid
    hash_input = dict(payload)
    hash_input.pop("runtime_release_sha256")
    payload["runtime_release_sha256"] = hashlib.sha256(canonical_json_bytes(hash_input)).hexdigest()
    with pytest.raises(ValueError, match="identity is invalid"):
        EvidenceRuntimeReleaseIdentity.from_mapping(payload)


def test_runtime_release_hash_is_verified() -> None:
    payload = _identity_payload()
    payload["runtime_release_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="identity is invalid"):
        EvidenceRuntimeReleaseIdentity.from_mapping(payload)


def test_v2_http_happy_path_binds_bearer_get_post_and_snapshot(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    request = _request()
    readiness_response = _Response(_readiness(identity))
    retrieval_response = _Response(
        _response_payload(request, identity.knowledge_snapshot_id),
        _headers(identity),
    )
    sent: list[Any] = []
    responses = iter([readiness_response, retrieval_response])

    def fake_urlopen(http_request: Any, timeout: float) -> _Response:
        sent.append((http_request, timeout))
        return next(responses)

    monkeypatch.setattr(
        retrieval_module,
        "load_released_evidence_runtime_identity",
        lambda: identity,
    )
    monkeypatch.setattr(retrieval_module, "urlopen", fake_urlopen)
    retriever = ProductionHttpRegulationRetriever(
        "http://127.0.0.1:8010",
        service_token="Abcdefghijklmnopqrstuvwxyz0123456789",
        timeout_seconds=3,
        max_retries=0,
    )
    response = retriever.retrieve(request)

    assert response.request_id == request.request_id
    assert [item[0].full_url for item in sent] == [
        "http://127.0.0.1:8010/internal/rag/v2/health/ready",
        "http://127.0.0.1:8010/internal/rag/v2/retrieve",
    ]
    assert all(
        item[0].get_header("Authorization") == "Bearer Abcdefghijklmnopqrstuvwxyz0123456789"
        for item in sent
    )
    sent_headers = {name.casefold(): value for name, value in sent[1][0].header_items()}
    for header_name, expected in _headers(identity).items():
        assert sent_headers[header_name.casefold()] == expected


def test_every_production_retrieval_fetches_fresh_readiness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    request = _request()
    responses = iter(
        [
            _Response(_readiness(identity)),
            _Response(
                _response_payload(request, identity.knowledge_snapshot_id),
                _headers(identity),
            ),
            _Response(_readiness(identity)),
            _Response(
                _response_payload(request, identity.knowledge_snapshot_id),
                _headers(identity),
            ),
        ]
    )
    methods: list[str] = []

    def fake_urlopen(http_request: Any, timeout: float) -> _Response:
        assert timeout > 0
        methods.append(http_request.get_method())
        return next(responses)

    monkeypatch.setattr(
        retrieval_module,
        "load_released_evidence_runtime_identity",
        lambda: identity,
    )
    monkeypatch.setattr(retrieval_module, "urlopen", fake_urlopen)
    retriever = ProductionHttpRegulationRetriever(
        "http://127.0.0.1:8010",
        service_token="Abcdefghijklmnopqrstuvwxyz0123456789",
        max_retries=0,
    )
    retriever.retrieve(request)
    retriever.retrieve(request)
    assert methods == ["GET", "POST", "GET", "POST"]


@pytest.mark.parametrize("header_name", list(_headers(_identity())))
def test_response_identity_mismatch_is_rejected_before_body_read(
    monkeypatch: pytest.MonkeyPatch,
    header_name: str,
) -> None:
    identity = _identity()
    request = _request()
    bad_headers = _headers(identity)
    bad_headers[header_name] = "mismatch"
    readiness_response = _Response(_readiness(identity))
    retrieval_response = _Response(
        _response_payload(request, identity.knowledge_snapshot_id),
        bad_headers,
    )
    responses = iter([readiness_response, retrieval_response])
    monkeypatch.setattr(
        retrieval_module,
        "load_released_evidence_runtime_identity",
        lambda: identity,
    )
    monkeypatch.setattr(
        retrieval_module,
        "urlopen",
        lambda *_args, **_kwargs: next(responses),
    )
    retriever = ProductionHttpRegulationRetriever(
        "http://127.0.0.1:8010",
        service_token="Abcdefghijklmnopqrstuvwxyz0123456789",
        max_retries=0,
    )
    with pytest.raises(RetrievalError, match="identity mismatch"):
        retriever.retrieve(request)
    assert retrieval_response.read_calls == 0


def test_duplicate_response_identity_header_is_rejected_before_body_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    request = _request()
    readiness_response = _Response(_readiness(identity))
    retrieval_response = _Response(
        _response_payload(request, identity.knowledge_snapshot_id),
        _headers(identity),
    )
    retrieval_response.headers = _DuplicateIdentityHeaders(
        _headers(identity),
        EVIDENCE_RUNTIME_RELEASE_SHA256_HEADER,
    )  # type: ignore[assignment]
    responses = iter([readiness_response, retrieval_response])
    monkeypatch.setattr(
        retrieval_module,
        "load_released_evidence_runtime_identity",
        lambda: identity,
    )
    monkeypatch.setattr(
        retrieval_module,
        "urlopen",
        lambda *_args, **_kwargs: next(responses),
    )
    retriever = ProductionHttpRegulationRetriever(
        "http://127.0.0.1:8010",
        service_token="Abcdefghijklmnopqrstuvwxyz0123456789",
        max_retries=0,
    )
    with pytest.raises(RetrievalError, match="identity mismatch"):
        retriever.retrieve(request)
    assert retrieval_response.read_calls == 0


def test_snapshot_mismatch_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    identity = _identity()
    request = _request()
    responses = iter(
        [
            _Response(_readiness(identity)),
            _Response(_response_payload(request, "different-snapshot"), _headers(identity)),
        ]
    )
    monkeypatch.setattr(
        retrieval_module,
        "load_released_evidence_runtime_identity",
        lambda: identity,
    )
    monkeypatch.setattr(
        retrieval_module,
        "urlopen",
        lambda *_args, **_kwargs: next(responses),
    )
    retriever = ProductionHttpRegulationRetriever(
        "http://127.0.0.1:8010",
        service_token="Abcdefghijklmnopqrstuvwxyz0123456789",
        max_retries=0,
    )
    with pytest.raises(RetrievalError, match="snapshot"):
        retriever.retrieve(request)


def test_request_id_mismatch_and_duplicate_json_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    request = _request()
    monkeypatch.setattr(
        retrieval_module,
        "load_released_evidence_runtime_identity",
        lambda: identity,
    )
    for body in (
        _response_payload(
            request.model_copy(update={"request_id": "different-request"}),
            identity.knowledge_snapshot_id,
        ),
        b'{"request_id":"a","request_id":"b"}',
    ):
        responses = iter(
            [
                _Response(_readiness(identity)),
                _Response(body, _headers(identity)),
            ]
        )
        monkeypatch.setattr(
            retrieval_module,
            "urlopen",
            lambda *_args, **_kwargs: next(responses),
        )
        retriever = ProductionHttpRegulationRetriever(
            "http://127.0.0.1:8010",
            service_token="Abcdefghijklmnopqrstuvwxyz0123456789",
            max_retries=0,
        )
        with pytest.raises(RetrievalError):
            retriever.retrieve(request)


def test_412_does_not_retry_or_leak_bearer(monkeypatch: pytest.MonkeyPatch) -> None:
    identity = _identity()
    secret = "SecretBearer_abcdefghijklmnopqrstuvwxyz0123456789"
    calls = 0

    def fake_urlopen(request: Any, timeout: float) -> _Response:
        assert timeout > 0
        nonlocal calls
        calls += 1
        if calls == 1:
            return _Response(_readiness(identity))
        raise HTTPError(request.full_url, 412, "private provider detail", {}, None)

    monkeypatch.setattr(
        retrieval_module,
        "load_released_evidence_runtime_identity",
        lambda: identity,
    )
    monkeypatch.setattr(retrieval_module, "urlopen", fake_urlopen)
    retriever = ProductionHttpRegulationRetriever(
        "http://127.0.0.1:8010",
        service_token=secret,
        max_retries=3,
    )
    with pytest.raises(RetrievalError) as captured:
        retriever.retrieve(_request())
    assert calls == 2
    assert "HTTP status 412" in str(captured.value)
    assert secret not in str(captured.value)
    assert "private provider detail" not in str(captured.value)


def test_post_timeout_retries_only_the_same_attested_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    observed_headers: list[dict[str, str]] = []
    calls = 0

    def fake_urlopen(request: Any, timeout: float) -> _Response:
        assert timeout > 0
        nonlocal calls
        calls += 1
        if calls == 1:
            return _Response(_readiness(identity))
        observed_headers.append({name.casefold(): value for name, value in request.header_items()})
        raise TimeoutError

    monkeypatch.setattr(
        retrieval_module,
        "load_released_evidence_runtime_identity",
        lambda: identity,
    )
    monkeypatch.setattr(retrieval_module, "urlopen", fake_urlopen)
    retriever = ProductionHttpRegulationRetriever(
        "http://127.0.0.1:8010",
        service_token="Abcdefghijklmnopqrstuvwxyz0123456789",
        max_retries=1,
    )
    with pytest.raises(RetrievalError, match="timeout"):
        retriever.retrieve(_request())
    assert len(observed_headers) == 2
    assert observed_headers[0] == observed_headers[1]


def test_get_and_post_share_one_absolute_deadline_against_slow_drip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    slow_response = _SlowDripResponse(
        _response_payload(_request(), identity.knowledge_snapshot_id),
        headers=_headers(identity),
    )
    calls = 0

    def fake_urlopen(_request: Any, timeout: float) -> _Response | _SlowDripResponse:
        assert 0 < timeout <= 0.081
        nonlocal calls
        calls += 1
        if calls == 1:
            return _Response(_readiness(identity))
        return slow_response

    monkeypatch.setattr(
        retrieval_module,
        "load_released_evidence_runtime_identity",
        lambda: identity,
    )
    monkeypatch.setattr(retrieval_module, "urlopen", fake_urlopen)
    retriever = ProductionHttpRegulationRetriever(
        "http://127.0.0.1:8010",
        service_token="Abcdefghijklmnopqrstuvwxyz0123456789",
        timeout_seconds=0.08,
        max_retries=3,
    )
    started = time.monotonic()
    with pytest.raises(RetrievalError, match="timeout"):
        retriever.retrieve(_request())
    elapsed = time.monotonic() - started

    assert elapsed < 0.3
    assert calls == 2
    assert slow_response.closed.wait(0.2)


def test_readiness_absolute_deadline_bounds_uncooperative_header_open(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    identity = _identity()
    calls = 0

    def uncooperative_urlopen(_request: Any, timeout: float) -> _Response:
        assert 0 < timeout <= 0.051
        nonlocal calls
        calls += 1
        time.sleep(0.2)
        return _Response(_readiness(identity))

    monkeypatch.setattr(
        retrieval_module,
        "load_released_evidence_runtime_identity",
        lambda: identity,
    )
    monkeypatch.setattr(retrieval_module, "urlopen", uncooperative_urlopen)
    started = time.monotonic()
    with pytest.raises(RetrievalError, match="timeout"):
        fetch_evidence_runtime_readiness(
            "http://127.0.0.1:8010",
            service_token="Abcdefghijklmnopqrstuvwxyz0123456789",
            timeout_seconds=0.05,
            max_retries=2,
        )
    elapsed = time.monotonic() - started

    assert elapsed < 0.15
    assert calls == 1


def test_redirect_status_is_not_followed(monkeypatch: pytest.MonkeyPatch) -> None:
    identity = _identity()
    calls = 0

    def fake_urlopen(request: Any, timeout: float) -> _Response:
        assert timeout > 0
        nonlocal calls
        calls += 1
        if calls == 1:
            return _Response(_readiness(identity))
        raise HTTPError(
            request.full_url,
            302,
            "redirect",
            {"Location": "https://attacker.invalid"},
            None,
        )

    monkeypatch.setattr(
        retrieval_module,
        "load_released_evidence_runtime_identity",
        lambda: identity,
    )
    monkeypatch.setattr(retrieval_module, "urlopen", fake_urlopen)
    retriever = ProductionHttpRegulationRetriever(
        "http://127.0.0.1:8010",
        service_token="Abcdefghijklmnopqrstuvwxyz0123456789",
        max_retries=0,
    )
    with pytest.raises(RetrievalError, match="HTTP status 302"):
        retriever.retrieve(_request())
    assert calls == 2


def test_missing_production_bearer_fails_before_lock_or_network(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def forbidden(*_args: Any, **_kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        raise AssertionError

    monkeypatch.setattr(retrieval_module, "urlopen", forbidden)
    monkeypatch.setattr(
        retrieval_module,
        "load_released_evidence_runtime_identity",
        forbidden,
    )
    with pytest.raises(ValueError, match="bearer token is required"):
        ProductionHttpRegulationRetriever(
            "http://127.0.0.1:8010",
            service_token=None,
        )
    assert calls == 0


def test_release_lock_builder_requires_exact_data_code_and_schema_lock() -> None:
    fixture_root = Path(__file__).parent / "fixtures" / "evidence_runtime_v2"
    lock_bytes = (fixture_root / "provider-runtime-lock.json").read_bytes()
    assert hashlib.sha256(lock_bytes).hexdigest() == (
        "e3e16ef56e2e2ce7106b7a5dce2eb3b3fb2618e7658f97742c76bbf47de99a5e"
    )
    expected = json.loads(lock_bytes)
    identity = _identity(
        provider_revision=expected["provider_revision"],
        provider_build_sha256=expected["provider_build_sha256"],
        provider_source_manifest_sha256=expected["provider_source_manifest_sha256"],
    )
    readiness_schema = release_module._read_resource_bytes("runtime-release-v2.schema.json")
    retrieval_schema = release_module._read_resource_bytes("v1.schema.json")
    assert expected["readiness_schema_sha256"] == hashlib.sha256(readiness_schema).hexdigest()
    assert expected["retrieval_schema_sha256"] == hashlib.sha256(retrieval_schema).hexdigest()
    result = build_released_lock(
        readiness=json.loads(_readiness(identity)),
        expected_data_lock=expected,
        readiness_schema_bytes=readiness_schema,
        retrieval_schema_bytes=retrieval_schema,
    )
    assert result["release_state"] == "released"
    assert result["identity"] == identity.to_wire()

    obsolete_provider_build = {
        "contract": "evidence-retrieval",
        "schema_version": "1.0",
        "provider_release": "evidence-api-v1.2.0",
        "provider_revision": expected["provider_revision"],
        "provider_build_sha256": expected["provider_build_sha256"],
        "schema_sha256": expected["retrieval_schema_sha256"],
        "source_manifest_sha256": expected["provider_source_manifest_sha256"],
    }
    with pytest.raises(ReleaseLockBuildError, match="expected_data_lock_invalid"):
        build_released_lock(
            readiness=json.loads(_readiness(identity)),
            expected_data_lock=obsolete_provider_build,
            readiness_schema_bytes=readiness_schema,
            retrieval_schema_bytes=retrieval_schema,
        )

    tampered = {**expected, "readiness_schema_sha256": "f" * 64}
    with pytest.raises(ReleaseLockBuildError, match="data_schema_does_not_match"):
        build_released_lock(
            readiness=json.loads(_readiness(identity)),
            expected_data_lock=tampered,
            readiness_schema_bytes=readiness_schema,
            retrieval_schema_bytes=retrieval_schema,
        )

    with pytest.raises(ReleaseLockBuildError, match="schema_bytes"):
        build_released_lock(
            readiness=json.loads(_readiness(identity)),
            expected_data_lock=expected,
            readiness_schema_bytes=readiness_schema + b"\n",
            retrieval_schema_bytes=retrieval_schema,
        )
