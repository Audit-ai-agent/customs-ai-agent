"""Strict HTTP consumer for the released Evidence API v3."""

from __future__ import annotations

import json
import threading
import time
from datetime import date
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request

from jsonschema import Draft202012Validator, FormatChecker
from pydantic import ValidationError

from customs_ai_agent.adapters.retrieval import (
    RetrievalError,
    _EvidenceDeadlineExceeded,
    _ResponseIdentityValidationError,
    _load_strict_json,
    _new_deadline,
    _optional_auth_header,
    _perform_http_exchange,
    _response_header_value,
    _safe_failure_reason,
    _validate_transport_settings,
    validate_evidence_destination,
)
from customs_ai_agent.contracts.evidence_v3 import (
    EvidenceRuntimeReleaseIdentityV3,
    EvidenceV3ContractError,
    load_evidence_v3_schema,
    load_released_evidence_runtime_identity_v3,
    parse_evidence_runtime_readiness_v3,
)
from customs_ai_agent.schemas.models import EvidenceRetrievalRequest, EvidenceRetrievalResponse


def _validate_payload(name: str, payload: Any) -> None:
    schema = load_evidence_v3_schema(name)
    error = next(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(payload),
        None,
    )
    if error is not None:
        raise EvidenceV3ContractError("Evidence v3 payload does not satisfy pinned Schema")


def _validate_response_headers(
    response: Any,
    identity: EvidenceRuntimeReleaseIdentityV3,
    *,
    request_id: str,
) -> None:
    for name, expected in identity.binding_headers().items():
        if _response_header_value(response, name) != expected:
            raise _ResponseIdentityValidationError("runtime v3 release identity mismatch")
    if _response_header_value(response, "X-Request-ID") != request_id:
        raise _ResponseIdentityValidationError("request identity mismatch")


def fetch_evidence_runtime_readiness_v3(
    base_url: str,
    *,
    service_token: str | None,
    allowed_origins: set[str] | frozenset[str] | None = None,
    timeout_seconds: float = 10.0,
    max_retries: int = 1,
    _deadline: float | None = None,
) -> EvidenceRuntimeReleaseIdentityV3:
    normalized = validate_evidence_destination(base_url, allowed_origins)
    _validate_transport_settings(timeout_seconds, max_retries)
    expected = load_released_evidence_runtime_identity_v3()
    deadline = _new_deadline(timeout_seconds) if _deadline is None else _deadline
    headers = {"Accept": "application/json"}
    headers.update(_optional_auth_header(service_token, "Authorization"))
    url = f"{normalized}/internal/rag/v3/health/ready"
    last_error: Exception | None = None
    for _ in range(max_retries + 1):
        try:
            request = Request(url, headers=headers, method="GET")  # noqa: S310
            payload = _load_strict_json(_perform_http_exchange(request, deadline=deadline))
            observed = parse_evidence_runtime_readiness_v3(payload)
            if observed != expected:
                raise ValueError("runtime v3 release identity mismatch")
            return observed
        except (HTTPError, URLError, TimeoutError, ValueError, EvidenceV3ContractError) as exc:
            last_error = exc
            if isinstance(exc, _EvidenceDeadlineExceeded) or time.monotonic() >= deadline:
                last_error = _EvidenceDeadlineExceeded()
                break
    reason = _safe_failure_reason(last_error or RuntimeError("internal state"))
    raise RetrievalError(f"Evidence v3 readiness request failed: {reason}") from None


def _project_v3_response_to_v1(payload: dict[str, Any]) -> dict[str, Any]:
    projected = json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False))
    projected["schema_version"] = "1.0"
    for result in projected.get("results") or []:
        for evidence in result.get("evidence") or []:
            evidence.pop("requirement_verification", None)
    return projected


class ProductionHttpRegulationRetrieverV3:
    """No-fallback consumer bound to all seventeen V3 runtime headers."""

    endpoint_path = "/internal/rag/v3/retrieve"

    def __init__(
        self,
        base_url: str,
        *,
        service_token: str | None,
        allowed_origins: set[str] | frozenset[str] | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 1,
    ) -> None:
        self._base_url = validate_evidence_destination(base_url, allowed_origins)
        _validate_transport_settings(timeout_seconds, max_retries)
        if service_token is None:
            raise ValueError("Production Evidence bearer token is required")
        self._auth_header = _optional_auth_header(service_token, "Authorization")
        self._expected_identity = load_released_evidence_runtime_identity_v3()
        self._service_token = service_token
        self._allowed_origins = allowed_origins
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.url = f"{self._base_url}{self.endpoint_path}"
        self._local = threading.local()

    @property
    def expected_identity(self) -> EvidenceRuntimeReleaseIdentityV3:
        return self._expected_identity

    def preflight(
        self,
        review_as_of: date | None = None,
        *,
        _deadline: float | None = None,
    ) -> EvidenceRuntimeReleaseIdentityV3:
        deadline = _new_deadline(self.timeout_seconds) if _deadline is None else _deadline
        identity = fetch_evidence_runtime_readiness_v3(
            self._base_url,
            service_token=self._service_token,
            allowed_origins=self._allowed_origins,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            _deadline=deadline,
        )
        if identity != self._expected_identity:
            raise RetrievalError("Evidence runtime v3 release identity does not match")
        if (
            review_as_of is not None
            and review_as_of != identity.base_runtime_release.supported_review_as_of
        ):
            raise RetrievalError("Evidence runtime v3 does not support review_as_of")
        return identity

    def retrieve(self, request: EvidenceRetrievalRequest) -> EvidenceRetrievalResponse:
        # A failed or concurrent request must never expose a prior request's
        # approved policy observations to the compliance stage.
        self._local.request_id = None
        self._local.policy_bindings = {}
        deadline = _new_deadline(self.timeout_seconds)
        request_payload = request.model_dump(mode="json")
        request_payload["schema_version"] = "3.0"
        _validate_payload("rag_retrieval_v3.schema.json", request_payload)
        observed = self.preflight(request.review_as_of, _deadline=deadline)
        body = json.dumps(
            request_payload,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json; charset=utf-8",
            "X-Request-ID": request.request_id,
            **observed.binding_headers(),
            **self._auth_header,
        }
        last_error: Exception | None = None
        for _ in range(self.max_retries + 1):
            try:
                http_request = Request(  # noqa: S310
                    self.url,
                    data=body,
                    headers=headers,
                    method="POST",
                )
                raw = _perform_http_exchange(
                    http_request,
                    deadline=deadline,
                    header_validator=lambda response: _validate_response_headers(
                        response,
                        observed,
                        request_id=request.request_id,
                    ),
                )
                response_payload = _load_strict_json(raw)
                _validate_payload("rag_retrieval_v3.schema.json", response_payload)
                if response_payload.get("request_id") != request.request_id:
                    raise ValueError("Evidence v3 request identity mismatch")
                if response_payload.get("knowledge_snapshot_id") != (
                    observed.base_runtime_release.knowledge_snapshot_id
                ):
                    raise ValueError("Evidence v3 snapshot mismatch")
                policy_bindings: dict[str, dict[str, Any]] = {}
                for result in response_payload.get("results") or []:
                    for evidence in result.get("evidence") or []:
                        if evidence.get("source_type") != "customs_requirement":
                            continue
                        binding = evidence.get("requirement_verification")
                        if (
                            not isinstance(binding, dict)
                            or not binding.get("approved")
                            or binding.get("policy_release") != observed.requirement_policy_release
                        ):
                            if response_payload.get("status") == "ok":
                                raise ValueError(
                                    "Evidence v3 returned unapproved policy as complete"
                                )
                        code = evidence.get("required_document_code")
                        if isinstance(code, str) and isinstance(binding, dict):
                            existing = policy_bindings.get(code)
                            if existing is not None and existing != binding:
                                raise ValueError(
                                    "Evidence v3 returned conflicting requirement policy"
                                )
                            policy_bindings[code] = dict(binding)
                parsed = EvidenceRetrievalResponse.model_validate(
                    _project_v3_response_to_v1(response_payload)
                )
                self._local.request_id = request.request_id
                self._local.policy_bindings = policy_bindings
                return parsed
            except _ResponseIdentityValidationError as exc:
                raise RetrievalError(
                    f"Evidence v3 request failed: {_safe_failure_reason(exc)}"
                ) from None
            except HTTPError as exc:
                if exc.code == 412:
                    raise RetrievalError("Evidence v3 request failed: HTTP status 412") from None
                last_error = exc
            except (
                URLError,
                TimeoutError,
                ValidationError,
                EvidenceV3ContractError,
                ValueError,
            ) as exc:
                last_error = exc
                if isinstance(exc, _EvidenceDeadlineExceeded) or time.monotonic() >= deadline:
                    last_error = _EvidenceDeadlineExceeded()
                    break
        reason = _safe_failure_reason(last_error or RuntimeError("internal state"))
        raise RetrievalError(f"Evidence v3 request failed: {reason}") from None

    def policy_bindings(self, request_id: str) -> dict[str, dict[str, Any]]:
        if getattr(self._local, "request_id", None) != request_id:
            return {}
        value = getattr(self._local, "policy_bindings", None)
        return dict(value) if isinstance(value, dict) else {}


__all__ = ["ProductionHttpRegulationRetrieverV3", "fetch_evidence_runtime_readiness_v3"]
