from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Any, Literal, Protocol, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import HTTPRedirectHandler, Request, build_opener

from pydantic import ValidationError

from customs_ai_agent.contracts.evidence import (
    EvidenceContractIdentity,
    EvidenceContractLockError,
    EvidenceContractPayloadError,
    load_pinned_evidence_contract_identity,
    load_pinned_evidence_schema,
    validate_pinned_evidence_payload,
)
from customs_ai_agent.fixtures.knowledge import KNOWLEDGE_BASE
from customs_ai_agent.schemas.models import (
    Coverage,
    EvidenceRetrievalRequest,
    EvidenceRetrievalResponse,
    HsResolution,
    RagEvidence,
    RetrievalStatus,
    RetrievalErrorDetail,
    RetrievalItemResult,
    RetrievalQueryItem,
    stable_evidence_id,
)


class RetrievalError(RuntimeError):
    """The Evidence Service could not return a contract-valid response."""


EvidenceContractIdentityMode = Literal["disabled", "optional", "required"]


@dataclass(frozen=True, slots=True)
class EvidenceReadiness:
    knowledge_snapshot_id: str
    supported_review_as_of: date
    contract_identity: EvidenceContractIdentity | None


class _ReadinessValidationError(ValueError):
    def __init__(self, safe_reason: str) -> None:
        super().__init__(safe_reason)
        self.safe_reason = safe_reason


class _ResponseIdentityValidationError(ValueError):
    def __init__(self, safe_reason: str) -> None:
        super().__init__(safe_reason)
        self.safe_reason = safe_reason


_HEADER_NAME = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_ISO_CALENDAR_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
EVIDENCE_SCHEMA_SHA256_HEADER = "X-Evidence-Schema-SHA256"
EVIDENCE_PROVIDER_RELEASE_HEADER = "X-Evidence-Provider-Release"
EVIDENCE_PROVIDER_REVISION_HEADER = "X-Evidence-Provider-Revision"
EVIDENCE_PROVIDER_BUILD_SHA256_HEADER = "X-Evidence-Provider-Build-SHA256"
_EVIDENCE_IDENTITY_HEADER_FIELDS = (
    (EVIDENCE_SCHEMA_SHA256_HEADER, "schema_sha256"),
    (EVIDENCE_PROVIDER_RELEASE_HEADER, "provider_release"),
    (EVIDENCE_PROVIDER_REVISION_HEADER, "provider_revision"),
    (EVIDENCE_PROVIDER_BUILD_SHA256_HEADER, "provider_build_sha256"),
)
_RESERVED_HEADERS = {
    "accept",
    "connection",
    "content-length",
    "content-type",
    "host",
    "transfer-encoding",
    EVIDENCE_SCHEMA_SHA256_HEADER.casefold(),
    EVIDENCE_PROVIDER_RELEASE_HEADER.casefold(),
    EVIDENCE_PROVIDER_REVISION_HEADER.casefold(),
    EVIDENCE_PROVIDER_BUILD_SHA256_HEADER.casefold(),
    "x-request-id",
}
MAX_EVIDENCE_RESPONSE_BYTES = 2 * 1024 * 1024


class _NoRedirectHandler(HTTPRedirectHandler):
    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        # Evidence endpoints are exact-origin contracts. Redirects could
        # forward document data or authorization headers to another origin.
        return None


_NO_REDIRECT_OPENER = build_opener(_NoRedirectHandler())


def urlopen(request: Request, timeout: float):
    """Open one exact Evidence URL without following redirects."""

    return _NO_REDIRECT_OPENER.open(request, timeout=timeout)


def _read_bounded_response(response: Any) -> bytes:
    payload = response.read(MAX_EVIDENCE_RESPONSE_BYTES + 1)
    if len(payload) > MAX_EVIDENCE_RESPONSE_BYTES:
        raise ValueError("Evidence Service response exceeded the byte limit")
    return payload


def _reject_nonstandard_json_constant(_value: str) -> None:
    raise ValueError("Evidence Service response contains a non-standard JSON number")


def _reject_duplicate_json_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for key, value in pairs:
        if key in payload:
            raise ValueError("Evidence Service response contains a duplicate JSON key")
        payload[key] = value
    return payload


def _load_strict_json(payload: bytes) -> Any:
    return json.loads(
        payload,
        parse_constant=_reject_nonstandard_json_constant,
        object_pairs_hook=_reject_duplicate_json_keys,
    )


def _is_loopback_host(hostname: str) -> bool:
    if hostname.casefold() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def validate_evidence_base_url(base_url: str) -> str:
    """Validate and normalize an Evidence Service origin without resolving DNS.

    Plain HTTP is intentionally limited to an explicit loopback hostname or IP.
    Network services must use HTTPS. Credentials, query strings, fragments and path
    prefixes are rejected so secrets cannot be smuggled into a logged request URL.
    """

    value = base_url.strip()
    if not value:
        raise ValueError("Evidence Service base URL is required")
    if any(character.isspace() for character in value):
        raise ValueError("Evidence Service base URL cannot contain whitespace")
    try:
        parsed = urlsplit(value)
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Evidence Service base URL is invalid") from exc
    scheme = parsed.scheme.casefold()
    if scheme not in {"http", "https"}:
        raise ValueError("Evidence Service base URL must use http or https")
    if not parsed.hostname:
        raise ValueError("Evidence Service base URL must include a hostname")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("Evidence Service base URL cannot contain credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("Evidence Service base URL cannot contain a query or fragment")
    if parsed.path not in {"", "/"}:
        raise ValueError("Evidence Service base URL cannot contain a path prefix")
    hostname = parsed.hostname.casefold()
    if scheme == "http" and not _is_loopback_host(hostname):
        raise ValueError("Plain HTTP Evidence Service URLs are allowed only on loopback")
    rendered_host = f"[{hostname}]" if ":" in hostname else hostname
    netloc = f"{rendered_host}:{port}" if port is not None else rendered_host
    return urlunsplit((scheme, netloc, "", "", ""))


def validate_evidence_destination(
    base_url: str,
    allowed_origins: set[str] | frozenset[str] | None,
) -> str:
    normalized = validate_evidence_base_url(base_url)
    hostname = urlsplit(normalized).hostname
    if hostname is None:
        raise ValueError("Evidence Service base URL must include a host")
    if _is_loopback_host(hostname):
        return normalized
    normalized_allowed = {
        validate_evidence_base_url(origin) for origin in (allowed_origins or set())
    }
    if normalized not in normalized_allowed:
        raise ValueError("External Evidence Service origin is not in the explicit allowlist")
    return normalized


def _optional_auth_header(
    service_token: str | None,
    token_header: str,
) -> dict[str, str]:
    if not _HEADER_NAME.fullmatch(token_header):
        raise ValueError("Evidence Service token header name is invalid")
    if token_header.casefold() in _RESERVED_HEADERS:
        raise ValueError("Evidence Service token header cannot override a protocol header")
    if service_token is None:
        return {}
    if not service_token.strip():
        raise ValueError("Evidence Service token cannot be blank")
    if "\r" in service_token or "\n" in service_token:
        raise ValueError("Evidence Service token cannot contain line breaks")
    return {token_header: service_token}


def _safe_failure_reason(exc: Exception) -> str:
    if isinstance(exc, (_ReadinessValidationError, _ResponseIdentityValidationError)):
        return exc.safe_reason
    if isinstance(exc, HTTPError):
        return f"HTTP status {exc.code}"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, URLError):
        return "network error"
    return "invalid contract response"


def _evidence_identity_headers(
    identity: EvidenceContractIdentity,
) -> dict[str, str]:
    return {
        header_name: str(getattr(identity, field_name))
        for header_name, field_name in _EVIDENCE_IDENTITY_HEADER_FIELDS
    }


def _response_header_value(response: Any, header_name: str) -> str | None:
    """Read one response header without trusting provider exception details."""

    try:
        getheader = getattr(response, "getheader", None)
        if callable(getheader):
            value = getheader(header_name)
        else:
            response_headers = getattr(response, "headers", None)
            get_value = getattr(response_headers, "get", None)
            value = get_value(header_name) if callable(get_value) else None
    except Exception:
        raise _ResponseIdentityValidationError("contract identity mismatch") from None
    return value if isinstance(value, str) else None


def _validate_response_identity_headers(
    response: Any,
    expected_identity: EvidenceContractIdentity,
) -> None:
    """Reject a rollout/build mismatch before reading any Evidence body bytes."""

    for header_name, expected_value in _evidence_identity_headers(expected_identity).items():
        if _response_header_value(response, header_name) != expected_value:
            raise _ResponseIdentityValidationError("contract identity mismatch")


def _validate_transport_settings(timeout_seconds: float, max_retries: int) -> None:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")


def _validate_contract_identity_mode(mode: object) -> EvidenceContractIdentityMode:
    if mode not in {"disabled", "optional", "required"}:
        raise ValueError("Evidence contract identity mode is invalid")
    return mode


def _parse_evidence_readiness(
    payload: object,
    *,
    contract_identity_mode: EvidenceContractIdentityMode,
    expected_identity: EvidenceContractIdentity | None,
) -> EvidenceReadiness:
    if not isinstance(payload, dict) or payload.get("status") != "ready":
        raise _ReadinessValidationError("invalid contract response")
    snapshot_id = payload.get("knowledge_snapshot_id")
    if not isinstance(snapshot_id, str) or not snapshot_id.strip():
        raise _ReadinessValidationError("invalid contract response")
    raw_date = payload.get("supported_review_as_of")
    if not isinstance(raw_date, str) or not _ISO_CALENDAR_DATE.fullmatch(raw_date):
        raise _ReadinessValidationError("invalid contract response")
    try:
        supported_date = date.fromisoformat(raw_date)
    except ValueError:
        raise _ReadinessValidationError("invalid contract response") from None

    identity: EvidenceContractIdentity | None = None
    if contract_identity_mode != "disabled":
        raw_identity = payload.get("evidence_contract")
        if raw_identity is None:
            if contract_identity_mode == "required":
                raise _ReadinessValidationError("contract identity missing")
        else:
            try:
                identity = EvidenceContractIdentity.from_mapping(raw_identity)
            except ValueError:
                raise _ReadinessValidationError("invalid contract response") from None
            if expected_identity is None or identity != expected_identity:
                raise _ReadinessValidationError("contract identity mismatch")

    return EvidenceReadiness(
        knowledge_snapshot_id=snapshot_id.strip(),
        supported_review_as_of=supported_date,
        contract_identity=identity,
    )


def fetch_evidence_readiness(
    base_url: str,
    *,
    service_token: str | None = None,
    token_header: str = "X-Demo-Token",  # noqa: S107 - header name, not a credential
    allowed_origins: set[str] | frozenset[str] | None = None,
    timeout_seconds: float = 10.0,
    max_retries: int = 1,
    contract_identity_mode: EvidenceContractIdentityMode = "required",
) -> EvidenceReadiness:
    """Read and optionally attest the provider's Evidence readiness identity."""

    normalized_base_url = validate_evidence_destination(base_url, allowed_origins)
    _validate_transport_settings(timeout_seconds, max_retries)
    identity_mode = _validate_contract_identity_mode(contract_identity_mode)
    expected_identity: EvidenceContractIdentity | None = None
    if identity_mode != "disabled":
        try:
            expected_identity = load_pinned_evidence_contract_identity()
            load_pinned_evidence_schema()
        except EvidenceContractLockError:
            raise RetrievalError("Evidence consumer contract bundle is invalid") from None
    headers = {"Accept": "application/json"}
    headers.update(_optional_auth_header(service_token, token_header))
    readiness_url = f"{normalized_base_url}/internal/rag/v1/health/ready"
    last_error: Exception | None = None
    for _ in range(max_retries + 1):
        try:
            # validate_evidence_destination already restricted scheme/origin.
            request = Request(  # noqa: S310
                readiness_url,
                headers=headers,
                method="GET",
            )
            with urlopen(request, timeout=timeout_seconds) as response:
                payload = _load_strict_json(_read_bounded_response(response))
            return _parse_evidence_readiness(
                payload,
                contract_identity_mode=identity_mode,
                expected_identity=expected_identity,
            )
        except (
            HTTPError,
            URLError,
            TimeoutError,
            json.JSONDecodeError,
            ValueError,
        ) as exc:
            last_error = exc
    if last_error is None:
        raise RetrievalError("Evidence Service readiness request failed: internal state") from None
    reason = _safe_failure_reason(last_error)
    raise RetrievalError(f"Evidence Service readiness request failed: {reason}") from None


def fetch_supported_review_as_of(
    base_url: str,
    *,
    service_token: str | None = None,
    token_header: str = "X-Demo-Token",  # noqa: S107 - header name, not a credential
    allowed_origins: set[str] | frozenset[str] | None = None,
    timeout_seconds: float = 10.0,
    max_retries: int = 1,
    contract_identity_mode: EvidenceContractIdentityMode = "required",
) -> date:
    """Return the provider-supported date after the configured readiness check."""

    return fetch_evidence_readiness(
        base_url,
        service_token=service_token,
        token_header=token_header,
        allowed_origins=allowed_origins,
        timeout_seconds=timeout_seconds,
        max_retries=max_retries,
        contract_identity_mode=contract_identity_mode,
    ).supported_review_as_of


@runtime_checkable
class RegulationRetriever(Protocol):
    def retrieve(self, request: EvidenceRetrievalRequest) -> EvidenceRetrievalResponse:
        """Retrieve item-bound regulatory evidence for one review run."""


def _hs_resolution(item: RetrievalQueryItem, matched: bool = False) -> HsResolution:
    if not item.hs_code:
        return "missing"
    if not item.hs_code.isdigit() or len(item.hs_code) not in {6, 10}:
        return "invalid"
    if len(item.hs_code) == 10:
        return "exact"
    return "unique_descendant" if matched else "invalid"


def _failure_result(
    item: RetrievalQueryItem,
    *,
    coverage: Coverage,
    code: str,
    message: str,
) -> RetrievalItemResult:
    return RetrievalItemResult(
        item_id=item.item_id,
        hs_resolution="missing" if not item.hs_code else "invalid",
        resolved_hsk10=None,
        candidate_hsk10s=[],
        coverage=coverage,
        requirements_found=False,
        evidence=[],
        errors=[RetrievalErrorDetail(code=code, message=message, retryable=True)],
    )


class MockRegulationRetriever:
    """Contract-faithful adapter around the existing deterministic fixture."""

    _resolved_codes = {
        "731815": "7318151000",
        "030354": "0303540000",
    }
    _document_names = {
        "quarantine_certificate": "Quarantine certificate",
    }

    @staticmethod
    def _matches(item: RetrievalQueryItem, entry: dict[str, Any]) -> bool:
        description = (item.description or "").casefold()
        if item.hs_code and item.hs_code in entry.get("hs_codes", []):
            return True
        return any(keyword in description for keyword in entry.get("keywords", []))

    def retrieve(self, request: EvidenceRetrievalRequest) -> EvidenceRetrievalResponse:
        results: list[RetrievalItemResult] = []
        for item in request.items:
            matches = [entry for entry in KNOWLEDGE_BASE if self._matches(item, entry)]
            resolution = _hs_resolution(item, matched=bool(matches))
            resolved = (
                item.hs_code
                if item.hs_code and len(item.hs_code) == 10 and item.hs_code.isdigit()
                else self._resolved_codes.get(item.hs_code or "")
            )
            evidence: list[RagEvidence] = []
            for entry in matches:
                required_code = entry.get("required_document")
                score = float(entry["score"])
                version_hash = hashlib.sha256(entry["version_hash"].encode("utf-8")).hexdigest()
                evidence_id = stable_evidence_id(entry["chunk_id"], version_hash)
                evidence.append(
                    RagEvidence(
                        evidence_id=evidence_id,
                        source_record_id=entry["chunk_id"],
                        source_type=(
                            "hsk_entry" if entry["source_type"] == "hs_code" else "notice"
                        ),
                        item_id=item.item_id,
                        title=entry["title"],
                        text=entry["text"],
                        law_id=None,
                        law_name=entry.get("law_name"),
                        article_no=entry.get("article_no"),
                        source_uri=entry["source_uri"],
                        valid_from=entry.get("effective_date"),
                        valid_to=None,
                        version_hash=version_hash,
                        matched_hs_code=resolved or item.hs_code,
                        trade_type=request.trade_type,
                        required_document_code=required_code,
                        required_document_name=(
                            self._document_names.get(required_code, required_code)
                            if required_code
                            else None
                        ),
                        agencies=[],
                        retrieval_methods=["exact_hs"],
                        # Fixture HSK/notice hits may support an explanation, but only
                        # structured customs_requirement records can create findings.
                        eligibility="supporting",
                        score=score,
                        raw_scores={"mock": score},
                    )
                )

            requirements_found = any(
                evidence_item.source_type == "customs_requirement"
                and evidence_item.eligibility == "finding"
                for evidence_item in evidence
            )
            complete = (
                resolution in {"exact", "unique_descendant"}
                and bool(matches)
                and all(
                    evidence_item.eligibility != "blocked" and (evidence_item.score or 0) >= 0.75
                    for evidence_item in evidence
                )
            )
            coverage: Coverage = "complete" if complete else ("partial" if matches else "unknown")
            errors = []
            if coverage != "complete":
                errors.append(
                    RetrievalErrorDetail(
                        code="MOCK_COVERAGE_NOT_VERIFIED",
                        message="The mock fixture cannot establish complete regulatory coverage.",
                        retryable=False,
                    )
                )
            results.append(
                RetrievalItemResult(
                    item_id=item.item_id,
                    hs_resolution=resolution,
                    resolved_hsk10=resolved,
                    candidate_hsk10s=[resolved] if resolved else [],
                    coverage=coverage,
                    requirements_found=requirements_found,
                    evidence=evidence,
                    errors=errors,
                )
            )

        status: RetrievalStatus = (
            "ok" if all(result.coverage == "complete" for result in results) else "partial"
        )
        return EvidenceRetrievalResponse(
            request_id=request.request_id,
            status=status,
            knowledge_snapshot_id="mock-knowledge-v1",
            results=results,
            errors=[],
        )


class EmptyRegulationRetriever:
    """Returns no hits without claiming that absence is a verified negative."""

    def retrieve(self, request: EvidenceRetrievalRequest) -> EvidenceRetrievalResponse:
        return EvidenceRetrievalResponse(
            request_id=request.request_id,
            status="partial",
            knowledge_snapshot_id="empty-test-snapshot",
            results=[
                _failure_result(
                    item,
                    coverage="unknown",
                    code="NO_VERIFIED_COVERAGE",
                    message="No evidence was returned and negative coverage was not verified.",
                )
                for item in request.items
            ],
            errors=[
                RetrievalErrorDetail(
                    code="NO_VERIFIED_COVERAGE",
                    message="No evidence was returned and negative coverage was not verified.",
                    retryable=False,
                )
            ],
        )


class PartialRegulationRetriever:
    def retrieve(self, request: EvidenceRetrievalRequest) -> EvidenceRetrievalResponse:
        return EvidenceRetrievalResponse(
            request_id=request.request_id,
            status="partial",
            knowledge_snapshot_id="partial-test-snapshot",
            results=[
                _failure_result(
                    item,
                    coverage="partial",
                    code="PARTIAL_RETRIEVAL",
                    message="One or more evidence stores were unavailable.",
                )
                for item in request.items
            ],
            errors=[
                RetrievalErrorDetail(
                    code="PARTIAL_RETRIEVAL",
                    message="One or more evidence stores were unavailable.",
                    retryable=True,
                )
            ],
        )


class FailingRegulationRetriever:
    def retrieve(self, request: EvidenceRetrievalRequest) -> EvidenceRetrievalResponse:
        raise RetrievalError("Evidence Service is unavailable")


class HttpRegulationRetriever:
    endpoint_path = "/internal/rag/v1/retrieve"

    def __init__(
        self,
        base_url: str,
        *,
        service_token: str | None = None,
        token_header: str = "X-Demo-Token",  # noqa: S107 - header name, not a credential
        allowed_origins: set[str] | frozenset[str] | None = None,
        timeout_seconds: float = 60.0,
        max_retries: int = 1,
        contract_identity_mode: EvidenceContractIdentityMode = "required",
    ) -> None:
        normalized_base_url = validate_evidence_destination(
            base_url,
            allowed_origins,
        )
        _validate_transport_settings(timeout_seconds, max_retries)
        identity_mode = _validate_contract_identity_mode(contract_identity_mode)
        try:
            load_pinned_evidence_contract_identity()
            load_pinned_evidence_schema()
        except EvidenceContractLockError:
            raise RetrievalError("Evidence consumer contract bundle is invalid") from None
        self._base_url = normalized_base_url
        self.url = f"{normalized_base_url}{self.endpoint_path}"
        self._service_token = service_token
        self._token_header = token_header
        self._allowed_origins = allowed_origins
        self._auth_header = _optional_auth_header(service_token, token_header)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.contract_identity_mode = identity_mode

    def _ensure_readiness(
        self,
        request: EvidenceRetrievalRequest,
    ) -> EvidenceReadiness | None:
        if self.contract_identity_mode == "disabled":
            return None
        # Re-attest every retrieval. A long-lived AgentRuntime must not keep
        # trusting an identity observed before a provider restart or rollout.
        readiness = fetch_evidence_readiness(
            self._base_url,
            service_token=self._service_token,
            token_header=self._token_header,
            allowed_origins=self._allowed_origins,
            timeout_seconds=self.timeout_seconds,
            max_retries=self.max_retries,
            contract_identity_mode=self.contract_identity_mode,
        )
        if request.review_as_of != readiness.supported_review_as_of:
            raise RetrievalError(
                "Evidence Service readiness does not support the requested review_as_of"
            )
        return readiness

    def retrieve(self, request: EvidenceRetrievalRequest) -> EvidenceRetrievalResponse:
        request_payload = request.model_dump(mode="json")
        try:
            validate_pinned_evidence_payload(request_payload)
        except (EvidenceContractLockError, EvidenceContractPayloadError):
            raise RetrievalError("Evidence request does not satisfy the pinned contract") from None
        readiness = self._ensure_readiness(request)
        payload = json.dumps(
            request_payload,
            ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json; charset=utf-8",
            "X-Request-ID": request.request_id,
        }
        response_identity = readiness.contract_identity if readiness is not None else None
        if response_identity is not None:
            headers.update(_evidence_identity_headers(response_identity))
        headers.update(self._auth_header)
        last_error: Exception | None = None
        for _ in range(self.max_retries + 1):
            try:
                # Constructor validation restricted this URL's scheme/origin.
                http_request = Request(  # noqa: S310
                    self.url,
                    data=payload,
                    headers=headers,
                    method="POST",
                )
                with urlopen(
                    http_request,
                    timeout=self.timeout_seconds,
                ) as response:
                    if response_identity is not None:
                        _validate_response_identity_headers(
                            response,
                            response_identity,
                        )
                    body = _read_bounded_response(response)
                response_payload = _load_strict_json(body)
                validate_pinned_evidence_payload(response_payload)
                parsed = EvidenceRetrievalResponse.model_validate(response_payload)
                if parsed.request_id != request.request_id:
                    raise RetrievalError("Evidence Service response request_id does not match")
                if (
                    readiness is not None
                    and parsed.knowledge_snapshot_id != readiness.knowledge_snapshot_id
                ):
                    raise RetrievalError(
                        "Evidence Service response snapshot does not match readiness"
                    )
                return parsed
            except _ResponseIdentityValidationError as exc:
                reason = _safe_failure_reason(exc)
                raise RetrievalError(f"Evidence Service request failed: {reason}") from None
            except HTTPError as exc:
                if exc.code == 412:
                    raise RetrievalError(
                        "Evidence Service request failed: HTTP status 412"
                    ) from None
                last_error = exc
            except (
                URLError,
                TimeoutError,
                ValidationError,
                EvidenceContractPayloadError,
                json.JSONDecodeError,
                ValueError,
            ) as exc:
                last_error = exc
        if last_error is None:
            raise RetrievalError("Evidence Service request failed: internal state") from None
        reason = _safe_failure_reason(last_error)
        raise RetrievalError(f"Evidence Service request failed: {reason}") from None
