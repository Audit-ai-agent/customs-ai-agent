from __future__ import annotations

import hashlib
import ipaddress
import json
import re
from datetime import date
from typing import Any, Protocol, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit, urlunsplit
from urllib.request import Request, urlopen

from pydantic import ValidationError

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


_HEADER_NAME = re.compile(r"^[!#$%&'*+.^_`|~0-9A-Za-z-]+$")
_ISO_CALENDAR_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_RESERVED_HEADERS = {
    "accept",
    "connection",
    "content-length",
    "content-type",
    "host",
    "transfer-encoding",
    "x-request-id",
}


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
    if isinstance(exc, HTTPError):
        return f"HTTP status {exc.code}"
    if isinstance(exc, TimeoutError):
        return "timeout"
    if isinstance(exc, URLError):
        return "network error"
    return "invalid contract response"


def _validate_transport_settings(timeout_seconds: float, max_retries: int) -> None:
    if timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be greater than zero")
    if max_retries < 0:
        raise ValueError("max_retries cannot be negative")


def fetch_supported_review_as_of(
    base_url: str,
    *,
    service_token: str | None = None,
    token_header: str = "X-Demo-Token",
    timeout_seconds: float = 10.0,
    max_retries: int = 1,
) -> date:
    """Read the provider-advertised snapshot date through its HTTP readiness contract."""

    normalized_base_url = validate_evidence_base_url(base_url)
    _validate_transport_settings(timeout_seconds, max_retries)
    headers = {"Accept": "application/json"}
    headers.update(_optional_auth_header(service_token, token_header))
    readiness_url = f"{normalized_base_url}/internal/rag/v1/health/ready"
    last_error: Exception | None = None
    for _ in range(max_retries + 1):
        try:
            request = Request(readiness_url, headers=headers, method="GET")
            with urlopen(request, timeout=timeout_seconds) as response:
                payload = json.loads(response.read())
            if not isinstance(payload, dict) or payload.get("status") != "ready":
                raise ValueError("readiness response did not report ready")
            raw_date = payload.get("supported_review_as_of")
            if not isinstance(raw_date, str) or not _ISO_CALENDAR_DATE.fullmatch(raw_date):
                raise ValueError("readiness response omitted supported_review_as_of")
            return date.fromisoformat(raw_date)
        except (HTTPError, URLError, TimeoutError, json.JSONDecodeError, ValueError) as exc:
            last_error = exc
    assert last_error is not None
    reason = _safe_failure_reason(last_error)
    raise RetrievalError(
        f"Evidence Service readiness request failed: {reason}"
    ) from None


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
                version_hash = hashlib.sha256(
                    entry["version_hash"].encode("utf-8")
                ).hexdigest()
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
                        eligibility=(
                            "finding" if required_code and score >= 0.75 else "supporting"
                        ),
                        score=score,
                        raw_scores={"mock": score},
                    )
                )

            requirements_found = any(
                evidence_item.source_type == "customs_requirement"
                and evidence_item.eligibility == "finding"
                for evidence_item in evidence
            )
            complete = resolution in {"exact", "unique_descendant"} and bool(matches) and all(
                evidence_item.eligibility != "blocked" and (evidence_item.score or 0) >= 0.75
                for evidence_item in evidence
            )
            coverage: Coverage = (
                "complete" if complete else ("partial" if matches else "unknown")
            )
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
        token_header: str = "X-Demo-Token",
        timeout_seconds: float = 60.0,
        max_retries: int = 1,
    ) -> None:
        normalized_base_url = validate_evidence_base_url(base_url)
        _validate_transport_settings(timeout_seconds, max_retries)
        self.url = f"{normalized_base_url}{self.endpoint_path}"
        self._auth_header = _optional_auth_header(service_token, token_header)
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def retrieve(self, request: EvidenceRetrievalRequest) -> EvidenceRetrievalResponse:
        payload = json.dumps(request.model_dump(mode="json"), ensure_ascii=False).encode("utf-8")
        headers = {
            "Accept": "application/json",
            "Content-Type": "application/json; charset=utf-8",
            "X-Request-ID": request.request_id,
        }
        headers.update(self._auth_header)
        last_error: Exception | None = None
        for _ in range(self.max_retries + 1):
            try:
                http_request = Request(
                    self.url,
                    data=payload,
                    headers=headers,
                    method="POST",
                )
                with urlopen(http_request, timeout=self.timeout_seconds) as response:
                    body = response.read()
                parsed = EvidenceRetrievalResponse.model_validate_json(body)
                if parsed.request_id != request.request_id:
                    raise RetrievalError("Evidence Service response request_id does not match")
                return parsed
            except (HTTPError, URLError, TimeoutError, ValidationError, ValueError) as exc:
                last_error = exc
        assert last_error is not None
        reason = _safe_failure_reason(last_error)
        raise RetrievalError(f"Evidence Service request failed: {reason}") from None
