"""Explicit loopback-only Evidence-v3 consumer for non-release conformance UAT.

This module is under ``tools`` and is not part of the Agent wheel.  It accepts
an identity object only from the caller, never an environment variable.  The
diagnostic consumer owns that identity directly and never patches a packaged
release loader.  The production adapter therefore remains fail-closed on its
packaged unreleased lock while local tools can exercise real HTTP GET/POST
interoperability safely.
"""

from __future__ import annotations

import time
from datetime import date
from typing import Literal
from urllib.error import HTTPError, URLError
from urllib.request import Request
from urllib.parse import urlsplit

from customs_ai_agent.adapters.retrieval import (
    RetrievalError,
    _EvidenceDeadlineExceeded,
    _load_strict_json,
    _new_deadline,
    _optional_auth_header,
    _perform_http_exchange,
    _safe_failure_reason,
    _validate_transport_settings,
    validate_evidence_destination,
)
from customs_ai_agent.adapters.retrieval_v3 import ProductionHttpRegulationRetrieverV3
from customs_ai_agent.contracts.evidence_v3 import (
    EvidenceRuntimeReleaseIdentityV3,
    EvidenceV3ContractError,
    parse_evidence_runtime_readiness_v3,
)
from customs_ai_agent.schemas.models import EvidenceRetrievalRequest, EvidenceRetrievalResponse


def _assert_loopback(base_url: str) -> None:
    hostname = urlsplit(base_url).hostname
    if hostname in {"127.0.0.1", "::1"}:
        return
    raise ValueError("non-release Evidence v3 conformance is loopback-only")


class LoopbackEvidenceV3DiagnosticRetriever(ProductionHttpRegulationRetrieverV3):
    """Real HTTP consumer with explicit, non-release identity injection.

    The caller owns the exact V3 runtime identity and the meaning of the
    diagnostic profile.  This adapter only proves that the loopback provider's
    readiness and retrieval response are bound to that identity.  It never
    infers which databases the provider used, never loads a Production lock,
    and has no Mock fallback.
    """

    def __init__(
        self,
        base_url: str,
        *,
        conformance_identity: EvidenceRuntimeReleaseIdentityV3,
        service_token: str,
        acknowledge_non_release: Literal[True],
        timeout_seconds: float = 30.0,
        max_retries: int = 0,
        readiness_identity_cache_ttl_seconds: float = 0.0,
    ) -> None:
        if acknowledge_non_release is not True:
            raise ValueError("explicit non-release conformance acknowledgement is required")
        _assert_loopback(base_url)
        normalized = validate_evidence_destination(base_url, {base_url.rstrip("/")})
        _validate_transport_settings(timeout_seconds, max_retries)
        if service_token is None:
            raise ValueError("Diagnostic Evidence bearer token is required")
        if not 0.0 <= readiness_identity_cache_ttl_seconds <= 600.0:
            raise ValueError("Diagnostic Evidence readiness cache TTL is invalid")
        self._conformance_identity = conformance_identity
        self._successful_readiness_get_count = 0
        self._successful_retrieval_post_count = 0
        self._readiness_identity_cache_ttl_seconds = readiness_identity_cache_ttl_seconds
        self._readiness_identity_cache: (
            tuple[float, EvidenceRuntimeReleaseIdentityV3] | None
        ) = None
        # Initialize the parent-owned transport fields explicitly. Calling the
        # Production constructor here would load the packaged *released*
        # identity and would either fail or tempt a process-global monkeypatch.
        self._base_url = normalized
        self._auth_header = _optional_auth_header(service_token, "Authorization")
        self._expected_identity = conformance_identity
        self._service_token = service_token
        self._allowed_origins = frozenset({normalized})
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.url = f"{normalized}{self.endpoint_path}"
        # The parent uses thread-local request policy bindings. Importing
        # threading locally keeps the diagnostic-only boundary obvious.
        import threading

        self._local = threading.local()
        self._readiness_identity_cache_lock = threading.Lock()

    def preflight(
        self,
        review_as_of: date | None = None,
        *,
        _deadline: float | None = None,
    ) -> EvidenceRuntimeReleaseIdentityV3:
        with self._readiness_identity_cache_lock:
            cached = self._readiness_identity_cache
        if review_as_of is not None and cached is not None and time.monotonic() < cached[0]:
            identity = cached[1]
            if (
                review_as_of is None
                or review_as_of == identity.base_runtime_release.supported_review_as_of
            ):
                return identity

        deadline = _new_deadline(self.timeout_seconds) if _deadline is None else _deadline
        headers = {"Accept": "application/json", **self._auth_header}
        url = f"{self._base_url}/internal/rag/v3/health/ready"
        last_error: Exception | None = None
        for _ in range(self.max_retries + 1):
            try:
                request = Request(url, headers=headers, method="GET")  # noqa: S310
                payload = _load_strict_json(_perform_http_exchange(request, deadline=deadline))
                identity = parse_evidence_runtime_readiness_v3(payload)
                if identity != self._conformance_identity:
                    raise ValueError("diagnostic runtime v3 release identity mismatch")
                if (
                    review_as_of is not None
                    and review_as_of != identity.base_runtime_release.supported_review_as_of
                ):
                    raise ValueError("diagnostic runtime v3 does not support review_as_of")
                self._successful_readiness_get_count += 1
                if (
                    review_as_of is not None
                    and self._readiness_identity_cache_ttl_seconds > 0.0
                ):
                    with self._readiness_identity_cache_lock:
                        self._readiness_identity_cache = (
                            time.monotonic() + self._readiness_identity_cache_ttl_seconds,
                            identity,
                        )
                return identity
            except (
                HTTPError,
                URLError,
                TimeoutError,
                ValueError,
                EvidenceV3ContractError,
            ) as exc:
                last_error = exc
                if isinstance(exc, _EvidenceDeadlineExceeded) or time.monotonic() >= deadline:
                    last_error = _EvidenceDeadlineExceeded()
                    break
        reason = _safe_failure_reason(last_error or RuntimeError("internal state"))
        raise RetrievalError(f"Evidence v3 readiness request failed: {reason}") from None

    def invalidate_readiness_identity_cache(self) -> None:
        """Start each review with a fresh dated readiness attestation."""

        with self._readiness_identity_cache_lock:
            self._readiness_identity_cache = None

    def retrieve(self, request: EvidenceRetrievalRequest) -> EvidenceRetrievalResponse:
        response = super().retrieve(request)
        self._successful_retrieval_post_count += 1
        return response

    @property
    def successful_readiness_get_count(self) -> int:
        return self._successful_readiness_get_count

    @property
    def successful_retrieval_post_count(self) -> int:
        return self._successful_retrieval_post_count


class LoopbackEvidenceV3ConformanceRetriever(LoopbackEvidenceV3DiagnosticRetriever):
    """Backward-compatible name for the synthetic V3 conformance runner."""


__all__ = [
    "LoopbackEvidenceV3ConformanceRetriever",
    "LoopbackEvidenceV3DiagnosticRetriever",
]
