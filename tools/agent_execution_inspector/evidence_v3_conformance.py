"""Explicit loopback-only Evidence-v3 consumer for non-release conformance UAT.

This module is under ``tools`` and is not part of the Agent wheel.  It accepts
an identity object only from the caller, never an environment variable, and
temporarily overlays the production loader under one process-wide lock.  The
production adapter therefore remains fail-closed on its packaged unreleased
lock while tests can exercise real HTTP GET/POST interoperability.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from datetime import date
from typing import Iterator, Literal
from urllib.parse import urlsplit

import customs_ai_agent.adapters.retrieval_v3 as retrieval_v3_module
from customs_ai_agent.adapters.retrieval_v3 import ProductionHttpRegulationRetrieverV3
from customs_ai_agent.contracts.evidence_v3 import EvidenceRuntimeReleaseIdentityV3
from customs_ai_agent.schemas.models import EvidenceRetrievalRequest, EvidenceRetrievalResponse


_IDENTITY_OVERLAY_LOCK = threading.RLock()


def _assert_loopback(base_url: str) -> None:
    hostname = urlsplit(base_url).hostname
    if hostname in {"127.0.0.1", "::1"}:
        return
    raise ValueError("non-release Evidence v3 conformance is loopback-only")


@contextmanager
def non_release_v3_identity_overlay(
    identity: EvidenceRuntimeReleaseIdentityV3,
) -> Iterator[None]:
    """Overlay one caller-owned synthetic identity for one single-flight call."""

    with _IDENTITY_OVERLAY_LOCK:
        original = retrieval_v3_module.load_released_evidence_runtime_identity_v3
        retrieval_v3_module.load_released_evidence_runtime_identity_v3 = lambda: identity
        try:
            yield
        finally:
            retrieval_v3_module.load_released_evidence_runtime_identity_v3 = original


class LoopbackEvidenceV3ConformanceRetriever(ProductionHttpRegulationRetrieverV3):
    """Real HTTP consumer with explicit, non-release identity injection."""

    def __init__(
        self,
        base_url: str,
        *,
        conformance_identity: EvidenceRuntimeReleaseIdentityV3,
        service_token: str,
        acknowledge_non_release: Literal[True],
        timeout_seconds: float = 30.0,
        max_retries: int = 0,
    ) -> None:
        if acknowledge_non_release is not True:
            raise ValueError("explicit non-release conformance acknowledgement is required")
        _assert_loopback(base_url)
        self._conformance_identity = conformance_identity
        self._successful_readiness_get_count = 0
        self._successful_retrieval_post_count = 0
        with non_release_v3_identity_overlay(conformance_identity):
            super().__init__(
                base_url,
                service_token=service_token,
                allowed_origins={base_url.rstrip("/")},
                timeout_seconds=timeout_seconds,
                max_retries=max_retries,
            )

    def preflight(
        self,
        review_as_of: date | None = None,
        *,
        _deadline: float | None = None,
    ) -> EvidenceRuntimeReleaseIdentityV3:
        with non_release_v3_identity_overlay(self._conformance_identity):
            identity = super().preflight(review_as_of, _deadline=_deadline)
        self._successful_readiness_get_count += 1
        return identity

    def retrieve(self, request: EvidenceRetrievalRequest) -> EvidenceRetrievalResponse:
        with non_release_v3_identity_overlay(self._conformance_identity):
            response = super().retrieve(request)
        self._successful_retrieval_post_count += 1
        return response

    @property
    def successful_readiness_get_count(self) -> int:
        return self._successful_readiness_get_count

    @property
    def successful_retrieval_post_count(self) -> int:
        return self._successful_retrieval_post_count


__all__ = [
    "LoopbackEvidenceV3ConformanceRetriever",
    "non_release_v3_identity_overlay",
]
