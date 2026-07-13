from __future__ import annotations

import hashlib
import json
from typing import Any, Protocol, runtime_checkable
from urllib.error import HTTPError, URLError
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
        timeout_seconds: float = 60.0,
        max_retries: int = 1,
    ) -> None:
        if not base_url.strip():
            raise ValueError("base_url is required")
        self.url = f"{base_url.rstrip('/')}{self.endpoint_path}"
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries

    def retrieve(self, request: EvidenceRetrievalRequest) -> EvidenceRetrievalResponse:
        payload = json.dumps(request.model_dump(mode="json"), ensure_ascii=False).encode("utf-8")
        http_request = Request(
            self.url,
            data=payload,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json; charset=utf-8",
                "X-Request-ID": request.request_id,
            },
            method="POST",
        )
        last_error: Exception | None = None
        for _ in range(self.max_retries + 1):
            try:
                with urlopen(http_request, timeout=self.timeout_seconds) as response:
                    body = response.read()
                parsed = EvidenceRetrievalResponse.model_validate_json(body)
                if parsed.request_id != request.request_id:
                    raise RetrievalError("Evidence Service response request_id does not match")
                return parsed
            except (HTTPError, URLError, TimeoutError, ValidationError, ValueError) as exc:
                last_error = exc
        raise RetrievalError(f"Evidence Service request failed: {last_error}") from last_error
