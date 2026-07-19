from __future__ import annotations

import json
from copy import deepcopy
from datetime import date
from urllib.error import HTTPError, URLError
from urllib.request import Request

from pydantic import ValidationError
import pytest

from customs_ai_agent.adapters.retrieval import (
    MAX_EVIDENCE_RESPONSE_BYTES,
    EmptyRegulationRetriever,
    EVIDENCE_PROVIDER_BUILD_SHA256_HEADER,
    EVIDENCE_PROVIDER_RELEASE_HEADER,
    EVIDENCE_PROVIDER_REVISION_HEADER,
    EVIDENCE_SCHEMA_SHA256_HEADER,
    FailingRegulationRetriever,
    HttpRegulationRetriever,
    PartialRegulationRetriever,
    RetrievalError,
    fetch_evidence_readiness,
    fetch_supported_review_as_of,
    validate_evidence_base_url,
)
from customs_ai_agent.contracts.evidence import load_pinned_evidence_contract_identity
from customs_ai_agent.contracts.evidence import EvidenceContractLockError
from customs_ai_agent.adapters.retrieval import _NoRedirectHandler
from customs_ai_agent.runner import (
    rerun_review,
    run_review,
    run_review_result,
    to_review_run_result_v2,
)
from customs_ai_agent.nodes.compliance import compliance_review
from customs_ai_agent.nodes.retrieve import _context_errors, _hs_result_matches_query
from customs_ai_agent.schemas.models import (
    EvidenceRetrievalRequest,
    EvidenceRetrievalResponse,
    RagEvidence,
    RetrievalItemResult,
    RetrievalQueryItem,
    ReviewRunRequest,
    ReviewRunResultV2,
    stable_evidence_id,
)
from customs_ai_agent.fixtures.scenarios import SCENARIOS


def _http_test_request() -> EvidenceRetrievalRequest:
    return EvidenceRetrievalRequest(
        request_id="request-http-1",
        case_id="case-http-1",
        review_run_id="run-http-1",
        review_as_of="2026-07-10",
        jurisdiction="KR",
        trade_type="import",
        items=[
            {
                "item_id": "item-http-1",
                "description": "Machine screw",
                "hs_code": "7318151000",
                "origin_country": "NO",
                "destination_country": "KR",
            }
        ],
    )


def _http_test_response(request_id: str) -> EvidenceRetrievalResponse:
    return EvidenceRetrievalResponse(
        request_id=request_id,
        status="ok",
        knowledge_snapshot_id="snapshot-http-v1",
        results=[
            RetrievalItemResult(
                item_id="item-http-1",
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


def _readiness_payload(
    *,
    include_identity: bool = True,
    snapshot_id: str = "snapshot-1",
) -> dict[str, object]:
    payload: dict[str, object] = {
        "status": "ready",
        "knowledge_snapshot_id": snapshot_id,
        "supported_review_as_of": "2026-07-10",
    }
    if include_identity:
        identity = load_pinned_evidence_contract_identity()
        payload["evidence_contract"] = {
            "contract": identity.contract,
            "schema_version": identity.schema_version,
            "provider_release": identity.provider_release,
            "provider_revision": identity.provider_revision,
            "provider_build_sha256": identity.provider_build_sha256,
            "schema_sha256": identity.schema_sha256,
        }
    return payload


def _identity_headers() -> dict[str, str]:
    identity = load_pinned_evidence_contract_identity()
    return {
        EVIDENCE_SCHEMA_SHA256_HEADER: identity.schema_sha256,
        EVIDENCE_PROVIDER_RELEASE_HEADER: identity.provider_release,
        EVIDENCE_PROVIDER_REVISION_HEADER: identity.provider_revision,
        EVIDENCE_PROVIDER_BUILD_SHA256_HEADER: identity.provider_build_sha256,
    }


class _HttpBody:
    def __init__(
        self,
        payload: bytes,
        *,
        headers: dict[str, str] | None = None,
    ) -> None:
        self.payload = payload
        self.headers = _identity_headers() if headers is None else headers
        self.read_count = 0

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self, size: int = -1) -> bytes:
        self.read_count += 1
        return self.payload if size < 0 else self.payload[:size]

    def getheader(self, name: str) -> str | None:
        return self.headers.get(name)


class RequirementRetriever:
    def __init__(self) -> None:
        self.request: EvidenceRetrievalRequest | None = None

    def retrieve(self, request: EvidenceRetrievalRequest) -> EvidenceRetrievalResponse:
        self.request = request
        results = []
        for item in request.items:
            source_record_id = "requirement:731815:import:1"
            version_hash = "b" * 64
            evidence = RagEvidence(
                evidence_id=stable_evidence_id(source_record_id, version_hash),
                source_record_id=source_record_id,
                source_type="customs_requirement",
                item_id=item.item_id,
                title="Test import requirement",
                text="A test import confirmation is required.",
                law_id="law-test",
                law_name="Test Import Act",
                article_no="12",
                source_uri="https://example.test/requirements/1",
                valid_from="2026-01-01",
                version_hash=version_hash,
                matched_hs_code="7318151000",
                trade_type=request.trade_type,
                required_document_code="test_import_confirmation",
                required_document_name="Test import confirmation",
                agencies=["Test Customs Agency"],
                retrieval_methods=["exact_hs", "graph"],
                eligibility="finding",
                score=0.93,
                raw_scores={"exact_hs": 1.0},
            )
            results.append(
                RetrievalItemResult(
                    item_id=item.item_id,
                    hs_resolution="unique_descendant",
                    resolved_hsk10="7318151000",
                    candidate_hsk10s=["7318151000"],
                    coverage="complete",
                    requirements_found=True,
                    evidence=[evidence],
                    errors=[],
                )
            )
        return EvidenceRetrievalResponse(
            request_id=request.request_id,
            status="ok",
            knowledge_snapshot_id="snapshot-test-v1",
            results=results,
            errors=[],
        )


class OutOfDateRequirementRetriever(RequirementRetriever):
    def __init__(self, validity: str) -> None:
        super().__init__()
        self.validity = validity

    def retrieve(self, request: EvidenceRetrievalRequest) -> EvidenceRetrievalResponse:
        payload = super().retrieve(request).model_dump(mode="python")
        evidence = payload["results"][0]["evidence"][0]
        if self.validity == "not_yet_valid":
            evidence["valid_from"] = date(2026, 7, 11)
        else:
            evidence["valid_to"] = date(2026, 7, 9)
        return EvidenceRetrievalResponse.model_validate(payload)


class NeverCalledRetriever:
    def __init__(self) -> None:
        self.called = False

    def retrieve(self, _request):
        self.called = True
        raise AssertionError("Evidence retrieval must not receive an invalid HS code")


def _documents_with_replacements(replacements: dict[str, str]):
    documents = deepcopy(SCENARIOS["happy_path"]["documents"])
    return [
        document.model_copy(
            update={
                "content_text": replacements.get(
                    document.document_id,
                    document.content_text,
                )
            }
        )
        for document in documents
    ]


def test_contaminated_hs_code_is_a_validation_issue_and_never_reaches_evidence():
    invoice = next(
        document.content_text
        for document in SCENARIOS["happy_path"]["documents"]
        if document.document_id == "doc_inv"
    )
    retriever = NeverCalledRetriever()
    state = run_review(
        ReviewRunRequest(
            case_id="case-invalid-hs-contamination",
            review_run_id="run-invalid-hs-contamination",
            documents=_documents_with_replacements(
                {"doc_inv": invoice.replace("HS: 7318.15", "HS: ABC731815")}
            ),
        ),
        retriever=retriever,
    )

    assert retriever.called is False
    assert state["normalized_documents"]["doc_inv"]["line_items"][0]["hs_code"] is None
    invalid_issues = [
        issue
        for issue in state["validation_issues"]
        if issue["issue_code"] == "HS_CODE_FORMAT_INVALID"
        and issue["source_document_id"] == "doc_inv"
    ]
    assert len(invalid_issues) == 1
    assert invalid_issues[0]["observed_value"] == "ABC731815"
    assert state["retrieval_status"] == "failed"
    assert state["retrieval_request"] == {}
    assert state["retrieval_errors"] == [
        {
            "code": "INVALID_HS_CODE_RETRIEVAL_BLOCKED",
            "message": (
                "Regulatory retrieval was not called because invoice items have "
                f"invalid HS codes: {state['normalized_documents']['doc_inv']['line_items'][0]['item_id']}."
            ),
            "retryable": False,
        }
    ]
    assert state["requires_human_review"] is True
    assert state["final_status"] == "needs_review"


def test_ambiguous_numeric_dates_are_not_assumed_to_be_month_first():
    invoice = next(
        document.content_text
        for document in SCENARIOS["happy_path"]["documents"]
        if document.document_id == "doc_inv"
    )
    packing = next(
        document.content_text
        for document in SCENARIOS["happy_path"]["documents"]
        if document.document_id == "doc_pl"
    )
    state = run_review(
        ReviewRunRequest(
            case_id="case-ambiguous-numeric-dates",
            review_run_id="run-ambiguous-numeric-dates",
            documents=_documents_with_replacements(
                {
                    "doc_inv": invoice.replace("2026.06.01", "05/06/2025"),
                    "doc_pl": packing.replace("2026.06.01", "05/06/2025"),
                }
            ),
        )
    )

    assert state["normalized_documents"]["doc_inv"]["fields"]["invoice_date"] is None
    assert state["normalized_documents"]["doc_pl"]["fields"]["invoice_date"] is None
    ambiguous_issues = [
        issue
        for issue in state["validation_issues"]
        if issue["issue_code"] == "DATE_FORMAT_AMBIGUOUS"
    ]
    assert {issue["source_document_id"] for issue in ambiguous_issues} == {
        "doc_inv",
        "doc_pl",
    }
    assert {issue["observed_value"] for issue in ambiguous_issues} == {"05/06/2025"}
    assert state["requires_human_review"] is True
    assert state["final_status"] == "needs_review"


def _non_requirement_finding_payload(source_type: str, item_id: str) -> dict[str, object]:
    source_record_id = f"malicious-{source_type}-finding"
    version_hash = "c" * 64
    return {
        "evidence_id": stable_evidence_id(source_record_id, version_hash),
        "source_record_id": source_record_id,
        "source_type": source_type,
        "item_id": item_id,
        "title": "Unstructured semantic hit",
        "text": "This is not a structured customs requirement record.",
        "law_id": "untrusted-law-label",
        "law_name": "Untrusted legal label",
        "article_no": "untrusted-article",
        "source_uri": f"https://example.test/{source_type}/malicious",
        "valid_from": date(2026, 1, 1),
        "version_hash": version_hash,
        "matched_hs_code": "7318151000",
        "trade_type": "import",
        "required_document_code": "UNTRUSTED-CERT",
        "required_document_name": "Untrusted certificate",
        "agencies": ["Untrusted Agency"],
        "retrieval_methods": ["dense"],
        "eligibility": "finding",
        "score": 0.99,
        "raw_scores": {"dense": 0.99},
    }


class ConstructedNonRequirementFindingRetriever:
    """Simulate a provider bypassing Pydantic validators with model_construct."""

    def __init__(self, source_type: str) -> None:
        self.source_type = source_type

    def retrieve(self, request: EvidenceRetrievalRequest) -> EvidenceRetrievalResponse:
        results = []
        for item in request.items:
            evidence = RagEvidence.model_construct(
                **_non_requirement_finding_payload(self.source_type, item.item_id)
            )
            results.append(
                RetrievalItemResult.model_construct(
                    item_id=item.item_id,
                    hs_resolution="unique_descendant",
                    resolved_hsk10="7318151000",
                    candidate_hsk10s=["7318151000"],
                    coverage="complete",
                    requirements_found=False,
                    evidence=[evidence],
                    errors=[],
                )
            )
        return EvidenceRetrievalResponse.model_construct(
            schema_version="1.0",
            request_id=request.request_id,
            status="ok",
            knowledge_snapshot_id="malicious-snapshot",
            results=results,
            errors=[],
        )


class ConstructedRequirementsFlagMismatchRetriever(RequirementRetriever):
    """Simulate a validated requirement whose result-level flag was tampered with."""

    def retrieve(self, request: EvidenceRetrievalRequest) -> EvidenceRetrievalResponse:
        valid_response = super().retrieve(request)
        tampered_results = [
            RetrievalItemResult.model_construct(
                item_id=result.item_id,
                hs_resolution=result.hs_resolution,
                resolved_hsk10=result.resolved_hsk10,
                candidate_hsk10s=result.candidate_hsk10s,
                coverage=result.coverage,
                requirements_found=False,
                evidence=result.evidence,
                errors=result.errors,
            )
            for result in valid_response.results
        ]
        return EvidenceRetrievalResponse.model_construct(
            schema_version="1.0",
            request_id=request.request_id,
            status="ok",
            knowledge_snapshot_id="tampered-snapshot",
            results=tampered_results,
            errors=[],
        )


class AmbiguousCompleteRetriever:
    def retrieve(self, request: EvidenceRetrievalRequest) -> EvidenceRetrievalResponse:
        return EvidenceRetrievalResponse(
            request_id=request.request_id,
            status="ok",
            knowledge_snapshot_id="snapshot-test-v1",
            results=[
                RetrievalItemResult(
                    item_id=item.item_id,
                    hs_resolution="ambiguous",
                    resolved_hsk10=None,
                    candidate_hsk10s=["7318151000", "7318159000"],
                    coverage="complete",
                    requirements_found=False,
                    evidence=[],
                    errors=[],
                )
                for item in request.items
            ],
            errors=[],
        )


class CompleteNegativeHsRetriever:
    def __init__(
        self,
        *,
        hs_resolution: object,
        resolved_hsk10: object,
        candidate_hsk10s: object,
        response_status: object = "ok",
        constructed: bool,
    ) -> None:
        self.hs_resolution = hs_resolution
        self.resolved_hsk10 = resolved_hsk10
        self.candidate_hsk10s = candidate_hsk10s
        self.response_status = response_status
        self.constructed = constructed

    def retrieve(self, request: EvidenceRetrievalRequest) -> EvidenceRetrievalResponse:
        result_values = {
            "item_id": request.items[0].item_id,
            "hs_resolution": self.hs_resolution,
            "resolved_hsk10": self.resolved_hsk10,
            "candidate_hsk10s": self.candidate_hsk10s,
            "coverage": "complete",
            "requirements_found": False,
            "evidence": [],
            "errors": [],
        }
        if not self.constructed:
            result = RetrievalItemResult.model_validate(result_values)
            return EvidenceRetrievalResponse(
                request_id=request.request_id,
                status="ok",
                knowledge_snapshot_id="wrong-hs-negative-snapshot",
                results=[result],
                errors=[],
            )
        result = RetrievalItemResult.model_construct(**result_values)
        return EvidenceRetrievalResponse.model_construct(
            schema_version="1.0",
            request_id=request.request_id,
            status=self.response_status,
            knowledge_snapshot_id="constructed-wrong-hs-negative-snapshot",
            results=[result],
            errors=[],
        )


@pytest.mark.parametrize(
    ("retriever", "expected_status", "expected_coverage"),
    [
        (EmptyRegulationRetriever(), "partial", "unknown"),
        (PartialRegulationRetriever(), "partial", "partial"),
        (FailingRegulationRetriever(), "failed", "unknown"),
    ],
)
def test_unverified_or_failed_retrieval_is_fail_closed(
    retriever, expected_status, expected_coverage
):
    state = run_review(
        ReviewRunRequest(scenario_id="happy_path", review_run_id="run-fail-closed"),
        retriever=retriever,
    )

    assert state["retrieval_status"] == expected_status
    assert state["retrieval_coverage"] == expected_coverage
    assert state["final_status"] == "needs_review"
    assert state["requires_human_review"] is True
    assert state["report_json"]["overall_status"] == state["final_status"]


@pytest.mark.parametrize("source_type", ["notice", "hsk_entry"])
def test_non_requirement_evidence_cannot_claim_finding_eligibility(source_type):
    with pytest.raises(
        ValidationError,
        match="finding eligibility is reserved for structured customs requirements",
    ):
        RagEvidence.model_validate(_non_requirement_finding_payload(source_type, "item-1"))


@pytest.mark.parametrize(
    ("field_name", "invalid_value"),
    [
        ("title", " "),
        ("text", " "),
        ("law_id", ""),
        ("law_name", " "),
        ("valid_from", None),
        ("matched_hs_code", ""),
        ("required_document_code", ""),
        ("required_document_name", " "),
        ("agencies", [""]),
        ("retrieval_methods", []),
    ],
)
def test_requirement_finding_rejects_blank_or_missing_provenance(field_name, invalid_value):
    response = RequirementRetriever().retrieve(_http_test_request())
    payload = response.results[0].evidence[0].model_dump(mode="python")
    payload[field_name] = invalid_value

    with pytest.raises(
        ValidationError,
        match="finding customs requirement requires source, law, date, matched HS",
    ):
        RagEvidence.model_validate(payload)


@pytest.mark.parametrize("source_uri", [" ", "not-a-uri", "https://"])
def test_evidence_rejects_non_absolute_source_uri(source_uri):
    response = RequirementRetriever().retrieve(_http_test_request())
    payload = response.results[0].evidence[0].model_dump(mode="python")
    payload["source_uri"] = source_uri

    with pytest.raises(ValidationError, match="source_uri must be an absolute URI"):
        RagEvidence.model_validate(payload)


def test_requirement_finding_rejects_blank_source_record_id():
    response = RequirementRetriever().retrieve(_http_test_request())
    payload = response.results[0].evidence[0].model_dump(mode="python")
    payload["source_record_id"] = " "
    payload["evidence_id"] = stable_evidence_id(" ", payload["version_hash"])

    with pytest.raises(
        ValidationError,
        match="finding customs requirement requires source, law, date, matched HS",
    ):
        RagEvidence.model_validate(payload)


@pytest.mark.parametrize("source_type", ["notice", "hsk_entry"])
def test_constructed_non_requirement_finding_response_is_fail_closed(source_type):
    state = run_review(
        ReviewRunRequest(
            scenario_id="happy_path",
            review_run_id=f"run-malicious-{source_type}",
            review_as_of="2026-07-10",
            trade_type="import",
            jurisdiction="KR",
        ),
        retriever=ConstructedNonRequirementFindingRetriever(source_type),
    )

    assert state["retrieval_status"] == "failed"
    assert state["retrieval_coverage"] == "unknown"
    assert state["rag_evidence"] == []
    assert state["regulatory_findings"] == []
    assert state["requires_human_review"] is True
    assert state["final_status"] == "needs_review"
    assert "EVIDENCE_FINDING_SOURCE_NOT_STRUCTURED" in {
        error["code"] for error in state["retrieval_errors"]
    }


def test_constructed_requirements_found_mismatch_is_fail_closed():
    state = run_review(
        ReviewRunRequest(
            scenario_id="happy_path",
            review_run_id="run-tampered-requirements-flag",
            review_as_of=date(2026, 7, 10),
            trade_type="import",
            jurisdiction="KR",
        ),
        retriever=ConstructedRequirementsFlagMismatchRetriever(),
    )

    assert state["retrieval_status"] == "failed"
    assert state["retrieval_coverage"] == "unknown"
    assert state["rag_evidence"] == []
    assert state["regulatory_findings"] == []
    assert state["requires_human_review"] is True
    assert state["final_status"] == "needs_review"
    assert "EVIDENCE_REQUIREMENTS_FOUND_MISMATCH" in {
        error["code"] for error in state["retrieval_errors"]
    }


@pytest.mark.parametrize("source_type", ["notice", "hsk_entry"])
def test_compliance_defense_rejects_unstructured_high_finding(source_type):
    update = compliance_review(
        {
            "retrieval_status": "ok",
            "retrieval_coverage": "complete",
            "knowledge_snapshot_id": "malicious-snapshot",
            "rag_evidence": [_non_requirement_finding_payload(source_type, "item-1")],
            "normalized_documents": {},
            "uploaded_requirement_document_codes": [],
            "documents": [],
            "requires_human_review": False,
            "audit_trail": [],
        }
    )

    assert update["regulatory_findings"] == []
    assert update["requires_human_review"] is True
    assert update["audit_trail"][-1]["metadata"]["rejected_or_low_evidence"] == 1


def test_ambiguous_hs_cannot_complete_even_with_complete_coverage():
    state = run_review(
        ReviewRunRequest(scenario_id="happy_path", review_run_id="run-ambiguous"),
        retriever=AmbiguousCompleteRetriever(),
    )

    assert state["retrieval_status"] == "ok"
    assert state["retrieval_coverage"] == "complete"
    assert state["retrieval_results"][0]["hs_resolution"] == "ambiguous"
    assert state["final_status"] == "needs_review"


@pytest.mark.parametrize(
    ("validity", "expected_error"),
    [
        ("not_yet_valid", "EVIDENCE_NOT_YET_VALID"),
        ("expired", "EVIDENCE_EXPIRED"),
    ],
)
def test_out_of_date_evidence_cannot_complete(validity, expected_error):
    state = run_review(
        ReviewRunRequest(
            scenario_id="happy_path",
            review_run_id=f"run-evidence-{validity}",
            review_as_of="2026-07-10",
        ),
        retriever=OutOfDateRequirementRetriever(validity),
    )

    assert state["retrieval_status"] == "failed"
    assert state["retrieval_coverage"] == "unknown"
    assert state["rag_evidence"] == []
    assert state["regulatory_findings"] == []
    assert expected_error in {error["code"] for error in state["retrieval_errors"]}
    assert state["requires_human_review"] is True
    assert state["final_status"] == "needs_review"


@pytest.mark.parametrize(
    ("requested_hs", "resolution", "resolved_hsk10", "expected"),
    [
        ("7318151000", "exact", "7318151000", True),
        ("7318151000", "exact", "7318159000", False),
        ("731815", "unique_descendant", "7318151000", True),
        ("731815", "unique_descendant", "9999999999", False),
        ("731815", "exact", "7318151000", False),
        ("7318151000", "unique_descendant", "7318151000", False),
    ],
)
def test_hs_result_binding_matches_only_exact_or_true_descendant(
    requested_hs,
    resolution,
    resolved_hsk10,
    expected,
):
    query = RetrievalQueryItem(
        item_id="item-hs-binding",
        description="Machine screw",
        hs_code=requested_hs,
        origin_country="NO",
        destination_country="KR",
    )
    result = RetrievalItemResult(
        item_id=query.item_id,
        hs_resolution=resolution,
        resolved_hsk10=resolved_hsk10,
        candidate_hsk10s=[resolved_hsk10],
        coverage="complete",
        requirements_found=False,
        evidence=[],
        errors=[],
    )

    assert _hs_result_matches_query(query, result) is expected


def test_mismatched_complete_negative_hs_cannot_complete():
    state = run_review(
        ReviewRunRequest(
            scenario_id="happy_path",
            review_run_id="run-mismatched-complete-negative-hs",
        ),
        retriever=CompleteNegativeHsRetriever(
            hs_resolution="unique_descendant",
            resolved_hsk10="9999999999",
            candidate_hsk10s=["9999999999"],
            constructed=False,
        ),
    )

    assert state["retrieval_status"] == "failed"
    assert state["retrieval_coverage"] == "unknown"
    assert state["retrieval_results"] == []
    assert state["rag_evidence"] == []
    assert state["retrieval_errors"][0]["code"] == "HS_RESULT_MISMATCH"
    assert state["requires_human_review"] is True
    assert state["final_status"] == "needs_review"


@pytest.mark.parametrize(
    ("hs_resolution", "resolved_hsk10", "candidate_hsk10s", "response_status"),
    [
        ("exact", "", [], "ok"),
        ("exact", None, [], "ok"),
        ("exact", 7_318_151_000, [], "ok"),
        ("", "7318151000", ["7318151000"], "ok"),
        (["exact"], "7318151000", ["7318151000"], "ok"),
        ("unique_descendant", "7318151000", "7318151000", "ok"),
        ("unique_descendant", "7318151000", ["7318151000"], []),
        (
            "unique_descendant",
            "7318151000",
            ["7318151000", "7318152000"],
            "ok",
        ),
    ],
)
def test_constructed_malformed_hs_result_is_fail_closed(
    hs_resolution,
    resolved_hsk10,
    candidate_hsk10s,
    response_status,
):
    state = run_review(
        ReviewRunRequest(
            scenario_id="happy_path",
            review_run_id="run-constructed-malformed-hs",
        ),
        retriever=CompleteNegativeHsRetriever(
            hs_resolution=hs_resolution,
            resolved_hsk10=resolved_hsk10,
            candidate_hsk10s=candidate_hsk10s,
            response_status=response_status,
            constructed=True,
        ),
    )

    assert state["retrieval_status"] == "failed"
    assert state["retrieval_coverage"] == "unknown"
    assert state["retrieval_results"] == []
    assert state["rag_evidence"] == []
    assert state["retrieval_errors"][0]["code"] in {
        "HS_RESULT_MISMATCH",
        "INVALID_RETRIEVAL_RESPONSE",
    }
    assert state["requires_human_review"] is True
    assert state["final_status"] == "needs_review"


def test_required_document_finding_is_item_and_evidence_bound():
    retriever = RequirementRetriever()
    state = run_review(
        ReviewRunRequest(
            scenario_id="happy_path",
            review_run_id="run-required-document",
            review_as_of="2026-07-10",
            trade_type="import",
            jurisdiction="KR",
        ),
        retriever=retriever,
    )

    assert retriever.request is not None
    assert retriever.request.review_run_id == "run-required-document"
    assert retriever.request.items[0].item_id == "doc_inv_line_1"
    finding = state["regulatory_findings"][0]
    evidence = state["rag_evidence"][0]
    assert finding["item_id"] == evidence["item_id"] == "doc_inv_line_1"
    assert finding["evidence_ids"] == [evidence["evidence_id"]]
    assert finding["required_document_code"] == "test_import_confirmation"
    assert finding["required_document_name"] == "Test import confirmation"
    assert finding["confidence_basis"] == "retrieval_score"
    assert state["final_status"] == "needs_review"


@pytest.mark.parametrize("tampering", ["missing_evidence", "cross_item"])
def test_v2_regulatory_finding_cannot_detach_from_its_item_evidence(tampering):
    state = run_review(
        ReviewRunRequest(
            scenario_id="happy_path",
            review_run_id="run-v2-finding-binding",
            review_as_of="2026-07-10",
            trade_type="import",
            jurisdiction="KR",
        ),
        retriever=RequirementRetriever(),
    )
    payload = to_review_run_result_v2(state).model_dump(mode="json")
    ReviewRunResultV2.model_validate(payload)

    if tampering == "missing_evidence":
        replacement = ["ev_" + "f" * 64]
        payload["regulatory_findings"][0]["evidence_ids"] = replacement
        payload["report"]["regulatory_findings"][0]["evidence_ids"] = replacement
    else:
        payload["regulatory_findings"][0]["item_id"] = "different-item"
        payload["report"]["regulatory_findings"][0]["item_id"] = "different-item"

    with pytest.raises(
        ValidationError,
        match="finding-eligible Evidence|public line-item HS source",
    ):
        ReviewRunResultV2.model_validate(payload)


def test_v2_regulatory_finding_cannot_forge_an_item_detached_from_document_source():
    state = run_review(
        ReviewRunRequest(
            scenario_id="happy_path",
            review_run_id="run-v2-forged-item-source",
            review_as_of="2026-07-10",
            trade_type="import",
            jurisdiction="KR",
        ),
        retriever=RequirementRetriever(),
    )
    payload = to_review_run_result_v2(state).model_dump(mode="json")
    forged_item_id = "forged-item-not-in-document"

    for item in payload["retrieval_request"]["items"]:
        item["item_id"] = forged_item_id
    for result in payload["retrieval_results"]:
        result["item_id"] = forged_item_id
        for evidence in result["evidence"]:
            evidence["item_id"] = forged_item_id
    for evidence in payload["rag_evidence"]:
        evidence["item_id"] = forged_item_id
    for evidence in payload["report"]["rag_evidence"]:
        evidence["item_id"] = forged_item_id
    for finding in payload["regulatory_findings"]:
        finding["item_id"] = forged_item_id
    for finding in payload["report"]["regulatory_findings"]:
        finding["item_id"] = forged_item_id

    with pytest.raises(ValidationError, match="public line-item HS source"):
        ReviewRunResultV2.model_validate(payload)


def test_same_document_multi_agency_finding_cites_all_evidence():
    evidence = []
    for index, agency in enumerate(("Agency A", "Agency B"), start=1):
        source_record_id = f"record-{index}"
        version_hash = str(index) * 64
        evidence.append(
            RagEvidence(
                evidence_id=stable_evidence_id(source_record_id, version_hash),
                source_record_id=source_record_id,
                source_type="customs_requirement",
                item_id="item-1",
                title=f"CITES requirement from {agency}",
                text="A CITES permit is required.",
                law_id="law-cites",
                law_name="CITES Implementation Act",
                article_no="12",
                source_uri=f"https://example.test/{index}",
                valid_from="2026-01-01",
                version_hash=version_hash,
                matched_hs_code="7318151000",
                trade_type="import",
                required_document_code="DOC-CITES",
                required_document_name="CITES permit",
                agencies=[agency],
                retrieval_methods=["exact_hs"],
                eligibility="finding",
                score=None,
            ).model_dump(mode="json")
        )
    update = compliance_review(
        {
            "retrieval_status": "ok",
            "retrieval_coverage": "complete",
            "knowledge_snapshot_id": "snapshot-test-v1",
            "rag_evidence": evidence,
            "normalized_documents": {},
            "uploaded_requirement_document_codes": [],
            "documents": [],
            "requires_human_review": False,
            "audit_trail": [],
        }
    )
    assert len(update["regulatory_findings"]) == 1
    finding = update["regulatory_findings"][0]
    assert finding["evidence_ids"] == sorted(item["evidence_id"] for item in evidence)
    assert finding["confidence"] == 0.95
    assert finding["confidence_basis"] == "structured_requirement_policy"


def test_submitted_requirement_code_suppresses_missing_finding():
    state = run_review(
        ReviewRunRequest(
            scenario_id="happy_path",
            review_run_id="run-submitted-document",
            uploaded_requirement_document_codes=["test_import_confirmation"],
        ),
        retriever=RequirementRetriever(),
    )

    assert state["regulatory_findings"] == []
    assert state["final_status"] == "completed"
    assert state["requires_human_review"] is False


def test_response_rejects_cross_item_evidence_link():
    source_record_id = "record-1"
    version_hash = "d" * 64
    with pytest.raises(ValidationError):
        RetrievalItemResult(
            item_id="item-1",
            hs_resolution="exact",
            resolved_hsk10="7318151000",
            candidate_hsk10s=["7318151000"],
            coverage="complete",
            requirements_found=True,
            evidence=[
                RagEvidence(
                    evidence_id=stable_evidence_id(source_record_id, version_hash),
                    source_record_id=source_record_id,
                    source_type="customs_requirement",
                    item_id="item-2",
                    title="Requirement",
                    text="Requirement text",
                    law_id="law-test",
                    law_name="Test Import Act",
                    source_uri="https://example.test/1",
                    valid_from="2026-01-01",
                    version_hash=version_hash,
                    matched_hs_code="7318151000",
                    trade_type="import",
                    required_document_code="doc-code",
                    required_document_name="Document",
                    agencies=["Test Customs Agency"],
                    retrieval_methods=["exact_hs"],
                    eligibility="finding",
                )
            ],
        )


def test_initial_and_rerun_status_sources_are_consistent():
    initial = run_review(
        ReviewRunRequest(scenario_id="happy_path", review_run_id="run-consistency"),
        retriever=FailingRegulationRetriever(),
    )
    rerun = rerun_review(initial, [], retriever=FailingRegulationRetriever())

    for state in (initial, rerun):
        assert state["final_status"] == state["report_json"]["overall_status"]
        assert state["requires_human_review"] is (state["final_status"] != "completed")


def test_retriever_exception_never_leaks_exception_or_document_canary():
    canary = "SECRET-DOCUMENT-AND-TOKEN-CANARY"

    class CanaryRetriever:
        def retrieve(self, _request):
            raise RuntimeError(canary)

    state = run_review(
        ReviewRunRequest(scenario_id="happy_path", review_run_id="run-canary"),
        retriever=CanaryRetriever(),
    )
    observable_failure = json.dumps(
        {
            "retrieval_errors": state["retrieval_errors"],
            "audit_trail": state["audit_trail"],
        }
    )

    assert canary not in observable_failure
    assert state["retrieval_errors"][0]["code"] == "RETRIEVER_EXCEPTION"
    assert state["retrieval_errors"][0]["message"] == "Regulatory retrieval failed."


def test_real_retriever_is_not_silently_replaced_by_mock_on_rerun():
    initial = run_review(
        ReviewRunRequest(scenario_id="happy_path", review_run_id="run-real-rerun"),
        retriever=RequirementRetriever(),
    )

    rerun = rerun_review(initial, [])

    assert initial["knowledge_snapshot_id"] == "snapshot-test-v1"
    assert rerun["retrieval_status"] == "failed"
    assert rerun["knowledge_snapshot_id"] is None
    assert rerun["final_status"] == "needs_review"


def test_public_review_run_result_is_contract_valid():
    result = run_review_result(
        ReviewRunRequest(scenario_id="happy_path", review_run_id="run-public-result")
    )

    assert result.review_run_id == "run-public-result"
    assert result.status == result.report.overall_status == "completed"
    assert result.retrieval_status == "ok"
    assert result.retrieval_coverage == "complete"
    assert result.retrieval_request is not None
    assert result.retrieval_request.review_run_id == "run-public-result"
    assert result.retrieval_errors == []
    assert result.retrieval_results[0].item_id == "doc_inv_line_1"


def test_http_retriever_uses_v1_endpoint_and_retries_once(monkeypatch):
    request = _http_test_request()
    response = _http_test_response(request.request_id)
    calls = []

    def fake_urlopen(http_request, timeout):
        calls.append((http_request, timeout))
        if len(calls) == 1:
            raise URLError("temporary failure")
        return _HttpBody(json.dumps(response.model_dump(mode="json")).encode("utf-8"))

    monkeypatch.setattr("customs_ai_agent.adapters.retrieval.urlopen", fake_urlopen)
    actual = HttpRegulationRetriever(
        "https://evidence.local",
        allowed_origins={"https://evidence.local"},
        contract_identity_mode="disabled",
    ).retrieve(request)

    assert len(calls) == 2
    sent_request, timeout = calls[-1]
    assert sent_request.full_url == "https://evidence.local/internal/rag/v1/retrieve"
    assert sent_request.get_method() == "POST"
    assert json.loads(sent_request.data)["review_run_id"] == "run-http-1"
    assert timeout == 60.0
    assert actual == response


def test_http_retriever_rejects_invalid_response(monkeypatch):
    monkeypatch.setattr(
        "customs_ai_agent.adapters.retrieval.urlopen",
        lambda *_args, **_kwargs: _HttpBody(b'{"status":"ok"}'),
    )

    with pytest.raises(RetrievalError):
        HttpRegulationRetriever(
            "https://evidence.local",
            allowed_origins={"https://evidence.local"},
            contract_identity_mode="disabled",
        ).retrieve(_http_test_request())


@pytest.mark.parametrize("nonstandard_value", ["NaN", "Infinity", "-Infinity"])
def test_http_retriever_rejects_nonstandard_json_numbers(
    monkeypatch,
    nonstandard_value,
):
    response = RequirementRetriever().retrieve(_http_test_request())
    payload = response.model_dump(mode="json")
    payload["results"][0]["evidence"][0]["raw_scores"]["exact_hs"] = float("nan")
    body = json.dumps(payload, allow_nan=True).replace("NaN", nonstandard_value).encode("utf-8")
    monkeypatch.setattr(
        "customs_ai_agent.adapters.retrieval.urlopen",
        lambda *_args, **_kwargs: _HttpBody(body),
    )

    with pytest.raises(
        RetrievalError,
        match="Evidence Service request failed: invalid contract response",
    ):
        HttpRegulationRetriever(
            "http://localhost:8010",
            max_retries=0,
            contract_identity_mode="disabled",
        ).retrieve(_http_test_request())


def test_http_retriever_rejects_duplicate_json_keys(monkeypatch):
    response = _http_test_response("request-http-1")
    body = json.dumps(response.model_dump(mode="json")).replace(
        '"status": "ok"',
        '"status": "failed", "status": "ok"',
        1,
    )
    monkeypatch.setattr(
        "customs_ai_agent.adapters.retrieval.urlopen",
        lambda *_args, **_kwargs: _HttpBody(body.encode("utf-8")),
    )

    with pytest.raises(
        RetrievalError,
        match="Evidence Service request failed: invalid contract response",
    ):
        HttpRegulationRetriever(
            "http://localhost:8010",
            max_retries=0,
            contract_identity_mode="disabled",
        ).retrieve(_http_test_request())


def test_pinned_provider_schema_rejects_response_looser_pydantic_model_accepts(
    monkeypatch,
):
    payload = {
        "schema_version": "1.0",
        "request_id": "request-http-1",
        "status": "failed",
        "knowledge_snapshot_id": None,
        "results": [],
        "errors": [
            {
                "code": "x" * 129,
                "message": "bounded safe error",
                "retryable": False,
            }
        ],
    }
    # This regression demonstrates why the provider-owned packaged schema is
    # authoritative at HTTP boundaries even while internal models stay
    # backward compatible with released Agent result schemas.
    EvidenceRetrievalResponse.model_validate(payload)
    monkeypatch.setattr(
        "customs_ai_agent.adapters.retrieval.urlopen",
        lambda *_args, **_kwargs: _HttpBody(json.dumps(payload).encode("utf-8")),
    )

    with pytest.raises(
        RetrievalError,
        match="Evidence Service request failed: invalid contract response",
    ):
        HttpRegulationRetriever(
            "http://localhost:8010",
            max_retries=0,
            contract_identity_mode="disabled",
        ).retrieve(_http_test_request())


def test_http_retriever_rejects_schema_invalid_constructed_request_before_network(
    monkeypatch,
):
    monkeypatch.setattr(
        "customs_ai_agent.adapters.retrieval.urlopen",
        lambda *_args, **_kwargs: pytest.fail("schema-invalid request must not reach HTTP"),
    )
    request = _http_test_request().model_copy(update={"request_id": "x" * 129})

    with pytest.raises(
        RetrievalError,
        match="request does not satisfy the pinned contract",
    ):
        HttpRegulationRetriever(
            "http://localhost:8010",
            contract_identity_mode="disabled",
        ).retrieve(request)


def test_consumer_contract_bundle_failure_stops_before_network(monkeypatch):
    monkeypatch.setattr(
        "customs_ai_agent.adapters.retrieval.urlopen",
        lambda *_args, **_kwargs: pytest.fail("invalid consumer bundle must not reach HTTP"),
    )
    monkeypatch.setattr(
        "customs_ai_agent.adapters.retrieval.load_pinned_evidence_contract_identity",
        lambda: (_ for _ in ()).throw(EvidenceContractLockError("private detail")),
    )

    with pytest.raises(
        RetrievalError,
        match="consumer contract bundle is invalid",
    ):
        HttpRegulationRetriever("http://localhost:8010")


@pytest.mark.parametrize(
    ("raw", "normalized"),
    [
        ("http://localhost:8010/", "http://localhost:8010"),
        ("http://127.0.0.1:8010", "http://127.0.0.1:8010"),
        ("http://[::1]:8010", "http://[::1]:8010"),
        ("https://evidence.example:8443", "https://evidence.example:8443"),
    ],
)
def test_evidence_base_url_accepts_loopback_http_or_https(raw, normalized):
    assert validate_evidence_base_url(raw) == normalized


@pytest.mark.parametrize(
    "url",
    [
        "http://evidence.example",
        "file:///tmp/evidence",
        "https://user:secret@evidence.example",
        "https://evidence.example?token=secret",
        "https://evidence.example#fragment",
        "https://evidence.example/prefix",
        "https://evidence.example:99999",
    ],
)
def test_evidence_base_url_rejects_unsafe_or_ambiguous_values(url):
    with pytest.raises(ValueError):
        validate_evidence_base_url(url)


def test_external_evidence_client_requires_exact_origin_allowlist():
    with pytest.raises(ValueError, match="not in the explicit allowlist"):
        HttpRegulationRetriever("https://evidence.example")

    retriever = HttpRegulationRetriever(
        "https://evidence.example:8443",
        allowed_origins={
            "https://other.example",
            "https://evidence.example:8443",
        },
    )

    assert retriever.url == ("https://evidence.example:8443/internal/rag/v1/retrieve")


def test_http_retriever_defaults_to_required_contract_identity_handshake():
    retriever = HttpRegulationRetriever("http://localhost:8010")

    assert retriever.contract_identity_mode == "required"


def test_evidence_redirect_handler_never_creates_cross_origin_request():
    original = Request(
        "https://evidence.example/internal/rag/v1/retrieve",
        headers={"X-Demo-Token": "must-not-forward"},
    )

    redirected = _NoRedirectHandler().redirect_request(
        original,
        None,
        302,
        "Found",
        {},
        "https://attacker.example/collect",
    )

    assert redirected is None


def test_oversized_evidence_response_fails_without_exposing_body(monkeypatch):
    monkeypatch.setattr(
        "customs_ai_agent.adapters.retrieval.urlopen",
        lambda *_args, **_kwargs: _HttpBody(b"x" * (MAX_EVIDENCE_RESPONSE_BYTES + 1)),
    )

    with pytest.raises(RetrievalError) as exc_info:
        HttpRegulationRetriever(
            "http://127.0.0.1:8010",
            max_retries=0,
            contract_identity_mode="disabled",
        ).retrieve(_http_test_request())

    assert str(exc_info.value) == ("Evidence Service request failed: invalid contract response")


def test_http_retriever_sends_token_only_in_configured_header(monkeypatch):
    secret = "header-only-secret"
    response = _http_test_response("request-http-1")
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _HttpBody(json.dumps(response.model_dump(mode="json")).encode("utf-8"))

    monkeypatch.setattr("customs_ai_agent.adapters.retrieval.urlopen", fake_urlopen)
    retriever = HttpRegulationRetriever(
        "http://127.0.0.1:8010",
        service_token=secret,
        token_header="X-Demo-Token",
        contract_identity_mode="disabled",
    )

    retriever.retrieve(_http_test_request())
    request = captured["request"]
    sent_headers = {name.casefold(): value for name, value in request.header_items()}
    assert sent_headers["x-demo-token"] == secret
    assert secret not in request.full_url
    assert secret not in retriever.url


@pytest.mark.parametrize(
    ("token", "header"),
    [
        ("secret\r\ninjected: yes", "X-Demo-Token"),
        ("secret", "X-Demo-Token\r\nInjected"),
        ("secret", "X-Request-ID"),
        ("secret", EVIDENCE_PROVIDER_BUILD_SHA256_HEADER),
    ],
)
def test_http_retriever_rejects_unsafe_token_or_header(token, header):
    with pytest.raises(ValueError):
        HttpRegulationRetriever(
            "http://127.0.0.1:8010",
            service_token=token,
            token_header=header,
        )


def test_http_error_does_not_expose_service_token(monkeypatch):
    secret = "do-not-leak-this-token"
    monkeypatch.setattr(
        "customs_ai_agent.adapters.retrieval.urlopen",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(URLError(secret)),
    )

    with pytest.raises(RetrievalError) as captured:
        HttpRegulationRetriever(
            "http://127.0.0.1:8010",
            service_token=secret,
            contract_identity_mode="disabled",
        ).retrieve(_http_test_request())

    assert str(captured.value) == "Evidence Service request failed: network error"
    assert secret not in str(captured.value)


def test_readiness_returns_provider_supported_review_date_and_token(monkeypatch):
    captured = {}

    def fake_urlopen(request, timeout):
        captured["request"] = request
        captured["timeout"] = timeout
        return _HttpBody(
            json.dumps(
                {
                    "status": "ready",
                    "knowledge_snapshot_id": "snapshot-1",
                    "supported_review_as_of": "2026-07-10",
                }
            ).encode("utf-8")
        )

    monkeypatch.setattr("customs_ai_agent.adapters.retrieval.urlopen", fake_urlopen)

    actual = fetch_supported_review_as_of(
        "http://localhost:8010",
        service_token="readiness-secret",
        max_retries=0,
        contract_identity_mode="optional",
    )
    request = captured["request"]
    sent_headers = {name.casefold(): value for name, value in request.header_items()}
    assert actual == date(2026, 7, 10)
    assert request.get_method() == "GET"
    assert request.full_url == "http://localhost:8010/internal/rag/v1/health/ready"
    assert sent_headers["x-demo-token"] == "readiness-secret"
    assert captured["timeout"] == 10.0


def test_required_readiness_verifies_exact_pinned_contract_identity(monkeypatch):
    monkeypatch.setattr(
        "customs_ai_agent.adapters.retrieval.urlopen",
        lambda *_args, **_kwargs: _HttpBody(json.dumps(_readiness_payload()).encode("utf-8")),
    )

    readiness = fetch_evidence_readiness(
        "http://localhost:8010",
        max_retries=0,
        contract_identity_mode="required",
    )
    expected = load_pinned_evidence_contract_identity()

    assert readiness.knowledge_snapshot_id == "snapshot-1"
    assert readiness.supported_review_as_of == date(2026, 7, 10)
    assert readiness.contract_identity == expected


def test_optional_readiness_allows_missing_identity_only_for_explicit_compatibility(
    monkeypatch,
):
    monkeypatch.setattr(
        "customs_ai_agent.adapters.retrieval.urlopen",
        lambda *_args, **_kwargs: _HttpBody(
            json.dumps(_readiness_payload(include_identity=False)).encode("utf-8")
        ),
    )

    readiness = fetch_evidence_readiness(
        "http://localhost:8010",
        max_retries=0,
        contract_identity_mode="optional",
    )

    assert readiness.contract_identity is None
    assert readiness.supported_review_as_of == date(2026, 7, 10)


def test_required_readiness_rejects_provider_that_omits_contract_identity(monkeypatch):
    monkeypatch.setattr(
        "customs_ai_agent.adapters.retrieval.urlopen",
        lambda *_args, **_kwargs: _HttpBody(
            json.dumps(_readiness_payload(include_identity=False)).encode("utf-8")
        ),
    )

    with pytest.raises(
        RetrievalError,
        match="readiness request failed: contract identity missing",
    ):
        fetch_evidence_readiness(
            "http://localhost:8010",
            max_retries=0,
            contract_identity_mode="required",
        )


@pytest.mark.parametrize(
    ("field", "mismatch"),
    [
        ("provider_release", "evidence-api-v1.0.1"),
        ("provider_revision", "b" * 40),
        ("provider_build_sha256", "b" * 64),
        ("schema_sha256", "b" * 64),
    ],
)
def test_readiness_rejects_well_formed_contract_identity_mismatch(
    monkeypatch,
    field,
    mismatch,
):
    payload = _readiness_payload()
    identity = dict(payload["evidence_contract"])
    identity[field] = mismatch
    payload["evidence_contract"] = identity
    monkeypatch.setattr(
        "customs_ai_agent.adapters.retrieval.urlopen",
        lambda *_args, **_kwargs: _HttpBody(json.dumps(payload).encode("utf-8")),
    )

    with pytest.raises(
        RetrievalError,
        match="readiness request failed: contract identity mismatch",
    ):
        fetch_evidence_readiness(
            "http://localhost:8010",
            max_retries=0,
            contract_identity_mode="required",
        )


def test_required_http_retrieval_attests_before_every_post(monkeypatch):
    response = _http_test_response("request-http-1")
    methods = []
    post_headers = []

    def fake_urlopen(request, timeout):
        assert timeout == 60.0
        methods.append(request.get_method())
        if request.get_method() == "POST":
            post_headers.append({name.casefold(): value for name, value in request.header_items()})
        payload = (
            _readiness_payload(snapshot_id="snapshot-http-v1")
            if request.get_method() == "GET"
            else response.model_dump(mode="json")
        )
        return _HttpBody(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr("customs_ai_agent.adapters.retrieval.urlopen", fake_urlopen)
    retriever = HttpRegulationRetriever(
        "http://localhost:8010",
        max_retries=0,
        contract_identity_mode="required",
    )

    assert retriever.retrieve(_http_test_request()) == response
    assert retriever.retrieve(_http_test_request()) == response
    assert methods == ["GET", "POST", "GET", "POST"]
    expected_headers = {name.casefold(): value for name, value in _identity_headers().items()}
    assert all(
        {name: headers[name] for name in expected_headers} == expected_headers
        for headers in post_headers
    )


def test_required_http_retrieval_rejects_get_post_build_mismatch_before_body_read(
    monkeypatch,
):
    response = _http_test_response("request-http-1")
    methods = []
    post_body = _HttpBody(
        json.dumps(response.model_dump(mode="json")).encode("utf-8"),
        headers={
            **_identity_headers(),
            EVIDENCE_PROVIDER_BUILD_SHA256_HEADER: "b" * 64,
        },
    )

    def fake_urlopen(request, timeout):
        assert timeout == 60.0
        methods.append(request.get_method())
        if request.get_method() == "GET":
            return _HttpBody(
                json.dumps(_readiness_payload(snapshot_id="snapshot-http-v1")).encode("utf-8")
            )
        return post_body

    monkeypatch.setattr("customs_ai_agent.adapters.retrieval.urlopen", fake_urlopen)

    with pytest.raises(
        RetrievalError,
        match=r"^Evidence Service request failed: contract identity mismatch$",
    ):
        HttpRegulationRetriever(
            "http://localhost:8010",
            max_retries=0,
            contract_identity_mode="required",
        ).retrieve(_http_test_request())

    assert methods == ["GET", "POST"]
    assert post_body.read_count == 0


def test_required_http_retrieval_rejects_missing_response_identity_before_body_read(
    monkeypatch,
):
    response = _http_test_response("request-http-1")
    post_body = _HttpBody(
        json.dumps(response.model_dump(mode="json")).encode("utf-8"),
        headers={},
    )

    def fake_urlopen(request, **_kwargs):
        if request.get_method() == "GET":
            return _HttpBody(
                json.dumps(_readiness_payload(snapshot_id="snapshot-http-v1")).encode("utf-8")
            )
        return post_body

    monkeypatch.setattr("customs_ai_agent.adapters.retrieval.urlopen", fake_urlopen)

    with pytest.raises(
        RetrievalError,
        match=r"^Evidence Service request failed: contract identity mismatch$",
    ):
        HttpRegulationRetriever(
            "http://localhost:8010",
            max_retries=1,
        ).retrieve(_http_test_request())

    assert post_body.read_count == 0


def test_retrieval_retry_revalidates_identity_and_rejects_build_drift(
    monkeypatch,
):
    response = _http_test_response("request-http-1")
    methods = []
    drifted_body = _HttpBody(
        json.dumps(response.model_dump(mode="json")).encode("utf-8"),
        headers={
            **_identity_headers(),
            EVIDENCE_PROVIDER_BUILD_SHA256_HEADER: "c" * 64,
        },
    )
    post_attempt = 0

    def fake_urlopen(request, **_kwargs):
        nonlocal post_attempt
        method = request.get_method()
        methods.append(method)
        if method == "GET":
            return _HttpBody(
                json.dumps(_readiness_payload(snapshot_id="snapshot-http-v1")).encode("utf-8")
            )
        post_attempt += 1
        if post_attempt == 1:
            raise URLError("private transient detail")
        return drifted_body

    monkeypatch.setattr("customs_ai_agent.adapters.retrieval.urlopen", fake_urlopen)

    with pytest.raises(
        RetrievalError,
        match=r"^Evidence Service request failed: contract identity mismatch$",
    ):
        HttpRegulationRetriever(
            "http://localhost:8010",
            max_retries=1,
        ).retrieve(_http_test_request())

    assert methods == ["GET", "POST", "POST"]
    assert drifted_body.read_count == 0


def test_http_412_is_safe_and_never_reads_provider_error_body(monkeypatch):
    private_reason = "private rollout detail"
    methods = []

    def fake_urlopen(request, **_kwargs):
        methods.append(request.get_method())
        if request.get_method() == "GET":
            return _HttpBody(
                json.dumps(_readiness_payload(snapshot_id="snapshot-http-v1")).encode("utf-8")
            )
        raise HTTPError(
            request.full_url,
            412,
            private_reason,
            hdrs=None,
            fp=None,
        )

    monkeypatch.setattr("customs_ai_agent.adapters.retrieval.urlopen", fake_urlopen)

    with pytest.raises(RetrievalError) as captured:
        HttpRegulationRetriever(
            "http://localhost:8010",
            max_retries=1,
        ).retrieve(_http_test_request())

    assert str(captured.value) == "Evidence Service request failed: HTTP status 412"
    assert private_reason not in str(captured.value)
    assert methods == ["GET", "POST"]


def test_required_http_retrieval_rejects_snapshot_changed_after_readiness(monkeypatch):
    response = _http_test_response("request-http-1")
    methods = []

    def fake_urlopen(request, timeout):
        assert timeout == 60.0
        methods.append(request.get_method())
        payload = (
            _readiness_payload(snapshot_id="readiness-snapshot")
            if request.get_method() == "GET"
            else response.model_dump(mode="json")
        )
        return _HttpBody(json.dumps(payload).encode("utf-8"))

    monkeypatch.setattr("customs_ai_agent.adapters.retrieval.urlopen", fake_urlopen)

    with pytest.raises(RetrievalError, match="snapshot does not match readiness"):
        HttpRegulationRetriever(
            "http://localhost:8010",
            max_retries=0,
        ).retrieve(_http_test_request())

    assert methods == ["GET", "POST"]


def test_required_http_retrieval_rejects_unsupported_as_of_before_post(monkeypatch):
    methods = []

    def fake_urlopen(request, timeout):
        assert timeout == 60.0
        methods.append(request.get_method())
        return _HttpBody(json.dumps(_readiness_payload()).encode("utf-8"))

    monkeypatch.setattr("customs_ai_agent.adapters.retrieval.urlopen", fake_urlopen)
    retriever = HttpRegulationRetriever(
        "http://localhost:8010",
        max_retries=0,
        contract_identity_mode="required",
    )
    request = _http_test_request().model_copy(update={"review_as_of": date(2026, 7, 11)})

    with pytest.raises(RetrievalError, match="does not support the requested review_as_of"):
        retriever.retrieve(request)

    assert methods == ["GET"]


def test_missing_runtime_contract_identity_fails_closed_without_retrieval_post(monkeypatch):
    methods = []

    def fake_urlopen(request, timeout):
        assert timeout == 60.0
        methods.append(request.get_method())
        return _HttpBody(json.dumps(_readiness_payload(include_identity=False)).encode("utf-8"))

    monkeypatch.setattr("customs_ai_agent.adapters.retrieval.urlopen", fake_urlopen)
    retriever = HttpRegulationRetriever(
        "http://localhost:8010",
        max_retries=0,
        contract_identity_mode="required",
    )
    state = run_review(
        ReviewRunRequest(
            scenario_id="happy_path",
            review_run_id="run-missing-runtime-contract-identity",
            review_as_of="2026-07-10",
        ),
        retriever=retriever,
    )

    assert methods == ["GET"]
    assert state["retrieval_status"] == "failed"
    assert state["retrieval_coverage"] == "unknown"
    assert state["knowledge_snapshot_id"] is None
    assert state["rag_evidence"] == []
    assert state["requires_human_review"] is True
    assert state["final_status"] == "needs_review"


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "starting", "supported_review_as_of": "2026-07-10"},
        {"status": "ready"},
        {"status": "ready", "supported_review_as_of": "not-a-date"},
        {"status": "ready", "supported_review_as_of": "20260710"},
    ],
)
def test_readiness_rejects_missing_or_invalid_supported_date(monkeypatch, payload):
    monkeypatch.setattr(
        "customs_ai_agent.adapters.retrieval.urlopen",
        lambda *_args, **_kwargs: _HttpBody(json.dumps(payload).encode("utf-8")),
    )

    with pytest.raises(
        RetrievalError,
        match="readiness request failed: invalid contract response",
    ):
        fetch_supported_review_as_of("http://localhost:8010", max_retries=0)


def test_evidence_valid_to_is_inclusive_contract_boundary():
    state = {
        "review_as_of": "2026-07-10",
        "trade_type": "import",
    }
    results = [
        {
            "item_id": "item-1",
            "resolved_hsk10": "0303540000",
            "evidence": [
                {
                    "evidence_id": "ev-boundary",
                    "source_type": "customs_requirement",
                    "matched_hs_code": "0303540000",
                    "trade_type": "import",
                    "valid_from": "2026-01-01",
                    "valid_to": "2026-07-10",
                }
            ],
        }
    ]

    assert _context_errors(state, results) == []
    state["review_as_of"] = "2026-07-11"
    errors = _context_errors(state, results)
    assert [error["code"] for error in errors] == ["EVIDENCE_EXPIRED"]
