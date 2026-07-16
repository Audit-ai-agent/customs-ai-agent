from __future__ import annotations

import json
from datetime import date
from urllib.error import URLError

from pydantic import ValidationError
import pytest

from customs_ai_agent.adapters.retrieval import (
    EmptyRegulationRetriever,
    FailingRegulationRetriever,
    HttpRegulationRetriever,
    PartialRegulationRetriever,
    RetrievalError,
    fetch_supported_review_as_of,
    validate_evidence_base_url,
)
from customs_ai_agent.runner import rerun_review, run_review, run_review_result
from customs_ai_agent.nodes.compliance import compliance_review
from customs_ai_agent.nodes.retrieve import _context_errors
from customs_ai_agent.schemas.models import (
    EvidenceRetrievalRequest,
    EvidenceRetrievalResponse,
    RagEvidence,
    RetrievalItemResult,
    ReviewRunRequest,
    stable_evidence_id,
)


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


class _HttpBody:
    def __init__(self, payload: bytes) -> None:
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None

    def read(self) -> bytes:
        return self.payload


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


def test_ambiguous_hs_cannot_complete_even_with_complete_coverage():
    state = run_review(
        ReviewRunRequest(scenario_id="happy_path", review_run_id="run-ambiguous"),
        retriever=AmbiguousCompleteRetriever(),
    )

    assert state["retrieval_status"] == "ok"
    assert state["retrieval_coverage"] == "complete"
    assert state["retrieval_results"][0]["hs_resolution"] == "ambiguous"
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


def test_same_document_multi_agency_finding_cites_all_evidence():
    evidence = []
    for index, agency in enumerate(("Agency A", "Agency B"), start=1):
        evidence.append({
            "evidence_id": f"ev-{index}",
            "source_record_id": f"record-{index}",
            "source_uri": f"https://example.test/{index}",
            "version_hash": str(index) * 64,
            "source_type": "customs_requirement",
            "item_id": "item-1",
            "eligibility": "finding",
            "required_document_code": "DOC-CITES",
            "required_document_name": "CITES permit",
            "agencies": [agency],
            "score": None,
        })
    update = compliance_review({
        "rag_evidence": evidence,
        "normalized_documents": {},
        "uploaded_requirement_document_codes": [],
        "documents": [],
        "requires_human_review": False,
        "audit_trail": [],
    })
    assert len(update["regulatory_findings"]) == 1
    finding = update["regulatory_findings"][0]
    assert finding["evidence_ids"] == ["ev-1", "ev-2"]
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
                    source_uri="https://example.test/1",
                    version_hash=version_hash,
                    trade_type="import",
                    required_document_code="doc-code",
                    required_document_name="Document",
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
    actual = HttpRegulationRetriever("https://evidence.local").retrieve(request)

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
        HttpRegulationRetriever("https://evidence.local").retrieve(_http_test_request())


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
    )
    request = captured["request"]
    sent_headers = {name.casefold(): value for name, value in request.header_items()}
    assert actual == date(2026, 7, 10)
    assert request.get_method() == "GET"
    assert request.full_url == "http://localhost:8010/internal/rag/v1/health/ready"
    assert sent_headers["x-demo-token"] == "readiness-secret"
    assert captured["timeout"] == 10.0


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
