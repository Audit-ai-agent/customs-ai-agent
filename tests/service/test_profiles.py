from __future__ import annotations

import json
from dataclasses import replace
from typing import Any
from urllib.error import HTTPError

import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

import customs_ai_agent.service.profiles as profile_module
from customs_ai_agent.adapters.retrieval import RetrievalError
from customs_ai_agent.schemas.v3 import ServiceReviewRunRequestV3, canonical_json_sha256
from customs_ai_agent.service.app import create_app
from customs_ai_agent.service.profiles import (
    _DeterministicDocumentAnalyzer,
    _actions,
    _dedupe_evidence_records,
    _findings_from_evidence,
    ProfileEvidenceRetriever,
    ProfileRuntime,
    SERVICE_MODE_HEADER,
)
from customs_ai_agent.service.runtime_errors import RuntimeConfigurationError
from customs_ai_agent.service.settings import ServiceConfigurationError, ServiceSettings


_TOKEN = "Abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJK"
_GUIDANCE_TOKEN = "Zyxwvutsrqponmlkjihgfedcba9876543210ABCDEFGHIJK"


def _settings(**changes: Any) -> ServiceSettings:
    return replace(
        ServiceSettings(
            bearer_token=SecretStr(_TOKEN),
            evidence_base_url="http://127.0.0.1:18083",
            evidence_service_token=SecretStr(_TOKEN),
        ),
        **changes,
    )


def test_default_profile_is_local_development() -> None:
    settings = ServiceSettings.from_environment(
        {
            "CUSTOMS_AGENT_SERVICE_BEARER_TOKEN": _TOKEN,
            "EVIDENCE_SERVICE_BASE_URL": "http://127.0.0.1:18083",
            "EVIDENCE_SERVICE_TOKEN": _TOKEN,
            "EVIDENCE_SERVICE_TOKEN_HEADER": "Authorization",
        }
    )
    assert settings.service_mode == "local-development"
    assert settings.port == 18090
    assert settings.guidance_base_url is None
    assert settings.guidance_service_token is None


def test_regulatory_requirement_is_one_finding_and_action_across_linked_documents() -> None:
    evidence = [
        {
            "evidence_id": "evidence-food-1",
            "item_id": "commodity-one",
            "title": "수입식품등 수입신고확인증",
            "eligibility": "finding",
            "law_id": "food-law",
            "required_document_code": "FOOD_IMPORT_CONFIRMATION",
            "required_document_name": "수입식품등 수입신고확인증",
        },
        {
            "evidence_id": "evidence-food-2",
            "item_id": "commodity-two",
            "title": "수입식품등 수입신고확인증",
            "eligibility": "finding",
            "law_id": "food-law",
            "required_document_code": "FOOD_IMPORT_CONFIRMATION",
            "required_document_name": "수입식품등 수입신고확인증",
        },
    ]

    deduped = _dedupe_evidence_records(evidence)
    findings = _findings_from_evidence(
        deduped,
        document_ids_by_item={
            "commodity-one": ["invoice", "packing"],
            "commodity-two": ["certificate"],
        },
    )
    actions = _actions(findings)

    assert len(deduped) == 1
    assert len(findings) == 1
    assert findings[0].document_ids == ["certificate", "invoice", "packing"]
    assert findings[0].evidence_ids == ["evidence-food-1"]
    regulatory_actions = [action for action in actions if action.finding_ids]
    assert len(regulatory_actions) == 1
    assert regulatory_actions[0].finding_ids == [findings[0].finding_id]


def test_deterministic_classifier_keeps_a_packing_header_over_a_later_bl_reference() -> None:
    analysis = _DeterministicDocumentAnalyzer().analyze(
        {
            "document_id": "packing",
            "filename": "packing.txt",
            "content_text": "PACKING LIST\nB/L No: BL-001",
        }
    )

    assert analysis.candidate.document_type == "packing_list"


def test_deterministic_extractor_does_not_treat_an_invoice_total_as_a_line_item() -> None:
    analysis = _DeterministicDocumentAnalyzer().analyze(
        {
            "document_id": "invoice",
            "filename": "invoice.txt",
            "content_text": (
                "COMMERCIAL INVOICE\n"
                "Total Amount: USD 30\n"
                "Description: Widget | HS Code: 123456 | Quantity: 3 PCS | "
                "Unit Price: USD 10 | Amount: USD 30"
            ),
        }
    )

    assert len(analysis.candidate.line_items) == 1


@pytest.mark.parametrize(
    ("document_id", "content_text", "expected_type", "expected_awb"),
    [
        (
            "packing-awb-number",
            "PACKING LIST\nPacking No: PL-AWB-001\nAWB Number: AWB-EXPORT-001",
            "packing_list",
            "AWB-EXPORT-001",
        ),
        (
            "air-waybill-awb-no",
            "AIR WAYBILL\nAWB No: AWB-EXPORT-001\nGross Weight: 100 KG",
            "air_waybill",
            "AWB-EXPORT-001",
        ),
    ],
)
def test_deterministic_extractor_recognizes_standard_awb_number_labels(
    document_id: str, content_text: str, expected_type: str, expected_awb: str
) -> None:
    """Packing and AWB documents retain the same source-grounded AWB value."""

    analysis = _DeterministicDocumentAnalyzer().analyze(
        {
            "document_id": document_id,
            "filename": "transport.txt",
            "content_text": content_text,
        }
    )

    values = {field.name: field.value for field in analysis.candidate.fields}
    assert analysis.candidate.document_type == expected_type
    assert values["awb_no"] == expected_awb


def test_deterministic_extractor_keeps_missing_awb_as_a_reviewable_absence() -> None:
    """An AWB heading alone must not manufacture an identifier for comparison."""

    analysis = _DeterministicDocumentAnalyzer().analyze(
        {
            "document_id": "packing-with-awb-like-reference",
            "filename": "transport.txt",
            "content_text": "PACKING LIST\nPacking No: PL-AWB-001\nTotal Gross Weight: 100 KG",
        }
    )

    assert analysis.candidate.document_type == "packing_list"
    assert "awb_no" not in {field.name for field in analysis.candidate.fields}


def test_deterministic_extractor_supports_korean_export_and_submission_documents() -> None:
    declaration = _DeterministicDocumentAnalyzer().analyze(
        {
            "document_id": "export-declaration",
            "filename": "export.txt",
            "content_text": (
                "수출신고필증\n수출신고번호: ED-2026-001\n신고일자: 2026-08-05\n"
                "처리상태: 수리\n송장번호: INV-2026-001\n총액: USD 20\n원산지: KR\n"
                "목적국: US\n품명: Widget | HS 코드: 123456 | 수량: 2 PCS | 단위: PCS | 금액: USD 20"
            ),
        }
    )
    support = _DeterministicDocumentAnalyzer().analyze(
        {
            "document_id": "support",
            "filename": "support.txt",
            "content_text": "보완 증빙\n접수번호: RECEIPT-001\n제출일자: 2026-08-05\n처리상태: 승인",
        }
    )

    assert declaration.candidate.document_type == "export_declaration"
    assert {field.name for field in declaration.candidate.fields} >= {
        "export_declaration_no",
        "filing_date",
        "submission_status",
        "invoice_no",
        "total_amount",
        "origin_country",
        "destination_country",
    }
    assert declaration.candidate.line_items[0].fields[0].name == "description"
    assert support.candidate.document_type == "supporting_evidence"
    assert {field.name for field in support.candidate.fields} == {
        "submission_receipt_no",
        "filing_date",
        "submission_status",
    }


def test_backend_profile_loader_mode_is_accepted() -> None:
    settings = ServiceSettings.from_environment(
        {
            "CUSTOMS_AGENT_SERVICE_BEARER_TOKEN": _TOKEN,
            "EVIDENCE_SERVICE_BASE_URL": "http://127.0.0.1:18083",
            "EVIDENCE_SERVICE_TOKEN": _TOKEN,
            "EVIDENCE_SERVICE_TOKEN_HEADER": "Authorization",
            "CUSTOMS_AGENT_SERVICE_MODE": "functional-uat",
        }
    )
    assert settings.service_mode == "functional-uat"


def test_profile_rejects_non_loopback_evidence() -> None:
    with pytest.raises(RuntimeConfigurationError) as raised:
        ProfileEvidenceRetriever(replace(_settings(), evidence_base_url="https://example.invalid"))
    assert raised.value.code == "evidence_base_url_invalid"


@pytest.mark.parametrize(
    "environment",
    [
        {
            "CUSTOMS_AGENT_SERVICE_BEARER_TOKEN": _TOKEN,
            "EVIDENCE_SERVICE_BASE_URL": "http://127.0.0.1:18083",
            "EVIDENCE_SERVICE_TOKEN": _TOKEN,
            "GUIDANCE_SERVICE_BASE_URL": "http://127.0.0.1:18093",
        },
        {
            "CUSTOMS_AGENT_SERVICE_BEARER_TOKEN": _TOKEN,
            "EVIDENCE_SERVICE_BASE_URL": "http://127.0.0.1:18083",
            "EVIDENCE_SERVICE_TOKEN": _TOKEN,
            "GUIDANCE_SERVICE_TOKEN": _GUIDANCE_TOKEN,
        },
    ],
)
def test_guidance_peer_requires_a_complete_url_and_token_pair(
    environment: dict[str, str],
) -> None:
    with pytest.raises(ServiceConfigurationError) as raised:
        ServiceSettings.from_environment(environment)
    assert raised.value.code == "guidance_service_peer_incomplete"


def test_profile_rejects_non_loopback_guidance_peer() -> None:
    settings = replace(
        _settings(),
        guidance_base_url="https://example.invalid",
        guidance_service_token=SecretStr(_GUIDANCE_TOKEN),
    )
    with pytest.raises(RuntimeConfigurationError) as raised:
        ProfileEvidenceRetriever(settings)
    assert raised.value.code == "guidance_base_url_invalid"


def test_guidance_uses_its_own_loopback_origin_and_token_while_review_stays_on_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The composed profile must never redirect document review to guidance."""

    class Headers(dict[str, str]):
        def get_all(self, _name: str) -> None:
            return None

    calls: list[tuple[str, str | None]] = []

    def open_request(request: Any, **_kwargs: Any) -> object:
        headers = {key.casefold(): value for key, value in request.header_items()}
        calls.append((request.full_url, headers.get("authorization")))
        if request.full_url.endswith(ProfileEvidenceRetriever.readiness_path):
            payload: dict[str, Any] = {"status": "ready"}
        elif request.full_url.endswith(ProfileEvidenceRetriever.retrieve_path):
            body = json.loads(request.data.decode("utf-8"))
            payload = {
                "schema_version": "3.0",
                "request_id": body["request_id"],
                "status": "ok",
                "knowledge_snapshot_id": "functional-uat-3law",
                "results": [],
            }
        else:
            body = json.loads(request.data.decode("utf-8"))
            payload = {
                "schema_version": "3.0",
                "request_id": body["request_id"],
                "status": "covered",
                "snapshot_id": "functional-uat-kr-trade",
                "as_of": body["as_of"],
                "coverage": "covered",
                "registry_version": "1.0",
                "topic": body["topic_hint"],
                "results": [],
                "errors": [],
            }

        class Response:
            headers = Headers({SERVICE_MODE_HEADER: "functional-uat"})

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self, _limit: int) -> bytes:
                return json.dumps(payload).encode("utf-8")

        return Response()

    monkeypatch.setattr(profile_module._HTTP, "open", open_request)
    retriever = ProfileEvidenceRetriever(
        replace(
            _settings(service_mode="functional-uat"),
            guidance_base_url="http://127.0.0.1:18093",
            guidance_service_token=SecretStr(_GUIDANCE_TOKEN),
        )
    )
    retriever.retrieve({"request_id": "evidence-request", "review_run_id": "review-run"})
    retriever.guidance_query(
        {
            "schema_version": "3.0",
            "request_id": "guidance-request",
            "query": "수출 신고 절차를 알려 주세요.",
            "trade_direction": "export",
            "topic_hint": "customs_declaration",
            "as_of": "2026-08-05",
            "top_k": 3,
        }
    )

    assert calls == [
        ("http://127.0.0.1:18083/internal/rag/v3/health/ready", f"Bearer {_TOKEN}"),
        ("http://127.0.0.1:18083/internal/rag/v3/retrieve", f"Bearer {_TOKEN}"),
        ("http://127.0.0.1:18093/internal/rag/v3/guidance/query", f"Bearer {_GUIDANCE_TOKEN}"),
    ]


@pytest.mark.parametrize("failure", ["mode", "token"])
def test_guidance_peer_mode_or_token_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    """A separate peer cannot degrade into a successful general answer."""

    class Headers(dict[str, str]):
        def get_all(self, _name: str) -> None:
            return None

    def open_request(request: Any, **_kwargs: Any) -> object:
        if failure == "token":
            assert request.get_header("Authorization") == f"Bearer {_GUIDANCE_TOKEN}"
            raise HTTPError(request.full_url, 401, "", Headers(), None)

        class Response:
            headers = Headers({SERVICE_MODE_HEADER: "local-development"})

            def __enter__(self) -> "Response":
                return self

            def __exit__(self, *_args: object) -> None:
                return None

            def read(self, _limit: int) -> bytes:
                return b"{}"

        return Response()

    monkeypatch.setattr(profile_module._HTTP, "open", open_request)
    retriever = ProfileEvidenceRetriever(
        replace(
            _settings(service_mode="functional-uat"),
            guidance_base_url="http://127.0.0.1:18093",
            guidance_service_token=SecretStr(_GUIDANCE_TOKEN),
        )
    )
    with pytest.raises(RetrievalError):
        retriever.guidance_query(
            {
                "schema_version": "3.0",
                "request_id": "guidance-failed",
                "query": "수출 신고 절차를 알려 주세요.",
                "trade_direction": "export",
                "topic_hint": "customs_declaration",
                "as_of": "2026-08-05",
                "top_k": 3,
            }
        )


def test_review_input_manifest_hashes_the_exact_backend_wire_for_baseline_and_child_runs() -> None:
    baseline_wire = {
        "case_id": "case-input-manifest",
        "review_run_id": "run-input-manifest",
        "review_as_of": "2026-08-05",
        "trade_type": "import",
        "jurisdiction": "KR",
        "documents": [
            {
                "document_id": "document-input-manifest",
                "filename": "invoice.txt",
                "content_text": "COMMERCIAL INVOICE\\nInvoice No: INV-001",
            }
        ],
    }
    baseline = ServiceReviewRunRequestV3.model_validate(baseline_wire)
    assert baseline.input_manifest_wire() == baseline_wire
    assert canonical_json_sha256(baseline.input_manifest_wire()) == canonical_json_sha256(
        baseline_wire
    )

    child_wire = {
        **baseline_wire,
        "review_run_id": "run-input-manifest-child",
        "parent_review_run_id": "run-input-manifest",
        "applied_correction_proposal_ids": ["correction-input-manifest"],
        "human_corrections": [
            {
                "correction_proposal_id": "correction-input-manifest",
                "document_id": "document-input-manifest",
                "field_name": "invoice_number",
                "new_value": "INV-001-CORRECTED",
                "reason": "Reviewer confirmed the text document.",
                "source_kind": "human_provided",
            }
        ],
    }
    child = ServiceReviewRunRequestV3.model_validate(child_wire)
    assert child.input_manifest_wire() == child_wire
    assert canonical_json_sha256(child.input_manifest_wire()) == canonical_json_sha256(child_wire)


def test_review_uses_one_readiness_then_one_retrieve_and_stays_human_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str]] = []

    def data_request(
        _self: ProfileEvidenceRetriever,
        path: str,
        *,
        method: str,
        payload: dict[str, Any] | None = None,
        review_run_id: str | None = None,
    ) -> dict[str, Any]:
        assert review_run_id == "run-profile-v3"
        calls.append((method, path))
        if path == ProfileEvidenceRetriever.readiness_path:
            assert method == "GET"
            assert payload is None
            return {"status": "ready"}
        assert path == ProfileEvidenceRetriever.retrieve_path
        assert method == "POST"
        assert payload is not None
        return {
            "schema_version": "3.0",
            "request_id": payload["request_id"],
            "status": "ok",
            "knowledge_snapshot_id": "dev-fixture",
            "results": [],
        }

    monkeypatch.setattr(ProfileEvidenceRetriever, "_request", data_request)
    runtime = ProfileRuntime(_settings())
    result = runtime.review_v3_json(
        json.dumps(
            {
                "case_id": "case-profile-v3",
                "documents": [
                    {
                        "document_id": "doc-profile-v3",
                        "filename": "invoice.txt",
                        "content_text": "COMMERCIAL INVOICE\nInvoice No: INV-003",
                    }
                ],
                "review_run_id": "run-profile-v3",
                "review_as_of": "2026-08-05",
                "trade_type": "import",
                "jurisdiction": "KR",
            }
        ).encode()
    )
    assert result.schema_version == "3.2"
    assert result.status == "needs_review"
    assert result.requires_human_review is True
    assert result.engine is not None and result.engine.kind == "deterministic"
    assert any(action.action_code == "human_review_required" for action in result.required_actions)
    assert [stage.stage for stage in result.workflow.stages] == [
        "classify_documents",
        "extract_facts",
        "normalize_facts",
        "match_documents",
        "validate_documents",
        "reextract_documents",
        "retrieve_evidence",
        "generate_result",
    ]
    assert result.document_analyses[0].document_id == "doc-profile-v3"
    assert result.workflow.planner.cycle_count <= 2
    assert len(result.workflow.planner.actions) <= 6
    assert {section.section for section in result.report.sections} == {
        "conclusion",
        "execution_process",
        "document_comparison",
        "issues_and_causes",
        "legal_evidence",
        "impact",
        "next_actions",
        "limitations",
    }
    # No HS-qualified commodity exists in this deliberately incomplete
    # invoice.  Sending an empty or no-HS fallback to Data would manufacture a
    # partial Data status, so the bounded workflow records an input-side skip.
    assert calls == []
    retrieval = next(
        stage for stage in result.workflow.stages if stage.stage == "retrieve_evidence"
    )
    assert retrieval.status == "skipped"
    legal_evidence = next(
        section for section in result.report.sections if section.section == "legal_evidence"
    )
    assert "HS 코드가 확인된 품목이 없어" in legal_evidence.claims[0].text
    assert "Data" not in legal_evidence.claims[0].text


def test_reextraction_failure_is_exposed_as_partial_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Analyzer:
        model_id = "test-reextract-failure"

        def __init__(self) -> None:
            self.calls = 0
            self._fallback = _DeterministicDocumentAnalyzer()

        def analyze(self, document: dict[str, Any], **_kwargs: Any) -> Any:
            self.calls += 1
            if self.calls > 1:
                raise RuntimeError("controlled reextract failure")
            return self._fallback.analyze(document)

    monkeypatch.setattr(
        ProfileEvidenceRetriever,
        "retrieve",
        lambda _self, request: {
            "schema_version": "3.0",
            "request_id": request["request_id"],
            "status": "ok",
            "knowledge_snapshot_id": "dev-fixture",
            "results": [],
        },
    )
    runtime = ProfileRuntime(_settings())
    analyzer = Analyzer()
    runtime._analyzer = analyzer  # type: ignore[assignment]

    result = runtime.review_v3_json(
        json.dumps(
            {
                "case_id": "case-reextract-failure",
                "documents": [
                    {
                        "document_id": "doc-reextract-failure",
                        "filename": "invoice.txt",
                        "content_text": "COMMERCIAL INVOICE\nInvoice No: INV-RETRY",
                    }
                ],
                "review_run_id": "run-reextract-failure",
                "review_as_of": "2026-08-05",
                "trade_type": "import",
                "jurisdiction": "KR",
            }
        ).encode()
    )

    assert analyzer.calls == 2
    assert result.workflow.status == "partial"
    assert result.execution_status == "partial"


def test_internal_api_requires_matching_mode_header() -> None:
    class Runtime:
        def review_v3_json(self, _payload: bytes) -> object:
            raise AssertionError("review is not part of this test")

        def close(self) -> None:
            return None

    app = create_app(
        _settings(),
        runtime_factory=lambda _settings: Runtime(),  # type: ignore[arg-type]
        readiness_probe_factory=lambda _settings: lambda: object(),
    )
    with TestClient(app, raise_server_exceptions=False) as client:
        missing = client.get(
            "/internal/agent/v2/metadata", headers={"Authorization": f"Bearer {_TOKEN}"}
        )
        accepted = client.get(
            "/internal/agent/v2/metadata",
            headers={
                "Authorization": f"Bearer {_TOKEN}",
                SERVICE_MODE_HEADER: "local-development",
            },
        )
    assert missing.status_code == 403
    assert missing.json()["error"]["code"] == "service_mode_mismatch"
    assert accepted.status_code == 200
    assert accepted.json()["human_review_only"] is True


def test_http_knowledge_rejects_document_material_before_runtime_execution() -> None:
    """The public Agent route must not enqueue a pasted Case/document payload."""

    calls = 0

    class Runtime:
        def knowledge_answer_json(self, _payload: bytes) -> object:
            nonlocal calls
            calls += 1
            raise AssertionError("knowledge runtime must not receive rejected input")

        def close(self) -> None:
            return None

    app = create_app(
        _settings(),
        runtime_factory=lambda _settings: Runtime(),  # type: ignore[arg-type]
        readiness_probe_factory=lambda _settings: lambda: object(),
    )
    unsafe_question = "상업송장\n송장번호: INV-001"
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/internal/agent/v2/knowledge/answers",
            headers={
                "Authorization": f"Bearer {_TOKEN}",
                SERVICE_MODE_HEADER: "local-development",
            },
            json={
                "schema_version": "1.0",
                "conversation_id": "knowledge-boundary",
                "question": unsafe_question,
                "trade_type": "both",
                "as_of": "2026-08-05",
                "recent_turns": [],
            },
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "knowledge_input_boundary"
    assert "일반적인 통관 지식 질문" in response.json()["error"]["message"]
    assert unsafe_question not in response.text
    assert calls == 0


def test_http_knowledge_rejects_document_material_in_recent_turns_before_execution() -> None:
    """History is an input boundary too, not a trusted provider context."""

    calls = 0

    class Runtime:
        def knowledge_answer_json(self, _payload: bytes) -> object:
            nonlocal calls
            calls += 1
            raise AssertionError("knowledge runtime must not receive rejected history")

        def close(self) -> None:
            return None

    app = create_app(
        _settings(),
        runtime_factory=lambda _settings: Runtime(),  # type: ignore[arg-type]
        readiness_probe_factory=lambda _settings: lambda: object(),
    )
    unsafe_history = "원산지증명서\nHS 코드: 0303540000"
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/internal/agent/v2/knowledge/answers",
            headers={
                "Authorization": f"Bearer {_TOKEN}",
                SERVICE_MODE_HEADER: "local-development",
            },
            json={
                "schema_version": "1.0",
                "conversation_id": "knowledge-history-boundary",
                "question": "HS 품목분류의 기본 절차를 알려 주세요.",
                "trade_type": "both",
                "as_of": "2026-08-05",
                "recent_turns": [{"role": "user", "content": unsafe_history}],
            },
        )

    assert response.status_code == 422
    assert response.json()["error"]["code"] == "knowledge_input_boundary"
    assert unsafe_history not in response.text
    assert calls == 0


def test_http_review_keeps_human_only_v3_result(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        ProfileEvidenceRetriever,
        "retrieve",
        lambda _self, request: {
            "schema_version": "3.0",
            "request_id": request["request_id"],
            "status": "ok",
            "knowledge_snapshot_id": "dev-fixture",
            "results": [],
        },
    )
    app = create_app(_settings())
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/internal/agent/v2/reviews",
            headers={
                "Authorization": f"Bearer {_TOKEN}",
                SERVICE_MODE_HEADER: "local-development",
            },
            json={
                "case_id": "case-profile-http",
                "documents": [
                    {
                        "document_id": "doc-profile-http",
                        "filename": "invoice.txt",
                        "content_text": "COMMERCIAL INVOICE\nInvoice No: INV-004",
                    }
                ],
                "review_run_id": "run-profile-http",
                "review_as_of": "2026-08-05",
                "trade_type": "import",
                "jurisdiction": "KR",
            },
        )
    assert response.status_code == 200
    assert response.json()["status"] == "needs_review"
    assert response.json()["requires_human_review"] is True


def test_http_review_accepts_bounded_child_correction_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A schema-valid child run must pass HTTP admission before the runtime."""

    monkeypatch.setattr(
        ProfileEvidenceRetriever,
        "retrieve",
        lambda _self, request: {
            "schema_version": "3.0",
            "request_id": request["request_id"],
            "status": "ok",
            "knowledge_snapshot_id": "dev-fixture",
            "results": [],
        },
    )
    app = create_app(_settings())
    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post(
            "/internal/agent/v2/reviews",
            headers={
                "Authorization": f"Bearer {_TOKEN}",
                SERVICE_MODE_HEADER: "local-development",
            },
            json={
                "case_id": "case-profile-child",
                "documents": [
                    {
                        "document_id": "invoice",
                        "filename": "invoice.txt",
                        "content_text": "COMMERCIAL INVOICE\nInvoice No: INV-005",
                    }
                ],
                "review_run_id": "run-profile-child",
                "review_as_of": "2026-08-05",
                "trade_type": "import",
                "jurisdiction": "KR",
                "parent_review_run_id": "run-profile-parent",
                "applied_correction_proposal_ids": ["proposal-origin"],
                "human_corrections": [
                    {
                        "correction_proposal_id": "proposal-origin",
                        "document_id": "invoice",
                        "field_name": "origin_country",
                        "new_value": "KR",
                        "reason": "승인된 원산지 보완값입니다.",
                        "source_kind": "human_provided",
                    }
                ],
            },
        )

    assert response.status_code == 200
    actions = response.json()["workflow"]["planner"]["actions"]
    correction = next(action for action in actions if action["action"] == "apply_human_corrections")
    assert correction["correction_proposal_ids"] == ["proposal-origin"]
