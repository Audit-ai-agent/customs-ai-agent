import pytest

from customs_ai_agent.adapters.llm import (
    DocumentAnalysis,
    LLMConfigurationError,
    LLMProviderError,
    ModelCallMetadata,
)
from customs_ai_agent.adapters.retrieval import MockRegulationRetriever
from customs_ai_agent.runner import rerun_review, run_review, to_review_run_result
from customs_ai_agent.schemas.llm import ClaudeDocumentCandidate
from customs_ai_agent.schemas.models import ReviewRunRequest, UserCorrection


def _metadata(document_id):
    return ModelCallMetadata(
        provider="anthropic",
        model_id="claude-sonnet-5",
        prompt_id="customs_document_analysis",
        prompt_version="1.1.0",
        request_id=f"req_{document_id}",
        message_id=f"msg_{document_id}",
        stop_reason="end_turn",
        input_tokens=100,
        output_tokens=50,
        cache_creation_input_tokens=0,
        cache_read_input_tokens=0,
        latency_ms=10,
    )


def _field(name, value, source_quote=None):
    return {
        "name": name,
        "value": value,
        "source_quote": source_quote or value,
    }


def _candidate_for(document_id):
    if document_id == "doc_bl":
        return {
            "schema_version": "1.0",
            "document_type": "bill_of_lading",
            "classification_source_quote": "BILL OF LADING",
            "fields": [
                _field("bl_no", "BL-2026-001"),
                _field("shipper", "Blue Ocean Export Co."),
                _field("consignee", "Korea Foods Import Inc."),
                _field("container_numbers", "MSKU1234567"),
                _field("packages", "100 CTNS"),
                _field("gross_weight_kg", "1200 KG"),
            ],
            "line_items": [],
        }
    if document_id == "doc_inv":
        line = (
            "ITEM: Stainless bolts | HS: 7318.15 | QTY: 1200 KG | "
            "UNIT PRICE: 10 | AMOUNT: 12000 | ORIGIN: NO"
        )
        return {
            "schema_version": "1.0",
            "document_type": "invoice",
            "classification_source_quote": "COMMERCIAL INVOICE",
            "fields": [
                _field("invoice_no", "INV-2026-001"),
                _field("invoice_date", "2026.06.01"),
                _field("seller", "Blue Ocean Export Co."),
                _field("buyer", "Korea Foods Import Inc."),
                _field("incoterms", "CIF"),
                _field("currency", "USD"),
                _field("total_amount", "12000", "TOTAL AMOUNT: 12000"),
            ],
            "line_items": [
                {
                    "source_quote": line,
                    "fields": [
                        _field("description", "Stainless bolts"),
                        _field("hs_code", "7318.15"),
                        _field("quantity", "1200 KG"),
                        _field("unit_price", "10"),
                        _field("amount", "12000"),
                        _field("origin_country", "NO"),
                    ],
                }
            ],
        }
    if document_id == "doc_pl":
        line = (
            "ITEM: Stainless bolts | QTY: 1200 KG | NET WEIGHT: 1100 KG | "
            "GROSS WEIGHT: 1200 KG | PACKAGE COUNT: 100"
        )
        return {
            "schema_version": "1.0",
            "document_type": "packing_list",
            "classification_source_quote": "PACKING LIST",
            "fields": [
                _field("packing_no", "PL-2026-001"),
                _field("seller", "Blue Ocean Export Co."),
                _field("buyer", "Korea Foods Import Inc."),
                _field("container_numbers", "MSKU1234567"),
                _field("packages", "100 CTNS"),
            ],
            "line_items": [
                {
                    "source_quote": line,
                    "fields": [
                        _field("description", "Stainless bolts"),
                        _field("quantity", "1200 KG"),
                        _field("net_weight_kg", "1100 KG"),
                        _field("gross_weight_kg", "1200 KG"),
                        _field("package_count", "100"),
                    ],
                }
            ],
        }
    if document_id == "doc_co":
        return {
            "schema_version": "1.0",
            "document_type": "certificate_of_origin",
            "classification_source_quote": "CERTIFICATE OF ORIGIN",
            "fields": [
                _field("certificate_no", "CO-2026-001"),
                _field("exporter", "Blue Ocean Export Co."),
                _field("importer", "Korea Foods Import Inc."),
                _field("origin_country", "Norway"),
                _field("hs_codes", "7318.15"),
                _field("goods_description", "Stainless bolts"),
                _field("issue_date", "Jun 1 2026"),
            ],
            "line_items": [],
        }
    raise AssertionError(document_id)


class StaticAnalyzer:
    def __init__(self, fail_document_id=None):
        self.calls = []
        self.fail_document_id = fail_document_id

    def analyze(self, document):
        document_id = document["document_id"]
        self.calls.append(document_id)
        if document_id == self.fail_document_id:
            raise LLMProviderError(
                "provider_unavailable",
                "Claude was unavailable after bounded SDK retries.",
                retryable=True,
            )
        return DocumentAnalysis(
            candidate=ClaudeDocumentCandidate.model_validate(_candidate_for(document_id)),
            metadata=_metadata(document_id),
        )


def test_llm_workflow_calls_analyzer_once_per_document_and_preserves_policy():
    analyzer = StaticAnalyzer()
    state = run_review(
        ReviewRunRequest(scenario_id="happy_path", llm_mode="llm"),
        retriever=MockRegulationRetriever(),
        analyzer=analyzer,
    )

    assert analyzer.calls == ["doc_bl", "doc_inv", "doc_pl", "doc_co"]
    assert state["model_errors"] == []
    assert len(state["model_calls"]) == 4
    assert state["final_status"] == "needs_review"
    assert state["requires_human_review"] is True
    assert state["validation_issues"] == []
    evidence = state["extracted_documents"]["doc_inv"]["fields"]["invoice_no"]["evidence"]
    assert evidence["bbox"] is None
    assert evidence["raw_text"] == "INV-2026-001"
    assert (
        state["extracted_documents"]["doc_inv"]["fields"]["invoice_no"][
            "confidence_basis"
        ]
        == "source_quote_grounded_uncalibrated"
    )
    assert all(
        "api_key" not in str(call).casefold() and "content_text" not in call
        for call in state["model_calls"]
    )
    public_result = to_review_run_result(state)
    assert len(public_result.model_calls) == 4
    assert public_result.model_errors == []
    reasons = state["report_json"]["summary"]["outcome_reasons"]
    assert "llm_extraction_uncalibrated" in reasons
    assert "extraction_confidence_below_threshold" in reasons


def test_llm_mode_rejects_implicit_mock_rag_before_analyzer_call():
    analyzer = StaticAnalyzer()

    with pytest.raises(LLMConfigurationError) as exc_info:
        run_review(
            ReviewRunRequest(scenario_id="happy_path", llm_mode="llm"),
            analyzer=analyzer,
        )

    assert exc_info.value.code == "explicit_retriever_required"
    assert analyzer.calls == []


def test_provider_failure_never_falls_back_to_rule_extraction():
    analyzer = StaticAnalyzer(fail_document_id="doc_inv")
    state = run_review(
        ReviewRunRequest(scenario_id="happy_path", llm_mode="llm"),
        retriever=MockRegulationRetriever(),
        analyzer=analyzer,
    )

    assert state["classified_documents"]["doc_inv"]["document_type"] == "unknown"
    assert state["extracted_documents"]["doc_inv"]["fields"] == {}
    assert analyzer.calls == ["doc_bl", "doc_inv"]
    assert [
        (item["document_id"], item["code"])
        for item in state["model_errors"]
    ] == [
        ("doc_inv", "provider_unavailable"),
        ("doc_pl", "provider_not_attempted_after_failure"),
        ("doc_co", "provider_not_attempted_after_failure"),
    ]
    assert all(item["provider"] == "anthropic" for item in state["model_errors"])
    assert all(
        item["prompt_version"] == "1.1.0" for item in state["model_errors"]
    )
    assert state["retrieval_status"] == "failed"
    assert state["final_status"] == "needs_review"


def test_invalid_injected_analysis_fails_closed_during_mapping():
    class AmbiguousQuoteAnalyzer(StaticAnalyzer):
        def analyze(self, document):
            analysis = super().analyze(document)
            if document["document_id"] != "doc_inv":
                return analysis
            payload = analysis.candidate.model_dump(mode="json")
            for field in payload["fields"]:
                if field["name"] == "total_amount":
                    field["source_quote"] = "12000"
            return DocumentAnalysis(
                candidate=ClaudeDocumentCandidate.model_validate(payload),
                metadata=analysis.metadata,
            )

    state = run_review(
        ReviewRunRequest(scenario_id="happy_path", llm_mode="llm"),
        retriever=MockRegulationRetriever(),
        analyzer=AmbiguousQuoteAnalyzer(),
    )

    assert state["extracted_documents"]["doc_inv"]["document_type"] == "unknown"
    assert any(
        item["document_id"] == "doc_inv"
        and item["code"] == "source_quote_ambiguous"
        for item in state["model_errors"]
    )
    assert state["final_status"] == "needs_review"


def test_incomplete_but_grounded_candidate_has_deterministic_blocker():
    class IncompleteAnalyzer(StaticAnalyzer):
        def analyze(self, document):
            analysis = super().analyze(document)
            if document["document_id"] != "doc_bl":
                return analysis
            payload = analysis.candidate.model_dump(mode="json")
            payload["fields"] = [
                field for field in payload["fields"] if field["name"] == "bl_no"
            ]
            return DocumentAnalysis(
                candidate=ClaudeDocumentCandidate.model_validate(payload),
                metadata=analysis.metadata,
            )

    state = run_review(
        ReviewRunRequest(scenario_id="happy_path", llm_mode="llm"),
        retriever=MockRegulationRetriever(),
        analyzer=IncompleteAnalyzer(),
    )
    reasons = state["report_json"]["summary"]["outcome_reasons"]

    assert state["final_status"] == "needs_review"
    assert "doc_bl:required_field_missing:shipper" in reasons
    assert "doc_bl:required_field_missing:gross_weight_kg" in reasons


def test_uncalibrated_llm_critical_issue_cannot_autonomously_block_case():
    class SemanticallyWrongAnalyzer(StaticAnalyzer):
        def analyze(self, document):
            analysis = super().analyze(document)
            if document["document_id"] != "doc_co":
                return analysis
            payload = analysis.candidate.model_dump(mode="json")
            for field in payload["fields"]:
                if field["name"] == "origin_country":
                    # "Korea" is a real exact source span, but it belongs to
                    # the importer name rather than the origin-country field.
                    field["value"] = "Korea"
                    field["source_quote"] = "Korea"
            return DocumentAnalysis(
                candidate=ClaudeDocumentCandidate.model_validate(payload),
                metadata=analysis.metadata,
            )

    state = run_review(
        ReviewRunRequest(scenario_id="happy_path", llm_mode="llm"),
        retriever=MockRegulationRetriever(),
        analyzer=SemanticallyWrongAnalyzer(),
    )
    origin_issue = next(
        issue
        for issue in state["validation_issues"]
        if issue["issue_code"] == "ORIGIN_MISMATCH"
    )

    assert origin_issue["severity"] == "critical"
    assert origin_issue["confidence"] <= 0.85
    assert origin_issue["confidence_basis"] == "source_field_minimum"
    report_origin_issue = next(
        issue
        for issue in state["report_json"]["issues"]
        if issue["issue_code"] == "ORIGIN_MISMATCH"
    )
    assert report_origin_issue["confidence"] == origin_issue["confidence"]
    assert report_origin_issue["confidence_basis"] == "source_field_minimum"
    assert state["final_status"] == "needs_review"
    assert state["requires_human_review"] is True
    assert "critical_finding" in state["report_json"]["summary"]["outcome_reasons"]


def test_deterministic_self_correction_preserves_observed_source_quote():
    state = run_review(
        ReviewRunRequest(scenario_id="low_confidence_self_correction")
    )
    field = state["extracted_documents"]["doc_bl"]["fields"]["gross_weight_kg"]

    assert field["value"] == "1200 KG"
    assert field["value_source"] == "deterministic_ocr_correction"
    assert field["evidence"]["raw_text"] == "12OO KG"


def test_human_correction_preserves_original_source_quote():
    state = run_review(ReviewRunRequest(scenario_id="invoice_total_mismatch"))
    before = state["extracted_documents"]["doc_inv"]["fields"]["total_amount"][
        "evidence"
    ]["raw_text"]

    corrected = rerun_review(
        state,
        [
            UserCorrection(
                document_id="doc_inv",
                field_path="fields.total_amount",
                value="12000",
            )
        ],
    )
    field = corrected["extracted_documents"]["doc_inv"]["fields"]["total_amount"]

    assert field["value"] == "12000"
    assert field["value_source"] == "human_correction"
    assert field["evidence"]["raw_text"] == before
