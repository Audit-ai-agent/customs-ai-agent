from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import pytest

from customs_ai_agent.adapters.llm import DocumentAnalysis
from customs_ai_agent.schemas.llm import (
    ClaudeDocumentCandidate,
    ClaudeFieldCandidate,
    ClaudeLineItemCandidate,
)
from customs_ai_agent.schemas.v3 import ServiceReviewRunRequestV3
from customs_ai_agent.service.graph import (
    EvidenceStageResult,
    ReviewGraphConfig,
    derive_statuses,
    run_review_graph,
)
from customs_ai_agent.service.workflow import BoundedWorkflowOutput, run_bounded_workflow


_INVOICE_TEXT = "\n".join(
    (
        "COMMERCIAL INVOICE",
        "Invoice No: INV-001",
        "Total Amount: USD 20",
        "Country of Origin: KR",
        "Description: Widget | HS Code: 123456 | Quantity: 2 PCS | Unit: PCS | "
        "Unit Price: USD 10 | Amount: USD 20",
    )
)
_PACKING_TEXT = "\n".join(
    (
        "PACKING LIST",
        "Packing No: PL-001",
        "B/L No: BL-001",
        "Container No: ABCD1234567",
        "Total Gross Weight: 100 KG",
        "Description: Widget | HS Code: 123456 | Quantity: 3 PCS | Unit: PCS | Gross Weight: 100 KG",
    )
)


def _field(name: str, value: str, quote: str) -> ClaudeFieldCandidate:
    return ClaudeFieldCandidate(name=name, value=value, source_quote=quote)


def _candidates(*, packing_quantity: str = "3 PCS") -> dict[str, list[ClaudeDocumentCandidate]]:
    invoice_line = _INVOICE_TEXT.splitlines()[-1]
    packing_line = _PACKING_TEXT.splitlines()[-1]
    invoice = ClaudeDocumentCandidate(
        document_type="invoice",
        classification_source_quote="COMMERCIAL INVOICE",
        fields=[
            _field("invoice_no", "INV-001", "Invoice No: INV-001"),
            _field("total_amount", "USD 20", "Total Amount: USD 20"),
            _field("origin_country", "KR", "Country of Origin: KR"),
            _field("hs_code", "123456", "HS Code: 123456"),
        ],
        line_items=[
            ClaudeLineItemCandidate(
                source_quote=invoice_line,
                fields=[
                    _field("description", "Widget", invoice_line),
                    _field("hs_code", "123456", invoice_line),
                    _field("quantity", "2 PCS", invoice_line),
                    _field("unit", "PCS", invoice_line),
                    _field("unit_price", "USD 10", invoice_line),
                    _field("amount", "USD 20", invoice_line),
                ],
            )
        ],
    )
    packing = ClaudeDocumentCandidate(
        document_type="packing_list",
        classification_source_quote="PACKING LIST",
        fields=[
            _field("packing_no", "PL-001", "Packing No: PL-001"),
            _field("bl_no", "BL-001", "B/L No: BL-001"),
            _field("container_numbers", "ABCD1234567", "Container No: ABCD1234567"),
            _field("total_gross_weight_kg", "100 KG", "Total Gross Weight: 100 KG"),
        ],
        line_items=[
            ClaudeLineItemCandidate(
                source_quote=packing_line,
                fields=[
                    _field("description", "Widget", packing_line),
                    _field("hs_code", "123456", packing_line),
                    _field("quantity", packing_quantity, packing_line),
                    _field("unit", "PCS", packing_line),
                ],
            )
        ],
    )
    return {"invoice": [invoice], "packing": [packing]}


def _request() -> ServiceReviewRunRequestV3:
    return ServiceReviewRunRequestV3.model_validate(
        {
            "case_id": "case-graph",
            "review_run_id": "run-graph",
            "review_as_of": "2026-08-05",
            "trade_type": "import",
            "jurisdiction": "KR",
            "documents": [
                {
                    "document_id": "invoice",
                    "filename": "invoice.txt",
                    "content_text": _INVOICE_TEXT,
                    "extraction": {"source": "raster_ocr", "layout_summary": None},
                },
                {
                    "document_id": "packing",
                    "filename": "packing.txt",
                    "content_text": _PACKING_TEXT,
                },
            ],
        }
    )


@dataclass
class _SequenceAnalyzer:
    candidates: dict[str, list[ClaudeDocumentCandidate]]
    model_id: str = "test-extractor"

    def __post_init__(self) -> None:
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    def analyze(
        self,
        document: Mapping[str, Any],
        *,
        transcript: object | None = None,
        focus_fields: tuple[str, ...] = (),
    ) -> DocumentAnalysis:
        del transcript
        document_id = str(document["document_id"])
        self.calls.append((document_id, focus_fields))
        values = self.candidates[document_id]
        index = min(sum(call[0] == document_id for call in self.calls) - 1, len(values) - 1)
        return DocumentAnalysis(candidate=values[index], model_id=self.model_id)


def _evidence_stage(status: str = "ok", *, skipped: bool = False) -> Any:
    def stage(
        request: ServiceReviewRunRequestV3, workflow_output: BoundedWorkflowOutput
    ) -> EvidenceStageResult:
        del request, workflow_output
        return EvidenceStageResult(
            data_status=status,  # type: ignore[arg-type]
            skipped=skipped,
            evidence_records=[],
            evidence_findings=[],
            public_evidence=[],
            temporal_context=None,
            temporal_coverage=None,
        )

    return stage


class _Responder:
    def __init__(self, answers: dict[str, dict[str, Any]]) -> None:
        self.answers = answers
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def respond(self, payload: dict[str, Any], *, purpose: str, **_kwargs: Any) -> dict[str, Any]:
        self.calls.append((purpose, payload))
        return self.answers.get(purpose, {})


def test_wrapper_graph_stops_after_cross_validation_and_records_its_trace() -> None:
    analyzer = _SequenceAnalyzer(_candidates())
    output = run_bounded_workflow(_request(), analyzer)

    trace = output.workflow.graph_trace
    assert trace is not None
    assert trace.framework == "langgraph"
    assert trace.iteration_count == 1
    assert trace.loop_reason is None
    assert [(run.node, run.iteration) for run in trace.node_runs] == [
        ("extraction_agent", 1),
        ("cross_validation_agent", 1),
    ]
    assert [entry.action for entry in output.workflow.planner.actions] == [
        "analyze_documents",
        "compare_documents",
    ]
    assert output.document_analyses[0].extraction_source == "raster_ocr"
    assert output.document_analyses[1].extraction_source is None
    # A deterministic extractor never triggers the mismatch-driven audit pass.
    assert analyzer.calls == [("invoice", ()), ("packing", ())]
    assert any(comparison.status == "mismatch" for comparison in output.comparisons)


def test_full_graph_reaches_confidence_verdict_and_report() -> None:
    analyzer = _SequenceAnalyzer(_candidates())
    result = run_review_graph(
        _request(),
        ReviewGraphConfig(
            analyzer=analyzer, evidence_stage=_evidence_stage(), confidence_threshold=0.8
        ),
    )

    assert [run.node for run in result.graph_trace.node_runs] == [
        "extraction_agent",
        "cross_validation_agent",
        "evidence_retrieval",
        "expert_review_agent",
        "report_generation",
    ]
    assert [entry.action for entry in result.workflow_output.workflow.planner.actions] == [
        "analyze_documents",
        "compare_documents",
        "retrieve_evidence",
        "generate_report",
    ]
    assert result.status == "needs_review"
    assert result.execution_status == "succeeded"
    assert result.assessment == "issues_detected"
    assert result.confidence is not None and result.confidence.method == "deterministic_signals"
    assert result.eligibility_verdict is not None
    assert result.eligibility_verdict.verdict == "ineligible"
    assert result.eligibility_verdict.advisory is True
    assert result.eligibility_verdict.source == "rule"
    assert all(finding.confidence is not None for finding in result.findings)
    assert all(
        finding.classification is not None and finding.classification.source == "rule"
        for finding in result.findings
    )
    mismatch = next(finding for finding in result.findings if finding.code.endswith("_mismatch"))
    assert mismatch.classification is not None
    assert mismatch.classification.error_category == "value_mismatch"
    assert result.report_sections is not None and len(result.report_sections) == 8
    assert result.actions is not None and result.actions[-1].action_code == "human_review_required"


def test_model_extractor_gets_one_audit_pass_on_a_mismatch() -> None:
    # The second candidate for packing agrees with the invoice, so the re-read
    # resolves the mismatch and the comparison is recorded once, in cycle 2.
    candidates = _candidates()
    candidates["packing"].append(_candidates(packing_quantity="2 PCS")["packing"][0])
    analyzer = _SequenceAnalyzer(candidates)
    result = run_review_graph(
        _request(),
        ReviewGraphConfig(
            analyzer=analyzer, mode="functional-uat", evidence_stage=_evidence_stage()
        ),
    )

    trace = result.graph_trace
    assert trace.iteration_count == 2
    assert trace.loop_reason == "mismatch_reread"
    assert [(run.node, run.iteration) for run in trace.node_runs][:4] == [
        ("extraction_agent", 1),
        ("cross_validation_agent", 1),
        ("extraction_agent", 2),
        ("cross_validation_agent", 2),
    ]
    assert trace.node_runs[1].status == "needs_review"
    assert "재판독을 요청" in trace.node_runs[1].summary
    focused = [call for call in analyzer.calls if call[1]]
    assert {document_id for document_id, _ in focused} == {"invoice", "packing"}
    assert all("quantity" in fields for _, fields in focused)
    assert len(focused) == 2
    planner = result.workflow_output.workflow.planner
    assert planner.cycle_count == 2
    assert [(entry.cycle, entry.action) for entry in planner.actions] == [
        (1, "analyze_documents"),
        (2, "reextract_documents"),
        (2, "compare_documents"),
        (2, "retrieve_evidence"),
        (2, "generate_report"),
    ]
    assert [entry.document_id for entry in result.workflow_output.workflow.reextractions] == [
        "invoice",
        "packing",
    ]
    assert not any(
        comparison.status == "mismatch" for comparison in result.workflow_output.comparisons
    )
    assert result.eligibility_verdict is not None
    assert result.eligibility_verdict.verdict != "ineligible"


def test_llm_opinions_are_validated_and_never_relax_the_rules() -> None:
    analyzer = _SequenceAnalyzer(_candidates())
    responder = _Responder(
        {
            "finding_classification": {
                "classifications": [
                    {
                        "finding_id": "will-be-filled",
                        "error_category": "unit_or_format",
                        "explanation": "x",
                    }
                ],
                "reread_requests": [{"document_id": "packing", "fields": ["not_a_field"]}],
            },
            "expert_review": {"verdict": "eligible", "reasons": []},
        }
    )
    request = _request()
    first = run_review_graph(
        request,
        ReviewGraphConfig(
            analyzer=analyzer,
            mode="functional-uat",
            evidence_stage=_evidence_stage(),
            responder=responder,
        ),
    )
    mismatch = next(finding for finding in first.findings if finding.code.endswith("_mismatch"))
    # Unknown finding ids and disallowed re-read fields are ignored; the rule
    # label stays, and the mismatch itself still triggers the audit pass.
    assert mismatch.classification is not None and mismatch.classification.source == "rule"
    assert first.graph_trace.loop_reason == "mismatch_reread"
    assert first.eligibility_verdict is not None
    assert first.eligibility_verdict.verdict == "ineligible"
    assert first.eligibility_verdict.source == "rule"
    purposes = [purpose for purpose, _ in responder.calls]
    assert purposes.count("finding_classification") == 2
    assert purposes.count("expert_review") == 1

    responder = _Responder(
        {
            "finding_classification": {
                "classifications": [
                    {
                        "finding_id": mismatch.finding_id,
                        "error_category": "unit_or_format",
                        "explanation": "수량 단위 표기가 문서마다 다릅니다.",
                    }
                ]
            },
            "expert_review": {
                "verdict": "ineligible",
                "reasons": [
                    {
                        "code": "quantity_conflict",
                        "message": "수량 불일치가 해소되지 않았습니다.",
                        "finding_ids": [mismatch.finding_id],
                    }
                ],
            },
        }
    )
    second = run_review_graph(
        request,
        ReviewGraphConfig(
            analyzer=_SequenceAnalyzer(_candidates()),
            mode="functional-uat",
            evidence_stage=_evidence_stage(),
            responder=responder,
        ),
    )
    labelled = next(
        finding for finding in second.findings if finding.finding_id == mismatch.finding_id
    )
    assert labelled.classification is not None
    assert labelled.classification.source == "llm"
    assert labelled.classification.error_category == "unit_or_format"
    assert second.eligibility_verdict is not None
    assert second.eligibility_verdict.source == "rule+llm"
    assert second.eligibility_verdict.reasons[-1].code == "llm_quantity_conflict"


def test_evidence_failure_blocks_and_skip_is_indeterminate() -> None:
    failed = run_review_graph(
        _request(),
        ReviewGraphConfig(
            analyzer=_SequenceAnalyzer(_candidates()), evidence_stage=_evidence_stage("failed")
        ),
    )
    assert failed.status == "blocked"
    assert failed.execution_status == "failed"
    assert failed.workflow_output.workflow.status == "failed"
    assert failed.graph_trace.node_runs[2].status == "failed"

    skipped = run_review_graph(
        _request(),
        ReviewGraphConfig(
            analyzer=_SequenceAnalyzer(_candidates()), evidence_stage=_evidence_stage(skipped=True)
        ),
    )
    assert skipped.graph_trace.node_runs[2].status == "skipped"
    assert skipped.confidence is not None and skipped.confidence.components.evidence == 0.7
    assert skipped.eligibility_verdict is not None
    assert any(
        reason.code == "evidence_not_requested" for reason in skipped.eligibility_verdict.reasons
    ) or (skipped.eligibility_verdict.verdict == "ineligible")


def test_analyzer_errors_propagate_unwrapped() -> None:
    class Broken:
        model_id = "broken"

        def analyze(self, *_args: Any, **_kwargs: Any) -> DocumentAnalysis:
            raise ValueError("extractor exploded")

    with pytest.raises(ValueError, match="extractor exploded"):
        run_review_graph(_request(), ReviewGraphConfig(analyzer=Broken()))


def test_derive_statuses_matches_the_profile_rules() -> None:
    assert derive_statuses(data_status="ok", workflow_status="completed", has_findings=False) == (
        "needs_review",
        "succeeded",
        "no_issue_detected",
    )
    assert derive_statuses(
        data_status="partial", workflow_status="completed", has_findings=True
    ) == (
        "needs_review",
        "partial",
        "issues_detected",
    )
    assert derive_statuses(
        data_status="failed", workflow_status="completed", has_findings=False
    ) == (
        "blocked",
        "failed",
        "indeterminate",
    )
