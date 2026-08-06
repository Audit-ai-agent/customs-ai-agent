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
from customs_ai_agent.service.report import (
    build_grounded_report_sections,
    validate_grounded_report_sections,
)
from customs_ai_agent.service.workflow import run_bounded_workflow


def _field(name: str, value: str, quote: str) -> ClaudeFieldCandidate:
    return ClaudeFieldCandidate(name=name, value=value, source_quote=quote)


def _line(quote: str, *fields: tuple[str, str]) -> ClaudeLineItemCandidate:
    return ClaudeLineItemCandidate(
        source_quote=quote,
        fields=[_field(name, value, quote) for name, value in fields],
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


def _scenario(
    *,
    invoice_quantity: str = "2 PCS",
    packing_quantity: str = "2 PCS",
    duplicate_packing_item: bool = False,
    invoice_line_amount: str | None = None,
    invoice_total_amount: str | None = None,
    certificate_origin: str = "KR",
    packing_bl_no: str = "BL-001",
    bill_bl_no: str = "BL-001",
    packing_container: str = "ABCD1234567",
    bill_container: str = "ABCD1234567",
    packing_gross_weight: str = "100 KG",
    bill_gross_weight: str = "100 KG",
) -> tuple[ServiceReviewRunRequestV3, dict[str, list[ClaudeDocumentCandidate]]]:
    invoice_amount = invoice_line_amount or (
        "USD 30" if invoice_quantity.startswith("3") else "USD 20"
    )
    total_amount = invoice_total_amount or invoice_amount
    invoice_text = "\n".join(
        (
            "COMMERCIAL INVOICE",
            "Invoice No: INV-001",
            f"Total Amount: {total_amount}",
            "Country of Origin: KR",
            (
                "Description: Widget | HS Code: 123456 | Quantity: "
                f"{invoice_quantity} | Unit: PCS | Unit Price: USD 10 | Amount: {invoice_amount}"
            ),
        )
    )
    packing_item = (
        "Description: Widget | HS Code: 123456 | Quantity: "
        f"{packing_quantity} | Unit: PCS | Gross Weight: 100 KG"
    )
    packing_text_lines = [
        "PACKING LIST",
        "Packing No: PL-001",
        f"B/L No: {packing_bl_no}",
        f"Container No: {packing_container}",
        f"Total Gross Weight: {packing_gross_weight}",
        packing_item,
    ]
    if duplicate_packing_item:
        packing_text_lines.append(packing_item)
    packing_text = "\n".join(packing_text_lines)
    certificate_text = "\n".join(
        (
            "CERTIFICATE OF ORIGIN",
            "Certificate No: COO-001",
            f"Origin Country: {certificate_origin}",
            "HS Code: 123456",
        )
    )
    bill_text = "\n".join(
        (
            "BILL OF LADING",
            f"B/L No: {bill_bl_no}",
            f"Container No: {bill_container}",
            f"Gross Weight: {bill_gross_weight}",
        )
    )
    request = ServiceReviewRunRequestV3.model_validate(
        {
            "case_id": "case-workflow",
            "review_run_id": "run-workflow",
            "review_as_of": "2026-08-05",
            "trade_type": "import",
            "jurisdiction": "KR",
            "documents": [
                {"document_id": "invoice", "filename": "invoice.txt", "content_text": invoice_text},
                {"document_id": "packing", "filename": "packing.txt", "content_text": packing_text},
                {
                    "document_id": "certificate",
                    "filename": "coo.txt",
                    "content_text": certificate_text,
                },
                {"document_id": "bill", "filename": "bill.txt", "content_text": bill_text},
            ],
        }
    )
    invoice_line = invoice_text.splitlines()[-1]
    packing_lines = packing_text.splitlines()
    candidates = {
        "invoice": [
            ClaudeDocumentCandidate(
                document_type="invoice",
                classification_source_quote="COMMERCIAL INVOICE",
                fields=[
                    _field("invoice_no", "INV-001", "Invoice No: INV-001"),
                    _field("total_amount", total_amount, f"Total Amount: {total_amount}"),
                    _field("country_of_origin", "KR", "Country of Origin: KR"),
                ],
                line_items=[
                    _line(
                        invoice_line,
                        ("description", "Widget"),
                        ("hs_code", "123456"),
                        ("quantity", invoice_quantity),
                        ("unit", "PCS"),
                        ("unit_price", "USD 10"),
                        ("amount", invoice_amount),
                    )
                ],
            )
        ],
        "packing": [
            ClaudeDocumentCandidate(
                document_type="packing_list",
                classification_source_quote="PACKING LIST",
                fields=[
                    _field("packing_no", "PL-001", "Packing No: PL-001"),
                    _field("bl_no", packing_bl_no, f"B/L No: {packing_bl_no}"),
                    _field(
                        "container_numbers", packing_container, f"Container No: {packing_container}"
                    ),
                    _field(
                        "total_gross_weight_kg",
                        packing_gross_weight,
                        f"Total Gross Weight: {packing_gross_weight}",
                    ),
                ],
                line_items=[
                    _line(
                        line,
                        ("description", "Widget"),
                        ("hs_code", "123456"),
                        ("quantity", packing_quantity),
                        ("unit", "PCS"),
                        ("gross_weight_kg", "100 KG"),
                    )
                    for line in packing_lines
                    if line.startswith("Description:")
                ],
            )
        ],
        "certificate": [
            ClaudeDocumentCandidate(
                document_type="certificate_of_origin",
                classification_source_quote="CERTIFICATE OF ORIGIN",
                fields=[
                    _field("certificate_no", "COO-001", "Certificate No: COO-001"),
                    _field(
                        "origin_country",
                        certificate_origin,
                        f"Origin Country: {certificate_origin}",
                    ),
                    _field("hs_code", "123456", "HS Code: 123456"),
                ],
            )
        ],
        "bill": [
            ClaudeDocumentCandidate(
                document_type="bill_of_lading",
                classification_source_quote="BILL OF LADING",
                fields=[
                    _field("bl_no", bill_bl_no, f"B/L No: {bill_bl_no}"),
                    _field("container_numbers", bill_container, f"Container No: {bill_container}"),
                    _field(
                        "gross_weight_kg", bill_gross_weight, f"Gross Weight: {bill_gross_weight}"
                    ),
                ],
            )
        ],
    }
    return request, candidates


def test_bounded_workflow_records_grounded_passes_and_links() -> None:
    request, candidates = _scenario()
    output = run_bounded_workflow(request, _SequenceAnalyzer(candidates))

    assert output.workflow.reextractions == []
    assert all(comparison.status == "pass" for comparison in output.comparisons)
    assert {analysis.document_type for analysis in output.document_analyses} == {
        "invoice",
        "packing_list",
        "certificate_of_origin",
        "bill_of_lading",
    }
    invoice_fact = output.document_analyses[0].facts[0]
    assert invoice_fact.source.document_id == "invoice"
    assert invoice_fact.source.quote == "Invoice No: INV-001"
    assert output.findings == []
    assert output.workflow.planner.cycle_count <= 2
    assert len(output.workflow.planner.actions) <= 6
    # The pre-Evidence output reports only actions that really ran.  It does
    # not pretend that a correction, retry, retrieval, or report happened.
    assert [entry.action for entry in output.workflow.planner.actions] == [
        "analyze_documents",
        "compare_documents",
    ]
    assert all(entry.status != "skipped" for entry in output.workflow.planner.actions)
    with_evidence = output.with_evidence_status("ok")
    complete = with_evidence.with_report_generated()
    assert [entry.action for entry in complete.workflow.planner.actions] == [
        "analyze_documents",
        "compare_documents",
        "retrieve_evidence",
        "generate_report",
    ]
    assert complete.workflow.planner.cycle_count == 1


def test_quantity_mismatch_produces_stable_finding_reference() -> None:
    request, candidates = _scenario(invoice_quantity="3 PCS")
    output = run_bounded_workflow(request, _SequenceAnalyzer(candidates))

    comparison = next(
        comparison
        for comparison in output.comparisons
        if comparison.rule_code == "invoice_packing_quantity_unit"
    )
    assert comparison.status == "mismatch"
    finding = next(
        finding for finding in output.findings if finding.comparison_id == comparison.comparison_id
    )
    assert finding.finding_id.startswith("finding-")
    assert finding.document_ids == ["invoice", "packing"]
    assert set(finding.fact_ids) == set(comparison.left.fact_ids + comparison.right.fact_ids)


def test_quantity_comparison_uses_explicit_unit_fact_when_quantity_has_no_suffix() -> None:
    request, candidates = _scenario(invoice_quantity="2", packing_quantity="2")
    output = run_bounded_workflow(request, _SequenceAnalyzer(candidates))

    comparison = next(
        value for value in output.comparisons if value.rule_code == "invoice_packing_quantity_unit"
    )
    assert comparison.status == "pass"
    assert comparison.unit == "PCS"


def test_retrieval_does_not_duplicate_a_document_hs_already_linked_to_a_line_item() -> None:
    request, candidates = _scenario()
    output = run_bounded_workflow(request, _SequenceAnalyzer(candidates))

    item_ids = [item["item_id"] for item in output.retrieval_items]
    assert "invoice-1" in item_ids
    assert "invoice-document" not in item_ids


@pytest.mark.parametrize(
    ("changes", "rule_code"),
    [
        (
            {"invoice_line_amount": "USD 25", "invoice_total_amount": "USD 25"},
            "invoice_line_amount",
        ),
        ({"invoice_total_amount": "USD 25"}, "invoice_total_amount"),
        ({"certificate_origin": "CN"}, "invoice_certificate_origin"),
        ({"packing_bl_no": "BL-002"}, "bill_packing_bl_no"),
        ({"packing_container": "EFGH7654321"}, "bill_packing_container"),
        ({"packing_gross_weight": "120 KG"}, "bill_packing_gross_weight"),
    ],
)
def test_specified_cross_document_rules_report_real_mismatches(
    changes: dict[str, str], rule_code: str
) -> None:
    request, candidates = _scenario(**changes)
    output = run_bounded_workflow(request, _SequenceAnalyzer(candidates))

    comparison = next(value for value in output.comparisons if value.rule_code == rule_code)
    assert comparison.status == "mismatch"
    assert any(finding.comparison_id == comparison.comparison_id for finding in output.findings)


def test_ambiguous_item_pair_is_needs_review_not_false_mismatch() -> None:
    request, candidates = _scenario(duplicate_packing_item=True)
    output = run_bounded_workflow(request, _SequenceAnalyzer(candidates))

    pairing = [
        comparison
        for comparison in output.comparisons
        if comparison.rule_code == "invoice_packing_item_match"
    ]
    assert pairing and all(comparison.status == "needs_review" for comparison in pairing)
    assert not any(
        comparison.rule_code == "invoice_packing_quantity_unit" and comparison.status == "mismatch"
        for comparison in output.comparisons
    )


def test_invalid_source_quote_is_excluded_and_marked_for_review() -> None:
    request, candidates = _scenario()
    candidates["invoice"][0] = candidates["invoice"][0].model_copy(
        update={
            "fields": [
                _field("invoice_no", "INV-001", "not present in source"),
                *candidates["invoice"][0].fields[1:],
            ]
        }
    )
    output = run_bounded_workflow(request, _SequenceAnalyzer(candidates))

    invoice = next(
        analysis for analysis in output.document_analyses if analysis.document_id == "invoice"
    )
    assert all(fact.name != "invoice_no" for fact in invoice.facts)
    assert any(warning.code == "field_source_invalid" for warning in invoice.warnings)
    assert any(finding.code == "field_source_invalid" for finding in output.findings)


def test_reextracts_each_document_at_most_once_for_missing_targeted_fields() -> None:
    text = "\n".join(
        (
            "COMMERCIAL INVOICE",
            "Invoice No: INV-RETRY",
            "Total Amount: USD 10",
            "HS Code: 123456",
        )
    )
    request = ServiceReviewRunRequestV3.model_validate(
        {
            "case_id": "case-retry",
            "review_run_id": "run-retry",
            "review_as_of": "2026-08-05",
            "trade_type": "import",
            "jurisdiction": "KR",
            "documents": [
                {"document_id": "invoice", "filename": "invoice.txt", "content_text": text}
            ],
        }
    )
    first = ClaudeDocumentCandidate(
        document_type="invoice",
        classification_source_quote="COMMERCIAL INVOICE",
        fields=[_field("invoice_no", "INV-RETRY", "Invoice No: INV-RETRY")],
    )
    retry = ClaudeDocumentCandidate(
        document_type="invoice",
        classification_source_quote="COMMERCIAL INVOICE",
        fields=[
            _field("invoice_no", "INV-RETRY", "Invoice No: INV-RETRY"),
            _field("total_amount", "USD 10", "Total Amount: USD 10"),
            _field("hs_code", "123456", "HS Code: 123456"),
        ],
    )
    analyzer = _SequenceAnalyzer({"invoice": [first, retry]})
    output = run_bounded_workflow(request, analyzer)

    assert analyzer.calls == [
        ("invoice", ()),
        ("invoice", ("hs_code", "total_amount")),
    ]
    assert output.workflow.reextractions[0].document_id == "invoice"
    assert output.workflow.reextractions[0].attempt_count == 1
    assert output.workflow.reextractions[0].status == "completed"
    assert [entry.action for entry in output.workflow.planner.actions] == [
        "analyze_documents",
        "reextract_documents",
        "compare_documents",
    ]
    assert output.workflow.planner.cycle_count == 2


def test_approved_human_correction_preserves_original_fact_and_drives_comparison() -> None:
    request, candidates = _scenario(invoice_quantity="3 PCS", packing_quantity="2 PCS")
    payload = request.model_dump(mode="json")
    payload.update(
        {
            "parent_review_run_id": "run-parent",
            "applied_correction_proposal_ids": ["proposal-quantity"],
            "human_corrections": [
                {
                    "correction_proposal_id": "proposal-quantity",
                    "document_id": "invoice",
                    "item_id": "invoice:item:1",
                    "fact_id": None,
                    "field_name": "quantity",
                    "new_value": "2 PCS",
                    "unit": "PCS",
                    "reason": "수량 정정본을 담당자가 확인했습니다.",
                    "source_kind": "human_provided",
                }
            ],
        }
    )
    child_request = ServiceReviewRunRequestV3.model_validate(payload)

    output = run_bounded_workflow(child_request, _SequenceAnalyzer(candidates))

    comparison = next(
        value for value in output.comparisons if value.rule_code == "invoice_packing_quantity_unit"
    )
    assert comparison.status == "pass"
    invoice = next(value for value in output.document_analyses if value.document_id == "invoice")
    item = next(value for value in invoice.line_items if value.item_id == "invoice:item:1")
    assert any(
        fact.source.source_kind == "human_provided" and fact.raw_value == "2 PCS"
        for fact in item.facts
    )
    correction_action = next(
        value
        for value in output.workflow.planner.actions
        if value.action == "apply_human_corrections"
    )
    assert correction_action.correction_proposal_ids == ["proposal-quantity"]
    assert [value.action for value in output.workflow.planner.actions] == [
        "analyze_documents",
        "apply_human_corrections",
        "compare_documents",
    ]
    sections = build_grounded_report_sections(
        analyses=output.document_analyses,
        comparisons=output.comparisons,
        evidence=[],
        findings=output.findings,
        actions=[],
    )
    correction_claim = next(
        claim
        for section in sections
        if section.section == "execution_process"
        for claim in section.claims
        if "proposal-quantity" in claim.text
    )
    assert correction_claim.fact_ids == ["human:proposal-quantity"]
    validate_grounded_report_sections(
        sections,
        analyses=output.document_analyses,
        comparisons=output.comparisons,
        evidence=[],
    )
    missing_correction = [section.model_copy() for section in sections]
    execution_index = next(
        index
        for index, section in enumerate(missing_correction)
        if section.section == "execution_process"
    )
    missing_correction[execution_index] = missing_correction[execution_index].model_copy(
        update={"claims": missing_correction[execution_index].claims[:1]}
    )
    with pytest.raises(ValueError, match="report_human_correction_claim_missing"):
        validate_grounded_report_sections(
            missing_correction,
            analyses=output.document_analyses,
            comparisons=output.comparisons,
            evidence=[],
        )


def test_human_correction_with_another_document_item_fails_closed() -> None:
    request, candidates = _scenario(invoice_quantity="3 PCS", packing_quantity="2 PCS")
    payload = request.model_dump(mode="json")
    payload.update(
        {
            "parent_review_run_id": "run-parent",
            "applied_correction_proposal_ids": ["proposal-wrong-item"],
            "human_corrections": [
                {
                    "correction_proposal_id": "proposal-wrong-item",
                    "document_id": "invoice",
                    "item_id": "packing:item:1",
                    "field_name": "quantity",
                    "new_value": "2 PCS",
                    "reason": "잘못된 품목 대상은 적용할 수 없습니다.",
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="human_correction_item_target_not_found"):
        run_bounded_workflow(
            ServiceReviewRunRequestV3.model_validate(payload), _SequenceAnalyzer(candidates)
        )
