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


def _export_scenario(
    *,
    declaration_hs: str = "123456",
    declaration_quantity: str = "2 PCS",
    declaration_amount: str = "USD 20",
    destination_country: str = "US",
    include_declaration: bool = True,
) -> tuple[ServiceReviewRunRequestV3, dict[str, list[ClaudeDocumentCandidate]]]:
    """A source-grounded air-export document set for bounded workflow tests."""

    invoice_line = (
        "Description: Widget | HS Code: 123456 | Quantity: 2 PCS | Unit: PCS | "
        "Unit Price: USD 10 | Amount: USD 20"
    )
    declaration_line = (
        f"Description: Widget | HS Code: {declaration_hs} | Quantity: {declaration_quantity} | "
        f"Unit: PCS | Amount: {declaration_amount}"
    )
    documents = [
        {
            "document_id": "invoice",
            "filename": "invoice.txt",
            "content_text": "\n".join(
                (
                    "COMMERCIAL INVOICE",
                    "Invoice No: INV-EXPORT-001",
                    "Total Amount: USD 20",
                    "Country of Origin: KR",
                    invoice_line,
                )
            ),
        },
        {
            "document_id": "packing",
            "filename": "packing.txt",
            "content_text": "\n".join(
                (
                    "PACKING LIST",
                    "Packing No: PL-EXPORT-001",
                    "AWB No: AWB-EXPORT-001",
                    "Total Gross Weight: 100 KG",
                    "Description: Widget | HS Code: 123456 | Quantity: 2 PCS | Unit: PCS",
                )
            ),
        },
        {
            "document_id": "certificate",
            "filename": "certificate.txt",
            "content_text": "\n".join(
                (
                    "CERTIFICATE OF ORIGIN",
                    "Certificate No: COO-EXPORT-001",
                    "Origin Country: KR",
                    "HS Code: 123456",
                )
            ),
        },
        {
            "document_id": "awb",
            "filename": "awb.txt",
            "content_text": "\n".join(
                (
                    "AIR WAYBILL",
                    "AWB No: AWB-EXPORT-001",
                    "Gross Weight: 100 KG",
                )
            ),
        },
    ]
    candidates: dict[str, list[ClaudeDocumentCandidate]] = {
        "invoice": [
            ClaudeDocumentCandidate(
                document_type="invoice",
                classification_source_quote="COMMERCIAL INVOICE",
                fields=[
                    _field("invoice_no", "INV-EXPORT-001", "Invoice No: INV-EXPORT-001"),
                    _field("total_amount", "USD 20", "Total Amount: USD 20"),
                    _field("country_of_origin", "KR", "Country of Origin: KR"),
                ],
                line_items=[
                    _line(
                        invoice_line,
                        ("description", "Widget"),
                        ("hs_code", "123456"),
                        ("quantity", "2 PCS"),
                        ("unit", "PCS"),
                        ("unit_price", "USD 10"),
                        ("amount", "USD 20"),
                    )
                ],
            )
        ],
        "packing": [
            ClaudeDocumentCandidate(
                document_type="packing_list",
                classification_source_quote="PACKING LIST",
                fields=[
                    _field("packing_no", "PL-EXPORT-001", "Packing No: PL-EXPORT-001"),
                    _field("awb_no", "AWB-EXPORT-001", "AWB No: AWB-EXPORT-001"),
                    _field("total_gross_weight_kg", "100 KG", "Total Gross Weight: 100 KG"),
                ],
                line_items=[
                    _line(
                        "Description: Widget | HS Code: 123456 | Quantity: 2 PCS | Unit: PCS",
                        ("description", "Widget"),
                        ("hs_code", "123456"),
                        ("quantity", "2 PCS"),
                        ("unit", "PCS"),
                    )
                ],
            )
        ],
        "certificate": [
            ClaudeDocumentCandidate(
                document_type="certificate_of_origin",
                classification_source_quote="CERTIFICATE OF ORIGIN",
                fields=[
                    _field("certificate_no", "COO-EXPORT-001", "Certificate No: COO-EXPORT-001"),
                    _field("origin_country", "KR", "Origin Country: KR"),
                    _field("hs_code", "123456", "HS Code: 123456"),
                ],
            )
        ],
        "awb": [
            ClaudeDocumentCandidate(
                document_type="air_waybill",
                classification_source_quote="AIR WAYBILL",
                fields=[
                    _field("awb_no", "AWB-EXPORT-001", "AWB No: AWB-EXPORT-001"),
                    _field("gross_weight_kg", "100 KG", "Gross Weight: 100 KG"),
                ],
            )
        ],
    }
    if include_declaration:
        documents.append(
            {
                "document_id": "declaration",
                "filename": "declaration.txt",
                "content_text": "\n".join(
                    (
                        "EXPORT DECLARATION",
                        "Export Declaration No: ED-EXPORT-001",
                        "Filing Date: 2026-08-05",
                        "Submission Status: Accepted",
                        "Invoice No: INV-EXPORT-001",
                        f"Total Amount: {declaration_amount}",
                        "Origin Country: KR",
                        f"Destination Country: {destination_country}",
                        declaration_line,
                    )
                ),
            }
        )
        candidates["declaration"] = [
            ClaudeDocumentCandidate(
                document_type="export_declaration",
                classification_source_quote="EXPORT DECLARATION",
                fields=[
                    _field(
                        "export_declaration_no",
                        "ED-EXPORT-001",
                        "Export Declaration No: ED-EXPORT-001",
                    ),
                    _field("filing_date", "2026-08-05", "Filing Date: 2026-08-05"),
                    _field("submission_status", "Accepted", "Submission Status: Accepted"),
                    _field("invoice_no", "INV-EXPORT-001", "Invoice No: INV-EXPORT-001"),
                    _field(
                        "total_amount", declaration_amount, f"Total Amount: {declaration_amount}"
                    ),
                    _field("origin_country", "KR", "Origin Country: KR"),
                    _field(
                        "destination_country",
                        destination_country,
                        f"Destination Country: {destination_country}",
                    ),
                ],
                line_items=[
                    _line(
                        declaration_line,
                        ("description", "Widget"),
                        ("hs_code", declaration_hs),
                        ("quantity", declaration_quantity),
                        ("unit", "PCS"),
                        ("amount", declaration_amount),
                    )
                ],
            )
        ]
    request = ServiceReviewRunRequestV3.model_validate(
        {
            "case_id": "case-export-workflow",
            "review_run_id": "run-export-workflow",
            "review_as_of": "2026-08-05",
            "trade_type": "export",
            "jurisdiction": "KR",
            "documents": documents,
        }
    )
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


def test_export_workflow_compares_declaration_and_awb_with_export_destination() -> None:
    request, candidates = _export_scenario()

    output = run_bounded_workflow(request, _SequenceAnalyzer(candidates))

    statuses = {item.rule_code: item.status for item in output.comparisons}
    assert statuses["awb_packing_awb_no"] == "pass"
    assert statuses["awb_packing_gross_weight"] == "pass"
    assert statuses["export_declaration_destination_country"] == "pass"
    assert statuses["invoice_export_declaration_invoice_no"] == "pass"
    assert statuses["invoice_export_declaration_total_amount"] == "pass"
    assert statuses["invoice_export_declaration_origin"] == "pass"
    assert statuses["invoice_export_declaration_hs_code"] == "pass"
    assert statuses["invoice_export_declaration_quantity_unit"] == "pass"
    assert statuses["invoice_export_declaration_amount"] == "pass"
    assert output.retrieval_skip_reason is None
    assert len(output.retrieval_items) == 1
    assert output.retrieval_items[0]["destination_country"] == "US"


def test_export_workflow_reports_declared_item_mismatch_without_guessing_destination() -> None:
    request, candidates = _export_scenario(
        declaration_hs="654321", declaration_quantity="3 PCS", declaration_amount="USD 30"
    )

    output = run_bounded_workflow(request, _SequenceAnalyzer(candidates))

    statuses = {item.rule_code: item.status for item in output.comparisons}
    assert statuses["invoice_export_declaration_hs_code"] == "mismatch"
    assert statuses["invoice_export_declaration_quantity_unit"] == "mismatch"
    assert statuses["invoice_export_declaration_amount"] == "mismatch"
    assert output.retrieval_items[0]["destination_country"] == "US"


def test_export_workflow_skips_evidence_when_destination_is_not_a_two_letter_code() -> None:
    request, candidates = _export_scenario(destination_country="United States")

    output = run_bounded_workflow(request, _SequenceAnalyzer(candidates))

    statuses = {item.rule_code: item.status for item in output.comparisons}
    assert statuses["export_declaration_destination_country"] == "needs_review"
    assert output.retrieval_items == []
    assert output.retrieval_skip_reason == "export_destination_country_missing"
    assert output.with_evidence_skipped().workflow.stages[-2].summary == (
        "수출 목적국 코드가 문서에서 확인되지 않아 법령 근거 조회를 실행하지 않았습니다."
    )


def test_export_workflow_requires_an_export_declaration() -> None:
    request, candidates = _export_scenario(include_declaration=False)

    output = run_bounded_workflow(request, _SequenceAnalyzer(candidates))

    presence = next(
        item for item in output.comparisons if item.rule_code == "export_declaration_presence"
    )
    assert presence.status == "needs_review"


def test_weight_field_schema_unit_normalizes_naked_model_number() -> None:
    """A verified KG source may be returned by the model as only its number."""
    request, candidates = _scenario()
    bill = candidates["bill"][0]
    bill.fields[2] = _field("gross_weight_kg", "100", "Gross Weight: 100 KG")

    output = run_bounded_workflow(request, _SequenceAnalyzer(candidates))

    comparison = next(
        comparison
        for comparison in output.comparisons
        if comparison.rule_code == "bill_packing_gross_weight"
    )
    assert comparison.status == "pass"
    assert comparison.unit == "KG"


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


def test_retrieval_groups_one_commodity_and_excludes_no_hs_bill_of_lading() -> None:
    request, candidates = _scenario()
    output = run_bounded_workflow(request, _SequenceAnalyzer(candidates))

    assert len(output.retrieval_items) == 1
    item = output.retrieval_items[0]
    assert item["item_id"].startswith("commodity-")
    assert item["hs_code"] == "123456"
    assert output.retrieval_document_ids[item["item_id"]] == ["certificate", "invoice", "packing"]
    # A B/L normally has no HS code.  It is not a law-retrieval target and it
    # must not become a visible missing-HS condition.
    assert output.retrieval_excluded_document_ids == []


def _no_hs_invoice_scenario(
    *, reviewer_code: str | None
) -> tuple[ServiceReviewRunRequestV3, dict[str, list[ClaudeDocumentCandidate]]]:
    invoice_line = (
        "Description: Widget | Quantity: 2 PCS | Unit: PCS | Unit Price: USD 10 | Amount: USD 20"
    )
    invoice_text = "\n".join(
        (
            "COMMERCIAL INVOICE",
            "Invoice No: INV-001",
            "Total Amount: USD 20",
            "Country of Origin: KR",
            invoice_line,
        )
    )
    payload: dict[str, Any] = {
        "case_id": "case-no-hs",
        "review_run_id": "run-no-hs",
        "review_as_of": "2026-08-05",
        "trade_type": "import",
        "jurisdiction": "KR",
        "documents": [
            {"document_id": "invoice", "filename": "invoice.txt", "content_text": invoice_text}
        ],
    }
    if reviewer_code is not None:
        payload["reviewer_hs_selection"] = {"code": reviewer_code, "source_kind": "human_provided"}
    request = ServiceReviewRunRequestV3.model_validate(payload)
    candidates = {
        "invoice": [
            ClaudeDocumentCandidate(
                document_type="invoice",
                classification_source_quote="COMMERCIAL INVOICE",
                fields=[
                    _field("invoice_no", "INV-001", "Invoice No: INV-001"),
                    _field("total_amount", "USD 20", "Total Amount: USD 20"),
                    _field("country_of_origin", "KR", "Country of Origin: KR"),
                ],
                line_items=[
                    _line(
                        invoice_line,
                        ("description", "Widget"),
                        ("quantity", "2 PCS"),
                        ("unit", "PCS"),
                        ("unit_price", "USD 10"),
                        ("amount", "USD 20"),
                    )
                ],
            )
        ]
    }
    return request, candidates


def test_retrieval_uses_reviewer_hs_selection_when_no_document_cites_a_code() -> None:
    request, candidates = _no_hs_invoice_scenario(reviewer_code="6109101000")
    output = run_bounded_workflow(request, _SequenceAnalyzer(candidates))

    assert output.retrieval_skip_reason is None
    assert output.retrieval_reviewer_hs_code == "6109101000"
    assert len(output.retrieval_items) == 1
    item = output.retrieval_items[0]
    assert item["hs_code"] == "6109101000"
    assert item["description"] == "Widget"
    assert item["origin_country"] == "KR"
    assert output.retrieval_document_ids[item["item_id"]] == ["invoice"]
    assert output.retrieval_excluded_document_ids == []
    complete = output.with_evidence_status("ok")
    retrieve = next(
        stage for stage in complete.workflow.stages if stage.stage == "retrieve_evidence"
    )
    assert "담당자가 확정한 HSK 코드(6109101000)" in retrieve.summary


def test_retrieval_without_reviewer_selection_still_skips_when_no_code() -> None:
    request, candidates = _no_hs_invoice_scenario(reviewer_code=None)
    output = run_bounded_workflow(request, _SequenceAnalyzer(candidates))

    assert output.retrieval_items == []
    assert output.retrieval_skip_reason == "no_hs_qualified_commodity"
    assert output.retrieval_reviewer_hs_code is None


def test_reviewer_hs_selection_completes_a_document_cited_hs6_prefix() -> None:
    request, candidates = _scenario()
    payload = request.input_manifest_wire()
    payload["reviewer_hs_selection"] = {"code": "1234561234", "source_kind": "human_provided"}
    request = ServiceReviewRunRequestV3.model_validate(payload)
    output = run_bounded_workflow(request, _SequenceAnalyzer(candidates))

    assert output.retrieval_reviewer_hs_code == "1234561234"
    assert [item["hs_code"] for item in output.retrieval_items] == ["1234561234"]
    complete = output.with_evidence_status("ok")
    retrieve = next(
        stage for stage in complete.workflow.stages if stage.stage == "retrieve_evidence"
    )
    assert "담당자가 확정한 HSK 코드(1234561234)" in retrieve.summary


def test_reviewer_hs_selection_never_overrides_document_cited_codes() -> None:
    request, candidates = _scenario()
    payload = request.input_manifest_wire()
    payload["reviewer_hs_selection"] = {"code": "6109101000", "source_kind": "human_provided"}
    request = ServiceReviewRunRequestV3.model_validate(payload)
    output = run_bounded_workflow(request, _SequenceAnalyzer(candidates))

    assert output.retrieval_reviewer_hs_code is None
    assert [item["hs_code"] for item in output.retrieval_items] == ["123456"]


def test_hs_code_alias_is_canonical_for_required_field_checks() -> None:
    request, candidates = _scenario()
    certificate = candidates["certificate"][0]
    certificate.fields[-1] = _field("hs_codes", "123456", "HS Code: 123456")

    output = run_bounded_workflow(request, _SequenceAnalyzer(candidates))

    certificate_hs = next(
        comparison
        for comparison in output.comparisons
        if comparison.rule_code == "hs_code_presence"
        and comparison.left.document_id == "certificate"
    )
    assert certificate_hs.status == "pass"
    assert not any(
        finding.code == "hs_code_not_found" and "certificate" in finding.document_ids
        for finding in output.findings
    )


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
        if "원문 인용" in claim.text
    )
    assert correction_claim.fact_ids == ["human:proposal-quantity"]
    assert "proposal-quantity" not in correction_claim.text
    assert "human:proposal-quantity" not in correction_claim.text
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


def test_document_level_correction_survives_claude_field_reordering() -> None:
    """A child run may reorder extraction fields without moving the target."""

    request, candidates = _scenario()
    original_invoice = candidates["invoice"][0]
    parent_origin_field = original_invoice.fields[2].model_copy(update={"name": "origin_country"})
    parent_invoice_candidate = original_invoice.model_copy(
        update={"fields": [*original_invoice.fields[:2], parent_origin_field]}
    )
    parent = run_bounded_workflow(
        request,
        _SequenceAnalyzer({**candidates, "invoice": [parent_invoice_candidate]}),
    )
    parent_invoice = next(
        item for item in parent.document_analyses if item.document_id == "invoice"
    )
    parent_origin = next(item for item in parent_invoice.facts if item.name == "origin_country")

    child_origin_field = parent_origin_field.model_copy(update={"name": "country_of_origin"})
    reordered_candidate = parent_invoice_candidate.model_copy(
        update={"fields": [child_origin_field, *parent_invoice_candidate.fields[:2]]}
    )
    child_candidates = {**candidates, "invoice": [reordered_candidate]}
    child_payload = request.model_dump(mode="json")
    child_payload.update(
        {
            "parent_review_run_id": "run-parent",
            "applied_correction_proposal_ids": ["proposal-origin"],
            "human_corrections": [
                {
                    "correction_proposal_id": "proposal-origin",
                    "document_id": "invoice",
                    "fact_id": parent_origin.fact_id,
                    "field_name": "origin_country",
                    "new_value": "KR",
                    "reason": "승인된 원산지 보완값입니다.",
                    "source_kind": "human_provided",
                }
            ],
        }
    )

    output = run_bounded_workflow(
        ServiceReviewRunRequestV3.model_validate(child_payload),
        _SequenceAnalyzer(child_candidates),
    )

    invoice = next(item for item in output.document_analyses if item.document_id == "invoice")
    assert any(
        fact.fact_id == "human:proposal-origin" and fact.source.source_kind == "human_provided"
        for fact in invoice.facts
    )


def test_document_level_correction_does_not_choose_an_ambiguous_reordered_fact() -> None:
    request, candidates = _scenario()
    invoice_candidate = candidates["invoice"][0]
    duplicate_origin = _field("country_of_origin", "KR", "Country of Origin: KR")
    ambiguous_candidate = invoice_candidate.model_copy(
        update={"fields": [*invoice_candidate.fields, duplicate_origin]}
    )
    payload = request.model_dump(mode="json")
    payload.update(
        {
            "parent_review_run_id": "run-parent",
            "applied_correction_proposal_ids": ["proposal-ambiguous"],
            "human_corrections": [
                {
                    "correction_proposal_id": "proposal-ambiguous",
                    "document_id": "invoice",
                    "fact_id": "invoice:field:origin_country:99",
                    "field_name": "country_of_origin",
                    "new_value": "KR",
                    "reason": "모호한 대상은 적용할 수 없습니다.",
                    "source_kind": "human_provided",
                }
            ],
        }
    )

    with pytest.raises(ValueError, match="human_correction_fact_target_ambiguous"):
        run_bounded_workflow(
            ServiceReviewRunRequestV3.model_validate(payload),
            _SequenceAnalyzer({**candidates, "invoice": [ambiguous_candidate]}),
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


def _item_quantity_correction(
    request: ServiceReviewRunRequestV3, *, fact_id: str, item_id: str | None
) -> ServiceReviewRunRequestV3:
    payload = request.model_dump(mode="json")
    payload.update(
        {
            "parent_review_run_id": "run-parent",
            "applied_correction_proposal_ids": ["proposal-quantity"],
            "human_corrections": [
                {
                    "correction_proposal_id": "proposal-quantity",
                    "document_id": "invoice",
                    "item_id": item_id,
                    "fact_id": fact_id,
                    "field_name": "quantity",
                    "new_value": "2 PCS",
                    "unit": "PCS",
                    "reason": "수량 정정본을 담당자가 확인했습니다.",
                    "source_kind": "human_provided",
                }
            ],
        }
    )
    return ServiceReviewRunRequestV3.model_validate(payload)


@pytest.mark.parametrize("item_id", ["invoice:item:1", None])
def test_item_correction_survives_claude_field_reordering(item_id: str | None) -> None:
    """The parent saw quantity second; the child extraction returns it third."""

    request, candidates = _scenario(invoice_quantity="3 PCS", packing_quantity="2 PCS")
    child_request = _item_quantity_correction(
        request, fact_id="invoice:item:1:field:quantity:2", item_id=item_id
    )

    output = run_bounded_workflow(child_request, _SequenceAnalyzer(candidates))

    invoice = next(value for value in output.document_analyses if value.document_id == "invoice")
    item = next(value for value in invoice.line_items if value.item_id == "invoice:item:1")
    assert not any(fact.fact_id == "invoice:item:1:field:quantity:2" for fact in item.facts)
    assert any(fact.fact_id == "human:proposal-quantity" for fact in item.facts)
    assert not any(fact.source.source_kind == "human_provided" for fact in invoice.facts)
    comparison = next(
        value for value in output.comparisons if value.rule_code == "invoice_packing_quantity_unit"
    )
    assert comparison.status == "pass"


@pytest.mark.parametrize("item_id", ["invoice:item:1", None])
def test_item_correction_does_not_choose_an_ambiguous_reordered_fact(item_id: str | None) -> None:
    request, candidates = _scenario(invoice_quantity="3 PCS", packing_quantity="2 PCS")
    invoice_candidate = candidates["invoice"][0]
    line = invoice_candidate.line_items[0]
    duplicate = line.model_copy(
        update={"fields": [*line.fields, _field("quantity", "3 PCS", line.source_quote)]}
    )
    ambiguous_candidate = invoice_candidate.model_copy(update={"line_items": [duplicate]})
    child_request = _item_quantity_correction(
        request, fact_id="invoice:item:1:field:quantity:2", item_id=item_id
    )

    with pytest.raises(ValueError, match="human_correction_fact_target_ambiguous"):
        run_bounded_workflow(
            child_request,
            _SequenceAnalyzer({**candidates, "invoice": [ambiguous_candidate]}),
        )


def test_item_correction_fact_id_claiming_another_item_fails_closed() -> None:
    request, candidates = _scenario(invoice_quantity="3 PCS", packing_quantity="2 PCS")
    child_request = _item_quantity_correction(
        request, fact_id="invoice:item:2:field:quantity:2", item_id="invoice:item:1"
    )

    with pytest.raises(ValueError, match="human_correction_fact_not_owned_by_item"):
        run_bounded_workflow(child_request, _SequenceAnalyzer(candidates))


def test_item_correction_fact_id_for_another_field_fails_closed() -> None:
    request, candidates = _scenario(invoice_quantity="3 PCS", packing_quantity="2 PCS")
    child_request = _item_quantity_correction(
        request, fact_id="invoice:item:1:field:amount:9", item_id=None
    )

    with pytest.raises(ValueError, match="human_correction_field_does_not_match_fact"):
        run_bounded_workflow(child_request, _SequenceAnalyzer(candidates))


def test_carton_mark_extracted_as_container_is_needs_review_not_mismatch() -> None:
    request, candidates = _scenario(packing_container="C/NO. 1-101")
    output = run_bounded_workflow(request, _SequenceAnalyzer(candidates))

    comparison = next(
        value for value in output.comparisons if value.rule_code == "bill_packing_container"
    )
    assert comparison.status == "needs_review"


@pytest.mark.parametrize("item_id", ["invoice:item:1", None])
def test_item_correction_applies_when_child_extraction_omits_the_field(
    item_id: str | None,
) -> None:
    """E0827: the child extraction returned the item without any quantity fact."""

    request, candidates = _scenario(invoice_quantity="3 PCS", packing_quantity="2 PCS")
    invoice_candidate = candidates["invoice"][0]
    line = invoice_candidate.line_items[0]
    without_quantity = line.model_copy(
        update={"fields": [field for field in line.fields if field.name != "quantity"]}
    )
    child_candidate = invoice_candidate.model_copy(update={"line_items": [without_quantity]})
    child_request = _item_quantity_correction(
        request, fact_id="invoice:item:1:field:quantity:2", item_id=item_id
    )

    output = run_bounded_workflow(
        child_request, _SequenceAnalyzer({**candidates, "invoice": [child_candidate]})
    )

    invoice = next(value for value in output.document_analyses if value.document_id == "invoice")
    item = next(value for value in invoice.line_items if value.item_id == "invoice:item:1")
    assert [fact.fact_id for fact in item.facts if fact.name == "quantity"] == [
        "human:proposal-quantity"
    ]
