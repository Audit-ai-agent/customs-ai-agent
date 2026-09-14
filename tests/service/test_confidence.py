from __future__ import annotations

from customs_ai_agent.schemas.v3 import (
    ComparisonOperandV3,
    ComparisonV3,
    DocumentAnalysisV3,
    ExtractionWarningV3,
    FindingV3,
)
from customs_ai_agent.service.confidence import (
    attach_finding_confidence,
    comparison_component,
    comparison_key,
    evidence_component,
    extraction_component,
    finding_confidence,
    review_confidence,
)


def _comparison(status: str, *, left_facts: int = 1, right_facts: int = 1) -> ComparisonV3:
    return ComparisonV3(
        comparison_id=f"comparison-{status}-{left_facts}{right_facts}",
        rule_code="invoice_packing_quantity",
        status=status,  # type: ignore[arg-type]
        left=ComparisonOperandV3(
            document_id="invoice", item_id="invoice:item:1", fact_ids=["f"] * left_facts
        ),
        right=ComparisonOperandV3(
            document_id="packing", item_id="packing:item:1", fact_ids=["g"] * right_facts
        ),
        message="수량이 다릅니다.",
    )


def _finding(comparison: ComparisonV3 | None, *, kind: str = "validation_issue") -> FindingV3:
    return FindingV3(
        finding_id="finding-1",
        kind=kind,  # type: ignore[arg-type]
        code="invoice_packing_quantity_mismatch",
        severity="high",
        message="수량이 다릅니다.",
        comparison_id=comparison.comparison_id if comparison else None,
        evidence_ids=["evidence-1"] if kind == "regulatory_finding" else [],
    )


def _analysis(document_type: str = "invoice", *, warnings: int = 0) -> DocumentAnalysisV3:
    return DocumentAnalysisV3.model_validate(
        {
            "document_id": "invoice",
            "filename": "invoice.txt",
            "document_type": document_type,
            "extraction_status": "complete",
            "facts": [
                {
                    "fact_id": "invoice:field:invoice_no:1",
                    "name": "invoice_no",
                    "raw_value": "INV-1",
                    "source": {"document_id": "invoice", "quote": "INV-1", "occurrence": 0},
                }
            ],
            "warnings": [
                ExtractionWarningV3(code="field_source_invalid", message="x").model_dump()
                for _ in range(warnings)
            ],
        }
    )


def test_finding_confidence_rewards_grounded_mismatches_and_penalises_gaps() -> None:
    grounded = finding_confidence(
        _finding(_comparison("mismatch")),
        comparison=_comparison("mismatch"),
        preliminary_status=None,
        reextracted_document_ids=frozenset(),
        unavailable_document_ids=frozenset(),
        data_status="ok",
    )
    half_grounded = finding_confidence(
        _finding(_comparison("mismatch", right_facts=0)),
        comparison=_comparison("mismatch", right_facts=0),
        preliminary_status=None,
        reextracted_document_ids=frozenset(),
        unavailable_document_ids=frozenset(),
        data_status="ok",
    )
    needs_review = finding_confidence(
        _finding(_comparison("needs_review")),
        comparison=_comparison("needs_review"),
        preliminary_status=None,
        reextracted_document_ids=frozenset(),
        unavailable_document_ids=frozenset(),
        data_status="ok",
    )
    assert grounded == 0.9
    assert half_grounded == 0.675
    assert needs_review == 0.6


def test_finding_confidence_reflects_the_audit_loop_outcome() -> None:
    comparison = _comparison("mismatch")
    confirmed = finding_confidence(
        _finding(comparison),
        comparison=comparison,
        preliminary_status="mismatch",
        reextracted_document_ids=frozenset({"invoice"}),
        unavailable_document_ids=frozenset(),
        data_status="ok",
    )
    unavailable = finding_confidence(
        _finding(comparison),
        comparison=comparison,
        preliminary_status="mismatch",
        reextracted_document_ids=frozenset(),
        unavailable_document_ids=frozenset({"packing"}),
        data_status="ok",
    )
    changed = finding_confidence(
        _finding(comparison),
        comparison=comparison,
        preliminary_status="needs_review",
        reextracted_document_ids=frozenset({"invoice"}),
        unavailable_document_ids=frozenset(),
        data_status="ok",
    )
    assert confirmed == 0.95
    assert unavailable == 0.75
    assert changed == 0.9


def test_regulatory_and_warning_findings_use_fixed_signals() -> None:
    regulatory = finding_confidence(
        _finding(None, kind="regulatory_finding"),
        comparison=None,
        preliminary_status=None,
        reextracted_document_ids=frozenset(),
        unavailable_document_ids=frozenset(),
        data_status="partial",
    )
    warning = finding_confidence(
        _finding(None),
        comparison=None,
        preliminary_status=None,
        reextracted_document_ids=frozenset(),
        unavailable_document_ids=frozenset(),
        data_status="ok",
    )
    assert regulatory == 0.6
    assert warning == 0.5


def test_components_are_bounded_and_deterministic() -> None:
    assert extraction_component([_analysis()]) == 1.0
    assert extraction_component([_analysis(warnings=1)]) == 0.5
    assert extraction_component([_analysis("unknown")]) == 0.4
    assert comparison_component([]) == 0.5
    assert comparison_component([_comparison("pass"), _comparison("needs_review")]) == 0.75
    assert comparison_component([_comparison("skip")]) == 0.5
    assert evidence_component("ok", retrieval_skipped=False) == 1.0
    assert evidence_component("failed", retrieval_skipped=False) == 0.0
    assert evidence_component("failed", retrieval_skipped=True) == 0.7


def test_review_confidence_selects_the_track_from_the_threshold() -> None:
    strong = review_confidence(
        analyses=[_analysis()],
        comparisons=[_comparison("pass")],
        data_status="ok",
        retrieval_skipped=False,
        threshold=0.8,
    )
    weak = review_confidence(
        analyses=[_analysis("unknown")],
        comparisons=[_comparison("needs_review")],
        data_status="partial",
        retrieval_skipped=False,
        threshold=0.8,
    )
    assert strong.overall == 1.0
    assert strong.review_track == "expedited_check"
    assert weak.overall == 0.48
    assert weak.review_track == "detailed_review"
    assert weak.method == "deterministic_signals"
    twice = review_confidence(
        analyses=[_analysis("unknown")],
        comparisons=[_comparison("needs_review")],
        data_status="partial",
        retrieval_skipped=False,
        threshold=0.8,
    )
    assert twice == weak


def test_attach_finding_confidence_uses_the_stable_comparison_key() -> None:
    preliminary = _comparison("mismatch")
    final = preliminary.model_copy(update={"comparison_id": "comparison-after-reread"})
    assert comparison_key(preliminary) == comparison_key(final)
    scored = attach_finding_confidence(
        [_finding(final)],
        comparisons_by_id={final.comparison_id: final},
        preliminary_status_by_key={comparison_key(preliminary): preliminary.status},
        reextracted_document_ids=frozenset({"packing"}),
        unavailable_document_ids=frozenset(),
        data_status="ok",
    )
    assert scored[0].confidence == 0.95
    assert scored[0].finding_id == "finding-1"
