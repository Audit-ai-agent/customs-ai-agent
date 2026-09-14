from __future__ import annotations

from customs_ai_agent.schemas.v3 import ComparisonOperandV3, ComparisonV3, FindingV3
from customs_ai_agent.service.expert_review import merge_llm_verdict, rule_verdict


def _comparison(status: str, comparison_id: str = "comparison-1") -> ComparisonV3:
    return ComparisonV3(
        comparison_id=comparison_id,
        rule_code="invoice_packing_quantity",
        status=status,  # type: ignore[arg-type]
        left=ComparisonOperandV3(document_id="invoice", fact_ids=["f"]),
        right=ComparisonOperandV3(document_id="packing", fact_ids=["g"]),
        message="수량이 다릅니다.",
    )


def _finding(
    finding_id: str,
    *,
    severity: str = "high",
    comparison_id: str | None = "comparison-1",
    kind: str = "validation_issue",
    evidence_ids: list[str] | None = None,
) -> FindingV3:
    return FindingV3(
        finding_id=finding_id,
        kind=kind,  # type: ignore[arg-type]
        code="invoice_packing_quantity_mismatch",
        severity=severity,  # type: ignore[arg-type]
        message="확인이 필요합니다.",
        comparison_id=comparison_id,
        evidence_ids=evidence_ids or [],
    )


def _verdict(**changes: object):  # type: ignore[no-untyped-def]
    arguments: dict[str, object] = {
        "findings": [],
        "comparisons_by_id": {},
        "execution_status": "succeeded",
        "data_status": "ok",
        "retrieval_skipped": False,
        "review_track": "expedited_check",
    }
    arguments.update(changes)
    return rule_verdict(**arguments)  # type: ignore[arg-type]


def test_rule_verdict_is_ineligible_only_for_high_severity_mismatches() -> None:
    mismatch = _comparison("mismatch")
    verdict = _verdict(
        findings=[_finding("finding-a"), _finding("finding-b", severity="medium")],
        comparisons_by_id={mismatch.comparison_id: mismatch},
    )
    assert verdict.verdict == "ineligible"
    assert verdict.advisory is True
    assert verdict.source == "rule"
    assert [reason.code for reason in verdict.reasons] == ["document_mismatch"]
    assert verdict.reasons[0].finding_ids == ["finding-a"]


def test_rule_verdict_is_indeterminate_for_open_questions() -> None:
    needs_review = _comparison("needs_review")
    regulatory = _verdict(
        findings=[
            _finding(
                "finding-r",
                comparison_id=None,
                kind="regulatory_finding",
                evidence_ids=["evidence-1"],
            )
        ]
    )
    assert regulatory.verdict == "indeterminate"
    assert regulatory.reasons[0].code == "requirement_confirmation_needed"
    assert regulatory.reasons[0].evidence_ids == ["evidence-1"]
    review = _verdict(
        findings=[_finding("finding-n", severity="medium")],
        comparisons_by_id={needs_review.comparison_id: needs_review},
    )
    assert review.verdict == "indeterminate"
    assert review.reasons[0].code == "comparison_needs_review"
    partial = _verdict(execution_status="partial")
    assert partial.verdict == "indeterminate"
    assert partial.reasons[0].code == "execution_incomplete"
    skipped = _verdict(retrieval_skipped=True)
    assert skipped.reasons[0].code == "evidence_not_requested"
    weak = _verdict(review_track="detailed_review")
    assert weak.verdict == "indeterminate"
    assert weak.reasons[0].code == "low_confidence"


def test_rule_verdict_is_eligible_only_when_nothing_is_open() -> None:
    verdict = _verdict()
    assert verdict.verdict == "eligible"
    assert [reason.code for reason in verdict.reasons] == ["no_blocking_finding"]


def test_llm_verdict_may_only_tighten_and_must_cite_known_ids() -> None:
    rule = _verdict(review_track="detailed_review")
    assert rule.verdict == "indeterminate"
    known = frozenset({"finding-a"})
    evidence = frozenset({"evidence-1"})

    lenient = merge_llm_verdict(
        rule,
        {"verdict": "eligible", "reasons": []},
        known_finding_ids=known,
        known_evidence_ids=evidence,
    )
    assert lenient == rule

    uncited = merge_llm_verdict(
        rule,
        {"verdict": "ineligible", "reasons": [{"code": "x", "message": "m", "finding_ids": []}]},
        known_finding_ids=known,
        known_evidence_ids=evidence,
    )
    assert uncited == rule

    unknown_id = merge_llm_verdict(
        rule,
        {
            "verdict": "ineligible",
            "reasons": [{"code": "x", "message": "m", "finding_ids": ["finding-zzz"]}],
        },
        known_finding_ids=known,
        known_evidence_ids=evidence,
    )
    assert unknown_id == rule

    stricter = merge_llm_verdict(
        rule,
        {
            "verdict": "ineligible",
            "reasons": [
                {
                    "code": "requirement_document_missing",
                    "message": "원산지증명서가 요건서류 목록에 없습니다.",
                    "finding_ids": ["finding-a"],
                    "evidence_ids": ["evidence-1"],
                }
            ],
        },
        known_finding_ids=known,
        known_evidence_ids=evidence,
    )
    assert stricter.verdict == "ineligible"
    assert stricter.source == "rule+llm"
    assert stricter.reasons[-1].code == "llm_requirement_document_missing"
    assert stricter.reasons[-1].finding_ids == ["finding-a"]
    assert stricter.reasons[0] == rule.reasons[0]

    malformed = merge_llm_verdict(
        rule,
        {"verdict": "ineligible", "reasons": "no"},
        known_finding_ids=known,
        known_evidence_ids=evidence,
    )
    assert malformed == rule
