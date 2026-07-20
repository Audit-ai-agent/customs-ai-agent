from customs_ai_agent.runner import run_review
from customs_ai_agent.schemas.models import ReviewRunRequest


def _run(scenario_id: str):
    return run_review(ReviewRunRequest(scenario_id=scenario_id))


def _issue_codes(state):
    return {issue["issue_code"] for issue in state["validation_issues"]}


def test_happy_path_completes():
    state = _run("happy_path")
    assert state["final_status"] == "completed"
    assert state["requires_human_review"] is False
    assert state["validation_issues"] == []
    assert state["report_json"]["overall_status"] == "completed"


def test_invoice_total_mismatch_is_detected():
    state = _run("invoice_total_mismatch")
    assert "INV_TOTAL_MISMATCH" in _issue_codes(state)
    assert state["final_status"] == "needs_review"
    assert state["requires_human_review"] is True


def test_quantity_mismatch_is_detected():
    state = _run("quantity_mismatch")
    assert "QTY_MISMATCH_INVOICE_PACKING" in _issue_codes(state)
    assert state["final_status"] == "needs_review"


def test_gross_weight_mismatch_is_detected():
    state = _run("gross_weight_mismatch")
    assert "GROSS_WEIGHT_MISMATCH" in _issue_codes(state)
    assert state["final_status"] == "needs_review"


def test_origin_mismatch_blocks_case():
    state = _run("origin_mismatch_blocked")
    assert "ORIGIN_MISMATCH" in _issue_codes(state)
    assert state["final_status"] == "blocked"
    assert state["requires_human_review"] is True


def test_missing_hs_code_needs_review():
    state = _run("missing_hs_code")
    assert "HS_CODE_MISSING" in _issue_codes(state)
    assert state["final_status"] == "needs_review"


def test_low_confidence_self_correction_runs_once():
    state = _run("low_confidence_self_correction")
    assert state["retry_count"] == 1
    assert state["final_status"] == "needs_review"
    assert "GROSS_WEIGHT_MISMATCH" not in _issue_codes(state)
    correction_events = [
        event for event in state["audit_trail"] if event["stage"] == "self_correct_extraction"
    ]
    assert len(correction_events) == 1
    assert correction_events[0]["metadata"]["changed_fields"] == 1


def test_low_evidence_routes_to_review_without_unsupported_finding():
    state = _run("low_evidence_needs_review")
    assert state["final_status"] == "needs_review"
    assert state["requires_human_review"] is True
    assert state["rag_evidence"]
    assert state["regulatory_findings"] == []
