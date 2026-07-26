from __future__ import annotations

from pathlib import Path

import pytest

from customs_ai_agent.schemas.v3 import ServiceReviewRunRequestV3
from tools.agent_execution_inspector.server import (
    InspectorExecutionError,
    _matching_scenario,
    evaluate_scenario_gold,
    load_scenario_catalog,
    load_scenario_gold,
)


INSPECTOR_DIR = Path(__file__).parents[1] / "tools" / "agent_execution_inspector"


def _catalog_and_gold():
    catalog = load_scenario_catalog(
        INSPECTOR_DIR / "scenario_catalog.json",
        INSPECTOR_DIR / "scenario_catalog.schema.json",
    )
    gold = load_scenario_gold(
        INSPECTOR_DIR / "scenario_gold.json",
        INSPECTOR_DIR / "scenario_gold.schema.json",
        catalog=catalog,
    )
    return catalog, gold


def _request_for(scenario: dict) -> ServiceReviewRunRequestV3:
    return ServiceReviewRunRequestV3.model_validate(
        {
            "case_id": scenario["case_id"],
            "documents": scenario["documents"],
            "llm_mode": "llm",
            "review_run_id": "inspector_machine_assertion_test",
            "review_as_of": "2026-07-22",
            "trade_type": scenario["trade_type"],
            "document_language": scenario["document_language"],
            "jurisdiction": "KR",
            "uploaded_requirement_document_codes": scenario["uploaded_requirement_document_codes"],
            "supporting_documents": scenario["supporting_documents"],
        }
    )


def _clean_envelope(scenario: dict) -> dict:
    document_results = []
    for document in scenario["documents"]:
        document_results.append(
            {
                "document_id": document["document_id"],
                "document_type": document["declared_type"],
                "extraction_status": "completed",
                "fields": [
                    {
                        "name": "test_field",
                        "normalized_value": "normalized",
                        "source": {
                            "source_modality": "text",
                            "text_start": 0,
                            "text_end": 1,
                        },
                    }
                ],
                "line_items": [],
            }
        )
    return {
        "pairing": {"document_set_status": "deterministic"},
        "result": {
            "document_results": document_results,
            "model_errors": [],
            "validation_issues": [],
            "regulatory_findings": [],
            "completion_eligibility": {
                "unmet_reason_codes": ["supporting_document_verification_incomplete"]
            },
            "retrieval_status": "ok",
            "retrieval_coverage": "complete",
            "assessment": "no_issue_detected",
            "review_disposition": "human_review_required",
            "narrative_report": {
                "summary": [
                    {
                        "material": True,
                        "citations": [{"kind": "outcome", "reference_id": "outcome-1"}],
                    }
                ],
                "document_consistency": [],
                "regulatory_review": [],
                "required_actions": [],
            },
        },
    }


def test_gold_has_exact_catalog_coverage_and_both_positive_controls() -> None:
    catalog, gold = _catalog_and_gold()
    assert set(gold) == {scenario["id"] for scenario in catalog["scenarios"]}
    assert gold["clean_text_diagnostic"]["control_type"] == "diagnostic_positive_control"
    assert (
        gold["clean_text_production_candidate"]["control_type"]
        == "production_candidate_positive_control"
    )
    assert all(
        control["review_disposition"] == "human_review_required" for control in gold.values()
    )


def test_machine_assertions_pass_clean_control_without_synthesizing_production() -> None:
    catalog, gold = _catalog_and_gold()
    scenario = next(item for item in catalog["scenarios"] if item["id"] == "clean_text_diagnostic")
    evaluated = evaluate_scenario_gold(
        scenario_id=scenario["id"],
        gold=gold[scenario["id"]],
        envelope=_clean_envelope(scenario),
    )
    assert evaluated["status"] == "passed"
    assert evaluated["release_evidence"] is False
    assert {item["layer"] for item in evaluated["checks"]} == {
        "classification",
        "extraction",
        "normalization",
        "pairing",
        "validation",
        "supporting_document_verification",
        "evidence",
        "finding",
        "outcome",
        "narrative_citation",
    }


def test_defect_gold_fails_when_required_defect_is_not_observed() -> None:
    catalog, gold = _catalog_and_gold()
    scenario = next(item for item in catalog["scenarios"] if item["id"] == "quantity_mismatch")
    evaluated = evaluate_scenario_gold(
        scenario_id=scenario["id"],
        gold=gold[scenario["id"]],
        envelope=_clean_envelope(scenario),
    )
    assert evaluated["status"] == "failed"
    assert "required_issue_codes" in evaluated["failed_check_ids"]


def test_scenario_assertions_cannot_be_attached_to_modified_input() -> None:
    catalog, _gold = _catalog_and_gold()
    scenario = next(item for item in catalog["scenarios"] if item["id"] == "clean_text_diagnostic")
    request = _request_for(scenario)
    assert _matching_scenario(scenario["id"], request, catalog)["id"] == scenario["id"]
    modified = request.model_copy(deep=True)
    modified.documents[0].content_text += "MODIFIED"
    with pytest.raises(InspectorExecutionError, match="scenario_payload_mismatch"):
        _matching_scenario(scenario["id"], modified, catalog)
