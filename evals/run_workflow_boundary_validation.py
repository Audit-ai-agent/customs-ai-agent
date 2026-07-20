"""Evaluate fail-closed workflow boundaries from a hash-pinned descriptor.

This is deliberately separate from business-field delta scoring.  It checks
that incomplete document sets, OCR conflicts, and non-complete Evidence
responses can never produce an automatic ``completed`` outcome.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from decimal import Decimal
from pathlib import Path
from typing import Any, Callable

from customs_ai_agent.nodes.pair import pair_documents
from customs_ai_agent.nodes.validate import validate_cross_documents
from customs_ai_agent.policies.outcome import (
    REQUIRED_DOCUMENT_FIELDS,
    REQUIRED_LINE_ITEM_FIELDS,
    determine_outcome,
)


REPORT_SCHEMA = "workflow-boundary-evaluation-report/v1"
DESCRIPTOR_SCHEMA = "trade-document-workflow-boundary-descriptor/v1"
CANONICAL_DESCRIPTOR_SHA256 = "5680801df1066a65a82cff4b425fb1410f5a393a30b376fcf084d2277fc6e9e6"
REQUIRED_OPERATOR_IDS = (
    "required_document_missing",
    "duplicate_document_type",
    "ocr_conflict_unresolved",
    "evidence_partial",
    "evidence_failed",
    "evidence_ambiguous",
)


class WorkflowBoundaryEvaluationError(ValueError):
    """The descriptor or one generated scenario violated the evaluation contract."""


def _sha256(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError as exc:
        raise WorkflowBoundaryEvaluationError("workflow descriptor is not readable") from exc


def _load_descriptor(path: Path) -> dict[str, Any]:
    digest = _sha256(path)
    if digest != CANONICAL_DESCRIPTOR_SHA256:
        raise WorkflowBoundaryEvaluationError(
            "workflow descriptor hash is not the canonical release hash"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise WorkflowBoundaryEvaluationError("workflow descriptor is not readable JSON") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != DESCRIPTOR_SCHEMA
        or value.get("descriptor_id") != "trade-document-workflow-boundaries/evaluation-v2"
        or value.get("dataset_id") != "trade_document_triplets"
        or value.get("dataset_version") != "source-v1"
        or value.get("evaluation_version") != "evaluation-v2"
    ):
        raise WorkflowBoundaryEvaluationError("workflow descriptor identity is invalid")
    scenarios = value.get("scenarios")
    if not isinstance(scenarios, list):
        raise WorkflowBoundaryEvaluationError("workflow descriptor scenarios are invalid")
    operator_ids = [
        scenario.get("operator_id") if isinstance(scenario, dict) else None
        for scenario in scenarios
    ]
    if operator_ids != list(REQUIRED_OPERATOR_IDS):
        raise WorkflowBoundaryEvaluationError(
            "workflow descriptor does not cover every required boundary exactly once"
        )
    for scenario in scenarios:
        if (
            not isinstance(scenario, dict)
            or set(scenario)
            != {
                "operator_id",
                "boundary",
                "expected_outcome",
                "forbidden_outcomes",
                "expected_issue_codes",
                "expected_blocker_codes",
            }
            or scenario.get("expected_outcome") not in {"needs_review", "blocked"}
            or "completed" not in (scenario.get("forbidden_outcomes") or [])
            or not isinstance(scenario.get("expected_issue_codes"), list)
            or not isinstance(scenario.get("expected_blocker_codes"), list)
            or not all(
                isinstance(code, str) and re.fullmatch(r"[A-Za-z0-9_:.+-]+", code)
                for code in [
                    *scenario["expected_issue_codes"],
                    *scenario["expected_blocker_codes"],
                ]
            )
        ):
            raise WorkflowBoundaryEvaluationError(
                "workflow descriptor scenario contract is invalid"
            )
    return value


def _field(value: Any) -> dict[str, Any]:
    return {"value": value}


def _base_state() -> dict[str, Any]:
    invoice: dict[str, Any] = {
        "document_id": "doc_inv",
        "document_type": "invoice",
        "fields": {
            "invoice_no": "INV-CONTROLLED",
            "seller": "CONTROLLED SELLER",
            "buyer": "CONTROLLED BUYER",
            "currency": "USD",
            "total_amount": Decimal("10"),
        },
        "line_items": [
            {
                "item_id": "invoice-item-1",
                "description": "CONTROLLED ITEM",
                "hs_code": "7318151000",
                "quantity": Decimal("1"),
                "unit_price": Decimal("10"),
                "amount": Decimal("10"),
            }
        ],
        "extraction_confidence": 1.0,
    }
    packing: dict[str, Any] = {
        "document_id": "doc_pl",
        "document_type": "packing_list",
        "fields": {
            "packing_no": "PL-CONTROLLED",
            "seller": "CONTROLLED SELLER",
            "buyer": "CONTROLLED BUYER",
            "packages": "1 CTN",
        },
        "line_items": [
            {
                "item_id": "packing-item-1",
                "description": "CONTROLLED ITEM",
                "quantity": Decimal("1"),
                "gross_weight_kg": Decimal("1"),
                "package_count": 1,
            }
        ],
        "extraction_confidence": 1.0,
    }
    bill: dict[str, Any] = {
        "document_id": "doc_bl",
        "document_type": "bill_of_lading",
        "fields": {
            "bl_no": "BL-CONTROLLED",
            "shipper": "CONTROLLED SELLER",
            "consignee": "CONTROLLED BUYER",
            "packages": "1 CTN",
            "gross_weight_kg": Decimal("1"),
            "freight_prepaid": "Y",
            "freight_collect": "",
        },
        "line_items": [],
        "extraction_confidence": 1.0,
    }
    extracted_documents: dict[str, dict[str, Any]] = {
        "doc_inv": {
            **copy.deepcopy(invoice),
            "extraction_status": "completed",
            "fields": {name: _field(value) for name, value in invoice["fields"].items()},
            "line_items": [
                {
                    name: _field(value)
                    for name, value in invoice["line_items"][0].items()
                    if name != "item_id"
                }
            ],
        },
        "doc_pl": {
            **copy.deepcopy(packing),
            "extraction_status": "completed",
            "fields": {name: _field(value) for name, value in packing["fields"].items()},
            "line_items": [
                {
                    name: _field(value)
                    for name, value in packing["line_items"][0].items()
                    if name != "item_id"
                }
            ],
        },
        "doc_bl": {
            **copy.deepcopy(bill),
            "extraction_status": "completed",
            "fields": {name: _field(value) for name, value in bill["fields"].items()},
            "line_items": [],
        },
    }
    for document in extracted_documents.values():
        document_type = str(document["document_type"])
        for field_name in REQUIRED_DOCUMENT_FIELDS.get(document_type, frozenset()):
            document["fields"].setdefault(field_name, _field("CONTROLLED"))
        required_line_fields = REQUIRED_LINE_ITEM_FIELDS.get(document_type)
        if required_line_fields is not None and document["line_items"]:
            for field_name in required_line_fields:
                document["line_items"][0].setdefault(field_name, _field("CONTROLLED"))
    return {
        "case_id": "workflow-boundary-controlled",
        "trade_type": "import",
        "llm_mode": "mock",
        "normalized_documents": {
            "doc_inv": invoice,
            "doc_pl": packing,
            "doc_bl": bill,
        },
        "extracted_documents": extracted_documents,
        "validation_issues": [],
        "regulatory_findings": [],
        "requires_human_review": False,
        "retry_count": 0,
        "retrieval_status": "ok",
        "retrieval_coverage": "complete",
        "knowledge_snapshot_id": "controlled-snapshot",
        "retrieval_errors": [],
        "retrieval_results": [
            {
                "item_id": "invoice-item-1",
                "hs_resolution": "exact",
                "coverage": "complete",
                "errors": [],
                "evidence": [],
                "requirements_found": False,
            }
        ],
        "audit_trail": [],
    }


def _document_set_scenario(operator_id: str) -> dict[str, Any]:
    state = _base_state()
    if operator_id == "required_document_missing":
        del state["normalized_documents"]["doc_bl"]
        del state["extracted_documents"]["doc_bl"]
    elif operator_id == "duplicate_document_type":
        duplicate = copy.deepcopy(state["normalized_documents"]["doc_inv"])
        duplicate["document_id"] = "doc_inv_duplicate"
        state["normalized_documents"]["doc_inv_duplicate"] = duplicate
        extracted = copy.deepcopy(state["extracted_documents"]["doc_inv"])
        extracted["document_id"] = "doc_inv_duplicate"
        state["extracted_documents"]["doc_inv_duplicate"] = extracted
    else:  # pragma: no cover - guarded by the dispatch table
        raise WorkflowBoundaryEvaluationError("unsupported document-set scenario")
    state.update(pair_documents(state))
    state.update(validate_cross_documents(state))
    return state


def _ocr_conflict_scenario() -> dict[str, Any]:
    state = _base_state()
    state["extracted_documents"]["doc_inv"]["extraction_status"] = "failed"
    state["requires_human_review"] = True
    return state


def _evidence_scenario(operator_id: str) -> dict[str, Any]:
    state = _base_state()
    result = state["retrieval_results"][0]
    if operator_id == "evidence_partial":
        state["retrieval_status"] = "partial"
        state["retrieval_coverage"] = "partial"
        result["coverage"] = "partial"
    elif operator_id == "evidence_failed":
        state["retrieval_status"] = "failed"
        state["retrieval_coverage"] = "unknown"
        state["knowledge_snapshot_id"] = None
        state["retrieval_errors"] = ["controlled_failure"]
        result["coverage"] = "unknown"
        result["errors"] = ["controlled_failure"]
    elif operator_id == "evidence_ambiguous":
        result["hs_resolution"] = "ambiguous"
    else:  # pragma: no cover - guarded by the dispatch table
        raise WorkflowBoundaryEvaluationError("unsupported Evidence scenario")
    return state


def _scenario_state(operator_id: str) -> dict[str, Any]:
    builders: dict[str, Callable[[], dict[str, Any]]] = {
        "required_document_missing": lambda: _document_set_scenario("required_document_missing"),
        "duplicate_document_type": lambda: _document_set_scenario("duplicate_document_type"),
        "ocr_conflict_unresolved": _ocr_conflict_scenario,
        "evidence_partial": lambda: _evidence_scenario("evidence_partial"),
        "evidence_failed": lambda: _evidence_scenario("evidence_failed"),
        "evidence_ambiguous": lambda: _evidence_scenario("evidence_ambiguous"),
    }
    try:
        return builders[operator_id]()
    except KeyError as exc:  # pragma: no cover - descriptor is validated first
        raise WorkflowBoundaryEvaluationError("unknown workflow operator") from exc


def evaluate(descriptor_path: Path) -> tuple[int, dict[str, Any]]:
    descriptor = _load_descriptor(descriptor_path)
    results: list[dict[str, Any]] = []
    for expected in descriptor["scenarios"]:
        operator_id = expected["operator_id"]
        state = _scenario_state(operator_id)
        outcome, requires_review, blockers = determine_outcome(state)
        issue_codes = sorted(
            {
                str(issue.get("issue_code"))
                for issue in state.get("validation_issues") or []
                if issue.get("issue_code")
            }
        )
        expected_issues = set(expected["expected_issue_codes"])
        expected_blockers = set(expected["expected_blocker_codes"])
        passed = (
            outcome == expected["expected_outcome"]
            and outcome not in expected["forbidden_outcomes"]
            and requires_review
            and expected_issues <= set(issue_codes)
            and expected_blockers <= set(blockers)
        )
        results.append(
            {
                "operator_id": operator_id,
                "boundary": expected["boundary"],
                "expected_outcome": expected["expected_outcome"],
                "actual_outcome": outcome,
                "expected_issue_codes": sorted(expected_issues),
                "observed_issue_codes": issue_codes,
                "expected_blocker_codes": sorted(expected_blockers),
                "observed_blocker_codes": blockers,
                "requires_human_review": requires_review,
                "pass": passed,
            }
        )
    passed = len(results) == len(REQUIRED_OPERATOR_IDS) and all(
        result["pass"] for result in results
    )
    report = {
        "schema_version": REPORT_SCHEMA,
        "scope": "WORKFLOW_BOUNDARY_FAIL_CLOSED",
        "status": "PASS" if passed else "FAIL",
        "descriptor_sha256": _sha256(descriptor_path),
        "operator_ids": list(REQUIRED_OPERATOR_IDS),
        "case_count": len(results),
        "automatic_completed_count": sum(
            result["actual_outcome"] == "completed" for result in results
        ),
        "results": results,
        "redaction": {
            "source_values": "NOT_USED",
            "absolute_paths": "OMITTED",
            "api_keys": "OMITTED",
        },
    }
    return (0 if passed else 1), report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate hash-pinned fail-closed workflow boundary scenarios."
    )
    parser.add_argument("--descriptor", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        exit_code, report = evaluate(args.descriptor)
    except WorkflowBoundaryEvaluationError as exc:
        print(
            json.dumps(
                {
                    "schema_version": REPORT_SCHEMA,
                    "scope": "WORKFLOW_BOUNDARY_FAIL_CLOSED",
                    "status": "FAIL",
                    "error": str(exc),
                },
                sort_keys=True,
            )
        )
        return 2
    payload = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(payload, encoding="utf-8")
    print(payload, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
