from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

import evals.run_workflow_boundary_validation as workflow_eval
from evals.run_workflow_boundary_validation import (
    WorkflowBoundaryEvaluationError,
    evaluate,
)


ROOT = Path(__file__).parents[1]
DESCRIPTOR_PATH = (
    ROOT / "evals" / "trade_document_triplets" / "v1" / "workflow-boundary.descriptor.json"
)
SCHEMA_PATH = (
    ROOT
    / "evals"
    / "trade_document_triplets"
    / "v1"
    / "schemas"
    / "workflow-boundary-evaluation.schema.json"
)


def _write(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def test_workflow_descriptor_is_schema_valid_and_all_boundaries_fail_closed():
    descriptor = json.loads(DESCRIPTOR_PATH.read_text(encoding="utf-8"))
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    Draft202012Validator(schema).validate(descriptor)
    exit_code, report = evaluate(DESCRIPTOR_PATH)

    assert exit_code == 0
    assert report["status"] == "PASS"
    assert report["case_count"] == 6
    assert report["automatic_completed_count"] == 0
    assert report["operator_ids"] == list(workflow_eval.REQUIRED_OPERATOR_IDS)
    assert all(result["pass"] for result in report["results"])


def test_workflow_evaluator_rejects_a_self_consistent_missing_boundary(tmp_path, monkeypatch):
    descriptor = json.loads(DESCRIPTOR_PATH.read_text(encoding="utf-8"))
    tampered = copy.deepcopy(descriptor)
    tampered["scenarios"].pop()
    path = tmp_path / "descriptor.json"
    _write(path, tampered)
    monkeypatch.setattr(
        workflow_eval,
        "CANONICAL_DESCRIPTOR_SHA256",
        hashlib.sha256(path.read_bytes()).hexdigest(),
    )

    with pytest.raises(
        WorkflowBoundaryEvaluationError,
        match="every required boundary",
    ):
        evaluate(path)


def test_workflow_evaluator_rejects_descriptor_hash_tampering(tmp_path):
    path = tmp_path / "descriptor.json"
    path.write_bytes(DESCRIPTOR_PATH.read_bytes() + b"\n")

    with pytest.raises(WorkflowBoundaryEvaluationError, match="canonical release hash"):
        evaluate(path)
