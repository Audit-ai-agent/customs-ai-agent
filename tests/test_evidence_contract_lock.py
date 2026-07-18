from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from jsonschema import Draft202012Validator, FormatChecker

from customs_ai_agent.schemas.models import (
    EvidenceRetrievalRequest,
    EvidenceRetrievalResponse,
    RetrievalItemResult,
)


ROOT = Path(__file__).parents[1]
LOCK_PATH = ROOT / "contracts" / "evidence" / "v1.lock.json"


def _locked_contract() -> tuple[dict, dict]:
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    schema_path = ROOT / lock["consumer_snapshot_path"]
    schema_bytes = schema_path.read_bytes()
    assert hashlib.sha256(schema_bytes).hexdigest() == lock["sha256"]
    return lock, json.loads(schema_bytes)


def test_provider_owned_contract_snapshot_matches_lock() -> None:
    lock, schema = _locked_contract()

    assert lock["canonical_owner"] == "customs-data-platform"
    assert lock["consumer"] == "customs-ai-agent"
    assert lock["schema_version"] == "1.0"
    assert re.fullmatch(r"evidence-api-v\d+\.\d+\.\d+", lock["provider_release"])
    assert re.fullmatch(r"[0-9a-f]{40}", lock["provider_revision"])
    assert schema["$defs"]["RetrieveRequest"]["properties"]["schema_version"] == {"const": "1.0"}


def test_agent_wire_examples_validate_against_locked_provider_schema() -> None:
    _, schema = _locked_contract()
    validator = Draft202012Validator(schema, format_checker=FormatChecker())
    request = EvidenceRetrievalRequest(
        request_id="contract-request-1",
        case_id="contract-case-1",
        review_run_id="contract-run-1",
        review_as_of="2026-07-10",
        jurisdiction="KR",
        trade_type="import",
        items=[
            {
                "item_id": "contract-item-1",
                "description": "Machine screw",
                "hs_code": "7318151000",
                "origin_country": "NO",
                "destination_country": "KR",
            }
        ],
    )
    response = EvidenceRetrievalResponse(
        request_id=request.request_id,
        status="ok",
        knowledge_snapshot_id="contract-snapshot-1",
        results=[
            RetrievalItemResult(
                item_id="contract-item-1",
                hs_resolution="exact",
                resolved_hsk10="7318151000",
                candidate_hsk10s=["7318151000"],
                coverage="complete",
                requirements_found=False,
                evidence=[],
                errors=[],
            )
        ],
        errors=[],
    )

    validator.validate(request.model_dump(mode="json"))
    validator.validate(response.model_dump(mode="json"))
