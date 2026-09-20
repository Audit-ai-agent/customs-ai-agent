"""Admission-boundary checks for the child-run parent_document_analyses field."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import SecretStr

from customs_ai_agent.service.errors import ServiceHttpError
from customs_ai_agent.service.input import inspect_text_request
from customs_ai_agent.service.settings import ServiceSettings

_ANALYSIS: dict[str, Any] = {
    "document_id": "invoice",
    "filename": "invoice.txt",
    "document_type": "invoice",
    "extraction_status": "complete",
    "facts": [],
    "line_items": [{"item_id": "invoice:item:1", "facts": []}],
    "warnings": [],
}
_CHILD: dict[str, Any] = {
    "parent_review_run_id": "run-parent",
    "applied_correction_proposal_ids": ["proposal-1"],
    "human_corrections": [
        {
            "correction_proposal_id": "proposal-1",
            "document_id": "invoice",
            "field_name": "quantity",
            "new_value": "2 PCS",
            "reason": "담당자가 원본을 대조했습니다.",
            "source_kind": "human_provided",
        }
    ],
}


def _payload(**extra: Any) -> bytes:
    body: dict[str, Any] = {
        "case_id": "case-input",
        "review_run_id": "run-input",
        "review_as_of": "2026-08-05",
        "trade_type": "import",
        "jurisdiction": "KR",
        "documents": [
            {
                "document_id": "invoice",
                "filename": "invoice.txt",
                "content_text": "COMMERCIAL INVOICE",
            }
        ],
        **extra,
    }
    return json.dumps(body, ensure_ascii=False).encode("utf-8")


def _settings() -> ServiceSettings:
    token = "Abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJK"
    return ServiceSettings(
        bearer_token=SecretStr(token),
        evidence_base_url="http://127.0.0.1:18083",
        evidence_service_token=SecretStr(token),
    )


def test_admission_accepts_child_parent_document_analyses() -> None:
    inspected = inspect_text_request(
        _payload(**_CHILD, parent_document_analyses=[_ANALYSIS]), _settings()
    )

    assert inspected.review_run_id == "run-input"


@pytest.mark.parametrize(
    "analyses",
    [
        [{**_ANALYSIS, "document_id": ""}],
        [{**_ANALYSIS, "facts": [{"fact_id": f"f{index}"} for index in range(65)]}],
        [{**_ANALYSIS, "line_items": [{"item_id": "x", "facts": []}] * 101}],
        [{**_ANALYSIS, "line_items": [{"item_id": "x", "facts": [{}] * 33}]}],
        [_ANALYSIS] * 21,
        ["not-an-object"],
        {"document_id": "invoice"},
    ],
)
def test_admission_rejects_unbounded_parent_document_analyses(analyses: Any) -> None:
    with pytest.raises(ServiceHttpError):
        inspect_text_request(_payload(**_CHILD, parent_document_analyses=analyses), _settings())


def test_admission_rejects_parent_document_analyses_without_a_parent_run() -> None:
    with pytest.raises(ServiceHttpError):
        inspect_text_request(_payload(parent_document_analyses=[_ANALYSIS]), _settings())
