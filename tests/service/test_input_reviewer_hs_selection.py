"""Admission-boundary checks for the optional reviewer HS selection field."""

from __future__ import annotations

import json
from typing import Any

import pytest
from pydantic import SecretStr

from customs_ai_agent.service.errors import ServiceHttpError
from customs_ai_agent.service.input import inspect_text_request
from customs_ai_agent.service.settings import ServiceSettings


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


def test_admission_accepts_a_valid_reviewer_hs_selection() -> None:
    inspected = inspect_text_request(
        _payload(reviewer_hs_selection={"code": "6109101000", "source_kind": "human_provided"}),
        _settings(),
    )
    assert inspected.review_run_id == "run-input"


def test_admission_accepts_a_request_without_reviewer_hs_selection() -> None:
    inspected = inspect_text_request(_payload(), _settings())
    assert inspected.review_run_id == "run-input"


@pytest.mark.parametrize(
    "selection",
    [
        {"code": "123456", "source_kind": "human_provided"},
        {"code": "6109101000"},
        {"code": "6109101000", "source_kind": "document"},
        {"code": "6109101000", "source_kind": "human_provided", "extra": "x"},
        "6109101000",
        [],
    ],
)
def test_admission_rejects_a_malformed_reviewer_hs_selection(selection: Any) -> None:
    with pytest.raises(ServiceHttpError) as caught:
        inspect_text_request(_payload(reviewer_hs_selection=selection), _settings())
    assert caught.value.status_code == 422


def test_admission_accepts_per_item_reviewer_hs_selections() -> None:
    inspected = inspect_text_request(
        _payload(
            reviewer_hs_selection={"code": "6109101000", "source_kind": "human_provided"},
            reviewer_hs_selections=[
                {
                    "code": "6109101000",
                    "source_kind": "human_provided",
                    "item_id": "item-a",
                    "description": "MEN'S COTTON T-SHIRTS",
                },
                {"code": "0303540000", "source_kind": "human_provided", "item_id": None},
            ],
        ),
        _settings(),
    )
    assert inspected.review_run_id == "run-input"


@pytest.mark.parametrize(
    "selections",
    [
        {"code": "6109101000", "source_kind": "human_provided"},
        [{"code": "6109101000", "source_kind": "human_provided", "item_id": ""}],
        [{"code": "6109101000", "source_kind": "human_provided", "item_id": "x" * 65}],
        [{"code": "6109101000", "source_kind": "human_provided", "description": "y" * 301}],
        [{"code": "6109101000", "source_kind": "human_provided", "reason": "not on the wire"}],
        [{"code": "6109101000", "source_kind": "human_provided"}] * 21,
    ],
)
def test_admission_rejects_malformed_reviewer_hs_selections(selections: Any) -> None:
    with pytest.raises(ServiceHttpError) as caught:
        inspect_text_request(_payload(reviewer_hs_selections=selections), _settings())
    assert caught.value.status_code == 422
