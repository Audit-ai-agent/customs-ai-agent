from __future__ import annotations

import json
from typing import Any

from fastapi.testclient import TestClient
from pydantic import SecretStr

from customs_ai_agent.schemas.v3 import HsCandidatesRequestV1
from customs_ai_agent.service import hs_candidates
from customs_ai_agent.service.app import create_app
from customs_ai_agent.service.settings import ServiceSettings

_TOKEN = "Abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJK"


def _settings() -> ServiceSettings:
    return ServiceSettings(SecretStr(_TOKEN), "http://127.0.0.1:18083", SecretStr(_TOKEN))


def _request(**overrides: Any) -> HsCandidatesRequestV1:
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "case_id": "case-hs",
        "documents": [
            {
                "document_id": "invoice",
                "document_type": "CI",
                "content_text": "COMMERCIAL INVOICE\nDescription of Goods\nQuantity\nMEN'S COTTON T-SHIRTS\n64,444 PCS",
            }
        ],
        "reference_candidates": [
            {
                "code": "6109101000",
                "name_ko": "티셔츠",
                "name_en": "T-shirts",
                "ancestors": ["제61류", "면"],
            },
            {"code": "6105100000", "name_ko": "셔츠", "name_en": "Shirts", "ancestors": []},
        ],
        "max_picks": 3,
    }
    payload.update(overrides)
    return HsCandidatesRequestV1.model_validate(payload)


def test_validate_answer_keeps_only_reference_codes_and_known_documents() -> None:
    request = _request()
    items, picks = hs_candidates.validate_answer(
        {
            "items": [
                {
                    "description": "MEN'S COTTON T-SHIRTS",
                    "document_ids": ["invoice", "ghost"],
                    "cited_hs_codes": ["6109.10", "12"],
                    "search_terms": ["티셔츠", " 면 ", "티셔츠", 7, "", "x" * 81, "t-shirts"],
                },
                {"description": ""},
                "junk",
            ],
            "picks": [
                {"code": "6109101000", "rank": 1, "confidence": 0.92, "reason": "면 티셔츠"},
                {"code": "9999999999", "rank": 2, "confidence": 0.5, "reason": "invented"},
                {"code": "6109101000", "rank": 3, "confidence": 0.1, "reason": "duplicate"},
                {"code": "6105100000", "rank": 4, "confidence": 1.7, "reason": "out of range"},
                {"code": "6105100000", "rank": 5, "confidence": 0.4, "reason": "셔츠"},
            ],
        },
        request,
    )
    assert [item.description for item in items] == ["MEN'S COTTON T-SHIRTS"]
    assert items[0].document_ids == ["invoice"]
    assert items[0].cited_hs_codes == ["610910"]
    # Lookup terms are trimmed and de-duplicated; non-strings and over-long ones are dropped.
    assert items[0].search_terms == ["티셔츠", "면", "t-shirts"]
    assert [(pick.code, pick.rank, pick.confidence) for pick in picks] == [
        ("6109101000", 1, 0.92),
        ("6105100000", 2, 0.4),
    ]


def test_validate_answer_reads_items_without_search_terms_or_reference_codes() -> None:
    # The extraction-only round sends no reference codes: items still come
    # back, an older answer shape without search_terms is accepted, and no
    # pick can be honoured because there is nothing to pick from.
    request = _request(reference_candidates=[])
    items, picks = hs_candidates.validate_answer(
        {
            "items": [{"description": "LED MONITOR 24 INCH", "document_ids": ["invoice"]}],
            "picks": [{"code": "8528726000", "rank": 1, "confidence": 0.9, "reason": "모니터"}],
        },
        request,
    )
    assert [(item.description, item.search_terms) for item in items] == [
        ("LED MONITOR 24 INCH", [])
    ]
    assert picks == []


def test_validate_answer_drops_a_malformed_answer_entirely() -> None:
    assert hs_candidates.validate_answer("not json", _request()) == ([], [])
    assert hs_candidates.validate_answer({"picks": "x", "items": None}, _request()) == ([], [])


def test_request_payload_never_forwards_case_id() -> None:
    payload = hs_candidates.request_payload(_request())
    assert set(payload) == {"documents", "reference_candidates", "max_picks"}
    assert payload["reference_candidates"][0]["code"] == "6109101000"


def test_local_development_route_answers_deterministically_without_a_model() -> None:
    with TestClient(create_app(_settings())) as client:
        _exercise_route(client)


def _exercise_route(client: TestClient) -> None:
    headers = {"Authorization": f"Bearer {_TOKEN}", "X-Customs-Service-Mode": "local-development"}
    response = client.post(
        "/internal/agent/v3/hs-candidates",
        content=json.dumps(_request().model_dump(mode="json")).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json"},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "deterministic"
    assert body["picks"] == [] and body["items"] == []
    assert body["engine"]["kind"] == "deterministic"

    rejected = client.post(
        "/internal/agent/v3/hs-candidates",
        content=json.dumps(
            {"schema_version": "1.0", "case_id": "c", "documents": [], "extra": 1}
        ).encode("utf-8"),
        headers={**headers, "Content-Type": "application/json"},
    )
    assert rejected.status_code == 422
