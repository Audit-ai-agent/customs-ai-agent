from __future__ import annotations

import json

from customs_ai_agent.service.input import TEXT_REVIEW_REQUEST_SCHEMA
from tools.verify_service_openapi import (
    LOCK_PATH,
    canonical_openapi,
    verify_openapi_lock,
)


def test_service_openapi_matches_frozen_lock() -> None:
    assert (
        verify_openapi_lock() == "c7b4f174e80e1594c092bc9e29d23fed87ca254e5968177eccca6d0219625859"
    )


def test_text_service_contract_requires_reproducibility_and_excludes_visual_fields() -> None:
    assert set(TEXT_REVIEW_REQUEST_SCHEMA["required"]) >= {
        "review_as_of",
        "review_run_id",
    }
    document = TEXT_REVIEW_REQUEST_SCHEMA["properties"]["documents"]["items"]
    assert "image_pages" not in document["properties"]
    assert "pdf_pages" not in document["properties"]
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert lock["paths"]["/internal/agent/v1/reviews"] == ["post"]


def test_openapi_declares_bearer_auth_and_exact_v2_result_schema() -> None:
    schema, _encoded = canonical_openapi()
    assert schema["components"]["securitySchemes"]["ServiceBearer"] == {
        "type": "http",
        "scheme": "bearer",
        "bearerFormat": "opaque",
        "description": "High-entropy internal service credential.",
    }
    operation = schema["paths"]["/internal/agent/v1/reviews"]["post"]
    assert operation["security"] == [{"ServiceBearer": []}]
    result = operation["responses"]["200"]["content"]["application/json"]["schema"]
    assert result["$id"] == "https://schemas.customs.local/agent/v2/review-result.schema.json"
    for path in (
        "/internal/agent/v1/reviews",
        "/internal/agent/v2/reviews",
        "/internal/agent/v2/revisions",
        "/internal/agent/v2/answers",
    ):
        pre_execution_header = schema["paths"][path]["post"]["responses"]["503"]["headers"][
            "X-Agent-Execution-State"
        ]
        assert pre_execution_header["schema"] == {
            "type": "string",
            "enum": ["not-started"],
        }


def test_openapi_adds_revision_experience_without_changing_v2_revision_wire() -> None:
    schema, _encoded = canonical_openapi()
    legacy = schema["paths"]["/internal/agent/v2/revisions"]["post"]
    enhanced = schema["paths"]["/internal/agent/v3/revisions"]["post"]

    assert enhanced["requestBody"] == legacy["requestBody"]
    response_schema = enhanced["responses"]["200"]["content"]["application/json"]["schema"]
    assert response_schema["additionalProperties"] is False
    assert set(response_schema["properties"]) == {
        "schema_version",
        "parent_result_sha256",
        "applied_decision_ids",
        "experience_envelope",
    }
    assert set(response_schema["required"]) == {
        "parent_result_sha256",
        "applied_decision_ids",
        "experience_envelope",
    }
