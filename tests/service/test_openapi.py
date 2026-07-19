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
        verify_openapi_lock() == "c56c0bb6ebeb9bf74e7a274fd3670a5849793e09c6d34c2cd6c3e6375fcad803"
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
