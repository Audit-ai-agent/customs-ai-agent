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
        verify_openapi_lock() == "1d094981c137d9e3ed3222344864ae8e7357b6d5bdaa92b5aa7930bb1ca4f9dd"
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
