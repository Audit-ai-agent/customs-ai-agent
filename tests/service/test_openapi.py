from __future__ import annotations

from pydantic import SecretStr

from customs_ai_agent.service.app import create_app
from customs_ai_agent.service.settings import ServiceSettings


def _settings() -> ServiceSettings:
    token = "Abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJK"
    return ServiceSettings(SecretStr(token), "http://127.0.0.1:18083", SecretStr(token))


def test_openapi_exposes_only_active_backend_surface() -> None:
    schema = create_app(_settings()).openapi()
    assert set(schema["paths"]) == {
        "/health/live",
        "/health/ready",
        "/internal/agent/v2/metadata",
        "/internal/agent/v2/reviews",
    }
    review = schema["paths"]["/internal/agent/v2/reviews"]["post"]
    request_schema = review["requestBody"]["content"]["application/json"]["schema"]
    response_schema = review["responses"]["200"]["content"]["application/json"]["schema"]
    assert request_schema["$id"].endswith("/v3/review-request.schema.json")
    assert response_schema["$id"].endswith("/v3/review-result.schema.json")
    assert set(request_schema["required"]) == {
        "case_id",
        "review_run_id",
        "review_as_of",
        "trade_type",
        "jurisdiction",
        "documents",
    }
    assert "runtime_release" not in response_schema["properties"]
    assert response_schema["properties"]["requires_human_review"] == {
        "const": True,
        "type": "boolean",
    }
