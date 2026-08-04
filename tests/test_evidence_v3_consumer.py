from __future__ import annotations

from customs_ai_agent.contracts import get_schema, schema_sha256


def test_data_contract_resources_are_the_only_packaged_evidence_contracts() -> None:
    retrieval = get_schema("evidence", "v3", "rag-retrieval")
    readiness = get_schema("evidence", "v3", "readiness")
    assert retrieval["$id"].endswith("rag_retrieval_v3.schema.json")
    assert readiness["$id"].endswith("readiness_v3.schema.json")
    assert len(schema_sha256("evidence", "v3", "rag-retrieval")) == 64
    assert len(schema_sha256("evidence", "v3", "readiness")) == 64
    assert readiness["required"] == ["status"]
    assert readiness["properties"] == {
        "status": {"const": "ready", "title": "Status", "type": "string"}
    }
    assert "requirement_verification" not in str(retrieval)
    assert "policy_release" not in str(retrieval)


def test_backend_v3_contract_has_one_packaged_canonical_source() -> None:
    request = get_schema("backend", "v3", "review-request")
    result = get_schema("backend", "v3", "review-result")
    assert request["$id"].endswith("/v3/review-request.schema.json")
    assert result["$id"].endswith("/v3/review-result.schema.json")
    assert len(schema_sha256("backend", "v3", "review-request")) == 64
    assert len(schema_sha256("backend", "v3", "review-result")) == 64
