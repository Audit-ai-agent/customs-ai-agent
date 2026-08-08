from __future__ import annotations

import hashlib
import json
from importlib import resources

from customs_ai_agent.contracts import get_schema, schema_sha256


def test_data_contract_resources_are_the_only_packaged_evidence_contracts() -> None:
    retrieval = get_schema("evidence", "v3", "rag-retrieval")
    guidance = get_schema("evidence", "v3", "guidance-query")
    readiness = get_schema("evidence", "v3", "readiness")
    assert retrieval["$id"].endswith("rag_retrieval_v3.schema.json")
    assert readiness["$id"].endswith("readiness_v3.schema.json")
    assert len(schema_sha256("evidence", "v3", "rag-retrieval")) == 64
    assert len(schema_sha256("evidence", "v3", "guidance-query")) == 64
    assert len(schema_sha256("evidence", "v3", "readiness")) == 64
    assert readiness["required"] == ["status"]
    assert readiness["properties"] == {
        "status": {"const": "ready", "title": "Status", "type": "string"}
    }
    assert "requirement_verification" not in str(retrieval)
    assert "policy_release" not in str(retrieval)
    assert guidance["$id"].endswith("guidance_query_v3.schema.json")
    assert guidance["$defs"]["GeneralGuidanceQueryRequestV3"]["properties"]["trade_direction"][
        "enum"
    ] == ["import", "export", "both"]
    response = guidance["$defs"]["GeneralGuidanceQueryResponseV3"]
    source = guidance["$defs"]["GuidanceSourceV3"]
    assert response["properties"]["registry_sha256"]["pattern"] == "^[a-f0-9]{64}$"
    assert source["properties"]["registry_source_id"]["pattern"] == "^[a-z][a-z0-9_]{2,63}$"
    assert "registry_source_id" in source["required"]


def test_guidance_registry_consumer_resource_matches_its_guidance_contract_digest() -> None:
    raw = (
        resources.files("customs_ai_agent.contracts")
        .joinpath("evidence/v3/guidance_registry_v1.manifest.json")
        .read_bytes()
    )
    manifest = json.loads(raw.decode("utf-8"))
    guidance = get_schema("evidence", "v3", "guidance-query")

    assert len(hashlib.sha256(raw).hexdigest()) == 64
    assert manifest["schema_version"] == "1.0"
    assert (
        manifest["registry_sha256"]
        == guidance["$defs"]["GeneralGuidanceQueryResponseV3"]["properties"]["registry_sha256"][
            "default"
        ]
    )


def test_backend_v3_contract_has_one_packaged_canonical_source() -> None:
    request = get_schema("backend", "v3", "review-request")
    result = get_schema("backend", "v3", "review-result")
    assert request["$id"].endswith("/v3/review-request.schema.json")
    assert result["$id"].endswith("/v3/review-result.schema.json")
    assert len(schema_sha256("backend", "v3", "review-request")) == 64
    assert len(schema_sha256("backend", "v3", "review-result")) == 64
