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
    assert schema_sha256("evidence", "v3", "rag-retrieval") == (
        "4bb8066c4836192844a1387c6197eaccd0d932a4b9f95e2cce7bb07301aff71d"
    )
    assert len(schema_sha256("evidence", "v3", "guidance-query")) == 64
    assert len(schema_sha256("evidence", "v3", "readiness")) == 64
    assert readiness["required"] == ["status"]
    assert readiness["properties"] == {
        "status": {"const": "ready", "title": "Status", "type": "string"}
    }
    assert "requirement_verification" not in str(retrieval)
    assert "policy_release" not in str(retrieval)
    evidence = retrieval["$defs"]["Evidence"]
    response = retrieval["$defs"]["RetrieveResponseV3"]
    assert evidence["properties"]["retrieval_methods"]["items"]["enum"] == [
        "exact_hs",
        "dense",
        "bm25",
        "graph",
        "rerank",
    ]
    assert response["properties"]["temporal_context"] == {
        "anyOf": [{"$ref": "#/$defs/TemporalContext"}, {"type": "null"}],
        "default": None,
    }
    assert retrieval["$defs"]["TemporalContext"]["required"] == [
        "law_as_of",
        "hsk_snapshot_as_of",
        "requirements_snapshot_as_of",
    ]
    assert guidance["$id"].endswith("guidance_query_v3.schema.json")
    assert guidance["$defs"]["GeneralGuidanceQueryRequestV3"]["properties"]["trade_direction"][
        "enum"
    ] == ["import", "export", "both"]
    assert "scope_hint" in guidance["$defs"]["GeneralGuidanceQueryRequestV3"]["properties"]
    response = guidance["$defs"]["GeneralGuidanceQueryResponseV3"]
    source = guidance["$defs"]["GuidanceSourceV3"]
    assert response["properties"]["registry_sha256"]["pattern"] == "^[a-f0-9]{64}$"
    assert source["properties"]["registry_source_id"]["pattern"] == "^[a-z][a-z0-9_]{2,63}$"
    assert "registry_source_id" in source["required"]
    assert "guidance_scope" in response["properties"]
    assert response["properties"]["registry_version"]["enum"] == ["1.0", "2.0"]


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


def test_v2_registry_consumer_resource_keeps_page_scope_provenance() -> None:
    raw = (
        resources.files("customs_ai_agent.contracts")
        .joinpath("evidence/v3/guidance_registry_v2.manifest.json")
        .read_bytes()
    )
    manifest = json.loads(raw.decode("utf-8"))

    assert hashlib.sha256(raw).hexdigest() == (
        "51375a77351e73a32763e65679099d9bf04190078956e28ed1161b3ec5e1b7bb"
    )
    assert manifest["schema_version"] == "2.0"
    assert manifest["registry_sha256"] == (
        "a450ae6286d9a12729c5459543a35fd9c1c3df2ff59977700f72887c3df6543e"
    )
    assert all(
        set(source)
        == {
            "authority",
            "availability",
            "directions",
            "jurisdiction",
            "official_uri",
            "scopes",
            "source_id",
            "source_kind",
            "title",
            "topics",
        }
        for source in manifest["sources"]
    )
    scope_pairs = {
        (topic["topic"], scope["scope"])
        for topic in manifest["topics"]
        for scope in topic["scopes"]
    }
    assert ("customs_clearance", "import_clearance_flow") in scope_pairs
    assert ("transport_documents", "air_waybill") in scope_pairs
    assert ("fta_procedure", "preference_procedure") in scope_pairs


def test_backend_v3_contract_has_one_packaged_canonical_source() -> None:
    request = get_schema("backend", "v3", "review-request")
    result = get_schema("backend", "v3", "review-result")
    assert request["$id"].endswith("/v3/review-request.schema.json")
    assert result["$id"].endswith("/v3/review-result.schema.json")
    assert len(schema_sha256("backend", "v3", "review-request")) == 64
    assert len(schema_sha256("backend", "v3", "review-result")) == 64
