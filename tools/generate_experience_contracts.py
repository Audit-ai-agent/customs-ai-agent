from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from customs_ai_agent.meaning_catalog import (
    meaning_catalog_payload,
    meaning_catalog_sha256,
)
from customs_ai_agent.schemas.experience import (
    AgentUatCapabilityAttestationV1,
    AssistantAnswerV2,
    CaseConversationRequestV2,
    ReviewExperienceEnvelopeV1,
    ReviewStageEventV1,
)


ROOT = Path(__file__).resolve().parents[1]
PACKAGE_TARGET = ROOT / "src" / "customs_ai_agent" / "contracts" / "schemas" / "service" / "v3"
PROVIDER_TARGET = ROOT / "contracts" / "agent" / "service" / "v3"
MODELS: dict[str, type[BaseModel]] = {
    "agent-uat-capability-v1.schema.json": AgentUatCapabilityAttestationV1,
    "assistant-answer-v2.schema.json": AssistantAnswerV2,
    "case-conversation-request-v2.schema.json": CaseConversationRequestV2,
    "review-experience-envelope-v1.schema.json": ReviewExperienceEnvelopeV1,
    "review-stage-event-v1.schema.json": ReviewStageEventV1,
}


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def main() -> int:
    manifest: dict[str, Any] = {
        "schema_version": "customs-agent-experience-contract-manifest/v1",
        "schemas": {},
    }
    for target in (PACKAGE_TARGET, PROVIDER_TARGET):
        target.mkdir(parents=True, exist_ok=True)
    for name, model in sorted(MODELS.items()):
        payload = model.model_json_schema(mode="validation")
        encoded = _canonical_bytes(payload)
        for target in (PACKAGE_TARGET, PROVIDER_TARGET):
            (target / name).write_bytes(encoded)
        manifest["schemas"][name] = {
            "canonical_sha256": hashlib.sha256(encoded[:-1]).hexdigest(),
            "file_sha256": hashlib.sha256(encoded).hexdigest(),
        }
    encoded_manifest = _canonical_bytes(manifest)
    for target in (PACKAGE_TARGET, PROVIDER_TARGET):
        (target / "manifest.json").write_bytes(encoded_manifest)
    meaning_payload = meaning_catalog_payload()
    meaning_version = str(meaning_payload["version"])
    version_suffix = meaning_version.rsplit("/", 1)[-1]
    current_catalog_name = f"meaning-catalog-{version_suffix}.json"
    if current_catalog_name == "meaning-catalog-v1.json":
        raise RuntimeError("the enhanced meaning catalog must not overwrite immutable v1")
    legacy_catalog_name = "meaning-catalog-v1.json"
    legacy_catalog = (PROVIDER_TARGET / legacy_catalog_name).read_bytes()
    if legacy_catalog != (PACKAGE_TARGET / legacy_catalog_name).read_bytes():
        raise RuntimeError("provider and packaged legacy meaning catalog drift")
    legacy_payload = json.loads(legacy_catalog)
    if legacy_payload.get("version") != "customs-user-meaning-catalog/v1":
        raise RuntimeError("immutable meaning catalog v1 was modified")
    meaning_catalog = _canonical_bytes(meaning_payload)
    if hashlib.sha256(meaning_catalog[:-1]).hexdigest() != meaning_catalog_sha256():
        raise RuntimeError("meaning catalog canonical identity drift")
    asset_manifest = {
        "schema_version": "customs-agent-experience-asset-manifest/v1",
        "assets": {
            legacy_catalog_name: {
                "canonical_sha256": hashlib.sha256(legacy_catalog.rstrip(b"\n")).hexdigest(),
                "file_sha256": hashlib.sha256(legacy_catalog).hexdigest(),
                "media_type": "application/json",
            },
            current_catalog_name: {
                "canonical_sha256": meaning_catalog_sha256(),
                "file_sha256": hashlib.sha256(meaning_catalog).hexdigest(),
                "media_type": "application/json",
            },
        },
    }
    encoded_asset_manifest = _canonical_bytes(asset_manifest)
    for target in (PACKAGE_TARGET, PROVIDER_TARGET):
        (target / legacy_catalog_name).write_bytes(legacy_catalog)
        (target / current_catalog_name).write_bytes(meaning_catalog)
        (target / "asset-manifest.json").write_bytes(encoded_asset_manifest)
    print(encoded_manifest.decode("utf-8").strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
