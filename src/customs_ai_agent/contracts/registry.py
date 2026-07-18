from __future__ import annotations

import copy
import hashlib
import json
import os
from datetime import date
from importlib import resources
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from customs_ai_agent.schemas.llm import (
    ClaudeCropReadCandidate,
    ClaudeDocumentCandidate,
    ClaudeVisionDocumentCandidate,
)
from customs_ai_agent.schemas.models import (
    DocumentType,
    OcrLayoutLedger,
    ReviewRunRequest,
    ReviewRunRequestV2,
    ReviewRunResult,
)


SchemaScope = Literal["backend", "internal", "local"]
SchemaVersion = Literal["v1", "v2"]
SchemaKind = Literal[
    "review-request",
    "review-result",
    "ocr-layout-ledger",
    "claude-text-candidate",
    "claude-vision-candidate",
    "claude-crop-read-candidate",
    "review-manifest",
]
SchemaKey = tuple[SchemaScope, SchemaVersion, SchemaKind]

_RESOURCE_PATHS: dict[SchemaKey, str] = {
    ("backend", "v1", "review-request"): "schemas/backend/v1/review-request.schema.json",
    ("backend", "v1", "review-result"): "schemas/backend/v1/review-result.schema.json",
    ("backend", "v2", "review-request"): "schemas/backend/v2/review-request.schema.json",
    ("backend", "v2", "review-result"): "schemas/backend/v2/review-result.schema.json",
    (
        "internal",
        "v1",
        "ocr-layout-ledger",
    ): "schemas/internal/v1/ocr-layout-ledger.schema.json",
    (
        "internal",
        "v1",
        "claude-text-candidate",
    ): "schemas/internal/v1/claude-text-candidate.schema.json",
    (
        "internal",
        "v1",
        "claude-vision-candidate",
    ): "schemas/internal/v1/claude-vision-candidate.schema.json",
    (
        "internal",
        "v1",
        "claude-crop-read-candidate",
    ): "schemas/internal/v1/claude-crop-read-candidate.schema.json",
    ("local", "v1", "review-manifest"): "schemas/local/v1/review-manifest.schema.json",
}


class ContractSchemaError(ValueError):
    """Raised when an unknown or unavailable versioned schema is requested."""


class LocalManifestSourceV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["text", "image", "pdf"]
    path: str = Field(min_length=1, max_length=512)
    ocr_ledger: str | None = Field(default=None, min_length=1, max_length=512)

    @field_validator("path", "ocr_ledger")
    @classmethod
    def _relative_local_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        candidate = Path(value)
        if (
            "\x00" in value
            or candidate.is_absolute()
            or bool(candidate.drive)
            or ".." in candidate.parts
        ):
            raise ValueError("manifest source references must be safe relative paths")
        return value


class LocalManifestDocumentV1(BaseModel):
    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    filename: str = Field(min_length=1, max_length=255)
    declared_type: DocumentType | None = None
    requirement_document_codes: list[str] = Field(default_factory=list)
    source: LocalManifestSourceV1

    @field_validator("filename")
    @classmethod
    def _safe_filename_label(cls, value: str) -> str:
        if (
            value != Path(value).name
            or "/" in value
            or "\\" in value
            or any(ord(character) < 32 for character in value)
        ):
            raise ValueError("filename must be a plain display label")
        return value


class LocalReviewManifestV1(BaseModel):
    """Local file adapter contract; never the Backend Agent wire contract."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["customs-agent-local-review-manifest/v1"]
    contract_version: Literal["v1", "v2"]
    llm_mode: Literal["mock", "llm"]
    documents: list[LocalManifestDocumentV1] = Field(min_length=1, max_length=20)
    case_id: str | None = Field(default=None, min_length=1, max_length=128)
    review_run_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    review_as_of: date | None = None
    trade_type: Literal["import", "export"] = "import"
    jurisdiction: Literal["KR"] = "KR"
    uploaded_requirement_document_codes: list[str] = Field(default_factory=list)


def available_schemas() -> tuple[SchemaKey, ...]:
    return tuple(sorted(_RESOURCE_PATHS))


def _resource_path(
    scope: SchemaScope,
    version: SchemaVersion,
    kind: SchemaKind,
) -> str:
    try:
        return _RESOURCE_PATHS[(scope, version, kind)]
    except KeyError:
        raise ContractSchemaError(
            f"unsupported schema selection: {scope}/{version}/{kind}"
        ) from None


def _resource_bytes(
    scope: SchemaScope,
    version: SchemaVersion,
    kind: SchemaKind,
) -> bytes:
    relative_path = _resource_path(scope, version, kind)
    try:
        return resources.files("customs_ai_agent.contracts").joinpath(relative_path).read_bytes()
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        raise ContractSchemaError(
            f"packaged schema resource is missing: {scope}/{version}/{kind}"
        ) from None


def get_schema(
    scope: SchemaScope,
    version: SchemaVersion,
    kind: SchemaKind,
) -> dict[str, Any]:
    """Return a detached copy of one immutable packaged JSON Schema."""

    try:
        payload = json.loads(_resource_bytes(scope, version, kind))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ContractSchemaError(
            f"packaged schema resource is invalid: {scope}/{version}/{kind}"
        ) from None
    if not isinstance(payload, dict):
        raise ContractSchemaError(f"packaged schema root is invalid: {scope}/{version}/{kind}")
    return copy.deepcopy(payload)


def schema_sha256(
    scope: SchemaScope,
    version: SchemaVersion,
    kind: SchemaKind,
) -> str:
    """Hash the exact bytes shipped in the installed wheel."""

    return hashlib.sha256(_resource_bytes(scope, version, kind)).hexdigest()


def _strict_object_definitions(schema: dict[str, Any]) -> None:
    schema["additionalProperties"] = False
    for definition in schema.get("$defs", {}).values():
        if isinstance(definition, dict) and definition.get("type") == "object":
            definition["additionalProperties"] = False


def _backend_request_schema(version: SchemaVersion) -> dict[str, Any]:
    request_model = ReviewRunRequest if version == "v1" else ReviewRunRequestV2
    schema = copy.deepcopy(request_model.model_json_schema(mode="validation"))
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = f"https://schemas.customs.local/agent/{version}/review-request.schema.json"
    schema["title"] = f"Customs AI Agent {version.upper()} review request"
    _strict_object_definitions(schema)
    schema["oneOf"] = [
        {
            "required": ["scenario_id"],
            "properties": {
                "scenario_id": {"type": "string", "minLength": 1},
                "documents": {"type": "null"},
            },
        },
        {
            "required": ["documents"],
            "properties": {
                "scenario_id": {"type": "null"},
                "documents": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": 20,
                },
            },
        },
    ]
    document_definition = "DocumentInput" if version == "v1" else "DocumentInputV2"
    document_schema = schema["$defs"][document_definition]
    document_schema["additionalProperties"] = False
    if version == "v1":
        schema["description"] = "V1 preserves the text-only ReviewRunRequest wire contract."
        return schema

    document_schema["oneOf"] = [
        {
            "required": ["content_text"],
            "properties": {
                "content_text": {"type": "string", "minLength": 1},
                "image_pages": {"type": "array", "maxItems": 0},
                "pdf_pages": {"type": "null"},
            },
        },
        {
            "required": ["image_pages"],
            "properties": {
                "content_text": {"type": "null"},
                "image_pages": {"type": "array", "minItems": 1},
                "pdf_pages": {"type": "null"},
            },
        },
        {
            "required": ["pdf_pages"],
            "properties": {
                "content_text": {"type": "null"},
                "image_pages": {"type": "array", "maxItems": 0},
                "pdf_pages": {"type": "object"},
            },
        },
    ]
    schema["allOf"] = [
        {
            "if": {
                "required": ["documents"],
                "properties": {
                    "documents": {
                        "type": "array",
                        "contains": {
                            "type": "object",
                            "anyOf": [
                                {
                                    "required": ["image_pages"],
                                    "properties": {
                                        "image_pages": {
                                            "type": "array",
                                            "minItems": 1,
                                        }
                                    },
                                },
                                {
                                    "required": ["pdf_pages"],
                                    "properties": {"pdf_pages": {"type": "object"}},
                                },
                            ],
                        },
                    }
                },
            },
            "then": {
                "required": ["llm_mode"],
                "properties": {"llm_mode": {"const": "llm"}},
            },
        }
    ]
    schema["description"] = (
        "V2 extends ReviewRunRequest with explicit text, direct-image, and "
        "source-bound PDF-page modalities; raw PDFs, paths, and URLs are not in the wire. "
        "Visual requests require llm_mode=llm by runtime validation."
    )
    return schema


def _result_model(version: SchemaVersion) -> type[BaseModel]:
    if version == "v1":
        return ReviewRunResult
    try:
        from customs_ai_agent.schemas.models import ReviewRunResultV2
    except ImportError:
        raise ContractSchemaError("ReviewRunResultV2 is not available in this build") from None
    return ReviewRunResultV2


def _backend_result_schema(version: SchemaVersion) -> dict[str, Any]:
    schema = copy.deepcopy(_result_model(version).model_json_schema(mode="validation"))
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = f"https://schemas.customs.local/agent/{version}/review-result.schema.json"
    schema["title"] = f"Customs AI Agent {version.upper()} review result"
    _strict_object_definitions(schema)
    schema["description"] = (
        f"Public, redacted Agent review result contract {'1.0' if version == 'v1' else '2.0'}."
    )
    if version == "v2":
        model_call_schema = schema.get("$defs", {}).get("ModelCallRecordV2")
        if isinstance(model_call_schema, dict):
            required = list(model_call_schema.get("required") or [])
            for field_name in (
                "output_schema_id",
                "output_schema_version",
                "output_schema_sha256",
            ):
                if field_name not in required:
                    required.append(field_name)
            model_call_schema["required"] = required
    return schema


def _internal_schema(kind: SchemaKind) -> dict[str, Any]:
    models: dict[SchemaKind, type[BaseModel]] = {
        "ocr-layout-ledger": OcrLayoutLedger,
        "claude-text-candidate": ClaudeDocumentCandidate,
        "claude-vision-candidate": ClaudeVisionDocumentCandidate,
        "claude-crop-read-candidate": ClaudeCropReadCandidate,
    }
    try:
        model = models[kind]
    except KeyError:
        raise ContractSchemaError(f"unsupported internal schema kind: {kind}") from None
    schema = copy.deepcopy(model.model_json_schema(mode="validation"))
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = f"https://schemas.customs.local/agent/internal/v1/{kind}.schema.json"
    schema["title"] = f"Customs AI Agent internal V1 {kind}"
    schema["description"] = (
        "Agent-internal model/OCR boundary. This is not a Backend request/result contract."
    )
    return schema


def _local_manifest_schema() -> dict[str, Any]:
    schema = copy.deepcopy(LocalReviewManifestV1.model_json_schema(mode="validation"))
    schema["$schema"] = "https://json-schema.org/draft/2020-12/schema"
    schema["$id"] = "https://schemas.customs.local/agent/local/v1/review-manifest.schema.json"
    schema["title"] = "Customs AI Agent local review manifest V1"
    schema["description"] = (
        "Local CLI file-adapter contract. Paths in this manifest never cross the "
        "Backend Agent request boundary."
    )
    return schema


def generate_schema(
    scope: SchemaScope,
    version: SchemaVersion,
    kind: SchemaKind,
) -> dict[str, Any]:
    """Generate one schema from the installed authoritative Pydantic model."""

    _resource_path(scope, version, kind)
    if scope == "backend" and kind == "review-request":
        return _backend_request_schema(version)
    if scope == "backend" and kind == "review-result":
        return _backend_result_schema(version)
    if scope == "internal" and version == "v1":
        return _internal_schema(kind)
    if scope == "local" and version == "v1" and kind == "review-manifest":
        return _local_manifest_schema()
    raise ContractSchemaError(f"unsupported schema generation: {scope}/{version}/{kind}")


def verify_packaged_schemas() -> dict[SchemaKey, str]:
    """Fail if checked-in wheel resources drift from current model schemas."""

    hashes: dict[SchemaKey, str] = {}
    for scope, version, kind in available_schemas():
        packaged = get_schema(scope, version, kind)
        generated = generate_schema(scope, version, kind)
        if packaged != generated:
            raise ContractSchemaError(f"packaged schema drift: {scope}/{version}/{kind}")
        hashes[(scope, version, kind)] = schema_sha256(scope, version, kind)
    return hashes


def write_schema_resources(
    output_root: str | os.PathLike[str],
) -> dict[SchemaKey, str]:
    """Materialize deterministic JSON resources for release/build preparation."""

    root = Path(output_root)
    hashes: dict[SchemaKey, str] = {}
    for key, relative_path in _RESOURCE_PATHS.items():
        scope, version, kind = key
        schema = generate_schema(scope, version, kind)
        encoded = (
            json.dumps(
                schema,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        destination = root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(encoded)
        hashes[key] = hashlib.sha256(encoded).hexdigest()
    return hashes


__all__ = [
    "ContractSchemaError",
    "LocalManifestDocumentV1",
    "LocalManifestSourceV1",
    "LocalReviewManifestV1",
    "SchemaKind",
    "SchemaScope",
    "SchemaVersion",
    "available_schemas",
    "generate_schema",
    "get_schema",
    "schema_sha256",
    "verify_packaged_schemas",
    "write_schema_resources",
]
