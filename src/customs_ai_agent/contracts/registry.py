"""Packaged schema bytes used by the live Backend and Data HTTP boundaries."""

from __future__ import annotations

import copy
import hashlib
import json
from importlib import resources
from typing import Any, Literal


SchemaScope = Literal["backend", "evidence"]
SchemaVersion = Literal["v3"]
SchemaKind = Literal[
    "review-request",
    "review-result",
    "assistant-answer-request",
    "assistant-answer-response",
    "rag-retrieval",
    "readiness",
]
SchemaKey = tuple[SchemaScope, SchemaVersion, SchemaKind]

_RESOURCE_PATHS: dict[SchemaKey, str] = {
    ("backend", "v3", "review-request"): "schemas/backend/v3/review-request.schema.json",
    ("backend", "v3", "review-result"): "schemas/backend/v3/review-result.schema.json",
    ("backend", "v3", "assistant-answer-request"): (
        "schemas/backend/v3/assistant-answer-request.schema.json"
    ),
    ("backend", "v3", "assistant-answer-response"): (
        "schemas/backend/v3/assistant-answer-response.schema.json"
    ),
    ("evidence", "v3", "rag-retrieval"): "evidence/v3/rag_retrieval_v3.schema.json",
    ("evidence", "v3", "readiness"): "evidence/v3/readiness_v3.schema.json",
}


class ContractSchemaError(ValueError):
    pass


def available_schemas() -> tuple[SchemaKey, ...]:
    return tuple(sorted(_RESOURCE_PATHS))


def _bytes(scope: SchemaScope, version: SchemaVersion, kind: SchemaKind) -> bytes:
    try:
        path = _RESOURCE_PATHS[(scope, version, kind)]
    except KeyError:
        raise ContractSchemaError(
            f"unsupported schema selection: {scope}/{version}/{kind}"
        ) from None
    try:
        return resources.files("customs_ai_agent.contracts").joinpath(path).read_bytes()
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        raise ContractSchemaError(
            f"packaged schema resource is missing: {scope}/{version}/{kind}"
        ) from None


def get_schema(scope: SchemaScope, version: SchemaVersion, kind: SchemaKind) -> dict[str, Any]:
    try:
        value = json.loads(_bytes(scope, version, kind))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise ContractSchemaError(
            f"packaged schema resource is invalid: {scope}/{version}/{kind}"
        ) from None
    if not isinstance(value, dict):
        raise ContractSchemaError(f"packaged schema root is invalid: {scope}/{version}/{kind}")
    return copy.deepcopy(value)


def schema_sha256(scope: SchemaScope, version: SchemaVersion, kind: SchemaKind) -> str:
    return hashlib.sha256(_bytes(scope, version, kind)).hexdigest()


def verify_packaged_schemas() -> dict[SchemaKey, str]:
    """Check that each intentionally shipped JSON schema resource parses."""

    hashes: dict[SchemaKey, str] = {}
    for key in available_schemas():
        get_schema(*key)
        hashes[key] = schema_sha256(*key)
    return hashes


__all__ = [
    "ContractSchemaError",
    "SchemaKey",
    "SchemaKind",
    "SchemaScope",
    "SchemaVersion",
    "available_schemas",
    "get_schema",
    "schema_sha256",
    "verify_packaged_schemas",
]
