from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError


_PROVIDER_RELEASE = re.compile(r"^evidence-api-v\d+\.\d+\.\d+$")
_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class EvidenceContractLockError(RuntimeError):
    """The packaged Evidence consumer lock is unavailable or malformed."""


class EvidenceContractPayloadError(ValueError):
    """A wire payload does not satisfy the packaged provider schema."""


@dataclass(frozen=True, slots=True)
class EvidenceContractIdentity:
    contract: str
    schema_version: str
    provider_release: str
    provider_revision: str
    schema_sha256: str

    @classmethod
    def from_mapping(
        cls,
        value: Any,
        *,
        sha_field: str = "schema_sha256",
    ) -> EvidenceContractIdentity:
        if not isinstance(value, dict):
            raise ValueError("Evidence contract identity must be an object")
        contract = value.get("contract")
        schema_version = value.get("schema_version")
        provider_release = value.get("provider_release")
        provider_revision = value.get("provider_revision")
        schema_sha256 = value.get(sha_field)
        if (
            contract != "evidence-retrieval"
            or schema_version != "1.0"
            or not isinstance(provider_release, str)
            or _PROVIDER_RELEASE.fullmatch(provider_release) is None
            or not isinstance(provider_revision, str)
            or _GIT_REVISION.fullmatch(provider_revision) is None
            or not isinstance(schema_sha256, str)
            or _SHA256.fullmatch(schema_sha256) is None
        ):
            raise ValueError("Evidence contract identity is invalid")
        return cls(
            contract=contract,
            schema_version=schema_version,
            provider_release=provider_release,
            provider_revision=provider_revision,
            schema_sha256=schema_sha256,
        )


@lru_cache(maxsize=1)
def _load_pinned_evidence_contract() -> tuple[EvidenceContractIdentity, dict[str, Any]]:
    try:
        evidence_resources = files("customs_ai_agent.contracts").joinpath("evidence")
        lock_payload = json.loads(
            evidence_resources.joinpath("v1.lock.json").read_text(encoding="utf-8")
        )
        if (
            not isinstance(lock_payload, dict)
            or lock_payload.get("canonical_owner") != "customs-data-platform"
            or lock_payload.get("provider_repository") != "customs-data-platform"
            or lock_payload.get("consumer") != "customs-ai-agent"
            or lock_payload.get("packaged_snapshot_path")
            != "customs_ai_agent/contracts/evidence/v1.schema.json"
        ):
            raise ValueError
        identity = EvidenceContractIdentity.from_mapping(lock_payload, sha_field="sha256")
        schema_bytes = evidence_resources.joinpath("v1.schema.json").read_bytes()
        if hashlib.sha256(schema_bytes).hexdigest() != identity.schema_sha256:
            raise ValueError
        schema = json.loads(schema_bytes)
        if not isinstance(schema, dict):
            raise ValueError
        Draft202012Validator.check_schema(schema)
        return identity, schema
    except (
        FileNotFoundError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        SchemaError,
        json.JSONDecodeError,
    ):
        raise EvidenceContractLockError(
            "The packaged Evidence consumer contract bundle is invalid."
        ) from None


def load_pinned_evidence_contract_identity() -> EvidenceContractIdentity:
    """Load the provider identity shipped in the Agent wheel's consumer lock."""

    return _load_pinned_evidence_contract()[0]


def load_pinned_evidence_schema() -> dict[str, Any]:
    """Load the hash-verified provider schema shipped beside the lock."""

    return _load_pinned_evidence_contract()[1]


@lru_cache(maxsize=1)
def _pinned_evidence_validator() -> Draft202012Validator:
    return Draft202012Validator(
        load_pinned_evidence_schema(),
        format_checker=FormatChecker(),
    )


def validate_pinned_evidence_payload(payload: object) -> None:
    """Fail without exposing payload values when the provider wire contract drifts."""

    if not isinstance(payload, dict):
        raise EvidenceContractPayloadError("Evidence wire payload is invalid.")
    if next(_pinned_evidence_validator().iter_errors(payload), None) is not None:
        raise EvidenceContractPayloadError("Evidence wire payload is invalid.")


__all__ = [
    "EvidenceContractIdentity",
    "EvidenceContractLockError",
    "EvidenceContractPayloadError",
    "load_pinned_evidence_contract_identity",
    "load_pinned_evidence_schema",
    "validate_pinned_evidence_payload",
]
