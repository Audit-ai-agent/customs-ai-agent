from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import date
from functools import lru_cache
from importlib.resources import files
from typing import Any, Literal

from jsonschema import Draft202012Validator, FormatChecker
from jsonschema.exceptions import SchemaError


_PROVIDER_RELEASE = re.compile(r"^evidence-api-v\d+\.\d+\.\d+$")
_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SNAPSHOT_ID = re.compile(r"^[A-Za-z0-9._:-]{1,200}$")
_EXPECTED_PROVIDER_RELEASE = "evidence-api-v1.2.0"
_IDENTITY_FIELDS = (
    "contract",
    "schema_version",
    "retrieval_schema_version",
    "provider_release",
    "provider_revision",
    "provider_build_sha256",
    "retrieval_schema_sha256",
    "provider_source_manifest_sha256",
    "knowledge_generation_id",
    "knowledge_generation_attestation_sha256",
    "knowledge_source_manifest_sha256",
    "model_manifest_sha256",
    "index_config_sha256",
    "evaluation_report_sha256",
    "seal_attestation_sha256",
    "knowledge_snapshot_id",
    "supported_review_as_of",
    "runtime_release_sha256",
)
_HASH_INPUT_FIELDS = _IDENTITY_FIELDS[:-1]
_PROVIDER_RUNTIME_LOCK_FIELDS = {
    "contract",
    "schema_version",
    "provider_release",
    "provider_revision",
    "provider_build_sha256",
    "provider_source_manifest_sha256",
    "retrieval_schema_sha256",
    "readiness_schema_sha256",
}


class EvidenceRuntimeReleaseLockError(RuntimeError):
    """The packaged production Evidence release lock cannot be trusted."""


class EvidenceRuntimeReleaseUnreleasedError(EvidenceRuntimeReleaseLockError):
    """No sealed Data release has been approved and pinned for production."""


class EvidenceRuntimeReleasePayloadError(ValueError):
    """A readiness payload does not satisfy the packaged provider schema."""


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _strict_json_object(raw: bytes) -> dict[str, Any]:
    def reject_constant(_value: str) -> None:
        raise ValueError

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError
            value[key] = item
        return value

    value = json.loads(
        raw,
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )
    if not isinstance(value, dict):
        raise ValueError
    return value


@dataclass(frozen=True, slots=True)
class EvidenceRuntimeReleaseIdentity:
    contract: Literal["evidence-runtime-release"]
    schema_version: Literal["2.0"]
    retrieval_schema_version: Literal["1.0"]
    provider_release: str
    provider_revision: str
    provider_build_sha256: str
    retrieval_schema_sha256: str
    provider_source_manifest_sha256: str
    knowledge_generation_id: str
    knowledge_generation_attestation_sha256: str
    knowledge_source_manifest_sha256: str
    model_manifest_sha256: str
    index_config_sha256: str
    evaluation_report_sha256: str
    seal_attestation_sha256: str
    knowledge_snapshot_id: str
    supported_review_as_of: date
    runtime_release_sha256: str

    @classmethod
    def from_mapping(cls, value: Any) -> EvidenceRuntimeReleaseIdentity:
        if not isinstance(value, dict) or set(value) != set(_IDENTITY_FIELDS):
            raise ValueError("Evidence runtime release identity is invalid")
        if (
            value.get("contract") != "evidence-runtime-release"
            or value.get("schema_version") != "2.0"
            or value.get("retrieval_schema_version") != "1.0"
            or value.get("provider_release") != _EXPECTED_PROVIDER_RELEASE
        ):
            raise ValueError("Evidence runtime release identity is invalid")
        provider_revision = value.get("provider_revision")
        generation_id = value.get("knowledge_generation_id")
        snapshot_id = value.get("knowledge_snapshot_id")
        if (
            not isinstance(provider_revision, str)
            or _GIT_REVISION.fullmatch(provider_revision) is None
            or not isinstance(generation_id, str)
            or _SHA256.fullmatch(generation_id) is None
            or not isinstance(snapshot_id, str)
            or _SNAPSHOT_ID.fullmatch(snapshot_id) is None
        ):
            raise ValueError("Evidence runtime release identity is invalid")
        for field_name in (
            "provider_build_sha256",
            "retrieval_schema_sha256",
            "provider_source_manifest_sha256",
            "knowledge_generation_attestation_sha256",
            "knowledge_source_manifest_sha256",
            "model_manifest_sha256",
            "index_config_sha256",
            "evaluation_report_sha256",
            "seal_attestation_sha256",
            "runtime_release_sha256",
        ):
            field_value = value.get(field_name)
            if not isinstance(field_value, str) or _SHA256.fullmatch(field_value) is None:
                raise ValueError("Evidence runtime release identity is invalid")
        raw_date = value.get("supported_review_as_of")
        if not isinstance(raw_date, str):
            raise ValueError("Evidence runtime release identity is invalid")
        try:
            supported_date = date.fromisoformat(raw_date)
        except ValueError:
            raise ValueError("Evidence runtime release identity is invalid") from None
        if supported_date.isoformat() != raw_date:
            raise ValueError("Evidence runtime release identity is invalid")

        hash_input = {field_name: value[field_name] for field_name in _HASH_INPUT_FIELDS}
        if (
            hashlib.sha256(canonical_json_bytes(hash_input)).hexdigest()
            != value["runtime_release_sha256"]
        ):
            raise ValueError("Evidence runtime release identity is invalid")
        return cls(
            contract="evidence-runtime-release",
            schema_version="2.0",
            retrieval_schema_version="1.0",
            provider_release=value["provider_release"],
            provider_revision=provider_revision,
            provider_build_sha256=value["provider_build_sha256"],
            retrieval_schema_sha256=value["retrieval_schema_sha256"],
            provider_source_manifest_sha256=value["provider_source_manifest_sha256"],
            knowledge_generation_id=generation_id,
            knowledge_generation_attestation_sha256=value[
                "knowledge_generation_attestation_sha256"
            ],
            knowledge_source_manifest_sha256=value["knowledge_source_manifest_sha256"],
            model_manifest_sha256=value["model_manifest_sha256"],
            index_config_sha256=value["index_config_sha256"],
            evaluation_report_sha256=value["evaluation_report_sha256"],
            seal_attestation_sha256=value["seal_attestation_sha256"],
            knowledge_snapshot_id=snapshot_id,
            supported_review_as_of=supported_date,
            runtime_release_sha256=value["runtime_release_sha256"],
        )

    def to_wire(self) -> dict[str, str]:
        payload = asdict(self)
        payload["supported_review_as_of"] = self.supported_review_as_of.isoformat()
        return payload


@dataclass(frozen=True, slots=True)
class EvidenceRuntimeReleaseLock:
    release_state: Literal["unreleased", "released"]
    identity: EvidenceRuntimeReleaseIdentity | None


@dataclass(frozen=True, slots=True)
class EvidenceProviderRuntimeIdentity:
    """Exact non-secret identity of the Data provider build shipped with this Agent."""

    contract: Literal["evidence-runtime-release-provider-lock"]
    schema_version: Literal["1.0"]
    provider_release: str
    provider_revision: str
    provider_build_sha256: str
    provider_source_manifest_sha256: str
    retrieval_schema_sha256: str
    readiness_schema_sha256: str

    @classmethod
    def from_mapping(cls, value: Any) -> EvidenceProviderRuntimeIdentity:
        if not isinstance(value, dict) or set(value) != _PROVIDER_RUNTIME_LOCK_FIELDS:
            raise ValueError("Evidence provider runtime identity is invalid")
        if (
            value.get("contract") != "evidence-runtime-release-provider-lock"
            or value.get("schema_version") != "1.0"
            or value.get("provider_release") != _EXPECTED_PROVIDER_RELEASE
        ):
            raise ValueError("Evidence provider runtime identity is invalid")
        revision = value.get("provider_revision")
        if not isinstance(revision, str) or _GIT_REVISION.fullmatch(revision) is None:
            raise ValueError("Evidence provider runtime identity is invalid")
        for field_name in (
            "provider_build_sha256",
            "provider_source_manifest_sha256",
            "retrieval_schema_sha256",
            "readiness_schema_sha256",
        ):
            field_value = value.get(field_name)
            if not isinstance(field_value, str) or _SHA256.fullmatch(field_value) is None:
                raise ValueError("Evidence provider runtime identity is invalid")

        # Data computes the build digest over its canonical provider-build projection.
        build_payload = {
            "contract": "evidence-retrieval",
            "schema_version": "1.0",
            "provider_release": value["provider_release"],
            "provider_revision": revision,
            "schema_sha256": value["retrieval_schema_sha256"],
            "source_manifest_sha256": value["provider_source_manifest_sha256"],
        }
        if (
            hashlib.sha256(canonical_json_bytes(build_payload)).hexdigest()
            != value["provider_build_sha256"]
        ):
            raise ValueError("Evidence provider runtime build digest is invalid")
        return cls(
            contract="evidence-runtime-release-provider-lock",
            schema_version="1.0",
            provider_release=value["provider_release"],
            provider_revision=revision,
            provider_build_sha256=value["provider_build_sha256"],
            provider_source_manifest_sha256=value["provider_source_manifest_sha256"],
            retrieval_schema_sha256=value["retrieval_schema_sha256"],
            readiness_schema_sha256=value["readiness_schema_sha256"],
        )

    def to_wire(self) -> dict[str, str]:
        return asdict(self)


def _read_resource_bytes(name: str) -> bytes:
    return files("customs_ai_agent.contracts").joinpath("evidence", name).read_bytes()


@lru_cache(maxsize=1)
def load_pinned_evidence_provider_runtime_identity() -> EvidenceProviderRuntimeIdentity:
    """Load and byte-bind the exact Data provider build pin from the Agent wheel."""

    try:
        payload = _strict_json_object(_read_resource_bytes("provider-runtime-lock.json"))
        identity = EvidenceProviderRuntimeIdentity.from_mapping(payload)
        if (
            hashlib.sha256(_read_resource_bytes("v1.schema.json")).hexdigest()
            != identity.retrieval_schema_sha256
            or hashlib.sha256(_read_resource_bytes("runtime-release-v2.schema.json")).hexdigest()
            != identity.readiness_schema_sha256
        ):
            raise ValueError
        return identity
    except (
        FileNotFoundError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
    ):
        raise EvidenceRuntimeReleaseLockError(
            "The packaged Evidence provider runtime identity is invalid."
        ) from None


@lru_cache(maxsize=1)
def load_evidence_runtime_release_schema() -> dict[str, Any]:
    try:
        payload = _strict_json_object(_read_resource_bytes("runtime-release-v2.schema.json"))
        Draft202012Validator.check_schema(payload)
        return payload
    except (
        FileNotFoundError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        SchemaError,
        json.JSONDecodeError,
    ):
        raise EvidenceRuntimeReleaseLockError(
            "The packaged Evidence runtime release schema is invalid."
        ) from None


@lru_cache(maxsize=1)
def load_evidence_runtime_release_lock() -> EvidenceRuntimeReleaseLock:
    try:
        raw = _read_resource_bytes("runtime-release-v2.lock.json")
        payload = _strict_json_object(raw)
        common_keys = {
            "contract",
            "schema_version",
            "canonical_owner",
            "consumer",
            "provider_release",
            "retrieval_schema_version",
            "retrieval_schema_sha256",
            "readiness_schema_sha256",
            "release_state",
            "identity",
        }
        if (
            not isinstance(payload, dict)
            or payload.get("contract") != "evidence-runtime-release-consumer-lock"
            or payload.get("schema_version") != "1.0"
            or payload.get("canonical_owner") != "customs-data-platform"
            or payload.get("consumer") != "customs-ai-agent"
            or payload.get("provider_release") != _EXPECTED_PROVIDER_RELEASE
            or payload.get("retrieval_schema_version") != "1.0"
            or payload.get("release_state") not in {"unreleased", "released"}
        ):
            raise ValueError
        schema_bytes = _read_resource_bytes("runtime-release-v2.schema.json")
        retrieval_schema_bytes = _read_resource_bytes("v1.schema.json")
        schema_sha256 = payload.get("readiness_schema_sha256")
        retrieval_schema_sha256 = payload.get("retrieval_schema_sha256")
        if (
            not isinstance(schema_sha256, str)
            or _SHA256.fullmatch(schema_sha256) is None
            or hashlib.sha256(schema_bytes).hexdigest() != schema_sha256
            or not isinstance(retrieval_schema_sha256, str)
            or _SHA256.fullmatch(retrieval_schema_sha256) is None
            or hashlib.sha256(retrieval_schema_bytes).hexdigest() != retrieval_schema_sha256
        ):
            raise ValueError
        provider_identity = load_pinned_evidence_provider_runtime_identity()
        if (
            payload["provider_release"] != provider_identity.provider_release
            or retrieval_schema_sha256 != provider_identity.retrieval_schema_sha256
            or schema_sha256 != provider_identity.readiness_schema_sha256
        ):
            raise ValueError
        release_state = payload["release_state"]
        if release_state == "unreleased":
            if (
                set(payload) != common_keys | {"unreleased_reason"}
                or payload.get("identity") is not None
                or not isinstance(payload.get("unreleased_reason"), str)
                or not payload["unreleased_reason"].strip()
            ):
                raise ValueError
            return EvidenceRuntimeReleaseLock(release_state="unreleased", identity=None)
        if set(payload) != common_keys:
            raise ValueError
        identity = EvidenceRuntimeReleaseIdentity.from_mapping(payload.get("identity"))
        if (
            identity.provider_release != payload["provider_release"]
            or identity.retrieval_schema_version != payload["retrieval_schema_version"]
            or identity.retrieval_schema_sha256 != payload["retrieval_schema_sha256"]
            or identity.provider_revision != provider_identity.provider_revision
            or identity.provider_build_sha256 != provider_identity.provider_build_sha256
            or identity.provider_source_manifest_sha256
            != provider_identity.provider_source_manifest_sha256
        ):
            raise ValueError
        return EvidenceRuntimeReleaseLock(release_state="released", identity=identity)
    except EvidenceRuntimeReleaseLockError:
        raise
    except (
        FileNotFoundError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
        json.JSONDecodeError,
    ):
        raise EvidenceRuntimeReleaseLockError(
            "The packaged Evidence runtime release lock is invalid."
        ) from None


def load_released_evidence_runtime_identity() -> EvidenceRuntimeReleaseIdentity:
    lock = load_evidence_runtime_release_lock()
    if lock.release_state != "released" or lock.identity is None:
        raise EvidenceRuntimeReleaseUnreleasedError(
            "No sealed Evidence runtime release is pinned for production."
        )
    return lock.identity


@lru_cache(maxsize=1)
def _runtime_release_validator() -> Draft202012Validator:
    return Draft202012Validator(
        load_evidence_runtime_release_schema(),
        format_checker=FormatChecker(),
    )


def parse_evidence_runtime_readiness(payload: object) -> EvidenceRuntimeReleaseIdentity:
    if not isinstance(payload, dict):
        raise EvidenceRuntimeReleasePayloadError("Evidence runtime readiness is invalid.")
    if next(_runtime_release_validator().iter_errors(payload), None) is not None:
        raise EvidenceRuntimeReleasePayloadError("Evidence runtime readiness is invalid.")
    try:
        return EvidenceRuntimeReleaseIdentity.from_mapping(payload["evidence_runtime_release"])
    except (KeyError, ValueError):
        raise EvidenceRuntimeReleasePayloadError("Evidence runtime readiness is invalid.") from None


__all__ = [
    "EvidenceProviderRuntimeIdentity",
    "EvidenceRuntimeReleaseIdentity",
    "EvidenceRuntimeReleaseLock",
    "EvidenceRuntimeReleaseLockError",
    "EvidenceRuntimeReleasePayloadError",
    "EvidenceRuntimeReleaseUnreleasedError",
    "canonical_json_bytes",
    "load_evidence_runtime_release_lock",
    "load_evidence_runtime_release_schema",
    "load_pinned_evidence_provider_runtime_identity",
    "load_released_evidence_runtime_identity",
    "parse_evidence_runtime_readiness",
]
