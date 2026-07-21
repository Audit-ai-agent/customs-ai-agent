"""Pinned consumer identity for the provider-owned Evidence HTTP API v3."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from typing import Any, Literal

from jsonschema import Draft202012Validator, FormatChecker

from customs_ai_agent.contracts.evidence_runtime_release import (
    EvidenceProviderRuntimeIdentity,
    EvidenceRuntimeReleaseIdentity,
    EvidenceRuntimeReleaseLockError,
    canonical_json_bytes,
    load_pinned_evidence_provider_runtime_identity,
)


_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_MAX_CONTRACT_RESOURCE_BYTES = 1024 * 1024
_V3_IDENTITY_FIELDS = {
    "contract",
    "schema_version",
    "retrieval_schema_version",
    "base_runtime_release",
    "retrieval_schema_sha256",
    "readiness_schema_sha256",
    "requirement_policy_schema_sha256",
    "requirement_policy_release",
    "requirement_policy_manifest_sha256",
    "coverage_matrix_sha256",
    "requirement_policy_approval_sha256",
    "reviewer_trust_store_sha256",
    "runtime_release_v3_sha256",
}
_V3_LOCK_COMMON_FIELDS = {
    "contract",
    "schema_version",
    "canonical_owner",
    "consumer",
    "provider_runtime_lock_sha256",
    "provider_runtime_identity",
    "retrieval_schema_sha256",
    "readiness_schema_sha256",
    "requirement_policy_schema_sha256",
    "release_state",
    "identity",
}


class EvidenceV3ContractError(RuntimeError):
    pass


class EvidenceV3UnreleasedError(EvidenceV3ContractError):
    pass


@dataclass(frozen=True, slots=True)
class EvidenceRuntimeReleaseIdentityV3:
    contract: Literal["evidence-runtime-release"]
    schema_version: Literal["3.0"]
    retrieval_schema_version: Literal["3.0"]
    base_runtime_release: EvidenceRuntimeReleaseIdentity
    retrieval_schema_sha256: str
    readiness_schema_sha256: str
    requirement_policy_schema_sha256: str
    requirement_policy_release: str
    requirement_policy_manifest_sha256: str
    coverage_matrix_sha256: str
    requirement_policy_approval_sha256: str
    reviewer_trust_store_sha256: str
    runtime_release_v3_sha256: str

    @classmethod
    def from_mapping(cls, value: Any) -> "EvidenceRuntimeReleaseIdentityV3":
        if not isinstance(value, dict) or set(value) != _V3_IDENTITY_FIELDS:
            raise ValueError("Evidence runtime v3 identity shape is invalid")
        if (
            value.get("contract") != "evidence-runtime-release"
            or value.get("schema_version") != "3.0"
            or value.get("retrieval_schema_version") != "3.0"
        ):
            raise ValueError("Evidence runtime v3 identity version is invalid")
        base = EvidenceRuntimeReleaseIdentity.from_mapping(value.get("base_runtime_release"))
        hash_fields = (
            "retrieval_schema_sha256",
            "readiness_schema_sha256",
            "requirement_policy_schema_sha256",
            "requirement_policy_manifest_sha256",
            "coverage_matrix_sha256",
            "requirement_policy_approval_sha256",
            "reviewer_trust_store_sha256",
            "runtime_release_v3_sha256",
        )
        if any(
            not isinstance(value.get(name), str) or _SHA256.fullmatch(value[name]) is None
            for name in hash_fields
        ):
            raise ValueError("Evidence runtime v3 digest is invalid")
        release = value.get("requirement_policy_release")
        if not isinstance(release, str) or not 1 <= len(release) <= 128:
            raise ValueError("Evidence requirement policy release is invalid")
        digest_payload = dict(value)
        digest_payload.pop("runtime_release_v3_sha256")
        if (
            hashlib.sha256(canonical_json_bytes(digest_payload)).hexdigest()
            != value["runtime_release_v3_sha256"]
        ):
            raise ValueError("Evidence runtime v3 identity digest mismatch")
        identity = cls(
            contract="evidence-runtime-release",
            schema_version="3.0",
            retrieval_schema_version="3.0",
            base_runtime_release=base,
            retrieval_schema_sha256=value["retrieval_schema_sha256"],
            readiness_schema_sha256=value["readiness_schema_sha256"],
            requirement_policy_schema_sha256=value["requirement_policy_schema_sha256"],
            requirement_policy_release=release,
            requirement_policy_manifest_sha256=value["requirement_policy_manifest_sha256"],
            coverage_matrix_sha256=value["coverage_matrix_sha256"],
            requirement_policy_approval_sha256=value["requirement_policy_approval_sha256"],
            reviewer_trust_store_sha256=value["reviewer_trust_store_sha256"],
            runtime_release_v3_sha256=value["runtime_release_v3_sha256"],
        )
        identity.assert_packaged_contracts()
        return identity

    def to_wire(self) -> dict[str, Any]:
        return {
            "contract": self.contract,
            "schema_version": self.schema_version,
            "retrieval_schema_version": self.retrieval_schema_version,
            "base_runtime_release": self.base_runtime_release.to_wire(),
            "retrieval_schema_sha256": self.retrieval_schema_sha256,
            "readiness_schema_sha256": self.readiness_schema_sha256,
            "requirement_policy_schema_sha256": self.requirement_policy_schema_sha256,
            "requirement_policy_release": self.requirement_policy_release,
            "requirement_policy_manifest_sha256": self.requirement_policy_manifest_sha256,
            "coverage_matrix_sha256": self.coverage_matrix_sha256,
            "requirement_policy_approval_sha256": self.requirement_policy_approval_sha256,
            "reviewer_trust_store_sha256": self.reviewer_trust_store_sha256,
            "runtime_release_v3_sha256": self.runtime_release_v3_sha256,
        }

    def assert_packaged_contracts(self) -> None:
        expected = packaged_evidence_v3_schema_hashes()
        observed = {
            "rag_retrieval_v3.schema.json": self.retrieval_schema_sha256,
            "readiness_v3.schema.json": self.readiness_schema_sha256,
            "requirement_policy_v1.schema.json": self.requirement_policy_schema_sha256,
        }
        if observed != expected:
            raise ValueError("Evidence v3 identity does not match packaged Schema bytes")

    def binding_headers(self) -> dict[str, str]:
        base = self.base_runtime_release
        return {
            "X-Evidence-Schema-SHA256": base.retrieval_schema_sha256,
            "X-Evidence-Provider-Release": base.provider_release,
            "X-Evidence-Provider-Revision": base.provider_revision,
            "X-Evidence-Provider-Build-SHA256": base.provider_build_sha256,
            "X-Evidence-Knowledge-Generation": base.knowledge_generation_id,
            "X-Evidence-Generation-Attestation-SHA256": (
                base.knowledge_generation_attestation_sha256
            ),
            "X-Evidence-Evaluation-SHA256": base.evaluation_report_sha256,
            "X-Evidence-Runtime-Release-SHA256": base.runtime_release_sha256,
            "X-Evidence-V3-Retrieval-Schema-SHA256": self.retrieval_schema_sha256,
            "X-Evidence-V3-Readiness-Schema-SHA256": self.readiness_schema_sha256,
            "X-Evidence-Requirement-Policy-Schema-SHA256": (self.requirement_policy_schema_sha256),
            "X-Evidence-Requirement-Policy-Release": self.requirement_policy_release,
            "X-Evidence-Requirement-Policy-SHA256": self.requirement_policy_manifest_sha256,
            "X-Evidence-Coverage-Matrix-SHA256": self.coverage_matrix_sha256,
            "X-Evidence-Requirement-Policy-Approval-SHA256": (
                self.requirement_policy_approval_sha256
            ),
            "X-Evidence-Reviewer-Trust-Store-SHA256": (self.reviewer_trust_store_sha256),
            "X-Evidence-Runtime-V3-SHA256": self.runtime_release_v3_sha256,
        }


@dataclass(frozen=True, slots=True)
class EvidenceRuntimeReleaseLockV3:
    release_state: Literal["unreleased", "released"]
    provider_runtime_identity: EvidenceProviderRuntimeIdentity
    identity: EvidenceRuntimeReleaseIdentityV3 | None


def _resource(name: str) -> bytes:
    raw = files("customs_ai_agent.contracts").joinpath("evidence", "v3", name).read_bytes()
    if not raw or len(raw) > _MAX_CONTRACT_RESOURCE_BYTES:
        raise EvidenceV3ContractError("Evidence v3 contract resource size is invalid")
    return raw


def _provider_runtime_lock_resource() -> bytes:
    raw = (
        files("customs_ai_agent.contracts")
        .joinpath("evidence", "provider-runtime-lock.json")
        .read_bytes()
    )
    if not raw or len(raw) > _MAX_CONTRACT_RESOURCE_BYTES:
        raise EvidenceV3ContractError("Evidence provider runtime lock size is invalid")
    return raw


def _strict_json_object(raw: bytes, *, description: str) -> dict[str, Any]:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError("duplicate key")
            value[key] = item
        return value

    def reject_nonfinite(_value: str) -> Any:
        raise ValueError("non-finite JSON number")

    try:
        value = json.loads(
            raw.decode("utf-8"),
            object_pairs_hook=unique_object,
            parse_constant=reject_nonfinite,
        )
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise EvidenceV3ContractError(f"{description} is invalid") from None
    if not isinstance(value, dict):
        raise EvidenceV3ContractError(f"{description} is invalid")
    return value


@lru_cache(maxsize=1)
def packaged_evidence_v3_schema_hashes() -> dict[str, str]:
    return {
        name: hashlib.sha256(_resource(name)).hexdigest()
        for name in (
            "rag_retrieval_v3.schema.json",
            "readiness_v3.schema.json",
            "requirement_policy_v1.schema.json",
        )
    }


@lru_cache(maxsize=3)
def load_evidence_v3_schema(name: str) -> dict[str, Any]:
    if name not in packaged_evidence_v3_schema_hashes():
        raise EvidenceV3ContractError("unknown Evidence v3 Schema resource")
    try:
        value = _strict_json_object(
            _resource(name),
            description="packaged Evidence v3 Schema",
        )
        Draft202012Validator.check_schema(value)
    except (EvidenceV3ContractError, Exception):
        raise EvidenceV3ContractError("packaged Evidence v3 Schema is invalid") from None
    return value


def load_evidence_runtime_release_lock_v3() -> EvidenceRuntimeReleaseLockV3:
    try:
        payload = _strict_json_object(
            _resource("runtime-release-v3.lock.json"),
            description="Evidence v3 release lock",
        )
    except EvidenceV3ContractError:
        raise EvidenceV3ContractError("Evidence v3 release lock is invalid") from None
    try:
        state = payload.get("release_state")
        expected_fields = (
            _V3_LOCK_COMMON_FIELDS | {"unreleased_reason"}
            if state == "unreleased"
            else _V3_LOCK_COMMON_FIELDS
        )
        if (
            set(payload) != expected_fields
            or payload.get("contract") != "evidence-runtime-release-consumer-lock"
            or payload.get("schema_version") != "3.0"
            or payload.get("canonical_owner") != "customs-data-platform"
            or payload.get("consumer") != "customs-ai-agent"
            or state not in {"unreleased", "released"}
        ):
            raise ValueError
        provider = EvidenceProviderRuntimeIdentity.from_mapping(
            payload.get("provider_runtime_identity")
        )
        if provider != load_pinned_evidence_provider_runtime_identity():
            raise ValueError
        provider_lock_hash = payload.get("provider_runtime_lock_sha256")
        if (
            not isinstance(provider_lock_hash, str)
            or hashlib.sha256(_provider_runtime_lock_resource()).hexdigest() != provider_lock_hash
        ):
            raise ValueError
        schema_hashes = packaged_evidence_v3_schema_hashes()
        if (
            payload.get("retrieval_schema_sha256") != schema_hashes["rag_retrieval_v3.schema.json"]
            or payload.get("readiness_schema_sha256") != schema_hashes["readiness_v3.schema.json"]
            or payload.get("requirement_policy_schema_sha256")
            != schema_hashes["requirement_policy_v1.schema.json"]
        ):
            raise ValueError
        if state == "unreleased":
            if (
                payload.get("identity") is not None
                or not isinstance(payload.get("unreleased_reason"), str)
                or not payload["unreleased_reason"].strip()
            ):
                raise ValueError
            return EvidenceRuntimeReleaseLockV3(
                release_state="unreleased",
                provider_runtime_identity=provider,
                identity=None,
            )
        identity = EvidenceRuntimeReleaseIdentityV3.from_mapping(payload.get("identity"))
        base = identity.base_runtime_release
        if (
            base.provider_release != provider.provider_release
            or base.provider_revision != provider.provider_revision
            or base.provider_build_sha256 != provider.provider_build_sha256
            or base.provider_source_manifest_sha256 != provider.provider_source_manifest_sha256
            or base.retrieval_schema_sha256 != provider.retrieval_schema_sha256
            or identity.retrieval_schema_sha256 != payload["retrieval_schema_sha256"]
            or identity.readiness_schema_sha256 != payload["readiness_schema_sha256"]
            or identity.requirement_policy_schema_sha256
            != payload["requirement_policy_schema_sha256"]
        ):
            raise ValueError
        return EvidenceRuntimeReleaseLockV3(
            release_state="released",
            provider_runtime_identity=provider,
            identity=identity,
        )
    except (
        EvidenceRuntimeReleaseLockError,
        FileNotFoundError,
        OSError,
        TypeError,
        UnicodeError,
        ValueError,
    ):
        raise EvidenceV3ContractError("Evidence v3 release lock is invalid") from None


@lru_cache(maxsize=1)
def load_released_evidence_runtime_identity_v3() -> EvidenceRuntimeReleaseIdentityV3:
    lock = load_evidence_runtime_release_lock_v3()
    if lock.release_state != "released" or lock.identity is None:
        raise EvidenceV3UnreleasedError("Evidence v3 release is not approved")
    return lock.identity


def parse_evidence_runtime_readiness_v3(payload: Any) -> EvidenceRuntimeReleaseIdentityV3:
    schema = load_evidence_v3_schema("readiness_v3.schema.json")
    if next(
        Draft202012Validator(schema, format_checker=FormatChecker()).iter_errors(payload), None
    ):
        raise EvidenceV3ContractError("Evidence v3 readiness payload is invalid")
    if not isinstance(payload, dict) or payload.get("status") != "ready":
        raise EvidenceV3ContractError("Evidence v3 readiness payload is invalid")
    try:
        return EvidenceRuntimeReleaseIdentityV3.from_mapping(
            payload.get("evidence_runtime_release")
        )
    except ValueError:
        raise EvidenceV3ContractError("Evidence v3 readiness identity is invalid") from None


__all__ = [
    "EvidenceRuntimeReleaseIdentityV3",
    "EvidenceRuntimeReleaseLockV3",
    "EvidenceV3ContractError",
    "EvidenceV3UnreleasedError",
    "load_evidence_v3_schema",
    "load_evidence_runtime_release_lock_v3",
    "load_released_evidence_runtime_identity_v3",
    "packaged_evidence_v3_schema_hashes",
    "parse_evidence_runtime_readiness_v3",
]
