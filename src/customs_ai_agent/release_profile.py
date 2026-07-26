"""Cryptographically approved Claude release profile and kill switch."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from datetime import date, datetime
from decimal import Decimal
from importlib.resources import files
from typing import Any, Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import BaseModel, ConfigDict, Field, model_validator


_MAX_RELEASE_RESOURCE_BYTES = 1024 * 1024
_REVIEW_TRUST_DOMAIN = "customs-ai-agent/release-review/v1"
_SIGNATURE_DOMAIN = b"customs-ai-agent/release-review/v1\x00"


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def _strict_json_resource(package: str, name: str) -> dict[str, Any]:
    raw = files(package).joinpath(name).read_bytes()
    if not raw or len(raw) > _MAX_RELEASE_RESOURCE_BYTES:
        raise ValueError("release resource size is invalid")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("release resource contains a duplicate key")
            result[key] = value
        return result

    payload = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=unique_object,
        parse_constant=lambda _value: (_ for _ in ()).throw(
            ValueError("release resource contains a non-finite number")
        ),
    )
    if not isinstance(payload, dict):
        raise ValueError("release resource must be one JSON object")
    return payload


class TrustedReviewerKeyV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reviewer_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    key_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    public_key_base64url: str = Field(pattern=r"^[A-Za-z0-9_-]{43}$")
    valid_from: date
    valid_until: date
    status: Literal["active", "revoked"]

    @model_validator(mode="after")
    def _valid_window(self) -> "TrustedReviewerKeyV1":
        if self.valid_until < self.valid_from:
            raise ValueError("reviewer key validity window is invalid")
        return self


class ReviewerTrustStoreV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract: Literal["customs-ai-agent-reviewer-trust-store"]
    schema_version: Literal["1.0"]
    release_state: Literal["unreleased", "active"]
    trust_domain: Literal["customs-ai-agent/release-review/v1"]
    keys: list[TrustedReviewerKeyV1]
    trust_store_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def _immutable_store(self) -> "ReviewerTrustStoreV1":
        identities = [(item.reviewer_id, item.key_id) for item in self.keys]
        if len(identities) != len(set(identities)):
            raise ValueError("reviewer trust-store key identities must be unique")
        public_keys = [item.public_key_base64url for item in self.keys]
        if len(public_keys) != len(set(public_keys)):
            raise ValueError(
                "reviewer trust-store public keys must represent independent key material"
            )
        if self.release_state == "unreleased" and self.keys:
            raise ValueError("unreleased reviewer trust store must be empty")
        active_reviewers = {item.reviewer_id for item in self.keys if item.status == "active"}
        if self.release_state == "active" and len(active_reviewers) < 2:
            raise ValueError("active reviewer trust store requires two independent reviewers")
        expected = _sha256(self.model_dump(mode="json", exclude={"trust_store_sha256"}))
        if self.trust_store_sha256 != expected:
            raise ValueError("reviewer trust-store digest mismatch")
        return self


class ReviewerAttestationV1(BaseModel):
    """A signed decision from one key in the immutable reviewer trust store."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    reviewer_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    key_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z0-9._:-]+$")
    review_artifact_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    review_subject_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    decision: Literal["approved"]
    signed_at: datetime
    signature_base64url: str = Field(pattern=r"^[A-Za-z0-9_-]{86}$")

    @model_validator(mode="after")
    def _timezone_bound(self) -> "ReviewerAttestationV1":
        if self.signed_at.tzinfo is None:
            raise ValueError("reviewer signature time must include a timezone")
        return self


class ReleaseEvaluationSummaryV1(BaseModel):
    """Machine-checkable minimum evidence for an automatic-completion release.

    Rates are decimals rather than binary floating-point numbers so the signed
    canonical subject is stable across runtimes.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    minimum_material_field_independent_occurrences: int = Field(ge=0)
    critical_identifier_exact_match_rate: Decimal = Field(ge=0, le=1)
    required_field_exact_match_rate: Decimal = Field(ge=0, le=1)
    hallucinated_required_field_count: int = Field(ge=0)
    source_grounding_rate: Decimal = Field(ge=0, le=1)
    false_automatic_completion_count: int = Field(ge=0)
    locked_holdout_case_count: int = Field(ge=0)
    clean_production_control_run_count: int = Field(ge=0)
    clean_production_material_consistency_rate: Decimal = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _production_thresholds(self) -> "ReleaseEvaluationSummaryV1":
        failures: list[str] = []
        if self.minimum_material_field_independent_occurrences < 300:
            failures.append("minimum_material_field_independent_occurrences")
        if self.critical_identifier_exact_match_rate != Decimal("1"):
            failures.append("critical_identifier_exact_match_rate")
        if self.required_field_exact_match_rate < Decimal("0.99"):
            failures.append("required_field_exact_match_rate")
        if self.hallucinated_required_field_count != 0:
            failures.append("hallucinated_required_field_count")
        if self.source_grounding_rate != Decimal("1"):
            failures.append("source_grounding_rate")
        if self.false_automatic_completion_count != 0:
            failures.append("false_automatic_completion_count")
        if self.locked_holdout_case_count < 500:
            failures.append("locked_holdout_case_count")
        if self.clean_production_control_run_count < 30:
            failures.append("clean_production_control_run_count")
        if self.clean_production_material_consistency_rate != Decimal("1"):
            failures.append("clean_production_material_consistency_rate")
        if failures:
            raise ValueError(
                "automatic-completion evaluation thresholds failed: " + ",".join(failures)
            )
        return self


class ClaudeExtractionReleaseProfileV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract: Literal["claude-extraction-release-profile"]
    schema_version: Literal["1.0"]
    release_state: Literal["unreleased", "approved"]
    profile_id: str
    auto_completion_enabled: bool
    kill_switch_active: bool
    bounded_adaptive_active_enabled: bool
    agent_code_snapshot_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    model_id: str | None
    model_revision: str | None = Field(default=None, max_length=128)
    extraction_prompt_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    coordinator_prompt_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    narrative_prompt_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    chat_prompt_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    extraction_schema_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    coordinator_schema_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    narrative_schema_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    chat_schema_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    coordinator_semantic_schema_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    narrative_semantic_schema_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    chat_semantic_schema_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    request_schema_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    result_schema_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    policy_id: str | None = Field(default=None, min_length=1, max_length=128)
    tool_registry_id: str | None = Field(default=None, min_length=1, max_length=128)
    gold_set_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    evaluation_report_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    evaluation_summary: ReleaseEvaluationSummaryV1 | None = None
    approved_hsk10: list[str]
    approved_languages: list[str]
    approved_document_types: list[str]
    approved_trade_types: list[Literal["import", "export"]]
    approved_at: date | None = None
    valid_until: date | None = None
    reviewer_trust_store_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    assertion_trust_store_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    review_subject_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    reviewer_attestations: list[ReviewerAttestationV1]
    adjudication_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    profile_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def _fail_closed_release(self) -> "ClaudeExtractionReleaseProfileV1":
        if self.release_state == "unreleased":
            if self.auto_completion_enabled or not self.kill_switch_active:
                raise ValueError("unreleased profile must keep automatic completion disabled")
            if self.bounded_adaptive_active_enabled:
                raise ValueError("unreleased profile cannot enable active adaptive execution")
            claimed = (
                self.model_id,
                self.model_revision,
                self.agent_code_snapshot_sha256,
                self.extraction_prompt_sha256,
                self.coordinator_prompt_sha256,
                self.narrative_prompt_sha256,
                self.chat_prompt_sha256,
                self.extraction_schema_sha256,
                self.coordinator_schema_sha256,
                self.narrative_schema_sha256,
                self.chat_schema_sha256,
                self.coordinator_semantic_schema_sha256,
                self.narrative_semantic_schema_sha256,
                self.chat_semantic_schema_sha256,
                self.request_schema_sha256,
                self.result_schema_sha256,
                self.policy_id,
                self.tool_registry_id,
                self.gold_set_sha256,
                self.evaluation_report_sha256,
                self.evaluation_summary,
                self.approved_at,
                self.valid_until,
                self.reviewer_trust_store_sha256,
                self.assertion_trust_store_sha256,
                self.review_subject_sha256,
                self.adjudication_sha256,
                self.profile_sha256,
            )
            if any(value is not None for value in claimed) or any(
                (
                    self.approved_hsk10,
                    self.approved_languages,
                    self.approved_document_types,
                    self.approved_trade_types,
                    self.reviewer_attestations,
                )
            ):
                raise ValueError("unreleased profile cannot contain approval claims")
            return self
        if self.kill_switch_active or not self.auto_completion_enabled:
            raise ValueError("approved profile must explicitly enable automatic completion")
        required = (
            self.model_id,
            self.model_revision,
            self.agent_code_snapshot_sha256,
            self.extraction_prompt_sha256,
            self.coordinator_prompt_sha256,
            self.narrative_prompt_sha256,
            self.chat_prompt_sha256,
            self.extraction_schema_sha256,
            self.coordinator_schema_sha256,
            self.narrative_schema_sha256,
            self.chat_schema_sha256,
            self.coordinator_semantic_schema_sha256,
            self.narrative_semantic_schema_sha256,
            self.chat_semantic_schema_sha256,
            self.request_schema_sha256,
            self.result_schema_sha256,
            self.policy_id,
            self.tool_registry_id,
            self.gold_set_sha256,
            self.evaluation_report_sha256,
            self.evaluation_summary,
            self.approved_at,
            self.valid_until,
            self.reviewer_trust_store_sha256,
            self.assertion_trust_store_sha256,
            self.review_subject_sha256,
            self.adjudication_sha256,
            self.profile_sha256,
        )
        if any(value is None for value in required):
            raise ValueError("approved profile requires complete release identity")
        if not self.model_revision or not re.fullmatch(
            r"claude-[A-Za-z0-9._-]*[0-9]{8}", self.model_revision
        ):
            raise ValueError("approved profile requires an exact Claude model revision")
        if self.valid_until < self.approved_at:  # type: ignore[operator]
            raise ValueError("approved profile validity window is invalid")
        reviewer_ids = {item.reviewer_id for item in self.reviewer_attestations}
        key_ids = {(item.reviewer_id, item.key_id) for item in self.reviewer_attestations}
        review_artifacts = {item.review_artifact_sha256 for item in self.reviewer_attestations}
        if len(reviewer_ids) < 2 or len(key_ids) < 2 or len(review_artifacts) < 2:
            raise ValueError("approved profile requires two independent signed reviews")
        if (
            not self.approved_hsk10
            or not self.approved_languages
            or not self.approved_document_types
            or not self.approved_trade_types
        ):
            raise ValueError("approved profile requires a closed automatic-completion scope")
        subject = {
            "profile_id": self.profile_id,
            "release_state": self.release_state,
            "auto_completion_enabled": self.auto_completion_enabled,
            "kill_switch_active": self.kill_switch_active,
            "model_id": self.model_id,
            "model_revision": self.model_revision,
            "bounded_adaptive_active_enabled": self.bounded_adaptive_active_enabled,
            "agent_code_snapshot_sha256": self.agent_code_snapshot_sha256,
            "extraction_prompt_sha256": self.extraction_prompt_sha256,
            "coordinator_prompt_sha256": self.coordinator_prompt_sha256,
            "narrative_prompt_sha256": self.narrative_prompt_sha256,
            "chat_prompt_sha256": self.chat_prompt_sha256,
            "extraction_schema_sha256": self.extraction_schema_sha256,
            "coordinator_schema_sha256": self.coordinator_schema_sha256,
            "narrative_schema_sha256": self.narrative_schema_sha256,
            "chat_schema_sha256": self.chat_schema_sha256,
            "coordinator_semantic_schema_sha256": self.coordinator_semantic_schema_sha256,
            "narrative_semantic_schema_sha256": self.narrative_semantic_schema_sha256,
            "chat_semantic_schema_sha256": self.chat_semantic_schema_sha256,
            "request_schema_sha256": self.request_schema_sha256,
            "result_schema_sha256": self.result_schema_sha256,
            "policy_id": self.policy_id,
            "tool_registry_id": self.tool_registry_id,
            "gold_set_sha256": self.gold_set_sha256,
            "evaluation_report_sha256": self.evaluation_report_sha256,
            "evaluation_summary": self.evaluation_summary.model_dump(mode="json"),  # type: ignore[union-attr]
            "approved_hsk10": sorted(self.approved_hsk10),
            "approved_languages": sorted(self.approved_languages),
            "approved_document_types": sorted(self.approved_document_types),
            "approved_trade_types": sorted(self.approved_trade_types),
            "approved_at": self.approved_at.isoformat(),  # type: ignore[union-attr]
            "valid_until": self.valid_until.isoformat(),  # type: ignore[union-attr]
            "reviewer_trust_store_sha256": self.reviewer_trust_store_sha256,
            "assertion_trust_store_sha256": self.assertion_trust_store_sha256,
            "adjudication_sha256": self.adjudication_sha256,
        }
        expected_subject = _sha256(subject)
        if self.review_subject_sha256 != expected_subject or any(
            item.review_subject_sha256 != expected_subject for item in self.reviewer_attestations
        ):
            raise ValueError("review approval does not bind the exact release subject")
        expected_profile = _sha256(self.model_dump(mode="json", exclude={"profile_sha256"}))
        if self.profile_sha256 != expected_profile:
            raise ValueError("release profile digest mismatch")
        return self


def _decode_base64url(value: str, *, expected_bytes: int) -> bytes:
    try:
        decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (ValueError, TypeError):
        raise ValueError("release signature encoding is invalid") from None
    if (
        len(decoded) != expected_bytes
        or base64.urlsafe_b64encode(decoded).decode().rstrip("=") != value
    ):
        raise ValueError("release signature encoding is invalid")
    return decoded


def _attestation_message(attestation: ReviewerAttestationV1) -> bytes:
    return _SIGNATURE_DOMAIN + _canonical_json_bytes(
        {
            "reviewer_id": attestation.reviewer_id,
            "key_id": attestation.key_id,
            "review_artifact_sha256": attestation.review_artifact_sha256,
            "review_subject_sha256": attestation.review_subject_sha256,
            "decision": attestation.decision,
            "signed_at": attestation.signed_at.isoformat(),
        }
    )


def load_reviewer_trust_store() -> ReviewerTrustStoreV1:
    payload = _strict_json_resource("customs_ai_agent", "reviewer-trust-store-v1.lock.json")
    return ReviewerTrustStoreV1.model_validate(payload)


def _verify_approved_profile(
    profile: ClaudeExtractionReleaseProfileV1,
    *,
    as_of: date,
) -> None:
    if profile.approved_at is None or profile.valid_until is None:
        raise ValueError("approved profile validity is missing")
    if not profile.approved_at <= as_of <= profile.valid_until:
        raise ValueError("approved profile is not currently valid")
    store = load_reviewer_trust_store()
    if (
        store.release_state != "active"
        or profile.reviewer_trust_store_sha256 != store.trust_store_sha256
    ):
        raise ValueError("approved profile reviewer trust store is unavailable or mismatched")
    from customs_ai_agent.assertion_trust import load_assertion_trust_store

    assertion_store = load_assertion_trust_store()
    if (
        assertion_store.release_state != "active"
        or profile.assertion_trust_store_sha256 != assertion_store.trust_store_sha256
    ):
        raise ValueError("approved profile assertion trust store is unavailable or mismatched")
    keys = {(item.reviewer_id, item.key_id): item for item in store.keys}
    verified_reviewers: set[str] = set()
    verified_key_identities: set[tuple[str, str]] = set()
    verified_public_keys: set[bytes] = set()
    verified_signatures: set[bytes] = set()
    for attestation in profile.reviewer_attestations:
        key = keys.get((attestation.reviewer_id, attestation.key_id))
        if key is None or key.status != "active":
            raise ValueError("approved profile contains an unknown or revoked reviewer key")
        signed_on = attestation.signed_at.date()
        if not key.valid_from <= signed_on <= key.valid_until or signed_on > profile.approved_at:
            raise ValueError("reviewer signature is outside its approved validity window")
        try:
            public_key_bytes = _decode_base64url(key.public_key_base64url, expected_bytes=32)
            signature_bytes = _decode_base64url(
                attestation.signature_base64url,
                expected_bytes=64,
            )
            key_identity = (attestation.reviewer_id, attestation.key_id)
            if (
                key_identity in verified_key_identities
                or public_key_bytes in verified_public_keys
                or signature_bytes in verified_signatures
            ):
                raise ValueError("reviewer attestations are not cryptographically independent")
            public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
            public_key.verify(
                signature_bytes,
                _attestation_message(attestation),
            )
        except (InvalidSignature, ValueError):
            raise ValueError("reviewer signature verification failed") from None
        verified_reviewers.add(attestation.reviewer_id)
        verified_key_identities.add(key_identity)
        verified_public_keys.add(public_key_bytes)
        verified_signatures.add(signature_bytes)
    if len(verified_reviewers) < 2:
        raise ValueError("approved profile lacks two independent valid reviewer signatures")


def load_release_profile(*, as_of: date | None = None) -> ClaudeExtractionReleaseProfileV1:
    payload = _strict_json_resource("customs_ai_agent", "release-profile-v1.lock.json")
    profile = ClaudeExtractionReleaseProfileV1.model_validate(payload)
    if profile.release_state == "approved":
        _verify_approved_profile(profile, as_of=as_of or date.today())
    return profile


__all__ = [
    "ClaudeExtractionReleaseProfileV1",
    "ReleaseEvaluationSummaryV1",
    "ReviewerAttestationV1",
    "ReviewerTrustStoreV1",
    "TrustedReviewerKeyV1",
    "load_release_profile",
    "load_reviewer_trust_store",
]
