"""Cryptographic trust boundary for supporting-document verification receipts."""

from __future__ import annotations

import base64
import hashlib
import json
import re
from datetime import date, datetime
from importlib.resources import files
from typing import Any, Literal

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from customs_ai_agent.schemas.v3 import (
    SupportingDocumentV1,
    VerificationAssertionV1,
    VerificationReceiptV1,
    supporting_document_claims_sha256,
    verification_receipt_unsigned_payload,
)


ASSERTION_TRUST_DOMAIN = "customs-ai-agent/supporting-document-verification/v1"
_SIGNATURE_DOMAIN = ASSERTION_TRUST_DOMAIN.encode("ascii") + b"\x00"
_MAX_TRUST_STORE_BYTES = 1024 * 1024


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


def _decode_base64url(value: str, *, expected_bytes: int) -> bytes:
    try:
        decoded = base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
    except (TypeError, ValueError):
        raise ValueError("verification signature encoding is invalid") from None
    if (
        len(decoded) != expected_bytes
        or base64.urlsafe_b64encode(decoded).decode("ascii").rstrip("=") != value
    ):
        raise ValueError("verification signature encoding is invalid")
    return decoded


class TrustedAssertionKeyV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    authority_id: str = Field(min_length=1, max_length=128)
    authority_release: str = Field(min_length=1, max_length=128)
    method: Literal["trusted_connector", "human_approver"]
    key_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    public_key_base64url: str = Field(pattern=r"^[A-Za-z0-9_-]{43}$")
    valid_from: date
    valid_until: date
    status: Literal["active", "revoked"]

    @field_validator("public_key_base64url")
    @classmethod
    def _canonical_public_key(cls, value: str) -> str:
        key_bytes = _decode_base64url(value, expected_bytes=32)
        Ed25519PublicKey.from_public_bytes(key_bytes)
        return value

    @model_validator(mode="after")
    def _valid_key_window(self) -> "TrustedAssertionKeyV1":
        if self.valid_until < self.valid_from:
            raise ValueError("assertion trust key validity window is invalid")
        if self.method == "human_approver":
            if (
                re.fullmatch(
                    r"backend:principal-sha256:[a-f0-9]{64}",
                    self.authority_id,
                )
                is None
                or self.authority_release != "backend-human-decision-policy/v1"
            ):
                raise ValueError("human assertion key authority is invalid")
        else:
            connector = self.authority_id.removeprefix("connector:")
            if (
                not connector
                or self.authority_id != f"connector:{connector}"
                or self.authority_release != f"connector-policy:{connector}/v1"
            ):
                raise ValueError("connector assertion key authority is invalid")
        return self


class AssertionTrustStoreV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    contract: Literal["customs-ai-agent-assertion-trust-store"]
    schema_version: Literal["1.0"]
    release_state: Literal["unreleased", "active"]
    trust_domain: Literal["customs-ai-agent/supporting-document-verification/v1"]
    keys: list[TrustedAssertionKeyV1]
    trust_store_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")

    @model_validator(mode="after")
    def _immutable_store(self) -> "AssertionTrustStoreV1":
        identities = [(item.authority_id, item.key_id) for item in self.keys]
        if len(identities) != len(set(identities)):
            raise ValueError("assertion trust key identities must be unique")
        public_keys = [item.public_key_base64url for item in self.keys]
        if len(public_keys) != len(set(public_keys)):
            raise ValueError("assertion trust keys must use unique public key material")
        if self.release_state == "unreleased" and self.keys:
            raise ValueError("unreleased assertion trust store must be empty")
        if self.release_state == "active" and not any(
            item.status == "active" for item in self.keys
        ):
            raise ValueError("active assertion trust store requires an active key")
        expected = _sha256(self.model_dump(mode="json", exclude={"trust_store_sha256"}))
        if self.trust_store_sha256 != expected:
            raise ValueError("assertion trust-store digest mismatch")
        return self


def _strict_packaged_store() -> dict[str, Any]:
    raw = files("customs_ai_agent").joinpath("assertion-trust-store-v1.lock.json").read_bytes()
    if not raw or len(raw) > _MAX_TRUST_STORE_BYTES:
        raise ValueError("assertion trust-store resource size is invalid")

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("assertion trust store contains a duplicate key")
            result[key] = value
        return result

    payload = json.loads(
        raw.decode("utf-8"),
        object_pairs_hook=unique_object,
        parse_constant=lambda _value: (_ for _ in ()).throw(
            ValueError("assertion trust store contains a non-finite number")
        ),
    )
    if not isinstance(payload, dict):
        raise ValueError("assertion trust store must be one JSON object")
    return payload


def load_assertion_trust_store() -> AssertionTrustStoreV1:
    return AssertionTrustStoreV1.model_validate(_strict_packaged_store())


def verification_receipt_signature_message(receipt: VerificationReceiptV1) -> bytes:
    """Canonical detached-signature message shared with Backend/connector signers."""

    return _SIGNATURE_DOMAIN + _canonical_json_bytes(verification_receipt_unsigned_payload(receipt))


class SupportingDocumentAssertionVerifier:
    """Agent-owned verifier; request data cannot replace its public-key roots."""

    def __init__(self, store: AssertionTrustStoreV1) -> None:
        self.store = AssertionTrustStoreV1.model_validate(store.model_dump(mode="python"))
        self._keys = {(item.authority_id, item.key_id): item for item in self.store.keys}

    @classmethod
    def from_packaged_store(cls) -> "SupportingDocumentAssertionVerifier":
        return cls(load_assertion_trust_store())

    @property
    def ready(self) -> bool:
        return self.store.release_state == "active"

    def __call__(
        self,
        *,
        assertion: Any,
        document: SupportingDocumentV1,
        policy: dict[str, Any],
        case_id: str,
        assertion_cutoff_at: datetime,
    ) -> bool:
        if (
            not self.ready
            or not isinstance(assertion, VerificationAssertionV1)
            or assertion.method == "test_issuer"
            or assertion.receipt is None
        ):
            return False
        receipt = assertion.receipt
        key = self._keys.get((receipt.authority_id, receipt.key_id))
        if (
            key is None
            or key.status != "active"
            or key.method != assertion.method
            or key.authority_release != assertion.authority_release
            or receipt.case_id != case_id
            or receipt.method != assertion.method
            or receipt.authority_id != assertion.authority_id
            or receipt.authority_release != assertion.authority_release
            or receipt.signed_at != assertion.asserted_at
            or assertion.source_reference_sha256 != document.source_sha256
            or receipt.supporting_document_claims_sha256
            != supporting_document_claims_sha256(document)
            or receipt.policy_source_sha256 != policy.get("policy_source_sha256")
            or receipt.policy_release != policy.get("policy_release")
            or assertion_cutoff_at.tzinfo is None
            or not receipt.signed_at <= assertion_cutoff_at <= receipt.expires_at
            or not key.valid_from <= receipt.signed_at.date() <= key.valid_until
            or assertion_cutoff_at.date() > key.valid_until
            or receipt.expires_at.date() > key.valid_until
        ):
            return False
        try:
            public_key = Ed25519PublicKey.from_public_bytes(
                _decode_base64url(key.public_key_base64url, expected_bytes=32)
            )
            signature = _decode_base64url(receipt.signature_base64url, expected_bytes=64)
            public_key.verify(signature, verification_receipt_signature_message(receipt))
        except (InvalidSignature, TypeError, ValueError):
            return False
        return True


__all__ = [
    "ASSERTION_TRUST_DOMAIN",
    "AssertionTrustStoreV1",
    "SupportingDocumentAssertionVerifier",
    "TrustedAssertionKeyV1",
    "load_assertion_trust_store",
    "verification_receipt_signature_message",
]
