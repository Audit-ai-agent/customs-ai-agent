from __future__ import annotations

import base64
import hashlib
from datetime import datetime
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import ValidationError

import customs_ai_agent.assertion_trust as assertion_trust
from customs_ai_agent.assertion_trust import (
    AssertionTrustStoreV1,
    SupportingDocumentAssertionVerifier,
    verification_receipt_signature_message,
)
from customs_ai_agent.schemas.v3 import (
    SupportingDocumentV1,
    VerificationAssertionV1,
    VerificationReceiptV1,
    stable_object_id,
    supporting_document_claims_sha256,
)


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _document() -> SupportingDocumentV1:
    return SupportingDocumentV1(
        document_id="support-1",
        document_code="REQ_CERT",
        issuer="Synthetic issuer",
        document_number="SYN-001",
        issued_on="2026-06-01",
        valid_until="2026-12-31",
        subject_item_ids=["item-1"],
        hs_codes=["0303540000"],
        source_sha256="a" * 64,
        verification_assertions=[],
    )


def _policy() -> dict[str, Any]:
    return {
        "required_document_code": "REQ_CERT",
        "verification_policy": "human_only",
        "policy_source_sha256": "b" * 64,
        "policy_release": "requirements-v1",
        "approved": True,
    }


def _store(
    private: Ed25519PrivateKey,
    *,
    status: str = "active",
    duplicate_private: Ed25519PrivateKey | None = None,
    valid_until: str = "2027-12-31",
) -> AssertionTrustStoreV1:
    asserted_by = "backend-reviewer-7"
    authority_id = (
        "backend:principal-sha256:" + hashlib.sha256(asserted_by.encode("utf-8")).hexdigest()
    )
    rows = [
        {
            "authority_id": authority_id,
            "authority_release": "backend-human-decision-policy/v1",
            "method": "human_approver",
            "key_id": "backend-human-key-1",
            "public_key_base64url": _b64(
                private.public_key().public_bytes(
                    encoding=serialization.Encoding.Raw,
                    format=serialization.PublicFormat.Raw,
                )
            ),
            "valid_from": "2026-01-01",
            "valid_until": valid_until,
            "status": status,
        }
    ]
    if duplicate_private is not None:
        rows.append(
            {
                **rows[0],
                "authority_id": "connector:alias",
                "authority_release": "connector-policy:alias/v1",
                "method": "trusted_connector",
                "key_id": "alias-key",
                "public_key_base64url": _b64(
                    duplicate_private.public_key().public_bytes(
                        encoding=serialization.Encoding.Raw,
                        format=serialization.PublicFormat.Raw,
                    )
                ),
                "status": "active",
            }
        )
    payload: dict[str, Any] = {
        "contract": "customs-ai-agent-assertion-trust-store",
        "schema_version": "1.0",
        "release_state": "active",
        "trust_domain": "customs-ai-agent/supporting-document-verification/v1",
        "keys": rows,
    }
    payload["trust_store_sha256"] = assertion_trust._sha256(payload)
    return AssertionTrustStoreV1.model_validate(payload)


def _signed_assertion(
    private: Ed25519PrivateKey,
    document: SupportingDocumentV1,
    *,
    case_id: str = "case-1",
    key_id: str = "backend-human-key-1",
    expires_at: str = "2026-07-23T00:00:00Z",
) -> VerificationAssertionV1:
    asserted_by = "backend-reviewer-7"
    authority_id = (
        "backend:principal-sha256:" + hashlib.sha256(asserted_by.encode("utf-8")).hexdigest()
    )
    receipt_values: dict[str, Any] = {
        "trust_domain": "customs-ai-agent/supporting-document-verification/v1",
        "case_id": case_id,
        "method": "human_approver",
        "authority_id": authority_id,
        "authority_release": "backend-human-decision-policy/v1",
        "key_id": key_id,
        "supporting_document_claims_sha256": supporting_document_claims_sha256(document),
        "policy_source_sha256": "b" * 64,
        "policy_release": "requirements-v1",
        "decision_record_sha256": "c" * 64,
        "connector_result_sha256": None,
        "signed_at": "2026-07-22T00:00:00Z",
        "expires_at": expires_at,
    }
    unsigned = VerificationReceiptV1(
        **receipt_values,
        signature_base64url="A" * 86,
    )
    receipt = VerificationReceiptV1(
        **receipt_values,
        signature_base64url=_b64(private.sign(verification_receipt_signature_message(unsigned))),
    )
    assertion_values: dict[str, Any] = {
        "method": "human_approver",
        "subject_document_id": document.document_id,
        "asserted_by": asserted_by,
        "asserted_at": "2026-07-22T00:00:00Z",
        "authority_id": authority_id,
        "authority_release": "backend-human-decision-policy/v1",
        "source_reference_sha256": document.source_sha256,
        "receipt": receipt,
    }
    unsigned_assertion = {
        **assertion_values,
        "receipt": receipt.model_dump(
            mode="json",
            exclude={"signature_base64url"},
            exclude_none=True,
            warnings=False,
        ),
    }
    return VerificationAssertionV1(
        assertion_id=stable_object_id("assert", unsigned_assertion),
        **assertion_values,
    )


def _verify(
    verifier: SupportingDocumentAssertionVerifier,
    assertion: VerificationAssertionV1,
    document: SupportingDocumentV1,
    *,
    case_id: str = "case-1",
    policy: dict[str, Any] | None = None,
    cutoff: str = "2026-07-22T12:00:00Z",
) -> bool:
    return verifier(
        assertion=assertion,
        document=document,
        policy=policy or _policy(),
        case_id=case_id,
        assertion_cutoff_at=datetime.fromisoformat(cutoff.replace("Z", "+00:00")),
    )


def test_exact_detached_backend_receipt_is_accepted() -> None:
    private = Ed25519PrivateKey.generate()
    document = _document()
    assertion = _signed_assertion(private, document)
    verifier = SupportingDocumentAssertionVerifier(_store(private))
    assert _verify(verifier, assertion, document) is True


@pytest.mark.parametrize("mutation", ["document_code", "hs_code", "item_id", "source"])
def test_receipt_cannot_be_reused_after_any_material_document_claim_changes(
    mutation: str,
) -> None:
    private = Ed25519PrivateKey.generate()
    document = _document()
    assertion = _signed_assertion(private, document)
    payload = document.model_dump(mode="python", warnings=False)
    if mutation == "document_code":
        payload["document_code"] = "OTHER_CERT"
    elif mutation == "hs_code":
        payload["hs_codes"] = ["0303550000"]
    elif mutation == "item_id":
        payload["subject_item_ids"] = ["item-2"]
    else:
        payload["source_sha256"] = "d" * 64
    altered = SupportingDocumentV1.model_validate(payload)
    assert (
        _verify(SupportingDocumentAssertionVerifier(_store(private)), assertion, altered) is False
    )


def test_case_policy_key_signature_and_expiry_drift_all_fail_closed() -> None:
    private = Ed25519PrivateKey.generate()
    document = _document()
    assertion = _signed_assertion(private, document)
    verifier = SupportingDocumentAssertionVerifier(_store(private))
    assert _verify(verifier, assertion, document, case_id="other-case") is False
    assert (
        _verify(
            verifier,
            assertion,
            document,
            policy={**_policy(), "policy_source_sha256": "e" * 64},
        )
        is False
    )
    assert (
        _verify(
            verifier,
            assertion,
            document,
            policy={**_policy(), "policy_release": "requirements-v2"},
        )
        is False
    )
    assert _verify(verifier, assertion, document, cutoff="2026-07-24T00:00:00Z") is False

    unknown_key = _signed_assertion(private, document, key_id="unknown-key")
    assert _verify(verifier, unknown_key, document) is False

    assertion_payload = assertion.model_dump(
        mode="python",
        exclude={"assertion_id"},
        warnings=False,
    )
    assertion_payload["source_reference_sha256"] = "e" * 64
    unsigned_source_drift = assertion.model_dump(
        mode="json",
        exclude={"assertion_id"},
        exclude_none=True,
        warnings=False,
    )
    unsigned_source_drift["source_reference_sha256"] = "e" * 64
    unsigned_source_drift["receipt"].pop("signature_base64url")
    source_drift = VerificationAssertionV1(
        assertion_id=stable_object_id("assert", unsigned_source_drift),
        **assertion_payload,
    )
    assert _verify(verifier, source_drift, document) is False

    receipt_payload = assertion.receipt.model_dump(mode="python", warnings=False)  # type: ignore[union-attr]
    signature = receipt_payload["signature_base64url"]
    receipt_payload["signature_base64url"] = ("A" if signature[0] != "A" else "B") + signature[1:]
    bad_receipt = VerificationReceiptV1.model_validate(receipt_payload)
    assertion_payload = assertion.model_dump(
        mode="python", exclude={"assertion_id"}, warnings=False
    )
    assertion_payload["receipt"] = bad_receipt
    bad_assertion = VerificationAssertionV1(
        assertion_id=assertion.assertion_id,
        **assertion_payload,
    )
    assert _verify(verifier, bad_assertion, document) is False


def test_revoked_key_and_reintroduced_public_key_alias_fail_closed() -> None:
    private = Ed25519PrivateKey.generate()
    document = _document()
    assertion = _signed_assertion(private, document)
    independent = Ed25519PrivateKey.generate()
    revoked_store = _store(private, status="revoked", duplicate_private=independent)
    assert _verify(SupportingDocumentAssertionVerifier(revoked_store), assertion, document) is False

    with pytest.raises(ValidationError, match="unique public key material"):
        _store(private, status="revoked", duplicate_private=private)

    expired_store = _store(private, valid_until="2026-07-22")
    assert _verify(SupportingDocumentAssertionVerifier(expired_store), assertion, document) is False


def test_packaged_empty_trust_root_is_valid_but_not_ready() -> None:
    store = assertion_trust.load_assertion_trust_store()
    assert store.release_state == "unreleased"
    assert store.keys == []
    assert SupportingDocumentAssertionVerifier(store).ready is False
