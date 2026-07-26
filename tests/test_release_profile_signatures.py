from __future__ import annotations

import base64
from datetime import date, datetime, timezone
from types import SimpleNamespace
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import ValidationError

import customs_ai_agent.release_profile as release_profile
import customs_ai_agent.assertion_trust as assertion_trust
from customs_ai_agent.release_profile import (
    ClaudeExtractionReleaseProfileV1,
    ReleaseEvaluationSummaryV1,
    ReviewerAttestationV1,
    ReviewerTrustStoreV1,
    TrustedReviewerKeyV1,
)


@pytest.fixture(autouse=True)
def _active_assertion_trust_store(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        assertion_trust,
        "load_assertion_trust_store",
        lambda: SimpleNamespace(release_state="active", trust_store_sha256="f" * 64),
    )


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode().rstrip("=")


def _trust_store(
    keys: list[tuple[str, str, Ed25519PrivateKey]],
) -> ReviewerTrustStoreV1:
    payload: dict[str, Any] = {
        "contract": "customs-ai-agent-reviewer-trust-store",
        "schema_version": "1.0",
        "release_state": "active",
        "trust_domain": "customs-ai-agent/release-review/v1",
        "keys": [
            {
                "reviewer_id": reviewer_id,
                "key_id": key_id,
                "public_key_base64url": _b64(
                    private.public_key().public_bytes(
                        encoding=serialization.Encoding.Raw,
                        format=serialization.PublicFormat.Raw,
                    )
                ),
                "valid_from": "2026-01-01",
                "valid_until": "2027-12-31",
                "status": "active",
            }
            for reviewer_id, key_id, private in keys
        ],
    }
    payload["trust_store_sha256"] = release_profile._sha256(payload)
    return ReviewerTrustStoreV1.model_validate(payload)


def _evaluation_summary(**overrides: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "minimum_material_field_independent_occurrences": 300,
        "critical_identifier_exact_match_rate": "1",
        "required_field_exact_match_rate": "0.99",
        "hallucinated_required_field_count": 0,
        "source_grounding_rate": "1",
        "false_automatic_completion_count": 0,
        "locked_holdout_case_count": 500,
        "clean_production_control_run_count": 30,
        "clean_production_material_consistency_rate": "1",
    }
    payload.update(overrides)
    return payload


def _approved_profile(
    store: ReviewerTrustStoreV1,
    signers: list[tuple[str, str, Ed25519PrivateKey]],
    *,
    model_revision: str = "claude-test-release-20260722",
) -> ClaudeExtractionReleaseProfileV1:
    subject = {
        "profile_id": "claude-release-test-v1",
        "release_state": "approved",
        "auto_completion_enabled": True,
        "kill_switch_active": False,
        "model_id": "claude-test-release",
        "model_revision": model_revision,
        "bounded_adaptive_active_enabled": True,
        "agent_code_snapshot_sha256": "0" * 64,
        "extraction_prompt_sha256": "1" * 64,
        "coordinator_prompt_sha256": "2" * 64,
        "narrative_prompt_sha256": "3" * 64,
        "chat_prompt_sha256": "4" * 64,
        "extraction_schema_sha256": "c" * 64,
        "coordinator_schema_sha256": "d" * 64,
        "narrative_schema_sha256": "e" * 64,
        "chat_schema_sha256": "f" * 64,
        "coordinator_semantic_schema_sha256": "1" * 64,
        "narrative_semantic_schema_sha256": "2" * 64,
        "chat_semantic_schema_sha256": "3" * 64,
        "request_schema_sha256": "5" * 64,
        "result_schema_sha256": "6" * 64,
        "policy_id": "bounded-adaptive-customs-review/v1",
        "tool_registry_id": "customs-agent-tools/text-v1",
        "gold_set_sha256": "7" * 64,
        "evaluation_report_sha256": "b" * 64,
        "evaluation_summary": _evaluation_summary(),
        "approved_hsk10": ["0303540000"],
        "approved_languages": ["en"],
        "approved_document_types": [
            "bill_of_lading",
            "certificate_of_origin",
            "invoice",
            "packing_list",
        ],
        "approved_trade_types": ["import"],
        "approved_at": "2026-07-22",
        "valid_until": "2026-12-31",
        "reviewer_trust_store_sha256": store.trust_store_sha256,
        "assertion_trust_store_sha256": "f" * 64,
        "adjudication_sha256": "a" * 64,
    }
    subject_sha256 = release_profile._sha256(subject)
    attestations: list[dict[str, Any]] = []
    for index, (reviewer_id, key_id, private) in enumerate(signers):
        unsigned = ReviewerAttestationV1(
            reviewer_id=reviewer_id,
            key_id=key_id,
            review_artifact_sha256=f"{index + 8:x}" * 64,
            review_subject_sha256=subject_sha256,
            decision="approved",
            signed_at=datetime(2026, 7, 21, tzinfo=timezone.utc),
            signature_base64url="A" * 86,
        )
        signature = private.sign(release_profile._attestation_message(unsigned))
        attestations.append(
            {
                **unsigned.model_dump(mode="json", exclude={"signature_base64url"}),
                "signature_base64url": _b64(signature),
            }
        )
    payload: dict[str, Any] = {
        "contract": "claude-extraction-release-profile",
        "schema_version": "1.0",
        "release_state": "approved",
        "profile_id": "claude-release-test-v1",
        "auto_completion_enabled": True,
        "kill_switch_active": False,
        **subject,
        "review_subject_sha256": subject_sha256,
        "reviewer_attestations": attestations,
    }
    payload["profile_sha256"] = release_profile._sha256(payload)
    return ClaudeExtractionReleaseProfileV1.model_validate(payload)


def test_two_independent_ed25519_reviewers_approve_the_exact_subject(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    keys = [
        ("reviewer-1", "key-1", Ed25519PrivateKey.generate()),
        ("reviewer-2", "key-2", Ed25519PrivateKey.generate()),
    ]
    store = _trust_store(keys)
    profile = _approved_profile(store, keys)
    monkeypatch.setattr(release_profile, "load_reviewer_trust_store", lambda: store)
    release_profile._verify_approved_profile(profile, as_of=date(2026, 7, 22))


def test_unknown_or_invalid_reviewer_signature_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trusted = [
        ("reviewer-1", "key-1", Ed25519PrivateKey.generate()),
        ("reviewer-2", "key-2", Ed25519PrivateKey.generate()),
    ]
    store = _trust_store(trusted)
    untrusted_signers = [
        ("reviewer-1", "key-1", Ed25519PrivateKey.generate()),
        trusted[1],
    ]
    profile = _approved_profile(store, untrusted_signers)
    monkeypatch.setattr(release_profile, "load_reviewer_trust_store", lambda: store)
    with pytest.raises(ValueError, match="signature verification"):
        release_profile._verify_approved_profile(profile, as_of=date(2026, 7, 22))


def test_duplicate_reviewer_and_expired_profile_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    key_1 = Ed25519PrivateKey.generate()
    key_2 = Ed25519PrivateKey.generate()
    store = _trust_store([("reviewer-1", "key-1", key_1), ("reviewer-2", "key-2", key_2)])
    # The profile itself rejects two signatures controlled by one reviewer.
    with pytest.raises(ValidationError, match="independent"):
        _approved_profile(
            store,
            [("reviewer-1", "key-1", key_1), ("reviewer-1", "key-2", key_2)],
        )

    valid = _approved_profile(
        store,
        [("reviewer-1", "key-1", key_1), ("reviewer-2", "key-2", key_2)],
    )
    monkeypatch.setattr(release_profile, "load_reviewer_trust_store", lambda: store)
    with pytest.raises(ValueError, match="not currently valid"):
        release_profile._verify_approved_profile(valid, as_of=date(2027, 1, 1))


def test_reviewer_aliases_cannot_reuse_one_ed25519_public_key() -> None:
    shared_key = Ed25519PrivateKey.generate()
    with pytest.raises(ValidationError, match="public keys.*independent"):
        _trust_store(
            [
                ("reviewer-1", "key-1", shared_key),
                ("reviewer-2", "key-2", shared_key),
            ]
        )


def test_revoked_key_material_cannot_be_reintroduced_under_active_alias() -> None:
    shared_key = Ed25519PrivateKey.generate()
    independent_key = Ed25519PrivateKey.generate()
    rows = [
        ("retired-reviewer", "retired-key", shared_key, "revoked"),
        ("reviewer-1", "replacement-alias", shared_key, "active"),
        ("reviewer-2", "independent-key", independent_key, "active"),
    ]
    payload: dict[str, Any] = {
        "contract": "customs-ai-agent-reviewer-trust-store",
        "schema_version": "1.0",
        "release_state": "active",
        "trust_domain": "customs-ai-agent/release-review/v1",
        "keys": [
            {
                "reviewer_id": reviewer_id,
                "key_id": key_id,
                "public_key_base64url": _b64(
                    private.public_key().public_bytes(
                        encoding=serialization.Encoding.Raw,
                        format=serialization.PublicFormat.Raw,
                    )
                ),
                "valid_from": "2026-01-01",
                "valid_until": "2027-12-31",
                "status": status,
            }
            for reviewer_id, key_id, private, status in rows
        ],
    }
    payload["trust_store_sha256"] = release_profile._sha256(payload)
    with pytest.raises(ValidationError, match="public keys.*independent"):
        ReviewerTrustStoreV1.model_validate(payload)


def test_revoked_reviewer_does_not_satisfy_active_two_reviewer_gate() -> None:
    key_1 = Ed25519PrivateKey.generate()
    key_2 = Ed25519PrivateKey.generate()
    payload: dict[str, Any] = {
        "contract": "customs-ai-agent-reviewer-trust-store",
        "schema_version": "1.0",
        "release_state": "active",
        "trust_domain": "customs-ai-agent/release-review/v1",
        "keys": [
            {
                "reviewer_id": reviewer_id,
                "key_id": key_id,
                "public_key_base64url": _b64(
                    private.public_key().public_bytes(
                        encoding=serialization.Encoding.Raw,
                        format=serialization.PublicFormat.Raw,
                    )
                ),
                "valid_from": "2026-01-01",
                "valid_until": "2027-12-31",
                "status": status,
            }
            for reviewer_id, key_id, private, status in (
                ("reviewer-1", "key-1", key_1, "active"),
                ("reviewer-2", "key-2", key_2, "revoked"),
            )
        ],
    }
    payload["trust_store_sha256"] = release_profile._sha256(payload)
    with pytest.raises(ValidationError, match="two independent reviewers"):
        ReviewerTrustStoreV1.model_validate(payload)


def test_signature_verifier_defends_against_alias_store_validation_bypass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shared_key = Ed25519PrivateKey.generate()
    public_key = _b64(
        shared_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
    )
    keys = [
        TrustedReviewerKeyV1(
            reviewer_id=reviewer_id,
            key_id=key_id,
            public_key_base64url=public_key,
            valid_from=date(2026, 1, 1),
            valid_until=date(2027, 12, 31),
            status="active",
        )
        for reviewer_id, key_id in (("reviewer-1", "key-1"), ("reviewer-2", "key-2"))
    ]
    # ``model_construct`` represents a compromised caller bypassing Pydantic;
    # the signature boundary must still enforce cryptographic independence.
    store = ReviewerTrustStoreV1.model_construct(
        contract="customs-ai-agent-reviewer-trust-store",
        schema_version="1.0",
        release_state="active",
        trust_domain="customs-ai-agent/release-review/v1",
        keys=keys,
        trust_store_sha256="f" * 64,
    )
    signers = [
        ("reviewer-1", "key-1", shared_key),
        ("reviewer-2", "key-2", shared_key),
    ]
    profile = _approved_profile(store, signers)
    monkeypatch.setattr(release_profile, "load_reviewer_trust_store", lambda: store)
    with pytest.raises(ValueError, match="signature verification"):
        release_profile._verify_approved_profile(profile, as_of=date(2026, 7, 22))


def test_unreleased_sentinel_loads_without_a_trust_store(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        release_profile,
        "load_reviewer_trust_store",
        lambda: (_ for _ in ()).throw(ValueError("must not be read")),
    )
    profile = release_profile.load_release_profile()
    assert profile.release_state == "unreleased"
    assert profile.kill_switch_active is True


@pytest.mark.parametrize(
    ("field", "bad_value"),
    [
        ("minimum_material_field_independent_occurrences", 299),
        ("critical_identifier_exact_match_rate", "0.999"),
        ("required_field_exact_match_rate", "0.989"),
        ("hallucinated_required_field_count", 1),
        ("source_grounding_rate", "0.999"),
        ("false_automatic_completion_count", 1),
        ("locked_holdout_case_count", 499),
        ("clean_production_control_run_count", 29),
        ("clean_production_material_consistency_rate", "0.999"),
    ],
)
def test_each_release_evaluation_threshold_fails_closed(
    field: str,
    bad_value: Any,
) -> None:
    with pytest.raises(ValidationError, match=field):
        ReleaseEvaluationSummaryV1.model_validate(_evaluation_summary(**{field: bad_value}))


def test_model_alias_without_exact_snapshot_revision_cannot_be_approved() -> None:
    keys = [
        ("reviewer-1", "key-1", Ed25519PrivateKey.generate()),
        ("reviewer-2", "key-2", Ed25519PrivateKey.generate()),
    ]
    store = _trust_store(keys)
    with pytest.raises(ValidationError, match="exact Claude model revision"):
        _approved_profile(store, keys, model_revision="claude-sonnet-latest")
