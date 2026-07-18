from __future__ import annotations

import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from evals.release_holdout_protocol import (
    RELEASE_INVALIDATION_CODE,
    ReleaseHoldoutProtocolError,
    validate_public_sealed_bundle_manifest,
)


SCHEMA_PATH = (
    Path(__file__).parents[1]
    / "evals"
    / "release_holdout_protocol"
    / "v1"
    / "sealed-bundle-public-manifest.schema.json"
)


def _manifest() -> dict:
    return {
        "schema_version": "sealed-release-holdout-bundle-manifest/v1",
        "dataset_id": "trade_document_triplets",
        "dataset_version": "source-v1",
        "evaluation_version": "evaluation-v3",
        "bundle_sha256": "a" * 64,
        "bundle_byte_size": 4096,
        "case_count": 20,
        "selection_provenance": "human_supplied_external_sealed_unseen",
        "public_disclosure": "bundle_count_and_sha256_only",
        "deterministic_seed_used": False,
        "ordinary_tooling_pre_access": False,
    }


def test_public_manifest_exposes_only_count_and_bundle_identity() -> None:
    value = _manifest()
    schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))

    Draft202012Validator(schema).validate(value)

    assert validate_public_sealed_bundle_manifest(value) == value
    assert not any(
        key in value
        for key in (
            "case_ids",
            "membership",
            "selection_seed",
            "holdout_seed",
            "rebalance_seed",
        )
    )


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("case_ids", ["redacted"]),
        ("selection_seed", "public-seed"),
        ("membership", "derivable"),
    ],
)
def test_public_manifest_rejects_membership_or_seed_extensions(
    key: str,
    value: object,
) -> None:
    manifest = _manifest()
    manifest[key] = value

    with pytest.raises(
        ReleaseHoldoutProtocolError,
        match="release_holdout_sealed_bundle_manifest_invalid",
    ):
        validate_public_sealed_bundle_manifest(manifest)


def test_compromised_v2_cannot_be_relabelled_as_a_sealed_bundle() -> None:
    manifest = _manifest()
    manifest["evaluation_version"] = "evaluation-v2"

    with pytest.raises(ReleaseHoldoutProtocolError, match=RELEASE_INVALIDATION_CODE):
        validate_public_sealed_bundle_manifest(manifest)
