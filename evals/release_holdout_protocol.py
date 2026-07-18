"""Versioned release-holdout boundary without public membership derivation.

Evaluation-v2 is permanently ineligible for release evidence.  Its membership
can be derived from repository-visible inputs, so authorization receipts or a
new label cannot make it unseen again.

The replacement protocol intentionally exposes only a sealed bundle's byte
identity and count.  Selection, membership, and any seed stay outside the
repository and outside ordinary development tooling.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any


DATASET_ID = "trade_document_triplets"
DATASET_VERSION = "source-v1"
INVALIDATED_EVALUATION_VERSION = "evaluation-v2"
REPLACEMENT_EVALUATION_VERSION = "evaluation-v3"
SEALED_BUNDLE_MANIFEST_SCHEMA = "sealed-release-holdout-bundle-manifest/v1"
RELEASE_INVALIDATION_CODE = "release_holdout_v2_publicly_derivable"
REPLACEMENT_NOT_PROVISIONED_CODE = "release_holdout_v3_sealed_bundle_not_provisioned"
INVALIDATED_RELEASE_EVALUATION_VERSIONS = frozenset(
    {
        "evaluation-v1",
        INVALIDATED_EVALUATION_VERSION,
    }
)

_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_PUBLIC_MANIFEST_KEYS = frozenset(
    {
        "schema_version",
        "dataset_id",
        "dataset_version",
        "evaluation_version",
        "bundle_sha256",
        "bundle_byte_size",
        "case_count",
        "selection_provenance",
        "public_disclosure",
        "deterministic_seed_used",
        "ordinary_tooling_pre_access",
    }
)


class ReleaseHoldoutProtocolError(ValueError):
    """A release-holdout input violates the permanent protocol boundary."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def require_release_eligible_version(evaluation_version: str) -> None:
    """Reject compromised or unimplemented release evaluation versions."""

    if evaluation_version in INVALIDATED_RELEASE_EVALUATION_VERSIONS:
        raise ReleaseHoldoutProtocolError(RELEASE_INVALIDATION_CODE)
    if evaluation_version != REPLACEMENT_EVALUATION_VERSION:
        raise ReleaseHoldoutProtocolError("release_holdout_evaluation_version_unsupported")


def validate_public_sealed_bundle_manifest(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate the only repository-visible metadata allowed for replacement.

    This validates a public, value-free envelope.  It does not open, decrypt,
    enumerate, or otherwise inspect the sealed bundle.
    """

    require_release_eligible_version(str(value.get("evaluation_version") or ""))
    if (
        set(value) != _PUBLIC_MANIFEST_KEYS
        or value.get("schema_version") != SEALED_BUNDLE_MANIFEST_SCHEMA
        or value.get("dataset_id") != DATASET_ID
        or value.get("dataset_version") != DATASET_VERSION
        or not isinstance(value.get("bundle_sha256"), str)
        or _SHA256.fullmatch(value["bundle_sha256"]) is None
        or not isinstance(value.get("bundle_byte_size"), int)
        or isinstance(value.get("bundle_byte_size"), bool)
        or value["bundle_byte_size"] < 1
        or not isinstance(value.get("case_count"), int)
        or isinstance(value.get("case_count"), bool)
        or value["case_count"] < 1
        or value.get("selection_provenance") != "human_supplied_external_sealed_unseen"
        or value.get("public_disclosure") != "bundle_count_and_sha256_only"
        or value.get("deterministic_seed_used") is not False
        or value.get("ordinary_tooling_pre_access") is not False
    ):
        raise ReleaseHoldoutProtocolError("release_holdout_sealed_bundle_manifest_invalid")
    return dict(value)
