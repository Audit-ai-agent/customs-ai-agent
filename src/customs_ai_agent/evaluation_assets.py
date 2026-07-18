"""Code-owned invalidations for evaluation assets that must not be reused."""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping
from types import MappingProxyType
from typing import Any


_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_SEMANTIC_DOCUMENT_KEYS = (
    "document_id",
    "source_document_type",
    "source_image_sha256",
    "expected_document_type",
    "expected_fields",
    "expected_line_item_count",
)

INVALIDATED_IMAGE_GOLD: MappingProxyType[str, str] = MappingProxyType(
    {
        # Two AI annotations agreed on six Packing List B/L values containing
        # a suffix that is absent from the source pixels. Human-independent
        # re-adjudication is not complete, so the entire Gold is ineligible.
        "75364e7c07f72633f06735fd1f4f36c719b54498c38eba2f43cb11500aca15c6": (
            "adjudicated_gold_invalidated_by_label_error"
        ),
    }
)

INVALIDATED_IMAGE_GOLD_SEMANTICS: MappingProxyType[str, str] = MappingProxyType(
    {
        # Value-bearing projection of the same invalid 12-case Gold. This pin
        # deliberately excludes JSON formatting, output paths, reviewer
        # references, worklist/gate hashes, and gold_set_id so reserialization
        # or metadata renaming cannot make the incorrect observations eligible.
        "ba07c100b59bcad78b9383f67b21dd0581a284baf64ccbe16c585728cdae7133": (
            "adjudicated_gold_invalidated_by_label_error"
        ),
    }
)

if any(
    _SHA256.fullmatch(digest) is None
    for digest in (*INVALIDATED_IMAGE_GOLD, *INVALIDATED_IMAGE_GOLD_SEMANTICS)
):
    raise RuntimeError("invalid image Gold invalidation registry")


def image_gold_invalidation_code(sha256: str | None) -> str | None:
    """Return the safe invalidation code for an exact image Gold hash."""

    if not isinstance(sha256, str):
        return None
    return INVALIDATED_IMAGE_GOLD.get(sha256.lower())


def image_gold_semantic_fingerprint(gold: object) -> str | None:
    """Hash source-bound, value-bearing Gold semantics independent of metadata.

    ``None`` means the input is not sufficiently well formed to produce a
    semantic identity. Schema validation remains the caller's responsibility.
    """

    if not isinstance(gold, Mapping):
        return None
    raw_cases = gold.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        return None

    normalized_cases: list[dict[str, Any]] = []
    seen_case_ids: set[str] = set()
    for raw_case in raw_cases:
        if not isinstance(raw_case, Mapping):
            return None
        case_id = raw_case.get("case_id")
        triplet_sha256 = raw_case.get("triplet_fingerprint_sha256")
        raw_documents = raw_case.get("documents")
        if (
            not isinstance(case_id, str)
            or not case_id
            or case_id in seen_case_ids
            or not isinstance(triplet_sha256, str)
            or _SHA256.fullmatch(triplet_sha256) is None
            or not isinstance(raw_documents, list)
            or not raw_documents
        ):
            return None
        seen_case_ids.add(case_id)

        normalized_documents: list[dict[str, Any]] = []
        document_sort_keys: set[tuple[str, str]] = set()
        for raw_document in raw_documents:
            if not isinstance(raw_document, Mapping):
                return None
            if any(key not in raw_document for key in _SEMANTIC_DOCUMENT_KEYS):
                return None
            document_id = raw_document["document_id"]
            source_document_type = raw_document["source_document_type"]
            source_image_sha256 = raw_document["source_image_sha256"]
            expected_document_type = raw_document["expected_document_type"]
            expected_fields = raw_document["expected_fields"]
            expected_line_item_count = raw_document["expected_line_item_count"]
            sort_key = (str(source_document_type), str(document_id))
            if (
                not isinstance(document_id, str)
                or not document_id
                or not isinstance(source_document_type, str)
                or not source_document_type
                or sort_key in document_sort_keys
                or not isinstance(source_image_sha256, str)
                or _SHA256.fullmatch(source_image_sha256) is None
                or not isinstance(expected_document_type, str)
                or not expected_document_type
                or not isinstance(expected_fields, Mapping)
                or not expected_fields
                or any(
                    not isinstance(name, str) or not isinstance(value, str)
                    for name, value in expected_fields.items()
                )
                or isinstance(expected_line_item_count, bool)
                or not isinstance(expected_line_item_count, int)
                or expected_line_item_count < 0
            ):
                return None
            document_sort_keys.add(sort_key)
            normalized_documents.append(
                {
                    "document_id": document_id,
                    "source_document_type": source_document_type,
                    "source_image_sha256": source_image_sha256,
                    "expected_document_type": expected_document_type,
                    "expected_fields": dict(expected_fields),
                    "expected_line_item_count": expected_line_item_count,
                }
            )

        normalized_cases.append(
            {
                "case_id": case_id,
                "triplet_fingerprint_sha256": triplet_sha256,
                "documents": sorted(
                    normalized_documents,
                    key=lambda document: (
                        document["source_document_type"],
                        document["document_id"],
                    ),
                ),
            }
        )

    payload = json.dumps(
        {"cases": sorted(normalized_cases, key=lambda case: case["case_id"])},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def image_gold_semantic_invalidation_code(gold: object) -> str | None:
    """Return an invalidation code for semantically identical bad Gold."""

    fingerprint = image_gold_semantic_fingerprint(gold)
    if fingerprint is None:
        return None
    return INVALIDATED_IMAGE_GOLD_SEMANTICS.get(fingerprint)


__all__ = [
    "INVALIDATED_IMAGE_GOLD",
    "INVALIDATED_IMAGE_GOLD_SEMANTICS",
    "image_gold_invalidation_code",
    "image_gold_semantic_fingerprint",
    "image_gold_semantic_invalidation_code",
]
