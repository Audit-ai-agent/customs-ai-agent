from __future__ import annotations

import hashlib


MAX_EVIDENCE_ITEM_ID_LENGTH = 128


def stable_line_item_id(document_id: str, line_number: int) -> str:
    """Create a deterministic Evidence-compatible line ID without truncation collisions."""

    if line_number < 1:
        raise ValueError("line_number must be positive")
    suffix = f"_line_{line_number}"
    candidate = f"{document_id}{suffix}"
    if len(candidate) <= MAX_EVIDENCE_ITEM_ID_LENGTH:
        return candidate
    digest = hashlib.sha256(document_id.encode("utf-8")).hexdigest()[:32]
    separator = "_"
    prefix_length = MAX_EVIDENCE_ITEM_ID_LENGTH - len(separator) - len(digest) - len(suffix)
    if prefix_length < 1:
        raise ValueError("line item suffix leaves no room for an ID prefix")
    return f"{document_id[:prefix_length]}{separator}{digest}{suffix}"


def stable_retrieval_request_id(review_run_id: str, retry_count: int) -> str:
    """Create a deterministic Evidence request ID within the v1 128-char bound."""

    if retry_count < 0:
        raise ValueError("retry_count cannot be negative")
    suffix = f":retrieve:{retry_count}"
    candidate = f"{review_run_id}{suffix}"
    if len(candidate) <= MAX_EVIDENCE_ITEM_ID_LENGTH:
        return candidate
    digest = hashlib.sha256(review_run_id.encode("utf-8")).hexdigest()[:32]
    separator = ":"
    prefix_length = MAX_EVIDENCE_ITEM_ID_LENGTH - len(separator) - len(digest) - len(suffix)
    if prefix_length < 1:
        raise ValueError("retrieval suffix leaves no room for an ID prefix")
    return f"{review_run_id[:prefix_length]}{separator}{digest}{suffix}"


__all__ = [
    "MAX_EVIDENCE_ITEM_ID_LENGTH",
    "stable_line_item_id",
    "stable_retrieval_request_id",
]
