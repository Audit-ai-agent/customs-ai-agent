from __future__ import annotations

import hashlib
import re


MAX_EVIDENCE_ITEM_ID_LENGTH = 128
_EVIDENCE_HTTP_REQUEST_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


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


def stable_retrieval_request_id(review_run_id: str, retrieval_attempt: int) -> str:
    """Create a deterministic ID accepted by the Evidence HTTP request boundary."""

    if retrieval_attempt < 0:
        raise ValueError("retrieval_attempt cannot be negative")
    suffix = f"-retrieve-{retrieval_attempt}"
    candidate = f"{review_run_id}{suffix}"
    if _EVIDENCE_HTTP_REQUEST_ID.fullmatch(candidate):
        return candidate
    digest = hashlib.sha256(
        f"evidence-retrieval-request-v1\0{review_run_id}\0{retrieval_attempt}".encode(),
    ).hexdigest()
    return f"retrieval-{digest}"


__all__ = [
    "MAX_EVIDENCE_ITEM_ID_LENGTH",
    "stable_line_item_id",
    "stable_retrieval_request_id",
]
