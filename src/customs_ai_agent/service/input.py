"""Small bounded JSON admission for the active text-only review endpoint."""

from __future__ import annotations

import asyncio
import hashlib
import json
import re
from dataclasses import dataclass
from datetime import date
from typing import Any

from starlette.requests import ClientDisconnect, Request

from customs_ai_agent.service.errors import ServiceHttpError
from customs_ai_agent.service.settings import ServiceSettings


_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_REQUIRED_REQUEST_FIELDS = {
    "case_id",
    "documents",
    "jurisdiction",
    "review_as_of",
    "review_run_id",
    "trade_type",
}
_CHILD_REQUEST_FIELDS = {
    "parent_review_run_id",
    "applied_correction_proposal_ids",
    "human_corrections",
}
_ALLOWED_REQUEST_FIELDS = _REQUIRED_REQUEST_FIELDS | _CHILD_REQUEST_FIELDS
_CORRECTION_REQUIRED_FIELDS = {
    "correction_proposal_id",
    "document_id",
    "field_name",
    "new_value",
    "reason",
    "source_kind",
}
_CORRECTION_ALLOWED_FIELDS = _CORRECTION_REQUIRED_FIELDS | {
    "item_id",
    "fact_id",
    "unit",
}
_ASSISTANT_REQUIRED_FIELDS = {
    "schema_version",
    "case_id",
    "review_run_id",
    "conversation_id",
    "review_result",
    "question",
    "recent_turns",
}
_ASSISTANT_ALLOWED_FIELDS = _ASSISTANT_REQUIRED_FIELDS

ERROR_RESPONSE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "required": ["error"],
    "properties": {
        "error": {
            "type": "object",
            "additionalProperties": False,
            "required": ["code", "message", "request_id", "retryable"],
            "properties": {
                "code": {"type": "string", "minLength": 1, "maxLength": 128},
                "message": {"type": "string", "minLength": 1, "maxLength": 256},
                "request_id": {"type": "string", "format": "uuid"},
                "retryable": {"type": "boolean"},
            },
        }
    },
}


class _DuplicateKey(ValueError):
    pass


def _object_without_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise _DuplicateKey
        value[key] = item
    return value


def _reject_constant(_value: str) -> Any:
    raise ValueError


def raw_header_values(request: Request, name: str) -> list[str]:
    expected = name.casefold().encode("ascii")
    return [
        value.decode("latin-1")
        for key, value in request.scope.get("headers", [])
        if key.lower() == expected
    ]


def _validate_media_type(request: Request) -> None:
    content_types = raw_header_values(request, "content-type")
    if len(content_types) != 1:
        raise ServiceHttpError("unsupported_media_type", 415)
    parts = [part.strip() for part in content_types[0].split(";")]
    if not parts or parts[0].casefold() != "application/json":
        raise ServiceHttpError("unsupported_media_type", 415)
    parameters = parts[1:]
    if len(parameters) > 1:
        raise ServiceHttpError("unsupported_media_type", 415)
    if parameters:
        name, separator, value = parameters[0].partition("=")
        if (
            not separator
            or name.strip().casefold() != "charset"
            or value.strip().strip('"').casefold() not in {"utf-8", "utf8"}
        ):
            raise ServiceHttpError("unsupported_media_type", 415)
    encodings = raw_header_values(request, "content-encoding")
    if len(encodings) > 1 or (
        encodings and encodings[0].strip().casefold() not in {"", "identity"}
    ):
        raise ServiceHttpError("content_encoding_unsupported", 415)


async def read_bounded_json_body(request: Request, settings: ServiceSettings) -> bytes:
    """Read one identity-encoded JSON body without crossing the configured cap."""

    _validate_media_type(request)
    lengths = raw_header_values(request, "content-length")
    declared_length: int | None = None
    if len(lengths) > 1:
        raise ServiceHttpError("content_length_invalid", 400)
    if lengths:
        raw_length = lengths[0].strip()
        if not raw_length.isascii() or not raw_length.isdecimal():
            raise ServiceHttpError("content_length_invalid", 400)
        declared_length = int(raw_length)
        if declared_length > settings.max_body_bytes:
            raise ServiceHttpError("request_too_large", 413)
    body = bytearray()
    try:
        async with asyncio.timeout(settings.body_read_timeout_seconds):
            async for chunk in request.stream():
                if len(body) + len(chunk) > settings.max_body_bytes:
                    body.clear()
                    raise ServiceHttpError("request_too_large", 413)
                body.extend(chunk)
    except TimeoutError:
        body.clear()
        raise ServiceHttpError("request_body_timeout", 408, retryable=True) from None
    except ClientDisconnect:
        body.clear()
        raise ServiceHttpError("request_invalid", 400) from None
    if declared_length is not None and declared_length != len(body):
        body.clear()
        raise ServiceHttpError("content_length_invalid", 400)
    if not body:
        raise ServiceHttpError("request_invalid", 422)
    return bytes(body)


@dataclass(frozen=True, slots=True)
class InspectedRequest:
    review_run_id: str
    review_as_of: date
    canonical_sha256: str


@dataclass(frozen=True, slots=True)
class InspectedAssistantRequest:
    review_run_id: str


def _bounded_string(value: object, *, minimum: int, maximum: int) -> bool:
    return isinstance(value, str) and minimum <= len(value) <= maximum


def _inspect_child_context(decoded: dict[str, Any]) -> None:
    """Validate the bounded, append-only child-run context before execution.

    The runtime still owns model-level semantic validation, but admission must
    accept every schema-valid V3.2 child request and reject unbounded nested
    values before a review can be scheduled.  This keeps correction payloads
    on the same small JSON surface as the text-only request itself.
    """

    parent_review_run_id = decoded.get("parent_review_run_id")
    if parent_review_run_id is not None and (
        not isinstance(parent_review_run_id, str) or _RUN_ID.fullmatch(parent_review_run_id) is None
    ):
        raise ServiceHttpError("request_invalid", 422)

    proposal_ids = decoded.get("applied_correction_proposal_ids", [])
    if not isinstance(proposal_ids, list) or len(proposal_ids) > 50:
        raise ServiceHttpError("request_invalid", 422)
    if not all(_bounded_string(value, minimum=1, maximum=128) for value in proposal_ids):
        raise ServiceHttpError("request_invalid", 422)
    if len(proposal_ids) != len(set(proposal_ids)):
        raise ServiceHttpError("request_invalid", 422)

    corrections = decoded.get("human_corrections", [])
    if not isinstance(corrections, list) or len(corrections) > 50:
        raise ServiceHttpError("request_invalid", 422)
    correction_ids: list[str] = []
    for correction in corrections:
        if (
            not isinstance(correction, dict)
            or set(correction) - _CORRECTION_ALLOWED_FIELDS
            or not _CORRECTION_REQUIRED_FIELDS.issubset(correction)
        ):
            raise ServiceHttpError("request_invalid", 422)
        if not _bounded_string(correction["correction_proposal_id"], minimum=1, maximum=128):
            raise ServiceHttpError("request_invalid", 422)
        if not _bounded_string(correction["document_id"], minimum=1, maximum=128):
            raise ServiceHttpError("request_invalid", 422)
        if not _bounded_string(correction["field_name"], minimum=1, maximum=64):
            raise ServiceHttpError("request_invalid", 422)
        if not _bounded_string(correction["new_value"], minimum=1, maximum=2_000):
            raise ServiceHttpError("request_invalid", 422)
        if not _bounded_string(correction["reason"], minimum=1, maximum=2_000):
            raise ServiceHttpError("request_invalid", 422)
        if correction["source_kind"] != "human_provided":
            raise ServiceHttpError("request_invalid", 422)
        for optional_field, maximum in (("item_id", 256), ("fact_id", 256), ("unit", 64)):
            value = correction.get(optional_field)
            if value is not None and not _bounded_string(value, minimum=0, maximum=maximum):
                raise ServiceHttpError("request_invalid", 422)
        correction_ids.append(correction["correction_proposal_id"])

    if len(correction_ids) != len(set(correction_ids)) or set(correction_ids) != set(proposal_ids):
        raise ServiceHttpError("request_invalid", 422)


def inspect_text_request(payload: bytes, settings: ServiceSettings) -> InspectedRequest:
    """Reject malformed, visual, or oversized calls before worker execution."""

    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise ServiceHttpError("request_invalid", 422) from None
    if not isinstance(decoded, dict):
        raise ServiceHttpError("request_invalid", 422)
    if set(decoded) - _ALLOWED_REQUEST_FIELDS:
        raise ServiceHttpError("request_invalid", 422)
    if not _REQUIRED_REQUEST_FIELDS.issubset(decoded):
        raise ServiceHttpError("required_field_missing", 422)
    review_run_id = decoded["review_run_id"]
    if not isinstance(review_run_id, str) or _RUN_ID.fullmatch(review_run_id) is None:
        raise ServiceHttpError("request_invalid", 422)
    review_as_of = decoded["review_as_of"]
    if not isinstance(review_as_of, str):
        raise ServiceHttpError("request_invalid", 422)
    try:
        parsed_date = date.fromisoformat(review_as_of)
    except ValueError:
        raise ServiceHttpError("request_invalid", 422) from None
    if parsed_date.isoformat() != review_as_of:
        raise ServiceHttpError("request_invalid", 422)
    documents = decoded["documents"]
    if not isinstance(documents, list) or not 1 <= len(documents) <= settings.max_documents:
        raise ServiceHttpError("request_invalid", 422)
    text_characters = 0
    for document in documents:
        if not isinstance(document, dict) or {"image_pages", "pdf_pages"} & set(document):
            raise ServiceHttpError("request_invalid", 422)
        content_text = document.get("content_text")
        if not isinstance(content_text, str):
            raise ServiceHttpError("request_invalid", 422)
        text_characters += len(content_text)
        if text_characters > settings.max_text_characters:
            raise ServiceHttpError("request_too_large", 413)
    _inspect_child_context(decoded)
    canonical = json.dumps(
        decoded,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return InspectedRequest(
        review_run_id=review_run_id,
        review_as_of=parsed_date,
        canonical_sha256=hashlib.sha256(canonical).hexdigest(),
    )


def inspect_assistant_request(payload: bytes) -> InspectedAssistantRequest:
    """Reject malformed or oversized assistant context before runtime execution."""

    try:
        decoded = json.loads(
            payload.decode("utf-8"),
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
        )
    except (UnicodeDecodeError, ValueError, RecursionError):
        raise ServiceHttpError("request_invalid", 422) from None
    if not isinstance(decoded, dict) or set(decoded) - _ASSISTANT_ALLOWED_FIELDS:
        raise ServiceHttpError("request_invalid", 422)
    if not _ASSISTANT_REQUIRED_FIELDS.issubset(decoded):
        raise ServiceHttpError("required_field_missing", 422)
    if decoded.get("schema_version") != "1.0":
        raise ServiceHttpError("request_invalid", 422)
    review_run_id = decoded.get("review_run_id")
    if not isinstance(review_run_id, str) or _RUN_ID.fullmatch(review_run_id) is None:
        raise ServiceHttpError("request_invalid", 422)
    question = decoded.get("question")
    turns = decoded.get("recent_turns")
    if not isinstance(question, str) or not 1 <= len(question) <= 4000:
        raise ServiceHttpError("request_invalid", 422)
    if not isinstance(turns, list) or len(turns) > 8:
        raise ServiceHttpError("request_invalid", 422)
    return InspectedAssistantRequest(review_run_id=review_run_id)


__all__ = [
    "ERROR_RESPONSE_SCHEMA",
    "InspectedRequest",
    "InspectedAssistantRequest",
    "inspect_assistant_request",
    "inspect_text_request",
    "raw_header_values",
    "read_bounded_json_body",
]
