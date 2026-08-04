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
from customs_ai_agent.transport_limits import TEXT_FIRST_MAX_DOCUMENTS


_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")

TEXT_REVIEW_REQUEST_SCHEMA: dict[str, Any] = {
    "additionalProperties": False,
    "description": (
        "Text-only production transport profile. Image and PDF properties are "
        "intentionally absent and rejected before the Agent runtime is invoked."
    ),
    "properties": {
        "case_id": {
            "type": "string",
            "minLength": 1,
            "maxLength": 128,
            "pattern": r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
        },
        "documents": {
            "type": "array",
            "minItems": 1,
            "maxItems": TEXT_FIRST_MAX_DOCUMENTS,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "content_text": {"type": "string", "minLength": 1, "maxLength": 200_000},
                    "declared_type": {
                        "type": ["string", "null"],
                        "enum": [
                            "invoice",
                            "packing_list",
                            "bill_of_lading",
                            "certificate_of_origin",
                            "unknown",
                            None,
                        ],
                    },
                    "document_id": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 128,
                    },
                    "filename": {"type": "string", "minLength": 1, "maxLength": 255},
                    "requirement_document_codes": {
                        "type": "array",
                        "maxItems": 100,
                        "items": {"type": "string", "minLength": 1, "maxLength": 128},
                    },
                },
                "required": ["document_id", "filename", "content_text"],
            },
        },
        "jurisdiction": {"type": "string", "const": "KR", "default": "KR"},
        "llm_mode": {"type": "string", "const": "llm"},
        "review_as_of": {"type": "string", "format": "date"},
        "review_run_id": {
            "type": "string",
            "minLength": 1,
            "maxLength": 128,
            "pattern": r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
        },
        "trade_type": {
            "type": "string",
            "enum": ["import", "export"],
            "default": "import",
        },
        "uploaded_requirement_document_codes": {
            "type": "array",
            "maxItems": 100,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1, "maxLength": 128},
        },
    },
    "required": [
        "case_id",
        "documents",
        "llm_mode",
        "review_as_of",
        "review_run_id",
    ],
    "title": "Customs AI Agent text service review request V1",
    "type": "object",
}

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
        raise ServiceHttpError(
            "request_body_timeout",
            408,
            retryable=True,
        ) from None
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
class InspectedAuxiliaryRequest:
    replay_id: str
    canonical_sha256: str


def _reject_non_text_document_set(
    documents: Any,
    settings: ServiceSettings,
    *,
    accept_empty_visual_placeholders: bool,
) -> None:
    if not isinstance(documents, list) or not 1 <= len(documents) <= settings.max_documents:
        raise ServiceHttpError("request_invalid", 422)
    text_characters = 0
    for document in documents:
        if not isinstance(document, dict):
            raise ServiceHttpError("request_invalid", 422)
        image_pages = document.get("image_pages", [])
        pdf_pages = document.get("pdf_pages")
        if accept_empty_visual_placeholders:
            if image_pages not in (None, []) or pdf_pages is not None:
                raise ServiceHttpError("visual_input_not_supported", 422)
        elif "image_pages" in document or "pdf_pages" in document:
            raise ServiceHttpError("visual_input_not_supported", 422)
        content_text = document.get("content_text")
        if not isinstance(content_text, str):
            raise ServiceHttpError("request_invalid", 422)
        text_characters += len(content_text)
        if text_characters > settings.max_text_characters:
            raise ServiceHttpError("request_too_large", 413)


def inspect_text_request(
    payload: bytes,
    settings: ServiceSettings,
    *,
    accept_empty_visual_placeholders: bool = False,
) -> InspectedRequest:
    """Reject visual and non-reproducible requests before the core runtime."""

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
    for name in ("case_id", "documents", "llm_mode", "review_run_id", "review_as_of"):
        if name not in decoded:
            raise ServiceHttpError("required_field_missing", 422)
    if decoded.get("llm_mode") != "llm":
        raise ServiceHttpError("request_invalid", 422)
    review_run_id = decoded.get("review_run_id")
    if not isinstance(review_run_id, str) or _RUN_ID.fullmatch(review_run_id) is None:
        raise ServiceHttpError("request_invalid", 422)
    review_as_of = decoded.get("review_as_of")
    if not isinstance(review_as_of, str):
        raise ServiceHttpError("request_invalid", 422)
    try:
        parsed_date = date.fromisoformat(review_as_of)
    except ValueError:
        raise ServiceHttpError("request_invalid", 422) from None
    if parsed_date.isoformat() != review_as_of:
        raise ServiceHttpError("request_invalid", 422)

    _reject_non_text_document_set(
        decoded.get("documents"),
        settings,
        accept_empty_visual_placeholders=accept_empty_visual_placeholders,
    )

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


def inspect_v3_auxiliary_request(
    payload: bytes,
    *,
    operation: str,
    settings: ServiceSettings,
) -> InspectedAuxiliaryRequest:
    """Inspect revision/answer envelopes without trusting nested result content."""

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
    if operation == "revision":
        parent = decoded.get("parent_request")
        revised = decoded.get("revised_request")
        if not isinstance(parent, dict) or not isinstance(revised, dict):
            raise ServiceHttpError("required_field_missing", 422)
        for nested_request in (parent, revised):
            _reject_non_text_document_set(
                nested_request.get("documents"),
                settings,
                accept_empty_visual_placeholders=True,
            )
        replay_id = revised.get("review_run_id")
    elif operation == "answer":
        replay_id = decoded.get("question_id")
    elif operation == "answer_v2":
        conversation_id = decoded.get("conversation_id")
        if not isinstance(conversation_id, str) or _RUN_ID.fullmatch(conversation_id) is None:
            raise ServiceHttpError("request_invalid", 422)
        replay_id = (
            "conversation:"
            + hashlib.sha256(
                json.dumps(
                    decoded,
                    ensure_ascii=False,
                    allow_nan=False,
                    separators=(",", ":"),
                    sort_keys=True,
                ).encode("utf-8")
            ).hexdigest()
        )
    else:
        raise ValueError("unsupported V3 auxiliary operation")
    if not isinstance(replay_id, str) or _RUN_ID.fullmatch(replay_id) is None:
        raise ServiceHttpError("request_invalid", 422)
    canonical = json.dumps(
        decoded,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return InspectedAuxiliaryRequest(
        replay_id=replay_id,
        canonical_sha256=hashlib.sha256(canonical).hexdigest(),
    )


__all__ = [
    "ERROR_RESPONSE_SCHEMA",
    "InspectedRequest",
    "InspectedAuxiliaryRequest",
    "TEXT_REVIEW_REQUEST_SCHEMA",
    "inspect_text_request",
    "inspect_v3_auxiliary_request",
    "raw_header_values",
    "read_bounded_json_body",
]
