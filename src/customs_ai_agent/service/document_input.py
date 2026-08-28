"""Bounded binary admission for the isolated document-preview endpoint.

The regular review endpoint intentionally remains JSON/text-only.  This module
is the sole Agent-side place that reads PDF/JPEG/PNG bytes, and it never logs,
serializes, or writes those bytes to a transcript.
"""

from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass
from typing import Literal

from starlette.requests import ClientDisconnect, Request

from customs_ai_agent.schemas.document_extraction import (
    DocumentExtractionRequestMetadataV1,
    PreviewInputV1,
)
from customs_ai_agent.schemas.raster_document_extraction import (
    RasterDocumentExtractionRequestMetadataV2,
    RasterPreviewInputV2,
)
from customs_ai_agent.schemas.raster_text_extraction import (
    RasterTextExtractionRequestMetadataV1,
    RasterTextInputV1,
)
from customs_ai_agent.service.errors import ServiceHttpError
from customs_ai_agent.service.input import raw_header_values
from customs_ai_agent.service.settings import ServiceSettings


_SOURCE_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_JPEG_SOF_MARKERS = frozenset(
    {
        0xC0,
        0xC1,
        0xC2,
        0xC3,
        0xC5,
        0xC6,
        0xC7,
        0xC9,
        0xCA,
        0xCB,
        0xCD,
        0xCE,
        0xCF,
    }
)
DocumentMediaKind = Literal["application/pdf", "image/jpeg", "image/png"]


@dataclass(frozen=True, slots=True)
class DocumentStructure:
    media_type: DocumentMediaKind
    image_width: int | None = None
    image_height: int | None = None
    pdf_page_hint: int | None = None
    # v1 ignores EXIF orientation exactly as before.  v2 reads this only to
    # stop before emitting geometry that would be wrong after browser rotation.
    jpeg_exif_orientation: int | None = None


@dataclass(frozen=True, slots=True)
class InspectedDocumentExtractionRequest:
    preview_input: PreviewInputV1
    structure: DocumentStructure


@dataclass(frozen=True, slots=True)
class InspectedRasterDocumentExtractionRequest:
    """Admitted raw document for the separate raster v2 boundary."""

    raster_input: RasterPreviewInputV2
    structure: DocumentStructure


@dataclass(frozen=True, slots=True)
class InspectedRasterTextExtractionRequest:
    """Admitted raw document for the independent text-only raster boundary."""

    raster_text_input: RasterTextInputV1
    structure: DocumentStructure


def _single_header(request: Request, name: str) -> str:
    values = raw_header_values(request, name)
    if len(values) != 1:
        raise ServiceHttpError("document_request_invalid", 422)
    value = values[0]
    if not value.isascii():
        raise ServiceHttpError("document_request_invalid", 422)
    return value


def _validate_content_encoding(request: Request) -> None:
    encodings = raw_header_values(request, "content-encoding")
    if len(encodings) > 1 or (
        encodings and encodings[0].strip().casefold() not in {"", "identity"}
    ):
        raise ServiceHttpError("content_encoding_unsupported", 415)


def _document_media_type(request: Request) -> DocumentMediaKind:
    content_type = _single_header(request, "content-type").strip().casefold()
    # No parameter is accepted: binary preview never performs implicit charset
    # decoding and a multipart wrapper would blur the binary boundary.
    if ";" in content_type or content_type not in {
        "application/pdf",
        "image/jpeg",
        "image/png",
    }:
        raise ServiceHttpError("unsupported_media_type", 415)
    return content_type  # type: ignore[return-value]


def _content_length(request: Request, settings: ServiceSettings) -> int | None:
    values = raw_header_values(request, "content-length")
    if len(values) > 1:
        raise ServiceHttpError("content_length_invalid", 400)
    if not values:
        return None
    value = values[0].strip()
    if not value.isascii() or not value.isdecimal():
        raise ServiceHttpError("content_length_invalid", 400)
    parsed = int(value)
    if parsed == 0:
        raise ServiceHttpError("document_request_invalid", 422)
    if parsed > settings.max_document_bytes:
        raise ServiceHttpError("document_too_large", 413)
    return parsed


def _request_metadata(
    request: Request, media_type: DocumentMediaKind
) -> DocumentExtractionRequestMetadataV1:
    asset_id = _single_header(request, "x-document-asset-id").strip()
    extraction_request_id = _single_header(request, "x-extraction-request-id").strip()
    source_sha256 = _single_header(request, "x-document-sha256").strip()
    if (
        _IDENTIFIER.fullmatch(asset_id) is None
        or _IDENTIFIER.fullmatch(extraction_request_id) is None
        or _SOURCE_SHA256.fullmatch(source_sha256) is None
    ):
        raise ServiceHttpError("document_request_invalid", 422)
    return DocumentExtractionRequestMetadataV1(
        document_asset_id=asset_id,
        extraction_request_id=extraction_request_id,
        source_sha256=source_sha256,
        media_type=media_type,
    )


def _raster_request_metadata(
    request: Request, media_type: DocumentMediaKind
) -> RasterDocumentExtractionRequestMetadataV2:
    """Read the v2 profile binding without changing the v1 header contract."""

    base = _request_metadata(request, media_type)
    raster_profile_revision = _single_header(request, "x-raster-profile-revision").strip()
    if _IDENTIFIER.fullmatch(raster_profile_revision) is None:
        raise ServiceHttpError("document_request_invalid", 422)
    return RasterDocumentExtractionRequestMetadataV2(
        document_asset_id=base.document_asset_id,
        extraction_request_id=base.extraction_request_id,
        source_sha256=base.source_sha256,
        media_type=base.media_type,
        raster_profile_revision=raster_profile_revision,
    )


def _raster_text_request_metadata(
    request: Request, media_type: DocumentMediaKind
) -> RasterTextExtractionRequestMetadataV1:
    """Read the text-only profile header without widening the v2 contract."""

    base = _request_metadata(request, media_type)
    raster_text_profile_revision = _single_header(
        request, "x-raster-text-profile-revision"
    ).strip()
    if _IDENTIFIER.fullmatch(raster_text_profile_revision) is None:
        raise ServiceHttpError("document_request_invalid", 422)
    return RasterTextExtractionRequestMetadataV1(
        document_asset_id=base.document_asset_id,
        extraction_request_id=base.extraction_request_id,
        source_sha256=base.source_sha256,
        media_type=base.media_type,
        raster_text_profile_revision=raster_text_profile_revision,
    )


def _bounded_pixels(width: int, height: int, settings: ServiceSettings) -> tuple[int, int]:
    if width < 1 or height < 1 or width * height > settings.max_document_image_pixels:
        raise ServiceHttpError("document_image_limits_exceeded", 422)
    return width, height


def _png_dimensions(content: bytes, settings: ServiceSettings) -> tuple[int, int]:
    if len(content) < 24 or not content.startswith(_PNG_SIGNATURE) or content[12:16] != b"IHDR":
        raise ServiceHttpError("document_magic_mismatch", 422)
    width = int.from_bytes(content[16:20], "big")
    height = int.from_bytes(content[20:24], "big")
    return _bounded_pixels(width, height, settings)


def _jpeg_dimensions(content: bytes, settings: ServiceSettings) -> tuple[int, int]:
    if len(content) < 4 or content[:3] != b"\xff\xd8\xff":
        raise ServiceHttpError("document_magic_mismatch", 422)
    index = 2
    while index < len(content):
        if content[index] != 0xFF:
            # Before scan data, JPEG segments must start with a marker.  Do
            # not attempt best-effort recovery because dimensions are a safety
            # boundary for decompression-bomb resistance.
            raise ServiceHttpError("document_decode_invalid", 422)
        while index < len(content) and content[index] == 0xFF:
            index += 1
        if index >= len(content):
            break
        marker = content[index]
        index += 1
        if marker in {0xD8, 0xD9, 0x01} or 0xD0 <= marker <= 0xD7:
            continue
        if index + 2 > len(content):
            break
        segment_length = int.from_bytes(content[index : index + 2], "big")
        if segment_length < 2 or index + segment_length > len(content):
            break
        if marker in _JPEG_SOF_MARKERS:
            if segment_length < 8:
                break
            height = int.from_bytes(content[index + 3 : index + 5], "big")
            width = int.from_bytes(content[index + 5 : index + 7], "big")
            return _bounded_pixels(width, height, settings)
        # SOS begins entropy-coded data.  The SOF marker is required before it.
        if marker == 0xDA:
            break
        index += segment_length
    raise ServiceHttpError("document_decode_invalid", 422)


def _jpeg_exif_orientation(content: bytes) -> int | None:
    """Return EXIF orientation, or ``0`` when an EXIF block is malformed.

    The v2 service turns every non-``None``/``1`` result into ``needs_review``.
    This is deliberately conservative: until a renderer and viewer share an
    orientation transform, guessing would make a correct OCR bbox appear in
    the wrong place.  The existing v1 admission is intentionally unaffected.
    """

    index = 2
    while index + 4 <= len(content):
        if content[index] != 0xFF:
            return 0
        while index < len(content) and content[index] == 0xFF:
            index += 1
        if index >= len(content):
            return 0
        marker = content[index]
        index += 1
        if marker in {0xD8, 0xD9, 0x01} or 0xD0 <= marker <= 0xD7:
            continue
        if index + 2 > len(content):
            return 0
        length = int.from_bytes(content[index : index + 2], "big")
        if length < 2 or index + length > len(content):
            return 0
        payload = content[index + 2 : index + length]
        if marker == 0xE1 and payload.startswith(b"Exif\x00\x00"):
            tiff = payload[6:]
            if len(tiff) < 8 or tiff[:2] not in {b"II", b"MM"}:
                return 0
            byteorder: Literal["little", "big"] = "little" if tiff[:2] == b"II" else "big"
            if int.from_bytes(tiff[2:4], byteorder) != 42:
                return 0
            offset = int.from_bytes(tiff[4:8], byteorder)
            if offset + 2 > len(tiff):
                return 0
            entries = int.from_bytes(tiff[offset : offset + 2], byteorder)
            if offset + 2 + entries * 12 > len(tiff):
                return 0
            for entry_index in range(entries):
                start = offset + 2 + entry_index * 12
                tag = int.from_bytes(tiff[start : start + 2], byteorder)
                if tag != 0x0112:
                    continue
                value_type = int.from_bytes(tiff[start + 2 : start + 4], byteorder)
                count = int.from_bytes(tiff[start + 4 : start + 8], byteorder)
                if value_type != 3 or count != 1:
                    return 0
                value = int.from_bytes(tiff[start + 8 : start + 10], byteorder)
                return value if 1 <= value <= 8 else 0
            return None
        if marker == 0xDA:
            return None
        index += length
    return None


def _pdf_page_hint(content: bytes, settings: ServiceSettings) -> int:
    if len(content) < 9 or not content.startswith(b"%PDF-"):
        raise ServiceHttpError("document_magic_mismatch", 422)
    # This inexpensive guard is deliberately only a hint; a selected local
    # renderer/parser must validate the actual page tree before it returns a
    # preview.  It still bounds obvious uncompressed page-object floods before
    # any optional parser is invoked.
    page_count = len(re.findall(rb"/Type\s*/Page\b", content))
    if page_count > settings.max_document_pages:
        raise ServiceHttpError("document_page_limit_exceeded", 422)
    if b"%%EOF" not in content[-4096:]:
        raise ServiceHttpError("document_decode_invalid", 422)
    return page_count


def inspect_document_structure(
    content: bytes,
    media_type: DocumentMediaKind,
    settings: ServiceSettings,
) -> DocumentStructure:
    """Check magic and inexpensive structural limits before adapter execution."""

    if media_type == "image/png":
        width, height = _png_dimensions(content, settings)
        return DocumentStructure(media_type=media_type, image_width=width, image_height=height)
    if media_type == "image/jpeg":
        width, height = _jpeg_dimensions(content, settings)
        return DocumentStructure(media_type=media_type, image_width=width, image_height=height)
    return DocumentStructure(media_type=media_type, pdf_page_hint=_pdf_page_hint(content, settings))


def inspect_raster_document_structure(
    content: bytes,
    media_type: DocumentMediaKind,
    settings: ServiceSettings,
) -> DocumentStructure:
    """V2 structural inspection with conservative JPEG orientation metadata."""

    structure = inspect_document_structure(content, media_type, settings)
    if media_type != "image/jpeg":
        return structure
    return DocumentStructure(
        media_type=structure.media_type,
        image_width=structure.image_width,
        image_height=structure.image_height,
        pdf_page_hint=structure.pdf_page_hint,
        jpeg_exif_orientation=_jpeg_exif_orientation(content),
    )


async def read_bounded_document_body(
    request: Request,
    settings: ServiceSettings,
) -> InspectedDocumentExtractionRequest:
    """Read one raw document exactly once, bounded and hash-verified.

    The body exists only in process memory for the duration of the explicitly
    local preview call.  It is not put into a JSON payload, logging event, or
    scenario transcript.
    """

    _validate_content_encoding(request)
    media_type = _document_media_type(request)
    metadata = _request_metadata(request, media_type)
    declared_length = _content_length(request, settings)
    content = bytearray()
    digest = hashlib.sha256()
    try:
        async with asyncio.timeout(settings.body_read_timeout_seconds):
            async for chunk in request.stream():
                if len(content) + len(chunk) > settings.max_document_bytes:
                    content.clear()
                    raise ServiceHttpError("document_too_large", 413)
                digest.update(chunk)
                content.extend(chunk)
    except TimeoutError:
        content.clear()
        raise ServiceHttpError("request_body_timeout", 408, retryable=True) from None
    except ClientDisconnect:
        content.clear()
        raise ServiceHttpError("document_request_invalid", 400) from None
    if not content:
        content.clear()
        raise ServiceHttpError("document_request_invalid", 422)
    if declared_length is not None and declared_length != len(content):
        content.clear()
        raise ServiceHttpError("content_length_invalid", 400)
    actual_digest = digest.hexdigest()
    if actual_digest != metadata.source_sha256:
        content.clear()
        raise ServiceHttpError("document_hash_mismatch", 422)
    immutable_content = bytes(content)
    try:
        structure = inspect_document_structure(immutable_content, media_type, settings)
        return InspectedDocumentExtractionRequest(
            preview_input=PreviewInputV1(
                metadata=metadata,
                byte_size=len(immutable_content),
                content=immutable_content,
            ),
            structure=structure,
        )
    except Exception:
        # Ensure a malformed document is not unnecessarily retained by this
        # stack frame before FastAPI maps the content-free error code.
        immutable_content = b""
        raise


async def read_bounded_raster_document_body(
    request: Request,
    settings: ServiceSettings,
) -> InspectedRasterDocumentExtractionRequest:
    """Read the v2 binary request once and bind it to its raster profile.

    This is intentionally separate from the v1 reader so an added v2 header
    can never alter the existing approved PDF preview request shape.
    """

    _validate_content_encoding(request)
    media_type = _document_media_type(request)
    metadata = _raster_request_metadata(request, media_type)
    declared_length = _content_length(request, settings)
    content = bytearray()
    digest = hashlib.sha256()
    try:
        async with asyncio.timeout(settings.body_read_timeout_seconds):
            async for chunk in request.stream():
                if len(content) + len(chunk) > settings.max_document_bytes:
                    content.clear()
                    raise ServiceHttpError("document_too_large", 413)
                digest.update(chunk)
                content.extend(chunk)
    except TimeoutError:
        content.clear()
        raise ServiceHttpError("request_body_timeout", 408, retryable=True) from None
    except ClientDisconnect:
        content.clear()
        raise ServiceHttpError("document_request_invalid", 400) from None
    if not content:
        content.clear()
        raise ServiceHttpError("document_request_invalid", 422)
    if declared_length is not None and declared_length != len(content):
        content.clear()
        raise ServiceHttpError("content_length_invalid", 400)
    if digest.hexdigest() != metadata.source_sha256:
        content.clear()
        raise ServiceHttpError("document_hash_mismatch", 422)
    immutable_content = bytes(content)
    try:
        structure = inspect_raster_document_structure(immutable_content, media_type, settings)
        return InspectedRasterDocumentExtractionRequest(
            raster_input=RasterPreviewInputV2(
                metadata=metadata,
                byte_size=len(immutable_content),
                content=immutable_content,
            ),
            structure=structure,
        )
    except Exception:
        immutable_content = b""
        raise


async def read_bounded_raster_text_document_body(
    request: Request,
    settings: ServiceSettings,
) -> InspectedRasterTextExtractionRequest:
    """Read text-only OCR input once under its own profile identity.

    This intentionally duplicates the narrow v2 admission sequence instead
    of adding an optional header to it.  A caller cannot accidentally turn a
    structured request into a text-only request or vice versa.
    """

    # Keep this independently versioned endpoint aligned with its public
    # schema even if a broader generic document setting is configured.
    # Never admit a body that the result contract cannot safely represent.
    raster_text_max_document_bytes = min(settings.max_document_bytes, 16 * 1024 * 1024)
    _validate_content_encoding(request)
    media_type = _document_media_type(request)
    metadata = _raster_text_request_metadata(request, media_type)
    declared_length = _content_length(request, settings)
    content = bytearray()
    digest = hashlib.sha256()
    try:
        async with asyncio.timeout(settings.body_read_timeout_seconds):
            async for chunk in request.stream():
                if len(content) + len(chunk) > raster_text_max_document_bytes:
                    content.clear()
                    raise ServiceHttpError("document_too_large", 413)
                digest.update(chunk)
                content.extend(chunk)
    except TimeoutError:
        content.clear()
        raise ServiceHttpError("request_body_timeout", 408, retryable=True) from None
    except ClientDisconnect:
        content.clear()
        raise ServiceHttpError("document_request_invalid", 400) from None
    if not content:
        content.clear()
        raise ServiceHttpError("document_request_invalid", 422)
    if declared_length is not None and declared_length != len(content):
        content.clear()
        raise ServiceHttpError("content_length_invalid", 400)
    if digest.hexdigest() != metadata.source_sha256:
        content.clear()
        raise ServiceHttpError("document_hash_mismatch", 422)
    immutable_content = bytes(content)
    try:
        structure = inspect_raster_document_structure(immutable_content, media_type, settings)
        return InspectedRasterTextExtractionRequest(
            raster_text_input=RasterTextInputV1(
                metadata=metadata,
                byte_size=len(immutable_content),
                content=immutable_content,
            ),
            structure=structure,
        )
    except Exception:
        immutable_content = b""
        raise


__all__ = [
    "DocumentStructure",
    "InspectedDocumentExtractionRequest",
    "InspectedRasterDocumentExtractionRequest",
    "InspectedRasterTextExtractionRequest",
    "inspect_document_structure",
    "inspect_raster_document_structure",
    "read_bounded_document_body",
    "read_bounded_raster_document_body",
    "read_bounded_raster_text_document_body",
]
