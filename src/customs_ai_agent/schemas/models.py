from __future__ import annotations

import base64
import binascii
import hashlib
import json
import math
from datetime import date, datetime, timezone
from decimal import Decimal
from io import BytesIO
from typing import Any, Literal
from urllib.parse import urlsplit
from uuid import uuid4

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from customs_ai_agent.ocr_reading_order import OCR_READING_ORDER_POLICY_ID


DocumentType = Literal[
    "bill_of_lading",
    "invoice",
    "packing_list",
    "certificate_of_origin",
    "unknown",
]
Severity = Literal["low", "medium", "high", "critical"]
ReviewStatus = Literal["completed", "needs_review", "blocked"]
TradeType = Literal["import", "export"]
RetrievalStatus = Literal["ok", "partial", "failed"]
Coverage = Literal["complete", "partial", "unknown"]
HsResolution = Literal["exact", "unique_descendant", "ambiguous", "invalid", "missing"]
ImageMediaType = Literal["image/jpeg", "image/png", "image/webp"]
SourceDocumentMediaType = Literal[
    "image/jpeg",
    "image/png",
    "image/webp",
    "application/pdf",
]
PageTransform = Literal["exif_transpose", "metadata_strip", "pdf_rasterize"]

MAX_IMAGE_BYTES = 5 * 1024 * 1024
MAX_IMAGE_BASE64_CHARS = ((MAX_IMAGE_BYTES + 2) // 3) * 4
MAX_IMAGE_LONG_EDGE = 2576
MAX_IMAGE_VISUAL_TOKENS = 4784
MAX_IMAGE_PAGES_PER_DOCUMENT = 5
MAX_IMAGE_PAGES_PER_RUN = 10
MAX_IMAGE_BYTES_PER_RUN = 20 * 1024 * 1024

_FORBIDDEN_PUBLIC_VALUE_KEYS = {
    "api_key",
    "authorization",
    "content_text",
    "data_base64",
    "documents",
    "dsn",
    "file_path",
    "filename",
    "image_pages",
    "pdf_pages",
    "manifest_path",
    "ocr_ledger_path",
    "raw_prompt",
    "raw_response",
    "source_path",
    "token",
}


def _validate_public_json_value(value: Any, *, depth: int = 0) -> Any:
    """Allow bounded JSON values while rejecting sensitive payload containers."""

    if depth > 8:
        raise ValueError("public structured value exceeds maximum nesting depth")
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("public numeric value must be finite")
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError("public decimal value must be finite")
        return str(value)
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if isinstance(value, str):
        if len(value) > 20_000:
            raise ValueError("public string value is too large")
        return value
    if isinstance(value, (list, tuple)):
        if len(value) > 1000:
            raise ValueError("public list value is too large")
        return [_validate_public_json_value(item, depth=depth + 1) for item in value]
    if isinstance(value, dict):
        if len(value) > 1000:
            raise ValueError("public object value is too large")
        validated: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError("public object keys must be strings")
            normalized_key = key.strip().lower().replace("-", "_")
            if normalized_key in _FORBIDDEN_PUBLIC_VALUE_KEYS:
                raise ValueError("public structured value contains a forbidden sensitive key")
            validated[key] = _validate_public_json_value(item, depth=depth + 1)
        return validated
    raise ValueError("public structured value must be JSON-compatible")


def stable_evidence_id(source_record_id: str, version_hash: str) -> str:
    canonical = f"{source_record_id}\x1f{version_hash}".encode("utf-8")
    return "ev_" + hashlib.sha256(canonical).hexdigest()


class BBox(BaseModel):
    page: int = Field(ge=1)
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    x2: float = Field(ge=0)
    y2: float = Field(ge=0)

    @model_validator(mode="after")
    def _valid_box(self) -> "BBox":
        if self.x2 <= self.x1 or self.y2 <= self.y1:
            raise ValueError("bbox x2/y2 must be greater than x1/y1")
        return self


OcrSpanRole = Literal[
    "text",
    "heading",
    "label",
    "value",
    "table",
    "table_row",
    "table_cell",
]


class OcrLayoutSpan(BaseModel):
    """One provider-neutral OCR/layout observation.

    Span IDs are local stable references within one immutable ledger. Claude
    may map business fields to these observations, but it cannot create or
    modify them.
    """

    model_config = ConfigDict(extra="forbid")

    span_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    page: int = Field(ge=1)
    reading_order: int = Field(ge=0)
    role: OcrSpanRole = "text"
    text: str = Field(min_length=1, max_length=12_000)
    bbox: BBox
    confidence: float = Field(ge=0, le=1)
    table_id: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    row_index: int | None = Field(default=None, ge=0)
    column_index: int | None = Field(default=None, ge=0)
    row_span: int | None = Field(default=None, ge=1, le=100)
    column_span: int | None = Field(default=None, ge=1, le=100)

    @model_validator(mode="after")
    def _valid_span(self) -> "OcrLayoutSpan":
        if self.bbox.page != self.page:
            raise ValueError("OCR span page must match bbox page")
        if not self.text.strip():
            raise ValueError("OCR span text cannot be blank")
        table_structure = (
            self.row_index,
            self.column_index,
            self.row_span,
            self.column_span,
        )
        if any(value is not None for value in table_structure) and self.table_id is None:
            raise ValueError("OCR table structure requires table_id")
        if self.role == "table_cell" and (
            self.table_id is None or self.row_index is None or self.column_index is None
        ):
            raise ValueError("OCR table_cell requires table_id, row_index and column_index")
        if self.role == "table_cell":
            if self.row_span is None:
                self.row_span = 1
            if self.column_span is None:
                self.column_span = 1
        elif self.row_span is not None or self.column_span is not None:
            raise ValueError("OCR cell spans are valid only for table_cell observations")
        return self


class OcrLayoutPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1)
    image_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    width_px: int = Field(ge=1, le=MAX_IMAGE_LONG_EDGE)
    height_px: int = Field(ge=1, le=MAX_IMAGE_LONG_EDGE)
    spans: list[OcrLayoutSpan] = Field(min_length=1, max_length=20_000)

    @model_validator(mode="after")
    def _valid_page(self) -> "OcrLayoutPage":
        span_ids = [span.span_id for span in self.spans]
        if len(span_ids) != len(set(span_ids)):
            raise ValueError("OCR span_id values must be unique within a page")
        reading_order = [span.reading_order for span in self.spans]
        if reading_order != sorted(reading_order):
            raise ValueError("OCR spans must be supplied in reading order")
        for span in self.spans:
            if span.page != self.page_number:
                raise ValueError("OCR span page must match its ledger page")
            if span.bbox.x2 > self.width_px or span.bbox.y2 > self.height_px:
                raise ValueError("OCR span bbox exceeds its source page")
        return self


def stable_ocr_ledger_id(
    *,
    document_id: str,
    provider_id: str,
    engine_id: str,
    engine_version: str,
    reading_order_policy_id: str = OCR_READING_ORDER_POLICY_ID,
    model_id: str = "unattested-model-set",
    model_assets_lock_sha256: str = "0" * 64,
    pages: list[OcrLayoutPage],
) -> str:
    canonical = json.dumps(
        {
            "document_id": document_id,
            "provider_id": provider_id,
            "engine_id": engine_id,
            "engine_version": engine_version,
            "reading_order_policy_id": reading_order_policy_id,
            "model_id": model_id,
            "model_assets_lock_sha256": model_assets_lock_sha256,
            "pages": [page.model_dump(mode="json") for page in pages],
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return "ocr_" + hashlib.sha256(canonical).hexdigest()


class OcrLayoutLedger(BaseModel):
    """Immutable OCR/layout provenance produced independently of Claude."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.0"] = "2.0"
    ledger_id: str = Field(pattern=r"^ocr_[a-f0-9]{64}$")
    document_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    provider_id: str = Field(min_length=1, max_length=128)
    engine_id: str = Field(min_length=1, max_length=128)
    engine_version: str = Field(min_length=1, max_length=128)
    reading_order_policy_id: Literal["worker-emitted-sequence/v1"] = OCR_READING_ORDER_POLICY_ID
    model_id: str = Field(
        default="unattested-model-set",
        min_length=1,
        max_length=256,
    )
    model_assets_lock_sha256: str = Field(
        default="0" * 64,
        pattern=r"^[a-f0-9]{64}$",
    )
    pages: list[OcrLayoutPage] = Field(min_length=1, max_length=MAX_IMAGE_PAGES_PER_DOCUMENT)

    @model_validator(mode="after")
    def _valid_ledger(self) -> "OcrLayoutLedger":
        page_numbers = [page.page_number for page in self.pages]
        if page_numbers != list(range(1, len(self.pages) + 1)):
            raise ValueError("OCR ledger pages must be consecutive and source ordered")
        span_ids = [span.span_id for page in self.pages for span in page.spans]
        if len(span_ids) != len(set(span_ids)):
            raise ValueError("OCR span_id values must be unique within a ledger")
        expected_id = stable_ocr_ledger_id(
            document_id=self.document_id,
            provider_id=self.provider_id,
            engine_id=self.engine_id,
            engine_version=self.engine_version,
            reading_order_policy_id=self.reading_order_policy_id,
            model_id=self.model_id,
            model_assets_lock_sha256=self.model_assets_lock_sha256,
            pages=self.pages,
        )
        if self.ledger_id != expected_id:
            raise ValueError("OCR ledger_id does not match canonical ledger content")
        return self


class ImagePageProvenance(BaseModel):
    """Trace the validated page bytes back to their immutable source."""

    model_config = ConfigDict(extra="forbid")

    source_media_type: SourceDocumentMediaType
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_page_number: int = Field(ge=1)
    transforms: list[PageTransform] = Field(default_factory=list, max_length=4)
    original_exif_orientation: int | None = Field(default=None, ge=1, le=8)
    rasterizer_id: str | None = Field(default=None, min_length=1, max_length=128)
    rasterizer_version: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
    )
    raster_dpi: int | None = Field(default=None, ge=72, le=600)

    @model_validator(mode="after")
    def _valid_provenance(self) -> "ImagePageProvenance":
        if len(self.transforms) != len(set(self.transforms)):
            raise ValueError("page provenance transforms must be unique")
        rasterizer_values = (
            self.rasterizer_id,
            self.rasterizer_version,
            self.raster_dpi,
        )
        if self.source_media_type == "application/pdf":
            if self.transforms != ["pdf_rasterize"] or any(
                value is None for value in rasterizer_values
            ):
                raise ValueError(
                    "PDF page provenance requires one pdf_rasterize transform "
                    "and complete rasterizer metadata"
                )
            if self.original_exif_orientation is not None:
                raise ValueError("PDF provenance cannot contain EXIF orientation")
        else:
            if any(value is not None for value in rasterizer_values):
                raise ValueError("image provenance cannot contain rasterizer metadata")
            orientation = self.original_exif_orientation
            if orientation is None:
                raise ValueError("image provenance requires original EXIF orientation")
            if orientation == 1 and self.transforms != ["metadata_strip"]:
                raise ValueError("unrotated image provenance requires one metadata_strip transform")
            if orientation != 1 and self.transforms != [
                "exif_transpose",
                "metadata_strip",
            ]:
                raise ValueError(
                    "rotated image provenance requires exif_transpose followed by metadata_strip"
                )
            if self.source_page_number != 1:
                raise ValueError("one image file can only identify source page 1")
        return self


class FieldEvidence(BaseModel):
    document_id: str
    page: int = Field(ge=1)
    bbox: BBox | None = None
    raw_text: str | None = None
    text_start: int | None = Field(default=None, ge=0)
    text_end: int | None = Field(default=None, ge=1)
    confidence: float = Field(ge=0, le=1)
    source_modality: Literal["text", "image"] = "text"
    image_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    image_width_px: int | None = Field(default=None, ge=1)
    image_height_px: int | None = Field(default=None, ge=1)
    coordinate_space: Literal["original_image_pixels_top_left"] | None = None
    ocr_ledger_id: str | None = Field(
        default=None,
        pattern=r"^ocr_[a-f0-9]{64}$",
    )
    ocr_span_ids: list[str] = Field(default_factory=list, max_length=128)

    @model_validator(mode="after")
    def _valid_text_span(self) -> "FieldEvidence":
        if (self.text_start is None) != (self.text_end is None):
            raise ValueError("text_start and text_end must be provided together")
        if (
            self.text_start is not None
            and self.text_end is not None
            and self.text_end <= self.text_start
        ):
            raise ValueError("text_end must be greater than text_start")
        image_values = (
            self.image_sha256,
            self.image_width_px,
            self.image_height_px,
            self.coordinate_space,
            self.ocr_ledger_id,
        )
        if self.source_modality == "image":
            if (
                self.bbox is None
                or self.raw_text is None
                or any(value is None for value in image_values)
            ):
                raise ValueError(
                    "image evidence requires independent OCR spans, bbox, raw_text, "
                    "image provenance and coordinate space"
                )
            if not self.ocr_span_ids:
                raise ValueError("image evidence requires at least one OCR span reference")
            if len(self.ocr_span_ids) != len(set(self.ocr_span_ids)):
                raise ValueError("image evidence OCR span references must be unique")
            if self.text_start is not None or self.text_end is not None:
                raise ValueError("image evidence cannot claim a text-buffer span")
        elif any(value is not None for value in image_values) or self.ocr_span_ids:
            raise ValueError("text evidence cannot contain image or OCR provenance")
        return self


class ExtractedField(BaseModel):
    name: str
    value: Any
    normalized_value: Any | None = None
    confidence: float = Field(ge=0, le=1)
    confidence_basis: Literal[
        "rule_heuristic",
        "source_quote_grounded_uncalibrated",
        "model_mapped_to_independent_ocr_uncalibrated",
    ] = "rule_heuristic"
    evidence: FieldEvidence


def _png_container_end(raw: bytes) -> int:
    if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError("image bytes are not an exact supported container")
    offset = 8
    while offset + 12 <= len(raw):
        chunk_length = int.from_bytes(raw[offset : offset + 4], "big")
        chunk_type = raw[offset + 4 : offset + 8]
        chunk_end = offset + 12 + chunk_length
        if chunk_end > len(raw):
            raise ValueError("image bytes are not an exact supported container")
        offset = chunk_end
        if chunk_type == b"IEND":
            if chunk_length != 0:
                raise ValueError("image bytes are not an exact supported container")
            return offset
    raise ValueError("image bytes are not an exact supported container")


def _jpeg_container_end(raw: bytes) -> int:
    if not raw.startswith(b"\xff\xd8"):
        raise ValueError("image bytes are not an exact supported container")
    offset = 2
    in_entropy_data = False
    while offset < len(raw):
        if in_entropy_data:
            marker_prefix = raw.find(b"\xff", offset)
            if marker_prefix < 0:
                break
            marker_index = marker_prefix + 1
            while marker_index < len(raw) and raw[marker_index] == 0xFF:
                marker_index += 1
            if marker_index >= len(raw):
                break
            marker = raw[marker_index]
            if marker == 0x00:
                offset = marker_index + 1
                continue
            if 0xD0 <= marker <= 0xD7:
                offset = marker_index + 1
                continue
            offset = marker_prefix
            in_entropy_data = False
            continue

        if raw[offset] != 0xFF:
            raise ValueError("image bytes are not an exact supported container")
        marker_index = offset + 1
        while marker_index < len(raw) and raw[marker_index] == 0xFF:
            marker_index += 1
        if marker_index >= len(raw):
            break
        marker = raw[marker_index]
        offset = marker_index + 1
        if marker == 0xD9:
            return offset
        if marker in {0x01, 0xD8} or 0xD0 <= marker <= 0xD7:
            continue
        if marker == 0x00 or offset + 2 > len(raw):
            raise ValueError("image bytes are not an exact supported container")
        segment_length = int.from_bytes(raw[offset : offset + 2], "big")
        if segment_length < 2 or offset + segment_length > len(raw):
            raise ValueError("image bytes are not an exact supported container")
        offset += segment_length
        if marker == 0xDA:
            in_entropy_data = True
    raise ValueError("image bytes are not an exact supported container")


def _webp_container_end(raw: bytes) -> int:
    if len(raw) < 12 or raw[:4] != b"RIFF" or raw[8:12] != b"WEBP":
        raise ValueError("image bytes are not an exact supported container")
    return 8 + int.from_bytes(raw[4:8], "little")


def _validate_sanitized_jpeg(raw: bytes) -> None:
    """Allow only the segments emitted by the metadata-stripping JPEG encoder."""

    safe_jfif = b"JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00"
    allowed_segments = frozenset({0xC0, 0xC4, 0xDA, 0xDB, 0xDD, 0xE0})
    offset = 2
    in_entropy_data = False
    segment_count = 0
    jfif_seen = False
    while offset < len(raw):
        if in_entropy_data:
            marker_prefix = raw.find(b"\xff", offset)
            if marker_prefix < 0:
                break
            marker_index = marker_prefix + 1
            while marker_index < len(raw) and raw[marker_index] == 0xFF:
                marker_index += 1
            if marker_index >= len(raw):
                break
            marker = raw[marker_index]
            if marker == 0x00:
                offset = marker_index + 1
                continue
            if 0xD0 <= marker <= 0xD7:
                offset = marker_index + 1
                continue
            offset = marker_prefix
            in_entropy_data = False
            continue

        if raw[offset] != 0xFF:
            raise ValueError("JPEG contains a non-canonical segment")
        marker_index = offset + 1
        while marker_index < len(raw) and raw[marker_index] == 0xFF:
            marker_index += 1
        if marker_index >= len(raw):
            break
        marker = raw[marker_index]
        offset = marker_index + 1
        if marker == 0xD9:
            return
        if marker in {0x01, 0xD8} or 0xD0 <= marker <= 0xD7:
            raise ValueError("JPEG contains a non-canonical marker")
        if offset + 2 > len(raw):
            raise ValueError("JPEG contains a malformed segment")
        segment_length = int.from_bytes(raw[offset : offset + 2], "big")
        segment_end = offset + segment_length
        if segment_length < 2 or segment_end > len(raw):
            raise ValueError("JPEG contains a malformed segment")
        payload = raw[offset + 2 : segment_end]
        if marker not in allowed_segments:
            raise ValueError("JPEG contains non-visual metadata")
        if marker == 0xE0:
            if segment_count != 0 or jfif_seen or payload != safe_jfif:
                raise ValueError("JPEG contains non-canonical application metadata")
            jfif_seen = True
        segment_count += 1
        offset = segment_end
        if marker == 0xDA:
            in_entropy_data = True
    raise ValueError("JPEG container is incomplete")


def _validate_sanitized_png(raw: bytes) -> None:
    """Require the canonical metadata-free PNG chunk topology."""

    offset = 8
    chunk_types: list[bytes] = []
    while offset + 12 <= len(raw):
        chunk_length = int.from_bytes(raw[offset : offset + 4], "big")
        chunk_type = raw[offset + 4 : offset + 8]
        chunk_end = offset + 12 + chunk_length
        if chunk_end > len(raw):
            raise ValueError("PNG contains a malformed chunk")
        chunk_types.append(chunk_type)
        offset = chunk_end
        if chunk_type == b"IEND":
            break
    if (
        offset != len(raw)
        or len(chunk_types) < 3
        or chunk_types[0] != b"IHDR"
        or chunk_types[-1] != b"IEND"
        or any(chunk_type != b"IDAT" for chunk_type in chunk_types[1:-1])
    ):
        raise ValueError("PNG contains non-visual metadata or non-canonical chunks")


def _validate_sanitized_webp(raw: bytes) -> None:
    """Allow one metadata-free lossy or lossless WebP image-data chunk."""

    offset = 12
    chunk_types: list[bytes] = []
    while offset < len(raw):
        if offset + 8 > len(raw):
            raise ValueError("WebP contains a malformed chunk")
        chunk_type = raw[offset : offset + 4]
        chunk_length = int.from_bytes(raw[offset + 4 : offset + 8], "little")
        payload_end = offset + 8 + chunk_length
        padded_end = payload_end + (chunk_length % 2)
        if padded_end > len(raw) or (padded_end != payload_end and raw[payload_end] != 0):
            raise ValueError("WebP contains a malformed chunk")
        chunk_types.append(chunk_type)
        offset = padded_end
    if len(chunk_types) != 1 or chunk_types[0] not in {b"VP8 ", b"VP8L"}:
        raise ValueError("WebP contains non-visual metadata or non-canonical chunks")


def _validate_exact_image_container(raw: bytes, detected_format: str) -> None:
    end_offset_functions = {
        "PNG": _png_container_end,
        "JPEG": _jpeg_container_end,
        "WEBP": _webp_container_end,
    }
    metadata_validators = {
        "PNG": _validate_sanitized_png,
        "JPEG": _validate_sanitized_jpeg,
        "WEBP": _validate_sanitized_webp,
    }
    end_offset = end_offset_functions.get(detected_format)
    metadata_validator = metadata_validators.get(detected_format)
    if end_offset is None or metadata_validator is None or end_offset(raw) != len(raw):
        raise ValueError("image container contains trailing or embedded payload bytes")
    metadata_validator(raw)


class ImagePageInput(BaseModel):
    """One validated, already-rasterized page supplied to Claude Vision.

    `data_base64` is part of the in-memory wire request but is excluded from repr.
    The model verifies bytes, media type, dimensions and hash before any provider
    or Evidence Service call.
    """

    model_config = ConfigDict(extra="forbid")

    page_number: int = Field(ge=1, le=50)
    media_type: ImageMediaType
    data_base64: str = Field(
        min_length=4,
        max_length=MAX_IMAGE_BASE64_CHARS,
        repr=False,
    )
    byte_length: int = Field(ge=1, le=MAX_IMAGE_BYTES)
    sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    width_px: int = Field(ge=1, le=MAX_IMAGE_LONG_EDGE)
    height_px: int = Field(ge=1, le=MAX_IMAGE_LONG_EDGE)
    provenance: ImagePageProvenance

    def decoded_bytes(self) -> bytes:
        try:
            return base64.b64decode(self.data_base64, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("image data_base64 must be canonical base64") from None

    @model_validator(mode="after")
    def _validate_image(self) -> "ImagePageInput":
        raw = self.decoded_bytes()
        if len(raw) != self.byte_length:
            raise ValueError("image byte_length does not match decoded bytes")
        if len(raw) > MAX_IMAGE_BYTES:
            raise ValueError("image exceeds the per-page byte limit")
        if hashlib.sha256(raw).hexdigest() != self.sha256:
            raise ValueError("image sha256 does not match decoded bytes")

        format_to_media_type = {
            "JPEG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
        }
        try:
            with Image.open(BytesIO(raw)) as image:
                detected_format = str(image.format or "").upper()
                detected_size = image.size
                frame_count = int(getattr(image, "n_frames", 1))
                orientation = int(image.getexif().get(274, 1) or 1)
                if detected_size[0] > MAX_IMAGE_LONG_EDGE or detected_size[1] > MAX_IMAGE_LONG_EDGE:
                    raise ValueError("image requires upstream resize before Claude bbox analysis")
                visual_tokens = ((detected_size[0] + 27) // 28) * ((detected_size[1] + 27) // 28)
                if visual_tokens > MAX_IMAGE_VISUAL_TOKENS:
                    raise ValueError("image requires upstream resize before Claude bbox analysis")
            with Image.open(BytesIO(raw)) as verified:
                verified.verify()
            with Image.open(BytesIO(raw)) as decoded:
                decoded.load()
        except ValueError:
            raise
        except (Image.DecompressionBombError, UnidentifiedImageError, OSError):
            raise ValueError("image bytes are not a safe decodable image") from None

        if format_to_media_type.get(detected_format) != self.media_type:
            raise ValueError("image media_type does not match decoded bytes")
        if detected_size != (self.width_px, self.height_px):
            raise ValueError("image dimensions do not match decoded bytes")
        if frame_count != 1:
            raise ValueError("animated or multi-frame images are not supported")
        if orientation != 1:
            raise ValueError("EXIF-rotated images require upstream orientation normalization")
        _validate_exact_image_container(raw, detected_format)
        provenance = self.provenance
        if provenance.source_media_type == "application/pdf":
            if provenance.source_page_number != self.page_number:
                raise ValueError("PDF source page number must match output page number")
        elif provenance.source_media_type != self.media_type:
            raise ValueError("direct image source media type must match output media type")
        if not provenance.transforms and provenance.source_sha256 != self.sha256:
            raise ValueError("untransformed page source hash must match output image hash")
        return self


class PdfPagesInput(BaseModel):
    """Normalized, rasterized PDF modality for the public V2 request.

    The Backend wire never carries a raw PDF, local path, or URL.  A trusted
    edge adapter validates and rasterizes the ephemeral PDF first, then sends
    these source-bound pages.  This makes PDF an explicit modality without
    expanding the core Agent's filesystem or network authority.
    """

    model_config = ConfigDict(extra="forbid")

    source_pdf_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    source_pdf_byte_length: int = Field(ge=1, le=MAX_IMAGE_BYTES_PER_RUN)
    page_count: int = Field(ge=1, le=MAX_IMAGE_PAGES_PER_DOCUMENT)
    rasterizer_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:+-]*$",
    )
    rasterizer_version: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._+-]*$",
    )
    raster_dpi: int = Field(ge=72, le=600)
    pages: list[ImagePageInput] = Field(
        min_length=1,
        max_length=MAX_IMAGE_PAGES_PER_DOCUMENT,
    )

    @model_validator(mode="after")
    def _bind_pages_to_pdf(self) -> "PdfPagesInput":
        if len(self.pages) != self.page_count:
            raise ValueError("PDF page_count must match the rasterized page list")
        if sum(page.byte_length for page in self.pages) > MAX_IMAGE_BYTES_PER_RUN:
            raise ValueError("PDF rasterized pages exceed the per-run byte limit")
        expected_numbers = list(range(1, self.page_count + 1))
        if [page.page_number for page in self.pages] != expected_numbers:
            raise ValueError("PDF rasterized pages must be consecutive and source ordered")
        for page in self.pages:
            provenance = page.provenance
            if (
                provenance.source_media_type != "application/pdf"
                or provenance.source_sha256 != self.source_pdf_sha256
                or provenance.source_page_number != page.page_number
                or provenance.transforms != ["pdf_rasterize"]
                or provenance.rasterizer_id != self.rasterizer_id
                or provenance.rasterizer_version != self.rasterizer_version
                or provenance.raster_dpi != self.raster_dpi
            ):
                raise ValueError("PDF page provenance does not match the PDF modality envelope")
        return self


class DocumentInput(BaseModel):
    document_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    filename: str = Field(min_length=1, max_length=255)
    content_text: str = Field(min_length=1, max_length=200_000)
    declared_type: DocumentType | None = None
    requirement_document_codes: list[str] = Field(default_factory=list)


class DocumentInputV2(BaseModel):
    """Additive multimodal input used only by the V2 Backend contract."""

    model_config = ConfigDict(extra="forbid")

    document_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    filename: str = Field(min_length=1, max_length=255)
    content_text: str | None = Field(default=None, min_length=1, max_length=200_000)
    image_pages: list[ImagePageInput] = Field(
        default_factory=list,
        max_length=MAX_IMAGE_PAGES_PER_DOCUMENT,
    )
    pdf_pages: PdfPagesInput | None = None
    declared_type: DocumentType | None = None
    requirement_document_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _exactly_one_document_modality(self) -> "DocumentInputV2":
        has_text = bool(self.content_text and self.content_text.strip())
        has_images = bool(self.image_pages)
        has_pdf = self.pdf_pages is not None
        if sum((has_text, has_images, has_pdf)) != 1:
            raise ValueError("exactly one of content_text, image_pages or pdf_pages is required")
        if has_images:
            if any(
                page.provenance.source_media_type == "application/pdf" for page in self.image_pages
            ):
                raise ValueError("PDF-derived pages require the explicit pdf_pages modality")
            page_numbers = [page.page_number for page in self.image_pages]
            if page_numbers != list(range(1, len(self.image_pages) + 1)):
                raise ValueError(
                    "image page_number values must be consecutive and source ordered from 1"
                )
        return self


class ReviewRunRequest(BaseModel):
    scenario_id: str | None = Field(default=None, min_length=1, max_length=128)
    case_id: str | None = Field(default=None, min_length=1, max_length=128)
    documents: list[DocumentInput] | None = Field(
        default=None,
        min_length=1,
        max_length=20,
    )
    llm_mode: Literal["mock", "llm"] = "mock"
    review_run_id: str = Field(
        default_factory=lambda: str(uuid4()),
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    review_as_of: date = Field(default_factory=date.today)
    trade_type: TradeType = "import"
    jurisdiction: Literal["KR"] = "KR"
    uploaded_requirement_document_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _scenario_or_documents(self) -> "ReviewRunRequest":
        if bool(self.scenario_id) == bool(self.documents):
            raise ValueError("exactly one of scenario_id or documents is required")
        if self.documents and sum(len(doc.content_text) for doc in self.documents) > 500_000:
            raise ValueError("combined document text exceeds the per-run safety limit")
        if self.documents:
            document_ids = [document.document_id for document in self.documents]
            if len(document_ids) != len(set(document_ids)):
                raise ValueError("document_id values must be unique within a review run")
        self.uploaded_requirement_document_codes = sorted(
            {code.strip() for code in self.uploaded_requirement_document_codes if code.strip()}
        )
        return self


class ReviewRunRequestV2(BaseModel):
    """Additive multimodal request used only by the V2 Backend contract."""

    model_config = ConfigDict(extra="forbid")

    scenario_id: str | None = Field(default=None, min_length=1, max_length=128)
    case_id: str | None = Field(default=None, min_length=1, max_length=128)
    documents: list[DocumentInputV2] | None = Field(
        default=None,
        min_length=1,
        max_length=20,
    )
    llm_mode: Literal["mock", "llm"] = "mock"
    review_run_id: str = Field(
        default_factory=lambda: str(uuid4()),
        min_length=1,
        max_length=128,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._:-]*$",
    )
    review_as_of: date = Field(default_factory=date.today)
    trade_type: TradeType = "import"
    jurisdiction: Literal["KR"] = "KR"
    uploaded_requirement_document_codes: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _scenario_or_documents(self) -> "ReviewRunRequestV2":
        if bool(self.scenario_id) == bool(self.documents):
            raise ValueError("exactly one of scenario_id or documents is required")
        if self.documents and sum(len(doc.content_text or "") for doc in self.documents) > 500_000:
            raise ValueError("combined document text exceeds the per-run safety limit")
        if self.documents:
            document_ids = [document.document_id for document in self.documents]
            if len(document_ids) != len(set(document_ids)):
                raise ValueError("document_id values must be unique within a review run")
            image_pages = [
                page
                for document in self.documents
                for page in (
                    document.image_pages if document.pdf_pages is None else document.pdf_pages.pages
                )
            ]
            if len(image_pages) > MAX_IMAGE_PAGES_PER_RUN:
                raise ValueError("combined image pages exceed the per-run safety limit")
            if sum(page.byte_length for page in image_pages) > MAX_IMAGE_BYTES_PER_RUN:
                raise ValueError("combined image bytes exceed the per-run safety limit")
            if image_pages and self.llm_mode != "llm":
                raise ValueError("image and PDF documents require llm mode")
        self.uploaded_requirement_document_codes = sorted(
            {code.strip() for code in self.uploaded_requirement_document_codes if code.strip()}
        )
        return self


class WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class BBoxV2(BBox):
    """Strict public V2 bounding box without changing the V1 BBox runtime."""

    model_config = ConfigDict(extra="forbid")


class ModelCallRecord(WireModel):
    document_id: str = Field(min_length=1, max_length=128)
    provider: Literal["anthropic"]
    model_id: str = Field(min_length=1, max_length=128)
    prompt_id: str = Field(min_length=1, max_length=128)
    prompt_version: str = Field(min_length=1, max_length=64)
    request_id: str = Field(min_length=1, max_length=256)
    message_id: str = Field(min_length=1, max_length=256)
    stop_reason: str = Field(min_length=1, max_length=64)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    cache_creation_input_tokens: int = Field(ge=0)
    cache_read_input_tokens: int = Field(ge=0)
    latency_ms: int = Field(ge=0)


class ModelCallRecordV2(ModelCallRecord):
    """V2 model-call audit with prompt/schema and OCR/crop provenance."""

    prompt_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    output_schema_id: str | None = Field(default=None, min_length=1, max_length=128)
    output_schema_version: str | None = Field(default=None, min_length=1, max_length=64)
    output_schema_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    input_modality: Literal["text", "image", "image_crop"] = "text"
    ocr_ledger_id: str | None = Field(
        default=None,
        pattern=r"^ocr_[a-f0-9]{64}$",
    )
    ocr_provider_id: str | None = Field(default=None, min_length=1, max_length=128)
    ocr_engine_id: str | None = Field(default=None, min_length=1, max_length=128)
    ocr_engine_version: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
    )
    ocr_model_id: str | None = Field(default=None, min_length=1, max_length=256)
    ocr_model_assets_lock_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    crop_source_image_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    crop_image_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    crop_transform_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    crop_bbox: BBoxV2 | None = None

    @model_validator(mode="after")
    def _valid_ocr_metadata(self) -> "ModelCallRecordV2":
        ocr_values = (
            self.ocr_ledger_id,
            self.ocr_provider_id,
            self.ocr_engine_id,
            self.ocr_engine_version,
            self.ocr_model_id,
            self.ocr_model_assets_lock_sha256,
        )
        if self.input_modality in {"image", "image_crop"} and any(
            value is None for value in ocr_values
        ):
            raise ValueError("image model call requires complete OCR provenance")
        if self.input_modality == "text" and any(value is not None for value in ocr_values):
            raise ValueError("text model call cannot contain OCR provenance")
        crop_values = (
            self.crop_source_image_sha256,
            self.crop_image_sha256,
            self.crop_transform_sha256,
            self.crop_bbox,
        )
        if self.input_modality == "image_crop" and any(value is None for value in crop_values):
            raise ValueError("crop model call requires complete crop provenance")
        if self.input_modality != "image_crop" and any(value is not None for value in crop_values):
            raise ValueError("non-crop model call cannot contain crop provenance")
        return self


class ModelErrorDetail(WireModel):
    document_id: str = Field(min_length=1, max_length=128)
    code: str = Field(min_length=1, max_length=128)
    retryable: bool
    provider: Literal["anthropic"] | None = None
    model_id: str | None = Field(default=None, min_length=1, max_length=128)
    prompt_id: str | None = Field(default=None, min_length=1, max_length=128)
    prompt_version: str | None = Field(default=None, min_length=1, max_length=64)
    latency_ms: int = Field(default=0, ge=0)
    provider_request_id: str | None = Field(default=None, max_length=256)
    provider_status_code: int | None = Field(default=None, ge=100, le=599)


class ModelErrorDetailV2(WireModel):
    """V2 model/OCR error audit without raw provider content."""

    document_id: str = Field(min_length=1, max_length=128)
    code: str = Field(min_length=1, max_length=128)
    retryable: bool
    provider: Literal["anthropic", "ocr_layout"] | None = None
    model_id: str | None = Field(default=None, min_length=1, max_length=128)
    prompt_id: str | None = Field(default=None, min_length=1, max_length=128)
    prompt_version: str | None = Field(default=None, min_length=1, max_length=64)
    prompt_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    latency_ms: int = Field(default=0, ge=0)
    provider_request_id: str | None = Field(default=None, max_length=256)
    provider_status_code: int | None = Field(default=None, ge=100, le=599)
    input_modality: Literal["text", "image", "image_crop"] | None = None


class ModelConflictRecord(WireModel):
    document_id: str = Field(min_length=1, max_length=128)
    field_path: str = Field(min_length=1, max_length=256)
    status: Literal[
        "resolved",
        "crop_not_eligible",
        "crop_budget_exhausted",
        "crop_provider_error",
        "crop_schema_error",
        "crop_unresolved",
    ]
    ocr_ledger_id: str = Field(pattern=r"^ocr_[a-f0-9]{64}$")
    ocr_span_ids: list[str] = Field(min_length=1, max_length=128)
    candidate_value_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    ocr_text_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    crop_attempted: bool
    crop_source_image_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    crop_image_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    crop_transform_sha256: str | None = Field(
        default=None,
        pattern=r"^[a-f0-9]{64}$",
    )
    crop_bbox: BBoxV2 | None = None

    @model_validator(mode="after")
    def _valid_crop_provenance(self) -> "ModelConflictRecord":
        if len(self.ocr_span_ids) != len(set(self.ocr_span_ids)):
            raise ValueError("conflict OCR span references must be unique")
        crop_values = (
            self.crop_source_image_sha256,
            self.crop_image_sha256,
            self.crop_transform_sha256,
            self.crop_bbox,
        )
        if self.crop_attempted and any(value is None for value in crop_values):
            raise ValueError("attempted crop conflict requires crop provenance")
        if not self.crop_attempted and any(value is not None for value in crop_values):
            raise ValueError("unattempted crop conflict cannot claim crop provenance")
        if self.status == "resolved" and not self.crop_attempted:
            raise ValueError("resolved conflict requires a crop attempt")
        return self


class RetrievalQueryItem(WireModel):
    item_id: str = Field(min_length=1, max_length=128)
    description: str = Field(min_length=1, max_length=2000)
    hs_code: str | None = Field(default=None, min_length=1, max_length=64)
    origin_country: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    destination_country: str = Field(pattern=r"^[A-Z]{2}$")


class EvidenceRetrievalRequest(WireModel):
    schema_version: Literal["1.0"] = "1.0"
    request_id: str = Field(min_length=1, max_length=128)
    case_id: str = Field(min_length=1, max_length=128)
    review_run_id: str = Field(min_length=1, max_length=128)
    review_as_of: date
    jurisdiction: Literal["KR"]
    trade_type: TradeType
    top_k: int = Field(default=8, ge=1, le=100)
    items: list[RetrievalQueryItem] = Field(min_length=1, max_length=100)
    uploaded_requirement_document_codes: list[str] = Field(default_factory=list)
    validation_issue_codes: list[str] = Field(default_factory=list)
    query_intent: str = Field(default="required_import_documents", min_length=1, max_length=2000)

    @model_validator(mode="after")
    def _unique_item_ids(self) -> "EvidenceRetrievalRequest":
        item_ids = [item.item_id for item in self.items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("retrieval request item_id values must be unique")
        if len(self.uploaded_requirement_document_codes) != len(
            set(self.uploaded_requirement_document_codes)
        ):
            raise ValueError("uploaded requirement document codes must be unique")
        if len(self.validation_issue_codes) != len(set(self.validation_issue_codes)):
            raise ValueError("validation issue codes must be unique")
        if any(not code or len(code) > 128 for code in self.uploaded_requirement_document_codes):
            raise ValueError("uploaded requirement document codes must be 1..128 chars")
        if any(not code or len(code) > 128 for code in self.validation_issue_codes):
            raise ValueError("validation issue codes must be 1..128 chars")
        return self


class UserCorrection(BaseModel):
    document_id: str
    field_path: str
    value: Any
    confidence: float = Field(default=1.0, ge=0, le=1)


class ValidationIssue(BaseModel):
    issue_code: str
    severity: Severity
    message: str
    expected_value: str | None = None
    observed_value: str | None = None
    source_document_id: str | None = None
    target_document_id: str | None = None
    source_bbox: BBox | None = None
    target_bbox: BBox | None = None
    confidence: float = Field(default=0.95, ge=0, le=1)
    confidence_basis: Literal["deterministic_rule", "source_field_minimum"] = "deterministic_rule"
    needs_regulatory_review: bool = False


class ValidationIssueV2(ValidationIssue):
    model_config = ConfigDict(extra="forbid")

    source_bbox: BBoxV2 | None = None
    target_bbox: BBoxV2 | None = None


class RetrievalErrorDetail(WireModel):
    code: str = Field(min_length=1)
    message: str = Field(min_length=1)
    retryable: bool = False


class RagEvidence(WireModel):
    evidence_id: str = Field(pattern=r"^ev_[a-f0-9]{64}$")
    source_record_id: str = Field(min_length=1)
    source_type: Literal["law_article", "customs_requirement", "hsk_entry", "notice"]
    item_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    text: str = Field(min_length=1)
    law_id: str | None = None
    law_name: str | None = None
    article_no: str | None = None
    source_uri: str = Field(min_length=1)
    valid_from: date | None = None
    valid_to: date | None = None
    version_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    matched_hs_code: str | None = None
    trade_type: TradeType
    required_document_code: str | None = None
    required_document_name: str | None = None
    agencies: list[str] = Field(default_factory=list)
    retrieval_methods: list[Literal["exact_hs", "dense", "bm25", "graph"]] = Field(
        default_factory=list
    )
    eligibility: Literal["finding", "supporting", "blocked"]
    score: float | None = Field(default=None, ge=0, le=1)
    raw_scores: dict[str, float] = Field(default_factory=dict)

    @field_validator("source_uri")
    @classmethod
    def _valid_source_uri(cls, value: str) -> str:
        try:
            parsed = urlsplit(value)
        except ValueError as exc:
            raise ValueError("source_uri must be an absolute URI") from exc
        if value != value.strip() or not parsed.scheme or not (parsed.netloc or parsed.path):
            raise ValueError("source_uri must be an absolute URI")
        return value

    @model_validator(mode="after")
    def _valid_evidence_metadata(self) -> "RagEvidence":
        if self.evidence_id != stable_evidence_id(self.source_record_id, self.version_hash):
            raise ValueError("evidence_id must be SHA-256 of source_record_id and version_hash")
        if self.matched_hs_code and not (
            self.matched_hs_code.isdigit() and len(self.matched_hs_code) in {6, 10}
        ):
            raise ValueError("matched_hs_code must be 6 or 10 digits")
        if len(self.agencies) != len(set(self.agencies)):
            raise ValueError("agencies must be unique")
        if len(self.retrieval_methods) != len(set(self.retrieval_methods)):
            raise ValueError("retrieval_methods must be unique")
        if self.valid_from and self.valid_to and self.valid_to < self.valid_from:
            raise ValueError("valid_to cannot be before valid_from")
        if self.eligibility == "finding":
            if self.source_type != "customs_requirement":
                raise ValueError(
                    "finding eligibility is reserved for structured customs requirements"
                )
            required_text = (
                self.source_record_id,
                self.title,
                self.text,
                self.law_id,
                self.law_name,
                self.source_uri,
                self.matched_hs_code,
                self.required_document_code,
                self.required_document_name,
            )
            if (
                self.valid_from is None
                or any(not value or not value.strip() for value in required_text)
                or not self.agencies
                or any(not agency.strip() for agency in self.agencies)
                or not self.retrieval_methods
            ):
                raise ValueError(
                    "finding customs requirement requires source, law, date, matched HS, "
                    "document, agency and retrieval provenance"
                )
        return self


class RetrievalItemResult(WireModel):
    item_id: str = Field(min_length=1)
    hs_resolution: HsResolution
    resolved_hsk10: str | None = Field(default=None, pattern=r"^\d{10}$")
    candidate_hsk10s: list[str] = Field(default_factory=list)
    coverage: Coverage
    requirements_found: bool
    evidence: list[RagEvidence] = Field(default_factory=list)
    errors: list[RetrievalErrorDetail] = Field(default_factory=list)

    @model_validator(mode="after")
    def _evidence_is_item_bound(self) -> "RetrievalItemResult":
        if any(item.item_id != self.item_id for item in self.evidence):
            raise ValueError("every evidence item_id must match its retrieval result item_id")
        if any(not (len(code) == 10 and code.isdigit()) for code in self.candidate_hsk10s):
            raise ValueError("candidate_hsk10s must contain only 10-digit codes")
        if len(self.candidate_hsk10s) != len(set(self.candidate_hsk10s)):
            raise ValueError("candidate_hsk10s must be unique")
        evidence_ids = [item.evidence_id for item in self.evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("evidence_id values must be unique within an item result")
        if self.hs_resolution in {"exact", "unique_descendant"}:
            if self.resolved_hsk10 is None:
                raise ValueError(f"resolved_hsk10 is required for {self.hs_resolution}")
            if self.candidate_hsk10s not in ([], [self.resolved_hsk10]):
                raise ValueError(
                    "successful HS resolution can contain only its resolved_hsk10 candidate"
                )
        elif self.hs_resolution == "ambiguous":
            if self.resolved_hsk10 is not None or len(self.candidate_hsk10s) < 2:
                raise ValueError(
                    "ambiguous resolution requires no resolved code and at least two candidates"
                )
        elif self.resolved_hsk10 is not None or self.candidate_hsk10s:
            raise ValueError(
                f"{self.hs_resolution} resolution cannot contain resolved or candidate codes"
            )
        finding_requirements = [
            item
            for item in self.evidence
            if item.source_type == "customs_requirement" and item.eligibility == "finding"
        ]
        if self.requirements_found != bool(finding_requirements):
            raise ValueError(
                "requirements_found must match finding-eligible customs requirement evidence"
            )
        return self


class EvidenceRetrievalResponse(WireModel):
    schema_version: Literal["1.0"] = "1.0"
    request_id: str = Field(min_length=1)
    status: RetrievalStatus
    knowledge_snapshot_id: str | None = None
    results: list[RetrievalItemResult] = Field(default_factory=list)
    errors: list[RetrievalErrorDetail] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_results(self) -> "EvidenceRetrievalResponse":
        item_ids = [result.item_id for result in self.results]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("retrieval response result item_id values must be unique")
        has_result_problem = any(
            result.coverage != "complete" or bool(result.errors) for result in self.results
        )
        if self.status == "ok" and (
            not self.knowledge_snapshot_id or self.errors or has_result_problem
        ):
            raise ValueError("ok response requires a snapshot and complete, error-free results")
        if self.status == "partial" and not (self.errors or has_result_problem):
            raise ValueError("partial response requires an incomplete or errored result")
        if self.status == "failed" and not (
            self.errors or any(result.errors for result in self.results)
        ):
            raise ValueError("failed response requires a response or result error")
        return self


class RegulatoryFinding(BaseModel):
    finding_code: str
    severity: Severity
    message: str
    item_id: str = Field(min_length=1)
    required_document_code: str = Field(min_length=1)
    required_document_name: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)
    confidence_basis: Literal["retrieval_score", "structured_requirement_policy"] = (
        "structured_requirement_policy"
    )


class RegulatoryFindingV2(RegulatoryFinding):
    model_config = ConfigDict(extra="forbid")

    source: FieldSourceReferenceV2 | None = None

    @model_validator(mode="after")
    def _require_source_for_material_finding(self) -> "RegulatoryFindingV2":
        if self.severity in {"critical", "high"} and self.source is None:
            raise ValueError("High/Critical regulatory finding requires a source reference")
        if (
            self.severity in {"critical", "high"}
            and self.source is not None
            and self.source.source_modality == "text"
            and (self.source.text_start is None or self.source.text_end is None)
        ):
            raise ValueError("High/Critical text finding requires an exact source span")
        return self


class AuditEvent(BaseModel):
    stage: str
    event: str
    message: str
    timestamp: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict[str, Any] = Field(default_factory=dict)


class AuditEventV2(AuditEvent):
    model_config = ConfigDict(extra="forbid")

    @field_validator("metadata", mode="before")
    @classmethod
    def _safe_public_metadata(cls, value: Any) -> Any:
        validated = _validate_public_json_value(value)
        if not isinstance(validated, dict):
            raise ValueError("audit metadata must be a JSON object")
        return validated


class XaiReport(BaseModel):
    case_id: str
    overall_status: ReviewStatus
    summary: dict[str, Any]
    document_summary: list[dict[str, Any]]
    issues: list[dict[str, Any]]
    regulatory_findings: list[dict[str, Any]]
    rag_evidence: list[dict[str, Any]]
    disclaimer: str = (
        "This result is a pre-clearance review aid. Final legal decisions must "
        "be made by a responsible human reviewer."
    )


XaiRawScoreMethodV2 = Literal[
    "exact_hs",
    "dense",
    "bm25",
    "graph",
    "rrf",
    "lexical_overlap",
    "mock",
]


class XaiReportSummaryV2(WireModel):
    total_documents: int = Field(ge=0)
    total_issues: int = Field(ge=0)
    critical: int = Field(ge=0)
    high: int = Field(ge=0)
    medium: int = Field(ge=0)
    low: int = Field(ge=0)
    average_extraction_confidence: float = Field(ge=0, le=1)
    retrieval_status: RetrievalStatus | None = None
    retrieval_coverage: Coverage | None = None
    knowledge_snapshot_id: str | None = Field(default=None, min_length=1, max_length=256)
    outcome_reasons: list[str] = Field(default_factory=list, max_length=256)


class XaiDocumentSummaryV2(WireModel):
    document_id: str = Field(min_length=1, max_length=128)
    document_type: DocumentType
    extraction_status: Literal["completed", "failed"]
    extraction_confidence: float = Field(ge=0, le=1)


class XaiBBoxV2(WireModel):
    page: int = Field(ge=1)
    x1: float = Field(ge=0)
    y1: float = Field(ge=0)
    x2: float = Field(ge=0)
    y2: float = Field(ge=0)

    @model_validator(mode="after")
    def _valid_box(self) -> "XaiBBoxV2":
        BBox.model_validate(self.model_dump(mode="python"))
        return self


class XaiValidationIssueV2(WireModel):
    issue_code: str = Field(min_length=1, max_length=128)
    severity: Severity
    message: str = Field(min_length=1, max_length=4000)
    expected_value: str | None = Field(default=None, max_length=4000)
    observed_value: str | None = Field(default=None, max_length=4000)
    source_document_id: str | None = Field(default=None, min_length=1, max_length=128)
    target_document_id: str | None = Field(default=None, min_length=1, max_length=128)
    source_bbox: XaiBBoxV2 | None = None
    target_bbox: XaiBBoxV2 | None = None
    confidence: float = Field(ge=0, le=1)
    confidence_basis: Literal[
        "deterministic_rule",
        "source_field_minimum",
    ]
    needs_regulatory_review: bool
    recommendation: str = Field(min_length=1, max_length=2000)

    @model_validator(mode="after")
    def _valid_domain_issue(self) -> "XaiValidationIssueV2":
        payload = self.model_dump(mode="python", exclude={"recommendation"})
        ValidationIssue.model_validate(payload)
        return self


class XaiRegulatoryFindingV2(WireModel):
    finding_code: str = Field(min_length=1, max_length=128)
    severity: Severity
    message: str = Field(min_length=1, max_length=4000)
    item_id: str = Field(min_length=1, max_length=128)
    required_document_code: str = Field(min_length=1, max_length=128)
    required_document_name: str = Field(min_length=1, max_length=512)
    evidence_ids: list[str] = Field(min_length=1, max_length=256)
    confidence: float = Field(ge=0, le=1)
    confidence_basis: Literal[
        "retrieval_score",
        "structured_requirement_policy",
    ]
    source: FieldSourceReferenceV2 | None = None

    @model_validator(mode="after")
    def _valid_domain_finding(self) -> "XaiRegulatoryFindingV2":
        RegulatoryFindingV2.model_validate(self.model_dump(mode="python"))
        return self


class XaiEvidenceScoreV2(WireModel):
    method: XaiRawScoreMethodV2
    score: float = Field(allow_inf_nan=False)


class XaiRagEvidenceV2(WireModel):
    evidence_id: str = Field(pattern=r"^ev_[a-f0-9]{64}$")
    source_record_id: str = Field(min_length=1, max_length=512)
    source_type: Literal["law_article", "customs_requirement", "hsk_entry", "notice"]
    item_id: str = Field(min_length=1, max_length=128)
    title: str = Field(min_length=1, max_length=2000)
    text: str = Field(min_length=1, max_length=20_000)
    law_id: str | None = Field(default=None, max_length=512)
    law_name: str | None = Field(default=None, max_length=2000)
    article_no: str | None = Field(default=None, max_length=512)
    source_uri: str = Field(min_length=1, max_length=4000)
    valid_from: date | None = None
    valid_to: date | None = None
    version_hash: str = Field(pattern=r"^[a-f0-9]{64}$")
    matched_hs_code: str | None = Field(default=None, max_length=64)
    trade_type: TradeType
    required_document_code: str | None = Field(default=None, max_length=128)
    required_document_name: str | None = Field(default=None, max_length=512)
    agencies: list[str] = Field(default_factory=list, max_length=256)
    retrieval_methods: list[Literal["exact_hs", "dense", "bm25", "graph"]] = Field(
        default_factory=list,
        max_length=4,
    )
    eligibility: Literal["finding", "supporting", "blocked"]
    score: float | None = Field(default=None, ge=0, le=1)
    raw_scores: list[XaiEvidenceScoreV2] = Field(default_factory=list, max_length=7)

    @field_validator("raw_scores", mode="before")
    @classmethod
    def _normalize_raw_scores(cls, value: Any) -> Any:
        if not isinstance(value, dict):
            return value
        return [{"method": method, "score": score} for method, score in sorted(value.items())]

    @model_validator(mode="after")
    def _valid_domain_evidence(self) -> "XaiRagEvidenceV2":
        payload = self.model_dump(mode="python")
        payload["raw_scores"] = {raw_score.method: raw_score.score for raw_score in self.raw_scores}
        RagEvidence.model_validate(payload)
        return self


class XaiReportV2(WireModel):
    case_id: str = Field(min_length=1, max_length=128)
    overall_status: ReviewStatus
    summary: XaiReportSummaryV2
    document_summary: list[XaiDocumentSummaryV2] = Field(
        default_factory=list,
        max_length=20,
    )
    issues: list[XaiValidationIssueV2] = Field(default_factory=list, max_length=1000)
    regulatory_findings: list[XaiRegulatoryFindingV2] = Field(
        default_factory=list,
        max_length=1000,
    )
    rag_evidence: list[XaiRagEvidenceV2] = Field(default_factory=list, max_length=10_000)
    disclaimer: str = Field(min_length=1, max_length=2000)


class FieldSourceReferenceV2(WireModel):
    """Redacted source reference for one public V2 value.

    The extracted value is carried by its parent field. This reference deliberately
    omits the full OCR text and all raw document bytes while preserving enough
    immutable provenance to locate and independently audit the observation.
    """

    document_id: str = Field(min_length=1, max_length=128)
    page: int = Field(ge=1)
    source_modality: Literal["text", "image"]
    bbox: BBoxV2 | None = None
    text_start: int | None = Field(default=None, ge=0)
    text_end: int | None = Field(default=None, ge=1)
    image_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    image_width_px: int | None = Field(default=None, ge=1)
    image_height_px: int | None = Field(default=None, ge=1)
    coordinate_space: Literal["original_image_pixels_top_left"] | None = None
    ocr_ledger_id: str | None = Field(
        default=None,
        pattern=r"^ocr_[a-f0-9]{64}$",
    )
    ocr_span_ids: list[str] = Field(default_factory=list, max_length=128)

    @model_validator(mode="after")
    def _complete_reference(self) -> "FieldSourceReferenceV2":
        if (self.text_start is None) != (self.text_end is None):
            raise ValueError("text offsets must be supplied together")
        if (
            self.text_start is not None
            and self.text_end is not None
            and self.text_end <= self.text_start
        ):
            raise ValueError("text_end must be greater than text_start")
        image_values = (
            self.image_sha256,
            self.image_width_px,
            self.image_height_px,
            self.coordinate_space,
            self.ocr_ledger_id,
        )
        if self.source_modality == "image":
            if (
                self.bbox is None
                or any(value is None for value in image_values)
                or not self.ocr_span_ids
            ):
                raise ValueError(
                    "image source reference requires bbox, image hash, dimensions, "
                    "coordinate space and independent OCR spans"
                )
            if self.text_start is not None or self.text_end is not None:
                raise ValueError("image source reference cannot contain text offsets")
            if self.bbox is not None and self.bbox.page != self.page:
                raise ValueError("image source bbox page must match source page")
            if (
                self.bbox is not None
                and self.image_width_px is not None
                and self.image_height_px is not None
                and (self.bbox.x2 > self.image_width_px or self.bbox.y2 > self.image_height_px)
            ):
                raise ValueError("image source bbox exceeds its source image")
        elif any(value is not None for value in image_values) or self.ocr_span_ids:
            raise ValueError("text source reference cannot claim image/OCR provenance")
        if len(self.ocr_span_ids) != len(set(self.ocr_span_ids)):
            raise ValueError("OCR span references must be unique")
        return self


class ReviewFieldV2(WireModel):
    name: str = Field(min_length=1, max_length=128)
    value: Any
    normalized_value: Any | None = None
    confidence: float = Field(ge=0, le=1)
    confidence_basis: Literal[
        "rule_heuristic",
        "source_quote_grounded_uncalibrated",
        "model_mapped_to_independent_ocr_uncalibrated",
    ]
    source: FieldSourceReferenceV2

    @field_validator("value", "normalized_value", mode="before")
    @classmethod
    def _safe_public_value(cls, value: Any) -> Any:
        return _validate_public_json_value(value)


class ReviewLineItemV2(WireModel):
    item_id: str = Field(min_length=1, max_length=128)
    fields: list[ReviewFieldV2] = Field(default_factory=list)


class OcrProvenanceV2(WireModel):
    schema_version: Literal["2.0"]
    ledger_id: str = Field(pattern=r"^ocr_[a-f0-9]{64}$")
    provider_id: str = Field(min_length=1, max_length=128)
    engine_id: str = Field(min_length=1, max_length=128)
    engine_version: str = Field(min_length=1, max_length=128)
    reading_order_policy_id: Literal["worker-emitted-sequence/v1"]
    model_id: str = Field(min_length=1, max_length=256)
    model_assets_lock_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class ReviewDocumentResultV2(WireModel):
    document_id: str = Field(min_length=1, max_length=128)
    document_type: DocumentType
    input_modality: Literal["text", "image"]
    extraction_status: Literal["completed", "failed"]
    extraction_confidence: float = Field(ge=0, le=1)
    extraction_provider: str = Field(min_length=1, max_length=128)
    model_id: str | None = Field(default=None, max_length=128)
    prompt_version: str | None = Field(default=None, max_length=64)
    ocr_layout: OcrProvenanceV2 | None = None
    fields: list[ReviewFieldV2] = Field(default_factory=list)
    line_items: list[ReviewLineItemV2] = Field(default_factory=list)

    @model_validator(mode="after")
    def _complete_document_provenance(self) -> "ReviewDocumentResultV2":
        if (
            self.input_modality == "image"
            and self.extraction_status == "completed"
            and self.ocr_layout is None
        ):
            raise ValueError("completed image result requires independent OCR provenance")
        if self.input_modality == "text" and self.ocr_layout is not None:
            raise ValueError("text result cannot claim OCR provenance")
        if self.extraction_status == "failed":
            if self.fields or self.line_items:
                raise ValueError("failed document result cannot contain fields or line items")
            if self.document_type != "unknown":
                raise ValueError("failed document result must use unknown document type")
            if self.extraction_confidence != 0:
                raise ValueError("failed document result must use zero confidence")
        field_names = [field.name for field in self.fields]
        if len(field_names) != len(set(field_names)):
            raise ValueError("document field names must be unique")
        item_ids = [item.item_id for item in self.line_items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("line item IDs must be unique")
        all_fields = [
            *self.fields,
            *(field for item in self.line_items for field in item.fields),
        ]
        for field in all_fields:
            if field.source.document_id != self.document_id:
                raise ValueError("field source document must match its parent document")
            if field.source.source_modality != self.input_modality:
                raise ValueError("field source modality must match its parent document")
            if self.input_modality == "image":
                if self.ocr_layout is None:
                    raise ValueError("image fields require independent OCR provenance")
                if field.source.ocr_ledger_id != self.ocr_layout.ledger_id:
                    raise ValueError("field source OCR ledger must match its parent document")
        return self


class _ReviewRunResultBase(BaseModel):
    case_id: str
    review_run_id: str
    review_as_of: date
    trade_type: TradeType
    jurisdiction: str
    status: ReviewStatus
    requires_human_review: bool
    retrieval_status: RetrievalStatus
    retrieval_coverage: Coverage
    knowledge_snapshot_id: str | None = None
    retrieval_request: EvidenceRetrievalRequest | None = None
    retrieval_results: list[RetrievalItemResult] = Field(default_factory=list)
    retrieval_errors: list[RetrievalErrorDetail] = Field(default_factory=list)
    validation_issues: list[ValidationIssue] = Field(default_factory=list)
    regulatory_findings: list[RegulatoryFinding] = Field(default_factory=list)
    rag_evidence: list[RagEvidence] = Field(default_factory=list)
    # Concrete wire versions override this compatibility slot with their
    # version-specific report model.
    report: Any
    audit_trail: list[AuditEvent] = Field(default_factory=list)


class ReviewRunResult(_ReviewRunResultBase):
    schema_version: Literal["1.0"] = "1.0"
    report: XaiReport
    model_calls: list[ModelCallRecord] = Field(default_factory=list)
    model_errors: list[ModelErrorDetail] = Field(default_factory=list)


class ReviewRunResultV2(_ReviewRunResultBase):
    """Additive, redacted multimodal result contract.

    V1 remains unchanged for text-only consumers. V2 adds structured document
    outputs and independent OCR references without exposing source payloads,
    full OCR text, prompts, provider responses or local paths.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["2.0"] = "2.0"
    # V2 deliberately narrows the inherited mutable wire field so source
    # grounding cannot be dropped during serialization.
    regulatory_findings: list[RegulatoryFindingV2] = Field(  # type: ignore[assignment]
        default_factory=list
    )
    report: XaiReportV2
    model_calls: list[ModelCallRecordV2] = Field(default_factory=list)
    model_errors: list[ModelErrorDetailV2] = Field(default_factory=list)
    model_conflicts: list[ModelConflictRecord] = Field(default_factory=list)
    document_results: list[ReviewDocumentResultV2] = Field(default_factory=list)

    @field_validator("validation_issues", mode="before")
    @classmethod
    def _strict_validation_issues(cls, value: Any) -> Any:
        if not isinstance(value, (list, tuple)):
            return value
        return [ValidationIssueV2.model_validate(item) for item in value]

    @field_validator("regulatory_findings", mode="before")
    @classmethod
    def _strict_regulatory_findings(cls, value: Any) -> Any:
        if not isinstance(value, (list, tuple)):
            return value
        return [RegulatoryFindingV2.model_validate(item) for item in value]

    @field_validator("audit_trail", mode="before")
    @classmethod
    def _strict_audit_trail(cls, value: Any) -> Any:
        if not isinstance(value, (list, tuple)):
            return value
        return [AuditEventV2.model_validate(item) for item in value]

    @model_validator(mode="after")
    def _complete_model_schema_provenance(self) -> "ReviewRunResultV2":
        for call in self.model_calls:
            if any(
                value is None
                for value in (
                    call.output_schema_id,
                    call.output_schema_version,
                    call.output_schema_sha256,
                )
            ):
                raise ValueError("V2 model calls require output schema identity and checksum")
        if self.case_id != self.report.case_id:
            raise ValueError("top-level case_id must match report case_id")
        if self.status != self.report.overall_status:
            raise ValueError("top-level status must match report overall_status")
        if self.requires_human_review != (self.status != "completed"):
            raise ValueError("requires_human_review must match the final status")

        summary = self.report.summary
        if summary.retrieval_status != self.retrieval_status:
            raise ValueError("report retrieval_status must match the top-level result")
        if summary.retrieval_coverage != self.retrieval_coverage:
            raise ValueError("report retrieval_coverage must match the top-level result")
        if summary.knowledge_snapshot_id != self.knowledge_snapshot_id:
            raise ValueError("report knowledge_snapshot_id must match the top-level result")
        retrieval_result_ids = [item.item_id for item in self.retrieval_results]
        if len(retrieval_result_ids) != len(set(retrieval_result_ids)):
            raise ValueError("retrieval result item IDs must be unique")
        if self.retrieval_request is None:
            if (
                self.retrieval_status != "failed"
                or self.retrieval_results
                or self.rag_evidence
                or self.knowledge_snapshot_id is not None
            ):
                raise ValueError(
                    "result without a retrieval request must fail without Evidence data"
                )
        else:
            request = self.retrieval_request
            if (
                request.case_id != self.case_id
                or request.review_run_id != self.review_run_id
                or request.review_as_of != self.review_as_of
                or request.trade_type != self.trade_type
                or request.jurisdiction != self.jurisdiction
            ):
                raise ValueError("retrieval request identity must match the review result")
            request_item_ids = [item.item_id for item in request.items]
            if not set(retrieval_result_ids).issubset(request_item_ids):
                raise ValueError("retrieval results must belong to the retrieval request")
            EvidenceRetrievalResponse(
                request_id=request.request_id,
                status=self.retrieval_status,
                knowledge_snapshot_id=self.knowledge_snapshot_id,
                results=self.retrieval_results,
                errors=self.retrieval_errors,
            )
            if self.retrieval_status == "ok" and set(retrieval_result_ids) != set(request_item_ids):
                raise ValueError("ok retrieval must cover every requested item")

        if not self.retrieval_results:
            derived_coverage: Coverage = "unknown"
        elif all(item.coverage == "complete" for item in self.retrieval_results):
            derived_coverage = "complete"
        elif any(item.coverage == "unknown" for item in self.retrieval_results):
            derived_coverage = "unknown"
        else:
            derived_coverage = "partial"
        if self.retrieval_coverage != derived_coverage:
            raise ValueError("retrieval_coverage must match the item-level Evidence results")
        if self.status == "completed" and (
            self.retrieval_request is None
            or self.retrieval_status != "ok"
            or self.retrieval_coverage != "complete"
            or not self.knowledge_snapshot_id
            or self.retrieval_errors
            or not self.retrieval_results
            or any(
                item.coverage != "complete"
                or item.errors
                or item.hs_resolution not in {"exact", "unique_descendant"}
                or not item.resolved_hsk10
                for item in self.retrieval_results
            )
        ):
            raise ValueError("completed result requires complete, error-free Evidence retrieval")

        if summary.total_documents != len(self.report.document_summary):
            raise ValueError("report total_documents must match document_summary")
        if summary.total_documents != len(self.document_results):
            raise ValueError("report total_documents must match document_results")
        document_summaries = {item.document_id: item for item in self.report.document_summary}
        if len(document_summaries) != len(self.report.document_summary):
            raise ValueError("report document IDs must be unique")
        document_results = {item.document_id: item for item in self.document_results}
        if len(document_results) != len(self.document_results):
            raise ValueError("document result IDs must be unique")
        if document_summaries.keys() != document_results.keys():
            raise ValueError("report documents must match public document results")
        for document_id, result in document_results.items():
            report_document = document_summaries[document_id]
            if (
                report_document.document_type != result.document_type
                or report_document.extraction_status != result.extraction_status
                or report_document.extraction_confidence != result.extraction_confidence
            ):
                raise ValueError("report document summary must match public document result")

        for call in self.model_calls:
            document = document_results.get(call.document_id)
            if document is None:
                raise ValueError("model call must reference a public document result")
            if call.input_modality == "text":
                if document.input_modality != "text":
                    raise ValueError("text model call must reference a text document")
            else:
                layout = document.ocr_layout
                if document.input_modality != "image" or layout is None:
                    raise ValueError("image model call requires its document OCR provenance")
                if (
                    call.ocr_ledger_id != layout.ledger_id
                    or call.ocr_provider_id != layout.provider_id
                    or call.ocr_engine_id != layout.engine_id
                    or call.ocr_engine_version != layout.engine_version
                    or call.ocr_model_id != layout.model_id
                    or call.ocr_model_assets_lock_sha256 != layout.model_assets_lock_sha256
                ):
                    raise ValueError("model call OCR provenance must match its document")
        for error in self.model_errors:
            document = document_results.get(error.document_id)
            if document is None:
                raise ValueError("model error must reference a public document result")
            if error.input_modality == "text" and document.input_modality != "text":
                raise ValueError("text model error must reference a text document")
            if error.input_modality in {"image", "image_crop"} and (
                document.input_modality != "image"
            ):
                raise ValueError("image model error must reference an image document")
        for conflict in self.model_conflicts:
            document = document_results.get(conflict.document_id)
            if (
                document is None
                or document.input_modality != "image"
                or document.ocr_layout is None
                or conflict.ocr_ledger_id != document.ocr_layout.ledger_id
            ):
                raise ValueError("model conflict must match its image document OCR provenance")

        expected_total_issues = len(self.validation_issues) + len(self.regulatory_findings)
        if summary.total_issues != expected_total_issues:
            raise ValueError("report total_issues must match top-level findings and issues")
        severity_counts = {
            severity: (
                sum(item.severity == severity for item in self.validation_issues)
                + sum(item.severity == severity for item in self.regulatory_findings)
            )
            for severity in ("critical", "high", "medium", "low")
        }
        if any(getattr(summary, severity) != count for severity, count in severity_counts.items()):
            raise ValueError("report severity counts must match top-level findings and issues")

        report_issues = [
            item.model_dump(mode="json", exclude={"recommendation"}) for item in self.report.issues
        ]
        top_level_issues = [item.model_dump(mode="json") for item in self.validation_issues]
        if report_issues != top_level_issues:
            raise ValueError("report issues must match top-level validation issues")
        report_findings = [item.model_dump(mode="json") for item in self.report.regulatory_findings]
        top_level_findings = [item.model_dump(mode="json") for item in self.regulatory_findings]
        if report_findings != top_level_findings:
            raise ValueError("report findings must match top-level regulatory findings")

        report_evidence = []
        for item in self.report.rag_evidence:
            payload = item.model_dump(mode="python")
            payload["raw_scores"] = {score.method: score.score for score in item.raw_scores}
            report_evidence.append(RagEvidence.model_validate(payload).model_dump(mode="json"))
        top_level_evidence = [item.model_dump(mode="json") for item in self.rag_evidence]
        if report_evidence != top_level_evidence:
            raise ValueError("report evidence must match top-level RAG evidence")
        retrieval_evidence = [
            evidence.model_dump(mode="json")
            for result in self.retrieval_results
            for evidence in result.evidence
        ]
        if retrieval_evidence != top_level_evidence:
            raise ValueError("public RAG evidence must match item-level retrieval evidence")
        evidence_ids = [item.evidence_id for item in self.rag_evidence]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("public RAG evidence IDs must be globally unique")
        evidence_by_id = {item.evidence_id: item for item in self.rag_evidence}
        line_item_sources: dict[str, FieldSourceReferenceV2] = {}
        for document in self.document_results:
            for line_item in document.line_items:
                if line_item.item_id in line_item_sources:
                    raise ValueError("public line item IDs must be globally unique")
                hs_fields = [field for field in line_item.fields if field.name == "hs_code"]
                if len(hs_fields) == 1:
                    line_item_sources[line_item.item_id] = hs_fields[0].source
        for evidence in self.rag_evidence:
            if evidence.trade_type != self.trade_type:
                raise ValueError("public RAG evidence trade type must match the review")
        for finding in self.regulatory_findings:
            source = line_item_sources.get(finding.item_id)
            if finding.severity in {"critical", "high"} and (
                source is None or finding.source != source
            ):
                raise ValueError("regulatory finding must bind to its public line-item HS source")
            if finding.source is not None and finding.source != source:
                raise ValueError(
                    "regulatory finding source must match its public line-item HS source"
                )
            if len(finding.evidence_ids) != len(set(finding.evidence_ids)):
                raise ValueError("finding evidence references must be unique")
            for evidence_id in finding.evidence_ids:
                cited_evidence = evidence_by_id.get(evidence_id)
                if (
                    cited_evidence is None
                    or cited_evidence.item_id != finding.item_id
                    or cited_evidence.source_type != "customs_requirement"
                    or cited_evidence.eligibility != "finding"
                    or cited_evidence.required_document_code != finding.required_document_code
                    or cited_evidence.required_document_name != finding.required_document_name
                ):
                    raise ValueError(
                        "regulatory finding must cite matching finding-eligible Evidence"
                    )
        return self


class ChatAnswer(BaseModel):
    answer: str
    evidence_ids: list[str] = Field(default_factory=list)
    grounded: bool
