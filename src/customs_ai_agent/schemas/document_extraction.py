"""Wire models for the isolated document-preview boundary.

This module intentionally has no dependency on the review workflow.  A
preview is an untrusted, human-editable extraction candidate; it never becomes
review input until Backend sends separately confirmed text to the existing
text-only review endpoint.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


_IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
_REVISION_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
DocumentMediaType = Literal["application/pdf", "image/jpeg", "image/png"]
DocumentPreviewStatus = Literal["preview_ready", "needs_review", "unavailable"]
PageCoordinateUnit = Literal["pixels", "pdf_points"]


class DocumentExtractionWireModel(BaseModel):
    """Reject undeclared properties at the binary-preview boundary."""

    model_config = ConfigDict(extra="forbid")


class DocumentExtractionRequestMetadataV1(DocumentExtractionWireModel):
    """The JSON-equivalent of the three required binary-request headers.

    The live endpoint carries these fields in HTTP headers because its body is
    raw PDF/JPEG/PNG bytes.  No filename, path, base64 string, or Case data is
    part of this contract.
    """

    document_asset_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=_IDENTIFIER_PATTERN,
    )
    extraction_request_id: str = Field(
        min_length=1,
        max_length=128,
        pattern=_IDENTIFIER_PATTERN,
    )
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    media_type: DocumentMediaType


class BoundingBoxV1(DocumentExtractionWireModel):
    """A normalized, page-relative rectangle for a text span.

    ``None`` is permitted on a span when a text-layer adapter cannot safely
    derive coordinates.  A page image's absolute dimensions remain available
    separately for the frontend's own rendering transform.
    """

    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)
    width: float = Field(gt=0, le=1)
    height: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def _fits_inside_page(self) -> "BoundingBoxV1":
        if self.x + self.width > 1 or self.y + self.height > 1:
            raise ValueError("bounding box must fit inside normalized page coordinates")
        return self


class DocumentSpanV1(DocumentExtractionWireModel):
    """One human-visible candidate text span, never a verified fact."""

    span_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER_PATTERN)
    ordinal: int = Field(ge=1, le=100_000)
    text: str = Field(min_length=1, max_length=8_000)
    bbox: BoundingBoxV1 | None = None
    reading_order: int = Field(ge=1, le=100_000)


class DocumentPreviewPageV1(DocumentExtractionWireModel):
    page_number: int = Field(ge=1, le=1_000)
    width: int = Field(ge=1, le=100_000)
    height: int = Field(ge=1, le=100_000)
    coordinate_unit: PageCoordinateUnit
    spans: list[DocumentSpanV1] = Field(default_factory=list, max_length=10_000)

    @model_validator(mode="after")
    def _spans_are_unique_and_ordered(self) -> "DocumentPreviewPageV1":
        span_ids = [span.span_id for span in self.spans]
        ordinals = [span.ordinal for span in self.spans]
        reading_order = [span.reading_order for span in self.spans]
        if len(span_ids) != len(set(span_ids)):
            raise ValueError("span_id values must be unique within a page")
        if len(ordinals) != len(set(ordinals)):
            raise ValueError("span ordinal values must be unique within a page")
        if len(reading_order) != len(set(reading_order)):
            raise ValueError("span reading_order values must be unique within a page")
        if ordinals != sorted(ordinals) or reading_order != sorted(reading_order):
            raise ValueError("page spans must be ordered")
        return self


class DocumentExtractionWarningV1(DocumentExtractionWireModel):
    """A content-free status code safe for logs and user-facing mapping."""

    code: str = Field(min_length=1, max_length=128, pattern=r"^[a-z][a-z0-9_]{1,127}$")


def canonical_preview_text(pages: list[DocumentPreviewPageV1]) -> str:
    """Build the stable text whose digest binds an extraction preview.

    The canonical form deliberately stays in memory only.  Service telemetry
    contains its digest, never the returned span text or a reconstructed page.
    """

    ordered_pages = sorted(pages, key=lambda page: page.page_number)
    lines = [
        unicodedata.normalize("NFC", span.text).replace("\r\n", "\n").replace("\r", "\n")
        for page in ordered_pages
        for span in sorted(page.spans, key=lambda item: (item.reading_order, item.ordinal))
    ]
    return "\n".join(lines)


def canonical_preview_text_sha256(pages: list[DocumentPreviewPageV1]) -> str:
    return hashlib.sha256(canonical_preview_text(pages).encode("utf-8")).hexdigest()


class DocumentExtractionResultV1(DocumentExtractionWireModel):
    """Metadata and human-review preview for one immutable uploaded asset."""

    schema_version: Literal["1.0"] = "1.0"
    document_asset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER_PATTERN)
    extraction_request_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER_PATTERN)
    status: DocumentPreviewStatus
    media_type: DocumentMediaType
    byte_size: int = Field(ge=1, le=32 * 1024 * 1024)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    canonical_text_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    extractor_revision: str | None = Field(
        default=None,
        min_length=1,
        max_length=128,
        pattern=_REVISION_PATTERN,
    )
    pages: list[DocumentPreviewPageV1] = Field(default_factory=list, max_length=1_000)
    warnings: list[DocumentExtractionWarningV1] = Field(default_factory=list, max_length=32)

    @model_validator(mode="after")
    def _status_and_provenance_are_consistent(self) -> "DocumentExtractionResultV1":
        page_numbers = [page.page_number for page in self.pages]
        if page_numbers != list(range(1, len(page_numbers) + 1)):
            raise ValueError("page_number values must be contiguous and ordered")
        if self.status == "unavailable":
            if (
                self.pages
                or self.canonical_text_sha256 is not None
                or self.extractor_revision is not None
            ):
                raise ValueError("unavailable previews must not claim extracted content")
            return self
        if not self.pages or self.extractor_revision is None or self.canonical_text_sha256 is None:
            raise ValueError("available previews require pages, revision, and canonical text hash")
        if self.canonical_text_sha256 != canonical_preview_text_sha256(self.pages):
            raise ValueError("canonical_text_sha256 does not match preview spans")
        return self


@dataclass(frozen=True, slots=True)
class PreviewInputV1:
    """Ephemeral binary input passed only to an explicitly local adapter."""

    metadata: DocumentExtractionRequestMetadataV1
    byte_size: int
    content: bytes


@dataclass(frozen=True, slots=True)
class PreviewCandidateV1:
    """Adapter output before service-owned provenance is bound."""

    status: Literal["preview_ready", "needs_review"]
    extractor_revision: str
    pages: list[DocumentPreviewPageV1]
    warnings: list[DocumentExtractionWarningV1]

    def __post_init__(self) -> None:
        # Validate every adapter result using the same immutable public model
        # shape, without having the adapter choose asset/request provenance.
        placeholder_hash = "0" * 64
        DocumentExtractionResultV1(
            document_asset_id=self._placeholder_identifier(),
            extraction_request_id=self._placeholder_identifier(),
            status=self.status,
            media_type="application/pdf",
            byte_size=1,
            source_sha256=placeholder_hash,
            canonical_text_sha256=canonical_preview_text_sha256(self.pages),
            extractor_revision=self.extractor_revision,
            pages=self.pages,
            warnings=self.warnings,
        )

    @staticmethod
    def _placeholder_identifier() -> str:
        return "preview-candidate"


def preview_result_json(result: DocumentExtractionResultV1) -> bytes:
    """Return a canonical JSON byte sequence for boundary-focused tests."""

    return json.dumps(
        result.model_dump(mode="json", warnings=False),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


__all__ = [
    "BoundingBoxV1",
    "DocumentExtractionRequestMetadataV1",
    "DocumentExtractionResultV1",
    "DocumentExtractionWarningV1",
    "DocumentMediaType",
    "DocumentPreviewPageV1",
    "DocumentPreviewStatus",
    "DocumentSpanV1",
    "PreviewCandidateV1",
    "PreviewInputV1",
    "canonical_preview_text",
    "canonical_preview_text_sha256",
    "preview_result_json",
]
