"""Text-only raster OCR wire models, isolated from the layout/table v2 API.

The existing review workflow consumes human-confirmed text.  This contract
therefore deliberately exposes an editable OCR draft only: it does not claim
document classification, fields, tables, cells, or source geometry.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from customs_ai_agent.schemas.document_extraction import DocumentMediaType


_IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
_REVISION_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
RasterTextPreviewStatus = Literal["preview_ready", "needs_review", "unavailable"]
_MAX_RASTER_TEXT_DOCUMENT_BYTES = 16 * 1024 * 1024
_MAX_RASTER_TEXT_CHARACTERS = 200_000


class RasterTextWireModel(BaseModel):
    """Reject undeclared fields at the text-only binary boundary."""

    model_config = ConfigDict(extra="forbid")


class RasterTextExtractionRequestMetadataV1(RasterTextWireModel):
    """Required headers carried beside one raw PDF/JPEG/PNG body."""

    document_asset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER_PATTERN)
    extraction_request_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER_PATTERN)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    media_type: DocumentMediaType
    raster_text_profile_revision: str = Field(
        min_length=1, max_length=128, pattern=_REVISION_PATTERN
    )


def _normalise_page_text(value: str) -> str:
    return unicodedata.normalize("NFC", value).replace("\r\n", "\n").replace("\r", "\n")


class RasterTextPageV1(RasterTextWireModel):
    """One ordered editable text draft.  No coordinates are promised."""

    page_number: int = Field(ge=1, le=20)
    text: str = Field(min_length=1, max_length=200_000)
    confidence: float | None = Field(default=None, ge=0, le=1)

    @model_validator(mode="after")
    def _safe_normalized_text(self) -> "RasterTextPageV1":
        normalized = _normalise_page_text(self.text)
        if not normalized or "\x00" in normalized:
            raise ValueError("raster text page is invalid")
        self.text = normalized
        return self


class RasterTextWarningV1(RasterTextWireModel):
    """Content-free code safe for logs and Backend job state mapping."""

    code: str = Field(min_length=2, max_length=128, pattern=r"^[a-z][a-z0-9_]{1,127}$")


def canonical_raster_text(pages: list[RasterTextPageV1]) -> str:
    """Return deterministic draft text without introducing table semantics."""

    return "\n\f\n".join(
        _normalise_page_text(page.text)
        for page in sorted(pages, key=lambda item: item.page_number)
    )


def canonical_raster_text_sha256(pages: list[RasterTextPageV1]) -> str:
    return hashlib.sha256(canonical_raster_text(pages).encode("utf-8")).hexdigest()


class RasterTextExtractionResultV1(RasterTextWireModel):
    """Provenance-bound OCR text draft for human correction only."""

    schema_version: Literal["1.0"] = "1.0"
    document_asset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER_PATTERN)
    extraction_request_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER_PATTERN)
    raster_text_profile_revision: str = Field(
        min_length=1, max_length=128, pattern=_REVISION_PATTERN
    )
    status: RasterTextPreviewStatus
    media_type: DocumentMediaType
    # This is deliberately narrower than the generic document limit.  The
    # text-only profile has one bounded CPU lane and a 60-second request
    # deadline, so it must share the Backend's 16 MiB raster-text contract.
    byte_size: int = Field(ge=1, le=_MAX_RASTER_TEXT_DOCUMENT_BYTES)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    canonical_text: str | None = Field(default=None, max_length=_MAX_RASTER_TEXT_CHARACTERS)
    canonical_text_sha256: str | None = Field(default=None, pattern=r"^[a-f0-9]{64}$")
    extractor_revision: str | None = Field(default=None, min_length=1, max_length=256)
    pages: list[RasterTextPageV1] = Field(max_length=20)
    warnings: list[RasterTextWarningV1] = Field(max_length=32)
    needs_review_reasons: list[RasterTextWarningV1] = Field(max_length=32)

    @model_validator(mode="after")
    def _status_content_and_provenance_are_consistent(self) -> "RasterTextExtractionResultV1":
        page_numbers = [page.page_number for page in self.pages]
        if page_numbers != list(range(1, len(page_numbers) + 1)):
            raise ValueError("raster text page numbers must be contiguous and ordered")
        has_content = bool(self.pages)
        has_provenance = (
            self.canonical_text is not None
            or self.canonical_text_sha256 is not None
            or self.extractor_revision is not None
        )
        if self.status == "unavailable":
            if has_content or has_provenance or self.needs_review_reasons:
                raise ValueError("unavailable raster text must not claim extracted content")
            return self
        if has_content:
            expected = canonical_raster_text(self.pages)
            if (
                self.canonical_text != expected
                or self.canonical_text_sha256 != canonical_raster_text_sha256(self.pages)
                or self.extractor_revision is None
            ):
                raise ValueError("raster text content provenance is invalid")
        elif has_provenance:
            raise ValueError("empty raster text must not claim extraction provenance")
        if self.status == "preview_ready":
            if not has_content or self.needs_review_reasons:
                raise ValueError("ready raster text must be complete and review-free")
        if self.status == "needs_review" and not self.needs_review_reasons:
            # A reviewer cannot distinguish a deliberate safety stop from a
            # malformed partial response without an explicit, content-free
            # reason code.  This applies even when a draft page is present.
            raise ValueError("needs-review raster text must state a reason")
        return self


@dataclass(frozen=True, slots=True)
class RasterTextInputV1:
    """Ephemeral binary input that never enters the text-review request."""

    metadata: RasterTextExtractionRequestMetadataV1
    byte_size: int
    content: bytes


@dataclass(frozen=True, slots=True)
class RasterTextPreviewCandidateV1:
    """Provider output before request-specific provenance is bound."""

    status: Literal["preview_ready", "needs_review"]
    extractor_revision: str | None
    pages: list[RasterTextPageV1] | None = None
    warnings: list[RasterTextWarningV1] | None = None
    needs_review_reasons: list[RasterTextWarningV1] | None = None

    def __post_init__(self) -> None:
        pages = self.pages or []
        warnings = self.warnings or []
        reasons = self.needs_review_reasons or []
        RasterTextExtractionResultV1(
            document_asset_id="raster-text-candidate",
            extraction_request_id="raster-text-candidate",
            raster_text_profile_revision="synthetic-raster-text-v1",
            status=self.status,
            media_type="application/pdf",
            byte_size=1,
            source_sha256="0" * 64,
            canonical_text=canonical_raster_text(pages) if pages else None,
            canonical_text_sha256=canonical_raster_text_sha256(pages) if pages else None,
            extractor_revision=self.extractor_revision if pages else None,
            pages=pages,
            warnings=warnings,
            needs_review_reasons=reasons,
        )


def raster_text_result_json(result: RasterTextExtractionResultV1) -> bytes:
    """Canonical JSON helper for contract-focused tests only."""

    return json.dumps(
        result.model_dump(mode="json", warnings=False),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


__all__ = [
    "RasterTextExtractionRequestMetadataV1",
    "RasterTextExtractionResultV1",
    "RasterTextPageV1",
    "RasterTextPreviewCandidateV1",
    "RasterTextPreviewStatus",
    "RasterTextInputV1",
    "RasterTextWarningV1",
    "canonical_raster_text",
    "canonical_raster_text_sha256",
    "raster_text_result_json",
]
