"""Wire and local-only models for the raster document extraction boundary.

The v2 boundary is deliberately parallel to the existing document-preview v1
contract.  It is an OCR/layout *candidate* for a reviewer, never a review
input or an automated customs decision.  Keeping the models separate avoids
silently changing the approved born-digital PDF v1 behaviour.
"""

from __future__ import annotations

import hashlib
import json
import unicodedata
from dataclasses import dataclass
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from customs_ai_agent.schemas.document_extraction import BoundingBoxV1, DocumentMediaType


_IDENTIFIER_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
_REVISION_PATTERN = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
RasterDocumentType = Literal[
    "bill_of_lading",
    "commercial_invoice",
    "packing_list",
    "unknown",
]
RasterDocumentPreviewStatus = Literal["preview_ready", "needs_review", "unavailable"]


class RasterDocumentExtractionWireModel(BaseModel):
    """Reject undeclared fields at the v2 binary-preview boundary."""

    model_config = ConfigDict(extra="forbid")


class RasterDocumentExtractionRequestMetadataV2(RasterDocumentExtractionWireModel):
    """Metadata carried in headers beside one raw PDF/JPEG/PNG body."""

    document_asset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER_PATTERN)
    extraction_request_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER_PATTERN)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    media_type: DocumentMediaType
    raster_profile_revision: str = Field(
        min_length=1, max_length=128, pattern=_REVISION_PATTERN
    )


class NormalizedPointV2(RasterDocumentExtractionWireModel):
    x: float = Field(ge=0, le=1)
    y: float = Field(ge=0, le=1)


class NormalizedPolygonV2(RasterDocumentExtractionWireModel):
    """Optional provider polygon in the same top-left normalized space as bbox."""

    points: list[NormalizedPointV2] = Field(min_length=4, max_length=32)

    @model_validator(mode="after")
    def _contains_distinct_points(self) -> "NormalizedPolygonV2":
        if len({(point.x, point.y) for point in self.points}) < 3:
            raise ValueError("polygon must contain at least three distinct points")
        return self


class RasterDocumentSpanV2(RasterDocumentExtractionWireModel):
    span_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER_PATTERN)
    ordinal: int = Field(ge=1, le=100_000)
    text: str = Field(min_length=1, max_length=8_000)
    bbox: BoundingBoxV1 | None
    polygon: NormalizedPolygonV2 | None
    confidence: float = Field(ge=0, le=1)
    reading_order: int = Field(ge=1, le=100_000)


class RasterDocumentPreviewPageV2(RasterDocumentExtractionWireModel):
    page_number: int = Field(ge=1, le=20)
    width: int = Field(ge=1, le=100_000)
    height: int = Field(ge=1, le=100_000)
    coordinate_unit: Literal["pixels"]
    spans: list[RasterDocumentSpanV2] = Field(max_length=20_000)

    @model_validator(mode="after")
    def _spans_are_unique_and_ordered(self) -> "RasterDocumentPreviewPageV2":
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


class RasterFieldCandidateV2(RasterDocumentExtractionWireModel):
    """Known document field candidate backed by one or more OCR spans."""

    field_id: str = Field(min_length=1, max_length=128, pattern=r"^[A-Za-z][A-Za-z0-9._:-]{0,127}$")
    value: str = Field(min_length=1, max_length=8_000)
    confidence: float = Field(ge=0, le=1)
    evidence_span_ids: list[str] = Field(min_length=1, max_length=256)
    bbox: BoundingBoxV1 | None

    @model_validator(mode="after")
    def _evidence_ids_are_unique(self) -> "RasterFieldCandidateV2":
        if len(self.evidence_span_ids) != len(set(self.evidence_span_ids)):
            raise ValueError("field evidence_span_ids must be unique")
        return self


class RasterTableCellV2(RasterDocumentExtractionWireModel):
    cell_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER_PATTERN)
    row_index: int = Field(ge=0, le=100_000)
    column_index: int = Field(ge=0, le=100_000)
    text: str = Field(min_length=1, max_length=8_000)
    bbox: BoundingBoxV1 | None
    confidence: float = Field(ge=0, le=1)
    evidence_span_ids: list[str] = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def _evidence_ids_are_unique(self) -> "RasterTableCellV2":
        if len(self.evidence_span_ids) != len(set(self.evidence_span_ids)):
            raise ValueError("table-cell evidence_span_ids must be unique")
        return self


class RasterTableRowV2(RasterDocumentExtractionWireModel):
    row_index: int = Field(ge=0, le=100_000)
    cells: list[RasterTableCellV2] = Field(min_length=1, max_length=1_000)

    @model_validator(mode="after")
    def _cells_match_the_row_and_are_unique(self) -> "RasterTableRowV2":
        columns = [cell.column_index for cell in self.cells]
        if any(cell.row_index != self.row_index for cell in self.cells):
            raise ValueError("table-cell row_index must match parent row")
        if len(columns) != len(set(columns)) or columns != sorted(columns):
            raise ValueError("table row columns must be unique and ordered")
        return self


class RasterTableV2(RasterDocumentExtractionWireModel):
    table_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER_PATTERN)
    page_number: int = Field(ge=1, le=20)
    bbox: BoundingBoxV1 | None
    rows: list[RasterTableRowV2] = Field(min_length=1, max_length=10_000)

    @model_validator(mode="after")
    def _rows_are_unique_and_ordered(self) -> "RasterTableV2":
        row_indexes = [row.row_index for row in self.rows]
        if len(row_indexes) != len(set(row_indexes)) or row_indexes != sorted(row_indexes):
            raise ValueError("table rows must be unique and ordered")
        return self


class RasterExtractionWarningV2(RasterDocumentExtractionWireModel):
    """A content-free code safe for logs and reviewer-state mapping."""

    code: str = Field(min_length=1, max_length=128, pattern=r"^[a-z][a-z0-9_]{1,127}$")


def canonical_raster_preview_text(pages: list[RasterDocumentPreviewPageV2]) -> str:
    """Return deterministic candidate text without exposing it to telemetry."""

    return "\n".join(
        unicodedata.normalize("NFC", span.text).replace("\r\n", "\n").replace("\r", "\n")
        for page in sorted(pages, key=lambda item: item.page_number)
        for span in sorted(page.spans, key=lambda item: (item.reading_order, item.ordinal))
    )


def canonical_raster_preview_text_sha256(pages: list[RasterDocumentPreviewPageV2]) -> str:
    return hashlib.sha256(canonical_raster_preview_text(pages).encode("utf-8")).hexdigest()


class RasterDocumentExtractionResultV2(RasterDocumentExtractionWireModel):
    """Provenance-bound v2 OCR/layout candidate for human review only."""

    schema_version: Literal["2.0"] = "2.0"
    document_asset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER_PATTERN)
    extraction_request_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER_PATTERN)
    raster_profile_revision: str = Field(
        min_length=1, max_length=128, pattern=_REVISION_PATTERN
    )
    status: RasterDocumentPreviewStatus
    media_type: DocumentMediaType
    byte_size: int = Field(ge=1, le=32 * 1024 * 1024)
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    canonical_text_sha256: str | None = Field(pattern=r"^[a-f0-9]{64}$")
    extractor_revision: str | None = Field(min_length=1, max_length=256)
    document_type: RasterDocumentType
    classification_confidence: float | None = Field(ge=0, le=1)
    pages: list[RasterDocumentPreviewPageV2] = Field(max_length=20)
    field_candidates: list[RasterFieldCandidateV2] = Field(max_length=256)
    tables: list[RasterTableV2] = Field(max_length=100)
    warnings: list[RasterExtractionWarningV2] = Field(max_length=32)
    needs_review_reasons: list[RasterExtractionWarningV2] = Field(max_length=32)

    @model_validator(mode="after")
    def _status_geometry_and_provenance_are_consistent(
        self,
    ) -> "RasterDocumentExtractionResultV2":
        page_numbers = [page.page_number for page in self.pages]
        if page_numbers != list(range(1, len(page_numbers) + 1)):
            raise ValueError("page_number values must be contiguous and ordered")
        span_ids = {span.span_id for page in self.pages for span in page.spans}
        total_spans = sum(len(page.spans) for page in self.pages)
        if len(span_ids) != total_spans:
            raise ValueError("span_id values must be unique across pages")
        if total_spans > 20_000:
            raise ValueError("raster preview exceeds span limit")
        table_ids = [table.table_id for table in self.tables]
        if len(table_ids) != len(set(table_ids)):
            raise ValueError("table_id values must be unique")
        field_ids = [field.field_id for field in self.field_candidates]
        if len(field_ids) != len(set(field_ids)):
            raise ValueError("field_id values must be unique")
        known_pages = set(page_numbers)
        if any(table.page_number not in known_pages for table in self.tables):
            raise ValueError("table page_number must refer to an extracted page")
        evidence_lists = [field.evidence_span_ids for field in self.field_candidates]
        evidence_lists.extend(
            cell.evidence_span_ids for table in self.tables for row in table.rows for cell in row.cells
        )
        if any(not set(ids) <= span_ids for ids in evidence_lists):
            raise ValueError("field and table evidence must refer to extracted span IDs")
        has_content = bool(self.pages or self.field_candidates or self.tables)
        has_provenance = self.extractor_revision is not None or self.canonical_text_sha256 is not None
        if self.status == "unavailable":
            if (
                has_content
                or has_provenance
                or self.document_type != "unknown"
                or self.classification_confidence is not None
                or self.needs_review_reasons
            ):
                raise ValueError("unavailable raster previews must not claim extracted content")
            return self
        if has_content:
            if self.extractor_revision is None or self.canonical_text_sha256 is None:
                raise ValueError("contentful raster previews require revision and canonical hash")
            if self.canonical_text_sha256 != canonical_raster_preview_text_sha256(self.pages):
                raise ValueError("canonical_text_sha256 does not match preview spans")
        elif has_provenance:
            raise ValueError("empty raster previews must not claim extraction provenance")
        if self.status == "preview_ready":
            if not self.pages or not span_ids:
                raise ValueError("preview_ready raster previews require positioned spans")
            if self.document_type == "unknown" or self.classification_confidence is None:
                raise ValueError("preview_ready raster previews require a known document type")
            if self.needs_review_reasons:
                raise ValueError("preview_ready raster previews must not contain review reasons")
            if any(span.bbox is None for page in self.pages for span in page.spans):
                raise ValueError("preview_ready raster spans require bbox")
            if any(field.bbox is None for field in self.field_candidates):
                raise ValueError("preview_ready raster fields require bbox")
            if any(
                table.bbox is None or any(cell.bbox is None for row in table.rows for cell in row.cells)
                for table in self.tables
            ):
                raise ValueError("preview_ready raster tables require bbox")
        return self


@dataclass(frozen=True, slots=True)
class RasterPreviewInputV2:
    """Ephemeral binary input for a pinned local raster adapter only."""

    metadata: RasterDocumentExtractionRequestMetadataV2
    byte_size: int
    content: bytes


@dataclass(frozen=True, slots=True)
class RasterPreviewCandidateV2:
    """Adapter output before service-owned asset/request provenance is bound."""

    status: Literal["preview_ready", "needs_review"]
    extractor_revision: str | None
    document_type: RasterDocumentType = "unknown"
    classification_confidence: float | None = None
    pages: list[RasterDocumentPreviewPageV2] | None = None
    field_candidates: list[RasterFieldCandidateV2] | None = None
    tables: list[RasterTableV2] | None = None
    warnings: list[RasterExtractionWarningV2] | None = None
    needs_review_reasons: list[RasterExtractionWarningV2] | None = None

    def __post_init__(self) -> None:
        pages = self.pages or []
        fields = self.field_candidates or []
        tables = self.tables or []
        warnings = self.warnings or []
        reasons = self.needs_review_reasons or []
        RasterDocumentExtractionResultV2(
            document_asset_id="raster-preview-candidate",
            extraction_request_id="raster-preview-candidate",
            raster_profile_revision="synthetic-raster-v1",
            status=self.status,
            media_type="application/pdf",
            byte_size=1,
            source_sha256="0" * 64,
            canonical_text_sha256=(canonical_raster_preview_text_sha256(pages) if pages else None),
            extractor_revision=self.extractor_revision,
            document_type=self.document_type,
            classification_confidence=self.classification_confidence,
            pages=pages,
            field_candidates=fields,
            tables=tables,
            warnings=warnings,
            needs_review_reasons=reasons,
        )


def raster_preview_result_json(result: RasterDocumentExtractionResultV2) -> bytes:
    """Canonical JSON helper used by boundary-focused tests."""

    return json.dumps(
        result.model_dump(mode="json", warnings=False),
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


__all__ = [
    "NormalizedPointV2",
    "NormalizedPolygonV2",
    "RasterDocumentExtractionRequestMetadataV2",
    "RasterDocumentExtractionResultV2",
    "RasterDocumentPreviewPageV2",
    "RasterDocumentPreviewStatus",
    "RasterDocumentSpanV2",
    "RasterDocumentType",
    "RasterExtractionWarningV2",
    "RasterFieldCandidateV2",
    "RasterPreviewCandidateV2",
    "RasterPreviewInputV2",
    "RasterTableCellV2",
    "RasterTableRowV2",
    "RasterTableV2",
    "canonical_raster_preview_text",
    "canonical_raster_preview_text_sha256",
    "raster_preview_result_json",
]
