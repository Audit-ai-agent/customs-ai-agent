"""Disabled-by-default raster OCR/layout execution boundary.

This module intentionally does not choose, download, or activate an OCR
model.  It defines the narrow local seams a future approved PP-StructureV3
profile must satisfy while preserving the existing PDF text-layer v1 path.
The provider receives raster pages only; it never receives original PDF bytes.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Protocol, Sequence

from customs_ai_agent.schemas.document_extraction import BoundingBoxV1
from customs_ai_agent.schemas.raster_document_extraction import (
    NormalizedPolygonV2,
    RasterDocumentExtractionResultV2,
    RasterDocumentPreviewPageV2,
    RasterDocumentType,
    RasterDocumentSpanV2,
    RasterExtractionWarningV2,
    RasterFieldCandidateV2,
    RasterPreviewCandidateV2,
    RasterPreviewInputV2,
    RasterTableCellV2,
    RasterTableRowV2,
    RasterTableV2,
    canonical_raster_preview_text_sha256,
)
from customs_ai_agent.service.document_input import DocumentStructure
from customs_ai_agent.service.settings import ServiceSettings


_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")


class RasterDocumentPreviewError(RuntimeError):
    """A content-free raster adapter failure safe for the HTTP boundary."""


class RasterDocumentPreviewUnavailable(RasterDocumentPreviewError):
    """No locally approved raster profile is installed."""


class RasterDocumentPreviewInputError(RasterDocumentPreviewError):
    """An adapter rejected unsafe or structurally inconsistent decoded input."""


class RasterDocumentPreviewUnsupported(RasterDocumentPreviewError):
    """A valid document is deliberately outside the sealed raster-v1 scope."""


class RasterDocumentPreviewExecutionError(RasterDocumentPreviewError):
    """An adapter failed without returning provider diagnostics to callers."""


@dataclass(frozen=True, slots=True)
class RasterizedPageV2:
    """One decoded/rasterized page passed to the OCR provider.

    ``pixels`` may contain a local image representation selected by the input
    adapter.  It must never be placed in result JSON, telemetry, or receipts.
    For PDF requests the input adapter, not the provider, is the only code
    that sees the original PDF bytes.
    """

    page_number: int
    width: int
    height: int
    pixels: bytes

    def __post_init__(self) -> None:
        if self.page_number < 1 or self.width < 1 or self.height < 1 or not self.pixels:
            raise ValueError("raster page is invalid")


@dataclass(frozen=True, slots=True)
class RasterLedgerSpanV2:
    """Provider-neutral, in-memory OCR span ledger entry."""

    span_id: str
    page_number: int
    ordinal: int
    reading_order: int
    text: str
    bbox: BoundingBoxV1 | None
    confidence: float
    polygon: NormalizedPolygonV2 | None = None

    def __post_init__(self) -> None:
        if (
            _IDENTIFIER.fullmatch(self.span_id) is None
            or self.page_number < 1
            or self.ordinal < 1
            or self.reading_order < 1
            or not self.text
            or not 0 <= self.confidence <= 1
        ):
            raise ValueError("raster ledger span is invalid")


@dataclass(frozen=True, slots=True)
class RasterLedgerFieldV2:
    field_id: str
    value: str
    confidence: float
    evidence_span_ids: tuple[str, ...]
    bbox: BoundingBoxV1 | None

    def __post_init__(self) -> None:
        if (
            _IDENTIFIER.fullmatch(self.field_id) is None
            or not self.value
            or not 0 <= self.confidence <= 1
            or not self.evidence_span_ids
            or len(self.evidence_span_ids) != len(set(self.evidence_span_ids))
        ):
            raise ValueError("raster ledger field is invalid")


@dataclass(frozen=True, slots=True)
class RasterLedgerCellV2:
    cell_id: str
    page_number: int
    row_index: int
    column_index: int
    text: str
    bbox: BoundingBoxV1 | None
    confidence: float
    evidence_span_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if (
            _IDENTIFIER.fullmatch(self.cell_id) is None
            or self.page_number < 1
            or self.row_index < 0
            or self.column_index < 0
            or not self.text
            or not 0 <= self.confidence <= 1
            or not self.evidence_span_ids
            or len(self.evidence_span_ids) != len(set(self.evidence_span_ids))
        ):
            raise ValueError("raster ledger cell is invalid")


@dataclass(frozen=True, slots=True)
class RasterLedgerTableV2:
    table_id: str
    page_number: int
    bbox: BoundingBoxV1 | None
    cells: tuple[RasterLedgerCellV2, ...]

    def __post_init__(self) -> None:
        if (
            _IDENTIFIER.fullmatch(self.table_id) is None
            or self.page_number < 1
            or not self.cells
            or any(cell.page_number != self.page_number for cell in self.cells)
        ):
            raise ValueError("raster ledger table is invalid")
        positions = [(cell.row_index, cell.column_index) for cell in self.cells]
        if len(positions) != len(set(positions)):
            raise ValueError("raster ledger table contains duplicate cells")


@dataclass(frozen=True, slots=True)
class RasterOcrLayoutLedgerV2:
    """Internal, provider-neutral OCR/layout representation.

    It deliberately retains provider revision and source binding locally so a
    result can be audited without exposing provider diagnostics to a browser.
    """

    source_sha256: str
    raster_profile_revision: str
    provider_revision: str
    pages: tuple[RasterizedPageV2, ...]
    spans: tuple[RasterLedgerSpanV2, ...]
    document_type: RasterDocumentType = "unknown"
    classification_confidence: float | None = None
    fields: tuple[RasterLedgerFieldV2, ...] = ()
    tables: tuple[RasterLedgerTableV2, ...] = ()
    warnings: tuple[RasterExtractionWarningV2, ...] = ()
    needs_review_reasons: tuple[RasterExtractionWarningV2, ...] = ()

    def __post_init__(self) -> None:
        if (
            _SHA256.fullmatch(self.source_sha256) is None
            or _IDENTIFIER.fullmatch(self.raster_profile_revision) is None
            or _IDENTIFIER.fullmatch(self.provider_revision) is None
            or not self.pages
        ):
            raise ValueError("raster OCR/layout ledger identity is invalid")
        page_numbers = [page.page_number for page in self.pages]
        if page_numbers != list(range(1, len(page_numbers) + 1)):
            raise ValueError("raster ledger page numbers must be contiguous")
        span_ids = [span.span_id for span in self.spans]
        if len(span_ids) != len(set(span_ids)) or any(
            span.page_number not in set(page_numbers) for span in self.spans
        ):
            raise ValueError("raster ledger spans are invalid")
        if self.document_type == "unknown" and self.classification_confidence is not None:
            raise ValueError("unknown document type must not claim classification confidence")
        if self.document_type != "unknown" and self.classification_confidence is None:
            raise ValueError("known document type requires classification confidence")
        if (
            self.classification_confidence is not None
            and not 0 <= self.classification_confidence <= 1
        ):
            raise ValueError("classification confidence is invalid")
        if len({field.field_id for field in self.fields}) != len(self.fields):
            raise ValueError("raster ledger fields must be unique")
        known_span_ids = set(span_ids)
        if any(not set(field.evidence_span_ids) <= known_span_ids for field in self.fields):
            raise ValueError("raster ledger field evidence is invalid")
        if len({table.table_id for table in self.tables}) != len(self.tables):
            raise ValueError("raster ledger tables must be unique")
        if any(
            not set(cell.evidence_span_ids) <= known_span_ids
            for table in self.tables
            for cell in table.cells
        ):
            raise ValueError("raster ledger table evidence is invalid")


class RasterInputAdapter(Protocol):
    """Strict local decode/rasterization seam before OCR provider invocation."""

    def rasterize_pdf(
        self, raster_input: RasterPreviewInputV2, structure: DocumentStructure
    ) -> Sequence[RasterizedPageV2]: ...

    def decode_image(
        self, raster_input: RasterPreviewInputV2, structure: DocumentStructure
    ) -> RasterizedPageV2: ...


class RasterOcrLayoutProvider(Protocol):
    """Pinned local OCR/layout provider seam (e.g. a future PP-StructureV3 profile)."""

    def extract_layout(
        self,
        pages: Sequence[RasterizedPageV2],
        *,
        source_sha256: str,
        raster_profile_revision: str,
    ) -> RasterOcrLayoutLedgerV2: ...


class LocalRasterDocumentPreviewEngine(Protocol):
    def extract(
        self, raster_input: RasterPreviewInputV2, structure: DocumentStructure
    ) -> RasterPreviewCandidateV2: ...


@dataclass(frozen=True, slots=True)
class MediaAwareRasterPreviewEngine:
    """Route PDF and image inputs without exposing a PDF to the OCR provider."""

    input_adapter: RasterInputAdapter
    provider: RasterOcrLayoutProvider

    def prepare_pages(
        self, raster_input: RasterPreviewInputV2, structure: DocumentStructure
    ) -> tuple[RasterizedPageV2, ...]:
        """Apply the service's strict PDF/image decode rule without OCR."""

        media_type = raster_input.metadata.media_type
        if media_type == "application/pdf":
            pages = tuple(self.input_adapter.rasterize_pdf(raster_input, structure))
        elif media_type in {"image/jpeg", "image/png"}:
            pages = (self.input_adapter.decode_image(raster_input, structure),)
        else:  # Defensive even though the Pydantic metadata model rejects it.
            raise RasterDocumentPreviewInputError("raster media type is unsupported")
        if not pages:
            raise RasterDocumentPreviewInputError("raster input adapter returned no pages")
        return pages

    @staticmethod
    def preview_from_ledger(
        *,
        raster_input: RasterPreviewInputV2,
        pages: Sequence[RasterizedPageV2],
        ledger: RasterOcrLayoutLedgerV2,
    ) -> RasterPreviewCandidateV2:
        """Validate a provider ledger and project it through the service shape."""

        if ledger.source_sha256 != raster_input.metadata.source_sha256:
            raise RasterDocumentPreviewExecutionError("raster ledger source binding is invalid")
        if ledger.raster_profile_revision != raster_input.metadata.raster_profile_revision:
            raise RasterDocumentPreviewExecutionError("raster ledger profile binding is invalid")
        if tuple(ledger.pages) != pages:
            raise RasterDocumentPreviewExecutionError("raster ledger page binding is invalid")
        return raster_ledger_to_preview_candidate(ledger)

    def extract(
        self, raster_input: RasterPreviewInputV2, structure: DocumentStructure
    ) -> RasterPreviewCandidateV2:
        pages = self.prepare_pages(raster_input, structure)
        ledger = self.provider.extract_layout(
            pages,
            source_sha256=raster_input.metadata.source_sha256,
            raster_profile_revision=raster_input.metadata.raster_profile_revision,
        )
        return self.preview_from_ledger(raster_input=raster_input, pages=pages, ledger=ledger)


class DisabledRasterDocumentPreviewEngine:
    """Safe default until a passing raster approval is explicitly wired in."""

    def extract(
        self, _raster_input: RasterPreviewInputV2, _structure: DocumentStructure
    ) -> RasterPreviewCandidateV2:
        raise RasterDocumentPreviewUnavailable("raster preview engine is not enabled")


def raster_ledger_to_preview_candidate(ledger: RasterOcrLayoutLedgerV2) -> RasterPreviewCandidateV2:
    """Project the internal ledger to the public human-review candidate shape."""

    spans_by_page: dict[int, list[RasterDocumentSpanV2]] = {
        page.page_number: [] for page in ledger.pages
    }
    for span in sorted(
        ledger.spans, key=lambda item: (item.page_number, item.reading_order, item.ordinal)
    ):
        spans_by_page[span.page_number].append(
            RasterDocumentSpanV2(
                span_id=span.span_id,
                ordinal=span.ordinal,
                text=span.text,
                bbox=span.bbox,
                polygon=span.polygon,
                confidence=span.confidence,
                reading_order=span.reading_order,
            )
        )
    pages = [
        RasterDocumentPreviewPageV2(
            page_number=page.page_number,
            width=page.width,
            height=page.height,
            coordinate_unit="pixels",
            spans=spans_by_page[page.page_number],
        )
        for page in ledger.pages
    ]
    fields = [
        RasterFieldCandidateV2(
            field_id=field.field_id,
            value=field.value,
            confidence=field.confidence,
            evidence_span_ids=list(field.evidence_span_ids),
            bbox=field.bbox,
        )
        for field in ledger.fields
    ]
    tables = [
        RasterTableV2(
            table_id=table.table_id,
            page_number=table.page_number,
            bbox=table.bbox,
            rows=[
                RasterTableRowV2(
                    row_index=row_index,
                    cells=[
                        RasterTableCellV2(
                            cell_id=cell.cell_id,
                            row_index=cell.row_index,
                            column_index=cell.column_index,
                            text=cell.text,
                            bbox=cell.bbox,
                            confidence=cell.confidence,
                            evidence_span_ids=list(cell.evidence_span_ids),
                        )
                        for cell in sorted(
                            (item for item in table.cells if item.row_index == row_index),
                            key=lambda item: item.column_index,
                        )
                    ],
                )
                for row_index in sorted({cell.row_index for cell in table.cells})
            ],
        )
        for table in ledger.tables
    ]
    reasons = list(ledger.needs_review_reasons)
    if ledger.document_type == "unknown":
        reasons.append(RasterExtractionWarningV2(code="document_type_unknown"))
    if not ledger.spans:
        reasons.append(RasterExtractionWarningV2(code="ocr_text_not_found"))
    # Nullable geometry is intentionally valid on the wire only for a
    # reviewer-held draft.  Never let the later preview_ready invariant turn
    # this expected condition into a 5xx response.
    if any(span.bbox is None for span in ledger.spans):
        reasons.append(RasterExtractionWarningV2(code="layout_incomplete"))
    if any(field.bbox is None for field in ledger.fields):
        reasons.append(RasterExtractionWarningV2(code="layout_incomplete"))
    if any(
        table.bbox is None or any(cell.bbox is None for cell in table.cells)
        for table in ledger.tables
    ):
        reasons.append(RasterExtractionWarningV2(code="table_structure_incomplete"))
    reasons = list({reason.code: reason for reason in reasons}.values())
    status: Literal["preview_ready", "needs_review"] = (
        "preview_ready"
        if ledger.document_type != "unknown" and ledger.spans and not reasons
        else "needs_review"
    )
    return RasterPreviewCandidateV2(
        status=status,
        extractor_revision=ledger.provider_revision,
        document_type=ledger.document_type,
        classification_confidence=ledger.classification_confidence,
        pages=pages,
        field_candidates=fields,
        tables=tables,
        warnings=list(ledger.warnings),
        needs_review_reasons=reasons,
    )


def _needs_review_for_jpeg_orientation(
    raster_input: RasterPreviewInputV2,
    structure: DocumentStructure,
) -> RasterDocumentExtractionResultV2 | None:
    orientation = structure.jpeg_exif_orientation
    if raster_input.metadata.media_type != "image/jpeg" or orientation in {None, 1}:
        return None
    return RasterDocumentExtractionResultV2(
        document_asset_id=raster_input.metadata.document_asset_id,
        extraction_request_id=raster_input.metadata.extraction_request_id,
        raster_profile_revision=raster_input.metadata.raster_profile_revision,
        status="needs_review",
        media_type=raster_input.metadata.media_type,
        byte_size=raster_input.byte_size,
        source_sha256=raster_input.metadata.source_sha256,
        canonical_text_sha256=None,
        extractor_revision=None,
        document_type="unknown",
        classification_confidence=None,
        pages=[],
        field_candidates=[],
        tables=[],
        warnings=[RasterExtractionWarningV2(code="jpeg_exif_orientation_unsupported")],
        needs_review_reasons=[RasterExtractionWarningV2(code="jpeg_exif_orientation_unsupported")],
    )


def _needs_review_for_unsupported_input(
    raster_input: RasterPreviewInputV2, *, code: str
) -> RasterDocumentExtractionResultV2:
    """Return an empty draft when a valid document is outside v1's Gate."""

    warning = RasterExtractionWarningV2(code=code)
    return RasterDocumentExtractionResultV2(
        document_asset_id=raster_input.metadata.document_asset_id,
        extraction_request_id=raster_input.metadata.extraction_request_id,
        raster_profile_revision=raster_input.metadata.raster_profile_revision,
        status="needs_review",
        media_type=raster_input.metadata.media_type,
        byte_size=raster_input.byte_size,
        source_sha256=raster_input.metadata.source_sha256,
        canonical_text_sha256=None,
        extractor_revision=None,
        document_type="unknown",
        classification_confidence=None,
        pages=[],
        field_candidates=[],
        tables=[],
        warnings=[warning],
        needs_review_reasons=[warning],
    )


def raster_preview_candidate_to_result(
    *, raster_input: RasterPreviewInputV2, candidate: RasterPreviewCandidateV2
) -> RasterDocumentExtractionResultV2:
    """Bind an already validated candidate to the public result contract."""

    pages = candidate.pages or []
    return RasterDocumentExtractionResultV2(
        document_asset_id=raster_input.metadata.document_asset_id,
        extraction_request_id=raster_input.metadata.extraction_request_id,
        raster_profile_revision=raster_input.metadata.raster_profile_revision,
        status=candidate.status,
        media_type=raster_input.metadata.media_type,
        byte_size=raster_input.byte_size,
        source_sha256=raster_input.metadata.source_sha256,
        canonical_text_sha256=(canonical_raster_preview_text_sha256(pages) if pages else None),
        extractor_revision=candidate.extractor_revision,
        document_type=candidate.document_type,
        classification_confidence=candidate.classification_confidence,
        pages=pages,
        field_candidates=candidate.field_candidates or [],
        tables=candidate.tables or [],
        warnings=candidate.warnings or [],
        needs_review_reasons=candidate.needs_review_reasons or [],
    )


def extract_raster_document_preview(
    engine: LocalRasterDocumentPreviewEngine,
    *,
    raster_input: RasterPreviewInputV2,
    structure: DocumentStructure,
    settings: ServiceSettings,
) -> RasterDocumentExtractionResultV2:
    """Bind a local raster candidate to immutable asset provenance.

    No fallback model is selected.  An unapproved profile remains an explicit
    content-free ``unavailable`` result, while EXIF rotations are deliberately
    routed to review until an orientation-preserving coordinate contract exists.
    """

    if raster_input.byte_size > settings.max_document_bytes:
        raise RasterDocumentPreviewInputError("raster input byte size is invalid")
    orientation_result = _needs_review_for_jpeg_orientation(raster_input, structure)
    if orientation_result is not None:
        return orientation_result
    try:
        candidate = engine.extract(raster_input, structure)
    except RasterDocumentPreviewUnsupported as error:
        # The source-attested synthetic Gate proves a single-page family only.
        # A valid multi-page PDF must not quietly enter an unmeasured OCR path.
        return _needs_review_for_unsupported_input(raster_input, code=str(error))
    except RasterDocumentPreviewUnavailable:
        return RasterDocumentExtractionResultV2(
            document_asset_id=raster_input.metadata.document_asset_id,
            extraction_request_id=raster_input.metadata.extraction_request_id,
            raster_profile_revision=raster_input.metadata.raster_profile_revision,
            status="unavailable",
            media_type=raster_input.metadata.media_type,
            byte_size=raster_input.byte_size,
            source_sha256=raster_input.metadata.source_sha256,
            canonical_text_sha256=None,
            extractor_revision=None,
            document_type="unknown",
            classification_confidence=None,
            pages=[],
            field_candidates=[],
            tables=[],
            warnings=[RasterExtractionWarningV2(code="raster_engine_unavailable")],
            needs_review_reasons=[],
        )
    return raster_preview_candidate_to_result(raster_input=raster_input, candidate=candidate)


__all__ = [
    "DisabledRasterDocumentPreviewEngine",
    "LocalRasterDocumentPreviewEngine",
    "MediaAwareRasterPreviewEngine",
    "RasterDocumentPreviewError",
    "RasterDocumentPreviewExecutionError",
    "RasterDocumentPreviewInputError",
    "RasterDocumentPreviewUnsupported",
    "RasterDocumentPreviewUnavailable",
    "RasterInputAdapter",
    "RasterLedgerCellV2",
    "RasterLedgerFieldV2",
    "RasterLedgerSpanV2",
    "RasterLedgerTableV2",
    "RasterOcrLayoutLedgerV2",
    "RasterOcrLayoutProvider",
    "RasterizedPageV2",
    "extract_raster_document_preview",
    "raster_preview_candidate_to_result",
    "raster_ledger_to_preview_candidate",
]
