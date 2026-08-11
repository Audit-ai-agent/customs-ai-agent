"""Local-only document-preview engine boundary.

No concrete OCR/VLM package is selected here.  An operator or test may inject a
pinned local adapter, while the shipped service remains disabled by default so
that an unmeasured engine cannot silently become an automated review source.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from customs_ai_agent.schemas.document_extraction import (
    DocumentExtractionResultV1,
    DocumentExtractionWarningV1,
    PreviewCandidateV1,
    PreviewInputV1,
    canonical_preview_text_sha256,
)
from customs_ai_agent.service.document_input import DocumentStructure
from customs_ai_agent.service.settings import ServiceSettings


class DocumentPreviewError(RuntimeError):
    """A content-free adapter error safe to map at the HTTP boundary."""


class DocumentPreviewUnavailable(DocumentPreviewError):
    """The configured profile has no approved local preview adapter."""


class DocumentPreviewInputError(DocumentPreviewError):
    """A local adapter rejected malformed or unsafe decoded content."""


class DocumentPreviewExecutionError(DocumentPreviewError):
    """A local adapter failed without exposing a provider error body."""


class LocalDocumentPreviewEngine(Protocol):
    """A pinned, local adapter only; network/model download is out of scope."""

    def extract(self, preview_input: PreviewInputV1) -> PreviewCandidateV1: ...


class PdfTextLayerAdapter(Protocol):
    """First-choice PDF adapter: native text layer before OCR/rasterization."""

    def extract_pdf_text_layer(
        self, preview_input: PreviewInputV1
    ) -> PreviewCandidateV1 | None: ...


class RasterOcrAdapter(Protocol):
    """Fallback local adapter for image documents or textless PDFs."""

    def extract_raster_or_image(self, preview_input: PreviewInputV1) -> PreviewCandidateV1: ...


@dataclass(frozen=True, slots=True)
class LayeredLocalPreviewEngine:
    """Use a PDF text layer before invoking any OCR-capable local adapter."""

    pdf_text_layer: PdfTextLayerAdapter
    raster_ocr: RasterOcrAdapter

    def extract(self, preview_input: PreviewInputV1) -> PreviewCandidateV1:
        if preview_input.metadata.media_type == "application/pdf":
            candidate = self.pdf_text_layer.extract_pdf_text_layer(preview_input)
            if candidate is not None:
                return candidate
        return self.raster_ocr.extract_raster_or_image(preview_input)


class DisabledDocumentPreviewEngine:
    """Safe default until a measured, locally pinned adapter is injected."""

    def extract(self, _preview_input: PreviewInputV1) -> PreviewCandidateV1:
        raise DocumentPreviewUnavailable("preview engine is not enabled")


def _validate_candidate_against_input(
    candidate: PreviewCandidateV1,
    *,
    preview_input: PreviewInputV1,
    structure: DocumentStructure,
    settings: ServiceSettings,
) -> None:
    pages = candidate.pages
    if not pages or len(pages) > settings.max_document_pages:
        raise DocumentPreviewInputError("preview page count is invalid")
    if preview_input.metadata.media_type != "application/pdf":
        if len(pages) != 1:
            raise DocumentPreviewInputError("image preview must contain one page")
        page = pages[0]
        if (
            page.coordinate_unit != "pixels"
            or page.width != structure.image_width
            or page.height != structure.image_height
        ):
            raise DocumentPreviewInputError("image preview geometry does not match admitted binary")
    elif structure.pdf_page_hint is not None and structure.pdf_page_hint > len(pages):
        # A nonzero uncompressed hint exceeding the adapter's output indicates
        # that the adapter did not safely traverse every admitted page.
        raise DocumentPreviewInputError("PDF preview omitted admitted page objects")


def extract_document_preview(
    engine: LocalDocumentPreviewEngine,
    *,
    preview_input: PreviewInputV1,
    structure: DocumentStructure,
    settings: ServiceSettings,
) -> DocumentExtractionResultV1:
    """Bind a local adapter candidate to immutable asset provenance.

    This function deliberately returns ``unavailable`` rather than selecting a
    fallback engine.  A caller cannot accidentally make preview/OCR results
    into review facts by calling this boundary.
    """

    try:
        candidate = engine.extract(preview_input)
    except DocumentPreviewUnavailable:
        return DocumentExtractionResultV1(
            document_asset_id=preview_input.metadata.document_asset_id,
            extraction_request_id=preview_input.metadata.extraction_request_id,
            status="unavailable",
            media_type=preview_input.metadata.media_type,
            byte_size=preview_input.byte_size,
            source_sha256=preview_input.metadata.source_sha256,
            warnings=[DocumentExtractionWarningV1(code="preview_engine_unavailable")],
        )
    _validate_candidate_against_input(
        candidate,
        preview_input=preview_input,
        structure=structure,
        settings=settings,
    )
    return DocumentExtractionResultV1(
        document_asset_id=preview_input.metadata.document_asset_id,
        extraction_request_id=preview_input.metadata.extraction_request_id,
        status=candidate.status,
        media_type=preview_input.metadata.media_type,
        byte_size=preview_input.byte_size,
        source_sha256=preview_input.metadata.source_sha256,
        canonical_text_sha256=canonical_preview_text_sha256(candidate.pages),
        extractor_revision=candidate.extractor_revision,
        pages=candidate.pages,
        warnings=candidate.warnings,
    )


__all__ = [
    "DisabledDocumentPreviewEngine",
    "DocumentPreviewError",
    "DocumentPreviewExecutionError",
    "DocumentPreviewInputError",
    "DocumentPreviewUnavailable",
    "LayeredLocalPreviewEngine",
    "LocalDocumentPreviewEngine",
    "PdfTextLayerAdapter",
    "RasterOcrAdapter",
    "extract_document_preview",
]
