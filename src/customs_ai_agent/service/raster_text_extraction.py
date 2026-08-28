"""Bounded text-only OCR execution boundary for scanned documents.

This module intentionally shares only strict decode/raster primitives with
the structured raster-v2 path.  It neither reads nor writes the v2 layout
approval pointer, and it never projects text into fields, tables, cells, or
geometry.  The output is an editable human-review draft.
"""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Literal, Protocol, Sequence

from customs_ai_agent.schemas.raster_text_extraction import (
    RasterTextExtractionResultV1,
    RasterTextInputV1,
    RasterTextPageV1,
    RasterTextPreviewCandidateV1,
    RasterTextWarningV1,
    canonical_raster_text,
    canonical_raster_text_sha256,
)
from customs_ai_agent.service.document_input import DocumentStructure
from customs_ai_agent.service.raster_document_extraction import RasterizedPageV2
from customs_ai_agent.service.settings import ServiceSettings


_MAX_RASTER_TEXT_CANONICAL_CHARACTERS = 200_000


class RasterTextPreviewError(RuntimeError):
    """Content-free text OCR error safe for the HTTP boundary."""


class RasterTextPreviewUnavailable(RasterTextPreviewError):
    """No passing local text profile is active."""


class RasterTextPreviewInputError(RasterTextPreviewError):
    """Strict decoding or input admission failed."""


class RasterTextPreviewUnsupported(RasterTextPreviewError):
    """A valid request is outside the measured text-only scope."""


class RasterTextPreviewExecutionError(RasterTextPreviewError):
    """Provider failed without exposing source text or native diagnostics."""


class RasterTextInputAdapter(Protocol):
    """Strict local decoder/rasterizer seam before a text-only provider."""

    def rasterize_pdf(
        self, raster_input: RasterTextInputV1, structure: DocumentStructure
    ) -> Sequence[RasterizedPageV2]: ...

    def decode_image(
        self, raster_input: RasterTextInputV1, structure: DocumentStructure
    ) -> RasterizedPageV2: ...


@dataclass(frozen=True, slots=True)
class RasterTextLedgerV1:
    """Provider-neutral, in-memory text result with no positional claims."""

    source_sha256: str
    raster_text_profile_revision: str
    provider_revision: str
    pages: tuple[RasterTextPageV1, ...]
    minimum_page_confidence: float | None = None
    warnings: tuple[RasterTextWarningV1, ...] = ()
    needs_review_reasons: tuple[RasterTextWarningV1, ...] = ()

    def __post_init__(self) -> None:
        page_numbers = [page.page_number for page in self.pages]
        if (
            len(self.source_sha256) != 64
            or not self.raster_text_profile_revision
            or not self.provider_revision
            or page_numbers != list(range(1, len(page_numbers) + 1))
            or (
                self.minimum_page_confidence is not None
                and not 0 <= self.minimum_page_confidence <= 1
            )
        ):
            raise ValueError("raster text ledger is invalid")


class RasterTextProvider(Protocol):
    """Pinned local OCR-only provider.  It does not receive original PDFs."""

    def extract_text(
        self,
        pages: Sequence[RasterizedPageV2],
        *,
        source_sha256: str,
        raster_text_profile_revision: str,
    ) -> RasterTextLedgerV1: ...


class LocalRasterTextPreviewEngine(Protocol):
    def extract(
        self, raster_input: RasterTextInputV1, structure: DocumentStructure
    ) -> RasterTextPreviewCandidateV1: ...


@dataclass(frozen=True, slots=True)
class MediaAwareRasterTextEngine:
    """Route PDFs and images without exposing original PDF bytes to OCR."""

    input_adapter: RasterTextInputAdapter
    provider: RasterTextProvider

    def prepare_pages(
        self, raster_input: RasterTextInputV1, structure: DocumentStructure
    ) -> tuple[RasterizedPageV2, ...]:
        media_type = raster_input.metadata.media_type
        if media_type == "application/pdf":
            # ``pdf_page_hint`` is produced during strict admission before a
            # native renderer is touched.  Text-v1 is deliberately measured
            # only on one-page files, so do not spend the 55-second lane
            # rasterizing an already-known unsupported multi-page document.
            if structure.pdf_page_hint != 1:
                raise RasterTextPreviewUnsupported("multi_page_pdf_unsupported")
            pages = tuple(self.input_adapter.rasterize_pdf(raster_input, structure))
        elif media_type in {"image/jpeg", "image/png"}:
            pages = (self.input_adapter.decode_image(raster_input, structure),)
        else:  # Defensive: the metadata schema has already constrained this.
            raise RasterTextPreviewInputError("raster text media type is unsupported")
        if not pages:
            raise RasterTextPreviewInputError("raster text input adapter returned no pages")
        # The source-attested text-v1 Gate covers exactly one-page documents.
        # Keep the schema forward-compatible, but never send an unmeasured
        # multi-page PDF to the direct OCR provider.
        if len(pages) != 1:
            raise RasterTextPreviewUnsupported("multi_page_pdf_unsupported")
        return pages

    def extract(
        self, raster_input: RasterTextInputV1, structure: DocumentStructure
    ) -> RasterTextPreviewCandidateV1:
        pages = self.prepare_pages(raster_input, structure)
        ledger = self.provider.extract_text(
            pages,
            source_sha256=raster_input.metadata.source_sha256,
            raster_text_profile_revision=raster_input.metadata.raster_text_profile_revision,
        )
        if (
            ledger.source_sha256 != raster_input.metadata.source_sha256
            or ledger.raster_text_profile_revision
            != raster_input.metadata.raster_text_profile_revision
            # A syntactically valid direct-OCR response may contain no text.
            # It is not a sidecar outage: preserve its provenance long enough
            # to map it to a content-free ``needs_review`` result below.
            # Any non-empty response still has to bind every input page.
            or (ledger.pages and len(ledger.pages) != len(pages))
        ):
            raise RasterTextPreviewExecutionError("raster text provider binding is invalid")
        if ledger.pages and any(
            page.page_number != source_page.page_number
            for page, source_page in zip(ledger.pages, pages, strict=True)
        ):
            raise RasterTextPreviewExecutionError("raster text provider page binding is invalid")
        reasons = list(ledger.needs_review_reasons)
        if not ledger.pages:
            reasons.append(RasterTextWarningV1(code="ocr_text_not_found"))
        elif ledger.minimum_page_confidence is not None and any(
            page.confidence is None or page.confidence < ledger.minimum_page_confidence
            for page in ledger.pages
        ):
            reasons.append(RasterTextWarningV1(code="ocr_confidence_below_calibrated_threshold"))
        status: Literal["preview_ready", "needs_review"] = (
            "preview_ready" if ledger.pages and not reasons else "needs_review"
        )
        return RasterTextPreviewCandidateV1(
            status=status,
            extractor_revision=ledger.provider_revision if ledger.pages else None,
            pages=list(ledger.pages),
            warnings=list(ledger.warnings),
            needs_review_reasons=list({item.code: item for item in reasons}.values()),
        )


class DisabledRasterTextPreviewEngine:
    """Safe default until a separately passing text profile is activated."""

    def extract(
        self, _raster_input: RasterTextInputV1, _structure: DocumentStructure
    ) -> RasterTextPreviewCandidateV1:
        raise RasterTextPreviewUnavailable("raster text preview engine is not enabled")


def raster_text_candidate_to_result(
    *,
    raster_input: RasterTextInputV1,
    candidate: RasterTextPreviewCandidateV1,
) -> RasterTextExtractionResultV1:
    pages = candidate.pages or []
    # A provider may return several individually valid pages whose joined
    # value would exceed the public v1 result contract.  Do not truncate a
    # customs document silently: return a content-free review state instead.
    if pages and len(canonical_raster_text(pages)) > _MAX_RASTER_TEXT_CANONICAL_CHARACTERS:
        return RasterTextExtractionResultV1(
            document_asset_id=raster_input.metadata.document_asset_id,
            extraction_request_id=raster_input.metadata.extraction_request_id,
            raster_text_profile_revision=raster_input.metadata.raster_text_profile_revision,
            status="needs_review",
            media_type=raster_input.metadata.media_type,
            byte_size=raster_input.byte_size,
            source_sha256=raster_input.metadata.source_sha256,
            canonical_text=None,
            canonical_text_sha256=None,
            extractor_revision=None,
            pages=[],
            warnings=[],
            needs_review_reasons=[RasterTextWarningV1(code="ocr_text_too_large")],
        )
    return RasterTextExtractionResultV1(
        document_asset_id=raster_input.metadata.document_asset_id,
        extraction_request_id=raster_input.metadata.extraction_request_id,
        raster_text_profile_revision=raster_input.metadata.raster_text_profile_revision,
        status=candidate.status,
        media_type=raster_input.metadata.media_type,
        byte_size=raster_input.byte_size,
        source_sha256=raster_input.metadata.source_sha256,
        canonical_text=canonical_raster_text(pages) if pages else None,
        canonical_text_sha256=canonical_raster_text_sha256(pages) if pages else None,
        extractor_revision=candidate.extractor_revision if pages else None,
        pages=pages,
        warnings=candidate.warnings or [],
        needs_review_reasons=candidate.needs_review_reasons or [],
    )


def _orientation_needs_review(
    raster_input: RasterTextInputV1,
    structure: DocumentStructure,
) -> RasterTextExtractionResultV1 | None:
    # The text draft does not expose geometry, but accepting rotated JPEGs
    # would silently alter reading order relative to the stored source.  Keep
    # the same v1 scope as raster-v2 until orientation handling is measured.
    if raster_input.metadata.media_type == "image/jpeg" and structure.jpeg_exif_orientation not in {
        None,
        1,
    }:
        warning = RasterTextWarningV1(code="jpeg_exif_orientation_unsupported")
        return RasterTextExtractionResultV1(
            document_asset_id=raster_input.metadata.document_asset_id,
            extraction_request_id=raster_input.metadata.extraction_request_id,
            raster_text_profile_revision=raster_input.metadata.raster_text_profile_revision,
            status="needs_review",
            media_type=raster_input.metadata.media_type,
            byte_size=raster_input.byte_size,
            source_sha256=raster_input.metadata.source_sha256,
            canonical_text=None,
            canonical_text_sha256=None,
            extractor_revision=None,
            pages=[],
            warnings=[],
            needs_review_reasons=[warning],
        )
    return None


def _unsupported_needs_review(
    raster_input: RasterTextInputV1, *, code: str
) -> RasterTextExtractionResultV1:
    warning = RasterTextWarningV1(
        code=code
        if re.fullmatch(r"[a-z][a-z0-9_]{1,127}", code) is not None
        else "raster_text_unsupported"
    )
    return RasterTextExtractionResultV1(
        document_asset_id=raster_input.metadata.document_asset_id,
        extraction_request_id=raster_input.metadata.extraction_request_id,
        raster_text_profile_revision=raster_input.metadata.raster_text_profile_revision,
        status="needs_review",
        media_type=raster_input.metadata.media_type,
        byte_size=raster_input.byte_size,
        source_sha256=raster_input.metadata.source_sha256,
        canonical_text=None,
        canonical_text_sha256=None,
        extractor_revision=None,
        pages=[],
        warnings=[],
        needs_review_reasons=[warning],
    )


def extract_raster_text_preview(
    engine: LocalRasterTextPreviewEngine,
    *,
    raster_input: RasterTextInputV1,
    structure: DocumentStructure,
    settings: ServiceSettings,
) -> RasterTextExtractionResultV1:
    """Execute only a separately approved text profile, fail-closed."""

    if raster_input.byte_size > settings.max_document_bytes:
        raise RasterTextPreviewInputError("raster text byte size is invalid")
    orientation_result = _orientation_needs_review(raster_input, structure)
    if orientation_result is not None:
        return orientation_result
    try:
        candidate = engine.extract(raster_input, structure)
    except RasterTextPreviewUnsupported as error:
        return _unsupported_needs_review(raster_input, code=str(error))
    except RasterTextPreviewUnavailable:
        return RasterTextExtractionResultV1(
            document_asset_id=raster_input.metadata.document_asset_id,
            extraction_request_id=raster_input.metadata.extraction_request_id,
            raster_text_profile_revision=raster_input.metadata.raster_text_profile_revision,
            status="unavailable",
            media_type=raster_input.metadata.media_type,
            byte_size=raster_input.byte_size,
            source_sha256=raster_input.metadata.source_sha256,
            canonical_text=None,
            canonical_text_sha256=None,
            extractor_revision=None,
            pages=[],
            warnings=[RasterTextWarningV1(code="raster_text_engine_unavailable")],
            needs_review_reasons=[],
        )
    return raster_text_candidate_to_result(raster_input=raster_input, candidate=candidate)


__all__ = [
    "DisabledRasterTextPreviewEngine",
    "LocalRasterTextPreviewEngine",
    "MediaAwareRasterTextEngine",
    "RasterTextInputAdapter",
    "RasterTextLedgerV1",
    "RasterTextPreviewError",
    "RasterTextPreviewExecutionError",
    "RasterTextPreviewInputError",
    "RasterTextPreviewUnavailable",
    "RasterTextPreviewUnsupported",
    "RasterTextProvider",
    "extract_raster_text_preview",
    "raster_text_candidate_to_result",
]
