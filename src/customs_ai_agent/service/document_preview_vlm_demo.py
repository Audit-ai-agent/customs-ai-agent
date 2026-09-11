"""Demonstration-only preview engine: pypdf text layer plus LLM vision OCR.

This module is selected exclusively through the explicit
``CUSTOMS_AGENT_DOCUMENT_PREVIEW_DEMO`` entry-point opt-in and never through the
approved-engine resolution path.  A PDF with a native text layer is extracted
locally with the preserved legacy ``pypdf`` semantics; only an image upload or a
textless (scanned) PDF is sent to the configured vision model (Anthropic Claude
or OpenAI), reusing the same provider configuration as the functional-UAT text
adapter.  Every VLM result is a
human-review candidate (``needs_review``); it can never claim ``preview_ready``.
"""

from __future__ import annotations

import base64
import io
import json
import threading
from typing import Any

from customs_ai_agent.adapters.llm import (
    ClaudeSettings,
    LLMConfigurationError,
    _json_text_from_model_output,
    build_provider_client,
)
from customs_ai_agent.schemas.document_extraction import (
    DocumentExtractionWarningV1,
    DocumentPreviewPageV1,
    DocumentSpanV1,
    PreviewCandidateV1,
    PreviewInputV1,
)
from customs_ai_agent.service.document_extraction import (
    DocumentPreviewExecutionError,
    DocumentPreviewInputError,
    DocumentPreviewUnavailable,
    LayeredLocalPreviewEngine,
)
from customs_ai_agent.service.document_input import inspect_document_structure
from customs_ai_agent.service.document_preview_runtime import (
    PdfTextLayerPreviewAdapter,
    _chunks,
    _pdf_page_dimensions,
    local_files_only,
)
from customs_ai_agent.service.errors import ServiceHttpError
from customs_ai_agent.service.llm_settings import LLM_SETTINGS
from customs_ai_agent.service.settings import ServiceSettings


_PDF_TEXT_LAYER_LEGACY_REVISION = "pypdf-5.3.1-text-layer-v1"
_VLM_OCR_REVISION_PREFIX = "vlm-ocr-demo-v1"
# Base64 expansion of the request body must stay under the providers' request
# size limits, and both providers read at most 100 PDF pages per request.
_MAX_VLM_PDF_BYTES = 24 * 1024 * 1024
_MAX_VLM_PDF_PAGES = 100

_OCR_SYSTEM_PROMPT = (
    "You are an OCR engine for customs trade documents. Transcribe all visible text "
    "from the supplied document exactly as written, in natural reading order, keeping "
    "line breaks between distinct lines. Do not translate, summarize, correct, or "
    "invent text. Return only one JSON object of the form "
    '{"pages": [{"page_number": 1, "text": "..."}]} with exactly one entry per page, '
    "page_number starting at 1, and an empty string for a page without text."
)


def _pdf_page_geometry(preview_input: PreviewInputV1) -> list[tuple[int, int]]:
    """Read page count and mediabox sizes locally so the wire pages stay honest."""

    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]
    except ImportError as error:
        raise DocumentPreviewUnavailable("pypdf is not provisioned") from error
    try:
        with local_files_only():
            reader = PdfReader(io.BytesIO(preview_input.content), strict=True)
            if reader.is_encrypted:
                raise DocumentPreviewInputError("encrypted PDF preview is unavailable")
            geometry = [_pdf_page_dimensions(page) for page in reader.pages]
    except DocumentPreviewInputError:
        raise
    except Exception:
        # A parser exception can include source text; keep the boundary content-free.
        raise DocumentPreviewInputError("PDF page geometry cannot be safely decoded") from None
    if not geometry:
        raise DocumentPreviewInputError("PDF has no pages")
    return geometry


class ClaudeVlmOcrAdapter:
    """LLM vision OCR for image uploads and PDFs without a native text layer."""

    def __init__(
        self,
        service_settings: ServiceSettings,
        *,
        claude_settings: ClaudeSettings | None = None,
        client: Any | None = None,
    ) -> None:
        self._service_settings = service_settings
        self._claude_settings = claude_settings
        self._client = client
        # An injected client is pinned; otherwise follow the operator-adjustable store.
        self._pinned = client is not None
        self._settings_version = LLM_SETTINGS.version
        self._lock = threading.Lock()

    def _ensure_client(self) -> tuple[ClaudeSettings, Any]:
        with self._lock:
            if not self._pinned and self._settings_version != LLM_SETTINGS.version:
                self._claude_settings = None
                self._client = None
            if self._claude_settings is None or self._client is None:
                self._settings_version = LLM_SETTINGS.version
                try:
                    settings = LLM_SETTINGS.current()
                except LLMConfigurationError:
                    raise DocumentPreviewUnavailable("VLM OCR is not configured") from None
                try:
                    client = build_provider_client(settings, max_retries=1)
                except LLMConfigurationError as error:
                    if "Install the functional-uat extra" in str(error):
                        raise DocumentPreviewUnavailable(
                            "the provider SDK package is not provisioned"
                        ) from None
                    raise DocumentPreviewExecutionError(
                        "the local VLM client could not be initialized"
                    ) from None
                self._claude_settings = settings
                self._client = client
            return self._claude_settings, self._client

    def extract_raster_or_image(self, preview_input: PreviewInputV1) -> PreviewCandidateV1:
        media_type = preview_input.metadata.media_type
        if media_type == "application/pdf":
            geometry = _pdf_page_geometry(preview_input)
            if len(geometry) > _MAX_VLM_PDF_PAGES:
                raise DocumentPreviewUnavailable("PDF exceeds the VLM OCR page limit")
            if preview_input.byte_size > _MAX_VLM_PDF_BYTES:
                raise DocumentPreviewUnavailable("PDF exceeds the VLM OCR size limit")
            coordinate_unit = "pdf_points"
            source_block: dict[str, Any] = {
                "type": "document",
                "source": {
                    "type": "base64",
                    "media_type": "application/pdf",
                    "data": base64.standard_b64encode(preview_input.content).decode("ascii"),
                },
            }
        else:
            try:
                structure = inspect_document_structure(
                    preview_input.content, media_type, self._service_settings
                )
            except ServiceHttpError:
                raise DocumentPreviewInputError("image preview cannot be safely decoded") from None
            if structure.image_width is None or structure.image_height is None:
                raise DocumentPreviewInputError("image preview cannot be safely decoded")
            geometry = [(structure.image_width, structure.image_height)]
            coordinate_unit = "pixels"
            source_block = {
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": media_type,
                    "data": base64.standard_b64encode(preview_input.content).decode("ascii"),
                },
            }
        model_id, page_texts = self._request_page_texts(source_block, expected_pages=len(geometry))
        pages = [
            DocumentPreviewPageV1(
                page_number=index,
                width=width,
                height=height,
                coordinate_unit=coordinate_unit,  # type: ignore[arg-type]
                spans=[
                    DocumentSpanV1(
                        span_id=f"page-{index}-span-{ordinal}",
                        ordinal=ordinal,
                        reading_order=ordinal,
                        text=chunk,
                    )
                    for ordinal, chunk in enumerate(_chunks(text), start=1)
                ],
            )
            for index, ((width, height), text) in enumerate(zip(geometry, page_texts), start=1)
        ]
        return PreviewCandidateV1(
            status="needs_review",
            extractor_revision=f"{_VLM_OCR_REVISION_PREFIX}:{model_id}",
            pages=pages,
            warnings=[DocumentExtractionWarningV1(code="vlm_ocr_used")],
        )

    def _request_page_texts(
        self, source_block: dict[str, Any], *, expected_pages: int
    ) -> tuple[str, list[str]]:
        claude_settings, client = self._ensure_client()
        instruction = (
            f"This document has {expected_pages} page(s). "
            "Transcribe every page and return only the JSON object."
        )
        try:
            response = client.messages.create(
                model=claude_settings.model,
                max_tokens=claude_settings.max_tokens,
                temperature=0,
                system=_OCR_SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": [source_block, {"type": "text", "text": instruction}],
                    }
                ],
            )
        except Exception:
            raise DocumentPreviewExecutionError("Claude VLM OCR request failed") from None
        content = getattr(response, "content", None)
        if not isinstance(content, list):
            raise DocumentPreviewExecutionError("Claude VLM OCR returned no text result")
        output = "".join(
            part
            for part in (getattr(item, "text", None) for item in content)
            if isinstance(part, str)
        ).strip()
        try:
            parsed = json.loads(_json_text_from_model_output(output))
        except (TypeError, ValueError):
            raise DocumentPreviewExecutionError("Claude VLM OCR returned invalid JSON") from None
        pages_value = parsed.get("pages") if isinstance(parsed, dict) else None
        if not isinstance(pages_value, list):
            raise DocumentPreviewExecutionError("Claude VLM OCR returned an invalid page list")
        texts = [""] * expected_pages
        for entry in pages_value:
            if not isinstance(entry, dict):
                continue
            number = entry.get("page_number")
            text = entry.get("text")
            if (
                isinstance(number, int)
                and not isinstance(number, bool)
                and 1 <= number <= expected_pages
                and isinstance(text, str)
            ):
                texts[number - 1] = text
        return claude_settings.model, texts


def build_claude_vlm_demo_engine(service_settings: ServiceSettings) -> LayeredLocalPreviewEngine:
    """Compose the demo engine: local pypdf text layer first, LLM VLM OCR fallback."""

    return LayeredLocalPreviewEngine(
        pdf_text_layer=PdfTextLayerPreviewAdapter(
            extractor_revision=_PDF_TEXT_LAYER_LEGACY_REVISION
        ),
        raster_ocr=ClaudeVlmOcrAdapter(service_settings),
    )


__all__ = [
    "ClaudeVlmOcrAdapter",
    "build_claude_vlm_demo_engine",
]
