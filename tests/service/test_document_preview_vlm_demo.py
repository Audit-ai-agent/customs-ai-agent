from __future__ import annotations

import base64
import json
import sys
from types import ModuleType, SimpleNamespace
from typing import Any

from pydantic import SecretStr
import pytest

from customs_ai_agent.adapters.llm import ClaudeSettings
from customs_ai_agent.schemas.document_extraction import (
    DocumentExtractionRequestMetadataV1,
    PreviewInputV1,
)
from customs_ai_agent.service.document_extraction import (
    DocumentPreviewExecutionError,
    DocumentPreviewUnavailable,
)
from customs_ai_agent.service.document_preview_vlm_demo import ClaudeVlmOcrAdapter
from customs_ai_agent.service.settings import ServiceSettings


_TOKEN = "Abcdefghijklmnopqrstuvwxyz0123456789ABCDEFGHIJK"


def _settings() -> ServiceSettings:
    return ServiceSettings(
        bearer_token=SecretStr(_TOKEN),
        evidence_base_url="http://127.0.0.1:18083",
        evidence_service_token=SecretStr(_TOKEN),
    )


def _claude_settings() -> ClaudeSettings:
    return ClaudeSettings(api_key=SecretStr("k" * 24))


def _png(width: int = 2, height: int = 3) -> bytes:
    return (
        b"\x89PNG\r\n\x1a\n"
        + b"\x00\x00\x00\rIHDR"
        + width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
    )


def _preview_input(content: bytes, media_type: str) -> PreviewInputV1:
    return PreviewInputV1(
        metadata=DocumentExtractionRequestMetadataV1(
            document_asset_id="asset-vlm",
            extraction_request_id="request-vlm",
            source_sha256="a" * 64,
            media_type=media_type,  # type: ignore[arg-type]
        ),
        byte_size=len(content),
        content=content,
    )


class _FakeClaudeClient:
    def __init__(self, output_text: str) -> None:
        self.requests: list[dict[str, Any]] = []
        self._output_text = output_text
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kwargs: Any) -> SimpleNamespace:
        self.requests.append(kwargs)
        return SimpleNamespace(content=[SimpleNamespace(text=self._output_text)])


def _install_fake_pypdf(monkeypatch: pytest.MonkeyPatch, page_count: int) -> None:
    module = ModuleType("pypdf")

    def reader_factory(_stream: object, *, strict: bool) -> SimpleNamespace:
        assert strict is True
        pages = [
            SimpleNamespace(mediabox=SimpleNamespace(width=200, height=100))
            for _ in range(page_count)
        ]
        return SimpleNamespace(is_encrypted=False, pages=pages)

    module.PdfReader = reader_factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pypdf", module)


def test_image_ocr_builds_needs_review_candidate_with_image_geometry() -> None:
    content = _png()
    client = _FakeClaudeClient(
        json.dumps({"pages": [{"page_number": 1, "text": "INVOICE NO INV-001"}]})
    )
    adapter = ClaudeVlmOcrAdapter(_settings(), claude_settings=_claude_settings(), client=client)

    candidate = adapter.extract_raster_or_image(_preview_input(content, "image/png"))

    assert candidate.status == "needs_review"
    assert candidate.extractor_revision.startswith("vlm-ocr-demo-v1:")
    assert [warning.code for warning in candidate.warnings] == ["vlm_ocr_used"]
    assert len(candidate.pages) == 1
    page = candidate.pages[0]
    assert (page.width, page.height, page.coordinate_unit) == (2, 3, "pixels")
    assert [span.text for span in page.spans] == ["INVOICE NO INV-001"]
    request = client.requests[0]
    source_block = request["messages"][0]["content"][0]
    assert source_block["type"] == "image"
    assert source_block["source"]["media_type"] == "image/png"
    assert source_block["source"]["data"] == base64.standard_b64encode(content).decode("ascii")


def test_textless_pdf_is_sent_as_document_block_with_local_page_geometry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_pypdf(monkeypatch, page_count=2)
    client = _FakeClaudeClient(
        json.dumps(
            {
                "pages": [
                    {"page_number": 1, "text": "page one text"},
                    {"page_number": 2, "text": ""},
                ]
            }
        )
    )
    adapter = ClaudeVlmOcrAdapter(_settings(), claude_settings=_claude_settings(), client=client)

    candidate = adapter.extract_raster_or_image(_preview_input(b"%PDF-scanned", "application/pdf"))

    assert candidate.status == "needs_review"
    assert len(candidate.pages) == 2
    assert all(page.coordinate_unit == "pdf_points" for page in candidate.pages)
    assert (candidate.pages[0].width, candidate.pages[0].height) == (200, 100)
    assert [span.text for span in candidate.pages[0].spans] == ["page one text"]
    assert candidate.pages[1].spans == []
    source_block = client.requests[0]["messages"][0]["content"][0]
    assert source_block["type"] == "document"
    assert source_block["source"]["media_type"] == "application/pdf"


def test_missing_api_key_maps_to_preview_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("ANTHROPIC_API_KEY", "ANTHROPIC_BASE_URL", "ANTHROPIC_AUTH_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    adapter = ClaudeVlmOcrAdapter(_settings())

    with pytest.raises(DocumentPreviewUnavailable):
        adapter.extract_raster_or_image(_preview_input(_png(), "image/png"))


def test_invalid_provider_output_maps_to_execution_error() -> None:
    client = _FakeClaudeClient("this is not a JSON object")
    adapter = ClaudeVlmOcrAdapter(_settings(), claude_settings=_claude_settings(), client=client)

    with pytest.raises(DocumentPreviewExecutionError):
        adapter.extract_raster_or_image(_preview_input(_png(), "image/png"))
