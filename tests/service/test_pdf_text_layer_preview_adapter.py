from __future__ import annotations

import socket
import sys
from types import ModuleType, SimpleNamespace
from typing import Callable

import pytest

import customs_ai_agent.service.document_preview_runtime as runtime_module
from customs_ai_agent.evaluation.synthetic_pdf_candidate_runner import _PdfTextLayerCandidateAdapter
from customs_ai_agent.schemas.document_extraction import (
    DocumentExtractionRequestMetadataV1,
    PreviewInputV1,
)
from customs_ai_agent.service.document_extraction import (
    DocumentPreviewInputError,
    DocumentPreviewUnavailable,
)
from customs_ai_agent.service.document_preview_runtime import PdfTextLayerPreviewAdapter


_IDENTITY = [1, 0, 0, 1, 0, 0]
_LEGACY_REVISION = "pypdf-5.3.1-text-layer-v1"
_VISITOR_REVISION = "pypdf-5.3.1-text-layer-visitor-v2"


def _preview_input() -> PreviewInputV1:
    content = b"%PDF-test-visitor"
    return PreviewInputV1(
        metadata=DocumentExtractionRequestMetadataV1(
            document_asset_id="asset-visitor",
            extraction_request_id="request-visitor",
            source_sha256="a" * 64,
            media_type="application/pdf",
        ),
        byte_size=len(content),
        content=content,
    )


class _FakeVisitorPage:
    def __init__(
        self,
        callback_plan: Callable[[Callable[..., None]], None],
        *,
        aggregate_text: str = "default-aggregation-must-not-be-used",
    ) -> None:
        self.mediabox = SimpleNamespace(width=200, height=100)
        self._callback_plan = callback_plan
        self._aggregate_text = aggregate_text
        self.visitor_calls = 0
        self.aggregate_calls = 0

    def extract_text(self, *, visitor_text: Callable[..., None] | None = None) -> str:
        if visitor_text is None:
            self.aggregate_calls += 1
            return self._aggregate_text
        self.visitor_calls += 1
        self._callback_plan(visitor_text)
        return self._aggregate_text


def _install_fake_pypdf(
    monkeypatch: pytest.MonkeyPatch,
    page_factory: Callable[[], _FakeVisitorPage],
) -> list[_FakeVisitorPage]:
    pages: list[_FakeVisitorPage] = []
    module = ModuleType("pypdf")

    def reader_factory(_stream: object, *, strict: bool) -> SimpleNamespace:
        assert strict is True
        page = page_factory()
        pages.append(page)
        return SimpleNamespace(is_encrypted=False, pages=[page])

    module.PdfReader = reader_factory  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pypdf", module)
    return pages


def test_pdf_text_visitor_trims_only_terminal_newlines_and_uses_no_aggregate_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def callback_plan(visitor: Callable[..., None]) -> None:
        non_identity = [1, 0, 0, 1, 1, 0]
        visitor("  first fragment\n", non_identity, _IDENTITY, {}, 12)
        visitor("\r\n", non_identity, _IDENTITY, {}, 12)
        visitor("second fragment\r\n", non_identity, _IDENTITY, {}, 12)

    pages = _install_fake_pypdf(monkeypatch, lambda: _FakeVisitorPage(callback_plan))

    adapter = PdfTextLayerPreviewAdapter(extractor_revision=_VISITOR_REVISION)
    fragments = adapter.extract_pdf_text_layer_fragments(_preview_input())
    candidate = adapter.extract_pdf_text_layer(_preview_input())

    assert candidate is not None
    spans = candidate.pages[0].spans
    assert [fragment.text for fragment in fragments] == ["  first fragment", "second fragment"]
    assert [span.text for span in spans] == ["  first fragment", "second fragment"]
    assert [(span.ordinal, span.reading_order) for span in spans] == [(1, 1), (2, 2)]
    assert all(span.bbox is None for span in spans)
    assert [fragment.anchor for fragment in fragments] == [
        None,
        None,
    ]
    assert [page.visitor_calls for page in pages] == [1, 1]
    assert [page.aggregate_calls for page in pages] == [0, 0]


def test_pdf_text_visitor_positions_are_deterministic_without_bbox_estimation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def callback_plan(visitor: Callable[..., None]) -> None:
        visitor("A\n", _IDENTITY, [1, 0, 0, 1, 50, 20], {}, 10)
        visitor("B\n", [1, 0, 0, 1, 10, 0], [1, 0, 0, 1, 70, 20], {}, 10)

    _install_fake_pypdf(monkeypatch, lambda: _FakeVisitorPage(callback_plan))
    adapter = PdfTextLayerPreviewAdapter(extractor_revision=_VISITOR_REVISION)

    first = adapter.extract_pdf_text_layer_fragments(_preview_input())
    second = adapter.extract_pdf_text_layer_fragments(_preview_input())

    assert first == second
    assert [(fragment.page_number, fragment.ordinal) for fragment in first] == [
        (1, 1),
        (1, 2),
    ]
    assert [
        None if fragment.anchor is None else (fragment.anchor.x, fragment.anchor.y)
        for fragment in first
    ] == [
        (0.25, 0.8),
        None,
    ]


def test_pdf_text_legacy_v1_preserves_the_aggregate_extract_text_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def callback_plan(_visitor: Callable[..., None]) -> None:
        raise AssertionError("legacy v1 must not invoke the visitor callback")

    pages = _install_fake_pypdf(
        monkeypatch,
        lambda: _FakeVisitorPage(
            callback_plan,
            aggregate_text="  legacy first\nlegacy second  ",
        ),
    )
    adapter = PdfTextLayerPreviewAdapter(extractor_revision=_LEGACY_REVISION)

    candidate = adapter.extract_pdf_text_layer(_preview_input())

    assert candidate is not None
    assert candidate.extractor_revision == _LEGACY_REVISION
    assert [span.text for span in candidate.pages[0].spans] == ["legacy first\nlegacy second"]
    assert [page.aggregate_calls for page in pages] == [1]
    assert [page.visitor_calls for page in pages] == [0]
    with pytest.raises(DocumentPreviewUnavailable, match="visitor is unavailable"):
        adapter.extract_pdf_text_layer_fragments(_preview_input())


def test_pdf_text_rejects_an_unsupported_extractor_revision() -> None:
    adapter = PdfTextLayerPreviewAdapter(
        extractor_revision="pypdf-5.3.1-text-layer-unreviewed-v3"
    )

    with pytest.raises(DocumentPreviewUnavailable, match="extractor revision is unavailable"):
        adapter.extract_pdf_text_layer(_preview_input())
    with pytest.raises(DocumentPreviewUnavailable, match="visitor is unavailable"):
        adapter.extract_pdf_text_layer_fragments(_preview_input())


def test_pdf_text_candidate_adapter_reuses_the_internal_fragment_projection() -> None:
    class FragmentOnlyAdapter:
        def extract_pdf_text_layer_fragments(
            self, _preview_input: PreviewInputV1
        ) -> tuple[runtime_module._PdfTextLayerVisitorFragment, ...]:
            return (
                runtime_module._PdfTextLayerVisitorFragment(
                    page_number=1,
                    ordinal=2,
                    text="candidate-fragment",
                    anchor=runtime_module._PdfTextLayerVisitorAnchor(x=0.25, y=0.8),
                ),
            )

        def extract_pdf_text_layer(self, _preview_input: PreviewInputV1) -> None:
            raise AssertionError("candidate path must not use the public preview projection")

    candidate = _PdfTextLayerCandidateAdapter(FragmentOnlyAdapter()).extract_pdf(_preview_input())  # type: ignore[arg-type]

    assert [
        (span.page_number, span.reading_order, span.text, span.bbox) for span in candidate.spans
    ] == [(1, 2, "candidate-fragment", None)]


def test_pdf_text_visitor_failure_does_not_leak_fragment_content(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    secret = "visitor-fragment-must-not-appear-in-errors"

    def callback_plan(visitor: Callable[..., None]) -> None:
        visitor(secret + "\n", _IDENTITY, _IDENTITY, {}, 12)
        raise RuntimeError(secret)

    _install_fake_pypdf(monkeypatch, lambda: _FakeVisitorPage(callback_plan))

    with pytest.raises(
        DocumentPreviewInputError, match="PDF text layer cannot be safely decoded"
    ) as error:
        PdfTextLayerPreviewAdapter(extractor_revision=_VISITOR_REVISION).extract_pdf_text_layer(
            _preview_input()
        )

    captured = capsys.readouterr()
    assert secret not in str(error.value)
    assert secret not in captured.out
    assert secret not in captured.err
    assert error.value.__cause__ is None
    assert error.value.__suppress_context__ is True


def test_pdf_text_visitor_runs_under_the_local_network_guard(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_connect_calls: list[object] = []

    def original_connect(_socket: socket.socket, address: object) -> None:
        original_connect_calls.append(address)

    monkeypatch.setattr(runtime_module, "_ORIGINAL_SOCKET_CONNECT", original_connect)

    def callback_plan(_visitor: Callable[..., None]) -> None:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as connection:
            connection.connect(("127.0.0.1", 9))

    _install_fake_pypdf(monkeypatch, lambda: _FakeVisitorPage(callback_plan))

    with pytest.raises(DocumentPreviewInputError, match="PDF text layer cannot be safely decoded"):
        PdfTextLayerPreviewAdapter(extractor_revision=_VISITOR_REVISION).extract_pdf_text_layer(
            _preview_input()
        )

    assert original_connect_calls == []
