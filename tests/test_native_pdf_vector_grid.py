from __future__ import annotations

import ast
import inspect
import sys
from types import ModuleType, SimpleNamespace
from typing import Sequence

import pytest

import customs_ai_agent.evaluation.native_pdf_vector_grid as vector_grid
from customs_ai_agent.schemas.document_extraction import (
    DocumentExtractionRequestMetadataV1,
    PreviewInputV1,
)


def _preview_input() -> PreviewInputV1:
    content = b"%PDF-test-native-vector-grid"
    return PreviewInputV1(
        metadata=DocumentExtractionRequestMetadataV1(
            document_asset_id="asset-vector-grid",
            extraction_request_id="request-vector-grid",
            source_sha256="a" * 64,
            media_type="application/pdf",
        ),
        byte_size=len(content),
        content=content,
    )


class _FakePage:
    def __init__(
        self, operations: Sequence[tuple[object, object]], *, width: int, height: int
    ) -> None:
        self.mediabox = SimpleNamespace(width=width, height=height)
        self._content = SimpleNamespace(operations=operations)

    def get_contents(self) -> object:
        return self._content


class _FakeByteStringObject(bytes):
    def __str__(self) -> str:
        return bytes(self).decode("ascii")


def _install_fake_pypdf(
    monkeypatch: pytest.MonkeyPatch,
    pages: Sequence[_FakePage],
) -> None:
    pypdf = ModuleType("pypdf")
    generic = ModuleType("pypdf.generic")

    def reader_factory(_stream: object, *, strict: bool) -> SimpleNamespace:
        assert strict is True
        return SimpleNamespace(is_encrypted=False, pages=list(pages))

    class ContentStream:
        def __init__(self, content: object, _reader: object) -> None:
            self.operations = getattr(content, "operations")

    pypdf.PdfReader = reader_factory  # type: ignore[attr-defined]
    generic.ByteStringObject = _FakeByteStringObject  # type: ignore[attr-defined]
    generic.ContentStream = ContentStream  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pypdf", pypdf)
    monkeypatch.setitem(sys.modules, "pypdf.generic", generic)


def _operation(name: str, operands: Sequence[object] = ()) -> tuple[object, object]:
    return list(operands), name.encode("ascii")


def _grid_operations(
    *,
    columns: int,
    x_offset: float = 10.0,
    y_offset: float = 0.0,
    scale: float = 1.0,
    prefix: str = "cell",
    include_all_text: bool = True,
    include_second_grid: bool = False,
) -> list[tuple[object, object]]:
    """Build a generic visible rectangle grid; values are not fixture data."""

    operations: list[tuple[object, object]] = [
        _operation("cm", (1, 0, 0, 1, 0, 0)),
        _operation("w", (1,)),
    ]

    def append_grid(*, left: float, top: float) -> None:
        width = 10.0 * scale
        pitch = 12.0 * scale
        header_height = 10.0 * scale
        body_height = 30.0 * scale
        body_y = top - body_height
        for y, height in ((top, header_height), (body_y, body_height)):
            for column in range(columns):
                operations.append(_operation("re", (left + (column * pitch), y, width, height)))
                operations.append(_operation("S"))
                operations.append(_operation("n"))
        operations.append(_operation("BT"))
        operations.append(_operation("Tf", ("F1", 10)))
        operations.append(_operation("TL", (5,)))
        for row, y in enumerate((top + (header_height / 2), body_y + (body_height / 2))):
            for column in range(columns):
                if not include_all_text and row == 1 and column == columns - 1:
                    continue
                x = left + (column * pitch) + 1
                operations.append(_operation("Tm", (1, 0, 0, 1, x, y)))
                operations.append(_operation("Tj", (f"{prefix}-{row}-{column}",)))
        operations.append(_operation("ET"))

    append_grid(left=x_offset, top=y_offset + (70.0 * scale))
    if include_second_grid:
        append_grid(left=x_offset + (120.0 * scale), top=y_offset + (70.0 * scale))
    return operations


def test_extracts_one_generic_five_column_grid_with_text_anchors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations = _grid_operations(columns=5)
    _install_fake_pypdf(monkeypatch, [_FakePage(operations, width=200, height=120)])

    extraction = vector_grid.extract_native_pdf_vector_grid(preview_input=_preview_input())

    assert [(page.page_number, page.width, page.height) for page in extraction.pages] == [
        (1, 200, 120)
    ]
    assert len(extraction.spans) == 10
    assert [(span.page_number, span.reading_order) for span in extraction.spans] == [
        (1, index) for index in range(1, 11)
    ]
    assert extraction.spans[0].anchor_x == pytest.approx(11 / 200)
    assert extraction.spans[0].anchor_y == pytest.approx((120 - 75) / 120)
    assert [(cell.row_index, cell.column_index) for cell in extraction.table_cells] == [
        (row, column) for row in (0, 1) for column in range(5)
    ]
    assert extraction.table_cells[0].text == "cell-0-0"
    assert extraction.table_cells[-1].text == "cell-1-4"


def test_byte_only_seam_extracts_without_preview_request_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations = _grid_operations(columns=5)
    _install_fake_pypdf(monkeypatch, [_FakePage(operations, width=200, height=120)])

    extraction = vector_grid.extract_native_pdf_vector_grid_from_pdf_bytes(
        pdf_bytes=b"%PDF-byte-only-native-vector-grid"
    )

    assert len(extraction.spans) == 10
    assert len(extraction.table_cells) == 10


def test_grid_detection_is_translation_and_scale_invariant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = _FakePage(_grid_operations(columns=6), width=200, height=120)
    _install_fake_pypdf(monkeypatch, [first])
    baseline = vector_grid.extract_native_pdf_vector_grid(preview_input=_preview_input())

    translated = _FakePage(
        _grid_operations(columns=6, x_offset=40.0, y_offset=35.0, scale=1.5),
        width=400,
        height=300,
    )
    _install_fake_pypdf(monkeypatch, [translated])
    shifted = vector_grid.extract_native_pdf_vector_grid(preview_input=_preview_input())

    assert [(cell.row_index, cell.column_index, cell.text) for cell in baseline.table_cells] == [
        (cell.row_index, cell.column_index, cell.text) for cell in shifted.table_cells
    ]
    assert len(baseline.table_cells) == len(shifted.table_cells) == 12


def test_preserves_individual_text_show_boundaries_inside_a_cell(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations = _grid_operations(columns=5)
    text_start = operations.index(_operation("ET"))
    operations[text_start:text_start] = [
        _operation("Tm", (1, 0, 0, 1, 11, 50)),
        _operation("Tj", ("first",)),
        _operation("T*"),
        _operation("Tj", ("second",)),
    ]
    _install_fake_pypdf(monkeypatch, [_FakePage(operations, width=200, height=120)])

    extraction = vector_grid.extract_native_pdf_vector_grid(preview_input=_preview_input())

    body_first = next(
        cell for cell in extraction.table_cells if cell.row_index == 1 and cell.column_index == 0
    )
    assert body_first.text == "cell-1-0\nfirst\nsecond"


def test_text_next_line_without_local_leading_keeps_the_default_baseline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations = [
        _operation("BT"),
        _operation("Tf", ("F1", 10)),
        _operation("Tm", (1, 0, 0, 1, 20, 40)),
        _operation("T*"),
        _operation("Tj", ("generic-token",)),
        _operation("ET"),
    ]
    _install_fake_pypdf(monkeypatch, [_FakePage(operations, width=100, height=100)])

    extraction = vector_grid.extract_native_pdf_vector_grid(preview_input=_preview_input())

    assert len(extraction.spans) == 1
    assert extraction.spans[0].anchor_x == pytest.approx(0.2)
    assert extraction.spans[0].anchor_y == pytest.approx(0.6)


def test_decodes_pypdf_candidate_byte_string_without_external_font_mapping(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    operations = [
        _operation("BT"),
        _operation("Tf", ("F1", 10)),
        _operation("Tm", (1, 0, 0, 1, 20, 40)),
        _operation("Tj", (_FakeByteStringObject(b"generic-token"),)),
        _operation("ET"),
    ]
    _install_fake_pypdf(monkeypatch, [_FakePage(operations, width=100, height=100)])

    extraction = vector_grid.extract_native_pdf_vector_grid(preview_input=_preview_input())

    assert [span.text for span in extraction.spans] == ["generic-token"]


def test_ambiguous_multiple_rectangle_grids_emit_no_table_cells(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_pypdf(
        monkeypatch,
        [_FakePage(_grid_operations(columns=5, include_second_grid=True), width=300, height=120)],
    )

    extraction = vector_grid.extract_native_pdf_vector_grid(preview_input=_preview_input())

    assert len(extraction.spans) == 20
    assert extraction.table_cells == ()


def test_missing_visible_text_in_a_rectangle_fails_closed_for_the_table(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_pypdf(
        monkeypatch,
        [_FakePage(_grid_operations(columns=5, include_all_text=False), width=200, height=120)],
    )

    extraction = vector_grid.extract_native_pdf_vector_grid(preview_input=_preview_input())

    assert len(extraction.spans) == 9
    assert extraction.table_cells == ()


def test_unsupported_graphics_transform_is_content_free_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _install_fake_pypdf(
        monkeypatch,
        [
            _FakePage(
                [
                    _operation("cm", (1, 0, 0, 1, 3, 0)),
                    _operation("BT"),
                    _operation("Tf", ("F1", 10)),
                    _operation("Tm", (1, 0, 0, 1, 10, 10)),
                    _operation("Tj", ("not-in-error",)),
                    _operation("ET"),
                ],
                width=100,
                height=100,
            )
        ],
    )

    with pytest.raises(vector_grid.NativePdfVectorGridError) as raised:
        vector_grid.extract_native_pdf_vector_grid(preview_input=_preview_input())

    assert str(raised.value) == "native_vector_transform_unsupported"
    assert "not-in-error" not in str(raised.value)


def test_module_boundary_excludes_source_truth_and_has_one_candidate_input() -> None:
    source = inspect.getsource(vector_grid)
    tree = ast.parse(source)
    imported_modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert "customs_ai_agent.evaluation.source_attested_synthetic_pdf_truth" not in imported_modules
    assert "customs_ai_agent.evaluation.document_gold" not in imported_modules
    assert tuple(inspect.signature(vector_grid.extract_native_pdf_vector_grid).parameters) == (
        "preview_input",
    )
