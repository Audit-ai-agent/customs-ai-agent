from __future__ import annotations

import inspect

import pytest
from pydantic import ValidationError

from customs_ai_agent.evaluation.document_ocr_quality import CandidateSpanV1, GoldSpanV2
from customs_ai_agent.evaluation.synthetic_pdf_visitor_table_projector import (
    project_synthetic_pdf_visitor_table_cells,
)
from customs_ai_agent.schemas.document_extraction import BoundingBoxV1, DocumentSpanV1


_AUTO_BBOX = object()


def _span(
    *,
    text: str,
    reading_order: int,
    x: float,
    y: float,
    width: float = 0.035,
    height: float = 0.02,
    page_number: int = 1,
    bbox: BoundingBoxV1 | None | object = _AUTO_BBOX,
    anchor_x: float | None = None,
    anchor_y: float | None = None,
) -> CandidateSpanV1:
    resolved_bbox = (
        BoundingBoxV1(x=x, y=y, width=width, height=height) if bbox is _AUTO_BBOX else bbox
    )
    return CandidateSpanV1(
        page_number=page_number,
        reading_order=reading_order,
        text=text,
        bbox=resolved_bbox,
        anchor_x=anchor_x,
        anchor_y=anchor_y,
    )


def _fragmented_grid(
    *,
    columns: int,
    prefix: str,
    geometry: str = "bbox",
    reading_order_start: int = 1,
    header_y: float = 0.40,
    body_y: float = 0.46,
    page_number: int = 1,
) -> list[CandidateSpanV1]:
    """Build opaque two-fragment cells from geometry, never header labels."""

    spans: list[CandidateSpanV1] = []
    reading_order = reading_order_start
    for row_name, y in (("header", header_y), ("body", body_y)):
        for column_index in range(columns):
            x = 0.06 + column_index * 0.15
            for suffix, fragment_x in (("a", x), ("b", x + 0.045)):
                spans.append(
                    _span(
                        text=f"{prefix}-{row_name}-{column_index}-{suffix}",
                        reading_order=reading_order,
                        x=fragment_x,
                        y=y,
                        page_number=page_number,
                        bbox=(
                            None
                            if geometry == "anchor"
                            else BoundingBoxV1(x=fragment_x, y=y, width=0.035, height=0.02)
                        ),
                        anchor_x=fragment_x if geometry == "anchor" else None,
                        anchor_y=y if geometry == "anchor" else None,
                    )
                )
                reading_order += 1
    return spans


@pytest.mark.parametrize("columns", [5, 6], ids=["five-columns", "six-columns"])
@pytest.mark.parametrize("geometry", ["bbox", "anchor"], ids=["public-bbox", "local-anchor"])
def test_projects_one_fragmented_structural_grid_without_document_type_or_text_predicate(
    columns: int, geometry: str
) -> None:
    spans = _fragmented_grid(columns=columns, prefix="opaque", geometry=geometry)

    cells = project_synthetic_pdf_visitor_table_cells(spans=spans)

    assert [(cell.page_number, cell.table_index) for cell in cells] == [(1, 1)] * (columns * 2)
    assert [(cell.row_index, cell.column_index) for cell in cells] == [
        (row_index, column_index) for row_index in (0, 1) for column_index in range(columns)
    ]
    assert [cell.text for cell in cells] == [
        f"opaque-{row_name}-{column_index}-aopaque-{row_name}-{column_index}-b"
        for row_name in ("header", "body")
        for column_index in range(columns)
    ]


def test_uses_reading_order_for_header_when_vertical_coordinates_are_reversed() -> None:
    spans = _fragmented_grid(
        columns=5,
        prefix="opaque",
        header_y=0.62,
        body_y=0.56,
    )

    cells = project_synthetic_pdf_visitor_table_cells(spans=spans)

    assert [cell.text for cell in cells[:5]] == [
        f"opaque-header-{index}-aopaque-header-{index}-b" for index in range(5)
    ]


def test_bbox_path_has_priority_when_anchors_do_not_form_a_grid() -> None:
    spans = _fragmented_grid(columns=5, prefix="opaque")
    spans = [
        CandidateSpanV1(
            page_number=span.page_number,
            reading_order=span.reading_order,
            text=span.text,
            bbox=span.bbox,
            # All anchor points collapse into one line.  A fallback-first
            # implementation would fail, while the valid bbox grid must win.
            anchor_x=0.5,
            anchor_y=0.5,
        )
        for span in spans
    ]

    assert len(project_synthetic_pdf_visitor_table_cells(spans=spans)) == 10


def test_uses_complete_anchor_grid_when_partial_bbox_grid_fails_closed() -> None:
    anchored = _fragmented_grid(columns=5, prefix="opaque", geometry="anchor")
    spans = [
        CandidateSpanV1(
            page_number=span.page_number,
            reading_order=span.reading_order,
            text=span.text,
            bbox=(
                None
                if index == len(anchored) - 1
                else BoundingBoxV1(
                    x=span.anchor_x or 0.0,
                    y=span.anchor_y or 0.0,
                    width=0.035,
                    height=0.02,
                )
            ),
            anchor_x=span.anchor_x,
            anchor_y=span.anchor_y,
        )
        for index, span in enumerate(anchored)
    ]

    assert len(project_synthetic_pdf_visitor_table_cells(spans=spans)) == 10


@pytest.mark.parametrize("malformation", ["missing", "unlocated", "misaligned", "interleaved"])
def test_bbox_path_fails_closed_for_missing_or_ambiguous_grid(malformation: str) -> None:
    spans = _fragmented_grid(columns=5, prefix="opaque")
    if malformation == "missing":
        spans = spans[:-2]
    elif malformation == "unlocated":
        target = spans[-1]
        spans[-1] = CandidateSpanV1(
            page_number=target.page_number,
            reading_order=target.reading_order,
            text=target.text,
            bbox=None,
        )
    elif malformation == "misaligned":
        for index in (-2, -1):
            target = spans[index]
            assert target.bbox is not None
            spans[index] = _span(
                text=target.text,
                reading_order=target.reading_order,
                x=target.bbox.x + 0.09,
                y=target.bbox.y,
            )
    else:
        spans = [
            CandidateSpanV1(
                page_number=span.page_number,
                reading_order=(
                    span.reading_order * 2
                    if span.reading_order <= 10
                    else span.reading_order * 2 - 19
                ),
                text=span.text,
                bbox=span.bbox,
            )
            for span in spans
        ]

    assert project_synthetic_pdf_visitor_table_cells(spans=spans) == []


def test_fails_closed_for_invalid_model_constructed_bbox() -> None:
    spans = _fragmented_grid(columns=5, prefix="opaque")
    spans[-1] = CandidateSpanV1.model_construct(
        page_number=1,
        reading_order=spans[-1].reading_order,
        text=spans[-1].text,
        bbox=BoundingBoxV1.model_construct(x=float("nan"), y=0.46, width=0.035, height=0.02),
        anchor_x=None,
        anchor_y=None,
    )

    assert project_synthetic_pdf_visitor_table_cells(spans=spans) == []


def test_anchor_path_fails_closed_for_tied_boundaries_and_missing_anchor() -> None:
    tied = _fragmented_grid(columns=5, prefix="opaque", geometry="anchor")
    tied = [
        CandidateSpanV1(
            page_number=span.page_number,
            reading_order=span.reading_order,
            text=span.text,
            bbox=None,
            anchor_x=0.05 + ((span.reading_order - 1) % 10) * 0.02,
            anchor_y=0.40 if span.reading_order <= 10 else 0.46,
        )
        for span in tied
    ]
    assert project_synthetic_pdf_visitor_table_cells(spans=tied) == []

    missing = _fragmented_grid(columns=5, prefix="opaque", geometry="anchor")
    last = missing[-1]
    missing[-1] = CandidateSpanV1(
        page_number=last.page_number,
        reading_order=last.reading_order,
        text=last.text,
        bbox=None,
    )
    assert project_synthetic_pdf_visitor_table_cells(spans=missing) == []


def test_rejects_nearby_unassigned_anchor_line_that_could_be_cell_continuation() -> None:
    spans = _fragmented_grid(columns=5, prefix="opaque", geometry="anchor")
    spans.append(
        _span(
            text="continuation",
            reading_order=21,
            x=0.06,
            y=0.50,
            bbox=None,
            anchor_x=0.06,
            anchor_y=0.50,
        )
    )

    assert project_synthetic_pdf_visitor_table_cells(spans=spans) == []


def test_fails_closed_when_two_complete_structural_targets_exist() -> None:
    spans = [
        *_fragmented_grid(columns=5, prefix="first", header_y=0.30, body_y=0.36),
        *_fragmented_grid(
            columns=6,
            prefix="second",
            reading_order_start=30,
            header_y=0.60,
            body_y=0.66,
        ),
    ]

    assert project_synthetic_pdf_visitor_table_cells(spans=spans) == []


def test_ignores_unlocated_callback_outside_grid_order_and_rejects_page_two_only_grid() -> None:
    positioned = _fragmented_grid(columns=5, prefix="opaque")
    positioned.append(CandidateSpanV1(page_number=1, reading_order=99, text="outside", bbox=None))
    assert len(project_synthetic_pdf_visitor_table_cells(spans=positioned)) == 10

    page_two = _fragmented_grid(columns=5, prefix="opaque", page_number=2)
    assert project_synthetic_pdf_visitor_table_cells(spans=page_two) == []


def test_candidate_anchors_are_paired_local_evaluation_fields_only() -> None:
    with pytest.raises(ValidationError, match="both present or both absent"):
        CandidateSpanV1(page_number=1, reading_order=1, text="opaque", anchor_x=0.1)

    candidate = CandidateSpanV1(
        page_number=1,
        reading_order=1,
        text="opaque",
        anchor_x=0.1,
        anchor_y=0.2,
    )
    assert (candidate.anchor_x, candidate.anchor_y) == (0.1, 0.2)
    assert "anchor_x" not in GoldSpanV2.model_fields
    assert "anchor_x" not in DocumentSpanV1.model_fields


def test_projector_contract_accepts_only_candidate_spans() -> None:
    parameters = inspect.signature(project_synthetic_pdf_visitor_table_cells).parameters
    assert tuple(parameters) == ("spans",)
    assert all(
        forbidden not in parameter_name
        for parameter_name in parameters
        for forbidden in ("gold", "csv", "source", "fixture", "parser", "document_type")
    )
