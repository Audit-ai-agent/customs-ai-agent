"""Fail-closed geometry projection for native PDF visitor table candidates.

The native visitor adapter owns PDF parsing and supplies ``CandidateSpanV1``
objects.  This module accepts only those local candidate spans.  It does not
read a PDF, source layout, fixture, Gold, or CSV input.

Visitor callbacks are not a cell API: one visible cell can contain several
callbacks, and a callback can lack a usable font bounding box.  The projector
therefore supports two independent geometry paths:

* public-preview ``bbox`` values when a complete visible grid has them;
* local evaluation-only text-matrix ``anchor_x``/``anchor_y`` fallback.

Both paths use only visible geometry, preserve callback text without adding
characters, and emit no cells whenever the fixed synthetic grid is ambiguous.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Final, Literal, Sequence

from customs_ai_agent.evaluation.document_ocr_quality import (
    CandidateSpanV1,
    CandidateTableCellV1,
)
from customs_ai_agent.schemas.document_extraction import BoundingBoxV1


# Candidate-side structural selection deliberately has no document-type or
# header-text predicate.  The synthetic corpus contains exactly one complete
# target with either five or six columns; a second structural target is unsafe.
_COMPLETE_COLUMN_COUNTS: Final[frozenset[int]] = frozenset((5, 6))
_COLUMN_BREAK_GAP_HEIGHT_MULTIPLIER: Final = 2.0
_MAX_BBOX_ROW_GAP_HEIGHT_MULTIPLIER: Final = 8.0
_ANCHOR_BASELINE_EPSILON: Final = 0.0001
_MIN_ANCHOR_BOUNDARY_TO_INNER_GAP_RATIO: Final = 2.0
_MAX_ANCHOR_ROW_GAP_TO_COLUMN_PITCH_RATIO: Final = 1.0
_MAX_COLUMN_ALIGNMENT_PITCH_RATIO: Final = 0.35


@dataclass(frozen=True, slots=True)
class _InputVisitorSpan:
    """A validated non-empty page-one candidate callback."""

    reading_order: int
    text: str
    bbox: BoundingBoxV1 | None
    anchor_x: float | None
    anchor_y: float | None


@dataclass(frozen=True, slots=True)
class _GeometrySpan:
    """One callback represented by either a rectangle or a matrix point."""

    reading_order: int
    text: str
    left: float
    right: float
    top: float
    bottom: float

    @property
    def horizontal_center(self) -> float:
        return (self.left + self.right) / 2

    @property
    def height(self) -> float:
        return self.bottom - self.top


@dataclass(frozen=True, slots=True)
class _GeometryRow:
    """One baseline band with the original callback reading orders intact."""

    spans: tuple[_GeometrySpan, ...]
    top: float
    bottom: float

    @property
    def reading_orders(self) -> tuple[int, ...]:
        return tuple(span.reading_order for span in self.spans)

    @property
    def max_height(self) -> float:
        return max(span.height for span in self.spans)


@dataclass(frozen=True, slots=True)
class _CellRun:
    """A horizontal cell run; callback text is kept in visual x order."""

    spans: tuple[_GeometrySpan, ...]

    @property
    def left(self) -> float:
        return self.spans[0].left

    @property
    def right(self) -> float:
        return self.spans[-1].right

    @property
    def horizontal_center(self) -> float:
        return (self.left + self.right) / 2

    @property
    def text(self) -> str:
        # Do not infer whitespace from a point anchor.  Text-matrix geometry
        # proves relative placement, not whether adjacent callbacks represent
        # a word boundary.  Preserving the adapter output makes any text loss
        # visible to the normal quality gate instead of inventing characters.
        return "".join(span.text for span in self.spans)


@dataclass(frozen=True, slots=True)
class _VisibleGrid:
    """One complete visible header/body grid in stable candidate coordinates."""

    header: tuple[_CellRun, ...]
    body: tuple[_CellRun, ...]
    first_row_index: int
    second_row_index: int
    geometry_kind: Literal["bbox", "anchor"]
    column_pitch: float

    @property
    def reading_orders(self) -> tuple[int, ...]:
        return tuple(
            span.reading_order
            for row in (self.header, self.body)
            for run in row
            for span in run.spans
        )


def _is_finite_unit_value(value: object) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and isfinite(float(value))
        and 0 <= float(value) <= 1
    )


def _valid_bbox(bbox: object) -> bool:
    """Reject model-constructed or otherwise malformed local candidate bboxes."""

    if not isinstance(bbox, BoundingBoxV1):
        return False
    values = (bbox.x, bbox.y, bbox.width, bbox.height)
    if not all(_is_finite_unit_value(value) for value in values):
        return False
    x, y, width, height = (float(value) for value in values)
    return width > 0 and height > 0 and x + width <= 1 and y + height <= 1


def _validated_page_one_spans(
    spans: Sequence[CandidateSpanV1],
) -> tuple[_InputVisitorSpan, ...] | None:
    """Validate page-one provenance, callback order, bbox, and anchor pairs."""

    page_one: list[_InputVisitorSpan] = []
    seen_reading_orders: set[int] = set()
    for span in spans:
        page_number = getattr(span, "page_number", None)
        if (
            isinstance(page_number, bool)
            or not isinstance(page_number, int)
            or page_number < 1
        ):
            return None
        if page_number != 1:
            continue
        reading_order = getattr(span, "reading_order", None)
        if (
            isinstance(reading_order, bool)
            or not isinstance(reading_order, int)
            or reading_order < 1
            or reading_order in seen_reading_orders
        ):
            return None
        seen_reading_orders.add(reading_order)
        text = getattr(span, "text", None)
        if not isinstance(text, str):
            return None
        if not text.strip():
            continue
        bbox = getattr(span, "bbox", None)
        if bbox is not None and not _valid_bbox(bbox):
            return None
        anchor_x = getattr(span, "anchor_x", None)
        anchor_y = getattr(span, "anchor_y", None)
        if (anchor_x is None) != (anchor_y is None):
            return None
        if anchor_x is not None and (
            not _is_finite_unit_value(anchor_x) or not _is_finite_unit_value(anchor_y)
        ):
            return None
        page_one.append(
            _InputVisitorSpan(
                reading_order=reading_order,
                text=text,
                bbox=bbox,
                anchor_x=float(anchor_x) if anchor_x is not None else None,
                anchor_y=float(anchor_y) if anchor_y is not None else None,
            )
        )
    return tuple(page_one)


def _bbox_geometry(
    spans: Sequence[_InputVisitorSpan],
) -> tuple[tuple[_GeometrySpan, ...], frozenset[int]]:
    """Build the public-bbox path without mixing anchors into its geometry."""

    geometry: list[_GeometrySpan] = []
    unlocated_orders: set[int] = set()
    for span in spans:
        if span.bbox is None:
            unlocated_orders.add(span.reading_order)
            continue
        geometry.append(
            _GeometrySpan(
                reading_order=span.reading_order,
                text=span.text,
                left=span.bbox.x,
                right=span.bbox.x + span.bbox.width,
                top=span.bbox.y,
                bottom=span.bbox.y + span.bbox.height,
            )
        )
    return tuple(geometry), frozenset(unlocated_orders)


def _anchor_geometry(
    spans: Sequence[_InputVisitorSpan],
) -> tuple[tuple[_GeometrySpan, ...], frozenset[int]]:
    """Build the local matrix-anchor path without estimating public bboxes."""

    geometry: list[_GeometrySpan] = []
    unlocated_orders: set[int] = set()
    for span in spans:
        if span.anchor_x is None or span.anchor_y is None:
            unlocated_orders.add(span.reading_order)
            continue
        geometry.append(
            _GeometrySpan(
                reading_order=span.reading_order,
                text=span.text,
                left=span.anchor_x,
                right=span.anchor_x,
                top=span.anchor_y,
                bottom=span.anchor_y,
            )
        )
    return tuple(geometry), frozenset(unlocated_orders)


def _bbox_rows(spans: Sequence[_GeometrySpan]) -> list[_GeometryRow]:
    """Create rows only where all member rectangles share a real y interval."""

    if not spans:
        return []
    rows: list[_GeometryRow] = []
    current: list[_GeometrySpan] = []
    common_top = 0.0
    common_bottom = 0.0
    for span in sorted(spans, key=lambda item: (item.top, item.bottom, item.left, item.reading_order)):
        if not current:
            current = [span]
            common_top = span.top
            common_bottom = span.bottom
            continue
        next_top = max(common_top, span.top)
        next_bottom = min(common_bottom, span.bottom)
        if next_top < next_bottom:
            current.append(span)
            common_top = next_top
            common_bottom = next_bottom
            continue
        rows.append(_GeometryRow(spans=tuple(current), top=common_top, bottom=common_bottom))
        current = [span]
        common_top = span.top
        common_bottom = span.bottom
    rows.append(_GeometryRow(spans=tuple(current), top=common_top, bottom=common_bottom))
    return rows


def _anchor_rows(spans: Sequence[_GeometrySpan]) -> list[_GeometryRow]:
    """Cluster only nearly identical matrix baselines into one visible row."""

    if not spans:
        return []
    rows: list[_GeometryRow] = []
    current: list[_GeometrySpan] = []
    minimum_y = 0.0
    maximum_y = 0.0
    for span in sorted(spans, key=lambda item: (item.top, item.left, item.reading_order)):
        if not current:
            current = [span]
            minimum_y = span.top
            maximum_y = span.top
            continue
        next_minimum = min(minimum_y, span.top)
        next_maximum = max(maximum_y, span.top)
        if next_maximum - next_minimum <= _ANCHOR_BASELINE_EPSILON:
            current.append(span)
            minimum_y = next_minimum
            maximum_y = next_maximum
            continue
        rows.append(_GeometryRow(spans=tuple(current), top=minimum_y, bottom=maximum_y))
        current = [span]
        minimum_y = span.top
        maximum_y = span.top
    rows.append(_GeometryRow(spans=tuple(current), top=minimum_y, bottom=maximum_y))
    return rows


def _complete_runs(groups: Sequence[Sequence[_GeometrySpan]]) -> tuple[_CellRun, ...] | None:
    runs = tuple(_CellRun(spans=tuple(group)) for group in groups)
    if len(runs) not in _COMPLETE_COLUMN_COUNTS or any(not run.text.strip() for run in runs):
        return None
    centers = [run.horizontal_center for run in runs]
    if any(left >= right for left, right in zip(centers, centers[1:])):
        return None
    return runs


def _bbox_row_run_candidates(row: _GeometryRow) -> tuple[tuple[_CellRun, ...], ...]:
    """Split a bbox baseline at only unmistakably large visible gaps."""

    ordered = tuple(sorted(row.spans, key=lambda span: (span.left, span.right, span.reading_order)))
    if not ordered:
        return ()
    groups: list[list[_GeometrySpan]] = [[ordered[0]]]
    for span in ordered[1:]:
        previous = groups[-1][-1]
        gap = span.left - previous.right
        # Rectangle overlap does not expose a stable fragment sequence.
        if gap < 0:
            return ()
        if gap > max(previous.height, span.height) * _COLUMN_BREAK_GAP_HEIGHT_MULTIPLIER:
            groups.append([span])
        else:
            groups[-1].append(span)
    runs = _complete_runs(groups)
    return (runs,) if runs is not None else ()


def _anchor_runs_for_columns(
    ordered: Sequence[_GeometrySpan], *, columns: int
) -> tuple[_CellRun, ...] | None:
    """Choose only clearly dominant matrix-x gaps as grid boundaries."""

    if len(ordered) < columns:
        return None
    gaps = [right.left - left.left for left, right in zip(ordered, ordered[1:])]
    if any(gap <= 0 for gap in gaps):
        return None
    if len(ordered) == columns:
        return _complete_runs([[span] for span in ordered])
    ranked = sorted(range(len(gaps)), key=lambda index: gaps[index], reverse=True)
    boundary_indexes = set(ranked[: columns - 1])
    boundary_gaps = [gaps[index] for index in boundary_indexes]
    inner_gaps = [gaps[index] for index in range(len(gaps)) if index not in boundary_indexes]
    if not boundary_gaps or not inner_gaps:
        return None
    # Tied or weakly separated gaps do not reveal which fragment belongs to
    # which cell, so point geometry alone must not guess a partition.
    if min(boundary_gaps) <= max(inner_gaps) * _MIN_ANCHOR_BOUNDARY_TO_INNER_GAP_RATIO:
        return None
    groups: list[list[_GeometrySpan]] = [[]]
    for index, span in enumerate(ordered):
        groups[-1].append(span)
        if index in boundary_indexes:
            groups.append([])
    if any(not group for group in groups):
        return None
    return _complete_runs(groups)


def _anchor_row_run_candidates(row: _GeometryRow) -> tuple[tuple[_CellRun, ...], ...]:
    """Return all independently unambiguous five- or six-column partitions."""

    ordered = tuple(sorted(row.spans, key=lambda span: (span.left, span.reading_order)))
    candidates = [
        runs
        for columns in sorted(_COMPLETE_COLUMN_COUNTS)
        if (runs := _anchor_runs_for_columns(ordered, columns=columns)) is not None
    ]
    return tuple(candidates)


def _header_body_rows(
    first: _GeometryRow, second: _GeometryRow
) -> tuple[_GeometryRow, _GeometryRow] | None:
    """Use non-interleaved callback order so y-origin differences are harmless."""

    first_orders = first.reading_orders
    second_orders = second.reading_orders
    if max(first_orders) < min(second_orders):
        return first, second
    if max(second_orders) < min(first_orders):
        return second, first
    return None


def _matching_column_pitch(
    header: Sequence[_CellRun], body: Sequence[_CellRun]
) -> float | None:
    """Verify a stable one-to-one horizontal slot for every visible column."""

    if len(header) != len(body) or len(header) not in _COMPLETE_COLUMN_COUNTS:
        return None
    centers = [
        (header_run.horizontal_center + body_run.horizontal_center) / 2
        for header_run, body_run in zip(header, body)
    ]
    pitches = [right - left for left, right in zip(centers, centers[1:])]
    if not pitches or min(pitches) <= 0:
        return None
    smallest_pitch = min(pitches)
    if any(
        abs(header_run.horizontal_center - body_run.horizontal_center)
        > smallest_pitch * _MAX_COLUMN_ALIGNMENT_PITCH_RATIO
        for header_run, body_run in zip(header, body)
    ):
        return None
    return smallest_pitch


def _bbox_rows_are_nearby(first: _GeometryRow, second: _GeometryRow, _: float) -> bool:
    vertical_gap = max(0.0, second.top - first.bottom, first.top - second.bottom)
    height = max(first.max_height, second.max_height)
    return vertical_gap <= height * _MAX_BBOX_ROW_GAP_HEIGHT_MULTIPLIER


def _anchor_rows_are_nearby(first: _GeometryRow, second: _GeometryRow, pitch: float) -> bool:
    vertical_gap = max(0.0, second.top - first.bottom, first.top - second.bottom)
    return vertical_gap <= pitch * _MAX_ANCHOR_ROW_GAP_TO_COLUMN_PITCH_RATIO


def _visible_grids(
    *,
    rows: Sequence[_GeometryRow],
    geometry_kind: Literal["bbox", "anchor"],
) -> list[_VisibleGrid]:
    run_candidates = (
        _bbox_row_run_candidates if geometry_kind == "bbox" else _anchor_row_run_candidates
    )
    rows_are_nearby = (
        _bbox_rows_are_nearby if geometry_kind == "bbox" else _anchor_rows_are_nearby
    )
    grids: list[_VisibleGrid] = []
    for index, (first, second) in enumerate(zip(rows, rows[1:])):
        header_row, body_row = _header_body_rows(first, second) or (None, None)
        if header_row is None or body_row is None:
            continue
        for header in run_candidates(header_row):
            for body in run_candidates(body_row):
                column_pitch = _matching_column_pitch(header, body)
                if column_pitch is None or not rows_are_nearby(first, second, column_pitch):
                    continue
                grids.append(
                    _VisibleGrid(
                        header=header,
                        body=body,
                        first_row_index=index,
                        second_row_index=index + 1,
                        geometry_kind=geometry_kind,
                        column_pitch=column_pitch,
                    )
                )
    return grids


def _row_conflicts_with_grid(
    row: _GeometryRow, grid: _VisibleGrid, rows: Sequence[_GeometryRow]
) -> bool:
    """Reject nearby unassigned callbacks that could be a line continuation."""

    first = rows[grid.first_row_index]
    second = rows[grid.second_row_index]
    table_top = min(first.top, second.top)
    table_bottom = max(first.bottom, second.bottom)
    table_left = min(run.left for run in (*grid.header, *grid.body))
    table_right = max(run.right for run in (*grid.header, *grid.body))
    vertical_gap = max(0.0, table_top - row.bottom, row.top - table_bottom)
    if grid.geometry_kind == "bbox":
        height = max(first.max_height, second.max_height, row.max_height)
        nearby = vertical_gap <= height * _MAX_BBOX_ROW_GAP_HEIGHT_MULTIPLIER
    else:
        nearby = vertical_gap <= grid.column_pitch * _MAX_ANCHOR_ROW_GAP_TO_COLUMN_PITCH_RATIO
    return nearby and any(span.left <= table_right and table_left <= span.right for span in row.spans)


def _has_unassigned_nearby_row(rows: Sequence[_GeometryRow], grid: _VisibleGrid) -> bool:
    return any(
        index not in (grid.first_row_index, grid.second_row_index)
        and _row_conflicts_with_grid(row, grid, rows)
        for index, row in enumerate(rows)
    )


def _has_unlocated_fragment_within_grid_order(
    unlocated_orders: frozenset[int], grid: _VisibleGrid
) -> bool:
    """Reject an unlocated callback whose ordering may place it in the grid."""

    if not unlocated_orders:
        return False
    orders = grid.reading_orders
    # A missing final/initial callback is just outside the surviving located
    # range, so include the immediately adjacent callback positions as well.
    # This intentionally rejects a borderline unrelated callback rather than
    # silently dropping a table fragment.
    return any(min(orders) - 1 <= order <= max(orders) + 1 for order in unlocated_orders)


def _project_geometry(
    *,
    geometry: Sequence[_GeometrySpan],
    unlocated_orders: frozenset[int],
    geometry_kind: Literal["bbox", "anchor"],
) -> list[CandidateTableCellV1]:
    rows = _bbox_rows(geometry) if geometry_kind == "bbox" else _anchor_rows(geometry)
    targets = [
        grid
        for grid in _visible_grids(rows=rows, geometry_kind=geometry_kind)
        if not _has_unassigned_nearby_row(rows, grid)
        and not _has_unlocated_fragment_within_grid_order(unlocated_orders, grid)
    ]
    if len(targets) != 1:
        return []
    target = targets[0]
    return [
        CandidateTableCellV1(
            page_number=1,
            table_index=1,
            row_index=row_index,
            column_index=column_index,
            text=cell.text,
        )
        for row_index, row in enumerate((target.header, target.body))
        for column_index, cell in enumerate(row)
    ]


def project_synthetic_pdf_visitor_table_cells(
    *, spans: Sequence[CandidateSpanV1]
) -> list[CandidateTableCellV1]:
    """Project exactly one structural five- or six-column synthetic table.

    The predicate reads neither candidate text nor document type.  It accepts
    only one page-one, two-row grid whose columns visibly align.  Bboxes are
    attempted first; only if no complete bbox grid is available does the
    local anchor path run.  Anchor points are never turned into public bboxes.

    A missing coordinate in the target callback block, invalid provenance or
    geometry, tied/weak anchor gaps, overlapping bbox fragments, extra nearby
    text line, incomplete grid, or more than one complete grid returns an
    empty list.  This is intentional fail-closed candidate output that can be
    passed directly into ``CandidateExtractionV1.table_cells``.
    """

    page_one = _validated_page_one_spans(spans)
    if page_one is None:
        return []
    bbox_geometry, bbox_unlocated = _bbox_geometry(page_one)
    bbox_cells = _project_geometry(
        geometry=bbox_geometry,
        unlocated_orders=bbox_unlocated,
        geometry_kind="bbox",
    )
    if bbox_cells:
        return bbox_cells
    anchor_geometry, anchor_unlocated = _anchor_geometry(page_one)
    return _project_geometry(
        geometry=anchor_geometry,
        unlocated_orders=anchor_unlocated,
        geometry_kind="anchor",
    )


__all__ = ["project_synthetic_pdf_visitor_table_cells"]
