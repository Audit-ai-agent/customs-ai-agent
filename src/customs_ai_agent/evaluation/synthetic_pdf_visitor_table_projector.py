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
_ANCHOR_X_COINCIDENCE_EPSILON: Final = 0.0001
_ANCHOR_GAP_TIE_EPSILON: Final = 0.000001
_MIN_ANCHOR_BOUNDARY_TO_INNER_GAP_RATIO: Final = 2.0
_MAX_ANCHOR_ROW_GAP_TO_COLUMN_PITCH_RATIO: Final = 1.0
_MAX_COLUMN_ALIGNMENT_PITCH_RATIO: Final = 0.35
_MIN_ANCHOR_COLUMN_RANGE_OVERLAP: Final = 0.95
_MAX_ANCHOR_COLUMN_PROFILE_DEVIATION: Final = 0.10
_MIN_ANCHOR_PROFILE_SELECTION_MARGIN: Final = 0.02
_MAX_ANCHOR_CONTINUATION_LANE_PITCH_RATIO: Final = 0.15
_MAX_ANCHOR_TRAILING_PARTIAL_GAP_PITCH_RATIO: Final = 0.40


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
        return min(span.left for span in self.spans)

    @property
    def right(self) -> float:
        return max(span.right for span in self.spans)

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
    anchor_profile_deviation: float | None
    reading_order_kind: Literal["row_major", "column_interleaved"]

    @property
    def reading_orders(self) -> tuple[int, ...]:
        return tuple(
            span.reading_order
            for row in (self.header, self.body)
            for run in row
            for span in run.spans
        )


@dataclass(frozen=True, slots=True)
class _ColumnMatch:
    """A geometry-only correspondence between equally-sized row slots."""

    column_pitch: float
    anchor_profile_deviation: float | None


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
        if isinstance(page_number, bool) or not isinstance(page_number, int) or page_number < 1:
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
    for span in sorted(
        spans, key=lambda item: (item.top, item.bottom, item.left, item.reading_order)
    ):
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


def _anchor_x_clusters(
    ordered: Sequence[_GeometrySpan],
) -> tuple[tuple[_GeometrySpan, ...], ...]:
    """Group only coincident text-start points before choosing cell breaks.

    ``pypdf`` can issue several callbacks for one visible text start.  A point
    anchor cannot establish a horizontal word advance, but equal starts do
    establish that those callbacks belong to one column lane.  Grouping only
    a very small coincidence tolerance is safer than treating the zero gap as
    a malformed second column or arbitrarily splitting it.
    """

    if not ordered:
        return ()
    clusters: list[list[_GeometrySpan]] = [[ordered[0]]]
    for span in ordered[1:]:
        previous = clusters[-1][-1]
        if span.left - previous.left <= _ANCHOR_X_COINCIDENCE_EPSILON:
            clusters[-1].append(span)
            continue
        clusters.append([span])
    return tuple(tuple(cluster) for cluster in clusters)


def _anchor_runs_for_columns(
    ordered: Sequence[_GeometrySpan], *, columns: int
) -> tuple[_CellRun, ...] | None:
    """Choose only an unambiguous matrix-x partition into visible columns."""

    clusters = _anchor_x_clusters(ordered)
    if len(clusters) < columns:
        return None
    if len(clusters) == columns:
        return _complete_runs(clusters)

    gaps = [right[0].left - left[-1].left for left, right in zip(clusters, clusters[1:])]
    if any(gap <= _ANCHOR_X_COINCIDENCE_EPSILON for gap in gaps):
        return None
    ranked = sorted(range(len(gaps)), key=lambda index: gaps[index], reverse=True)
    boundary_count = columns - 1
    boundary_indexes = set(ranked[:boundary_count])
    boundary_gaps = [gaps[index] for index in boundary_indexes]
    inner_gaps = [gaps[index] for index in range(len(gaps)) if index not in boundary_indexes]
    if not boundary_gaps or not inner_gaps:
        return None
    # A tie at the selection cut cannot reveal which callback belongs to a
    # neighboring cell.  Do not let the sort's input order choose for us.
    if (
        abs(gaps[ranked[boundary_count - 1]] - gaps[ranked[boundary_count]])
        <= _ANCHOR_GAP_TIE_EPSILON
    ):
        return None
    # Weakly separated gaps likewise do not prove a cell boundary.
    if min(boundary_gaps) <= max(inner_gaps) * _MIN_ANCHOR_BOUNDARY_TO_INNER_GAP_RATIO:
        return None
    groups: list[list[_GeometrySpan]] = [[]]
    for index, cluster in enumerate(clusters):
        groups[-1].extend(cluster)
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


def _columnwise_interleaved(first: Sequence[_CellRun], second: Sequence[_CellRun]) -> bool:
    """Accept only a complete per-column paint-order alternation.

    Native PDF drawing order can be ``header[0], body[0], header[1],
    body[1], ...`` instead of one entire row followed by the other.  This is
    still a geometry-only, unambiguous two-row signal when every cell-run is
    strictly before the corresponding run and the next column.  General
    interleaving remains rejected.
    """

    if len(first) != len(second):
        return False

    def _orders(run: _CellRun) -> tuple[int, int]:
        values = tuple(span.reading_order for span in run.spans)
        return min(values), max(values)

    def _alternates(left: Sequence[_CellRun], right: Sequence[_CellRun]) -> bool:
        for index, (left_run, right_run) in enumerate(zip(left, right)):
            _left_minimum, left_maximum = _orders(left_run)
            right_minimum, right_maximum = _orders(right_run)
            if left_maximum >= right_minimum:
                return False
            if index + 1 < len(left):
                next_left_minimum, _next_left_maximum = _orders(left[index + 1])
                if right_maximum >= next_left_minimum:
                    return False
        return True

    return _alternates(first, second) or _alternates(second, first)


def _header_body_rows(
    first_row: _GeometryRow,
    second_row: _GeometryRow,
    first_runs: Sequence[_CellRun],
    second_runs: Sequence[_CellRun],
) -> (
    tuple[
        _GeometryRow,
        _GeometryRow,
        tuple[_CellRun, ...],
        tuple[_CellRun, ...],
        Literal["row_major", "column_interleaved"],
    ]
    | None
):
    """Resolve safe logical row order without treating arbitrary order as table evidence."""

    first_orders = first_row.reading_orders
    second_orders = second_row.reading_orders
    if max(first_orders) < min(second_orders):
        return first_row, second_row, tuple(first_runs), tuple(second_runs), "row_major"
    if max(second_orders) < min(first_orders):
        return second_row, first_row, tuple(second_runs), tuple(first_runs), "row_major"
    if not _columnwise_interleaved(first_runs, second_runs):
        return None
    # Rows are enumerated in top-origin visual order.  An alternating PDF
    # paint order has no whole-row order to prefer, so only visual top/bottom
    # determines header/body.  The visitor adapter itself rejects transforms.
    if first_row.top <= second_row.top:
        return (
            first_row,
            second_row,
            tuple(first_runs),
            tuple(second_runs),
            "column_interleaved",
        )
    return (
        second_row,
        first_row,
        tuple(second_runs),
        tuple(first_runs),
        "column_interleaved",
    )


def _matching_bbox_column_pitch(
    header: Sequence[_CellRun], body: Sequence[_CellRun]
) -> _ColumnMatch | None:
    """Verify one-to-one public rectangle slots with the historic strict bound."""

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
    return _ColumnMatch(column_pitch=smallest_pitch, anchor_profile_deviation=None)


def _matching_anchor_column_pitch(
    header: Sequence[_CellRun], body: Sequence[_CellRun]
) -> _ColumnMatch | None:
    """Match text-start lanes without mistaking text alignment for cell edges.

    A visitor reports where text starts, not a cell rectangle.  Header text
    may be centered while a body value is left- or right-aligned inside the
    same column, so absolute start-point equality is not evidence.  The two
    rows must instead have a nearly identical normalized column profile and
    almost the same horizontal envelope.  Both conditions are required.
    """

    header_centers = [run.horizontal_center for run in header]
    body_centers = [run.horizontal_center for run in body]
    header_span = header_centers[-1] - header_centers[0]
    body_span = body_centers[-1] - body_centers[0]
    if header_span <= 0 or body_span <= 0:
        return None
    overlap = max(
        0.0,
        min(header_centers[-1], body_centers[-1]) - max(header_centers[0], body_centers[0]),
    )
    if overlap / min(header_span, body_span) < _MIN_ANCHOR_COLUMN_RANGE_OVERLAP:
        return None
    profile_deviation = max(
        abs(
            (header_center - header_centers[0]) / header_span
            - (body_center - body_centers[0]) / body_span
        )
        for header_center, body_center in zip(header_centers, body_centers)
    )
    if profile_deviation > _MAX_ANCHOR_COLUMN_PROFILE_DEVIATION:
        return None
    pitches = [
        *(right - left for left, right in zip(header_centers, header_centers[1:])),
        *(right - left for left, right in zip(body_centers, body_centers[1:])),
    ]
    if not pitches or min(pitches) <= 0:
        return None
    return _ColumnMatch(
        column_pitch=min(pitches),
        anchor_profile_deviation=profile_deviation,
    )


def _matching_column_pitch(
    header: Sequence[_CellRun],
    body: Sequence[_CellRun],
    *,
    geometry_kind: Literal["bbox", "anchor"],
) -> _ColumnMatch | None:
    """Verify a stable one-to-one horizontal slot for every visible column."""

    if len(header) != len(body) or len(header) not in _COMPLETE_COLUMN_COUNTS:
        return None
    if geometry_kind == "bbox":
        return _matching_bbox_column_pitch(header, body)
    return _matching_anchor_column_pitch(header, body)


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
    rows_are_nearby = _bbox_rows_are_nearby if geometry_kind == "bbox" else _anchor_rows_are_nearby
    grids: list[_VisibleGrid] = []
    row_indexes = (
        ((index, index + 1) for index in range(len(rows) - 1))
        if geometry_kind == "bbox"
        else (
            (first_index, second_index)
            for first_index in range(len(rows) - 1)
            for second_index in range(first_index + 1, len(rows))
        )
    )
    for first_index, second_index in row_indexes:
        first = rows[first_index]
        second = rows[second_index]
        for first_runs in run_candidates(first):
            for second_runs in run_candidates(second):
                ordered = _header_body_rows(first, second, first_runs, second_runs)
                if ordered is None:
                    continue
                (
                    _header_row,
                    _body_row,
                    header,
                    body,
                    reading_order_kind,
                ) = ordered
                match = _matching_column_pitch(
                    header,
                    body,
                    geometry_kind=geometry_kind,
                )
                if match is None or not rows_are_nearby(first, second, match.column_pitch):
                    continue
                grids.append(
                    _VisibleGrid(
                        header=header,
                        body=body,
                        first_row_index=first_index,
                        second_row_index=second_index,
                        geometry_kind=geometry_kind,
                        column_pitch=match.column_pitch,
                        anchor_profile_deviation=match.anchor_profile_deviation,
                        reading_order_kind=reading_order_kind,
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
    return nearby and any(
        span.left <= table_right and table_left <= span.right for span in row.spans
    )


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


def _anchor_continuation_lanes(
    row: _GeometryRow,
    *,
    cells: Sequence[_CellRun],
    column_pitch: float,
) -> tuple[int, ...] | None:
    """Map only clearly lane-aligned auxiliary callbacks to one logical row."""

    lanes: list[int] = []
    centers = tuple(cell.horizontal_center for cell in cells)
    for span in row.spans:
        distances = tuple(abs(span.horizontal_center - center) for center in centers)
        smallest = min(distances)
        indexes = [
            index
            for index, distance in enumerate(distances)
            if abs(distance - smallest) <= _ANCHOR_GAP_TIE_EPSILON
        ]
        if len(indexes) != 1:
            return None
        if smallest > column_pitch * _MAX_ANCHOR_CONTINUATION_LANE_PITCH_RATIO:
            return None
        lanes.append(indexes[0])
    return tuple(lanes)


def _with_anchor_continuations(
    runs: Sequence[_CellRun], additions: Sequence[Sequence[_GeometrySpan]]
) -> tuple[_CellRun, ...]:
    """Append proven later callback fragments without inventing separators."""

    return tuple(
        _CellRun(
            spans=tuple(
                (
                    *run.spans,
                    *sorted(extra, key=lambda span: span.reading_order),
                )
            )
        )
        for run, extra in zip(runs, additions)
    )


def _anchor_row_is_closer_to_another_complete_lattice(
    *,
    row_index: int,
    rows: Sequence[_GeometryRow],
    grid: _VisibleGrid,
) -> bool:
    """Avoid assigning a partial line to the wrong neighboring lattice."""

    row = rows[row_index]
    selected_distance = min(
        abs(row.top - rows[grid.first_row_index].top),
        abs(row.top - rows[grid.second_row_index].top),
    )
    return any(
        other_index not in (grid.first_row_index, grid.second_row_index)
        and bool(_anchor_row_run_candidates(other))
        and abs(row.top - other.top) + _ANCHOR_BASELINE_EPSILON < selected_distance
        for other_index, other in enumerate(rows)
    )


def _anchor_grid_with_continuations(
    rows: Sequence[_GeometryRow], grid: _VisibleGrid
) -> _VisibleGrid | None:
    """Allow only partial, lane-aligned callbacks embedded in a logical row.

    Multiple visitor baselines can be one wrapped visible table cell.  They
    are retained only when their callback order is wholly inside one selected
    row, every point maps unambiguously to an existing lane, and they cover
    fewer than all columns.  A full extra line, an order gap, or an off-lane
    callback is not silently dropped: it invalidates this candidate grid.
    """

    header_orders: tuple[int, ...] = tuple(
        span.reading_order for run in grid.header for span in run.spans
    )
    body_orders: tuple[int, ...] = tuple(
        span.reading_order for run in grid.body for span in run.spans
    )
    if not header_orders or not body_orders:
        return None
    header_range = min(header_orders), max(header_orders)
    body_range = min(body_orders), max(body_orders)
    total_range = min(*header_range, *body_range), max(*header_range, *body_range)
    header_additions: list[list[_GeometrySpan]] = [[] for _ in grid.header]
    body_additions: list[list[_GeometrySpan]] = [[] for _ in grid.body]

    for index, row in enumerate(rows):
        if index in (grid.first_row_index, grid.second_row_index):
            continue
        if not _row_conflicts_with_grid(row, grid, rows):
            continue
        orders = row.reading_orders
        if not any(total_range[0] <= order <= total_range[1] for order in orders):
            # A partial line painted immediately after the selected body may
            # be a wrapped value, but its order no longer proves where it
            # belongs.  Reject it rather than dropping it.  A complete
            # five/six-column lattice is not itself an extra *grid*; it is
            # considered independently by target selection below.
            if min(orders) > total_range[1] and row.top >= max(
                rows[grid.first_row_index].top,
                rows[grid.second_row_index].top,
            ):
                trailing_gap = row.top - max(
                    rows[grid.first_row_index].top,
                    rows[grid.second_row_index].top,
                )
                if (
                    trailing_gap <= grid.column_pitch * _MAX_ANCHOR_TRAILING_PARTIAL_GAP_PITCH_RATIO
                    and not _anchor_row_is_closer_to_another_complete_lattice(
                        row_index=index,
                        rows=rows,
                        grid=grid,
                    )
                ):
                    return None
            continue
        # Alternating primary rows have no disjoint whole-row order interval
        # into which a wrapped continuation can be placed safely.
        if grid.reading_order_kind != "row_major":
            return None
        if all(header_range[0] <= order <= header_range[1] for order in orders):
            lanes = _anchor_continuation_lanes(
                row,
                cells=grid.header,
                column_pitch=grid.column_pitch,
            )
            additions = header_additions
        elif all(body_range[0] <= order <= body_range[1] for order in orders):
            lanes = _anchor_continuation_lanes(
                row,
                cells=grid.body,
                column_pitch=grid.column_pitch,
            )
            additions = body_additions
        else:
            return None
        if lanes is None or len(set(lanes)) >= len(additions):
            return None
        for span, lane in zip(row.spans, lanes):
            additions[lane].append(span)

    if not any(header_additions) and not any(body_additions):
        return grid
    return _VisibleGrid(
        header=_with_anchor_continuations(grid.header, header_additions),
        body=_with_anchor_continuations(grid.body, body_additions),
        first_row_index=grid.first_row_index,
        second_row_index=grid.second_row_index,
        geometry_kind=grid.geometry_kind,
        column_pitch=grid.column_pitch,
        anchor_profile_deviation=grid.anchor_profile_deviation,
        reading_order_kind=grid.reading_order_kind,
    )


def _select_anchor_grid(grids: Sequence[_VisibleGrid]) -> _VisibleGrid | None:
    """Choose only one geometrically dominant target, never a sort-order tie."""

    if not grids:
        return None
    if len(grids) == 1:
        return grids[0]
    if any(grid.anchor_profile_deviation is None for grid in grids):
        return None
    ranked = sorted(grids, key=lambda grid: grid.anchor_profile_deviation or 0.0)
    target = ranked[0]
    target_rows = frozenset((target.first_row_index, target.second_row_index))
    for alternative in ranked[1:]:
        alternative_rows = frozenset((alternative.first_row_index, alternative.second_row_index))
        # The same two physical rows with another partition are ambiguous;
        # disjoint full pairs are a second grid.  Neither is safe to select.
        if alternative_rows == target_rows or alternative_rows.isdisjoint(target_rows):
            return None
        if (alternative.anchor_profile_deviation or 0.0) - (
            target.anchor_profile_deviation or 0.0
        ) <= _MIN_ANCHOR_PROFILE_SELECTION_MARGIN:
            return None
    return target


def _project_geometry(
    *,
    geometry: Sequence[_GeometrySpan],
    unlocated_orders: frozenset[int],
    geometry_kind: Literal["bbox", "anchor"],
) -> list[CandidateTableCellV1]:
    rows = _bbox_rows(geometry) if geometry_kind == "bbox" else _anchor_rows(geometry)
    targets: list[_VisibleGrid] = []
    for grid in _visible_grids(rows=rows, geometry_kind=geometry_kind):
        if _has_unlocated_fragment_within_grid_order(unlocated_orders, grid):
            continue
        if geometry_kind == "bbox":
            if not _has_unassigned_nearby_row(rows, grid):
                targets.append(grid)
            continue
        projected = _anchor_grid_with_continuations(rows, grid)
        if projected is not None:
            targets.append(projected)
    target: _VisibleGrid | None
    if geometry_kind == "bbox":
        if len(targets) != 1:
            return []
        target = targets[0]
    else:
        target = _select_anchor_grid(targets)
    if target is None:
        return []
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
