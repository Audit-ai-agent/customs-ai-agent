"""Bounded candidate-only native PDF text/vector grid extraction.

This module is deliberately separate from source-attested layout truth.  It
reads only the selected candidate PDF bytes in a ``PreviewInputV1`` and
derives candidate spans from that PDF's own text-show operations plus a
single visible rectangle grid.  It does not import a Gold, manifest, CSV,
packet, source-layout profile, template coordinate, or fixed text-run index.

The supported operator subset is intentionally small.  An unsupported text
or graphics transform makes the extraction unavailable instead of guessing a
position.  That is appropriate for the current born-digital compatibility
candidate and must not be presented as optical OCR for scanned documents.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Final, Sequence

from customs_ai_agent.evaluation.document_ocr_quality import (
    CandidateSpanV1,
    CandidateTableCellV1,
)
from customs_ai_agent.schemas.document_extraction import PreviewInputV1


_IDENTITY_MATRIX: Final[tuple[float, float, float, float, float, float]] = (
    1.0,
    0.0,
    0.0,
    1.0,
    0.0,
    0.0,
)
_COMPLETE_GRID_COLUMN_COUNTS: Final[frozenset[int]] = frozenset({5, 6})
_MAX_PAGES: Final = 1_000
_MAX_TEXT_RUNS_PER_PAGE: Final = 10_000
_MAX_RECTANGLES_PER_PAGE: Final = 10_000
_MAX_TEXT_RUN_LENGTH: Final = 8_000
_COORDINATE_RELATIVE_EPSILON: Final = 0.000_001


class NativePdfVectorGridError(ValueError):
    """Content-free failure emitted when native vector evidence is unsafe."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class NativePdfVectorGridPage:
    """One parsed PDF page whose dimensions are safe for preview projection."""

    page_number: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class NativePdfVectorFormRegion:
    """One candidate-only non-table rectangle in normalized page coordinates."""

    page_number: int
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class NativePdfVectorGridExtraction:
    """Candidate-only native text spans and one optional direct table grid."""

    pages: tuple[NativePdfVectorGridPage, ...]
    spans: tuple[CandidateSpanV1, ...]
    table_cells: tuple[CandidateTableCellV1, ...]
    form_regions: tuple[NativePdfVectorFormRegion, ...]


@dataclass(frozen=True, slots=True)
class _TextRun:
    page_number: int
    reading_order: int
    x: float
    y: float
    text: str


@dataclass(frozen=True, slots=True)
class _Rectangle:
    ordinal: int
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class _RectangleBand:
    y: float
    height: float
    rectangles: tuple[_Rectangle, ...]


def _close(left: float, right: float) -> bool:
    """Compare PDF coordinates without a template-specific absolute position."""

    return abs(left - right) <= max(1.0, abs(left), abs(right)) * _COORDINATE_RELATIVE_EPSILON


def _finite_number(value: object, *, code: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise NativePdfVectorGridError(code)
    result = float(value)
    if not math.isfinite(result):
        raise NativePdfVectorGridError(code)
    return result


def _operation_name(value: object) -> str:
    if isinstance(value, bytes):
        try:
            return value.decode("ascii")
        except UnicodeDecodeError as error:
            raise NativePdfVectorGridError("native_vector_operator_invalid") from error
    if isinstance(value, str) and value.isascii():
        return value
    raise NativePdfVectorGridError("native_vector_operator_invalid")


def _page_dimensions(page: object) -> tuple[int, int]:
    try:
        mediabox = getattr(page, "mediabox")
        width = math.ceil(
            _finite_number(getattr(mediabox, "width"), code="native_vector_page_invalid")
        )
        height = math.ceil(
            _finite_number(getattr(mediabox, "height"), code="native_vector_page_invalid")
        )
    except NativePdfVectorGridError:
        raise
    except (AttributeError, TypeError, ValueError, OverflowError) as error:
        raise NativePdfVectorGridError("native_vector_page_invalid") from error
    if not 1 <= width <= 100_000 or not 1 <= height <= 100_000:
        raise NativePdfVectorGridError("native_vector_page_invalid")
    return width, height


def _require_operand_count(
    operands: Sequence[object], *, count: int, code: str
) -> tuple[object, ...]:
    if len(operands) != count:
        raise NativePdfVectorGridError(code)
    return tuple(operands)


def _matrix_from_operands(operands: Sequence[object], *, code: str) -> tuple[float, ...]:
    return tuple(
        _finite_number(value, code=code)
        for value in _require_operand_count(operands, count=6, code=code)
    )


def _identity_linear_transform(matrix: Sequence[float]) -> bool:
    return all(
        _close(actual, expected)
        for actual, expected in zip(matrix[:4], _IDENTITY_MATRIX[:4], strict=True)
    )


def _parse_page_operations(
    *,
    page: object,
    reader: object,
    page_number: int,
    width: int,
    height: int,
) -> tuple[tuple[_TextRun, ...], tuple[_Rectangle, ...]]:
    """Parse only safe untransformed text runs and stroked rectangles."""

    try:
        from pypdf.generic import (  # type: ignore[import-not-found]
            ByteStringObject,
            ContentStream,
        )

        content = getattr(page, "get_contents")()
        operations = ContentStream(content, reader).operations
    except NativePdfVectorGridError:
        raise
    except Exception as error:
        raise NativePdfVectorGridError("native_vector_content_unavailable") from error

    in_text = False
    text_matrix: tuple[float, float, float, float, float, float] = _IDENTITY_MATRIX
    # PDF text leading begins at zero and is a text-state parameter, so a
    # ``T*`` before a local ``TL`` is well-defined: it leaves the baseline
    # unchanged.  It also survives a later ``BT``/``ET`` pair.
    leading = 0.0
    runs: list[_TextRun] = []
    rectangles: list[_Rectangle] = []
    pending_rectangles: list[_Rectangle] = []

    for operation_entry in operations:
        if not isinstance(operation_entry, tuple) or len(operation_entry) != 2:
            raise NativePdfVectorGridError("native_vector_operator_invalid")
        operands_raw, operator_raw = operation_entry
        if not isinstance(operands_raw, Sequence) or isinstance(operands_raw, (str, bytes)):
            raise NativePdfVectorGridError("native_vector_operand_invalid")
        operands = tuple(operands_raw)
        operator = _operation_name(operator_raw)

        if operator == "cm":
            if in_text or pending_rectangles:
                raise NativePdfVectorGridError("native_vector_graphics_state_invalid")
            matrix = _matrix_from_operands(operands, code="native_vector_transform_invalid")
            if tuple(matrix) != _IDENTITY_MATRIX:
                raise NativePdfVectorGridError("native_vector_transform_unsupported")
            continue
        if operator == "BT":
            if operands or in_text or pending_rectangles:
                raise NativePdfVectorGridError("native_vector_text_state_invalid")
            in_text = True
            text_matrix = _IDENTITY_MATRIX
            continue
        if operator == "ET":
            if operands or not in_text:
                raise NativePdfVectorGridError("native_vector_text_state_invalid")
            in_text = False
            continue
        if operator == "Tf":
            values = _require_operand_count(
                operands, count=2, code="native_vector_text_state_invalid"
            )
            if not in_text or not isinstance(values[0], str):
                raise NativePdfVectorGridError("native_vector_text_state_invalid")
            _finite_number(values[1], code="native_vector_text_state_invalid")
            continue
        if operator == "TL":
            values = _require_operand_count(
                operands, count=1, code="native_vector_text_state_invalid"
            )
            if not in_text:
                raise NativePdfVectorGridError("native_vector_text_state_invalid")
            parsed_leading = _finite_number(values[0], code="native_vector_text_state_invalid")
            leading = parsed_leading
            continue
        if operator == "Tm":
            if not in_text:
                raise NativePdfVectorGridError("native_vector_text_state_invalid")
            matrix = _matrix_from_operands(operands, code="native_vector_transform_invalid")
            if not _identity_linear_transform(matrix):
                raise NativePdfVectorGridError("native_vector_transform_unsupported")
            text_matrix = (
                matrix[0],
                matrix[1],
                matrix[2],
                matrix[3],
                matrix[4],
                matrix[5],
            )
            continue
        if operator == "T*":
            if operands or not in_text:
                raise NativePdfVectorGridError("native_vector_text_state_invalid")
            text_matrix = (
                text_matrix[0],
                text_matrix[1],
                text_matrix[2],
                text_matrix[3],
                text_matrix[4],
                text_matrix[5] - leading,
            )
            continue
        if operator == "Tj":
            values = _require_operand_count(
                operands, count=1, code="native_vector_text_state_invalid"
            )
            if not in_text:
                raise NativePdfVectorGridError("native_vector_text_state_invalid")
            if isinstance(values[0], str):
                text = values[0]
            elif isinstance(values[0], ByteStringObject):
                # ``ByteStringObject`` is pypdf's candidate-PDF-local string
                # representation when its decoder cannot identify the source
                # encoding up front.  Use only pypdf's bounded conversion;
                # never guess a font map or consult external/source truth.
                try:
                    text = str(values[0])
                except Exception as error:
                    raise NativePdfVectorGridError("native_vector_text_decode_invalid") from error
            else:
                raise NativePdfVectorGridError("native_vector_text_state_invalid")
            if len(text) > _MAX_TEXT_RUN_LENGTH:
                raise NativePdfVectorGridError("native_vector_text_run_invalid")
            if not text.strip():
                continue
            x, y = text_matrix[4], text_matrix[5]
            if not 0 <= x <= width or not 0 <= y <= height:
                raise NativePdfVectorGridError("native_vector_text_position_invalid")
            if len(runs) >= _MAX_TEXT_RUNS_PER_PAGE:
                raise NativePdfVectorGridError("native_vector_text_run_invalid")
            runs.append(
                _TextRun(
                    page_number=page_number,
                    reading_order=len(runs) + 1,
                    x=x,
                    y=y,
                    text=text,
                )
            )
            continue
        if operator == "w":
            values = _require_operand_count(
                operands, count=1, code="native_vector_graphics_state_invalid"
            )
            if (
                in_text
                or _finite_number(values[0], code="native_vector_graphics_state_invalid") <= 0
            ):
                raise NativePdfVectorGridError("native_vector_graphics_state_invalid")
            continue
        if operator == "re":
            if in_text:
                raise NativePdfVectorGridError("native_vector_graphics_state_invalid")
            x, y, rectangle_width, rectangle_height = (
                _finite_number(value, code="native_vector_rectangle_invalid")
                for value in _require_operand_count(
                    operands, count=4, code="native_vector_rectangle_invalid"
                )
            )
            if rectangle_width <= 0 or rectangle_height <= 0:
                raise NativePdfVectorGridError("native_vector_rectangle_invalid")
            if len(rectangles) + len(pending_rectangles) >= _MAX_RECTANGLES_PER_PAGE:
                raise NativePdfVectorGridError("native_vector_rectangle_invalid")
            pending_rectangles.append(
                _Rectangle(
                    ordinal=len(rectangles) + len(pending_rectangles) + 1,
                    x=x,
                    y=y,
                    width=rectangle_width,
                    height=rectangle_height,
                )
            )
            continue
        if operator == "S":
            if operands or in_text or not pending_rectangles:
                raise NativePdfVectorGridError("native_vector_graphics_state_invalid")
            rectangles.extend(pending_rectangles)
            pending_rectangles.clear()
            continue
        if operator == "n":
            if operands or in_text:
                raise NativePdfVectorGridError("native_vector_graphics_state_invalid")
            pending_rectangles.clear()
            continue
        raise NativePdfVectorGridError("native_vector_operator_unsupported")

    if in_text or pending_rectangles:
        raise NativePdfVectorGridError("native_vector_content_invalid")
    return tuple(runs), tuple(rectangles)


def _complete_rectangle_bands(rectangles: Sequence[_Rectangle]) -> tuple[_RectangleBand, ...]:
    """Find horizontal 5/6-cell bands using only visible rectangle geometry."""

    grouped: list[list[_Rectangle]] = []
    for rectangle in sorted(
        rectangles, key=lambda item: (item.y, item.height, item.x, item.ordinal)
    ):
        if (
            grouped
            and _close(rectangle.y, grouped[-1][0].y)
            and _close(rectangle.height, grouped[-1][0].height)
        ):
            grouped[-1].append(rectangle)
        else:
            grouped.append([rectangle])
    bands: list[_RectangleBand] = []
    for group in grouped:
        ordered = tuple(sorted(group, key=lambda item: (item.x, item.width, item.ordinal)))
        if len(ordered) not in _COMPLETE_GRID_COLUMN_COUNTS:
            continue
        if any(
            left.x >= right.x or _close(left.x, right.x)
            for left, right in zip(ordered, ordered[1:])
        ):
            continue
        bands.append(_RectangleBand(y=ordered[0].y, height=ordered[0].height, rectangles=ordered))
    return tuple(bands)


def _same_column_geometry(left: _RectangleBand, right: _RectangleBand) -> bool:
    return len(left.rectangles) == len(right.rectangles) and all(
        _close(first.x, second.x) and _close(first.width, second.width)
        for first, second in zip(left.rectangles, right.rectangles, strict=True)
    )


def _select_rectangle_grid(
    rectangles: Sequence[_Rectangle],
) -> tuple[_RectangleBand, _RectangleBand] | None:
    """Select exactly one top/header and bottom/body rectangle pair, or none."""

    bands = _complete_rectangle_bands(rectangles)
    pairs = [
        (left, right)
        for index, left in enumerate(bands)
        for right in bands[index + 1 :]
        if _same_column_geometry(left, right)
    ]
    if len(pairs) != 1:
        return None
    upper, lower = sorted(pairs[0], key=lambda item: item.y, reverse=True)
    # A grid must have two distinct, non-overlapping horizontal bands.  Touching
    # borders are valid; a body rectangle may end exactly at the header edge.
    if upper.y <= lower.y or upper.y + _COORDINATE_RELATIVE_EPSILON < lower.y + lower.height:
        return None
    return upper, lower


def _point_in_rectangle(*, x: float, y: float, rectangle: _Rectangle) -> bool:
    tolerance = (
        max(
            1.0,
            abs(rectangle.x),
            abs(rectangle.y),
            abs(rectangle.width),
            abs(rectangle.height),
        )
        * _COORDINATE_RELATIVE_EPSILON
    )
    return (
        rectangle.x - tolerance <= x <= rectangle.x + rectangle.width + tolerance
        and rectangle.y - tolerance <= y <= rectangle.y + rectangle.height + tolerance
    )


def _cell_text(runs: Sequence[_TextRun]) -> str | None:
    """Retain text-show operation boundaries rather than inventing characters."""

    if not runs:
        return None
    value = "\n".join(run.text for run in sorted(runs, key=lambda item: item.reading_order))
    return value if value.strip() else None


def _table_cells_from_grid(
    *, runs: Sequence[_TextRun], rectangles: Sequence[_Rectangle]
) -> tuple[CandidateTableCellV1, ...]:
    """Map text anchors into exactly one proven rectangle grid.

    A run that lands on more than one grid rectangle is ambiguous.  The
    candidate emits no table rather than letting an inclusive border test pick
    an arbitrary neighboring cell.
    """

    selected = _select_rectangle_grid(rectangles)
    if selected is None:
        return ()
    upper, lower = selected
    grid_rectangles = tuple((*upper.rectangles, *lower.rectangles))
    members_by_ordinal: dict[int, list[_TextRun]] = {
        rectangle.ordinal: [] for rectangle in grid_rectangles
    }
    for run in runs:
        matched = [
            rectangle
            for rectangle in grid_rectangles
            if _point_in_rectangle(x=run.x, y=run.y, rectangle=rectangle)
        ]
        if len(matched) > 1:
            return ()
        if matched:
            members_by_ordinal[matched[0].ordinal].append(run)

    cells: list[CandidateTableCellV1] = []
    for row_index, band in enumerate((upper, lower)):
        for column_index, rectangle in enumerate(band.rectangles):
            text = _cell_text(members_by_ordinal[rectangle.ordinal])
            if text is None:
                return ()
            cells.append(
                CandidateTableCellV1(
                    page_number=1,
                    table_index=1,
                    row_index=row_index,
                    column_index=column_index,
                    text=text,
                )
            )
    return tuple(cells)


def _form_regions_from_rectangles(
    *,
    page_number: int,
    rectangles: Sequence[_Rectangle],
    width: int,
    height: int,
) -> tuple[NativePdfVectorFormRegion, ...]:
    """Expose non-table rectangles without retaining a document layout profile."""

    selected = _select_rectangle_grid(rectangles)
    table_ordinals = {rectangle.ordinal for band in selected or () for rectangle in band.rectangles}
    return tuple(
        NativePdfVectorFormRegion(
            page_number=page_number,
            x=rectangle.x / width,
            y=(height - (rectangle.y + rectangle.height)) / height,
            width=rectangle.width / width,
            height=rectangle.height / height,
        )
        for rectangle in rectangles
        if rectangle.ordinal not in table_ordinals
    )


def extract_native_pdf_vector_grid_from_pdf_bytes(
    *, pdf_bytes: bytes
) -> NativePdfVectorGridExtraction:
    """Read native text/vector evidence from selected PDF bytes only.

    This byte-only seam lets an offline semantic model consume the same
    candidate evidence without receiving request metadata, asset identifiers,
    or a filesystem path.  It remains deliberately independent from Gold,
    CSV, source truth, and template-coordinate inputs.

    The first page may contribute one direct 5/6-column table candidate.  All
    text-show operations remain candidate spans, including on later pages.
    Unsupported constructs produce a content-free failure instead of a
    partial coordinate approximation.
    """

    if not isinstance(pdf_bytes, bytes) or not pdf_bytes:
        raise NativePdfVectorGridError("native_vector_pdf_invalid")
    try:
        from pypdf import PdfReader  # type: ignore[import-not-found]

        reader = PdfReader(io.BytesIO(pdf_bytes), strict=True)
        if reader.is_encrypted:
            raise NativePdfVectorGridError("native_vector_encrypted_pdf")
        pages = tuple(reader.pages)
    except NativePdfVectorGridError:
        raise
    except Exception as error:
        raise NativePdfVectorGridError("native_vector_pdf_invalid") from error
    if not pages or len(pages) > _MAX_PAGES:
        raise NativePdfVectorGridError("native_vector_page_invalid")

    extracted_pages: list[NativePdfVectorGridPage] = []
    all_spans: list[CandidateSpanV1] = []
    first_page_runs: tuple[_TextRun, ...] = ()
    first_page_rectangles: tuple[_Rectangle, ...] = ()
    for page_number, page in enumerate(pages, start=1):
        width, height = _page_dimensions(page)
        runs, rectangles = _parse_page_operations(
            page=page,
            reader=reader,
            page_number=page_number,
            width=width,
            height=height,
        )
        extracted_pages.append(
            NativePdfVectorGridPage(page_number=page_number, width=width, height=height)
        )
        all_spans.extend(
            CandidateSpanV1(
                page_number=run.page_number,
                reading_order=run.reading_order,
                text=run.text,
                anchor_x=run.x / width,
                anchor_y=(height - run.y) / height,
            )
            for run in runs
        )
        if page_number == 1:
            first_page_runs = runs
            first_page_rectangles = rectangles
    return NativePdfVectorGridExtraction(
        pages=tuple(extracted_pages),
        spans=tuple(all_spans),
        table_cells=_table_cells_from_grid(runs=first_page_runs, rectangles=first_page_rectangles),
        form_regions=_form_regions_from_rectangles(
            page_number=1,
            rectangles=first_page_rectangles,
            width=extracted_pages[0].width,
            height=extracted_pages[0].height,
        ),
    )


def extract_native_pdf_vector_grid(
    *, preview_input: PreviewInputV1
) -> NativePdfVectorGridExtraction:
    """Read a PDF preview input's native text/vector evidence.

    The preview boundary retains its media-type no-op behavior.  Parsing is
    delegated to the byte-only implementation so non-preview callers cannot
    accidentally acquire request metadata as an inference dependency.
    """

    if preview_input.metadata.media_type != "application/pdf":
        return NativePdfVectorGridExtraction(pages=(), spans=(), table_cells=(), form_regions=())
    return extract_native_pdf_vector_grid_from_pdf_bytes(pdf_bytes=preview_input.content)


__all__ = [
    "NativePdfVectorGridError",
    "NativePdfVectorGridExtraction",
    "NativePdfVectorFormRegion",
    "NativePdfVectorGridPage",
    "extract_native_pdf_vector_grid",
    "extract_native_pdf_vector_grid_from_pdf_bytes",
]
