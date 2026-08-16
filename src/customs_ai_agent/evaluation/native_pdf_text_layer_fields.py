"""Fail-closed field projection for positioned native-PDF text fragments.

This helper deliberately has a much smaller boundary than the synthetic-PDF
runner.  It accepts only one document type and candidate-engine spans.  In
particular it cannot read a PDF, a manifest, Gold, a CSV, source layout truth,
or another document.  It is intended for an adapter that has already produced
native text-layer fragments with a trustworthy page-relative anchor.

The projector never reconstructs a value from source data.  A value is emitted
only when exactly one visible label has one direct, geometrically adjacent,
non-label visitor successor.  Missing geometry, duplicate labels or direct
successors, and ambiguous neighbourhoods are therefore omitted for the
evaluator to score as missing.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

from customs_ai_agent.evaluation.document_ocr_quality import (
    CandidateFieldV1,
    CandidateSpanV1,
    FieldNormalizer,
    SyntheticDocumentType,
    normalize_field_value,
)


@dataclass(frozen=True, slots=True)
class _NativeVisibleFieldSpec:
    """One visible form or table label that can anchor a candidate value."""

    field_name: str
    normalizer: FieldNormalizer
    labels: tuple[str, ...]


# This is the same candidate-only visible-label vocabulary as the existing
# synthetic-PDF normalizer.  It describes labels rendered in the PDFs, not
# source-schema fields or a coordinate profile.  Keeping it local to this
# native-text helper avoids a dependency on the runner (which would be cyclic
# once the runner calls this helper during integration).
_NATIVE_VISIBLE_LAYOUT_FIELDS: dict[SyntheticDocumentType, tuple[_NativeVisibleFieldSpec, ...]] = {
    "bill_of_lading": (
        _NativeVisibleFieldSpec("exporter_name", "text-v1", ("shipper", "exporter")),
        _NativeVisibleFieldSpec("consignee", "text-v1", ("consignee", "to applicant / consignee")),
        _NativeVisibleFieldSpec("load_port", "text-v1", ("port of loading", "load port")),
        _NativeVisibleFieldSpec("disch_port", "text-v1", ("port of discharge", "discharge port")),
        _NativeVisibleFieldSpec("final_dest", "text-v1", ("final destination",)),
        _NativeVisibleFieldSpec(
            "vessel", "text-v1", ("vessel / flight", "vessel/flight", "vessel")
        ),
        _NativeVisibleFieldSpec("voyage", "identifier-v1", ("voyage no", "voyage")),
        _NativeVisibleFieldSpec("bl_no", "identifier-v1", ("bill of lading no", "b/l no", "bl no")),
        _NativeVisibleFieldSpec(
            "container_no", "identifier-v1", ("container no", "container number")
        ),
        _NativeVisibleFieldSpec("seal_no", "identifier-v1", ("seal no", "seal number")),
        _NativeVisibleFieldSpec("good", "text-v1", ("goods description", "description of goods")),
        _NativeVisibleFieldSpec("gross_weight", "numeric-v1", ("gross weight",)),
        _NativeVisibleFieldSpec(
            "containers", "numeric-v1", ("no of containers", "number of containers", "containers")
        ),
    ),
    "invoice": (
        _NativeVisibleFieldSpec("exporter_name", "text-v1", ("exporter", "seller", "shipper")),
        _NativeVisibleFieldSpec("importer_name", "text-v1", ("notify party", "importer", "buyer")),
        _NativeVisibleFieldSpec("consignee", "text-v1", ("consignee", "to applicant / consignee")),
        _NativeVisibleFieldSpec("load_port", "text-v1", ("port of loading", "load port")),
        _NativeVisibleFieldSpec("disch_port", "text-v1", ("port of discharge", "discharge port")),
        _NativeVisibleFieldSpec(
            "vessel", "text-v1", ("vessel / flight", "vessel/flight", "vessel")
        ),
        _NativeVisibleFieldSpec(
            "inv_no", "identifier-v1", ("invoice number", "invoice no", "inv no")
        ),
        _NativeVisibleFieldSpec("inv_date", "identifier-v1", ("invoice date",)),
        _NativeVisibleFieldSpec("lc_no", "identifier-v1", ("letter of credit", "l/c no", "lc no")),
        _NativeVisibleFieldSpec("good", "text-v1", ("goods description", "description of goods")),
        _NativeVisibleFieldSpec("unit", "text-v1", ("unit",)),
        _NativeVisibleFieldSpec("qty", "numeric-v1", ("quantity", "qty")),
        _NativeVisibleFieldSpec("unit_price", "numeric-v1", ("unit price",)),
        _NativeVisibleFieldSpec("amount", "numeric-v1", ("amount",)),
        _NativeVisibleFieldSpec("total", "numeric-v1", ("total",)),
    ),
    "packing_list": (
        _NativeVisibleFieldSpec("exporter_name", "text-v1", ("exporter", "seller", "shipper")),
        _NativeVisibleFieldSpec("consignee", "text-v1", ("consignee", "to applicant / consignee")),
        _NativeVisibleFieldSpec(
            "inv_no", "identifier-v1", ("no of invoice", "invoice no", "inv no")
        ),
        _NativeVisibleFieldSpec(
            "lc_no", "identifier-v1", ("no of l/c", "letter of credit", "l/c no", "lc no")
        ),
        _NativeVisibleFieldSpec("good", "text-v1", ("goods description", "description of goods")),
        _NativeVisibleFieldSpec("unit", "text-v1", ("unit",)),
        _NativeVisibleFieldSpec("qty", "numeric-v1", ("quantity", "qty")),
        _NativeVisibleFieldSpec(
            "cartons", "numeric-v1", ("no of cartons / pkgs", "cartons", "pkgs")
        ),
        _NativeVisibleFieldSpec("net_weight", "numeric-v1", ("net weight",)),
        _NativeVisibleFieldSpec("gross_weight", "numeric-v1", ("gross weight",)),
        _NativeVisibleFieldSpec("cbm", "numeric-v1", ("cbm", "volume", "measurement")),
        _NativeVisibleFieldSpec("net_gross_weight", "text-v1", ("net/gross weight",)),
    ),
}


_QUANTITY_TOKEN = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_QUANTITY_WITH_UNIT = re.compile(rf"^\s*(?P<quantity>{_QUANTITY_TOKEN})\s+(?P<unit>[A-Za-z]+)\s*$")
_QUANTITY_ONLY = re.compile(rf"^\s*(?P<quantity>{_QUANTITY_TOKEN})\s*$")
_UNIT_ONLY = re.compile(r"^\s*(?P<unit>[A-Za-z]+)\s*$")

# All positions are normalized to one page.  These are broad enough for a
# label/value pair rendered in the same form cell, but intentionally much
# narrower than a page quadrant.  A unique geometrically bounded candidate is
# required, rather than a nearest-neighbour search over the page.
_SAME_ROW_Y_DELTA = 0.020
_MAX_RIGHT_GAP = 0.550
_MAX_BELOW_GAP = 0.100
_BELOW_X_DELTA = 0.100
_POSITION_EPSILON = 0.000_001


@dataclass(frozen=True, slots=True)
class _NativeAnchor:
    """A proven normalized start point, optionally with a text rectangle."""

    x: float
    y: float
    width: float
    height: float

    @property
    def right(self) -> float:
        return self.x + self.width

    @property
    def bottom(self) -> float:
        return self.y + self.height

    @property
    def center_y(self) -> float:
        return self.y + (self.height / 2)


@dataclass(frozen=True, slots=True)
class _LocatedCandidateSpan:
    span: CandidateSpanV1
    text: str
    anchor: _NativeAnchor


def _normalized_visible_label(value: str) -> str:
    """Normalize label typography only; emitted values retain their spelling."""

    normalized = re.sub(r"[^a-z0-9/]+", " ", value.casefold())
    normalized = re.sub(r"\s*/\s*", "/", normalized)
    return " ".join(normalized.split())


def _visible_value(value: str) -> str | None:
    compact = " ".join(value.replace("\r\n", "\n").replace("\r", "\n").split())
    return compact or None


def _finite_unit_coordinate(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    numeric = float(value)
    if not math.isfinite(numeric) or not 0 <= numeric <= 1:
        return None
    return numeric


def _anchor_from_span(span: CandidateSpanV1) -> _NativeAnchor | None:
    """Read only a candidate span's bbox or optional internal anchor attrs.

    ``CandidateSpanV1`` normally carries ``bbox``.  The guarded ``anchor_x`` /
    ``anchor_y`` fallback permits an internal visitor-fragment subclass to
    retain a proven start point until an exact public rectangle is available.
    It never accepts an arbitrary coordinate outside the normalized page.
    """

    if span.bbox is not None:
        values = (span.bbox.x, span.bbox.y, span.bbox.width, span.bbox.height)
        if all(math.isfinite(float(value)) for value in values):
            return _NativeAnchor(
                x=float(span.bbox.x),
                y=float(span.bbox.y),
                width=float(span.bbox.width),
                height=float(span.bbox.height),
            )
        return None
    x = _finite_unit_coordinate(getattr(span, "anchor_x", None))
    y = _finite_unit_coordinate(getattr(span, "anchor_y", None))
    if x is None or y is None:
        return None
    return _NativeAnchor(x=x, y=y, width=0.0, height=0.0)


def _located_spans(spans: Sequence[CandidateSpanV1]) -> tuple[_LocatedCandidateSpan, ...]:
    located: list[_LocatedCandidateSpan] = []
    for span in spans:
        text = _visible_value(span.text)
        anchor = _anchor_from_span(span)
        if text is not None and anchor is not None:
            located.append(_LocatedCandidateSpan(span=span, text=text, anchor=anchor))
    return tuple(
        sorted(
            located,
            key=lambda item: (
                item.span.page_number,
                item.span.reading_order,
                item.anchor.y,
                item.anchor.x,
            ),
        )
    )


def _field_specification(
    document_type: SyntheticDocumentType,
) -> tuple[_NativeVisibleFieldSpec, ...]:
    return _NATIVE_VISIBLE_LAYOUT_FIELDS[document_type]


def _matching_specs(
    *, document_type: SyntheticDocumentType, label: str
) -> tuple[_NativeVisibleFieldSpec, ...]:
    normalized = _normalized_visible_label(label)
    if not normalized:
        return ()
    return tuple(
        specification
        for specification in _field_specification(document_type)
        if normalized in {_normalized_visible_label(alias) for alias in specification.labels}
    )


def _is_geometrically_adjacent_label_value(
    *, label: _LocatedCandidateSpan, value: _LocatedCandidateSpan
) -> bool:
    """Return true only for one same-row-right or same-column-below relation."""

    if (
        label.span.page_number != value.span.page_number
        or value.span.reading_order <= label.span.reading_order
    ):
        return False
    same_row = (
        value.anchor.x + _POSITION_EPSILON >= label.anchor.right
        and value.anchor.x - label.anchor.right <= _MAX_RIGHT_GAP
        and abs(value.anchor.center_y - label.anchor.center_y) <= _SAME_ROW_Y_DELTA
    )
    below = (
        value.anchor.y + _POSITION_EPSILON >= label.anchor.bottom
        and value.anchor.y - label.anchor.bottom <= _MAX_BELOW_GAP
        and abs(value.anchor.x - label.anchor.x) <= _BELOW_X_DELTA
    )
    return same_row or below


def _single_geometric_value(
    *,
    document_type: SyntheticDocumentType,
    label: _LocatedCandidateSpan,
    located: Sequence[_LocatedCandidateSpan],
) -> str | None:
    successors = [
        candidate
        for candidate in located
        if candidate.span.page_number == label.span.page_number
        and candidate.span.reading_order > label.span.reading_order
    ]
    if not successors:
        return None
    candidates = [
        candidate
        for candidate in successors
        if not _matching_specs(document_type=document_type, label=candidate.text)
        and _is_geometrically_adjacent_label_value(label=label, value=candidate)
    ]
    # Visitor callback order can contain an unrelated fragment between a
    # visible label and its nearby value.  Geometry may safely skip that
    # distant fragment only when it leaves exactly one local non-label
    # candidate.  Two candidates, including a duplicate callback at the same
    # anchor, are ambiguous and deliberately omitted.
    if len(candidates) != 1:
        return None
    return candidates[0].text


def _append_value_for_field(
    *, candidates: dict[str, list[str]], field_name: str, value: str
) -> None:
    """Preserve one emitted candidate value; only exact quantity forms split."""

    if field_name == "qty":
        quantity_with_unit = _QUANTITY_WITH_UNIT.fullmatch(value)
        if quantity_with_unit is not None:
            candidates.setdefault("qty", []).append(quantity_with_unit.group("quantity"))
            candidates.setdefault("unit", []).append(quantity_with_unit.group("unit"))
            return
        quantity_only = _QUANTITY_ONLY.fullmatch(value)
        if quantity_only is not None:
            candidates.setdefault("qty", []).append(quantity_only.group("quantity"))
        return
    if field_name == "unit":
        unit_only = _UNIT_ONLY.fullmatch(value)
        if unit_only is not None:
            candidates.setdefault("unit", []).append(unit_only.group("unit"))
        return
    candidates.setdefault(field_name, []).append(value)


def project_native_pdf_text_layer_fields(
    *, document_type: SyntheticDocumentType, spans: Sequence[CandidateSpanV1]
) -> list[CandidateFieldV1]:
    """Project unambiguous native-PDF label/value pairs into candidate fields.

    Only positioned candidate spans are consulted.  Each field requires one
    and only one exact visible label and one uniquely bounded, nearby non-label
    candidate.  Consequently this function returns an empty or partial list
    when the adapter does not provide reliable geometry; that is deliberate
    fail-closed behaviour, not a fallback to a nearest page-text search.
    """

    specification = _field_specification(document_type)
    located = _located_spans(spans)
    candidates: dict[str, list[str]] = {}
    for field in specification:
        labels = [
            item
            for item in located
            if _matching_specs(document_type=document_type, label=item.text) == (field,)
        ]
        if len(labels) != 1:
            continue
        value = _single_geometric_value(
            document_type=document_type,
            label=labels[0],
            located=located,
        )
        if value is not None:
            _append_value_for_field(
                candidates=candidates,
                field_name=field.field_name,
                value=value,
            )

    fields: list[CandidateFieldV1] = []
    for field in specification:
        values = candidates.get(field.field_name, ())
        # Duplicates are intentionally an error of evidence, even when their
        # canonical spellings agree.  Selecting one would hide an ambiguous
        # layout instead of allowing the independent gate to reject it.
        if len(values) != 1:
            continue
        canonical = normalize_field_value(values[0], field.normalizer)
        if not canonical:
            continue
        fields.append(CandidateFieldV1(field_name=field.field_name, value=values[0]))
    return fields


__all__ = ["project_native_pdf_text_layer_fields"]
