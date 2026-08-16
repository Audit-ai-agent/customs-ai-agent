"""Fail-closed form-field projection from candidate text anchors and rectangles.

This helper is intentionally generic: it accepts only candidate spans,
candidate-visible label specifications, and normalized rectangle geometry.
It has no source-truth, Gold, CSV, manifest, template-coordinate, or fixed
text-run-index input.  A field is emitted only when one visible label has one
unambiguous candidate value block inside one non-table rectangle.
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
    normalize_field_value,
)
from customs_ai_agent.evaluation.native_pdf_vector_grid import NativePdfVectorFormRegion


_LABEL_WORDS = re.compile(r"[^a-z0-9/]+")
_POSITION_EPSILON = 0.000_001


@dataclass(frozen=True, slots=True)
class NativePdfVectorVisibleFieldSpec:
    """One candidate-visible field label, supplied by the caller's vocabulary."""

    field_name: str
    normalizer: FieldNormalizer
    labels: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _LocatedSpan:
    page_number: int
    x: float
    y: float
    text: str


def _normalized_label(value: str) -> str:
    normalized = _LABEL_WORDS.sub(" ", value.casefold())
    normalized = re.sub(r"\s*/\s*", "/", normalized)
    return " ".join(normalized.split())


def _normalized_aliases(specification: NativePdfVectorVisibleFieldSpec) -> frozenset[str]:
    return frozenset(
        normalized for label in specification.labels if (normalized := _normalized_label(label))
    )


def _valid_region(region: NativePdfVectorFormRegion) -> bool:
    values = (region.x, region.y, region.width, region.height)
    return (
        isinstance(region.page_number, int)
        and region.page_number >= 1
        and all(
            not isinstance(value, bool)
            and isinstance(value, (int, float))
            and math.isfinite(float(value))
            for value in values
        )
        and region.x >= 0
        and region.y >= 0
        and region.width > 0
        and region.height > 0
        and region.x + region.width <= 1
        and region.y + region.height <= 1
    )


def _located_spans(spans: Sequence[CandidateSpanV1]) -> tuple[_LocatedSpan, ...]:
    result: list[_LocatedSpan] = []
    for span in spans:
        if span.anchor_x is None or span.anchor_y is None:
            continue
        if not span.text.strip():
            continue
        x, y = span.anchor_x, span.anchor_y
        if (
            isinstance(x, bool)
            or isinstance(y, bool)
            or not isinstance(x, (int, float))
            or not isinstance(y, (int, float))
            or not math.isfinite(float(x))
            or not math.isfinite(float(y))
            or not 0 <= float(x) <= 1
            or not 0 <= float(y) <= 1
        ):
            continue
        result.append(
            _LocatedSpan(
                page_number=span.page_number,
                x=float(x),
                y=float(y),
                text=span.text,
            )
        )
    # Geometric order is deliberate.  The projector never selects a fixed
    # text-run ordinal from a known fixture profile.
    return tuple(sorted(result, key=lambda item: (item.page_number, item.y, item.x, item.text)))


def _contains(*, region: NativePdfVectorFormRegion, span: _LocatedSpan) -> bool:
    return (
        region.page_number == span.page_number
        and region.x - _POSITION_EPSILON <= span.x <= region.x + region.width + _POSITION_EPSILON
        and region.y - _POSITION_EPSILON <= span.y <= region.y + region.height + _POSITION_EPSILON
    )


def _matching_field_names(
    *, specifications: Sequence[NativePdfVectorVisibleFieldSpec], text: str
) -> tuple[str, ...]:
    normalized = _normalized_label(text)
    if not normalized:
        return ()
    return tuple(
        specification.field_name
        for specification in specifications
        if normalized in _normalized_aliases(specification)
    )


def _same_row(
    *, label: _LocatedSpan, value: _LocatedSpan, region: NativePdfVectorFormRegion
) -> bool:
    return value.x > label.x + _POSITION_EPSILON and abs(value.y - label.y) <= max(
        region.height * 0.20, _POSITION_EPSILON
    )


def _below(*, label: _LocatedSpan, value: _LocatedSpan, region: NativePdfVectorFormRegion) -> bool:
    return (
        value.y > label.y + max(region.height * 0.05, _POSITION_EPSILON)
        and abs(value.x - label.x) <= region.width * 0.45
    )


def _unambiguous_value_block(
    *,
    label: _LocatedSpan,
    region: NativePdfVectorFormRegion,
    values: Sequence[_LocatedSpan],
) -> str | None:
    """Choose one directional non-label block, otherwise omit the field."""

    if not values:
        return None
    same_row = [value for value in values if _same_row(label=label, value=value, region=region)]
    below = [value for value in values if _below(label=label, value=value, region=region)]
    if same_row and below:
        return None
    chosen = same_row or below
    # Text elsewhere in the rectangle could be another unlabeled field or a
    # title.  Do not silently discard it to make a label/value pair fit.
    if not chosen or len(chosen) != len(values):
        return None
    value = "\n".join(item.text for item in chosen)
    return value if value.strip() else None


def project_native_pdf_vector_form_fields(
    *,
    spans: Sequence[CandidateSpanV1],
    regions: Sequence[NativePdfVectorFormRegion],
    specifications: Sequence[NativePdfVectorVisibleFieldSpec],
) -> list[CandidateFieldV1]:
    """Emit only rectangle-contained, visible-label field values.

    Regions must be non-overlapping for any participating label.  Nested or
    overlapping rectangles are common in arbitrary PDFs and do not prove which
    visual box owns a value, so affected fields deliberately remain absent.
    """

    if (
        not specifications
        or any(
            not specification.field_name
            or not specification.labels
            or not _normalized_aliases(specification)
            for specification in specifications
        )
        or any(not _valid_region(region) for region in regions)
    ):
        return []
    located = _located_spans(spans)
    memberships: dict[_LocatedSpan, tuple[int, ...]] = {
        span: tuple(
            index for index, region in enumerate(regions) if _contains(region=region, span=span)
        )
        for span in located
    }
    fields: list[CandidateFieldV1] = []
    for specification in specifications:
        labels = [
            span
            for span in located
            if _matching_field_names(specifications=specifications, text=span.text)
            == (specification.field_name,)
            and len(memberships[span]) == 1
        ]
        if len(labels) != 1:
            continue
        label = labels[0]
        region_index = memberships[label][0]
        region = regions[region_index]
        region_spans = [span for span in located if memberships[span] == (region_index,)]
        values = [
            span
            for span in region_spans
            if span != label
            and not _matching_field_names(specifications=specifications, text=span.text)
        ]
        raw_value = _unambiguous_value_block(label=label, region=region, values=values)
        if raw_value is None or not normalize_field_value(raw_value, specification.normalizer):
            continue
        fields.append(CandidateFieldV1(field_name=specification.field_name, value=raw_value))
    return fields


__all__ = [
    "NativePdfVectorVisibleFieldSpec",
    "project_native_pdf_vector_form_fields",
]
