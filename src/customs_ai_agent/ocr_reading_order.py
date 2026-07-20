"""Bounded preservation of an OCR provider's explicit reading order.

PaddleOCR and Docling already return their observations in an engine-defined
semantic sequence.  Reconstructing that sequence from geometry is lossy:
multi-column forms, field grids, and tables can share the same y coordinate.
The worker therefore owns conversion from the native provider result and the
ordered worker list is the wire-level reading-order contract.  The Agent edge
adapter validates and preserves that order without inventing text or layout.

Structured table converters emit ``table -> row -> cell`` in their native
position.  If an engine does not return table structure, this module must not
infer cells from ordinary OCR lines; downstream policy keeps that document in
human review.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final, TypeVar

_SpanT = TypeVar("_SpanT")
OCR_READING_ORDER_POLICY_ID: Final = "worker-emitted-sequence/v1"

# The worker response is capped separately at 8 MiB.  This count cap also
# bounds normalization CPU and memory when an adversarial page contains many
# tiny boxes.
MAX_OCR_SPANS_PER_PAGE: Final = 4_096


class OcrSpanLimitExceededError(ValueError):
    """Raised before normalization when one page exceeds the safe span bound."""


def preserve_provider_span_order(spans: Sequence[_SpanT]) -> list[_SpanT]:
    """Return the provider sequence unchanged after enforcing the page cap."""

    if len(spans) > MAX_OCR_SPANS_PER_PAGE:
        raise OcrSpanLimitExceededError("ocr_span_limit_exceeded")
    return list(spans)
