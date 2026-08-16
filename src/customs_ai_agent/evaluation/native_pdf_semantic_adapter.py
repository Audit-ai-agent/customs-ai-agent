"""Pure offline inference for learned native-PDF template selectors.

``NativePdfSemanticAdapter.extract_pdf_bytes`` intentionally accepts only the
selected PDF byte sequence.  Its model is an in-memory hash-bound artifact.
The adapter never receives request metadata, asset IDs, Gold/CSV/source truth,
a source-parser profile, a filesystem path, or a network client.  It returns
native vector spans/table cells unchanged and adds conservative direct fields.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence

from customs_ai_agent.evaluation.document_ocr_quality import (
    CandidateFieldV1,
    CandidateSpanV1,
    CandidateTableCellV1,
    normalize_field_value,
)
from customs_ai_agent.evaluation.native_pdf_semantic_model import (
    HashBoundNativePdfSemanticModelV1,
    NativePdfSemanticDocumentModelV1,
    NativePdfSemanticFieldModelV1,
    NativePdfSemanticModelError,
    NativePdfSemanticSelectorV1,
)
from customs_ai_agent.evaluation.native_pdf_vector_grid import (
    NativePdfVectorFormRegion,
    NativePdfVectorGridError,
    NativePdfVectorGridPage,
    extract_native_pdf_vector_grid_from_pdf_bytes,
)
from customs_ai_agent.service.document_preview_runtime import local_files_only


# These bounded patterns intentionally mirror the immutable field-transform
# contract used when source-PDF training labels are compiled.  They live here
# (rather than importing the training/source parser) so byte-only runtime
# inference remains independent from source truth and training-only code.
_NUMERIC_TOKEN = re.compile(r"(?<![A-Za-z0-9])(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?![A-Za-z0-9])")
_ALPHA_TOKEN = re.compile(r"[A-Za-z]+")


class NativePdfSemanticAdapterError(ValueError):
    """Content-free inference error that cannot expose selected PDF text."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class NativePdfSemanticAdapterOutputV1:
    """Original native vector evidence plus conservative direct field output."""

    document_type: str | None
    model_id: str
    model_revision: str
    model_artifact_sha256: str
    fields: tuple[CandidateFieldV1, ...]
    spans: tuple[CandidateSpanV1, ...]
    table_cells: tuple[CandidateTableCellV1, ...]
    form_regions: tuple[NativePdfVectorFormRegion, ...]
    # Page geometry is internal preview support only.  Keep a default so the
    # byte-only candidate adapter's existing output construction contract
    # remains source-compatible for injected test adapters.
    pages: tuple[NativePdfVectorGridPage, ...] = ()


@dataclass(frozen=True, slots=True)
class _SelectorMatch:
    source_kind: str
    source_index: int
    text: str


def _single_token(*, text: str, pattern: re.Pattern[str]) -> str | None:
    """Return exactly one bounded token, never a best-effort substring.

    The semantic model's token transforms select one native text run and then
    conservatively emit its sole numeric or alphabetic token.  A zero- or
    multi-token run is ambiguous and therefore omitted.
    """

    tokens = pattern.findall(text)
    if len(tokens) != 1 or not tokens[0].strip():
        return None
    return tokens[0].strip()


def _resolve_selector(
    *,
    selector: NativePdfSemanticSelectorV1,
    spans: Sequence[CandidateSpanV1],
    table_cells: Sequence[CandidateTableCellV1],
) -> _SelectorMatch | None:
    """Resolve one selector to exactly one direct candidate-native value."""

    if selector.source_kind == "span_reading_order":
        matches = [
            (index, span.text)
            for index, span in enumerate(spans)
            if span.page_number == selector.page_number
            and span.reading_order == selector.reading_order
        ]
    elif selector.source_kind == "span_anchor":
        # Schema validation proves these fields are present for this source
        # kind.  The explicit assertions keep malformed in-memory subclasses
        # fail-closed instead of coercing an absent coordinate.
        if (
            selector.anchor_x is None
            or selector.anchor_y is None
            or selector.anchor_tolerance is None
        ):
            return None
        matches = [
            (index, span.text)
            for index, span in enumerate(spans)
            if span.page_number == selector.page_number
            and span.anchor_x is not None
            and span.anchor_y is not None
            and math.isfinite(span.anchor_x)
            and math.isfinite(span.anchor_y)
            and abs(span.anchor_x - selector.anchor_x) <= selector.anchor_tolerance
            and abs(span.anchor_y - selector.anchor_y) <= selector.anchor_tolerance
        ]
    else:
        if (
            selector.table_index is None
            or selector.row_index is None
            or selector.column_index is None
        ):
            return None
        matches = [
            (index, cell.text)
            for index, cell in enumerate(table_cells)
            if cell.page_number == selector.page_number
            and cell.table_index == selector.table_index
            and cell.row_index == selector.row_index
            and cell.column_index == selector.column_index
        ]
    if len(matches) != 1 or not matches[0][1].strip():
        return None
    source_kind = "span" if selector.source_kind.startswith("span_") else "table_cell"
    return _SelectorMatch(source_kind=source_kind, source_index=matches[0][0], text=matches[0][1])


def _structural_document_model(
    *,
    documents: Sequence[NativePdfSemanticDocumentModelV1],
    page_count: int,
    spans: Sequence[CandidateSpanV1],
    table_cells: Sequence[CandidateTableCellV1],
) -> NativePdfSemanticDocumentModelV1 | None:
    """Choose exactly one profile from shape and selector availability only."""

    matching: list[NativePdfSemanticDocumentModelV1] = []
    for document in documents:
        signature = document.structural_signature
        if (
            signature.page_count != page_count
            or len(spans) < signature.minimum_span_count
            or len(table_cells) < signature.minimum_table_cell_count
        ):
            continue
        if all(
            _resolve_selector(selector=selector, spans=spans, table_cells=table_cells) is not None
            for selector in signature.required_selectors
        ):
            matching.append(document)
    return matching[0] if len(matching) == 1 else None


def _transformed_field_value(
    *,
    field: NativePdfSemanticFieldModelV1,
    spans: Sequence[CandidateSpanV1],
    table_cells: Sequence[CandidateTableCellV1],
) -> str | None:
    values: list[str] = []
    source_keys: set[tuple[str, int]] = set()
    for selector in field.selectors:
        selected = _resolve_selector(selector=selector, spans=spans, table_cells=table_cells)
        if selected is None:
            return None
        source_key = (selected.source_kind, selected.source_index)
        if source_key in source_keys:
            return None
        source_keys.add(source_key)
        values.append(selected.text)
    result: str | None
    if field.value_transform == "raw-v1":
        result = values[0]
    elif field.value_transform == "single-numeric-token-v1":
        token = _single_token(text=values[0], pattern=_NUMERIC_TOKEN)
        if token is None:
            return None
        result = token
    elif field.value_transform == "single-alpha-token-v1":
        token = _single_token(text=values[0], pattern=_ALPHA_TOKEN)
        if token is None:
            return None
        result = token
    else:
        result = " / ".join(value.strip() for value in values)
        if not result or any(not value.strip() for value in values):
            return None
    if not normalize_field_value(result, field.normalizer):
        return None
    return result


def _direct_template_fields(
    *,
    document: NativePdfSemanticDocumentModelV1,
    spans: Sequence[CandidateSpanV1],
    table_cells: Sequence[CandidateTableCellV1],
) -> tuple[CandidateFieldV1, ...]:
    """Emit fields only where every learned selector has one matching source."""

    fields: list[CandidateFieldV1] = []
    for field in document.fields:
        value = _transformed_field_value(field=field, spans=spans, table_cells=table_cells)
        if value is not None:
            fields.append(CandidateFieldV1(field_name=field.field_name, value=value))
    return tuple(sorted(fields, key=lambda field: field.field_name))


@dataclass(frozen=True, slots=True)
class NativePdfSemanticAdapter:
    """Candidate-only selector runtime with no path or request-metadata input."""

    model: HashBoundNativePdfSemanticModelV1

    def extract_pdf_bytes(self, pdf_bytes: bytes) -> NativePdfSemanticAdapterOutputV1:
        """Extract selected PDF bytes with the immutable bound selector model."""

        if not isinstance(pdf_bytes, bytes):
            raise NativePdfSemanticAdapterError("native_pdf_semantic_pdf_invalid")
        try:
            artifact = self.model.verified_artifact()
            # The parser consumes ``BytesIO`` and the model is already in
            # memory.  This guard rejects any future parser network path.
            with local_files_only():
                extraction = extract_native_pdf_vector_grid_from_pdf_bytes(pdf_bytes=pdf_bytes)
            document = _structural_document_model(
                documents=artifact.documents,
                page_count=len(extraction.pages),
                spans=extraction.spans,
                table_cells=extraction.table_cells,
            )
        except (NativePdfVectorGridError, NativePdfSemanticModelError) as error:
            raise NativePdfSemanticAdapterError("native_pdf_semantic_execution_invalid") from error
        except Exception as error:
            raise NativePdfSemanticAdapterError("native_pdf_semantic_execution_invalid") from error
        fields = (
            _direct_template_fields(
                document=document,
                spans=extraction.spans,
                table_cells=extraction.table_cells,
            )
            if document is not None
            else ()
        )
        return NativePdfSemanticAdapterOutputV1(
            document_type=document.document_type if document is not None else None,
            model_id=artifact.model_id,
            model_revision=artifact.model_revision,
            model_artifact_sha256=self.model.artifact_sha256,
            fields=fields,
            spans=extraction.spans,
            table_cells=extraction.table_cells,
            form_regions=extraction.form_regions,
            pages=extraction.pages,
        )


__all__ = [
    "NativePdfSemanticAdapter",
    "NativePdfSemanticAdapterError",
    "NativePdfSemanticAdapterOutputV1",
]
