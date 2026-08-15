"""Local-only source-attested layout truth for the synthetic PDF fixture.

This module deliberately does *not* turn the older human-review Gold into an
automatic approval.  The synthetic PDF generator is not available in this
workspace, so the final, born-digital PDF content stream is the only visual
source that can be parsed without guessing.  A small, pinned parser accepts
only the one observed fixture profile.  Unsupported syntax, a changed template,
or an ambiguous table fails closed.

The resulting evaluation is named ``synthetic_content_stream_compatibility``.
It is useful for comparing local extractors on this exact synthetic template,
but it is neither an independent optical-OCR certification nor evidence about
real trade documents.
"""

from __future__ import annotations

import base64
import re
import stat
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from customs_ai_agent.evaluation.document_gold import (
    DatasetAssetReceiptV1,
    DatasetReceiptV1,
)
from customs_ai_agent.evaluation.document_ocr_quality import (
    CandidateTableCellV1,
    EngineKind,
    EvaluationPacketV2,
    GoldFieldV2,
    GoldSpanV2,
    GoldTableCellV2,
    StratifiedPacketItemV2,
    SyntheticDocumentType,
    SyntheticPdfCandidateOutputBundleV1,
    build_synthetic_pdf_draft_gold,
    candidate_bundle_sha256,
    canonical_json_bytes,
    canonical_json_file_bytes,
    canonical_span_text,
    candidate_output_from_synthetic_bundle,
    normalize_field_value,
    normalize_table_cell_text,
    packet_receipt_sha256,
    load_synthetic_error_labels,
    receipt_sha256,
    sha256_bytes,
    sha256_file,
    validate_synthetic_candidate_output_provenance,
    verify_receipt_source_assets,
)


_IDENTIFIER = r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$"
_SHA256 = r"^[a-f0-9]{64}$"
_MAX_PDF_BYTES = 8 * 1024 * 1024
_PDF_CONTENT_STREAM_RE = re.compile(
    rb"(?<!end)stream\r?\n?(?P<content>.*?)\r?\n?endstream", re.DOTALL
)
_MEDIA_BOX_RE = re.compile(
    rb"/MediaBox\s*\[\s*(?P<x0>-?\d+(?:\.\d+)?)\s+"
    rb"(?P<y0>-?\d+(?:\.\d+)?)\s+(?P<x1>-?\d+(?:\.\d+)?)\s+"
    rb"(?P<y1>-?\d+(?:\.\d+)?)\s*\]"
)
_ERROR_SET_ID_RE = re.compile(r"^E(?P<set_number>\d{4})$")
_DOCUMENT_FILENAME_SUFFIX: dict[SyntheticDocumentType, str] = {
    "bill_of_lading": "BL",
    "invoice": "INV",
    "packing_list": "PL",
}

SOURCE_ATTESTED_EVALUATION_BASIS: Final[Literal["source_attested_synthetic"]] = (
    "source_attested_synthetic"
)
SOURCE_ATTESTED_EVALUATION_SCOPE: Final[Literal["synthetic_content_stream_compatibility"]] = (
    "synthetic_content_stream_compatibility"
)
SOURCE_LAYOUT_TRUTH_KIND: Final[Literal["pdf_content_stream_layout_v1"]] = (
    "pdf_content_stream_layout_v1"
)
SOURCE_LAYOUT_COMPILER_REVISION: Final[
    Literal["synthetic-pdf-content-stream-layout-compiler-v1"]
] = "synthetic-pdf-content-stream-layout-compiler-v1"
SOURCE_LAYOUT_PARSER_SPEC_REVISION: Final[
    Literal["synthetic-pdf-content-stream-parser-spec-v1"]
] = "synthetic-pdf-content-stream-parser-spec-v1"


class SourceAttestedSyntheticPdfTruthError(ValueError):
    """A content-free error boundary for local source-truth processing."""


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


@dataclass(frozen=True, slots=True)
class _TextRun:
    reading_order: int
    x: float
    y: float
    text: str


@dataclass(frozen=True, slots=True)
class _Rectangle:
    x: float
    y: float
    width: float
    height: float


@dataclass(frozen=True, slots=True)
class _ParsedContentStream:
    decoded_sha256: str
    text_runs: tuple[_TextRun, ...]
    rectangles: tuple[_Rectangle, ...]
    media_box: tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class _FieldSlot:
    field_name: str
    normalizer: Literal["text-v1", "numeric-v1", "identifier-v1"]
    reading_order: int
    x: float | tuple[float, ...]
    y: float


@dataclass(frozen=True, slots=True)
class _TableSlot:
    header_rectangle_indices: tuple[int, ...]
    body_rectangle_indices: tuple[int, ...]
    header_rectangle_geometry: tuple[tuple[float, float, float, float], ...]
    body_rectangle_geometry: tuple[tuple[float, float, float, float], ...]


@dataclass(frozen=True, slots=True)
class _TemplateProfile:
    text_run_count: int
    rectangle_count: int
    field_slots: tuple[_FieldSlot, ...]
    table: _TableSlot


# These are positions and field identifiers, not copied document text.  The
# positions were profiled from the final fixture's stable content-stream shape.
# Each input is required to retain that exact shape; an altered template is not
# silently interpreted as this version.
_TEMPLATE_PROFILES: dict[SyntheticDocumentType, _TemplateProfile] = {
    "bill_of_lading": _TemplateProfile(
        text_run_count=90,
        rectangle_count=42,
        field_slots=(
            _FieldSlot("exporter_name", "text-v1", 4, 44.0, 755.8898),
            _FieldSlot("consignee", "text-v1", 11, 44.0, 709.8898),
            _FieldSlot("bl_no", "identifier-v1", 8, 324.0, 755.8898),
            _FieldSlot("vessel", "text-v1", 22, 44.0, 612.8898),
            _FieldSlot("voyage", "identifier-v1", 25, 184.0, 612.8898),
            _FieldSlot("load_port", "text-v1", 31, 44.0, 585.8898),
            _FieldSlot("disch_port", "text-v1", 34, 172.8189, 585.8898),
            _FieldSlot("final_dest", "text-v1", 40, 430.4567, 585.8898),
            _FieldSlot("container_no", "identifier-v1", 54, 43.0, 537.8898),
            _FieldSlot("seal_no", "identifier-v1", 55, 121.0, 537.8898),
            _FieldSlot("good", "text-v1", 61, 278.0, 537.8898),
        ),
        table=_TableSlot(
            tuple(range(16, 22)),
            tuple(range(22, 28)),
            (
                (40.0, 549.8898, 78.0, 20.0),
                (118.0, 549.8898, 72.0, 20.0),
                (190.0, 549.8898, 85.0, 20.0),
                (275.0, 549.8898, 125.0, 20.0),
                (400.0, 549.8898, 85.0, 20.0),
                (485.0, 549.8898, 70.27559, 20.0),
            ),
            (
                (40.0, 453.8898, 78.0, 96.0),
                (118.0, 453.8898, 72.0, 96.0),
                (190.0, 453.8898, 85.0, 96.0),
                (275.0, 453.8898, 125.0, 96.0),
                (400.0, 453.8898, 85.0, 96.0),
                (485.0, 453.8898, 70.27559, 96.0),
            ),
        ),
    ),
    "invoice": _TemplateProfile(
        text_run_count=41,
        rectangle_count=23,
        field_slots=(
            _FieldSlot("exporter_name", "text-v1", 3, 49.0, 743.8898),
            _FieldSlot("consignee", "text-v1", 9, 49.0, 673.8898),
            _FieldSlot("inv_no", "identifier-v1", 6, 301.6378, 743.8898),
            _FieldSlot("inv_date", "identifier-v1", 7, 301.6378, 734.3898),
            _FieldSlot("lc_no", "identifier-v1", 13, 301.6378, 664.3898),
            _FieldSlot("importer_name", "text-v1", 15, 49.0, 609.8898),
            _FieldSlot("load_port", "text-v1", 17, 49.0, 572.8898),
            _FieldSlot("disch_port", "text-v1", 19, 175.3189, 572.8898),
            _FieldSlot("vessel", "text-v1", 24, 49.0, 550.8898),
            _FieldSlot("good", "text-v1", 35, 153.0, 513.8898),
            _FieldSlot(
                "total",
                "text-v1",
                40,
                (480.4596, 487.1316, 491.5796, 496.0276, 502.6996),
                362.8898,
            ),
        ),
        table=_TableSlot(
            tuple(range(11, 16)),
            tuple(range(16, 21)),
            (
                (45.0, 527.8898, 105.0, 16.0),
                (150.0, 527.8898, 170.0, 16.0),
                (320.0, 527.8898, 80.0, 16.0),
                (400.0, 527.8898, 80.0, 16.0),
                (480.0, 527.8898, 70.27559, 16.0),
            ),
            (
                (45.0, 377.8898, 105.0, 150.0),
                (150.0, 377.8898, 170.0, 150.0),
                (320.0, 377.8898, 80.0, 150.0),
                (400.0, 377.8898, 80.0, 150.0),
                (480.0, 377.8898, 70.27559, 150.0),
            ),
        ),
    ),
    "packing_list": _TemplateProfile(
        text_run_count=51,
        rectangle_count=21,
        field_slots=(
            _FieldSlot("exporter_name", "text-v1", 3, 49.0, 753.8898),
            _FieldSlot("inv_no", "identifier-v1", 6, 324.0, 753.8898),
            _FieldSlot("lc_no", "identifier-v1", 9, 324.0, 717.8898),
            _FieldSlot("good", "text-v1", 42, 178.0, 563.8898),
            _FieldSlot("net_weight", "text-v1", 48, (346.31, 350.202, 356.04), 437.8898),
            _FieldSlot("gross_weight", "text-v1", 49, (421.31, 425.202, 431.04), 437.8898),
            _FieldSlot("cbm", "text-v1", 50, (514.7686, 518.6606), 437.8898),
        ),
        table=_TableSlot(
            tuple(range(11, 16)),
            tuple(range(16, 21)),
            (
                (45.0, 577.8898, 130.0, 22.0),
                (175.0, 577.8898, 140.0, 22.0),
                (315.0, 577.8898, 75.0, 22.0),
                (390.0, 577.8898, 75.0, 22.0),
                (465.0, 577.8898, 85.27559, 22.0),
            ),
            (
                (45.0, 427.8898, 130.0, 150.0),
                (175.0, 427.8898, 140.0, 150.0),
                (315.0, 427.8898, 75.0, 150.0),
                (390.0, 427.8898, 75.0, 150.0),
                (465.0, 427.8898, 85.27559, 150.0),
            ),
        ),
    ),
}


def _parser_spec_payload() -> dict[str, object]:
    return {
        "revision": SOURCE_LAYOUT_PARSER_SPEC_REVISION,
        "layout_truth_kind": SOURCE_LAYOUT_TRUTH_KIND,
        "profiles": {
            document_type: {
                "text_run_count": profile.text_run_count,
                "rectangle_count": profile.rectangle_count,
                "field_slots": [
                    {
                        "field_name": slot.field_name,
                        "normalizer": slot.normalizer,
                        "reading_order": slot.reading_order,
                        "x_values": list(slot.x) if isinstance(slot.x, tuple) else [slot.x],
                        "y": slot.y,
                    }
                    for slot in profile.field_slots
                ],
                "table": {
                    "header_rectangle_indices": list(profile.table.header_rectangle_indices),
                    "body_rectangle_indices": list(profile.table.body_rectangle_indices),
                    "header_rectangle_geometry": [
                        list(rectangle) for rectangle in profile.table.header_rectangle_geometry
                    ],
                    "body_rectangle_geometry": [
                        list(rectangle) for rectangle in profile.table.body_rectangle_geometry
                    ],
                },
            }
            for document_type, profile in sorted(_TEMPLATE_PROFILES.items())
        },
    }


def source_layout_parser_spec_sha256() -> str:
    return sha256_bytes(canonical_json_bytes(_parser_spec_payload()))


class SourceAttestedSyntheticGoldRecordV1(_WireModel):
    asset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    source_sha256: str = Field(pattern=_SHA256)
    fixture_kind: Literal["clean", "error"]
    document_type: SyntheticDocumentType
    set_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    content_stream_sha256: str = Field(pattern=_SHA256)
    fields: list[GoldFieldV2] = Field(min_length=1, max_length=1_000)
    spans: list[GoldSpanV2] = Field(min_length=1, max_length=100_000)
    table_cells: list[GoldTableCellV2] = Field(min_length=1, max_length=100_000)
    source_truth_record_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _record_is_unambiguous_and_bound(self) -> "SourceAttestedSyntheticGoldRecordV1":
        fields = [item.field_name for item in self.fields]
        spans = [(item.page_number, item.reading_order) for item in self.spans]
        cells = [
            (item.page_number, item.table_index, item.row_index, item.column_index)
            for item in self.table_cells
        ]
        if len(fields) != len(set(fields)):
            raise ValueError("source-attested field names must be unique")
        if len(spans) != len(set(spans)):
            raise ValueError("source-attested span reading order must be unique")
        if len(cells) != len(set(cells)):
            raise ValueError("source-attested table cells must be unique")
        if self.source_truth_record_sha256 != source_truth_record_sha256(self):
            raise ValueError("source-attested Gold record SHA-256 mismatch")
        return self


def _source_truth_record_payload(
    record: SourceAttestedSyntheticGoldRecordV1 | Mapping[str, object],
) -> dict[str, object]:
    payload = (
        record.model_dump(mode="json", warnings=False)
        if isinstance(record, SourceAttestedSyntheticGoldRecordV1)
        else dict(record)
    )
    payload.pop("source_truth_record_sha256", None)
    return payload


def source_truth_record_sha256(
    record: SourceAttestedSyntheticGoldRecordV1 | Mapping[str, object],
) -> str:
    return sha256_bytes(canonical_json_bytes(_source_truth_record_payload(record)))


class SourceAttestedSyntheticLayoutTruthV1(_WireModel):
    """Immutable local source reference, deliberately separate from human Gold."""

    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["source_attested_synthetic_pdf_layout_truth"] = (
        "source_attested_synthetic_pdf_layout_truth"
    )
    provenance_kind: Literal["source_attested_synthetic"] = SOURCE_ATTESTED_EVALUATION_BASIS
    evaluation_scope: Literal["synthetic_content_stream_compatibility"] = (
        SOURCE_ATTESTED_EVALUATION_SCOPE
    )
    layout_truth_kind: Literal["pdf_content_stream_layout_v1"] = SOURCE_LAYOUT_TRUTH_KIND
    compiler_revision: Literal["synthetic-pdf-content-stream-layout-compiler-v1"] = (
        SOURCE_LAYOUT_COMPILER_REVISION
    )
    parser_spec_sha256: str = Field(pattern=_SHA256)
    dataset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    dataset_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    dataset_receipt_sha256: str = Field(pattern=_SHA256)
    packet_kind: Literal["stratified-v2"] = "stratified-v2"
    packet_split: Literal["development", "sealed"]
    packet_receipt_sha256: str = Field(pattern=_SHA256)
    clean_manifest_sha256: str = Field(pattern=_SHA256)
    error_labels_sha256: str = Field(pattern=_SHA256)
    source_manifest_mismatch_count: int = Field(ge=0, le=100_000)
    source_manifest_mismatch_receipt_sha256: str = Field(pattern=_SHA256)
    records: list[SourceAttestedSyntheticGoldRecordV1] = Field(min_length=1, max_length=100_000)
    layout_truth_receipt_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _truth_is_complete_and_bound(self) -> "SourceAttestedSyntheticLayoutTruthV1":
        if self.parser_spec_sha256 != source_layout_parser_spec_sha256():
            raise ValueError("source layout parser specification SHA-256 mismatch")
        assets = [record.asset_id for record in self.records]
        if len(assets) != len(set(assets)):
            raise ValueError("source-attested layout truth asset IDs must be unique")
        if self.layout_truth_receipt_sha256 != layout_truth_receipt_sha256(self):
            raise ValueError("source layout truth receipt SHA-256 mismatch")
        return self


def _layout_truth_payload(
    truth: SourceAttestedSyntheticLayoutTruthV1 | Mapping[str, object],
) -> dict[str, object]:
    payload = (
        truth.model_dump(mode="json", warnings=False)
        if isinstance(truth, SourceAttestedSyntheticLayoutTruthV1)
        else dict(truth)
    )
    payload.pop("layout_truth_receipt_sha256", None)
    return payload


def layout_truth_receipt_sha256(
    truth: SourceAttestedSyntheticLayoutTruthV1 | Mapping[str, object],
) -> str:
    return sha256_bytes(canonical_json_bytes(_layout_truth_payload(truth)))


def _parse_pdf_number(token: tuple[str, object], *, code: str) -> float:
    kind, value = token
    if kind != "number" or isinstance(value, bool) or not isinstance(value, float):
        raise SourceAttestedSyntheticPdfTruthError(code)
    return value


class _ContentLexer:
    """Tiny PDF-content lexer for the one pinned, ASCII-only fixture subset."""

    def __init__(self, content: bytes) -> None:
        self._content = content
        self._index = 0

    def _skip_ignored(self) -> None:
        while self._index < len(self._content):
            byte = self._content[self._index]
            if byte in b" \t\r\n\x0c\x00":
                self._index += 1
                continue
            if byte == ord("%"):
                while (
                    self._index < len(self._content) and self._content[self._index] not in b"\r\n"
                ):
                    self._index += 1
                continue
            return

    def _literal_string(self) -> tuple[str, object]:
        self._index += 1
        depth = 1
        decoded = bytearray()
        while self._index < len(self._content):
            byte = self._content[self._index]
            self._index += 1
            if byte == ord("\\"):
                if self._index >= len(self._content):
                    raise SourceAttestedSyntheticPdfTruthError("source_layout_pdf_string_invalid")
                escaped = self._content[self._index]
                self._index += 1
                named = {
                    ord("n"): ord("\n"),
                    ord("r"): ord("\r"),
                    ord("t"): ord("\t"),
                    ord("b"): ord("\b"),
                    ord("f"): ord("\f"),
                }
                if escaped in named:
                    decoded.append(named[escaped])
                    continue
                if escaped in (ord("\r"), ord("\n")):
                    if (
                        escaped == ord("\r")
                        and self._index < len(self._content)
                        and self._content[self._index] == ord("\n")
                    ):
                        self._index += 1
                    continue
                if ord("0") <= escaped <= ord("7"):
                    octal = bytearray((escaped,))
                    for _ in range(2):
                        if self._index < len(self._content) and ord("0") <= self._content[
                            self._index
                        ] <= ord("7"):
                            octal.append(self._content[self._index])
                            self._index += 1
                    decoded.append(int(octal, 8))
                    continue
                decoded.append(escaped)
                continue
            if byte == ord("("):
                depth += 1
                decoded.append(byte)
                continue
            if byte == ord(")"):
                depth -= 1
                if depth == 0:
                    try:
                        return "string", decoded.decode("latin-1")
                    except UnicodeDecodeError as error:  # pragma: no cover - latin-1 is total
                        raise SourceAttestedSyntheticPdfTruthError(
                            "source_layout_pdf_string_invalid"
                        ) from error
                decoded.append(byte)
                continue
            decoded.append(byte)
        raise SourceAttestedSyntheticPdfTruthError("source_layout_pdf_string_invalid")

    def next_token(self) -> tuple[str, object] | None:
        self._skip_ignored()
        if self._index >= len(self._content):
            return None
        byte = self._content[self._index]
        if byte == ord("("):
            return self._literal_string()
        if byte in (ord("["), ord("]"), ord("<"), ord(">"), ord("{"), ord("}")):
            raise SourceAttestedSyntheticPdfTruthError("source_layout_pdf_construct_unsupported")
        if byte == ord("/"):
            start = self._index + 1
            self._index = start
            while (
                self._index < len(self._content)
                and self._content[self._index] not in b" \t\r\n\x0c[]<>()/{%"
            ):
                self._index += 1
            if self._index == start:
                raise SourceAttestedSyntheticPdfTruthError("source_layout_pdf_name_invalid")
            return "name", self._content[start : self._index].decode("latin-1")
        start = self._index
        while (
            self._index < len(self._content)
            and self._content[self._index] not in b" \t\r\n\x0c[]<>()/{%"
        ):
            self._index += 1
        raw = self._content[start : self._index]
        if not raw:
            raise SourceAttestedSyntheticPdfTruthError("source_layout_pdf_token_invalid")
        try:
            return "number", float(raw)
        except ValueError:
            try:
                return "operator", raw.decode("ascii")
            except UnicodeDecodeError as error:
                raise SourceAttestedSyntheticPdfTruthError(
                    "source_layout_pdf_operator_invalid"
                ) from error


def _extract_media_box(pdf_bytes: bytes) -> tuple[float, float, float, float]:
    matches = list(_MEDIA_BOX_RE.finditer(pdf_bytes))
    if len(matches) != 1:
        raise SourceAttestedSyntheticPdfTruthError("source_layout_pdf_page_invalid")
    values = tuple(float(matches[0].group(name)) for name in ("x0", "y0", "x1", "y1"))
    if values != (0.0, 0.0, 595.2756, 841.8898):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_pdf_page_invalid")
    return values


def _decode_content_stream(pdf_bytes: bytes) -> bytes:
    if (
        not pdf_bytes.startswith(b"%PDF-1.3")
        or b"/Encrypt" in pdf_bytes
        or pdf_bytes.count(b"stream") != 2
        or pdf_bytes.count(b"/Filter") != 1
        or pdf_bytes.count(b"/ASCII85Decode") != 1
        or pdf_bytes.count(b"/FlateDecode") != 1
    ):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_pdf_construct_unsupported")
    matches = list(_PDF_CONTENT_STREAM_RE.finditer(pdf_bytes))
    if len(matches) != 1:
        raise SourceAttestedSyntheticPdfTruthError("source_layout_pdf_stream_invalid")
    try:
        encoded = matches[0].group("content").strip()
        return zlib.decompress(base64.a85decode(encoded, adobe=True))
    except (ValueError, zlib.error) as error:
        raise SourceAttestedSyntheticPdfTruthError("source_layout_pdf_stream_invalid") from error


def _pop_operands(
    operands: list[tuple[str, object]], *, count: int, code: str
) -> list[tuple[str, object]]:
    if len(operands) != count:
        raise SourceAttestedSyntheticPdfTruthError(code)
    values = operands[:]
    operands.clear()
    return values


def _parse_pdf_content_stream_v1(pdf_bytes: bytes) -> _ParsedContentStream:
    """Parse only the known final-PDF content stream without external libraries."""

    if len(pdf_bytes) < 1 or len(pdf_bytes) > _MAX_PDF_BYTES:
        raise SourceAttestedSyntheticPdfTruthError("source_layout_pdf_size_invalid")
    media_box = _extract_media_box(pdf_bytes)
    decoded = _decode_content_stream(pdf_bytes)
    lexer = _ContentLexer(decoded)
    operands: list[tuple[str, object]] = []
    text_runs: list[_TextRun] = []
    rectangles: list[_Rectangle] = []
    matrix: tuple[float, ...] = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
    leading = 0.0
    in_text = False
    allowed_operators = {"cm", "BT", "ET", "Tf", "TL", "Tm", "T*", "Tj", "w", "re", "S", "n"}

    while token := lexer.next_token():
        kind, value = token
        if kind != "operator":
            operands.append(token)
            continue
        operator = value
        if not isinstance(operator, str) or operator not in allowed_operators:
            raise SourceAttestedSyntheticPdfTruthError("source_layout_pdf_operator_unsupported")
        if operator == "cm":
            values = _pop_operands(operands, count=6, code="source_layout_pdf_operand_invalid")
            matrix_values = tuple(
                _parse_pdf_number(item, code="source_layout_pdf_operand_invalid") for item in values
            )
            if matrix_values != (1.0, 0.0, 0.0, 1.0, 0.0, 0.0):
                raise SourceAttestedSyntheticPdfTruthError(
                    "source_layout_pdf_transform_unsupported"
                )
            continue
        if operator == "BT":
            if operands or in_text:
                raise SourceAttestedSyntheticPdfTruthError("source_layout_pdf_text_state_invalid")
            in_text = True
            matrix = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)
            leading = 0.0
            continue
        if operator == "ET":
            if operands or not in_text:
                raise SourceAttestedSyntheticPdfTruthError("source_layout_pdf_text_state_invalid")
            in_text = False
            continue
        if operator == "Tf":
            values = _pop_operands(operands, count=2, code="source_layout_pdf_operand_invalid")
            if not in_text or values[0][0] != "name":
                raise SourceAttestedSyntheticPdfTruthError("source_layout_pdf_text_state_invalid")
            _parse_pdf_number(values[1], code="source_layout_pdf_operand_invalid")
            continue
        if operator == "TL":
            values = _pop_operands(operands, count=1, code="source_layout_pdf_operand_invalid")
            if not in_text:
                raise SourceAttestedSyntheticPdfTruthError("source_layout_pdf_text_state_invalid")
            leading = _parse_pdf_number(values[0], code="source_layout_pdf_operand_invalid")
            continue
        if operator == "Tm":
            values = _pop_operands(operands, count=6, code="source_layout_pdf_operand_invalid")
            if not in_text:
                raise SourceAttestedSyntheticPdfTruthError("source_layout_pdf_text_state_invalid")
            matrix = tuple(
                _parse_pdf_number(item, code="source_layout_pdf_operand_invalid") for item in values
            )
            if matrix[:4] != (1.0, 0.0, 0.0, 1.0):
                raise SourceAttestedSyntheticPdfTruthError(
                    "source_layout_pdf_transform_unsupported"
                )
            continue
        if operator == "T*":
            if operands or not in_text:
                raise SourceAttestedSyntheticPdfTruthError("source_layout_pdf_text_state_invalid")
            matrix = (*matrix[:4], matrix[4], matrix[5] - leading)
            continue
        if operator == "Tj":
            values = _pop_operands(operands, count=1, code="source_layout_pdf_operand_invalid")
            if not in_text or values[0][0] != "string":
                raise SourceAttestedSyntheticPdfTruthError("source_layout_pdf_text_state_invalid")
            text = values[0][1]
            if not isinstance(text, str) or not text.strip():
                raise SourceAttestedSyntheticPdfTruthError("source_layout_pdf_text_invalid")
            text_runs.append(
                _TextRun(
                    reading_order=len(text_runs) + 1,
                    x=matrix[4],
                    y=matrix[5],
                    text=text,
                )
            )
            continue
        if operator == "w":
            values = _pop_operands(operands, count=1, code="source_layout_pdf_operand_invalid")
            _parse_pdf_number(values[0], code="source_layout_pdf_operand_invalid")
            continue
        if operator == "re":
            values = _pop_operands(operands, count=4, code="source_layout_pdf_operand_invalid")
            x, y, width, height = (
                _parse_pdf_number(item, code="source_layout_pdf_operand_invalid") for item in values
            )
            if width <= 0 or height <= 0:
                raise SourceAttestedSyntheticPdfTruthError("source_layout_pdf_rectangle_invalid")
            rectangles.append(_Rectangle(x=x, y=y, width=width, height=height))
            continue
        if operator in {"S", "n"}:
            if operands or in_text:
                raise SourceAttestedSyntheticPdfTruthError(
                    "source_layout_pdf_graphics_state_invalid"
                )
            continue
        raise SourceAttestedSyntheticPdfTruthError("source_layout_pdf_operator_unsupported")
    if operands or in_text or not text_runs or not rectangles:
        raise SourceAttestedSyntheticPdfTruthError("source_layout_pdf_text_state_invalid")
    return _ParsedContentStream(
        decoded_sha256=sha256_bytes(decoded),
        text_runs=tuple(text_runs),
        rectangles=tuple(rectangles),
        media_box=media_box,
    )


def _require_profile(
    *, document_type: SyntheticDocumentType, parsed: _ParsedContentStream
) -> _TemplateProfile:
    profile = _TEMPLATE_PROFILES.get(document_type)
    if profile is None:
        raise SourceAttestedSyntheticPdfTruthError("source_layout_document_type_invalid")
    if (
        len(parsed.text_runs) != profile.text_run_count
        or len(parsed.rectangles) != profile.rectangle_count
    ):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_template_profile_mismatch")
    all_indices = [
        *profile.table.header_rectangle_indices,
        *profile.table.body_rectangle_indices,
    ]
    if len(set(all_indices)) != len(all_indices) or any(
        index < 0 or index >= len(parsed.rectangles) for index in all_indices
    ):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_template_profile_invalid")
    expected_table_geometry = (
        *profile.table.header_rectangle_geometry,
        *profile.table.body_rectangle_geometry,
    )
    if len(expected_table_geometry) != len(all_indices):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_template_profile_invalid")
    for rectangle_index, expected in zip(all_indices, expected_table_geometry, strict=True):
        rectangle = parsed.rectangles[rectangle_index]
        actual = (rectangle.x, rectangle.y, rectangle.width, rectangle.height)
        # This is an authored-layout profile, not a fuzzy visual detector.
        # The tiny tolerance admits decimal parser representation only.
        if any(
            abs(observed - target) > 0.0001
            for observed, target in zip(actual, expected, strict=True)
        ):
            raise SourceAttestedSyntheticPdfTruthError("source_layout_template_profile_mismatch")
    if any(
        slot.reading_order < 1 or slot.reading_order > len(parsed.text_runs)
        for slot in profile.field_slots
    ):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_template_profile_invalid")
    for slot in profile.field_slots:
        run = parsed.text_runs[slot.reading_order - 1]
        x_values = slot.x if isinstance(slot.x, tuple) else (slot.x,)
        if (
            not any(abs(run.x - expected_x) <= 0.0001 for expected_x in x_values)
            or abs(run.y - slot.y) > 0.0001
        ):
            raise SourceAttestedSyntheticPdfTruthError("source_layout_template_profile_mismatch")
    return profile


def _text_in_rectangle(*, run: _TextRun, rectangle: _Rectangle) -> bool:
    # Text origin is authoritative for this content-stream profile.  Border
    # touching is accepted so an exact template round-trip cannot lose a cell.
    return (
        rectangle.x <= run.x <= rectangle.x + rectangle.width
        and rectangle.y <= run.y <= rectangle.y + rectangle.height
    )


def _table_cells_from_profile(
    *, parsed: _ParsedContentStream, profile: _TemplateProfile
) -> list[GoldTableCellV2]:
    cells: list[GoldTableCellV2] = []
    for row_index, indices in enumerate(
        (profile.table.header_rectangle_indices, profile.table.body_rectangle_indices)
    ):
        for column_index, rectangle_index in enumerate(indices):
            rectangle = parsed.rectangles[rectangle_index]
            matched = [
                run.text
                for run in parsed.text_runs
                if _text_in_rectangle(run=run, rectangle=rectangle)
            ]
            if not matched:
                raise SourceAttestedSyntheticPdfTruthError("source_layout_table_cell_ambiguous")
            # Content-stream order preserves the authored line order inside one
            # cell.  We do not infer rows or merge text across rectangles.
            text = "\n".join(matched).strip()
            if not text:
                raise SourceAttestedSyntheticPdfTruthError("source_layout_table_cell_ambiguous")
            cells.append(
                GoldTableCellV2(
                    page_number=1,
                    table_index=1,
                    row_index=row_index,
                    column_index=column_index,
                    text=text,
                )
            )
    return cells


def _fields_from_profile(
    *, parsed: _ParsedContentStream, profile: _TemplateProfile
) -> list[GoldFieldV2]:
    fields: list[GoldFieldV2] = []
    for slot in profile.field_slots:
        run = parsed.text_runs[slot.reading_order - 1]
        value = run.text.strip()
        if not value:
            raise SourceAttestedSyntheticPdfTruthError("source_layout_field_ambiguous")
        fields.append(
            GoldFieldV2(
                field_name=slot.field_name,
                value=value,
                normalizer=slot.normalizer,
            )
        )
    return fields


def _mismatch_receipt(entries: Sequence[Mapping[str, str]]) -> str:
    return sha256_bytes(canonical_json_bytes(list(entries)))


def _source_record(
    *,
    asset_id: str,
    source_sha256: str,
    fixture_kind: Literal["clean", "error"],
    document_type: SyntheticDocumentType,
    set_id: str,
    parsed: _ParsedContentStream,
    profile: _TemplateProfile,
) -> SourceAttestedSyntheticGoldRecordV1:
    fields = _fields_from_profile(parsed=parsed, profile=profile)
    payload: dict[str, object] = {
        "asset_id": asset_id,
        "source_sha256": source_sha256,
        "fixture_kind": fixture_kind,
        "document_type": document_type,
        "set_id": set_id,
        "content_stream_sha256": parsed.decoded_sha256,
        "fields": [field.model_dump(mode="json", warnings=False) for field in fields],
        "spans": [
            GoldSpanV2(
                page_number=1,
                text=run.text,
                reading_order=run.reading_order,
            ).model_dump(mode="json", warnings=False)
            for run in parsed.text_runs
        ],
        "table_cells": [
            cell.model_dump(mode="json", warnings=False)
            for cell in _table_cells_from_profile(parsed=parsed, profile=profile)
        ],
    }
    payload["source_truth_record_sha256"] = source_truth_record_sha256(payload)
    return SourceAttestedSyntheticGoldRecordV1.model_validate(payload)


def _verify_receipt_packet_binding(
    *, receipt: DatasetReceiptV1, packet: EvaluationPacketV2
) -> None:
    if (
        receipt_sha256(receipt) != packet.receipt_sha256
        or receipt.dataset_id != packet.dataset_id
        or receipt.dataset_revision != packet.dataset_revision
    ):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_packet_receipt_mismatch")


def _source_asset_path(*, source_directory: Path, relative_path: str) -> Path:
    candidate = source_directory / Path(relative_path)
    if candidate.is_symlink():
        raise SourceAttestedSyntheticPdfTruthError("source_layout_source_asset_invalid")
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(source_directory.resolve(strict=True))
        mode = resolved.stat().st_mode
    except (OSError, ValueError) as error:
        raise SourceAttestedSyntheticPdfTruthError("source_layout_source_asset_invalid") from error
    if not stat.S_ISREG(mode) or resolved.is_symlink():
        raise SourceAttestedSyntheticPdfTruthError("source_layout_source_asset_invalid")
    return resolved


def _read_verified_source_pdf(*, source_directory: Path, asset: DatasetAssetReceiptV1) -> bytes:
    if asset.media_type != "application/pdf" or asset.source_role != "preview_candidate":
        raise SourceAttestedSyntheticPdfTruthError("source_layout_source_asset_invalid")
    path = _source_asset_path(source_directory=source_directory, relative_path=asset.relative_path)
    try:
        pdf_bytes = path.read_bytes()
    except OSError as error:
        raise SourceAttestedSyntheticPdfTruthError("source_layout_source_asset_invalid") from error
    if len(pdf_bytes) != asset.byte_size or sha256_bytes(pdf_bytes) != asset.sha256:
        raise SourceAttestedSyntheticPdfTruthError("source_layout_source_asset_invalid")
    return pdf_bytes


def _require_error_content_stream_difference(
    *,
    item: StratifiedPacketItemV2,
    parsed_error: _ParsedContentStream,
    assets_by_relative_path: Mapping[str, DatasetAssetReceiptV1],
    source_directory: Path,
) -> None:
    """Prove an error stratum visibly differs from its exact clean sibling.

    Labels alone are not visual evidence.  The stable fixture naming makes the
    counterpart deterministic; a missing or unsupported sibling is an error,
    never a reason to quietly keep the labelled item.
    """

    matched = _ERROR_SET_ID_RE.fullmatch(item.set_id)
    suffix = _DOCUMENT_FILENAME_SUFFIX.get(item.document_type)
    if matched is None or suffix is None:
        raise SourceAttestedSyntheticPdfTruthError("source_layout_error_counterpart_invalid")
    set_number = matched.group("set_number")
    clean_relative_path = f"clean_sets/set_{set_number}/C{set_number}_{suffix}.pdf"
    clean_asset = assets_by_relative_path.get(clean_relative_path)
    if clean_asset is None:
        raise SourceAttestedSyntheticPdfTruthError("source_layout_error_counterpart_missing")
    clean_pdf_bytes = _read_verified_source_pdf(
        source_directory=source_directory, asset=clean_asset
    )
    parsed_clean = _parse_pdf_content_stream_v1(clean_pdf_bytes)
    _require_profile(document_type=item.document_type, parsed=parsed_clean)
    if parsed_clean.decoded_sha256 == parsed_error.decoded_sha256:
        raise SourceAttestedSyntheticPdfTruthError("source_layout_error_content_unchanged")


def compile_source_attested_synthetic_pdf_layout_truth(
    *,
    receipt: DatasetReceiptV1,
    packet: EvaluationPacketV2,
    source_directory: Path,
    clean_manifest_path: Path,
    error_labels_path: Path,
) -> SourceAttestedSyntheticLayoutTruthV1:
    """Compile local content-stream layout truth without candidate OCR output.

    The two CSVs are used only to retain their immutable receipt bindings,
    selected-error eligibility, and a compact manifest-vs-render mismatch
    receipt.  They never provide the source truth values returned by this
    compiler.
    """

    source_directory = _local_input_directory(
        source_directory, code="source_layout_source_directory_invalid"
    )
    clean_manifest_path = _local_input_file(
        clean_manifest_path, code="source_layout_manifest_source_invalid"
    )
    error_labels_path = _local_input_file(
        error_labels_path, code="source_layout_label_source_invalid"
    )
    _verify_receipt_packet_binding(receipt=receipt, packet=packet)
    verify_receipt_source_assets(receipt=receipt, source_directory=source_directory)
    # This validates CSV headers, error-label eligibility, and their exact
    # binding to the selected packet.  Its values are only compared below; they
    # cannot overwrite the content-stream-derived values.
    manifest_drafts = build_synthetic_pdf_draft_gold(
        receipt=receipt,
        packet=packet,
        clean_manifest_path=clean_manifest_path,
        error_labels_path=error_labels_path,
    )
    labels_by_key = load_synthetic_error_labels(error_labels_path)
    receipt_assets = {asset.asset_id: asset for asset in receipt.assets}
    receipt_assets_by_relative_path = {asset.relative_path: asset for asset in receipt.assets}
    draft_by_asset = {record.asset_id: record for record in manifest_drafts.records}
    if set(draft_by_asset) != {item.asset_id for item in packet.items}:
        raise SourceAttestedSyntheticPdfTruthError("source_layout_manifest_packet_mismatch")

    records: list[SourceAttestedSyntheticGoldRecordV1] = []
    mismatch_entries: list[dict[str, str]] = []
    for item in packet.items:
        asset = receipt_assets.get(item.asset_id)
        if (
            asset is None
            or asset.sha256 != item.source_sha256
            or asset.media_type != "application/pdf"
        ):
            raise SourceAttestedSyntheticPdfTruthError("source_layout_source_asset_invalid")
        pdf_bytes = _read_verified_source_pdf(source_directory=source_directory, asset=asset)
        parsed = _parse_pdf_content_stream_v1(pdf_bytes)
        profile = _require_profile(document_type=item.document_type, parsed=parsed)
        if item.fixture_kind == "error":
            if not labels_by_key.get((item.set_id, item.document_type)):
                raise SourceAttestedSyntheticPdfTruthError("source_layout_error_label_missing")
            _require_error_content_stream_difference(
                item=item,
                parsed_error=parsed,
                assets_by_relative_path=receipt_assets_by_relative_path,
                source_directory=source_directory,
            )
        record = _source_record(
            asset_id=item.asset_id,
            source_sha256=item.source_sha256,
            fixture_kind=item.fixture_kind,
            document_type=item.document_type,
            set_id=item.set_id,
            parsed=parsed,
            profile=profile,
        )
        draft = draft_by_asset[item.asset_id]
        draft_fields = {field.field_name: field for field in draft.fields}
        for field in record.fields:
            draft_field = draft_fields.get(field.field_name)
            if draft_field is None:
                continue
            if normalize_field_value(field.value, field.normalizer) != normalize_field_value(
                draft_field.value, draft_field.normalizer
            ):
                mismatch_entries.append(
                    {
                        "asset_id": item.asset_id,
                        "field_name": field.field_name,
                        "manifest_value_sha256": sha256_bytes(
                            normalize_field_value(draft_field.value, draft_field.normalizer).encode(
                                "utf-8"
                            )
                        ),
                        "content_stream_value_sha256": sha256_bytes(
                            normalize_field_value(field.value, field.normalizer).encode("utf-8")
                        ),
                    }
                )
        records.append(record)
    if len(records) != len(packet.items):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_packet_coverage_invalid")
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "artifact_kind": "source_attested_synthetic_pdf_layout_truth",
        "provenance_kind": SOURCE_ATTESTED_EVALUATION_BASIS,
        "evaluation_scope": SOURCE_ATTESTED_EVALUATION_SCOPE,
        "layout_truth_kind": SOURCE_LAYOUT_TRUTH_KIND,
        "compiler_revision": SOURCE_LAYOUT_COMPILER_REVISION,
        "parser_spec_sha256": source_layout_parser_spec_sha256(),
        "dataset_id": packet.dataset_id,
        "dataset_revision": packet.dataset_revision,
        "dataset_receipt_sha256": receipt_sha256(receipt),
        "packet_kind": packet.packet_kind,
        "packet_split": packet.split,
        "packet_receipt_sha256": packet.packet_receipt_sha256,
        "clean_manifest_sha256": sha256_file(clean_manifest_path),
        "error_labels_sha256": sha256_file(error_labels_path),
        "source_manifest_mismatch_count": len(mismatch_entries),
        "source_manifest_mismatch_receipt_sha256": _mismatch_receipt(mismatch_entries),
        "records": [record.model_dump(mode="json", warnings=False) for record in records],
    }
    payload["layout_truth_receipt_sha256"] = layout_truth_receipt_sha256(payload)
    return SourceAttestedSyntheticLayoutTruthV1.model_validate(payload)


def validate_source_attested_layout_truth_for_evaluation(
    *,
    receipt: DatasetReceiptV1,
    packet: EvaluationPacketV2,
    truth: SourceAttestedSyntheticLayoutTruthV1,
) -> None:
    """Verify receipt → packet → source-layout bindings before scoring."""

    _verify_receipt_packet_binding(receipt=receipt, packet=packet)
    if (
        truth.dataset_id != packet.dataset_id
        or truth.dataset_revision != packet.dataset_revision
        or truth.dataset_receipt_sha256 != receipt_sha256(receipt)
        or truth.packet_kind != packet.packet_kind
        or truth.packet_split != packet.split
        or truth.packet_receipt_sha256 != packet.packet_receipt_sha256
    ):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_evaluation_binding_mismatch")
    packet_by_asset = {item.asset_id: item for item in packet.items}
    if set(packet_by_asset) != {record.asset_id for record in truth.records}:
        raise SourceAttestedSyntheticPdfTruthError("source_layout_evaluation_coverage_mismatch")
    for record in truth.records:
        item = packet_by_asset[record.asset_id]
        if (
            record.source_sha256 != item.source_sha256
            or record.fixture_kind != item.fixture_kind
            or record.document_type != item.document_type
            or record.set_id != item.set_id
        ):
            raise SourceAttestedSyntheticPdfTruthError("source_layout_evaluation_binding_mismatch")


class SourceAttestedSyntheticOcrMetricsV1(_WireModel):
    field_exact_match: float = Field(ge=0, le=1)
    ocr_cer: float = Field(ge=0, le=1)
    table_cell_f1: float = Field(ge=0, le=1)


class SourceAttestedSyntheticOcrEvaluationResultV1(_WireModel):
    """Content-free result with the exact runtime source-attested contract."""

    schema_version: Literal["1.0"] = "1.0"
    evaluation_basis: Literal["source_attested_synthetic"] = SOURCE_ATTESTED_EVALUATION_BASIS
    dataset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    dataset_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    packet_kind: Literal["development", "sealed"]
    packet_receipt_sha256: str = Field(pattern=_SHA256)
    evaluation_scope: Literal["synthetic_content_stream_compatibility"] = (
        SOURCE_ATTESTED_EVALUATION_SCOPE
    )
    layout_truth_kind: Literal["pdf_content_stream_layout_v1"] = SOURCE_LAYOUT_TRUTH_KIND
    layout_truth_receipt_sha256: str = Field(pattern=_SHA256)
    layout_truth_parser_spec_sha256: str = Field(pattern=_SHA256)
    development_selection_receipt_sha256: str | None = Field(default=None, pattern=_SHA256)
    sealed_run_receipt_sha256: str | None = Field(default=None, pattern=_SHA256)
    engine_kind: EngineKind
    extractor_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    configuration_sha256: str = Field(pattern=_SHA256)
    model_inventory_sha256: str = Field(pattern=_SHA256)
    candidate_bundle_sha256: str = Field(pattern=_SHA256)
    evaluation_result_sha256: str = Field(pattern=_SHA256)
    passed: bool
    metrics: SourceAttestedSyntheticOcrMetricsV1

    @model_validator(mode="after")
    def _result_is_hash_bound_and_gate_derived(
        self,
    ) -> "SourceAttestedSyntheticOcrEvaluationResultV1":
        if self.evaluation_result_sha256 != source_attested_evaluation_result_sha256(self):
            raise ValueError("source-attested evaluation result SHA-256 mismatch")
        if self.layout_truth_parser_spec_sha256 != source_layout_parser_spec_sha256():
            raise ValueError("source-attested layout parser specification SHA-256 mismatch")
        if self.packet_kind == "development":
            if (
                self.development_selection_receipt_sha256 is not None
                or self.sealed_run_receipt_sha256 is not None
            ):
                raise ValueError(
                    "development source-attested result must not carry sealed receipts"
                )
        elif (
            self.development_selection_receipt_sha256 is None
            or self.sealed_run_receipt_sha256 is None
        ):
            raise ValueError(
                "sealed source-attested result requires selection and sealed-run receipts"
            )
        expected = (
            self.metrics.field_exact_match >= 0.99
            and self.metrics.ocr_cer <= 0.01
            and self.metrics.table_cell_f1 >= 0.98
        )
        if self.passed != expected:
            raise ValueError("source-attested result pass flag does not match fixed release gates")
        return self


def _source_attested_result_payload(
    result: SourceAttestedSyntheticOcrEvaluationResultV1 | Mapping[str, object],
) -> dict[str, object]:
    payload = (
        result.model_dump(mode="json", warnings=False)
        if isinstance(result, SourceAttestedSyntheticOcrEvaluationResultV1)
        else dict(result)
    )
    payload.pop("evaluation_result_sha256", None)
    return payload


def source_attested_evaluation_result_sha256(
    result: SourceAttestedSyntheticOcrEvaluationResultV1 | Mapping[str, object],
) -> str:
    return sha256_bytes(canonical_json_bytes(_source_attested_result_payload(result)))


class SourceAttestedDevelopmentCandidateResultV1(_WireModel):
    engine_kind: EngineKind
    extractor_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    configuration_sha256: str = Field(pattern=_SHA256)
    model_inventory_sha256: str = Field(pattern=_SHA256)
    candidate_bundle_sha256: str = Field(pattern=_SHA256)
    evaluation_result_sha256: str = Field(pattern=_SHA256)
    passed: bool


class SourceAttestedSyntheticDevelopmentSelectionV1(_WireModel):
    """Immutable all-three-engine development choice for a source-truth run."""

    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["source_attested_synthetic_pdf_development_selection"] = (
        "source_attested_synthetic_pdf_development_selection"
    )
    evaluation_basis: Literal["source_attested_synthetic"] = SOURCE_ATTESTED_EVALUATION_BASIS
    evaluation_scope: Literal["synthetic_content_stream_compatibility"] = (
        SOURCE_ATTESTED_EVALUATION_SCOPE
    )
    layout_truth_kind: Literal["pdf_content_stream_layout_v1"] = SOURCE_LAYOUT_TRUTH_KIND
    dataset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    dataset_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    development_packet_receipt_sha256: str = Field(pattern=_SHA256)
    layout_truth_receipt_sha256: str = Field(pattern=_SHA256)
    layout_truth_parser_spec_sha256: str = Field(pattern=_SHA256)
    candidate_results: list[SourceAttestedDevelopmentCandidateResultV1] = Field(
        min_length=3, max_length=3
    )
    selected_engine_kind: EngineKind
    selected_extractor_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    selected_configuration_sha256: str = Field(pattern=_SHA256)
    selected_model_inventory_sha256: str = Field(pattern=_SHA256)
    selected_evaluation_result_sha256: str = Field(pattern=_SHA256)
    development_selection_receipt_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _selection_is_complete_and_bound(self) -> "SourceAttestedSyntheticDevelopmentSelectionV1":
        if self.layout_truth_parser_spec_sha256 != source_layout_parser_spec_sha256():
            raise ValueError("source-attested selection parser specification SHA-256 mismatch")
        expected_engines = {"pdf_text_layer", "paddleocr", "docling"}
        if {item.engine_kind for item in self.candidate_results} != expected_engines:
            raise ValueError(
                "source-attested selection requires exactly three unique engine results"
            )
        selected = [
            item
            for item in self.candidate_results
            if (
                item.engine_kind == self.selected_engine_kind
                and item.extractor_revision == self.selected_extractor_revision
                and item.configuration_sha256 == self.selected_configuration_sha256
                and item.model_inventory_sha256 == self.selected_model_inventory_sha256
                and item.evaluation_result_sha256 == self.selected_evaluation_result_sha256
            )
        ]
        if len(selected) != 1 or not selected[0].passed:
            raise ValueError("source-attested selected engine is not a passing development member")
        if self.development_selection_receipt_sha256 != development_selection_receipt_sha256(self):
            raise ValueError("source-attested development selection SHA-256 mismatch")
        return self


def _development_selection_payload(
    selection: SourceAttestedSyntheticDevelopmentSelectionV1 | Mapping[str, object],
) -> dict[str, object]:
    payload = (
        selection.model_dump(mode="json", warnings=False)
        if isinstance(selection, SourceAttestedSyntheticDevelopmentSelectionV1)
        else dict(selection)
    )
    payload.pop("development_selection_receipt_sha256", None)
    return payload


def development_selection_receipt_sha256(
    selection: SourceAttestedSyntheticDevelopmentSelectionV1 | Mapping[str, object],
) -> str:
    return sha256_bytes(canonical_json_bytes(_development_selection_payload(selection)))


class SourceAttestedSyntheticSealedRunV1(_WireModel):
    """One pre-result sealed candidate binding; a second attempt is consumed."""

    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["source_attested_synthetic_pdf_sealed_run"] = (
        "source_attested_synthetic_pdf_sealed_run"
    )
    evaluation_basis: Literal["source_attested_synthetic"] = SOURCE_ATTESTED_EVALUATION_BASIS
    evaluation_scope: Literal["synthetic_content_stream_compatibility"] = (
        SOURCE_ATTESTED_EVALUATION_SCOPE
    )
    layout_truth_kind: Literal["pdf_content_stream_layout_v1"] = SOURCE_LAYOUT_TRUTH_KIND
    dataset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    dataset_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    sealed_packet_receipt_sha256: str = Field(pattern=_SHA256)
    layout_truth_receipt_sha256: str = Field(pattern=_SHA256)
    layout_truth_parser_spec_sha256: str = Field(pattern=_SHA256)
    development_selection_receipt_sha256: str = Field(pattern=_SHA256)
    selected_engine_kind: EngineKind
    selected_extractor_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    selected_configuration_sha256: str = Field(pattern=_SHA256)
    selected_model_inventory_sha256: str = Field(pattern=_SHA256)
    selected_development_evaluation_result_sha256: str = Field(pattern=_SHA256)
    candidate_bundle_sha256: str = Field(pattern=_SHA256)
    candidate_parser_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    sealed_candidate_reservation_sha256: str = Field(pattern=_SHA256)
    sealed_run_receipt_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _sealed_run_is_bound(self) -> "SourceAttestedSyntheticSealedRunV1":
        if self.layout_truth_parser_spec_sha256 != source_layout_parser_spec_sha256():
            raise ValueError("source-attested sealed run parser specification SHA-256 mismatch")
        if self.sealed_run_receipt_sha256 != sealed_run_receipt_sha256(self):
            raise ValueError("source-attested sealed-run SHA-256 mismatch")
        return self


def _sealed_run_payload(
    sealed_run: SourceAttestedSyntheticSealedRunV1 | Mapping[str, object],
) -> dict[str, object]:
    payload = (
        sealed_run.model_dump(mode="json", warnings=False)
        if isinstance(sealed_run, SourceAttestedSyntheticSealedRunV1)
        else dict(sealed_run)
    )
    payload.pop("sealed_run_receipt_sha256", None)
    return payload


def sealed_run_receipt_sha256(
    sealed_run: SourceAttestedSyntheticSealedRunV1 | Mapping[str, object],
) -> str:
    return sha256_bytes(canonical_json_bytes(_sealed_run_payload(sealed_run)))


def _table_cell_set(
    cells: Iterable[GoldTableCellV2 | CandidateTableCellV1],
) -> set[tuple[int, int, int, int, str]]:
    return {
        (
            cell.page_number,
            cell.table_index,
            cell.row_index,
            cell.column_index,
            normalize_table_cell_text(cell.text),
        )
        for cell in cells
    }


def _levenshtein_distance(reference: str, candidate: str) -> int:
    if len(reference) < len(candidate):
        reference, candidate = candidate, reference
    previous = list(range(len(candidate) + 1))
    for reference_index, reference_character in enumerate(reference, start=1):
        current = [reference_index]
        for candidate_index, candidate_character in enumerate(candidate, start=1):
            substitution = previous[candidate_index - 1] + (
                reference_character != candidate_character
            )
            current.append(min(current[-1] + 1, previous[candidate_index] + 1, substitution))
        previous = current
    return previous[-1]


def score_source_attested_synthetic_pdf_candidate_output(
    *,
    receipt: DatasetReceiptV1,
    packet: EvaluationPacketV2,
    layout_truth: SourceAttestedSyntheticLayoutTruthV1,
    candidate_output: SyntheticPdfCandidateOutputBundleV1,
    engine_kind: EngineKind,
    extractor_revision: str,
    configuration_sha256: str,
    model_inventory_sha256: str,
    parser_revision: str,
    development_selection: SourceAttestedSyntheticDevelopmentSelectionV1 | None = None,
    sealed_run: SourceAttestedSyntheticSealedRunV1 | None = None,
) -> SourceAttestedSyntheticOcrEvaluationResultV1:
    """Score one local candidate against final-PDF source-layout truth.

    Native text-layer candidates are permitted as diagnostics.  The result's
    fixed scope makes clear that their score is content-stream compatibility,
    not independent optical OCR evidence.
    """

    validate_source_attested_layout_truth_for_evaluation(
        receipt=receipt, packet=packet, truth=layout_truth
    )
    validate_synthetic_candidate_output_provenance(
        candidate_output=candidate_output,
        packet=packet,
        engine_kind=engine_kind,
        extractor_revision=extractor_revision,
        configuration_sha256=configuration_sha256,
        model_inventory_sha256=model_inventory_sha256,
        parser_revision=parser_revision,
    )
    if packet.split == "development":
        if development_selection is not None or sealed_run is not None:
            raise SourceAttestedSyntheticPdfTruthError(
                "source_layout_development_receipt_not_permitted"
            )
    else:
        if development_selection is None or sealed_run is None:
            raise SourceAttestedSyntheticPdfTruthError("source_layout_sealed_receipt_required")
        _validate_sealed_run_binding(
            packet=packet,
            layout_truth=layout_truth,
            candidate_output=candidate_output,
            engine_kind=engine_kind,
            extractor_revision=extractor_revision,
            configuration_sha256=configuration_sha256,
            model_inventory_sha256=model_inventory_sha256,
            parser_revision=parser_revision,
            development_selection=development_selection,
            sealed_run=sealed_run,
        )
    candidate = candidate_output_from_synthetic_bundle(candidate_output)
    candidate_by_asset = {item.asset_id: item for item in candidate.documents}
    truth_by_asset = {item.asset_id: item for item in layout_truth.records}
    if set(candidate_by_asset) != set(truth_by_asset):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_candidate_coverage_mismatch")

    exact_fields = 0
    total_fields = 0
    total_distance = 0
    total_reference_characters = 0
    source_cells: set[tuple[str, int, int, int, int, str]] = set()
    candidate_cells: set[tuple[str, int, int, int, int, str]] = set()
    for asset_id, source in truth_by_asset.items():
        output = candidate_by_asset[asset_id]
        if output.source_sha256 != source.source_sha256:
            raise SourceAttestedSyntheticPdfTruthError("source_layout_candidate_source_mismatch")
        candidate_fields = {field.field_name: field.value for field in output.fields}
        for field in source.fields:
            total_fields += 1
            if field.field_name in candidate_fields and normalize_field_value(
                candidate_fields[field.field_name], field.normalizer
            ) == normalize_field_value(field.value, field.normalizer):
                exact_fields += 1
        reference_text = canonical_span_text(source.spans)
        candidate_text = canonical_span_text(output.spans)
        if not reference_text:
            raise SourceAttestedSyntheticPdfTruthError("source_layout_reference_text_missing")
        total_distance += _levenshtein_distance(reference_text, candidate_text)
        total_reference_characters += len(reference_text)
        source_cells.update(
            (asset_id, page, table, row, column, text)
            for page, table, row, column, text in _table_cell_set(source.table_cells)
        )
        candidate_cells.update(
            (asset_id, page, table, row, column, text)
            for page, table, row, column, text in _table_cell_set(output.table_cells)
        )
    if total_fields == 0 or total_reference_characters == 0 or not source_cells:
        raise SourceAttestedSyntheticPdfTruthError("source_layout_release_target_missing")
    true_positive = len(source_cells & candidate_cells)
    denominator = len(source_cells) + len(candidate_cells)
    table_f1 = 0.0 if denominator == 0 else (2.0 * true_positive) / denominator
    metrics = SourceAttestedSyntheticOcrMetricsV1(
        field_exact_match=exact_fields / total_fields,
        ocr_cer=min(1.0, total_distance / total_reference_characters),
        table_cell_f1=table_f1,
    )
    passed = (
        metrics.field_exact_match >= 0.99
        and metrics.ocr_cer <= 0.01
        and metrics.table_cell_f1 >= 0.98
    )
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "evaluation_basis": SOURCE_ATTESTED_EVALUATION_BASIS,
        "dataset_id": packet.dataset_id,
        "dataset_revision": packet.dataset_revision,
        "packet_kind": packet.split,
        "packet_receipt_sha256": packet.packet_receipt_sha256,
        "evaluation_scope": SOURCE_ATTESTED_EVALUATION_SCOPE,
        "layout_truth_kind": SOURCE_LAYOUT_TRUTH_KIND,
        "layout_truth_receipt_sha256": layout_truth.layout_truth_receipt_sha256,
        "layout_truth_parser_spec_sha256": layout_truth.parser_spec_sha256,
        "development_selection_receipt_sha256": (
            development_selection.development_selection_receipt_sha256
            if development_selection is not None
            else None
        ),
        "sealed_run_receipt_sha256": (
            sealed_run.sealed_run_receipt_sha256 if sealed_run is not None else None
        ),
        "engine_kind": engine_kind,
        "extractor_revision": extractor_revision,
        "configuration_sha256": configuration_sha256,
        "model_inventory_sha256": model_inventory_sha256,
        "candidate_bundle_sha256": candidate_output.candidate_bundle_sha256,
        "passed": passed,
        "metrics": metrics.model_dump(mode="json", warnings=False),
    }
    payload["evaluation_result_sha256"] = source_attested_evaluation_result_sha256(payload)
    return SourceAttestedSyntheticOcrEvaluationResultV1.model_validate(payload)


def select_best_source_attested_synthetic_development_result(
    results: Sequence[SourceAttestedSyntheticOcrEvaluationResultV1],
) -> SourceAttestedSyntheticOcrEvaluationResultV1:
    """Apply the fixed all-three-engine selection rule to source-attested runs."""

    expected_engines = {"pdf_text_layer", "paddleocr", "docling"}
    if len(results) != len(expected_engines):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_development_result_set_invalid")
    first = results[0] if results else None
    if first is None or first.packet_kind != "development":
        raise SourceAttestedSyntheticPdfTruthError("source_layout_development_result_set_invalid")
    identity = (
        first.dataset_id,
        first.dataset_revision,
        first.packet_receipt_sha256,
        first.layout_truth_receipt_sha256,
    )
    if {result.engine_kind for result in results} != expected_engines or any(
        result.packet_kind != "development"
        or (
            result.dataset_id,
            result.dataset_revision,
            result.packet_receipt_sha256,
            result.layout_truth_receipt_sha256,
        )
        != identity
        for result in results
    ):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_development_result_set_invalid")
    passing = [result for result in results if result.passed]
    if not passing:
        raise SourceAttestedSyntheticPdfTruthError("source_layout_development_no_passing_engine")
    return sorted(
        passing,
        key=lambda result: (
            -result.metrics.field_exact_match,
            result.metrics.ocr_cer,
            -result.metrics.table_cell_f1,
            result.engine_kind,
        ),
    )[0]


def build_source_attested_synthetic_development_selection(
    *, results: Sequence[SourceAttestedSyntheticOcrEvaluationResultV1]
) -> SourceAttestedSyntheticDevelopmentSelectionV1:
    """Bind one deterministic winner to all three immutable development runs."""

    selected = select_best_source_attested_synthetic_development_result(results)
    identity = (
        selected.dataset_id,
        selected.dataset_revision,
        selected.packet_receipt_sha256,
        selected.layout_truth_receipt_sha256,
        selected.layout_truth_parser_spec_sha256,
    )
    if any(
        (
            item.evaluation_basis,
            item.evaluation_scope,
            item.layout_truth_kind,
            item.dataset_id,
            item.dataset_revision,
            item.packet_receipt_sha256,
            item.layout_truth_receipt_sha256,
            item.layout_truth_parser_spec_sha256,
        )
        != (
            SOURCE_ATTESTED_EVALUATION_BASIS,
            SOURCE_ATTESTED_EVALUATION_SCOPE,
            SOURCE_LAYOUT_TRUTH_KIND,
            *identity,
        )
        for item in results
    ):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_development_result_set_invalid")
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "artifact_kind": "source_attested_synthetic_pdf_development_selection",
        "evaluation_basis": SOURCE_ATTESTED_EVALUATION_BASIS,
        "evaluation_scope": SOURCE_ATTESTED_EVALUATION_SCOPE,
        "layout_truth_kind": SOURCE_LAYOUT_TRUTH_KIND,
        "dataset_id": selected.dataset_id,
        "dataset_revision": selected.dataset_revision,
        "development_packet_receipt_sha256": selected.packet_receipt_sha256,
        "layout_truth_receipt_sha256": selected.layout_truth_receipt_sha256,
        "layout_truth_parser_spec_sha256": selected.layout_truth_parser_spec_sha256,
        "candidate_results": [
            {
                "engine_kind": item.engine_kind,
                "extractor_revision": item.extractor_revision,
                "configuration_sha256": item.configuration_sha256,
                "model_inventory_sha256": item.model_inventory_sha256,
                "candidate_bundle_sha256": item.candidate_bundle_sha256,
                "evaluation_result_sha256": item.evaluation_result_sha256,
                "passed": item.passed,
            }
            for item in sorted(results, key=lambda item: item.engine_kind)
        ],
        "selected_engine_kind": selected.engine_kind,
        "selected_extractor_revision": selected.extractor_revision,
        "selected_configuration_sha256": selected.configuration_sha256,
        "selected_model_inventory_sha256": selected.model_inventory_sha256,
        "selected_evaluation_result_sha256": selected.evaluation_result_sha256,
    }
    payload["development_selection_receipt_sha256"] = development_selection_receipt_sha256(payload)
    return SourceAttestedSyntheticDevelopmentSelectionV1.model_validate(payload)


def _is_local_data_path(path: Path) -> bool:
    parts = [part.casefold() for part in path.parts]
    return any(parts[index : index + 2] == ["_local", "data"] for index in range(len(parts) - 1))


def _local_output_directory(path: Path, *, code: str) -> Path:
    try:
        resolved = path.resolve()
        if not _is_local_data_path(resolved):
            raise SourceAttestedSyntheticPdfTruthError(code)
        resolved.mkdir(parents=True, exist_ok=True)
        if not resolved.is_dir() or resolved.is_symlink():
            raise SourceAttestedSyntheticPdfTruthError(code)
    except SourceAttestedSyntheticPdfTruthError:
        raise
    except OSError as error:
        raise SourceAttestedSyntheticPdfTruthError(code) from error
    return resolved


def _local_input_directory(path: Path, *, code: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
        if not _is_local_data_path(resolved) or not resolved.is_dir() or resolved.is_symlink():
            raise SourceAttestedSyntheticPdfTruthError(code)
    except SourceAttestedSyntheticPdfTruthError:
        raise
    except OSError as error:
        raise SourceAttestedSyntheticPdfTruthError(code) from error
    return resolved


def _local_input_file(path: Path, *, code: str) -> Path:
    try:
        resolved = path.resolve(strict=True)
        if not _is_local_data_path(resolved) or not resolved.is_file() or resolved.is_symlink():
            raise SourceAttestedSyntheticPdfTruthError(code)
    except SourceAttestedSyntheticPdfTruthError:
        raise
    except OSError as error:
        raise SourceAttestedSyntheticPdfTruthError(code) from error
    return resolved


def _write_hash_named_immutable(
    *, directory: Path, digest: str, payload: object, code: str
) -> Path:
    destination = directory / f"{digest}.json"
    encoded = canonical_json_file_bytes(payload)
    try:
        if destination.exists():
            if destination.is_symlink() or destination.read_bytes() != encoded:
                raise SourceAttestedSyntheticPdfTruthError(code)
            return destination
        with destination.open("xb") as stream:
            stream.write(encoded)
            stream.flush()
    except SourceAttestedSyntheticPdfTruthError:
        raise
    except OSError as error:
        raise SourceAttestedSyntheticPdfTruthError(code) from error
    return destination


def write_source_attested_synthetic_dataset_receipt(
    *, receipt: DatasetReceiptV1, output_directory: Path
) -> Path:
    """Mirror the non-content dataset receipt at its deterministic local path."""

    directory = _local_output_directory(
        output_directory, code="source_layout_dataset_receipt_directory_invalid"
    )
    return _write_hash_named_immutable(
        directory=directory,
        digest=receipt_sha256(receipt),
        payload=receipt.model_dump(mode="json", warnings=False),
        code="source_layout_dataset_receipt_immutable_conflict",
    )


def read_canonical_source_attested_synthetic_dataset_receipt(path: Path) -> DatasetReceiptV1:
    try:
        raw = path.read_bytes()
        receipt = DatasetReceiptV1.model_validate_json(raw)
    except (OSError, ValidationError) as error:
        raise SourceAttestedSyntheticPdfTruthError(
            "source_layout_dataset_receipt_invalid"
        ) from error
    if raw != canonical_json_file_bytes(receipt.model_dump(mode="json", warnings=False)):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_dataset_receipt_not_canonical")
    if path.name != f"{receipt_sha256(receipt)}.json":
        raise SourceAttestedSyntheticPdfTruthError("source_layout_dataset_receipt_filename_invalid")
    return receipt


def write_source_attested_synthetic_packet(
    *, packet: EvaluationPacketV2, output_directory: Path
) -> Path:
    """Mirror a selected source-attested packet at its immutable local path."""

    directory = _local_output_directory(
        output_directory, code="source_layout_packet_directory_invalid"
    )
    return _write_hash_named_immutable(
        directory=directory,
        digest=packet.packet_receipt_sha256,
        payload=packet.model_dump(mode="json", warnings=False),
        code="source_layout_packet_immutable_conflict",
    )


def read_canonical_source_attested_synthetic_packet(path: Path) -> EvaluationPacketV2:
    try:
        raw = path.read_bytes()
        packet = EvaluationPacketV2.model_validate_json(raw)
    except (OSError, ValidationError) as error:
        raise SourceAttestedSyntheticPdfTruthError("source_layout_packet_invalid") from error
    if raw != canonical_json_file_bytes(packet.model_dump(mode="json", warnings=False)):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_packet_not_canonical")
    if path.name != f"{packet_receipt_sha256(packet)}.json":
        raise SourceAttestedSyntheticPdfTruthError("source_layout_packet_filename_invalid")
    return packet


def write_source_attested_synthetic_candidate_bundle(
    *, candidate_bundle: SyntheticPdfCandidateOutputBundleV1, output_directory: Path
) -> Path:
    """Store one candidate in a deterministic hash-named local source flow path."""

    directory = _local_output_directory(
        output_directory, code="source_layout_candidate_bundle_directory_invalid"
    )
    return _write_hash_named_immutable(
        directory=directory,
        digest=candidate_bundle.candidate_bundle_sha256,
        payload=candidate_bundle.model_dump(mode="json", warnings=False),
        code="source_layout_candidate_bundle_immutable_conflict",
    )


def read_canonical_source_attested_synthetic_candidate_bundle(
    path: Path,
) -> SyntheticPdfCandidateOutputBundleV1:
    try:
        raw = path.read_bytes()
        candidate_bundle = SyntheticPdfCandidateOutputBundleV1.model_validate_json(raw)
    except (OSError, ValidationError) as error:
        raise SourceAttestedSyntheticPdfTruthError(
            "source_layout_candidate_bundle_invalid"
        ) from error
    if raw != canonical_json_file_bytes(candidate_bundle.model_dump(mode="json", warnings=False)):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_candidate_bundle_not_canonical")
    if path.name != f"{candidate_bundle_sha256(candidate_bundle)}.json":
        raise SourceAttestedSyntheticPdfTruthError(
            "source_layout_candidate_bundle_filename_invalid"
        )
    return candidate_bundle


def write_source_attested_synthetic_layout_truth(
    *, truth: SourceAttestedSyntheticLayoutTruthV1, output_directory: Path
) -> Path:
    directory = _local_output_directory(
        output_directory, code="source_layout_truth_directory_invalid"
    )
    return _write_hash_named_immutable(
        directory=directory,
        digest=truth.layout_truth_receipt_sha256,
        payload=truth.model_dump(mode="json", warnings=False),
        code="source_layout_truth_immutable_conflict",
    )


def read_canonical_source_attested_synthetic_layout_truth(
    path: Path,
) -> SourceAttestedSyntheticLayoutTruthV1:
    try:
        raw = path.read_bytes()
        truth = SourceAttestedSyntheticLayoutTruthV1.model_validate_json(raw)
    except (OSError, ValidationError) as error:
        raise SourceAttestedSyntheticPdfTruthError("source_layout_truth_invalid") from error
    if raw != canonical_json_file_bytes(truth.model_dump(mode="json", warnings=False)):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_truth_not_canonical")
    if path.name != f"{truth.layout_truth_receipt_sha256}.json":
        raise SourceAttestedSyntheticPdfTruthError("source_layout_truth_filename_invalid")
    return truth


def write_source_attested_synthetic_development_selection(
    *, selection: SourceAttestedSyntheticDevelopmentSelectionV1, output_directory: Path
) -> Path:
    directory = _local_output_directory(
        output_directory, code="source_layout_selection_directory_invalid"
    )
    return _write_hash_named_immutable(
        directory=directory,
        digest=selection.development_selection_receipt_sha256,
        payload=selection.model_dump(mode="json", warnings=False),
        code="source_layout_selection_immutable_conflict",
    )


def read_canonical_source_attested_synthetic_development_selection(
    path: Path,
) -> SourceAttestedSyntheticDevelopmentSelectionV1:
    try:
        raw = path.read_bytes()
        selection = SourceAttestedSyntheticDevelopmentSelectionV1.model_validate_json(raw)
    except (OSError, ValidationError) as error:
        raise SourceAttestedSyntheticPdfTruthError("source_layout_selection_invalid") from error
    if raw != canonical_json_file_bytes(selection.model_dump(mode="json", warnings=False)):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_selection_not_canonical")
    if path.name != f"{selection.development_selection_receipt_sha256}.json":
        raise SourceAttestedSyntheticPdfTruthError("source_layout_selection_filename_invalid")
    return selection


class _SealedCandidateReservationV1(_WireModel):
    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["sealed_synthetic_pdf_candidate_reservation"] = (
        "sealed_synthetic_pdf_candidate_reservation"
    )
    profile_mode: Literal["offline_synthetic_pdf_evaluation"] = "offline_synthetic_pdf_evaluation"
    parser_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    dataset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    dataset_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    packet_kind: Literal["sealed"] = "sealed"
    packet_receipt_sha256: str = Field(pattern=_SHA256)
    engine_kind: EngineKind
    extractor_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    configuration_sha256: str = Field(pattern=_SHA256)
    model_inventory_sha256: str = Field(pattern=_SHA256)


def _sealed_candidate_reservation_sha256(
    *,
    document_preview_root: Path,
    packet: EvaluationPacketV2,
    candidate_output: SyntheticPdfCandidateOutputBundleV1,
    engine_kind: EngineKind,
    extractor_revision: str,
    configuration_sha256: str,
    model_inventory_sha256: str,
    parser_revision: str,
) -> str:
    path = (
        document_preview_root
        / "candidate-run-reservations"
        / f"sealed-{packet.packet_receipt_sha256}.json"
    )
    try:
        if path.is_symlink():
            raise SourceAttestedSyntheticPdfTruthError("source_layout_sealed_reservation_invalid")
        raw = path.read_bytes()
        reservation = _SealedCandidateReservationV1.model_validate_json(raw)
    except SourceAttestedSyntheticPdfTruthError:
        raise
    except (OSError, ValidationError) as error:
        raise SourceAttestedSyntheticPdfTruthError(
            "source_layout_sealed_reservation_invalid"
        ) from error
    if raw != canonical_json_file_bytes(reservation.model_dump(mode="json", warnings=False)):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_sealed_reservation_invalid")
    if (
        reservation.dataset_id != packet.dataset_id
        or reservation.dataset_revision != packet.dataset_revision
        or reservation.packet_receipt_sha256 != packet.packet_receipt_sha256
        or reservation.engine_kind != engine_kind
        or reservation.extractor_revision != extractor_revision
        or reservation.configuration_sha256 != configuration_sha256
        or reservation.model_inventory_sha256 != model_inventory_sha256
        or reservation.parser_revision != parser_revision
        or candidate_output.candidate_bundle_sha256 == ""
    ):
        raise SourceAttestedSyntheticPdfTruthError(
            "source_layout_sealed_reservation_binding_mismatch"
        )
    return sha256_bytes(raw)


def _validate_development_selection_for_sealed(
    *,
    selection: SourceAttestedSyntheticDevelopmentSelectionV1,
    packet: EvaluationPacketV2,
    engine_kind: EngineKind,
    extractor_revision: str,
    configuration_sha256: str,
    model_inventory_sha256: str,
) -> None:
    if (
        selection.dataset_id != packet.dataset_id
        or selection.dataset_revision != packet.dataset_revision
        # The selection binds the immutable *development* layout truth; this
        # caller verifies a distinct immutable sealed layout truth.  Equal
        # layout receipts would be impossible because the packets are 60 vs 20
        # assets.  Both still pin the same parser specification revision.
        or selection.layout_truth_parser_spec_sha256 != source_layout_parser_spec_sha256()
        or selection.selected_engine_kind != engine_kind
        or selection.selected_extractor_revision != extractor_revision
        or selection.selected_configuration_sha256 != configuration_sha256
        or selection.selected_model_inventory_sha256 != model_inventory_sha256
    ):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_selection_binding_mismatch")


def build_source_attested_synthetic_sealed_run(
    *,
    packet: EvaluationPacketV2,
    layout_truth: SourceAttestedSyntheticLayoutTruthV1,
    development_selection: SourceAttestedSyntheticDevelopmentSelectionV1,
    candidate_output: SyntheticPdfCandidateOutputBundleV1,
    document_preview_root: Path,
    engine_kind: EngineKind,
    extractor_revision: str,
    configuration_sha256: str,
    model_inventory_sha256: str,
    parser_revision: str,
) -> SourceAttestedSyntheticSealedRunV1:
    if packet.split != "sealed":
        raise SourceAttestedSyntheticPdfTruthError("source_layout_sealed_run_packet_invalid")
    _validate_development_selection_for_sealed(
        selection=development_selection,
        packet=packet,
        engine_kind=engine_kind,
        extractor_revision=extractor_revision,
        configuration_sha256=configuration_sha256,
        model_inventory_sha256=model_inventory_sha256,
    )
    if (
        candidate_output.packet_receipt_sha256 != packet.packet_receipt_sha256
        or candidate_output.engine_kind != engine_kind
        or candidate_output.extractor_revision != extractor_revision
        or candidate_output.configuration_sha256 != configuration_sha256
        or candidate_output.model_inventory_sha256 != model_inventory_sha256
        or candidate_output.parser_revision != parser_revision
    ):
        raise SourceAttestedSyntheticPdfTruthError(
            "source_layout_sealed_candidate_binding_mismatch"
        )
    reservation_sha = _sealed_candidate_reservation_sha256(
        document_preview_root=document_preview_root,
        packet=packet,
        candidate_output=candidate_output,
        engine_kind=engine_kind,
        extractor_revision=extractor_revision,
        configuration_sha256=configuration_sha256,
        model_inventory_sha256=model_inventory_sha256,
        parser_revision=parser_revision,
    )
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "artifact_kind": "source_attested_synthetic_pdf_sealed_run",
        "evaluation_basis": SOURCE_ATTESTED_EVALUATION_BASIS,
        "evaluation_scope": SOURCE_ATTESTED_EVALUATION_SCOPE,
        "layout_truth_kind": SOURCE_LAYOUT_TRUTH_KIND,
        "dataset_id": packet.dataset_id,
        "dataset_revision": packet.dataset_revision,
        "sealed_packet_receipt_sha256": packet.packet_receipt_sha256,
        "layout_truth_receipt_sha256": layout_truth.layout_truth_receipt_sha256,
        "layout_truth_parser_spec_sha256": layout_truth.parser_spec_sha256,
        "development_selection_receipt_sha256": development_selection.development_selection_receipt_sha256,
        "selected_engine_kind": engine_kind,
        "selected_extractor_revision": extractor_revision,
        "selected_configuration_sha256": configuration_sha256,
        "selected_model_inventory_sha256": model_inventory_sha256,
        "selected_development_evaluation_result_sha256": development_selection.selected_evaluation_result_sha256,
        "candidate_bundle_sha256": candidate_output.candidate_bundle_sha256,
        "candidate_parser_revision": parser_revision,
        "sealed_candidate_reservation_sha256": reservation_sha,
    }
    payload["sealed_run_receipt_sha256"] = sealed_run_receipt_sha256(payload)
    return SourceAttestedSyntheticSealedRunV1.model_validate(payload)


class SourceAttestedSyntheticSealedRunStateV1(_WireModel):
    schema_version: Literal["1.0"] = "1.0"
    sealed_packet_receipt_sha256: str = Field(pattern=_SHA256)
    sealed_run_receipt_sha256: str = Field(pattern=_SHA256)


def write_source_attested_synthetic_sealed_run(
    *, sealed_run: SourceAttestedSyntheticSealedRunV1, output_directory: Path
) -> Path:
    """Create the sealed evaluator ledger before scoring; failure stays consumed."""

    directory = _local_output_directory(
        output_directory, code="source_layout_sealed_run_directory_invalid"
    )
    state_path = directory / f"sealed-run-state-{sealed_run.sealed_packet_receipt_sha256}.json"
    state = SourceAttestedSyntheticSealedRunStateV1(
        sealed_packet_receipt_sha256=sealed_run.sealed_packet_receipt_sha256,
        sealed_run_receipt_sha256=sealed_run.sealed_run_receipt_sha256,
    )
    try:
        with state_path.open("xb") as stream:
            stream.write(canonical_json_file_bytes(state.model_dump(mode="json", warnings=False)))
            stream.flush()
    except FileExistsError as error:
        raise SourceAttestedSyntheticPdfTruthError("source_layout_sealed_run_consumed") from error
    except OSError as error:
        raise SourceAttestedSyntheticPdfTruthError(
            "source_layout_sealed_run_write_failed"
        ) from error
    return _write_hash_named_immutable(
        directory=directory,
        digest=sealed_run.sealed_run_receipt_sha256,
        payload=sealed_run.model_dump(mode="json", warnings=False),
        code="source_layout_sealed_run_immutable_conflict",
    )


def read_canonical_source_attested_synthetic_sealed_run(
    path: Path,
) -> SourceAttestedSyntheticSealedRunV1:
    try:
        raw = path.read_bytes()
        sealed_run = SourceAttestedSyntheticSealedRunV1.model_validate_json(raw)
    except (OSError, ValidationError) as error:
        raise SourceAttestedSyntheticPdfTruthError("source_layout_sealed_run_invalid") from error
    if raw != canonical_json_file_bytes(sealed_run.model_dump(mode="json", warnings=False)):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_sealed_run_not_canonical")
    if path.name != f"{sealed_run.sealed_run_receipt_sha256}.json":
        raise SourceAttestedSyntheticPdfTruthError("source_layout_sealed_run_filename_invalid")
    return sealed_run


def _validate_sealed_run_binding(
    *,
    packet: EvaluationPacketV2,
    layout_truth: SourceAttestedSyntheticLayoutTruthV1,
    candidate_output: SyntheticPdfCandidateOutputBundleV1,
    engine_kind: EngineKind,
    extractor_revision: str,
    configuration_sha256: str,
    model_inventory_sha256: str,
    parser_revision: str,
    development_selection: SourceAttestedSyntheticDevelopmentSelectionV1,
    sealed_run: SourceAttestedSyntheticSealedRunV1,
) -> None:
    _validate_development_selection_for_sealed(
        selection=development_selection,
        packet=packet,
        engine_kind=engine_kind,
        extractor_revision=extractor_revision,
        configuration_sha256=configuration_sha256,
        model_inventory_sha256=model_inventory_sha256,
    )
    if (
        sealed_run.dataset_id != packet.dataset_id
        or sealed_run.dataset_revision != packet.dataset_revision
        or sealed_run.sealed_packet_receipt_sha256 != packet.packet_receipt_sha256
        or sealed_run.layout_truth_receipt_sha256 != layout_truth.layout_truth_receipt_sha256
        or sealed_run.layout_truth_parser_spec_sha256 != layout_truth.parser_spec_sha256
        or sealed_run.development_selection_receipt_sha256
        != development_selection.development_selection_receipt_sha256
        or sealed_run.selected_engine_kind != engine_kind
        or sealed_run.selected_extractor_revision != extractor_revision
        or sealed_run.selected_configuration_sha256 != configuration_sha256
        or sealed_run.selected_model_inventory_sha256 != model_inventory_sha256
        or sealed_run.selected_development_evaluation_result_sha256
        != development_selection.selected_evaluation_result_sha256
        or sealed_run.candidate_bundle_sha256 != candidate_output.candidate_bundle_sha256
        or sealed_run.candidate_parser_revision != parser_revision
    ):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_sealed_run_binding_mismatch")


class SourceAttestedSealedEvaluationStateV1(_WireModel):
    schema_version: Literal["1.0"] = "1.0"
    packet_kind: Literal["sealed"] = "sealed"
    packet_receipt_sha256: str = Field(pattern=_SHA256)
    layout_truth_receipt_sha256: str = Field(pattern=_SHA256)
    evaluation_result_sha256: str = Field(pattern=_SHA256)


def write_source_attested_synthetic_evaluation_result(
    *, result: SourceAttestedSyntheticOcrEvaluationResultV1, output_directory: Path
) -> Path:
    """Write immutable results and reserve a sealed source-attested packet once."""

    resolved = _local_output_directory(
        output_directory, code="source_layout_result_directory_invalid"
    )
    if result.packet_kind == "sealed":
        state = SourceAttestedSealedEvaluationStateV1(
            packet_receipt_sha256=result.packet_receipt_sha256,
            layout_truth_receipt_sha256=result.layout_truth_receipt_sha256,
            evaluation_result_sha256=result.evaluation_result_sha256,
        )
        state_path = resolved / f"sealed-evaluation-state-{result.packet_receipt_sha256}.json"
        try:
            with state_path.open("xb") as stream:
                stream.write(
                    canonical_json_file_bytes(state.model_dump(mode="json", warnings=False))
                )
                stream.flush()
        except FileExistsError as error:
            raise SourceAttestedSyntheticPdfTruthError(
                "source_layout_sealed_evaluation_consumed"
            ) from error
        except OSError as error:
            raise SourceAttestedSyntheticPdfTruthError(
                "source_layout_result_write_failed"
            ) from error
    return _write_hash_named_immutable(
        directory=resolved,
        digest=result.evaluation_result_sha256,
        payload=result.model_dump(mode="json", warnings=False),
        code="source_layout_result_immutable_conflict",
    )


def read_canonical_source_attested_synthetic_evaluation_result(
    path: Path,
) -> SourceAttestedSyntheticOcrEvaluationResultV1:
    try:
        raw = path.read_bytes()
        result = SourceAttestedSyntheticOcrEvaluationResultV1.model_validate_json(raw)
    except (OSError, ValidationError) as error:
        raise SourceAttestedSyntheticPdfTruthError("source_layout_result_invalid") from error
    if raw != canonical_json_file_bytes(result.model_dump(mode="json", warnings=False)):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_result_not_canonical")
    if path.name != f"{result.evaluation_result_sha256}.json":
        raise SourceAttestedSyntheticPdfTruthError("source_layout_result_filename_invalid")
    return result


__all__ = [
    "SOURCE_ATTESTED_EVALUATION_BASIS",
    "SOURCE_ATTESTED_EVALUATION_SCOPE",
    "SOURCE_LAYOUT_COMPILER_REVISION",
    "SOURCE_LAYOUT_PARSER_SPEC_REVISION",
    "SOURCE_LAYOUT_TRUTH_KIND",
    "SourceAttestedSyntheticGoldRecordV1",
    "SourceAttestedSyntheticLayoutTruthV1",
    "SourceAttestedSyntheticDevelopmentSelectionV1",
    "SourceAttestedSyntheticOcrEvaluationResultV1",
    "SourceAttestedSyntheticOcrMetricsV1",
    "SourceAttestedSyntheticPdfTruthError",
    "SourceAttestedSyntheticSealedRunV1",
    "build_source_attested_synthetic_development_selection",
    "build_source_attested_synthetic_sealed_run",
    "compile_source_attested_synthetic_pdf_layout_truth",
    "development_selection_receipt_sha256",
    "layout_truth_receipt_sha256",
    "read_canonical_source_attested_synthetic_development_selection",
    "read_canonical_source_attested_synthetic_candidate_bundle",
    "read_canonical_source_attested_synthetic_dataset_receipt",
    "read_canonical_source_attested_synthetic_evaluation_result",
    "read_canonical_source_attested_synthetic_layout_truth",
    "read_canonical_source_attested_synthetic_packet",
    "read_canonical_source_attested_synthetic_sealed_run",
    "score_source_attested_synthetic_pdf_candidate_output",
    "sealed_run_receipt_sha256",
    "select_best_source_attested_synthetic_development_result",
    "source_attested_evaluation_result_sha256",
    "source_layout_parser_spec_sha256",
    "validate_source_attested_layout_truth_for_evaluation",
    "write_source_attested_synthetic_development_selection",
    "write_source_attested_synthetic_candidate_bundle",
    "write_source_attested_synthetic_dataset_receipt",
    "write_source_attested_synthetic_evaluation_result",
    "write_source_attested_synthetic_layout_truth",
    "write_source_attested_synthetic_packet",
    "write_source_attested_synthetic_sealed_run",
]
