"""Local-only source-attested layout truth for the synthetic PDF fixture.

This module deliberately does *not* turn the older human-review Gold into an
automatic approval.  The synthetic PDF generator is not available in this
workspace, so the final, born-digital PDF content stream is the only visual
source that can be parsed without guessing.  A small, pinned parser accepts
only the one observed fixture profile.  Unsupported syntax, a changed template,
or an ambiguous table fails closed.

The resulting evaluation is named
``synthetic_r1_supervised_native_template_compatibility``.  It is useful for
comparing the explicitly supervised native-template extractor on this exact
synthetic template, but it is neither an independent optical-OCR certification
nor evidence about real trade documents.
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
from customs_ai_agent.evaluation.supervised_template_contract import (
    SUPERVISED_TEMPLATE_EXTRACTOR_REVISION,
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
_SYNTHETIC_PDF_RECEIPT_PATH_RE = re.compile(
    r"^(?P<fixture>clean_sets|error_sets)/set_(?P<set_number>\d{4})/"
    r"(?P<record>[CE]\d{4})_(?P<document>BL|INV|PL)\.pdf$"
)
_DOCUMENT_FILENAME_SUFFIX: dict[SyntheticDocumentType, str] = {
    "bill_of_lading": "BL",
    "invoice": "INV",
    "packing_list": "PL",
}

SOURCE_ATTESTED_EVALUATION_BASIS: Final[Literal["source_attested_synthetic"]] = (
    "source_attested_synthetic"
)
SOURCE_ATTESTED_EVALUATION_SCOPE: Final[
    Literal["synthetic_r1_supervised_native_template_compatibility"]
] = "synthetic_r1_supervised_native_template_compatibility"
SOURCE_LAYOUT_TRUTH_KIND: Final[Literal["pdf_content_stream_layout_v1"]] = (
    "pdf_content_stream_layout_v1"
)
SOURCE_LAYOUT_COMPILER_REVISION: Final[
    Literal["synthetic-pdf-content-stream-layout-compiler-v2"]
] = "synthetic-pdf-content-stream-layout-compiler-v2"
SOURCE_LAYOUT_PARSER_SPEC_REVISION: Final[
    Literal["synthetic-pdf-content-stream-parser-spec-v2"]
] = "synthetic-pdf-content-stream-parser-spec-v2"
# This is intentionally fixed at the source-truth writer as well as at the
# runtime boundary.  Merely requiring every receipt in a chain to agree would
# let a coherent, older parser version create a new automatic approval.
# Version 11 is the only normalizer accepted for this scope.  It is the
# separately reviewed supervised-native-template contract for the complete
# rendered B/L-13, Invoice-15, and Packing-list-12 field surface.  Version 10
# and every older revision must not regain eligibility merely by producing a
# self-consistent chain of newer receipts.
SOURCE_ATTESTED_CANDIDATE_PARSER_REVISION: Final[
    Literal["synthetic-pdf-supervised-template-normalizer-v11"]
] = "synthetic-pdf-supervised-template-normalizer-v11"
# The actual r1 development packet is not an engine bake-off.  It validates
# one fixed, family-safe supervised native-PDF profile.  Keeping this identity
# beside the source-truth scope makes an older text-layer extractor or an
# unrelated OCR engine ineligible even if it can mint a self-consistent result
# chain.
SOURCE_ATTESTED_SUPERVISED_TEMPLATE_ENGINE_KIND: Final[Literal["pdf_text_layer"]] = "pdf_text_layer"
SOURCE_ATTESTED_SUPERVISED_TEMPLATE_EXTRACTOR_REVISION: Final[
    Literal["pypdf-5.3.1-supervised-template-r1-v1"]
] = SUPERVISED_TEMPLATE_EXTRACTOR_REVISION

# This automatic path has one deliberately narrow approval target.  These
# values are a compact identity for the existing human-checked, 2,000-set PDF
# fixture; they contain no PDF, field, or OCR text.  A different fixture must
# receive a separately reviewed scope constant and parser revision instead of
# silently becoming eligible for this approval.
SOURCE_ATTESTED_SYNTHETIC_DATASET_ID: Final = "synthetic-customs-pdf-2000"
SOURCE_ATTESTED_SYNTHETIC_DATASET_REVISION: Final = "20260811-r1"
SOURCE_ATTESTED_SYNTHETIC_DATASET_RECEIPT_SHA256: Final = (
    "aa3c828f4fdf4afc6a02171b271c71f314eefb363e6676c65625bae803e1b39d"
)
SOURCE_ATTESTED_SYNTHETIC_TOTAL_PDF_ASSETS: Final = 6_000
SOURCE_ATTESTED_SYNTHETIC_CLEAN_PDF_ASSETS: Final = 3_000
SOURCE_ATTESTED_SYNTHETIC_ERROR_PDF_ASSETS: Final = 3_000
SOURCE_ATTESTED_SYNTHETIC_SETS_PER_FIXTURE_KIND: Final = 1_000
SOURCE_ATTESTED_SYNTHETIC_ERROR_LABELS_SHA256: Final = (
    "10bda04c08a27c89893d4101670991f66b9b37b02df2dbeea8ef1e92ffb9d92f"
)
SOURCE_ATTESTED_SYNTHETIC_DEVELOPMENT_PACKET_RECEIPT_SHA256: Final = (
    "5b8a804b022e97df0e58a60c4fadfc9d3637c900f368273b8c1b6ff347c3f7e2"
)
SOURCE_ATTESTED_SYNTHETIC_SEALED_PACKET_RECEIPT_SHA256: Final = (
    "174d7fad0040e89ee93d367f31694183049ea649a28515cde7a59458261aa271"
)


@dataclass(frozen=True, slots=True)
class SourceAttestedSyntheticDatasetScopeV1:
    """One exact synthetic-PDF corpus accepted by the automatic path.

    The injectable form exists solely for hermetic unit tests.  Every shipped
    CLI and runtime-facing verifier uses ``EXPECTED_SOURCE_ATTESTED...`` by
    default, so a caller cannot opt a different dataset into the production
    automatic pointer through a command-line option.
    """

    dataset_id: str
    dataset_revision: str
    dataset_receipt_sha256: str
    total_pdf_assets: int
    clean_pdf_assets: int
    error_pdf_assets: int
    sets_per_fixture_kind: int
    require_complete_canonical_inventory: bool = True
    require_canonical_packets: bool = True
    error_labels_sha256: str | None = None
    development_packet_receipt_sha256: str | None = None
    sealed_packet_receipt_sha256: str | None = None


EXPECTED_SOURCE_ATTESTED_SYNTHETIC_DATASET_SCOPE: Final = SourceAttestedSyntheticDatasetScopeV1(
    dataset_id=SOURCE_ATTESTED_SYNTHETIC_DATASET_ID,
    dataset_revision=SOURCE_ATTESTED_SYNTHETIC_DATASET_REVISION,
    dataset_receipt_sha256=SOURCE_ATTESTED_SYNTHETIC_DATASET_RECEIPT_SHA256,
    total_pdf_assets=SOURCE_ATTESTED_SYNTHETIC_TOTAL_PDF_ASSETS,
    clean_pdf_assets=SOURCE_ATTESTED_SYNTHETIC_CLEAN_PDF_ASSETS,
    error_pdf_assets=SOURCE_ATTESTED_SYNTHETIC_ERROR_PDF_ASSETS,
    sets_per_fixture_kind=SOURCE_ATTESTED_SYNTHETIC_SETS_PER_FIXTURE_KIND,
    error_labels_sha256=SOURCE_ATTESTED_SYNTHETIC_ERROR_LABELS_SHA256,
    development_packet_receipt_sha256=(SOURCE_ATTESTED_SYNTHETIC_DEVELOPMENT_PACKET_RECEIPT_SHA256),
    sealed_packet_receipt_sha256=SOURCE_ATTESTED_SYNTHETIC_SEALED_PACKET_RECEIPT_SHA256,
)


class SourceAttestedSyntheticPdfTruthError(ValueError):
    """A content-free error boundary for local source-truth processing."""


def _require_source_attested_candidate_parser_revision(parser_revision: str) -> None:
    if parser_revision != SOURCE_ATTESTED_CANDIDATE_PARSER_REVISION:
        raise SourceAttestedSyntheticPdfTruthError(
            "source_layout_candidate_parser_revision_invalid"
        )


def _is_canonical_r1_development_packet(
    *, dataset_id: str, dataset_revision: str, packet_receipt_sha256: str
) -> bool:
    """Whether a selection is for the one production r1 development packet.

    The source-truth schemas remain injectable for hermetic tests.  Production
    callers separately pin this exact corpus and packet, so only this identity
    receives the single-profile rule below; synthetic unit fixtures retain the
    historical three-engine exercise without becoming production-eligible.
    """

    return (
        dataset_id == SOURCE_ATTESTED_SYNTHETIC_DATASET_ID
        and dataset_revision == SOURCE_ATTESTED_SYNTHETIC_DATASET_REVISION
        and packet_receipt_sha256 == SOURCE_ATTESTED_SYNTHETIC_DEVELOPMENT_PACKET_RECEIPT_SHA256
    )


def _is_fixed_supervised_template_profile(
    *, engine_kind: EngineKind, extractor_revision: str
) -> bool:
    return (
        engine_kind == SOURCE_ATTESTED_SUPERVISED_TEMPLATE_ENGINE_KIND
        and extractor_revision == SOURCE_ATTESTED_SUPERVISED_TEMPLATE_EXTRACTOR_REVISION
    )


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


SourceLayoutValueTransform = Literal[
    "raw-v1",
    "single-numeric-token-v1",
    "single-alpha-token-v1",
    "join-slash-v1",
]


@dataclass(frozen=True, slots=True)
class _FieldRunAnchor:
    """One fixed run, or one fixed table-column range, in the pinned PDF."""

    reading_order: int
    y: float
    x_values: tuple[float, ...] = ()
    x_range: tuple[float, float] | None = None


@dataclass(frozen=True, slots=True)
class _FieldSlot:
    field_name: str
    normalizer: Literal["text-v1", "numeric-v1", "identifier-v1"]
    source_runs: tuple[_FieldRunAnchor, ...]
    value_transform: SourceLayoutValueTransform = "raw-v1"


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


# The automatic field gate covers the fields actually rendered by each final
# synthetic PDF template.  A visible table cell alone is not a substitute for
# a structured field: every entry below has a direct, pinned content-stream
# mapping.  Packing List deliberately has 12 fields: its final template has no
# semantic ``containers`` label/value slot.  The older manifest's non-rendered
# ``containers`` column is not inferred from a carton value, a B/L sibling, or
# CSV ground truth.  Missing mappings fail closed rather than falling back to
# clean_manifest.csv or error labels.
SOURCE_ATTESTED_REQUIRED_FIELDS: Final[
    Mapping[
        SyntheticDocumentType,
        tuple[tuple[str, Literal["text-v1", "numeric-v1", "identifier-v1"]], ...],
    ]
] = {
    "bill_of_lading": (
        ("exporter_name", "text-v1"),
        ("consignee", "text-v1"),
        ("load_port", "text-v1"),
        ("disch_port", "text-v1"),
        ("final_dest", "text-v1"),
        ("vessel", "text-v1"),
        ("voyage", "identifier-v1"),
        ("bl_no", "identifier-v1"),
        ("container_no", "identifier-v1"),
        ("seal_no", "identifier-v1"),
        ("good", "text-v1"),
        ("gross_weight", "numeric-v1"),
        ("containers", "numeric-v1"),
    ),
    "invoice": (
        ("exporter_name", "text-v1"),
        ("importer_name", "text-v1"),
        ("consignee", "text-v1"),
        ("load_port", "text-v1"),
        ("disch_port", "text-v1"),
        ("vessel", "text-v1"),
        ("inv_no", "identifier-v1"),
        ("inv_date", "identifier-v1"),
        ("lc_no", "identifier-v1"),
        ("good", "text-v1"),
        ("unit", "text-v1"),
        ("qty", "numeric-v1"),
        ("unit_price", "numeric-v1"),
        ("amount", "numeric-v1"),
        ("total", "numeric-v1"),
    ),
    "packing_list": (
        ("exporter_name", "text-v1"),
        ("consignee", "text-v1"),
        ("inv_no", "identifier-v1"),
        ("lc_no", "identifier-v1"),
        ("good", "text-v1"),
        ("unit", "text-v1"),
        ("qty", "numeric-v1"),
        ("cartons", "numeric-v1"),
        ("net_weight", "numeric-v1"),
        ("gross_weight", "numeric-v1"),
        ("cbm", "numeric-v1"),
        ("net_gross_weight", "text-v1"),
    ),
}


def _fixed_field_run(reading_order: int, x: float, y: float) -> _FieldRunAnchor:
    return _FieldRunAnchor(reading_order=reading_order, y=y, x_values=(x,))


def _variable_field_run(
    reading_order: int, x_min: float, x_max: float, y: float
) -> _FieldRunAnchor:
    return _FieldRunAnchor(
        reading_order=reading_order,
        y=y,
        x_range=(x_min, x_max),
    )


def _field_slot(
    field_name: str,
    normalizer: Literal["text-v1", "numeric-v1", "identifier-v1"],
    *source_runs: _FieldRunAnchor,
    value_transform: SourceLayoutValueTransform = "raw-v1",
) -> _FieldSlot:
    if not source_runs:
        raise ValueError("source-layout field slot requires a source run")
    return _FieldSlot(
        field_name=field_name,
        normalizer=normalizer,
        source_runs=source_runs,
        value_transform=value_transform,
    )


# ``error_labels.csv`` has a human-facing field caption.  The compiler never
# copies either CSV value into source truth; this small fixed mapping is used
# only to prove that a labelled error mutation is actually visible in the
# immutable error PDF compared with its clean sibling.
_SOURCE_ERROR_LABEL_FIELDS: Final[dict[tuple[SyntheticDocumentType, str], str]] = {
    ("bill_of_lading", "Container No."): "container_no",
    ("bill_of_lading", "Gross Weight"): "gross_weight",
    ("invoice", "Amount"): "amount",
    ("invoice", "Port of Discharge"): "disch_port",
    ("invoice", "TOTAL"): "total",
    ("invoice", "Vessel/Flight"): "vessel",
    ("packing_list", "Goods Description"): "good",
    ("packing_list", "Net/Gross Weight"): "net_gross_weight",
    ("packing_list", "No. of Cartons / PKGS"): "cartons",
    ("packing_list", "No. of Invoice"): "inv_no",
    ("packing_list", "No. of L/C"): "lc_no",
    ("packing_list", "Quantity"): "qty",
    ("packing_list", "To Applicant / Consignee"): "consignee",
}

# Error labels are an independent mutation assertion, not a source of Gold
# values.  Some rendered table cells express a numeric value with a document
# unit while the label expresses the same quantity with another fixed unit
# caption.  These projections verify the visible number(s) only for that
# assertion; the returned Gold field remains the direct PDF value above.
_SOURCE_ERROR_LABEL_PROJECTIONS: Final[
    dict[
        tuple[SyntheticDocumentType, str], Literal["normalized-v1", "numeric-v1", "numeric-pair-v1"]
    ]
] = {
    ("bill_of_lading", "gross_weight"): "numeric-v1",
    ("invoice", "amount"): "numeric-v1",
    ("invoice", "total"): "numeric-v1",
    ("packing_list", "cartons"): "numeric-v1",
    ("packing_list", "qty"): "numeric-v1",
    ("packing_list", "net_gross_weight"): "numeric-pair-v1",
}


# These are positions and field identifiers, not copied document text.  The
# positions were profiled from the final fixture's stable content-stream shape.
# Each input is required to retain that exact shape; an altered template is not
# silently interpreted as this version.
_TEMPLATE_PROFILES: dict[SyntheticDocumentType, _TemplateProfile] = {
    "bill_of_lading": _TemplateProfile(
        text_run_count=90,
        rectangle_count=42,
        field_slots=(
            _field_slot("exporter_name", "text-v1", _fixed_field_run(4, 44.0, 755.8898)),
            _field_slot("consignee", "text-v1", _fixed_field_run(11, 44.0, 709.8898)),
            _field_slot("load_port", "text-v1", _fixed_field_run(31, 44.0, 585.8898)),
            _field_slot("disch_port", "text-v1", _fixed_field_run(34, 172.8189, 585.8898)),
            _field_slot("final_dest", "text-v1", _fixed_field_run(40, 430.4567, 585.8898)),
            _field_slot("vessel", "text-v1", _fixed_field_run(22, 44.0, 612.8898)),
            _field_slot("voyage", "identifier-v1", _fixed_field_run(25, 184.0, 612.8898)),
            _field_slot("bl_no", "identifier-v1", _fixed_field_run(8, 324.0, 755.8898)),
            _field_slot("container_no", "identifier-v1", _fixed_field_run(54, 43.0, 537.8898)),
            _field_slot("seal_no", "identifier-v1", _fixed_field_run(55, 121.0, 537.8898)),
            _field_slot("good", "text-v1", _fixed_field_run(61, 278.0, 537.8898)),
            _field_slot(
                "gross_weight",
                "numeric-v1",
                _variable_field_run(63, 400.0, 485.0, 537.8898),
            ),
            _field_slot(
                "containers",
                "numeric-v1",
                _variable_field_run(59, 190.0, 275.0, 537.8898),
            ),
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
            _field_slot("exporter_name", "text-v1", _fixed_field_run(3, 49.0, 743.8898)),
            _field_slot("importer_name", "text-v1", _fixed_field_run(15, 49.0, 609.8898)),
            _field_slot("consignee", "text-v1", _fixed_field_run(9, 49.0, 673.8898)),
            _field_slot("load_port", "text-v1", _fixed_field_run(17, 49.0, 572.8898)),
            _field_slot("disch_port", "text-v1", _fixed_field_run(19, 175.3189, 572.8898)),
            _field_slot("vessel", "text-v1", _fixed_field_run(24, 49.0, 550.8898)),
            _field_slot("inv_no", "identifier-v1", _fixed_field_run(6, 301.6378, 743.8898)),
            _field_slot("inv_date", "identifier-v1", _fixed_field_run(7, 301.6378, 734.3898)),
            _field_slot("lc_no", "identifier-v1", _fixed_field_run(12, 301.6378, 673.8898)),
            _field_slot("good", "text-v1", _fixed_field_run(35, 153.0, 513.8898)),
            _field_slot(
                "unit",
                "text-v1",
                _variable_field_run(36, 320.0, 400.0, 513.8898),
                value_transform="single-alpha-token-v1",
            ),
            _field_slot(
                "qty",
                "numeric-v1",
                _variable_field_run(36, 320.0, 400.0, 513.8898),
                value_transform="single-numeric-token-v1",
            ),
            _field_slot(
                "unit_price",
                "numeric-v1",
                _variable_field_run(37, 400.0, 480.0, 513.8898),
            ),
            _field_slot(
                "amount",
                "numeric-v1",
                _variable_field_run(38, 480.0, 550.27559, 513.8898),
            ),
            _field_slot(
                "total",
                "numeric-v1",
                _variable_field_run(40, 480.0, 550.27559, 362.8898),
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
            _field_slot("exporter_name", "text-v1", _fixed_field_run(3, 49.0, 753.8898)),
            _field_slot("consignee", "text-v1", _fixed_field_run(12, 49.0, 717.8898)),
            _field_slot("inv_no", "identifier-v1", _fixed_field_run(6, 324.0, 753.8898)),
            _field_slot("lc_no", "identifier-v1", _fixed_field_run(9, 324.0, 717.8898)),
            _field_slot("good", "text-v1", _fixed_field_run(42, 178.0, 563.8898)),
            _field_slot(
                "unit",
                "text-v1",
                _fixed_field_run(43, 178.0, 554.3898),
                value_transform="single-alpha-token-v1",
            ),
            _field_slot(
                "qty",
                "numeric-v1",
                _fixed_field_run(43, 178.0, 554.3898),
                value_transform="single-numeric-token-v1",
            ),
            _field_slot("cartons", "numeric-v1", _fixed_field_run(41, 48.0, 535.3898)),
            _field_slot(
                "net_weight",
                "numeric-v1",
                _variable_field_run(44, 315.0, 390.0, 563.8898),
            ),
            _field_slot(
                "gross_weight",
                "numeric-v1",
                _variable_field_run(45, 390.0, 465.0, 563.8898),
            ),
            _field_slot(
                "cbm",
                "numeric-v1",
                _variable_field_run(46, 465.0, 550.27559, 563.8898),
            ),
            _field_slot(
                "net_gross_weight",
                "text-v1",
                _variable_field_run(44, 315.0, 390.0, 563.8898),
                _variable_field_run(45, 390.0, 465.0, 563.8898),
                value_transform="join-slash-v1",
            ),
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
        # Keep the release field universe in the parser-spec receipt as well
        # as in the compiler check.  Otherwise a future edit could weaken the
        # required-field gate without invalidating an already-recorded layout
        # truth or result.
        "required_fields": {
            document_type: [
                {"field_name": field_name, "normalizer": normalizer}
                for field_name, normalizer in fields
            ]
            for document_type, fields in sorted(SOURCE_ATTESTED_REQUIRED_FIELDS.items())
        },
        "profiles": {
            document_type: {
                "text_run_count": profile.text_run_count,
                "rectangle_count": profile.rectangle_count,
                "field_slots": [
                    {
                        "field_name": slot.field_name,
                        "normalizer": slot.normalizer,
                        "value_transform": slot.value_transform,
                        "source_runs": [
                            {
                                "reading_order": anchor.reading_order,
                                "x_values": list(anchor.x_values),
                                "x_range": (
                                    list(anchor.x_range) if anchor.x_range is not None else None
                                ),
                                "y": anchor.y,
                            }
                            for anchor in slot.source_runs
                        ],
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
    evaluation_scope: Literal["synthetic_r1_supervised_native_template_compatibility"] = (
        SOURCE_ATTESTED_EVALUATION_SCOPE
    )
    layout_truth_kind: Literal["pdf_content_stream_layout_v1"] = SOURCE_LAYOUT_TRUTH_KIND
    compiler_revision: Literal["synthetic-pdf-content-stream-layout-compiler-v2"] = (
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


def validate_source_attested_synthetic_dataset_scope(
    *,
    receipt: DatasetReceiptV1,
    expected_scope: SourceAttestedSyntheticDatasetScopeV1 = (
        EXPECTED_SOURCE_ATTESTED_SYNTHETIC_DATASET_SCOPE
    ),
) -> None:
    """Require the one approved 2,000-set corpus before automatic scoring.

    This is intentionally stricter than a matching ``dataset_id``.  A small
    hand-made 60+20 packet could otherwise satisfy the evaluator's strata
    checks and be mistaken for evidence about the existing 6,000-PDF corpus.
    No source content is returned or logged by this validator.
    """

    if (
        receipt.dataset_id != expected_scope.dataset_id
        or receipt.dataset_revision != expected_scope.dataset_revision
        or receipt_sha256(receipt) != expected_scope.dataset_receipt_sha256
        or len(receipt.assets) != expected_scope.total_pdf_assets
    ):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_dataset_scope_invalid")

    asset_ids = [asset.asset_id for asset in receipt.assets]
    relative_paths = [asset.relative_path for asset in receipt.assets]
    if len(asset_ids) != len(set(asset_ids)) or len(relative_paths) != len(set(relative_paths)):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_dataset_scope_invalid")

    clean_count = 0
    error_count = 0
    for asset in receipt.assets:
        matched = _SYNTHETIC_PDF_RECEIPT_PATH_RE.fullmatch(asset.relative_path)
        if (
            asset.media_type != "application/pdf"
            or asset.source_role != "preview_candidate"
            or matched is None
        ):
            raise SourceAttestedSyntheticPdfTruthError("source_layout_dataset_scope_invalid")
        fixture = matched.group("fixture")
        record = matched.group("record")
        set_number = int(matched.group("set_number"))
        expected_prefix = "C" if fixture == "clean_sets" else "E"
        if (
            not 1 <= set_number <= expected_scope.sets_per_fixture_kind
            or record != f"{expected_prefix}{set_number:04d}"
        ):
            raise SourceAttestedSyntheticPdfTruthError("source_layout_dataset_scope_invalid")
        if fixture == "clean_sets":
            clean_count += 1
        else:
            error_count += 1
    if (
        clean_count != expected_scope.clean_pdf_assets
        or error_count != expected_scope.error_pdf_assets
    ):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_dataset_scope_invalid")

    if not expected_scope.require_complete_canonical_inventory:
        return
    expected_paths = {
        f"{fixture}_sets/set_{set_number:04d}/{prefix}{set_number:04d}_{suffix}.pdf"
        for fixture, prefix in (("clean", "C"), ("error", "E"))
        for set_number in range(1, expected_scope.sets_per_fixture_kind + 1)
        for suffix in ("BL", "INV", "PL")
    }
    if set(relative_paths) != expected_paths:
        raise SourceAttestedSyntheticPdfTruthError("source_layout_dataset_scope_invalid")


def validate_source_attested_synthetic_packet_scope(
    *,
    receipt: DatasetReceiptV1,
    packet: EvaluationPacketV2,
    expected_scope: SourceAttestedSyntheticDatasetScopeV1 = (
        EXPECTED_SOURCE_ATTESTED_SYNTHETIC_DATASET_SCOPE
    ),
) -> None:
    """Require the preselected 60/20 packets for the exact corpus.

    A valid packet self-hash only says its own bytes are intact.  The automatic
    path additionally pins the two packet receipts generated from the existing
    6,000-PDF receipt, which prevents a hand-picked easy 60/20 subset from
    becoming release evidence.  The structural checks make the failure reason
    deterministic even before the fixed packet receipt comparison.
    """

    validate_source_attested_synthetic_dataset_scope(receipt=receipt, expected_scope=expected_scope)
    if (
        packet.packet_kind != "stratified-v2"
        or packet.dataset_id != receipt.dataset_id
        or packet.dataset_revision != receipt.dataset_revision
        or packet.receipt_sha256 != receipt_sha256(receipt)
    ):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_packet_scope_invalid")
    assets_by_id = {asset.asset_id: asset for asset in receipt.assets}
    strata: dict[tuple[str, SyntheticDocumentType], int] = {}
    for item in packet.items:
        asset = assets_by_id.get(item.asset_id)
        matched = (
            _SYNTHETIC_PDF_RECEIPT_PATH_RE.fullmatch(asset.relative_path)
            if asset is not None
            else None
        )
        if (
            asset is None
            or asset.sha256 != item.source_sha256
            or asset.media_type != "application/pdf"
            or asset.source_role != "preview_candidate"
            or matched is None
        ):
            raise SourceAttestedSyntheticPdfTruthError("source_layout_packet_scope_invalid")
        fixture_kind = "clean" if matched.group("fixture") == "clean_sets" else "error"
        document_type = {
            "BL": "bill_of_lading",
            "INV": "invoice",
            "PL": "packing_list",
        }[matched.group("document")]
        expected_set_id = ("C" if fixture_kind == "clean" else "E") + matched.group("set_number")
        if (
            item.fixture_kind != fixture_kind
            or item.document_type != document_type
            or item.set_id != expected_set_id
        ):
            raise SourceAttestedSyntheticPdfTruthError("source_layout_packet_scope_invalid")
        key = (fixture_kind, item.document_type)
        strata[key] = strata.get(key, 0) + 1
    if not expected_scope.require_canonical_packets:
        return
    expected_packet_sha256 = (
        expected_scope.development_packet_receipt_sha256
        if packet.split == "development"
        else expected_scope.sealed_packet_receipt_sha256
    )
    if (
        expected_scope.error_labels_sha256 is None
        or expected_packet_sha256 is None
        or packet.error_labels_sha256 != expected_scope.error_labels_sha256
        or packet.packet_receipt_sha256 != expected_packet_sha256
    ):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_packet_scope_invalid")
    expected_strata = {
        (fixture_kind, document_type)
        for fixture_kind in ("clean", "error")
        for document_type in ("bill_of_lading", "invoice", "packing_list")
    }
    if set(strata) != expected_strata:
        raise SourceAttestedSyntheticPdfTruthError("source_layout_packet_scope_invalid")
    if packet.split == "development":
        if len(packet.items) != 60 or any(count != 10 for count in strata.values()):
            raise SourceAttestedSyntheticPdfTruthError("source_layout_packet_scope_invalid")
    elif len(packet.items) != 20 or any(count < 3 for count in strata.values()):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_packet_scope_invalid")


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


def _validate_required_field_schema(
    *,
    document_type: SyntheticDocumentType,
    fields: Sequence[GoldFieldV2],
    required_field_schema: Mapping[
        SyntheticDocumentType,
        tuple[tuple[str, Literal["text-v1", "numeric-v1", "identifier-v1"]], ...],
    ] = SOURCE_ATTESTED_REQUIRED_FIELDS,
) -> None:
    """Reject partial structured-field truth before the field gate is used."""

    expected = required_field_schema.get(document_type)
    if expected is None:
        raise SourceAttestedSyntheticPdfTruthError("source_layout_required_field_schema_invalid")
    actual = tuple((field.field_name, field.normalizer) for field in fields)
    if len(actual) != len(set(actual)) or set(actual) != set(expected):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_required_field_mapping_missing")


def _require_profile(
    *,
    document_type: SyntheticDocumentType,
    parsed: _ParsedContentStream,
    required_field_schema: Mapping[
        SyntheticDocumentType,
        tuple[tuple[str, Literal["text-v1", "numeric-v1", "identifier-v1"]], ...],
    ] = SOURCE_ATTESTED_REQUIRED_FIELDS,
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
    for slot in profile.field_slots:
        source_indices = [anchor.reading_order for anchor in slot.source_runs]
        if (
            not source_indices
            or len(source_indices) != len(set(source_indices))
            or any(index < 1 or index > len(parsed.text_runs) for index in source_indices)
        ):
            raise SourceAttestedSyntheticPdfTruthError("source_layout_template_profile_invalid")
        for anchor in slot.source_runs:
            if bool(anchor.x_values) == (anchor.x_range is not None):
                raise SourceAttestedSyntheticPdfTruthError("source_layout_template_profile_invalid")
            if anchor.x_range is not None and anchor.x_range[0] > anchor.x_range[1]:
                raise SourceAttestedSyntheticPdfTruthError("source_layout_template_profile_invalid")
            run = parsed.text_runs[anchor.reading_order - 1]
            x_matches = (
                any(abs(run.x - expected_x) <= 0.0001 for expected_x in anchor.x_values)
                if anchor.x_values
                else (
                    anchor.x_range is not None
                    and anchor.x_range[0] - 0.0001 <= run.x <= anchor.x_range[1] + 0.0001
                )
            )
            if not x_matches or abs(run.y - anchor.y) > 0.0001:
                raise SourceAttestedSyntheticPdfTruthError(
                    "source_layout_template_profile_mismatch"
                )
    _validate_required_field_schema(
        document_type=document_type,
        fields=[
            GoldFieldV2(
                field_name=slot.field_name,
                value="profile-slot",
                normalizer=slot.normalizer,
            )
            for slot in profile.field_slots
        ],
        required_field_schema=required_field_schema,
    )
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


_SOURCE_NUMERIC_TOKEN_RE = re.compile(
    r"(?<![A-Za-z0-9])(?:\d{1,3}(?:,\d{3})*|\d+)(?:\.\d+)?(?![A-Za-z0-9])"
)
_SOURCE_ALPHA_TOKEN_RE = re.compile(r"[A-Za-z]+")


def _source_field_value(
    *, source_values: Sequence[str], value_transform: SourceLayoutValueTransform
) -> str:
    if not source_values or any(not value.strip() for value in source_values):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_field_ambiguous")
    if value_transform == "raw-v1":
        if len(source_values) != 1:
            raise SourceAttestedSyntheticPdfTruthError("source_layout_field_transform_invalid")
        return source_values[0].strip()
    if value_transform == "join-slash-v1":
        if len(source_values) < 2:
            raise SourceAttestedSyntheticPdfTruthError("source_layout_field_transform_invalid")
        return " / ".join(value.strip() for value in source_values)
    if len(source_values) != 1:
        raise SourceAttestedSyntheticPdfTruthError("source_layout_field_transform_invalid")
    source_value = source_values[0]
    if value_transform == "single-numeric-token-v1":
        tokens = _SOURCE_NUMERIC_TOKEN_RE.findall(source_value)
    elif value_transform == "single-alpha-token-v1":
        tokens = _SOURCE_ALPHA_TOKEN_RE.findall(source_value)
    else:
        raise SourceAttestedSyntheticPdfTruthError("source_layout_field_transform_invalid")
    if len(tokens) != 1 or not tokens[0].strip():
        raise SourceAttestedSyntheticPdfTruthError("source_layout_field_ambiguous")
    return tokens[0].strip()


def _fields_from_profile(
    *, parsed: _ParsedContentStream, profile: _TemplateProfile
) -> list[GoldFieldV2]:
    fields: list[GoldFieldV2] = []
    for slot in profile.field_slots:
        value = _source_field_value(
            source_values=[
                parsed.text_runs[anchor.reading_order - 1].text for anchor in slot.source_runs
            ],
            value_transform=slot.value_transform,
        )
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


def _error_label_projection(
    *,
    document_type: SyntheticDocumentType,
    field_name: str,
    value: str,
    normalizer: Literal["text-v1", "numeric-v1", "identifier-v1"],
) -> tuple[str, ...] | None:
    projection = _SOURCE_ERROR_LABEL_PROJECTIONS.get((document_type, field_name), "normalized-v1")
    if projection == "normalized-v1":
        return (normalize_field_value(value, normalizer),)
    tokens = _SOURCE_NUMERIC_TOKEN_RE.findall(value)
    expected_count = 1 if projection == "numeric-v1" else 2
    if len(tokens) != expected_count:
        return None
    return tuple(normalize_field_value(token, "numeric-v1") for token in tokens)


def _error_label_rendering_matches(
    *,
    document_type: SyntheticDocumentType,
    field_name: str,
    rendered_value: str,
    label_value: str,
    normalizer: Literal["text-v1", "numeric-v1", "identifier-v1"],
) -> bool:
    rendered = _error_label_projection(
        document_type=document_type,
        field_name=field_name,
        value=rendered_value,
        normalizer=normalizer,
    )
    label = _error_label_projection(
        document_type=document_type,
        field_name=field_name,
        value=label_value,
        normalizer=normalizer,
    )
    return rendered is not None and rendered == label


def _require_error_content_stream_difference(
    *,
    item: StratifiedPacketItemV2,
    parsed_error: _ParsedContentStream,
    error_profile: _TemplateProfile,
    error_labels: Sequence[Mapping[str, str]],
    assets_by_relative_path: Mapping[str, DatasetAssetReceiptV1],
    source_directory: Path,
    required_field_schema: Mapping[
        SyntheticDocumentType,
        tuple[tuple[str, Literal["text-v1", "numeric-v1", "identifier-v1"]], ...],
    ],
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
    clean_profile = _require_profile(
        document_type=item.document_type,
        parsed=parsed_clean,
        required_field_schema=required_field_schema,
    )
    if parsed_clean.decoded_sha256 == parsed_error.decoded_sha256:
        raise SourceAttestedSyntheticPdfTruthError("source_layout_error_content_unchanged")

    # The source parser remains the only producer of returned field values.
    # Here, and only here, a human-checked CSV label is used as a negative
    # fixture assertion: at least one labelled mutation must be visibly
    # present in the final error PDF and differ from the clean sibling.  A
    # merely different stream is not enough—an unrelated title/date edit
    # cannot be counted as an error-stratum rendering.
    error_rendered_fields = {
        field.field_name: field
        for field in _fields_from_profile(parsed=parsed_error, profile=error_profile)
    }
    clean_rendered_fields = {
        field.field_name: field
        for field in _fields_from_profile(parsed=parsed_clean, profile=clean_profile)
    }
    mutation_rendered = False
    for label in error_labels:
        field_name = _SOURCE_ERROR_LABEL_FIELDS.get((item.document_type, label["field"]))
        if field_name is None:
            continue
        error_rendered = error_rendered_fields.get(field_name)
        clean_rendered = clean_rendered_fields.get(field_name)
        if error_rendered is None or clean_rendered is None:
            continue
        wrong_projection = _error_label_projection(
            document_type=item.document_type,
            field_name=field_name,
            value=label["wrong_value"],
            normalizer=error_rendered.normalizer,
        )
        correct_projection = _error_label_projection(
            document_type=item.document_type,
            field_name=field_name,
            value=label["correct_value"],
            normalizer=clean_rendered.normalizer,
        )
        if (
            wrong_projection is None
            or correct_projection is None
            or wrong_projection == correct_projection
        ):
            continue
        if (
            _error_label_rendering_matches(
                document_type=item.document_type,
                field_name=field_name,
                rendered_value=error_rendered.value,
                label_value=label["wrong_value"],
                normalizer=error_rendered.normalizer,
            )
            and _error_label_rendering_matches(
                document_type=item.document_type,
                field_name=field_name,
                rendered_value=clean_rendered.value,
                label_value=label["correct_value"],
                normalizer=clean_rendered.normalizer,
            )
            and _error_label_projection(
                document_type=item.document_type,
                field_name=field_name,
                value=error_rendered.value,
                normalizer=error_rendered.normalizer,
            )
            != _error_label_projection(
                document_type=item.document_type,
                field_name=field_name,
                value=clean_rendered.value,
                normalizer=clean_rendered.normalizer,
            )
        ):
            mutation_rendered = True
            break
    if not mutation_rendered:
        raise SourceAttestedSyntheticPdfTruthError("source_layout_error_label_rendering_unverified")


def compile_source_attested_synthetic_pdf_layout_truth(
    *,
    receipt: DatasetReceiptV1,
    packet: EvaluationPacketV2,
    source_directory: Path,
    clean_manifest_path: Path,
    error_labels_path: Path,
    expected_dataset_scope: SourceAttestedSyntheticDatasetScopeV1 = (
        EXPECTED_SOURCE_ATTESTED_SYNTHETIC_DATASET_SCOPE
    ),
    required_field_schema: Mapping[
        SyntheticDocumentType,
        tuple[tuple[str, Literal["text-v1", "numeric-v1", "identifier-v1"]], ...],
    ] = SOURCE_ATTESTED_REQUIRED_FIELDS,
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
    validate_source_attested_synthetic_packet_scope(
        receipt=receipt, packet=packet, expected_scope=expected_dataset_scope
    )
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
        profile = _require_profile(
            document_type=item.document_type,
            parsed=parsed,
            required_field_schema=required_field_schema,
        )
        if item.fixture_kind == "error":
            error_labels = labels_by_key.get((item.set_id, item.document_type))
            if not error_labels:
                raise SourceAttestedSyntheticPdfTruthError("source_layout_error_label_missing")
            _require_error_content_stream_difference(
                item=item,
                parsed_error=parsed,
                error_profile=profile,
                error_labels=error_labels,
                assets_by_relative_path=receipt_assets_by_relative_path,
                source_directory=source_directory,
                required_field_schema=required_field_schema,
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
        _validate_required_field_schema(
            document_type=item.document_type,
            fields=record.fields,
            required_field_schema=required_field_schema,
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
    expected_dataset_scope: SourceAttestedSyntheticDatasetScopeV1 = (
        EXPECTED_SOURCE_ATTESTED_SYNTHETIC_DATASET_SCOPE
    ),
    required_field_schema: Mapping[
        SyntheticDocumentType,
        tuple[tuple[str, Literal["text-v1", "numeric-v1", "identifier-v1"]], ...],
    ] = SOURCE_ATTESTED_REQUIRED_FIELDS,
) -> None:
    """Verify receipt → packet → source-layout bindings before scoring."""

    _verify_receipt_packet_binding(receipt=receipt, packet=packet)
    validate_source_attested_synthetic_packet_scope(
        receipt=receipt, packet=packet, expected_scope=expected_dataset_scope
    )
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
        _validate_required_field_schema(
            document_type=record.document_type,
            fields=record.fields,
            required_field_schema=required_field_schema,
        )


class SourceAttestedSyntheticCandidateExecutionV1(_WireModel):
    """Content-free observation of the exact candidate bytes that were scored.

    This is not a claim that a local administrator cannot edit local files; the
    automatic path's trust boundary remains the local filesystem.  It does
    prevent an ordinary result from being relabelled after evaluation and gives
    the runtime enough immutable metadata to re-read the raw local candidate
    and replay the score without printing candidate text.
    """

    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["source_attested_synthetic_pdf_candidate_execution"] = (
        "source_attested_synthetic_pdf_candidate_execution"
    )
    profile_mode: Literal["offline_synthetic_pdf_evaluation"] = "offline_synthetic_pdf_evaluation"
    dataset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    dataset_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    dataset_receipt_sha256: str = Field(pattern=_SHA256)
    packet_kind: Literal["development", "sealed"]
    packet_receipt_sha256: str = Field(pattern=_SHA256)
    layout_truth_receipt_sha256: str = Field(pattern=_SHA256)
    layout_truth_parser_spec_sha256: str = Field(pattern=_SHA256)
    engine_kind: EngineKind
    extractor_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    configuration_sha256: str = Field(pattern=_SHA256)
    model_inventory_sha256: str = Field(pattern=_SHA256)
    candidate_parser_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    candidate_bundle_sha256: str = Field(pattern=_SHA256)
    candidate_output_sha256: str = Field(pattern=_SHA256)
    candidate_execution_receipt_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _execution_is_hash_bound(self) -> "SourceAttestedSyntheticCandidateExecutionV1":
        if self.candidate_parser_revision != SOURCE_ATTESTED_CANDIDATE_PARSER_REVISION:
            raise ValueError("source-attested candidate parser revision is not approved")
        if self.candidate_execution_receipt_sha256 != candidate_execution_receipt_sha256(self):
            raise ValueError("source-attested candidate execution SHA-256 mismatch")
        return self


def _candidate_execution_receipt_payload(
    execution: SourceAttestedSyntheticCandidateExecutionV1 | Mapping[str, object],
) -> dict[str, object]:
    payload = (
        execution.model_dump(mode="json", warnings=False)
        if isinstance(execution, SourceAttestedSyntheticCandidateExecutionV1)
        else dict(execution)
    )
    payload.pop("candidate_execution_receipt_sha256", None)
    return payload


def candidate_execution_receipt_sha256(
    execution: SourceAttestedSyntheticCandidateExecutionV1 | Mapping[str, object],
) -> str:
    return sha256_bytes(canonical_json_bytes(_candidate_execution_receipt_payload(execution)))


def _canonical_candidate_output_bytes(
    candidate_output: SyntheticPdfCandidateOutputBundleV1,
) -> bytes:
    return canonical_json_file_bytes(candidate_output.model_dump(mode="json", warnings=False))


def build_source_attested_synthetic_candidate_execution(
    *,
    receipt: DatasetReceiptV1,
    packet: EvaluationPacketV2,
    layout_truth: SourceAttestedSyntheticLayoutTruthV1,
    candidate_output: SyntheticPdfCandidateOutputBundleV1,
    candidate_output_raw: bytes,
    engine_kind: EngineKind,
    extractor_revision: str,
    configuration_sha256: str,
    model_inventory_sha256: str,
    parser_revision: str,
    expected_dataset_scope: SourceAttestedSyntheticDatasetScopeV1 = (
        EXPECTED_SOURCE_ATTESTED_SYNTHETIC_DATASET_SCOPE
    ),
    required_field_schema: Mapping[
        SyntheticDocumentType,
        tuple[tuple[str, Literal["text-v1", "numeric-v1", "identifier-v1"]], ...],
    ] = SOURCE_ATTESTED_REQUIRED_FIELDS,
) -> SourceAttestedSyntheticCandidateExecutionV1:
    """Bind an evaluated raw candidate file to its exact source-layout input."""

    _require_source_attested_candidate_parser_revision(parser_revision)
    validate_source_attested_layout_truth_for_evaluation(
        receipt=receipt,
        packet=packet,
        truth=layout_truth,
        expected_dataset_scope=expected_dataset_scope,
        required_field_schema=required_field_schema,
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
    if candidate_output_raw != _canonical_candidate_output_bytes(candidate_output):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_candidate_execution_invalid")
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "artifact_kind": "source_attested_synthetic_pdf_candidate_execution",
        "profile_mode": "offline_synthetic_pdf_evaluation",
        "dataset_id": packet.dataset_id,
        "dataset_revision": packet.dataset_revision,
        "dataset_receipt_sha256": receipt_sha256(receipt),
        "packet_kind": packet.split,
        "packet_receipt_sha256": packet.packet_receipt_sha256,
        "layout_truth_receipt_sha256": layout_truth.layout_truth_receipt_sha256,
        "layout_truth_parser_spec_sha256": layout_truth.parser_spec_sha256,
        "engine_kind": engine_kind,
        "extractor_revision": extractor_revision,
        "configuration_sha256": configuration_sha256,
        "model_inventory_sha256": model_inventory_sha256,
        "candidate_parser_revision": parser_revision,
        "candidate_bundle_sha256": candidate_output.candidate_bundle_sha256,
        "candidate_output_sha256": sha256_bytes(candidate_output_raw),
    }
    payload["candidate_execution_receipt_sha256"] = candidate_execution_receipt_sha256(payload)
    return SourceAttestedSyntheticCandidateExecutionV1.model_validate(payload)


def validate_source_attested_synthetic_candidate_execution(
    *,
    execution: SourceAttestedSyntheticCandidateExecutionV1,
    receipt: DatasetReceiptV1,
    packet: EvaluationPacketV2,
    layout_truth: SourceAttestedSyntheticLayoutTruthV1,
    candidate_output: SyntheticPdfCandidateOutputBundleV1,
    candidate_output_raw: bytes,
    expected_dataset_scope: SourceAttestedSyntheticDatasetScopeV1 = (
        EXPECTED_SOURCE_ATTESTED_SYNTHETIC_DATASET_SCOPE
    ),
    required_field_schema: Mapping[
        SyntheticDocumentType,
        tuple[tuple[str, Literal["text-v1", "numeric-v1", "identifier-v1"]], ...],
    ] = SOURCE_ATTESTED_REQUIRED_FIELDS,
) -> None:
    """Replay an execution receipt without logging or returning OCR content."""

    expected = build_source_attested_synthetic_candidate_execution(
        receipt=receipt,
        packet=packet,
        layout_truth=layout_truth,
        candidate_output=candidate_output,
        candidate_output_raw=candidate_output_raw,
        engine_kind=execution.engine_kind,
        extractor_revision=execution.extractor_revision,
        configuration_sha256=execution.configuration_sha256,
        model_inventory_sha256=execution.model_inventory_sha256,
        parser_revision=execution.candidate_parser_revision,
        expected_dataset_scope=expected_dataset_scope,
        required_field_schema=required_field_schema,
    )
    if execution != expected:
        raise SourceAttestedSyntheticPdfTruthError(
            "source_layout_candidate_execution_binding_mismatch"
        )


class SourceAttestedSyntheticFieldMetricDenominatorV1(_WireModel):
    """Content-free field-gate denominator for one rendered document type."""

    document_type: SyntheticDocumentType
    document_count: int = Field(ge=1, le=100_000)
    fields_per_document: int = Field(ge=1, le=1_000)
    total_fields: int = Field(ge=1, le=100_000_000)

    @model_validator(mode="after")
    def _total_is_derived(self) -> "SourceAttestedSyntheticFieldMetricDenominatorV1":
        if self.total_fields != self.document_count * self.fields_per_document:
            raise ValueError("source-attested field denominator is not derived")
        return self


class SourceAttestedSyntheticOcrMetricsV1(_WireModel):
    field_exact_match: float = Field(ge=0, le=1)
    field_exact_match_count: int = Field(ge=0, le=100_000_000)
    field_exact_match_total: int = Field(ge=1, le=100_000_000)
    field_exact_match_denominators: list[SourceAttestedSyntheticFieldMetricDenominatorV1] = Field(
        min_length=1, max_length=3
    )
    ocr_cer: float = Field(ge=0, le=1)
    table_cell_f1: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _field_ratio_and_denominator_are_derived(self) -> "SourceAttestedSyntheticOcrMetricsV1":
        if self.field_exact_match_count > self.field_exact_match_total:
            raise ValueError("source-attested field match count exceeds denominator")
        types = [item.document_type for item in self.field_exact_match_denominators]
        if (
            len(types) != len(set(types))
            or sum(item.total_fields for item in self.field_exact_match_denominators)
            != self.field_exact_match_total
        ):
            raise ValueError("source-attested field denominator coverage is invalid")
        if (
            abs(
                self.field_exact_match
                - (self.field_exact_match_count / self.field_exact_match_total)
            )
            > 1e-12
        ):
            raise ValueError("source-attested field exact match ratio is not derived")
        return self


class SourceAttestedSyntheticOcrEvaluationResultV1(_WireModel):
    """Content-free result with the exact runtime source-attested contract."""

    schema_version: Literal["1.0"] = "1.0"
    evaluation_basis: Literal["source_attested_synthetic"] = SOURCE_ATTESTED_EVALUATION_BASIS
    dataset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    dataset_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    packet_kind: Literal["development", "sealed"]
    packet_receipt_sha256: str = Field(pattern=_SHA256)
    evaluation_scope: Literal["synthetic_r1_supervised_native_template_compatibility"] = (
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
    candidate_parser_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    candidate_bundle_sha256: str = Field(pattern=_SHA256)
    candidate_execution_receipt_sha256: str = Field(pattern=_SHA256)
    evaluation_result_sha256: str = Field(pattern=_SHA256)
    passed: bool
    metrics: SourceAttestedSyntheticOcrMetricsV1

    @model_validator(mode="after")
    def _result_is_hash_bound_and_gate_derived(
        self,
    ) -> "SourceAttestedSyntheticOcrEvaluationResultV1":
        if self.candidate_parser_revision != SOURCE_ATTESTED_CANDIDATE_PARSER_REVISION:
            raise ValueError("source-attested candidate parser revision is not approved")
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
    candidate_parser_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    candidate_bundle_sha256: str = Field(pattern=_SHA256)
    candidate_execution_receipt_sha256: str = Field(pattern=_SHA256)
    evaluation_result_sha256: str = Field(pattern=_SHA256)
    passed: bool


class SourceAttestedSyntheticDevelopmentSelectionV1(_WireModel):
    """Immutable development validation binding for a source-truth run.

    The canonical r1 packet validates one fixed supervised native-PDF profile;
    it does not rank alternatives.  The broader one-to-three member shape is
    retained solely so hermetic fixtures can continue exercising the legacy
    three-engine selection path without weakening the production identity.
    """

    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["source_attested_synthetic_pdf_development_selection"] = (
        "source_attested_synthetic_pdf_development_selection"
    )
    evaluation_basis: Literal["source_attested_synthetic"] = SOURCE_ATTESTED_EVALUATION_BASIS
    evaluation_scope: Literal["synthetic_r1_supervised_native_template_compatibility"] = (
        SOURCE_ATTESTED_EVALUATION_SCOPE
    )
    layout_truth_kind: Literal["pdf_content_stream_layout_v1"] = SOURCE_LAYOUT_TRUTH_KIND
    dataset_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    dataset_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    development_packet_receipt_sha256: str = Field(pattern=_SHA256)
    layout_truth_receipt_sha256: str = Field(pattern=_SHA256)
    layout_truth_parser_spec_sha256: str = Field(pattern=_SHA256)
    candidate_parser_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    candidate_results: list[SourceAttestedDevelopmentCandidateResultV1] = Field(
        min_length=1, max_length=3
    )
    selected_engine_kind: EngineKind
    selected_extractor_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    selected_configuration_sha256: str = Field(pattern=_SHA256)
    selected_model_inventory_sha256: str = Field(pattern=_SHA256)
    selected_evaluation_result_sha256: str = Field(pattern=_SHA256)
    development_selection_receipt_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _selection_is_complete_and_bound(self) -> "SourceAttestedSyntheticDevelopmentSelectionV1":
        if self.candidate_parser_revision != SOURCE_ATTESTED_CANDIDATE_PARSER_REVISION:
            raise ValueError("source-attested candidate parser revision is not approved")
        if self.layout_truth_parser_spec_sha256 != source_layout_parser_spec_sha256():
            raise ValueError("source-attested selection parser specification SHA-256 mismatch")
        if _is_canonical_r1_development_packet(
            dataset_id=self.dataset_id,
            dataset_revision=self.dataset_revision,
            packet_receipt_sha256=self.development_packet_receipt_sha256,
        ):
            if (
                len(self.candidate_results) != 1
                or not self.candidate_results[0].passed
                or not _is_fixed_supervised_template_profile(
                    engine_kind=self.candidate_results[0].engine_kind,
                    extractor_revision=self.candidate_results[0].extractor_revision,
                )
            ):
                raise ValueError(
                    "canonical r1 selection requires one passing fixed supervised "
                    "pdf_text_layer profile"
                )
        else:
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
                and item.candidate_parser_revision == self.candidate_parser_revision
                and item.evaluation_result_sha256 == self.selected_evaluation_result_sha256
            )
        ]
        if len(selected) != 1 or not selected[0].passed:
            raise ValueError("source-attested selected engine is not a passing development member")
        if {item.candidate_parser_revision for item in self.candidate_results} != {
            self.candidate_parser_revision
        }:
            raise ValueError(
                "source-attested development candidates must share one parser revision"
            )
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
    evaluation_scope: Literal["synthetic_r1_supervised_native_template_compatibility"] = (
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
    candidate_execution_receipt_sha256: str = Field(pattern=_SHA256)
    sealed_candidate_reservation_sha256: str = Field(pattern=_SHA256)
    sealed_run_receipt_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _sealed_run_is_bound(self) -> "SourceAttestedSyntheticSealedRunV1":
        if self.candidate_parser_revision != SOURCE_ATTESTED_CANDIDATE_PARSER_REVISION:
            raise ValueError("source-attested candidate parser revision is not approved")
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


class SourceAttestedSyntheticCandidateRescoreV1(_WireModel):
    """Content-free output from replaying one canonical local candidate."""

    candidate_parser_revision: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER)
    candidate_bundle_sha256: str = Field(pattern=_SHA256)
    candidate_output_sha256: str = Field(pattern=_SHA256)
    metrics: SourceAttestedSyntheticOcrMetricsV1


def _parse_canonical_synthetic_candidate_output_raw(
    candidate_output_raw: bytes,
) -> SyntheticPdfCandidateOutputBundleV1:
    """Parse one candidate only when its raw local file is canonical.

    The caller deliberately receives no OCR text from this helper.  The raw
    bytes stay in memory, are hash-bound by the execution receipt, and must
    never be written to a transcript or general log.
    """

    try:
        candidate_output = SyntheticPdfCandidateOutputBundleV1.model_validate_json(
            candidate_output_raw
        )
    except (TypeError, ValueError, ValidationError) as error:
        raise SourceAttestedSyntheticPdfTruthError(
            "source_layout_candidate_output_invalid"
        ) from error
    if candidate_output_raw != _canonical_candidate_output_bytes(candidate_output):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_candidate_output_not_canonical")
    return candidate_output


def _score_source_attested_candidate_against_layout_truth(
    *,
    layout_truth: SourceAttestedSyntheticLayoutTruthV1,
    candidate_output: SyntheticPdfCandidateOutputBundleV1,
) -> SourceAttestedSyntheticOcrMetricsV1:
    """Calculate fixed metrics after all source/provenance checks have passed."""

    candidate = candidate_output_from_synthetic_bundle(candidate_output)
    candidate_by_asset = {item.asset_id: item for item in candidate.documents}
    truth_by_asset = {item.asset_id: item for item in layout_truth.records}
    if set(candidate_by_asset) != set(truth_by_asset):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_candidate_coverage_mismatch")

    exact_fields = 0
    total_fields = 0
    field_counts_by_document_type: dict[SyntheticDocumentType, int] = {}
    document_counts_by_document_type: dict[SyntheticDocumentType, int] = {}
    total_distance = 0
    total_reference_characters = 0
    source_cells: set[tuple[str, int, int, int, int, str]] = set()
    candidate_cells: set[tuple[str, int, int, int, int, str]] = set()
    for asset_id, source in truth_by_asset.items():
        output = candidate_by_asset[asset_id]
        if output.source_sha256 != source.source_sha256:
            raise SourceAttestedSyntheticPdfTruthError("source_layout_candidate_source_mismatch")
        candidate_fields = {field.field_name: field.value for field in output.fields}
        source_field_count = len(source.fields)
        prior_field_count = field_counts_by_document_type.get(source.document_type)
        if prior_field_count is not None and prior_field_count != source_field_count:
            raise SourceAttestedSyntheticPdfTruthError("source_layout_field_schema_inconsistent")
        field_counts_by_document_type[source.document_type] = source_field_count
        document_counts_by_document_type[source.document_type] = (
            document_counts_by_document_type.get(source.document_type, 0) + 1
        )
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
    return SourceAttestedSyntheticOcrMetricsV1(
        field_exact_match=exact_fields / total_fields,
        field_exact_match_count=exact_fields,
        field_exact_match_total=total_fields,
        field_exact_match_denominators=[
            SourceAttestedSyntheticFieldMetricDenominatorV1(
                document_type=document_type,
                document_count=document_counts_by_document_type[document_type],
                fields_per_document=field_counts_by_document_type[document_type],
                total_fields=(
                    document_counts_by_document_type[document_type]
                    * field_counts_by_document_type[document_type]
                ),
            )
            for document_type in sorted(field_counts_by_document_type)
        ],
        ocr_cer=min(1.0, total_distance / total_reference_characters),
        table_cell_f1=0.0 if denominator == 0 else (2.0 * true_positive) / denominator,
    )


def recompute_source_attested_synthetic_pdf_candidate_metrics(
    *,
    receipt: DatasetReceiptV1,
    packet: EvaluationPacketV2,
    layout_truth: SourceAttestedSyntheticLayoutTruthV1,
    candidate_output_raw: bytes,
    engine_kind: EngineKind,
    extractor_revision: str,
    configuration_sha256: str,
    model_inventory_sha256: str,
    parser_revision: str,
    expected_dataset_scope: SourceAttestedSyntheticDatasetScopeV1 = (
        EXPECTED_SOURCE_ATTESTED_SYNTHETIC_DATASET_SCOPE
    ),
    required_field_schema: Mapping[
        SyntheticDocumentType,
        tuple[tuple[str, Literal["text-v1", "numeric-v1", "identifier-v1"]], ...],
    ] = SOURCE_ATTESTED_REQUIRED_FIELDS,
) -> SourceAttestedSyntheticCandidateRescoreV1:
    """Re-score raw local candidate bytes without logging or exposing their text.

    This is the replay primitive used by the runtime activation verifier.  It
    accepts only canonical candidate bytes, rechecks the exact receipt/packet/
    source-layout chain and emits hashes plus metrics—not candidate contents.
    """

    _require_source_attested_candidate_parser_revision(parser_revision)
    validate_source_attested_layout_truth_for_evaluation(
        receipt=receipt,
        packet=packet,
        truth=layout_truth,
        expected_dataset_scope=expected_dataset_scope,
        required_field_schema=required_field_schema,
    )
    candidate_output = _parse_canonical_synthetic_candidate_output_raw(candidate_output_raw)
    validate_synthetic_candidate_output_provenance(
        candidate_output=candidate_output,
        packet=packet,
        engine_kind=engine_kind,
        extractor_revision=extractor_revision,
        configuration_sha256=configuration_sha256,
        model_inventory_sha256=model_inventory_sha256,
        parser_revision=parser_revision,
    )
    return SourceAttestedSyntheticCandidateRescoreV1(
        candidate_parser_revision=candidate_output.parser_revision,
        candidate_bundle_sha256=candidate_output.candidate_bundle_sha256,
        candidate_output_sha256=sha256_bytes(candidate_output_raw),
        metrics=_score_source_attested_candidate_against_layout_truth(
            layout_truth=layout_truth, candidate_output=candidate_output
        ),
    )


def score_source_attested_synthetic_pdf_candidate_output(
    *,
    receipt: DatasetReceiptV1,
    packet: EvaluationPacketV2,
    layout_truth: SourceAttestedSyntheticLayoutTruthV1,
    candidate_output_raw: bytes,
    candidate_execution: SourceAttestedSyntheticCandidateExecutionV1,
    engine_kind: EngineKind,
    extractor_revision: str,
    configuration_sha256: str,
    model_inventory_sha256: str,
    parser_revision: str,
    development_selection: SourceAttestedSyntheticDevelopmentSelectionV1 | None = None,
    sealed_run: SourceAttestedSyntheticSealedRunV1 | None = None,
    expected_dataset_scope: SourceAttestedSyntheticDatasetScopeV1 = (
        EXPECTED_SOURCE_ATTESTED_SYNTHETIC_DATASET_SCOPE
    ),
    required_field_schema: Mapping[
        SyntheticDocumentType,
        tuple[tuple[str, Literal["text-v1", "numeric-v1", "identifier-v1"]], ...],
    ] = SOURCE_ATTESTED_REQUIRED_FIELDS,
) -> SourceAttestedSyntheticOcrEvaluationResultV1:
    """Score one local candidate against final-PDF source-layout truth.

    Native text-layer candidates are permitted as diagnostics.  The result's
    fixed scope makes clear that their score is content-stream compatibility,
    not independent optical OCR evidence.
    """

    _require_source_attested_candidate_parser_revision(parser_revision)
    candidate_output = _parse_canonical_synthetic_candidate_output_raw(candidate_output_raw)
    rescore = recompute_source_attested_synthetic_pdf_candidate_metrics(
        receipt=receipt,
        packet=packet,
        layout_truth=layout_truth,
        candidate_output_raw=candidate_output_raw,
        engine_kind=engine_kind,
        extractor_revision=extractor_revision,
        configuration_sha256=configuration_sha256,
        model_inventory_sha256=model_inventory_sha256,
        parser_revision=parser_revision,
        expected_dataset_scope=expected_dataset_scope,
        required_field_schema=required_field_schema,
    )
    validate_source_attested_synthetic_candidate_execution(
        execution=candidate_execution,
        receipt=receipt,
        packet=packet,
        layout_truth=layout_truth,
        candidate_output=candidate_output,
        candidate_output_raw=candidate_output_raw,
        expected_dataset_scope=expected_dataset_scope,
        required_field_schema=required_field_schema,
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
            candidate_execution=candidate_execution,
            engine_kind=engine_kind,
            extractor_revision=extractor_revision,
            configuration_sha256=configuration_sha256,
            model_inventory_sha256=model_inventory_sha256,
            parser_revision=parser_revision,
            development_selection=development_selection,
            sealed_run=sealed_run,
        )
    metrics = rescore.metrics
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
        "candidate_parser_revision": parser_revision,
        "candidate_bundle_sha256": candidate_output.candidate_bundle_sha256,
        "candidate_execution_receipt_sha256": (
            candidate_execution.candidate_execution_receipt_sha256
        ),
        "passed": passed,
        "metrics": metrics.model_dump(mode="json", warnings=False),
    }
    payload["evaluation_result_sha256"] = source_attested_evaluation_result_sha256(payload)
    return SourceAttestedSyntheticOcrEvaluationResultV1.model_validate(payload)


def verify_source_attested_synthetic_pdf_evaluation_replay(
    *,
    result: SourceAttestedSyntheticOcrEvaluationResultV1,
    receipt: DatasetReceiptV1,
    packet: EvaluationPacketV2,
    layout_truth: SourceAttestedSyntheticLayoutTruthV1,
    candidate_output_raw: bytes,
    candidate_execution: SourceAttestedSyntheticCandidateExecutionV1,
    engine_kind: EngineKind,
    extractor_revision: str,
    configuration_sha256: str,
    model_inventory_sha256: str,
    parser_revision: str,
    development_selection: SourceAttestedSyntheticDevelopmentSelectionV1 | None = None,
    sealed_run: SourceAttestedSyntheticSealedRunV1 | None = None,
    expected_dataset_scope: SourceAttestedSyntheticDatasetScopeV1 = (
        EXPECTED_SOURCE_ATTESTED_SYNTHETIC_DATASET_SCOPE
    ),
    required_field_schema: Mapping[
        SyntheticDocumentType,
        tuple[tuple[str, Literal["text-v1", "numeric-v1", "identifier-v1"]], ...],
    ] = SOURCE_ATTESTED_REQUIRED_FIELDS,
) -> None:
    """Fail closed unless a result exactly matches a fresh raw-candidate replay.

    This pure verifier is intentionally importable by the runtime.  It reads
    no paths and produces no OCR/document text; callers supply already-read
    local bytes and may safely surface only its content-free error code.
    """

    replayed = score_source_attested_synthetic_pdf_candidate_output(
        receipt=receipt,
        packet=packet,
        layout_truth=layout_truth,
        candidate_output_raw=candidate_output_raw,
        candidate_execution=candidate_execution,
        engine_kind=engine_kind,
        extractor_revision=extractor_revision,
        configuration_sha256=configuration_sha256,
        model_inventory_sha256=model_inventory_sha256,
        parser_revision=parser_revision,
        development_selection=development_selection,
        sealed_run=sealed_run,
        expected_dataset_scope=expected_dataset_scope,
        required_field_schema=required_field_schema,
    )
    if result != replayed:
        raise SourceAttestedSyntheticPdfTruthError("source_layout_evaluation_replay_mismatch")


def select_best_source_attested_synthetic_development_result(
    results: Sequence[SourceAttestedSyntheticOcrEvaluationResultV1],
) -> SourceAttestedSyntheticOcrEvaluationResultV1:
    """Validate the r1 fixed profile or rank a hermetic fixture's engines.

    The canonical r1 development packet has a deliberately narrow contract:
    it carries exactly one passing supervised ``pdf_text_layer`` result.  It
    is therefore profile validation, not a comparison or ranking of engines.
    The old deterministic three-engine ranking remains only for injectable
    non-production fixtures used to test generic receipt handling.
    """

    first = results[0] if results else None
    if first is None or first.packet_kind != "development":
        raise SourceAttestedSyntheticPdfTruthError("source_layout_development_result_set_invalid")
    _require_source_attested_candidate_parser_revision(first.candidate_parser_revision)
    identity = (
        first.dataset_id,
        first.dataset_revision,
        first.packet_receipt_sha256,
        first.layout_truth_receipt_sha256,
        first.layout_truth_parser_spec_sha256,
        first.candidate_parser_revision,
    )
    if any(
        result.packet_kind != "development"
        or (
            result.dataset_id,
            result.dataset_revision,
            result.packet_receipt_sha256,
            result.layout_truth_receipt_sha256,
            result.layout_truth_parser_spec_sha256,
            result.candidate_parser_revision,
        )
        != identity
        for result in results
    ):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_development_result_set_invalid")
    if _is_canonical_r1_development_packet(
        dataset_id=first.dataset_id,
        dataset_revision=first.dataset_revision,
        packet_receipt_sha256=first.packet_receipt_sha256,
    ):
        if len(results) != 1 or not _is_fixed_supervised_template_profile(
            engine_kind=first.engine_kind,
            extractor_revision=first.extractor_revision,
        ):
            raise SourceAttestedSyntheticPdfTruthError(
                "source_layout_supervised_development_profile_invalid"
            )
        if not first.passed:
            raise SourceAttestedSyntheticPdfTruthError(
                "source_layout_supervised_development_gate_failed"
            )
        return first

    expected_engines = {"pdf_text_layer", "paddleocr", "docling"}
    if (
        len(results) != len(expected_engines)
        or {result.engine_kind for result in results} != expected_engines
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
    """Bind the selected development profile and its immutable evidence.

    On the canonical r1 packet this persists a single fixed-profile result;
    it is intentionally not an engine ranking artifact.
    """

    selected = select_best_source_attested_synthetic_development_result(results)
    _require_source_attested_candidate_parser_revision(selected.candidate_parser_revision)
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
        "candidate_parser_revision": selected.candidate_parser_revision,
        "candidate_results": [
            {
                "engine_kind": item.engine_kind,
                "extractor_revision": item.extractor_revision,
                "configuration_sha256": item.configuration_sha256,
                "model_inventory_sha256": item.model_inventory_sha256,
                "candidate_parser_revision": item.candidate_parser_revision,
                "candidate_bundle_sha256": item.candidate_bundle_sha256,
                "candidate_execution_receipt_sha256": (item.candidate_execution_receipt_sha256),
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


def write_source_attested_synthetic_candidate_execution(
    *,
    execution: SourceAttestedSyntheticCandidateExecutionV1,
    output_directory: Path,
) -> Path:
    """Write the content-free execution sidecar beside local candidate artifacts."""

    directory = _local_output_directory(
        output_directory, code="source_layout_candidate_execution_directory_invalid"
    )
    return _write_hash_named_immutable(
        directory=directory,
        digest=execution.candidate_execution_receipt_sha256,
        payload=execution.model_dump(mode="json", warnings=False),
        code="source_layout_candidate_execution_immutable_conflict",
    )


def read_canonical_source_attested_synthetic_candidate_execution(
    path: Path,
) -> SourceAttestedSyntheticCandidateExecutionV1:
    try:
        raw = path.read_bytes()
        execution = SourceAttestedSyntheticCandidateExecutionV1.model_validate_json(raw)
    except (OSError, ValidationError) as error:
        raise SourceAttestedSyntheticPdfTruthError(
            "source_layout_candidate_execution_invalid"
        ) from error
    if raw != canonical_json_file_bytes(execution.model_dump(mode="json", warnings=False)):
        raise SourceAttestedSyntheticPdfTruthError(
            "source_layout_candidate_execution_not_canonical"
        )
    if path.name != f"{execution.candidate_execution_receipt_sha256}.json":
        raise SourceAttestedSyntheticPdfTruthError(
            "source_layout_candidate_execution_filename_invalid"
        )
    return execution


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
    _require_source_attested_candidate_parser_revision(parser_revision)
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
    parser_revision: str,
) -> None:
    _require_source_attested_candidate_parser_revision(parser_revision)
    # Pydantic models are passed by reference and may have been mutated after
    # their initial construction.  Revalidate the complete receipt before a
    # sealed run consumes its one-shot reservation, including the canonical
    # r1 single-profile rule.
    try:
        SourceAttestedSyntheticDevelopmentSelectionV1.model_validate(
            selection.model_dump(mode="json", warnings=False)
        )
    except ValidationError as error:
        raise SourceAttestedSyntheticPdfTruthError(
            "source_layout_selection_binding_mismatch"
        ) from error
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
        or selection.candidate_parser_revision != parser_revision
    ):
        raise SourceAttestedSyntheticPdfTruthError("source_layout_selection_binding_mismatch")


def build_source_attested_synthetic_sealed_run(
    *,
    packet: EvaluationPacketV2,
    layout_truth: SourceAttestedSyntheticLayoutTruthV1,
    development_selection: SourceAttestedSyntheticDevelopmentSelectionV1,
    candidate_output: SyntheticPdfCandidateOutputBundleV1,
    candidate_execution: SourceAttestedSyntheticCandidateExecutionV1,
    document_preview_root: Path,
    engine_kind: EngineKind,
    extractor_revision: str,
    configuration_sha256: str,
    model_inventory_sha256: str,
    parser_revision: str,
) -> SourceAttestedSyntheticSealedRunV1:
    _require_source_attested_candidate_parser_revision(parser_revision)
    if packet.split != "sealed":
        raise SourceAttestedSyntheticPdfTruthError("source_layout_sealed_run_packet_invalid")
    _validate_development_selection_for_sealed(
        selection=development_selection,
        packet=packet,
        engine_kind=engine_kind,
        extractor_revision=extractor_revision,
        configuration_sha256=configuration_sha256,
        model_inventory_sha256=model_inventory_sha256,
        parser_revision=parser_revision,
    )
    if (
        candidate_output.packet_receipt_sha256 != packet.packet_receipt_sha256
        or candidate_output.engine_kind != engine_kind
        or candidate_output.extractor_revision != extractor_revision
        or candidate_output.configuration_sha256 != configuration_sha256
        or candidate_output.model_inventory_sha256 != model_inventory_sha256
        or candidate_output.parser_revision != parser_revision
        or candidate_execution.packet_kind != "sealed"
        or candidate_execution.dataset_id != packet.dataset_id
        or candidate_execution.dataset_revision != packet.dataset_revision
        or candidate_execution.packet_receipt_sha256 != packet.packet_receipt_sha256
        or candidate_execution.layout_truth_receipt_sha256
        != layout_truth.layout_truth_receipt_sha256
        or candidate_execution.layout_truth_parser_spec_sha256 != layout_truth.parser_spec_sha256
        or candidate_execution.engine_kind != engine_kind
        or candidate_execution.extractor_revision != extractor_revision
        or candidate_execution.configuration_sha256 != configuration_sha256
        or candidate_execution.model_inventory_sha256 != model_inventory_sha256
        or candidate_execution.candidate_parser_revision != parser_revision
        or candidate_execution.candidate_bundle_sha256 != candidate_output.candidate_bundle_sha256
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
        "candidate_execution_receipt_sha256": (
            candidate_execution.candidate_execution_receipt_sha256
        ),
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
    candidate_execution: SourceAttestedSyntheticCandidateExecutionV1,
    engine_kind: EngineKind,
    extractor_revision: str,
    configuration_sha256: str,
    model_inventory_sha256: str,
    parser_revision: str,
    development_selection: SourceAttestedSyntheticDevelopmentSelectionV1,
    sealed_run: SourceAttestedSyntheticSealedRunV1,
) -> None:
    _require_source_attested_candidate_parser_revision(parser_revision)
    _validate_development_selection_for_sealed(
        selection=development_selection,
        packet=packet,
        engine_kind=engine_kind,
        extractor_revision=extractor_revision,
        configuration_sha256=configuration_sha256,
        model_inventory_sha256=model_inventory_sha256,
        parser_revision=parser_revision,
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
        or sealed_run.candidate_execution_receipt_sha256
        != candidate_execution.candidate_execution_receipt_sha256
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
    "SOURCE_ATTESTED_CANDIDATE_PARSER_REVISION",
    "SOURCE_ATTESTED_EVALUATION_BASIS",
    "SOURCE_ATTESTED_EVALUATION_SCOPE",
    "SOURCE_ATTESTED_REQUIRED_FIELDS",
    "SOURCE_ATTESTED_SUPERVISED_TEMPLATE_ENGINE_KIND",
    "SOURCE_ATTESTED_SUPERVISED_TEMPLATE_EXTRACTOR_REVISION",
    "EXPECTED_SOURCE_ATTESTED_SYNTHETIC_DATASET_SCOPE",
    "SOURCE_LAYOUT_COMPILER_REVISION",
    "SOURCE_LAYOUT_PARSER_SPEC_REVISION",
    "SOURCE_LAYOUT_TRUTH_KIND",
    "SourceAttestedSyntheticCandidateExecutionV1",
    "SourceAttestedSyntheticCandidateRescoreV1",
    "SourceAttestedSyntheticDatasetScopeV1",
    "SourceAttestedSyntheticGoldRecordV1",
    "SourceAttestedSyntheticLayoutTruthV1",
    "SourceAttestedSyntheticDevelopmentSelectionV1",
    "SourceAttestedSyntheticFieldMetricDenominatorV1",
    "SourceAttestedSyntheticOcrEvaluationResultV1",
    "SourceAttestedSyntheticOcrMetricsV1",
    "SourceAttestedSyntheticPdfTruthError",
    "SourceAttestedSyntheticSealedRunV1",
    "build_source_attested_synthetic_candidate_execution",
    "build_source_attested_synthetic_development_selection",
    "build_source_attested_synthetic_sealed_run",
    "compile_source_attested_synthetic_pdf_layout_truth",
    "candidate_execution_receipt_sha256",
    "development_selection_receipt_sha256",
    "layout_truth_receipt_sha256",
    "read_canonical_source_attested_synthetic_development_selection",
    "read_canonical_source_attested_synthetic_candidate_bundle",
    "read_canonical_source_attested_synthetic_candidate_execution",
    "read_canonical_source_attested_synthetic_dataset_receipt",
    "read_canonical_source_attested_synthetic_evaluation_result",
    "read_canonical_source_attested_synthetic_layout_truth",
    "read_canonical_source_attested_synthetic_packet",
    "read_canonical_source_attested_synthetic_sealed_run",
    "recompute_source_attested_synthetic_pdf_candidate_metrics",
    "score_source_attested_synthetic_pdf_candidate_output",
    "sealed_run_receipt_sha256",
    "select_best_source_attested_synthetic_development_result",
    "source_attested_evaluation_result_sha256",
    "source_layout_parser_spec_sha256",
    "validate_source_attested_layout_truth_for_evaluation",
    "validate_source_attested_synthetic_candidate_execution",
    "validate_source_attested_synthetic_dataset_scope",
    "validate_source_attested_synthetic_packet_scope",
    "verify_source_attested_synthetic_pdf_evaluation_replay",
    "write_source_attested_synthetic_development_selection",
    "write_source_attested_synthetic_candidate_bundle",
    "write_source_attested_synthetic_candidate_execution",
    "write_source_attested_synthetic_dataset_receipt",
    "write_source_attested_synthetic_evaluation_result",
    "write_source_attested_synthetic_layout_truth",
    "write_source_attested_synthetic_packet",
    "write_source_attested_synthetic_sealed_run",
]
