"""Offline runner for synthetic-PDF OCR candidates.

This module is intentionally outside the HTTP document-preview and review
paths.  It reads only immutable local PDF receipts, packets, runtime sidecars,
and locally provisioned model files.  It never receives Gold, CSV ground truth,
case data, or a network-enabled adapter.  Its content-bearing output stays
below ``_local/data/document-preview`` and is labelled as an *evaluation
candidate*, never as an approved preview.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import re
import socket
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Literal, Protocol, Sequence

from pydantic import ValidationError

from customs_ai_agent.evaluation.document_gold import (
    DatasetAssetReceiptV1,
    DatasetReceiptV1,
    receipt_sha256,
)
from customs_ai_agent.evaluation.document_ocr_quality import (
    CandidateDocumentOutputV1,
    CandidateFieldV1,
    CandidateSpanV1,
    CandidateTableCellV1,
    EngineKind,
    EvaluationPacketV2,
    SyntheticDocumentType,
    SyntheticPdfCandidateOutputBundleV1,
    candidate_bundle_sha256,
    canonical_json_file_bytes,
    normalize_field_value,
    verify_receipt_source_assets,
)
from customs_ai_agent.schemas.document_extraction import (
    DocumentExtractionRequestMetadataV1,
    PreviewCandidateV1,
    PreviewInputV1,
)
from customs_ai_agent.service.document_extraction import DocumentPreviewError
from customs_ai_agent.service.document_preview_runtime import (
    DoclingPdfPreviewAdapter,
    DocumentPreviewRuntimeError,
    ModelInventoryV1,
    PaddleOcrPdfPreviewAdapter,
    PdfTextLayerPreviewAdapter,
    RuntimeConfigurationV1,
    local_files_only,
    parse_model_inventory,
    parse_runtime_configuration,
    validate_document_preview_root,
    verify_configured_model_directory_artifacts,
    verify_inventory_packages,
)


_SHA256 = re.compile(r"^[a-f0-9]{64}$")
# A complete immutable receipt enumerates every PDF in the fixed 6,000-asset
# synthetic corpus. It is metadata only, but is larger than the compact packet
# and runtime sidecars. Keep its explicit 8 MiB ceiling aligned with the
# runtime's source-attested receipt reader; do not weaken the 1 MiB limit for
# packets or sidecars merely because the receipt has a larger inventory.
_MAX_DATASET_RECEIPT_BYTES = 8 * 1024 * 1024
_MAX_JSON_BYTES = 1 * 1024 * 1024
_MAX_SOURCE_BYTES = 32 * 1024 * 1024
_PROFILE_MODE: Literal["offline_synthetic_pdf_evaluation"] = "offline_synthetic_pdf_evaluation"
# v5 binds the visible label/value transform and the direct-Docling five-column
# table projection.  It is deliberately a candidate-only revision: historical
# bundles cannot be relabelled as outputs from the stricter layout parser.
SYNTHETIC_PDF_LABEL_NORMALIZER_REVISION = "synthetic-pdf-visible-layout-normalizer-v5"


class SyntheticPdfCandidateRunnerError(ValueError):
    """Content-free local failure safe for an operator summary or a test."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class EvaluationRuntimeProfileV1:
    """Verified but deliberately unapproved local engine material."""

    document_preview_root: Path
    engine_kind: EngineKind
    extractor_revision: str
    configuration_sha256: str
    model_inventory_sha256: str
    configuration: RuntimeConfigurationV1
    inventory: ModelInventoryV1


@dataclass(frozen=True, slots=True)
class CandidateExtractionV1:
    """Raw local adapter output before the synthetic label normalizer runs."""

    spans: tuple[CandidateSpanV1, ...]
    table_cells: tuple[CandidateTableCellV1, ...] = ()


def select_synthetic_pdf_candidate_table_cells(
    *,
    table_cells: Sequence[CandidateTableCellV1],
) -> list[CandidateTableCellV1]:
    """Select one direct Docling table by coordinates, never by its text.

    This is intentionally a candidate-only structural projection.  A target
    has exactly five observed cells in each of row 0 and row 1, at columns
    0--4.  Later rows may exist only as partial footer rows: each is a
    non-empty strict subset of those same five columns.  A second complete
    five-column row, a row/column gap, a sixth column, or duplicate cell
    coordinate rejects that table.  Zero or multiple structural targets are
    omitted rather than guessed.

    The selected header/body pair is reindexed to the stable page-one/table-one
    slot.  The predicate reads neither document type nor cell text, so it
    cannot depend on raw header labels, Gold, CSV, source truth, or a PDF
    content stream.
    """

    targets: list[tuple[CandidateTableCellV1, ...]] = []
    for _table_key, table in sorted(_direct_docling_table_cells_by_coordinate(table_cells).items()):
        target = _direct_docling_five_column_target_pair(table)
        if target is not None:
            targets.append(target)
    if len(targets) != 1:
        return []
    return _reindexed_direct_docling_target_pair(cells=targets[0])


class SyntheticPdfCandidateAdapter(Protocol):
    """Evaluation-only adapter boundary; it has no Gold or approval access."""

    def extract_pdf(self, preview_input: PreviewInputV1) -> CandidateExtractionV1: ...


def _candidate_from_preview(candidate: PreviewCandidateV1 | None) -> CandidateExtractionV1:
    if candidate is None:
        return CandidateExtractionV1(spans=())
    spans: list[CandidateSpanV1] = []
    for page in candidate.pages:
        for span in page.spans:
            spans.append(
                CandidateSpanV1(
                    page_number=page.page_number,
                    text=span.text,
                    bbox=span.bbox,
                    reading_order=span.reading_order,
                )
            )
    return CandidateExtractionV1(spans=tuple(spans))


@dataclass(slots=True)
class _PdfTextLayerCandidateAdapter:
    adapter: PdfTextLayerPreviewAdapter

    def extract_pdf(self, preview_input: PreviewInputV1) -> CandidateExtractionV1:
        # Reuse the native adapter's one visitor conversion rather than
        # deriving candidate spans from the public preview wire projection.
        # Internal anchors deliberately stay out of CandidateSpanV1 until a
        # local-only geometry schema can represent points without inventing a
        # rectangle.
        return CandidateExtractionV1(
            spans=tuple(
                CandidateSpanV1(
                    page_number=fragment.page_number,
                    text=fragment.text,
                    reading_order=fragment.ordinal,
                )
                for fragment in self.adapter.extract_pdf_text_layer_fragments(preview_input)
            )
        )


@dataclass(slots=True)
class _PaddleOcrCandidateAdapter:
    adapter: PaddleOcrPdfPreviewAdapter

    def extract_pdf(self, preview_input: PreviewInputV1) -> CandidateExtractionV1:
        return _candidate_from_preview(self.adapter.extract_raster_or_image(preview_input))


@dataclass(slots=True)
class _DoclingCandidateAdapter:
    adapter: DoclingPdfPreviewAdapter

    def preflight_native_runtime(self) -> None:
        """Fail before a selected PDF is loaded when Docling is unusable."""

        self.adapter.verify_native_runtime()

    def extract_pdf(self, preview_input: PreviewInputV1) -> CandidateExtractionV1:
        # Preview and candidate evaluation consume the same direct Docling
        # object-model projection.  Do not re-parse a document-global Markdown
        # export here: that loses page provenance and can invent table layout.
        extraction = self.adapter.extract_structured_pdf(preview_input)
        spans = tuple(
            CandidateSpanV1(
                page_number=page.page_number,
                text=span.text,
                bbox=span.bbox,
                reading_order=span.reading_order,
            )
            for page in extraction.pages
            for span in page.spans
        )
        table_cells = tuple(
            CandidateTableCellV1(
                page_number=cell.page_number,
                table_index=cell.table_index,
                row_index=cell.row_index,
                column_index=cell.column_index,
                text=cell.text,
            )
            for cell in extraction.table_cells
        )
        return CandidateExtractionV1(spans=spans, table_cells=table_cells)


def build_default_synthetic_pdf_candidate_adapter(
    profile: EvaluationRuntimeProfileV1,
) -> SyntheticPdfCandidateAdapter:
    """Create one direct engine adapter with no text-layer fallback.

    PaddleOCR and Docling are deliberately invoked as their own candidates.
    A layered runtime fallback would make their development scores ambiguous.
    """

    if profile.engine_kind == "pdf_text_layer":
        return _PdfTextLayerCandidateAdapter(
            PdfTextLayerPreviewAdapter(extractor_revision=profile.extractor_revision)
        )
    if profile.engine_kind == "paddleocr":
        return _PaddleOcrCandidateAdapter(
            PaddleOcrPdfPreviewAdapter(
                extractor_revision=profile.extractor_revision,
                model_root=profile.document_preview_root,
                options=profile.configuration.options,
            )
        )
    if profile.engine_kind == "docling":
        return _DoclingCandidateAdapter(
            DoclingPdfPreviewAdapter(
                extractor_revision=profile.extractor_revision,
                model_root=profile.document_preview_root,
                options=profile.configuration.options,
            )
        )
    raise SyntheticPdfCandidateRunnerError("candidate_engine_kind_invalid")


def _canonical_runtime_sidecar_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _path_is_below_local_data(path: Path) -> bool:
    parts = [part.casefold() for part in path.parts]
    return any(parts[index : index + 2] == ["_local", "data"] for index in range(len(parts) - 1))


def _reject_symlink_components(path: Path, *, code: str) -> None:
    """Reject a link at any existing component, including a broken leaf link."""

    absolute = path.absolute()
    current = Path(absolute.anchor)
    for component in absolute.parts[1:]:
        current /= component
        try:
            if current.is_symlink():
                raise SyntheticPdfCandidateRunnerError(code)
        except OSError as error:
            raise SyntheticPdfCandidateRunnerError(code) from error


def _require_local_data_file(path: Path, *, limit: int, code: str) -> tuple[Path, bytes]:
    _reject_symlink_components(path, code=code)
    try:
        resolved = path.resolve(strict=True)
        metadata = resolved.stat()
        if (
            not _path_is_below_local_data(resolved)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size < 1
            or metadata.st_size > limit
        ):
            raise SyntheticPdfCandidateRunnerError(code)
        with resolved.open("rb") as stream:
            raw = stream.read(limit + 1)
    except SyntheticPdfCandidateRunnerError:
        raise
    except OSError as error:
        raise SyntheticPdfCandidateRunnerError(code) from error
    if len(raw) != metadata.st_size or len(raw) > limit:
        raise SyntheticPdfCandidateRunnerError(code)
    return resolved, raw


def _require_local_data_directory(path: Path, *, code: str) -> Path:
    _reject_symlink_components(path, code=code)
    try:
        resolved = path.resolve(strict=True)
        if not _path_is_below_local_data(resolved) or not resolved.is_dir():
            raise SyntheticPdfCandidateRunnerError(code)
    except SyntheticPdfCandidateRunnerError:
        raise
    except OSError as error:
        raise SyntheticPdfCandidateRunnerError(code) from error
    return resolved


def _runtime_sidecar(
    *, root: Path, path: Path, directory_name: str, code: str
) -> tuple[Path, bytes, str]:
    _reject_symlink_components(path, code=code)
    try:
        resolved = path.resolve(strict=True)
        expected_parent = (root / directory_name).resolve(strict=True)
    except OSError as error:
        raise SyntheticPdfCandidateRunnerError(code) from error
    if resolved.parent != expected_parent:
        raise SyntheticPdfCandidateRunnerError(code)
    resolved, raw = _require_local_data_file(resolved, limit=_MAX_JSON_BYTES, code=code)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise SyntheticPdfCandidateRunnerError(code) from error
    if not isinstance(value, dict) or raw != _canonical_runtime_sidecar_bytes(value):
        raise SyntheticPdfCandidateRunnerError(code)
    digest = hashlib.sha256(raw).hexdigest()
    if resolved.name != f"{digest}.json":
        raise SyntheticPdfCandidateRunnerError(code)
    return resolved, raw, digest


def load_unapproved_local_candidate_profile(
    *,
    document_preview_root: Path,
    configuration_path: Path,
    inventory_path: Path,
    engine_kind: EngineKind,
    package_version_reader: Callable[[str], str] = importlib.metadata.version,
) -> EvaluationRuntimeProfileV1:
    """Validate provisioned sidecars without reading an active approval pointer."""

    try:
        root = validate_document_preview_root(document_preview_root, create=False)
    except DocumentPreviewRuntimeError as error:
        raise SyntheticPdfCandidateRunnerError("candidate_runtime_root_invalid") from error
    _configuration_path, configuration_raw, configuration_sha256 = _runtime_sidecar(
        root=root,
        path=configuration_path,
        directory_name="runtime-configurations",
        code="candidate_runtime_configuration_invalid",
    )
    _inventory_path, inventory_raw, model_inventory_sha256 = _runtime_sidecar(
        root=root,
        path=inventory_path,
        directory_name="model-inventories",
        code="candidate_model_inventory_invalid",
    )
    try:
        configuration = parse_runtime_configuration(configuration_raw)
        inventory = parse_model_inventory(inventory_raw)
        verify_inventory_packages(inventory, package_version_reader=package_version_reader)
    except DocumentPreviewRuntimeError as error:
        raise SyntheticPdfCandidateRunnerError("candidate_runtime_profile_invalid") from error
    if (
        configuration.engine_kind != engine_kind
        or inventory.engine_kind != engine_kind
        or configuration.engine_kind != inventory.engine_kind
    ):
        raise SyntheticPdfCandidateRunnerError("candidate_engine_profile_mismatch")
    try:
        verify_configured_model_directory_artifacts(root, configuration, inventory)
    except DocumentPreviewRuntimeError as error:
        code = (
            "candidate_runtime_configuration_invalid"
            if error.code == "document_preview_configuration_invalid"
            else "candidate_model_inventory_invalid"
        )
        raise SyntheticPdfCandidateRunnerError(code) from error
    return EvaluationRuntimeProfileV1(
        document_preview_root=root,
        engine_kind=engine_kind,
        extractor_revision=configuration.extractor_revision,
        configuration_sha256=configuration_sha256,
        model_inventory_sha256=model_inventory_sha256,
        configuration=configuration,
        inventory=inventory,
    )


def load_local_synthetic_pdf_evaluation_inputs(
    *,
    receipt_path: Path,
    packet_path: Path,
    receipt_code: str = "candidate_receipt_invalid",
    packet_code: str = "candidate_packet_invalid",
    validation_code: str = "candidate_packet_or_receipt_invalid",
) -> tuple[DatasetReceiptV1, EvaluationPacketV2]:
    """Read a bounded local receipt and packet without following links.

    The receipt records the complete corpus and therefore has its own 8 MiB
    bound. The selected packet remains compact and retains the 1 MiB bound.
    Callers choose only content-free error codes, so this helper can be used
    by both the candidate runner and the source-attested evaluator without
    exposing document or OCR text.
    """

    _receipt_file, receipt_raw = _require_local_data_file(
        receipt_path, limit=_MAX_DATASET_RECEIPT_BYTES, code=receipt_code
    )
    _packet_file, packet_raw = _require_local_data_file(
        packet_path, limit=_MAX_JSON_BYTES, code=packet_code
    )
    try:
        receipt = DatasetReceiptV1.model_validate_json(receipt_raw)
        packet = EvaluationPacketV2.model_validate_json(packet_raw)
    except ValidationError as error:
        raise SyntheticPdfCandidateRunnerError(validation_code) from error
    return receipt, packet


def _load_receipt_and_packet(
    *, receipt_path: Path, packet_path: Path, source_directory: Path
) -> tuple[DatasetReceiptV1, EvaluationPacketV2, Path]:
    receipt, packet = load_local_synthetic_pdf_evaluation_inputs(
        receipt_path=receipt_path,
        packet_path=packet_path,
    )
    if (
        receipt_sha256(receipt) != packet.receipt_sha256
        or receipt.dataset_id != packet.dataset_id
        or receipt.dataset_revision != packet.dataset_revision
    ):
        raise SyntheticPdfCandidateRunnerError("candidate_packet_receipt_mismatch")
    source_root = _require_local_data_directory(
        source_directory, code="candidate_source_directory_invalid"
    )
    try:
        verify_receipt_source_assets(receipt=receipt, source_directory=source_root)
    except (OSError, ValueError) as error:
        raise SyntheticPdfCandidateRunnerError("candidate_source_receipt_invalid") from error
    receipt_assets = {asset.asset_id: asset for asset in receipt.assets}
    if len(receipt_assets) != len(receipt.assets):
        raise SyntheticPdfCandidateRunnerError("candidate_receipt_invalid")
    for item in packet.items:
        asset = receipt_assets.get(item.asset_id)
        if (
            asset is None
            or asset.media_type != "application/pdf"
            or asset.source_role != "preview_candidate"
            or asset.sha256 != item.source_sha256
        ):
            raise SyntheticPdfCandidateRunnerError("candidate_packet_source_mismatch")
    return receipt, packet, source_root


def _load_selected_pdf(*, source_root: Path, asset: DatasetAssetReceiptV1) -> bytes:
    path = source_root / Path(asset.relative_path)
    _reject_symlink_components(path, code="candidate_source_asset_invalid")
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(source_root)
        metadata = resolved.stat()
        if (
            not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size != asset.byte_size
            or metadata.st_size > _MAX_SOURCE_BYTES
        ):
            raise SyntheticPdfCandidateRunnerError("candidate_source_asset_invalid")
        with resolved.open("rb") as stream:
            content = stream.read(_MAX_SOURCE_BYTES + 1)
    except SyntheticPdfCandidateRunnerError:
        raise
    except (OSError, ValueError) as error:
        raise SyntheticPdfCandidateRunnerError("candidate_source_asset_invalid") from error
    if len(content) != asset.byte_size or hashlib.sha256(content).hexdigest() != asset.sha256:
        raise SyntheticPdfCandidateRunnerError("candidate_source_asset_invalid")
    return content


# This is deliberately a small, candidate-only visible-layout specification.
# It has no source-content-stream, Gold, CSV, or cross-document input.  Names
# below are *visible form labels and visible table headers*, not a coordinate
# profile or a source-of-truth mapping.  A missing/ambiguous output stays
# missing: the independent evaluator consequently cannot pass its field gate.
_VisibleNormalizer = Literal["text-v1", "numeric-v1", "identifier-v1"]


@dataclass(frozen=True, slots=True)
class _VisibleFieldSpec:
    field_name: str
    normalizer: _VisibleNormalizer
    labels: tuple[str, ...]
    # Adjacent label/value lines are accepted only for explicitly form-like
    # fields.  Table captions otherwise could consume a neighbouring cell.
    accepts_adjacent_value: bool = False


# The field universe is the actually visible synthetic PDF contract: B/L 13,
# Invoice 15, Packing List 12.  Packing List intentionally excludes the
# non-rendered ``containers`` manifest column.
_VISIBLE_LAYOUT_FIELDS: dict[SyntheticDocumentType, tuple[_VisibleFieldSpec, ...]] = {
    "bill_of_lading": (
        _VisibleFieldSpec("exporter_name", "text-v1", ("shipper", "exporter"), True),
        _VisibleFieldSpec("consignee", "text-v1", ("consignee", "to applicant / consignee"), True),
        _VisibleFieldSpec("load_port", "text-v1", ("port of loading", "load port"), True),
        _VisibleFieldSpec("disch_port", "text-v1", ("port of discharge", "discharge port"), True),
        _VisibleFieldSpec("final_dest", "text-v1", ("final destination",), True),
        _VisibleFieldSpec(
            "vessel", "text-v1", ("vessel / flight", "vessel/flight", "vessel"), True
        ),
        _VisibleFieldSpec("voyage", "identifier-v1", ("voyage no", "voyage"), True),
        _VisibleFieldSpec("bl_no", "identifier-v1", ("bill of lading no", "b/l no", "bl no"), True),
        _VisibleFieldSpec("container_no", "identifier-v1", ("container no", "container number")),
        _VisibleFieldSpec("seal_no", "identifier-v1", ("seal no", "seal number")),
        _VisibleFieldSpec("good", "text-v1", ("goods description", "description of goods")),
        _VisibleFieldSpec("gross_weight", "numeric-v1", ("gross weight",)),
        _VisibleFieldSpec(
            "containers", "numeric-v1", ("no of containers", "number of containers", "containers")
        ),
    ),
    "invoice": (
        _VisibleFieldSpec("exporter_name", "text-v1", ("exporter", "seller", "shipper"), True),
        # ``Notify Party`` is the visible invoice form label for this field;
        # no consignee/B/L sibling or data file is consulted.
        _VisibleFieldSpec("importer_name", "text-v1", ("notify party", "importer", "buyer"), True),
        _VisibleFieldSpec("consignee", "text-v1", ("consignee", "to applicant / consignee"), True),
        _VisibleFieldSpec("load_port", "text-v1", ("port of loading", "load port"), True),
        _VisibleFieldSpec("disch_port", "text-v1", ("port of discharge", "discharge port"), True),
        _VisibleFieldSpec(
            "vessel", "text-v1", ("vessel / flight", "vessel/flight", "vessel"), True
        ),
        _VisibleFieldSpec(
            "inv_no", "identifier-v1", ("invoice number", "invoice no", "inv no"), True
        ),
        _VisibleFieldSpec("inv_date", "identifier-v1", ("invoice date",), True),
        _VisibleFieldSpec("lc_no", "identifier-v1", ("letter of credit", "l/c no", "lc no"), True),
        _VisibleFieldSpec("good", "text-v1", ("goods description", "description of goods")),
        _VisibleFieldSpec("unit", "text-v1", ("unit",)),
        _VisibleFieldSpec("qty", "numeric-v1", ("quantity", "qty")),
        _VisibleFieldSpec("unit_price", "numeric-v1", ("unit price",)),
        _VisibleFieldSpec("amount", "numeric-v1", ("amount",)),
        _VisibleFieldSpec("total", "numeric-v1", ("total",), True),
    ),
    "packing_list": (
        _VisibleFieldSpec("exporter_name", "text-v1", ("exporter", "seller", "shipper"), True),
        _VisibleFieldSpec("consignee", "text-v1", ("consignee", "to applicant / consignee"), True),
        _VisibleFieldSpec(
            "inv_no", "identifier-v1", ("no of invoice", "invoice no", "inv no"), True
        ),
        _VisibleFieldSpec(
            "lc_no", "identifier-v1", ("no of l/c", "letter of credit", "l/c no", "lc no"), True
        ),
        _VisibleFieldSpec("good", "text-v1", ("goods description", "description of goods")),
        _VisibleFieldSpec("unit", "text-v1", ("unit",)),
        _VisibleFieldSpec("qty", "numeric-v1", ("quantity", "qty")),
        _VisibleFieldSpec("cartons", "numeric-v1", ("no of cartons / pkgs", "cartons", "pkgs")),
        _VisibleFieldSpec("net_weight", "numeric-v1", ("net weight",)),
        _VisibleFieldSpec("gross_weight", "numeric-v1", ("gross weight",)),
        _VisibleFieldSpec("cbm", "numeric-v1", ("cbm", "volume", "measurement")),
        _VisibleFieldSpec("net_gross_weight", "text-v1", ("net/gross weight",)),
    ),
}


# Table keys are intentionally separate from form labels.  Matching requires a
# complete normalized header, never a column position, bbox, or a value from a
# different document.
_VISIBLE_TABLE_HEADERS: dict[SyntheticDocumentType, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "bill_of_lading": (
        ("container_no", ("container no", "container number")),
        ("seal_no", ("seal no", "seal number")),
        (
            "containers",
            (
                "no and kind of packages/containers",
                "no kind of packages/containers",
                "no of packages/containers",
                "packages/containers",
            ),
        ),
        ("good", ("goods description", "description of goods", "description")),
        ("gross_weight", ("gross weight",)),
    ),
    "invoice": (
        ("good", ("description of goods", "goods description", "description")),
        ("qty_unit", ("quantity", "qty", "quantity unit")),
        ("unit_price", ("unit price",)),
        ("amount", ("amount",)),
    ),
    "packing_list": (
        (
            "cartons",
            ("marks/numbers/pkgs", "marks numbers pkgs", "marks numbers packages", "cartons"),
        ),
        ("good_with_quantity", ("goods description", "description of goods", "description")),
        ("qty_unit", ("quantity", "qty", "quantity unit")),
        ("net_weight", ("net weight",)),
        ("gross_weight", ("gross weight",)),
        ("cbm", ("measurement", "cbm", "volume")),
    ),
}


_INLINE_LABEL_VALUE = re.compile(r"^\s*(?P<label>.+?)\s*(?::|#|\s[-–—]\s)\s*(?P<value>\S.*?)\s*$")
_HTML_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)
_QUANTITY_TOKEN = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_QUANTITY_WITH_UNIT = re.compile(rf"^\s*(?P<quantity>{_QUANTITY_TOKEN})\s+(?P<unit>[A-Za-z]+)\s*$")
_QUANTITY_ONLY = re.compile(rf"^\s*(?P<quantity>{_QUANTITY_TOKEN})\s*$")
_UNIT_ONLY = re.compile(r"^\s*(?P<unit>[A-Za-z]+)\s*$")
# This deliberately finds only standalone token pairs.  It is used solely by
# the Packing List table-cell fallback below, where exactly one *terminal*
# pair is required; label matching and emitted values never use this rewrite.
_TERMINAL_QUANTITY_WITH_ASCII_UNIT = re.compile(
    rf"(?<!\S)(?P<quantity>{_QUANTITY_TOKEN})\s+(?P<unit>[A-Za-z]+)(?!\S)"
)


@dataclass(frozen=True, slots=True)
class _VisibleLine:
    page_number: int
    reading_order: int
    line_index: int
    text: str


def _normalized_visible_label(value: str) -> str:
    """Normalize only form/header typography, never an emitted field value."""

    normalized = re.sub(r"[^a-z0-9/]+", " ", value.casefold())
    # Labels such as ``L/C`` and ``L / C`` identify the same rendered form
    # header.  This is intentionally label-only: a field value containing a
    # slash keeps the candidate engine's spelling unchanged.
    normalized = re.sub(r"\s*/\s*", "/", normalized)
    return " ".join(normalized.split())


def _visible_value(value: str) -> str | None:
    compact = " ".join(
        _HTML_BREAK.sub("\n", value).replace("\r\n", "\n").replace("\r", "\n").split()
    )
    return compact or None


def _visible_table_value(value: str) -> str | None:
    """Normalize table whitespace while preserving an emitted cell line break."""

    lines = [
        " ".join(line.split())
        for line in _HTML_BREAK.sub("\n", value)
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .split("\n")
    ]
    nonempty = [line for line in lines if line]
    return "\n".join(nonempty) or None


def _visible_lines(spans: Sequence[CandidateSpanV1]) -> list[_VisibleLine]:
    lines: list[_VisibleLine] = []
    for span in sorted(spans, key=lambda item: (item.page_number, item.reading_order)):
        for line_index, raw in enumerate(
            span.text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
        ):
            text = raw.strip()
            if text:
                lines.append(
                    _VisibleLine(
                        page_number=span.page_number,
                        reading_order=span.reading_order,
                        line_index=line_index,
                        text=text,
                    )
                )
    return lines


def _field_specification(document_type: SyntheticDocumentType) -> tuple[_VisibleFieldSpec, ...]:
    specification = _VISIBLE_LAYOUT_FIELDS.get(document_type)
    if specification is None:
        raise SyntheticPdfCandidateRunnerError("candidate_document_type_invalid")
    return specification


def _matching_visible_field_names(
    *, document_type: SyntheticDocumentType, label: str
) -> tuple[str, ...]:
    normalized = _normalized_visible_label(label)
    if not normalized:
        return ()
    return tuple(
        specification.field_name
        for specification in _field_specification(document_type)
        if normalized in {_normalized_visible_label(alias) for alias in specification.labels}
    )


def _field_spec_by_name(
    *, document_type: SyntheticDocumentType, field_name: str
) -> _VisibleFieldSpec:
    for specification in _field_specification(document_type):
        if specification.field_name == field_name:
            return specification
    raise SyntheticPdfCandidateRunnerError("candidate_visible_field_invalid")


def _append_quantity_and_unit(*, candidates: dict[str, list[str]], value: str) -> None:
    """Emit a deterministic quantity/unit split only for an exact visible token pair."""

    pair = _QUANTITY_WITH_UNIT.fullmatch(value)
    if pair is not None:
        candidates.setdefault("qty", []).append(pair.group("quantity"))
        candidates.setdefault("unit", []).append(pair.group("unit"))
        return
    quantity = _QUANTITY_ONLY.fullmatch(value)
    if quantity is not None:
        candidates.setdefault("qty", []).append(quantity.group("quantity"))


def _append_field_candidate(
    *, candidates: dict[str, list[str]], field_name: str, raw_value: str
) -> None:
    value = _visible_value(raw_value)
    if value is None:
        return
    if field_name == "qty":
        _append_quantity_and_unit(candidates=candidates, value=value)
        return
    if field_name == "unit":
        unit = _UNIT_ONLY.fullmatch(value)
        if unit is not None:
            candidates.setdefault("unit", []).append(unit.group("unit"))
        return
    candidates.setdefault(field_name, []).append(value)


def _matching_table_header(*, document_type: SyntheticDocumentType, header: str) -> str | None:
    normalized = _normalized_visible_label(header)
    matched = {
        semantic
        for semantic, aliases in _VISIBLE_TABLE_HEADERS[document_type]
        if normalized in {_normalized_visible_label(alias) for alias in aliases}
    }
    return next(iter(matched)) if len(matched) == 1 else None


def _table_cells_by_layout(
    table_cells: Sequence[CandidateTableCellV1],
) -> dict[tuple[int, int], dict[tuple[int, int], str]]:
    tables: dict[tuple[int, int], dict[tuple[int, int], str]] = {}
    invalid_tables: set[tuple[int, int]] = set()
    for cell in table_cells:
        table_key = (cell.page_number, cell.table_index)
        coordinate = (cell.row_index, cell.column_index)
        table = tables.setdefault(table_key, {})
        if coordinate in table:
            invalid_tables.add(table_key)
            continue
        value = _visible_table_value(cell.text)
        if value is None:
            invalid_tables.add(table_key)
            continue
        table[coordinate] = value
    for table_key in invalid_tables:
        tables.pop(table_key, None)
    return tables


def _direct_docling_table_cells_by_coordinate(
    table_cells: Sequence[CandidateTableCellV1],
) -> dict[tuple[int, int], dict[tuple[int, int], CandidateTableCellV1]]:
    """Group direct cells without looking at their raw values or labels."""

    tables: dict[tuple[int, int], dict[tuple[int, int], CandidateTableCellV1]] = {}
    invalid_tables: set[tuple[int, int]] = set()
    for cell in table_cells:
        table_key = (cell.page_number, cell.table_index)
        coordinate = (cell.row_index, cell.column_index)
        table = tables.setdefault(table_key, {})
        if coordinate in table:
            invalid_tables.add(table_key)
            continue
        table[coordinate] = cell
    for table_key in invalid_tables:
        tables.pop(table_key, None)
    return tables


def _direct_docling_five_column_target_pair(
    table: dict[tuple[int, int], CandidateTableCellV1],
) -> tuple[CandidateTableCellV1, ...] | None:
    """Return one exact five-column header/body pair, if the grid is safe.

    Direct Docling grids can retain footer fragments after the actual one-line
    item.  The first two rows must still be a complete 5x2 rectangle.  A
    footer is allowed only after that pair and only when it is structurally
    partial; the function intentionally does not inspect its values.
    """

    if not table:
        return None
    row_indexes = {row_index for row_index, _column_index in table}
    if row_indexes != set(range(max(row_indexes) + 1)) or not {0, 1}.issubset(row_indexes):
        return None
    expected_columns = set(range(5))
    for row_index in (0, 1):
        columns = {
            column_index
            for candidate_row_index, column_index in table
            if candidate_row_index == row_index
        }
        if columns != expected_columns:
            return None
    for row_index in range(2, max(row_indexes) + 1):
        columns = {
            column_index
            for candidate_row_index, column_index in table
            if candidate_row_index == row_index
        }
        if not columns or not columns < expected_columns:
            return None
    return tuple(
        table[(row_index, column_index)] for row_index in (0, 1) for column_index in range(5)
    )


def _single_row_table_header_map(
    *, document_type: SyntheticDocumentType, table: dict[tuple[int, int], str]
) -> dict[str, str] | None:
    """Return semantic header -> body value for one explicit one-line-item table.

    The synthetic layouts have exactly one visible line item.  More than one
    body row is deliberately not guessed or merged, because doing so would
    create a hidden line-item policy in the quality evaluator.
    """

    row_indexes = {row for row, _column in table}
    if row_indexes != {0, 1}:
        return None
    header_by_column = {column: text for (row, column), text in table.items() if row == 0}
    body_by_column = {column: text for (row, column), text in table.items() if row == 1}
    if not header_by_column or set(header_by_column) != set(body_by_column):
        return None
    semantic_columns: dict[str, int] = {}
    for column, header in header_by_column.items():
        semantic = _matching_table_header(document_type=document_type, header=header)
        if semantic is None:
            continue
        if semantic in semantic_columns:
            return None
        semantic_columns[semantic] = column
    return {semantic: body_by_column[column] for semantic, column in semantic_columns.items()}


def _append_packing_list_good_with_quantity(
    *, candidates: dict[str, list[str]], value: str
) -> None:
    """Split the synthetic Packing List item cell only when it is unambiguous.

    The normal layout has a visible line break between the good and its
    quantity/unit.  Some local Markdown output flattens that one cell.  For
    that strictly limited case, accept exactly one terminal numeric + ASCII
    unit token pair.  More than one pair, a non-terminal pair, a missing good,
    or a non-ASCII unit stays absent rather than guessing an item boundary.
    """

    fragments = [
        _visible_value(fragment)
        for fragment in _HTML_BREAK.sub("\n", value)
        .replace("\r\n", "\n")
        .replace("\r", "\n")
        .split("\n")
    ]
    nonempty = [fragment for fragment in fragments if fragment is not None]
    if len(nonempty) >= 2:
        quantity_unit = nonempty[-1]
        if _QUANTITY_WITH_UNIT.fullmatch(quantity_unit) is None:
            return
        good = _visible_value(" ".join(nonempty[:-1]))
        if good is None:
            return
        candidates.setdefault("good", []).append(good)
        _append_quantity_and_unit(candidates=candidates, value=quantity_unit)
        return
    if len(nonempty) != 1:
        return

    flattened = nonempty[0]
    terminal_pairs = list(_TERMINAL_QUANTITY_WITH_ASCII_UNIT.finditer(flattened))
    if len(terminal_pairs) != 1:
        return
    terminal_pair = terminal_pairs[0]
    if terminal_pair.end() != len(flattened):
        return
    good = _visible_value(flattened[: terminal_pair.start()])
    if good is None:
        return
    candidates.setdefault("good", []).append(good)
    _append_quantity_and_unit(candidates=candidates, value=terminal_pair.group(0))


def _append_table_candidates(
    *,
    document_type: SyntheticDocumentType,
    table_cells: Sequence[CandidateTableCellV1],
    candidates: dict[str, list[str]],
) -> None:
    for _table_key, table in sorted(_table_cells_by_layout(table_cells).items()):
        values = _single_row_table_header_map(document_type=document_type, table=table)
        if values is None:
            continue
        if document_type == "bill_of_lading":
            required = {"container_no", "seal_no", "containers", "good", "gross_weight"}
            if not required.issubset(values):
                continue
            for field_name in sorted(required):
                _append_field_candidate(
                    candidates=candidates, field_name=field_name, raw_value=values[field_name]
                )
            continue
        if document_type == "invoice":
            required = {"good", "qty_unit", "unit_price", "amount"}
            if not required.issubset(values):
                continue
            _append_field_candidate(
                candidates=candidates, field_name="good", raw_value=values["good"]
            )
            _append_quantity_and_unit(candidates=candidates, value=values["qty_unit"])
            _append_field_candidate(
                candidates=candidates, field_name="unit_price", raw_value=values["unit_price"]
            )
            _append_field_candidate(
                candidates=candidates, field_name="amount", raw_value=values["amount"]
            )
            continue
        required = {"cartons", "net_weight", "gross_weight", "cbm"}
        if not required.issubset(values) or not (
            "good_with_quantity" in values or {"good", "qty_unit"}.issubset(values)
        ):
            continue
        _append_field_candidate(
            candidates=candidates, field_name="cartons", raw_value=values["cartons"]
        )
        _append_field_candidate(
            candidates=candidates, field_name="net_weight", raw_value=values["net_weight"]
        )
        _append_field_candidate(
            candidates=candidates, field_name="gross_weight", raw_value=values["gross_weight"]
        )
        _append_field_candidate(candidates=candidates, field_name="cbm", raw_value=values["cbm"])
        net_gross = _visible_value(f"{values['net_weight']} / {values['gross_weight']}")
        if net_gross is not None:
            candidates.setdefault("net_gross_weight", []).append(net_gross)
        if "good_with_quantity" in values:
            _append_packing_list_good_with_quantity(
                candidates=candidates, value=values["good_with_quantity"]
            )
        else:
            _append_field_candidate(
                candidates=candidates, field_name="good", raw_value=values["good"]
            )
            _append_quantity_and_unit(candidates=candidates, value=values["qty_unit"])


def _append_span_candidates(
    *,
    document_type: SyntheticDocumentType,
    spans: Sequence[CandidateSpanV1],
    candidates: dict[str, list[str]],
) -> None:
    lines = _visible_lines(spans)
    for index, line in enumerate(lines):
        inline = _INLINE_LABEL_VALUE.fullmatch(line.text)
        if inline is not None:
            matched = _matching_visible_field_names(
                document_type=document_type, label=inline.group("label")
            )
            if len(matched) == 1:
                _append_field_candidate(
                    candidates=candidates,
                    field_name=matched[0],
                    raw_value=inline.group("value"),
                )
            continue
        matched = _matching_visible_field_names(document_type=document_type, label=line.text)
        if len(matched) != 1:
            continue
        specification = _field_spec_by_name(document_type=document_type, field_name=matched[0])
        if not specification.accepts_adjacent_value or index + 1 >= len(lines):
            continue
        next_line = lines[index + 1]
        # Do not cross a page or let a second label/table row serve as a value.
        if (
            next_line.page_number != line.page_number
            or next_line.text.startswith("|")
            or _matching_visible_field_names(document_type=document_type, label=next_line.text)
        ):
            continue
        _append_field_candidate(
            candidates=candidates,
            field_name=matched[0],
            raw_value=next_line.text,
        )


def normalize_synthetic_pdf_fields(
    *,
    document_type: SyntheticDocumentType,
    spans: Sequence[CandidateSpanV1],
    table_cells: Sequence[CandidateTableCellV1] = (),
) -> list[CandidateFieldV1]:
    """Derive only unambiguous visible label/value and header/body fields.

    The caller supplies raw candidate-engine spans and, if the engine emitted
    them, table cells.  This pure transform has no Gold, CSV, content-stream,
    source-truth, or cross-document parameter.  It does not turn an unknown
    layout into an exception or a guessed value: missing/ambiguous fields are
    absent, which makes the independent quality gate fail closed.
    """

    specification = _field_specification(document_type)
    candidates: dict[str, list[str]] = {}
    _append_span_candidates(document_type=document_type, spans=spans, candidates=candidates)
    _append_table_candidates(
        document_type=document_type, table_cells=table_cells, candidates=candidates
    )
    fields: list[CandidateFieldV1] = []
    for field in specification:
        values = candidates.get(field.field_name, [])
        canonical_values = {
            normalize_field_value(value, field.normalizer)
            for value in values
            if _visible_value(value) is not None
        }
        if len(canonical_values) != 1:
            continue
        # Preserve the first candidate-engine value after the minimal visible
        # transform.  Canonicalization above only decides equivalence; it never
        # substitutes a value from another artifact.
        fields.append(CandidateFieldV1(field_name=field.field_name, value=values[0]))
    return fields


@dataclass(frozen=True, slots=True)
class _DoclingMarkdownPipeTable:
    """One explicit pipe table before document-type target selection."""

    rows: tuple[tuple[str, ...], ...]


def _markdown_pipe_cells(line: str) -> list[str] | None:
    """Parse one complete Markdown pipe row without inferring table geometry."""

    stripped = line.strip()
    if len(stripped) < 3 or not stripped.startswith("|") or not stripped.endswith("|"):
        return None
    cells: list[str] = []
    current: list[str] = []
    escaped = False
    for character in stripped[1:-1]:
        if escaped:
            if character == "|":
                current.append(character)
            else:
                current.extend(("\\", character))
            escaped = False
            continue
        if character == "\\":
            escaped = True
        elif character == "|":
            cells.append("".join(current).strip())
            current = []
        else:
            current.append(character)
    if escaped:
        current.append("\\")
    cells.append("".join(current).strip())
    if not cells or any(not cell for cell in cells):
        return None
    return cells


def _is_markdown_separator_row(cells: Sequence[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell.replace(" ", "")) for cell in cells)


def _candidate_table_layout_to_rows(
    table: dict[tuple[int, int], str],
) -> _DoclingMarkdownPipeTable | None:
    """Use only a complete direct grid; never infer missing cell coordinates."""

    if not table:
        return None
    row_indexes = {row for row, _column in table}
    column_indexes = {column for _row, column in table}
    if (
        min(row_indexes, default=-1) != 0
        or min(column_indexes, default=-1) != 0
        or row_indexes != set(range(max(row_indexes) + 1))
        or column_indexes != set(range(max(column_indexes) + 1))
    ):
        return None
    rows: list[tuple[str, ...]] = []
    for row_index in range(max(row_indexes) + 1):
        row = tuple(
            table.get((row_index, column_index), "")
            for column_index in range(max(column_indexes) + 1)
        )
        if any(not value for value in row):
            return None
        rows.append(row)
    return _DoclingMarkdownPipeTable(rows=tuple(rows))


def _docling_table_is_complete_target(
    *, document_type: SyntheticDocumentType, table: _DoclingMarkdownPipeTable
) -> bool:
    """Accept only one explicit one-line-item table for the document contract."""

    layout = {
        (row_index, column_index): text
        for row_index, row in enumerate(table.rows)
        for column_index, text in enumerate(row)
    }
    values = _single_row_table_header_map(document_type=document_type, table=layout)
    if values is None:
        return False
    if document_type == "bill_of_lading":
        return {"container_no", "seal_no", "containers", "good", "gross_weight"}.issubset(values)
    if document_type == "invoice":
        return {"good", "qty_unit", "unit_price", "amount"}.issubset(values)
    return {"cartons", "net_weight", "gross_weight", "cbm"}.issubset(values) and (
        "good_with_quantity" in values or {"good", "qty_unit"}.issubset(values)
    )


def _parse_docling_markdown_pipe_tables(markdown: str) -> list[_DoclingMarkdownPipeTable]:
    """Read complete explicit pipe tables without assigning output coordinates."""

    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    tables: list[_DoclingMarkdownPipeTable] = []
    line_index = 0
    in_fenced_code = False
    while line_index + 1 < len(lines):
        line = lines[line_index]
        if line.strip().startswith("```"):
            in_fenced_code = not in_fenced_code
            line_index += 1
            continue
        if in_fenced_code:
            line_index += 1
            continue
        header = _markdown_pipe_cells(line)
        separator = _markdown_pipe_cells(lines[line_index + 1])
        if (
            header is None
            or separator is None
            or len(header) != len(separator)
            or not _is_markdown_separator_row(separator)
        ):
            line_index += 1
            continue
        data_rows: list[list[str]] = []
        next_index = line_index + 2
        while next_index < len(lines):
            row = _markdown_pipe_cells(lines[next_index])
            if row is None or len(row) != len(header):
                break
            data_rows.append(row)
            next_index += 1
        # Header-only or malformed blocks are intentionally ignored.  A
        # reviewer cannot verify a table body that the candidate did not emit.
        if not data_rows:
            line_index += 1
            continue
        tables.append(
            _DoclingMarkdownPipeTable(rows=tuple(tuple(row) for row in [header, *data_rows]))
        )
        line_index = next_index
    return tables


def _reindexed_docling_table_cells(
    *, table: _DoclingMarkdownPipeTable
) -> list[CandidateTableCellV1]:
    """Materialize the single selected table in the fixed Gold coordinate slot."""

    return [
        CandidateTableCellV1(
            page_number=1,
            table_index=1,
            row_index=row_index,
            column_index=column_index,
            text=text,
        )
        for row_index, row in enumerate(table.rows)
        for column_index, text in enumerate(row)
    ]


def _reindexed_direct_docling_target_pair(
    *, cells: Sequence[CandidateTableCellV1]
) -> list[CandidateTableCellV1]:
    """Copy the already-validated direct header/body pair into its fixed slot."""

    return [
        CandidateTableCellV1(
            page_number=1,
            table_index=1,
            row_index=cell.row_index,
            column_index=cell.column_index,
            text=cell.text,
        )
        for cell in cells
    ]


def normalize_synthetic_pdf_docling_table_cells(
    *, document_type: SyntheticDocumentType, spans: Sequence[CandidateSpanV1]
) -> list[CandidateTableCellV1]:
    """Select one complete document-type pipe table from page-one Markdown.

    The pinned runtime's Docling adapter exports one document-global Markdown
    stream and conservatively places every chunk on page 1.  Rejoining those
    chunks exactly restores that stream even when a 8,000-character boundary
    splits a row.  A candidate must have the complete visible document-type
    headers and exactly one body row.  If zero or more than one table meets
    that target, no cells are emitted.  Unrelated Markdown tables do not shift
    the selected table's coordinates.

    Coordinate convention shared with the local Gold reviewer:
    ``page_number=1``; the one selected target has ``table_index=1``;
    ``row_index=0`` is the header, the single data row is 1; ``column_index``
    is zero-based.  The Markdown separator row is not a cell.  The parser
    uses only the document type and candidate spans; it has no Gold, CSV, or
    source-truth input.
    """

    markdown = "".join(
        span.text
        for span in sorted(spans, key=lambda span: (span.page_number, span.reading_order))
        if span.page_number == 1
    )
    if not markdown:
        return []
    targets = [
        table
        for table in _parse_docling_markdown_pipe_tables(markdown)
        if _docling_table_is_complete_target(document_type=document_type, table=table)
    ]
    if len(targets) != 1:
        return []
    return _reindexed_docling_table_cells(table=targets[0])


def _build_candidate_bundle(
    *,
    packet: EvaluationPacketV2,
    profile: EvaluationRuntimeProfileV1,
    documents: Sequence[CandidateDocumentOutputV1],
) -> SyntheticPdfCandidateOutputBundleV1:
    payload: dict[str, object] = {
        "schema_version": "1.0",
        "artifact_kind": "synthetic_pdf_candidate_output",
        "profile_mode": _PROFILE_MODE,
        "parser_revision": SYNTHETIC_PDF_LABEL_NORMALIZER_REVISION,
        "dataset_id": packet.dataset_id,
        "dataset_revision": packet.dataset_revision,
        "packet_receipt_sha256": packet.packet_receipt_sha256,
        "engine_kind": profile.engine_kind,
        "extractor_revision": profile.extractor_revision,
        "configuration_sha256": profile.configuration_sha256,
        "model_inventory_sha256": profile.model_inventory_sha256,
        "documents": [document.model_dump(mode="json", warnings=False) for document in documents],
    }
    payload["candidate_bundle_sha256"] = candidate_bundle_sha256(payload)
    return SyntheticPdfCandidateOutputBundleV1.model_validate(payload)


def _require_candidate_output_directory(*, root: Path, output_directory: Path) -> Path:
    _reject_symlink_components(output_directory, code="candidate_output_directory_invalid")
    try:
        resolved = output_directory.resolve()
        resolved.relative_to(root / "candidate-outputs")
        resolved.mkdir(parents=True, exist_ok=True)
        if not resolved.is_dir() or resolved.is_symlink():
            raise SyntheticPdfCandidateRunnerError("candidate_output_directory_invalid")
        _reject_symlink_components(resolved, code="candidate_output_directory_invalid")
    except SyntheticPdfCandidateRunnerError:
        raise
    except (OSError, ValueError) as error:
        raise SyntheticPdfCandidateRunnerError("candidate_output_directory_invalid") from error
    return resolved


def write_synthetic_pdf_candidate_output(
    *,
    root: Path,
    output_directory: Path,
    bundle: SyntheticPdfCandidateOutputBundleV1,
) -> Path:
    """Write raw candidate content once under the sole local output boundary."""

    directory = _require_candidate_output_directory(root=root, output_directory=output_directory)
    destination = directory / f"{bundle.candidate_bundle_sha256}.json"
    payload = canonical_json_file_bytes(bundle.model_dump(mode="json", warnings=False))
    _reject_symlink_components(destination, code="candidate_output_write_failed")
    try:
        if destination.exists():
            if destination.is_symlink() or destination.read_bytes() != payload:
                raise SyntheticPdfCandidateRunnerError("candidate_output_immutable_conflict")
            return destination
        # Create-only avoids replacing another evaluator's raw OCR artifact.
        with destination.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except SyntheticPdfCandidateRunnerError:
        raise
    except OSError as error:
        raise SyntheticPdfCandidateRunnerError("candidate_output_write_failed") from error
    return destination


def _sealed_reservation_path(*, root: Path, packet: EvaluationPacketV2) -> Path:
    return root / "candidate-run-reservations" / f"sealed-{packet.packet_receipt_sha256}.json"


def reserve_sealed_candidate_run(
    *, packet: EvaluationPacketV2, profile: EvaluationRuntimeProfileV1
) -> Path | None:
    """Consume a sealed packet before any adapter invocation, exactly once."""

    if packet.split != "sealed":
        return None
    destination = _sealed_reservation_path(root=profile.document_preview_root, packet=packet)
    payload_value: dict[str, object] = {
        "schema_version": "1.0",
        "artifact_kind": "sealed_synthetic_pdf_candidate_reservation",
        "profile_mode": _PROFILE_MODE,
        "parser_revision": SYNTHETIC_PDF_LABEL_NORMALIZER_REVISION,
        "dataset_id": packet.dataset_id,
        "dataset_revision": packet.dataset_revision,
        "packet_kind": "sealed",
        "packet_receipt_sha256": packet.packet_receipt_sha256,
        "engine_kind": profile.engine_kind,
        "extractor_revision": profile.extractor_revision,
        "configuration_sha256": profile.configuration_sha256,
        "model_inventory_sha256": profile.model_inventory_sha256,
    }
    payload = canonical_json_file_bytes(payload_value)
    _reject_symlink_components(destination, code="sealed_candidate_reservation_invalid")
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        _reject_symlink_components(destination.parent, code="sealed_candidate_reservation_invalid")
        # ``xb`` is process-safe across local evaluators.  Any prior attempt,
        # even one that crashed after reservation, remains consumed by design.
        with destination.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise SyntheticPdfCandidateRunnerError("sealed_candidate_run_already_reserved") from error
    except OSError as error:
        raise SyntheticPdfCandidateRunnerError(
            "sealed_candidate_reservation_write_failed"
        ) from error
    return destination


def run_synthetic_pdf_ocr_candidate(
    *,
    receipt_path: Path,
    packet_path: Path,
    source_directory: Path,
    document_preview_root: Path,
    configuration_path: Path,
    inventory_path: Path,
    engine_kind: EngineKind,
    output_directory: Path,
    confirm_local_evaluation: bool,
    adapter_factory: Callable[[EvaluationRuntimeProfileV1], SyntheticPdfCandidateAdapter] = (
        build_default_synthetic_pdf_candidate_adapter
    ),
    package_version_reader: Callable[[str], str] = importlib.metadata.version,
) -> dict[str, object]:
    """Extract one packet with exactly one offline engine into local-only storage."""

    if not confirm_local_evaluation:
        raise SyntheticPdfCandidateRunnerError("local_evaluation_confirmation_required")
    profile = load_unapproved_local_candidate_profile(
        document_preview_root=document_preview_root,
        configuration_path=configuration_path,
        inventory_path=inventory_path,
        engine_kind=engine_kind,
        package_version_reader=package_version_reader,
    )
    receipt, packet, source_root = _load_receipt_and_packet(
        receipt_path=receipt_path,
        packet_path=packet_path,
        source_directory=source_directory,
    )
    receipt_assets = {asset.asset_id: asset for asset in receipt.assets}
    # All cheap validation happens before a sealed reservation.  The next line
    # occurs before the first adapter construction or PDF byte extraction.
    reservation = reserve_sealed_candidate_run(packet=packet, profile=profile)
    try:
        adapter = adapter_factory(profile)
    except Exception as error:
        raise SyntheticPdfCandidateRunnerError("candidate_adapter_unavailable") from error
    if profile.engine_kind == "docling":
        native_preflight = getattr(adapter, "preflight_native_runtime", None)
        if callable(native_preflight):
            try:
                native_preflight()
            except (DocumentPreviewError, OSError, socket.error) as error:
                raise SyntheticPdfCandidateRunnerError("candidate_adapter_unavailable") from error
            except Exception as error:
                raise SyntheticPdfCandidateRunnerError("candidate_adapter_unavailable") from error

    documents: list[CandidateDocumentOutputV1] = []
    for item in packet.items:
        asset = receipt_assets[item.asset_id]
        content = _load_selected_pdf(source_root=source_root, asset=asset)
        preview_input = PreviewInputV1(
            metadata=DocumentExtractionRequestMetadataV1(
                document_asset_id=item.asset_id,
                extraction_request_id=f"candidate-{asset.sha256[:32]}",
                source_sha256=asset.sha256,
                media_type="application/pdf",
            ),
            byte_size=len(content),
            content=content,
        )
        try:
            # The runtime adapters also guard their model operations.  This
            # outer scope protects injected adapters and every byte read path.
            with local_files_only():
                extracted = adapter.extract_pdf(preview_input)
        except (DocumentPreviewError, OSError, socket.error) as error:
            raise SyntheticPdfCandidateRunnerError("candidate_engine_execution_failed") from error
        except Exception as error:
            raise SyntheticPdfCandidateRunnerError("candidate_engine_execution_failed") from error
        # Adapter-provided structured cells are the only table evidence.  The
        # runner never derives grid coordinates from Markdown, Gold, CSV, or
        # a PDF content stream.  Docling can emit several form tables, so only
        # its fixed synthetic target table is selected and reindexed.  Other
        # adapters preserve any future verified structured cells unchanged.
        raw_table_cells = tuple(extracted.table_cells)
        if profile.engine_kind == "docling":
            try:
                table_cells = select_synthetic_pdf_candidate_table_cells(
                    table_cells=raw_table_cells
                )
            except Exception as error:
                raise SyntheticPdfCandidateRunnerError(
                    "candidate_table_selection_invalid"
                ) from error
        else:
            table_cells = list(raw_table_cells)
        fields = normalize_synthetic_pdf_fields(
            document_type=item.document_type,
            spans=extracted.spans,
            table_cells=table_cells,
        )
        documents.append(
            CandidateDocumentOutputV1(
                asset_id=item.asset_id,
                source_sha256=item.source_sha256,
                fields=fields,
                spans=list(extracted.spans),
                table_cells=table_cells,
            )
        )
    bundle = _build_candidate_bundle(packet=packet, profile=profile, documents=documents)
    destination = write_synthetic_pdf_candidate_output(
        root=profile.document_preview_root,
        output_directory=output_directory,
        bundle=bundle,
    )
    return {
        "status": "local_evaluation_candidate_recorded",
        "packet_kind": packet.split,
        "engine_kind": profile.engine_kind,
        "extractor_revision": profile.extractor_revision,
        "configuration_sha256": profile.configuration_sha256,
        "model_inventory_sha256": profile.model_inventory_sha256,
        "parser_revision": SYNTHETIC_PDF_LABEL_NORMALIZER_REVISION,
        "candidate_bundle_sha256": bundle.candidate_bundle_sha256,
        "candidate_file": destination.name,
        "reservation_file": reservation.name if reservation is not None else None,
        "approved_preview": False,
    }


__all__ = [
    "CandidateExtractionV1",
    "EvaluationRuntimeProfileV1",
    "SYNTHETIC_PDF_LABEL_NORMALIZER_REVISION",
    "SyntheticPdfCandidateAdapter",
    "SyntheticPdfCandidateRunnerError",
    "build_default_synthetic_pdf_candidate_adapter",
    "load_local_synthetic_pdf_evaluation_inputs",
    "load_unapproved_local_candidate_profile",
    "normalize_synthetic_pdf_docling_table_cells",
    "normalize_synthetic_pdf_fields",
    "reserve_sealed_candidate_run",
    "run_synthetic_pdf_ocr_candidate",
    "select_synthetic_pdf_candidate_table_cells",
    "write_synthetic_pdf_candidate_output",
]
