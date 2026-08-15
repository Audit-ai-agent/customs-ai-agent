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
_MAX_JSON_BYTES = 1 * 1024 * 1024
_MAX_SOURCE_BYTES = 32 * 1024 * 1024
_PROFILE_MODE: Literal["offline_synthetic_pdf_evaluation"] = "offline_synthetic_pdf_evaluation"
# v2 adds an explicit Markdown-pipe-table transform for the local Docling
# candidate only.  It is part of the candidate provenance so an old field-only
# output cannot be relabelled as a table-capable one.
SYNTHETIC_PDF_LABEL_NORMALIZER_REVISION = "synthetic-pdf-label-normalizer-v2"


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
        return _candidate_from_preview(self.adapter.extract_pdf_text_layer(preview_input))


@dataclass(slots=True)
class _PaddleOcrCandidateAdapter:
    adapter: PaddleOcrPdfPreviewAdapter

    def extract_pdf(self, preview_input: PreviewInputV1) -> CandidateExtractionV1:
        return _candidate_from_preview(self.adapter.extract_raster_or_image(preview_input))


@dataclass(slots=True)
class _DoclingCandidateAdapter:
    adapter: DoclingPdfPreviewAdapter

    def extract_pdf(self, preview_input: PreviewInputV1) -> CandidateExtractionV1:
        # The current runtime adapter exposes preview spans only.  It does not
        # invent table coordinates; a future local Docling adapter may return a
        # CandidateExtractionV1 directly while preserving real table cells.
        return _candidate_from_preview(self.adapter.extract_raster_or_image(preview_input))


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


def _load_receipt_and_packet(
    *, receipt_path: Path, packet_path: Path, source_directory: Path
) -> tuple[DatasetReceiptV1, EvaluationPacketV2, Path]:
    _receipt_file, receipt_raw = _require_local_data_file(
        receipt_path, limit=_MAX_JSON_BYTES, code="candidate_receipt_invalid"
    )
    _packet_file, packet_raw = _require_local_data_file(
        packet_path, limit=_MAX_JSON_BYTES, code="candidate_packet_invalid"
    )
    try:
        receipt = DatasetReceiptV1.model_validate_json(receipt_raw)
        packet = EvaluationPacketV2.model_validate_json(packet_raw)
    except ValidationError as error:
        raise SyntheticPdfCandidateRunnerError("candidate_packet_or_receipt_invalid") from error
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


# The parser is a deliberately small, versioned document-specific specification.
# It never imports Gold, clean_manifest.csv, error_labels.csv, or a rendered-PDF
# template.  A label must appear in an emitted span with an explicit delimiter;
# unknown/ambiguous labels stay absent so the scorer fails naturally.
_SYNTHETIC_FIELD_LABELS: dict[SyntheticDocumentType, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "bill_of_lading": (
        ("exporter_name", ("exporter", "shipper")),
        ("consignee", ("consignee", "to applicant / consignee")),
        ("load_port", ("port of loading", "load port")),
        ("disch_port", ("port of discharge", "discharge port")),
        ("final_dest", ("final destination",)),
        ("vessel", ("vessel/flight", "vessel")),
        ("voyage", ("voyage no", "voyage")),
        ("bl_no", ("bill of lading no", "b/l no", "bl no")),
        ("container_no", ("container no",)),
        ("seal_no", ("seal no",)),
        ("good", ("goods description", "description of goods")),
        ("gross_weight", ("gross weight",)),
        ("containers", ("no. of containers", "number of containers", "containers")),
    ),
    "invoice": (
        ("exporter_name", ("exporter", "seller", "shipper")),
        ("importer_name", ("importer", "buyer")),
        ("consignee", ("consignee", "to applicant / consignee")),
        ("load_port", ("port of loading", "load port")),
        ("disch_port", ("port of discharge", "discharge port")),
        ("vessel", ("vessel/flight", "vessel")),
        ("inv_no", ("invoice number", "invoice no", "inv no")),
        ("inv_date", ("invoice date",)),
        ("lc_no", ("letter of credit", "l/c no", "lc no")),
        ("good", ("goods description", "description of goods")),
        ("unit", ("unit",)),
        ("qty", ("quantity", "qty")),
        ("unit_price", ("unit price",)),
        ("amount", ("amount",)),
        ("total", ("total",)),
    ),
    "packing_list": (
        ("exporter_name", ("exporter", "seller", "shipper")),
        ("consignee", ("consignee", "to applicant / consignee")),
        ("inv_no", ("no. of invoice", "invoice no", "inv no")),
        ("lc_no", ("no. of l/c", "letter of credit", "l/c no", "lc no")),
        ("good", ("goods description", "description of goods")),
        ("unit", ("unit",)),
        ("qty", ("quantity", "qty")),
        ("cartons", ("no. of cartons / pkgs", "cartons", "pkgs")),
        ("containers", ("no. of containers", "number of containers", "containers")),
        ("net_weight", ("net weight",)),
        ("gross_weight", ("gross weight",)),
        ("cbm", ("cbm", "volume")),
        ("net_gross_weight", ("net/gross weight",)),
    ),
}


def _label_expression(labels: Sequence[str]) -> re.Pattern[str]:
    escaped = sorted((re.escape(label) for label in labels), key=len, reverse=True)
    # A separator is required.  That prevents a heading or a value that merely
    # mentions a label from becoming a field candidate.
    return re.compile(
        rf"^\s*(?:{'|'.join(escaped)})\s*(?:[:#]|\s[-–—]\s)\s*(?P<value>[^\r\n]+?)\s*$",
        re.IGNORECASE,
    )


def normalize_synthetic_pdf_fields(
    *, document_type: SyntheticDocumentType, spans: Sequence[CandidateSpanV1]
) -> list[CandidateFieldV1]:
    """Extract only unambiguous labelled values from local candidate spans.

    The normalizer is intentionally not a production document parser.  It is
    only a stable synthetic-PDF evaluation transform and returns no value when
    a label is missing, malformed, or produces conflicting values.
    """

    specification = _SYNTHETIC_FIELD_LABELS.get(document_type)
    if specification is None:
        raise SyntheticPdfCandidateRunnerError("candidate_document_type_invalid")
    ordered_spans = sorted(spans, key=lambda span: (span.page_number, span.reading_order))
    fields: list[CandidateFieldV1] = []
    for field_name, labels in specification:
        matcher = _label_expression(labels)
        values: list[str] = []
        for span in ordered_spans:
            for line in span.text.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
                match = matcher.fullmatch(line)
                if match is not None:
                    value = match.group("value").strip()
                    if value:
                        values.append(value)
        unique_values = list(dict.fromkeys(values))
        if len(unique_values) == 1:
            fields.append(CandidateFieldV1(field_name=field_name, value=unique_values[0]))
    return fields


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


def normalize_synthetic_pdf_docling_table_cells(
    *, spans: Sequence[CandidateSpanV1]
) -> list[CandidateTableCellV1]:
    """Extract only explicit Docling Markdown pipe tables from page-one spans.

    The pinned runtime's Docling adapter exports one document-global Markdown
    stream and conservatively places every chunk on page 1.  Rejoining those
    chunks exactly restores that stream even when a 8,000-character boundary
    splits a row.  The parser emits a cell only for a complete header +
    separator + at least one equally wide data row.  It never uses Gold, CSV,
    coordinates, or a guessed table boundary.

    Coordinate convention shared with the local Gold reviewer:
    ``page_number=1``; ``table_index`` starts at 1 in Markdown order;
    ``row_index=0`` is the header, data rows follow at 1; ``column_index`` is
    zero-based.  The Markdown separator row is not a cell.
    """

    markdown = "".join(
        span.text
        for span in sorted(spans, key=lambda span: (span.page_number, span.reading_order))
        if span.page_number == 1
    )
    if not markdown:
        return []
    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    cells: list[CandidateTableCellV1] = []
    line_index = 0
    table_index = 0
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
        table_index += 1
        for row_index, row in enumerate([header, *data_rows]):
            for column_index, text in enumerate(row):
                cells.append(
                    CandidateTableCellV1(
                        page_number=1,
                        table_index=table_index,
                        row_index=row_index,
                        column_index=column_index,
                        text=text,
                    )
                )
        line_index = next_index
    return cells


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
        fields = normalize_synthetic_pdf_fields(
            document_type=item.document_type, spans=extracted.spans
        )
        # Preserve a future adapter's explicit structured cells.  The current
        # Docling adapter has only Markdown spans, so use its strict synthetic
        # pipe-table parser as the sole fallback.  PDF text/PaddleOCR are not
        # assigned guessed cells and consequently fail F1 when they emit none.
        table_cells = list(extracted.table_cells)
        if not table_cells and profile.engine_kind == "docling":
            table_cells = normalize_synthetic_pdf_docling_table_cells(spans=extracted.spans)
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
    "load_unapproved_local_candidate_profile",
    "normalize_synthetic_pdf_docling_table_cells",
    "normalize_synthetic_pdf_fields",
    "reserve_sealed_candidate_run",
    "run_synthetic_pdf_ocr_candidate",
    "write_synthetic_pdf_candidate_output",
]
