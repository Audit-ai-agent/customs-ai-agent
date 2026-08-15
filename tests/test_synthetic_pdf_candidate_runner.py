from __future__ import annotations

import hashlib
import inspect
import json
import socket
import stat
from types import SimpleNamespace
from dataclasses import dataclass
from datetime import date
from pathlib import Path
import pytest

from customs_ai_agent.evaluation.document_gold import (
    DatasetAssetReceiptV1,
    DatasetReceiptV1,
    receipt_sha256,
)
from customs_ai_agent.evaluation.document_ocr_quality import (
    CandidateSpanV1,
    CandidateTableCellV1,
    EvaluationPacketV2,
    StratifiedPacketItemV2,
    SyntheticPdfCandidateOutputBundleV1,
    canonical_json_file_bytes,
    packet_receipt_sha256,
    validate_synthetic_candidate_output_provenance,
)
from customs_ai_agent.evaluation.synthetic_pdf_candidate_runner import (
    CandidateExtractionV1,
    EvaluationRuntimeProfileV1,
    SYNTHETIC_PDF_LABEL_NORMALIZER_REVISION,
    SyntheticPdfCandidateAdapter,
    SyntheticPdfCandidateRunnerError,
    build_default_synthetic_pdf_candidate_adapter,
    load_unapproved_local_candidate_profile,
    normalize_synthetic_pdf_docling_table_cells,
    normalize_synthetic_pdf_fields,
    run_synthetic_pdf_ocr_candidate,
)
from customs_ai_agent.service.document_preview_runtime import (
    DocumentPreviewRuntimeError,
    write_runtime_sidecars,
)
from tools.run_synthetic_pdf_ocr_candidate import _parser, main


def _canonical_sidecar(value: object) -> bytes:
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


def _sha(value: bytes | str) -> str:
    return hashlib.sha256(value if isinstance(value, bytes) else value.encode("utf-8")).hexdigest()


def _write_sidecar(root: Path, directory: str, value: dict[str, object]) -> Path:
    raw = _canonical_sidecar(value)
    path = root / directory / f"{_sha(raw)}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return path


def _package(name: str) -> dict[str, object]:
    return {
        "name": name,
        "version": "test-v1",
        "license_id": "test-license",
        "source_uri": "https://example.test/models/",
    }


def _profile_sidecars(root: Path, engine_kind: str) -> tuple[Path, Path]:
    if engine_kind == "pdf_text_layer":
        configuration = {
            "schema_version": "1.0",
            "engine_kind": engine_kind,
            "extractor_revision": "native-test-v1",
            "options": {},
        }
        inventory = {
            "schema_version": "1.0",
            "engine_kind": engine_kind,
            "packages": [_package("pypdf")],
            "artifacts": [],
        }
    elif engine_kind == "paddleocr":
        artifacts = []
        for name in ("det", "rec", "cls"):
            path = root / "models" / name / "model.bin"
            path.parent.mkdir(parents=True, exist_ok=True)
            raw = f"{engine_kind}:{name}".encode("utf-8")
            path.write_bytes(raw)
            artifacts.append(
                {
                    "artifact_id": f"{name}-model",
                    "relative_path": f"models/{name}/model.bin",
                    "sha256": _sha(raw),
                    "byte_size": len(raw),
                    "license_id": "test-license",
                    "source_uri": "https://example.test/models/",
                }
            )
        configuration = {
            "schema_version": "1.0",
            "engine_kind": engine_kind,
            "extractor_revision": "paddle-test-v1",
            "options": {
                "det_model_directory": "models/det",
                "rec_model_directory": "models/rec",
                "cls_model_directory": "models/cls",
                "ocr_version": "PP-OCRv4",
                "use_gpu": False,
                "cpu_threads": 1,
            },
        }
        inventory = {
            "schema_version": "1.0",
            "engine_kind": engine_kind,
            "packages": [
                _package("pypdf"),
                _package("paddleocr"),
                _package("paddlepaddle"),
                _package("pypdfium2"),
            ],
            "artifacts": artifacts,
        }
    else:
        path = root / "models" / "docling" / "model.bin"
        path.parent.mkdir(parents=True, exist_ok=True)
        raw = b"docling:model"
        path.write_bytes(raw)
        configuration = {
            "schema_version": "1.0",
            "engine_kind": engine_kind,
            "extractor_revision": "docling-test-v1",
            "options": {
                "artifacts_directory": "models/docling",
                "ocr_enabled": False,
                "device": "cpu",
                "num_threads": 1,
                "do_table_structure": True,
                "table_structure_mode": "fast",
            },
        }
        inventory = {
            "schema_version": "1.0",
            "engine_kind": engine_kind,
            "packages": [_package("pypdf"), _package("docling")],
            "artifacts": [
                {
                    "artifact_id": "docling-model",
                    "relative_path": "models/docling/model.bin",
                    "sha256": _sha(raw),
                    "byte_size": len(raw),
                    "license_id": "test-license",
                    "source_uri": "https://example.test/models/",
                }
            ],
        }
    return (
        _write_sidecar(root, "runtime-configurations", configuration),
        _write_sidecar(root, "model-inventories", inventory),
    )


def _packet(
    *,
    receipt: DatasetReceiptV1,
    source_sha256: str,
    split: str = "development",
    receipt_digest: str | None = None,
) -> EvaluationPacketV2:
    item = StratifiedPacketItemV2(
        asset_id="asset-001",
        source_sha256=source_sha256,
        fixture_kind="clean",
        document_type="invoice",
        set_id="C0001",
    )
    value: dict[str, object] = {
        "schema_version": "2.0",
        "packet_kind": "stratified-v2",
        "split": split,
        "dataset_id": receipt.dataset_id,
        "dataset_revision": receipt.dataset_revision,
        "receipt_sha256": receipt_digest or receipt_sha256(receipt),
        "error_labels_sha256": _sha("labels"),
        "prepared_as_of": "2026-08-14",
        "items": [item.model_dump(mode="json")],
        "required_gates": ["field_exact_match", "ocr_cer", "table_cell_f1"],
        "release_thresholds": {
            "field_exact_match_min": 0.99,
            "ocr_cer_max": 0.01,
            "table_cell_f1_min": 0.98,
        },
    }
    value["packet_receipt_sha256"] = packet_receipt_sha256(value)
    return EvaluationPacketV2.model_validate(value)


def _fixture(
    tmp_path: Path, *, engine_kind: str = "pdf_text_layer", split: str = "development"
) -> dict[str, object]:
    local_data = tmp_path / "workspace" / "_local" / "data"
    root = local_data / "document-preview"
    root.mkdir(parents=True)
    source_directory = local_data / "synthetic-pdf"
    source_path = source_directory / "clean_sets" / "set_0001" / "C0001_INV.pdf"
    source_path.parent.mkdir(parents=True)
    source_bytes = b"%PDF-1.7\nsynthetic-candidate\n%%EOF\n"
    source_path.write_bytes(source_bytes)
    receipt = DatasetReceiptV1(
        dataset_id="synthetic-pdf",
        dataset_revision="human-source-v1",
        received_as_of=date(2026, 8, 14),
        assets=[
            DatasetAssetReceiptV1(
                asset_id="asset-001",
                relative_path="clean_sets/set_0001/C0001_INV.pdf",
                media_type="application/pdf",
                byte_size=len(source_bytes),
                sha256=_sha(source_bytes),
                source_role="preview_candidate",
            )
        ],
    )
    packet = _packet(receipt=receipt, source_sha256=_sha(source_bytes), split=split)
    receipt_path = local_data / "receipts" / "dataset-receipt.json"
    packet_path = local_data / "packets" / f"{split}.json"
    receipt_path.parent.mkdir(parents=True)
    packet_path.parent.mkdir(parents=True)
    receipt_path.write_bytes(canonical_json_file_bytes(receipt.model_dump(mode="json")))
    packet_path.write_bytes(canonical_json_file_bytes(packet.model_dump(mode="json")))
    configuration_path, inventory_path = _profile_sidecars(root, engine_kind)
    return {
        "local_data": local_data,
        "root": root,
        "source_directory": source_directory,
        "source_path": source_path,
        "receipt": receipt,
        "receipt_path": receipt_path,
        "packet": packet,
        "packet_path": packet_path,
        "configuration_path": configuration_path,
        "inventory_path": inventory_path,
        "engine_kind": engine_kind,
        "output_directory": root / "candidate-outputs" / engine_kind,
    }


@dataclass
class _FakeAdapter:
    calls: int = 0
    try_network: bool = False
    include_table_cells: bool = True
    markdown_table: bool = False

    def extract_pdf(self, _preview_input) -> CandidateExtractionV1:  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.try_network:
            socket.create_connection(("127.0.0.1", 9), timeout=0.01)
        text = "Invoice No: INV-001\nAmount: 10.00\nTOTAL: 10.00"
        if self.markdown_table:
            text += "\n\n| Item | Qty |\n| --- | ---: |\n| Widget | 2 |"
        table_cells = (
            (
                CandidateTableCellV1(
                    page_number=1,
                    table_index=1,
                    row_index=0,
                    column_index=0,
                    text="Widget",
                ),
            )
            if self.include_table_cells
            else ()
        )
        return CandidateExtractionV1(
            spans=(
                CandidateSpanV1(
                    page_number=1,
                    reading_order=1,
                    text=text,
                ),
            ),
            table_cells=table_cells,
        )


def _table_cells(
    *, headers: tuple[str, ...], values: tuple[str, ...]
) -> tuple[CandidateTableCellV1, ...]:
    assert len(headers) == len(values)
    return tuple(
        CandidateTableCellV1(
            page_number=1,
            table_index=1,
            row_index=row_index,
            column_index=column_index,
            text=text,
        )
        for row_index, row in enumerate((headers, values))
        for column_index, text in enumerate(row)
    )


def _full_visible_layout_extraction(document_type: str) -> CandidateExtractionV1:
    """Generic fake engine output; no local PDF/Gold/CSV values are used."""

    if document_type == "bill_of_lading":
        form_lines = (
            "Shipper",
            "Seller One",
            "Consignee",
            "Receiver One",
            "Port of Loading",
            "Port Alpha",
            "Port of Discharge",
            "Port Beta",
            "Final Destination",
            "City Gamma",
            "Vessel/Flight",
            "Vessel Delta",
            "Voyage No.",
            "V-01",
            "B/L No.",
            "BL-001",
        )
        cells = _table_cells(
            headers=(
                "Container No.",
                "Seal No.",
                "No. & Kind of Packages/Containers",
                "Goods Description",
                "Gross Weight",
                "Measurement",
            ),
            values=("CONT-1", "SEAL-1", "8 CTNS", "Widget", "100 KG", "1 CBM"),
        )
    elif document_type == "invoice":
        form_lines = (
            "Shipper",
            "Seller One",
            "Notify Party",
            "Buyer One",
            "Consignee",
            "Receiver One",
            "Port of Loading",
            "Port Alpha",
            "Port of Discharge",
            "Port Beta",
            "Vessel/Flight",
            "Vessel Delta",
            "Invoice No.",
            "INV-001",
            "Invoice Date",
            "2026-01-01",
            "L/C No.",
            "LC-001",
            "TOTAL",
            "62.50",
        )
        cells = _table_cells(
            headers=("Marks/No.", "Description of Goods", "Quantity", "Unit Price", "Amount"),
            values=("-", "Widget", "5 PCS", "12.50", "62.50"),
        )
    elif document_type == "packing_list":
        form_lines = (
            "Shipper",
            "Seller One",
            "To Applicant / Consignee",
            "Receiver One",
            "No. of Invoice",
            "INV-001",
            "No. of L/C",
            "LC-001",
        )
        cells = _table_cells(
            headers=(
                "Marks/Numbers/PKGS",
                "Goods Description",
                "Net Weight",
                "Gross Weight",
                "Measurement",
            ),
            values=("3 CTNS", "Widget\n5 PCS", "10 KG", "12 KG", "0.20 CBM"),
        )
    else:
        raise AssertionError("unexpected synthetic document type")
    return CandidateExtractionV1(
        spans=(CandidateSpanV1(page_number=1, reading_order=1, text="\n".join(form_lines)),),
        table_cells=cells,
    )


@dataclass
class _FullInvoiceVisibleLayoutFakeAdapter:
    calls: int = 0

    def extract_pdf(self, _preview_input) -> CandidateExtractionV1:  # type: ignore[no-untyped-def]
        self.calls += 1
        return _full_visible_layout_extraction("invoice")


@dataclass
class _FullInvoiceDoclingMarkdownFakeAdapter:
    calls: int = 0

    def extract_pdf(self, _preview_input) -> CandidateExtractionV1:  # type: ignore[no-untyped-def]
        self.calls += 1
        # The current pinned Docling boundary emits Markdown spans, not direct
        # table cells.  This generic engine-shaped output exercises the sole
        # allowed pipe-table fallback without consulting a local PDF or truth.
        return CandidateExtractionV1(
            spans=(
                CandidateSpanV1(
                    page_number=1,
                    reading_order=1,
                    text=(
                        "Shipper\nSeller One\nNotify Party\nBuyer One\nConsignee\nReceiver One\n"
                        "Port of Loading\nPort Alpha\nPort of Discharge\nPort Beta\n"
                        "Vessel/Flight\nVessel Delta\nInvoice No.\nINV-001\n"
                        "Invoice Date\n2026-01-01\nL/C No.\nLC-001\nTOTAL\n62.50\n\n"
                        "| Marks/No. | Description of Goods | Quantity | Unit Price | Amount |\n"
                        "| --- | --- | --- | ---: | ---: |\n"
                        "| - | Widget | 5 PCS | 12.50 | 62.50 |"
                    ),
                ),
            )
        )


def _run(
    fixture: dict[str, object],
    adapter: SyntheticPdfCandidateAdapter,
    *,
    factory_calls: list[int] | None = None,
) -> dict[str, object]:
    def factory(_profile: EvaluationRuntimeProfileV1) -> _FakeAdapter:
        if factory_calls is not None:
            factory_calls.append(1)
        return adapter

    return run_synthetic_pdf_ocr_candidate(
        receipt_path=fixture["receipt_path"],  # type: ignore[arg-type]
        packet_path=fixture["packet_path"],  # type: ignore[arg-type]
        source_directory=fixture["source_directory"],  # type: ignore[arg-type]
        document_preview_root=fixture["root"],  # type: ignore[arg-type]
        configuration_path=fixture["configuration_path"],  # type: ignore[arg-type]
        inventory_path=fixture["inventory_path"],  # type: ignore[arg-type]
        engine_kind=fixture["engine_kind"],  # type: ignore[arg-type]
        output_directory=fixture["output_directory"],  # type: ignore[arg-type]
        confirm_local_evaluation=True,
        adapter_factory=factory,
        package_version_reader=lambda _name: "test-v1",
    )


@pytest.mark.parametrize("engine_kind", ["pdf_text_layer", "paddleocr", "docling"])
def test_runner_binds_every_engine_profile_and_preserves_adapter_cells(
    tmp_path: Path, engine_kind: str
) -> None:
    fixture = _fixture(tmp_path, engine_kind=engine_kind)
    adapter = _FakeAdapter()

    summary = _run(fixture, adapter)

    output = fixture["output_directory"] / summary["candidate_file"]  # type: ignore[operator]
    bundle = SyntheticPdfCandidateOutputBundleV1.model_validate_json(output.read_bytes())
    assert summary["approved_preview"] is False
    assert bundle.engine_kind == engine_kind
    assert bundle.extractor_revision == summary["extractor_revision"]
    assert bundle.configuration_sha256 == summary["configuration_sha256"]
    assert bundle.model_inventory_sha256 == summary["model_inventory_sha256"]
    assert bundle.parser_revision == SYNTHETIC_PDF_LABEL_NORMALIZER_REVISION
    assert [field.field_name for field in bundle.documents[0].fields] == [
        "inv_no",
        "amount",
        "total",
    ]
    assert bundle.documents[0].table_cells[0].text == "Widget"
    assert adapter.calls == 1


def test_runner_v3_full_invoice_fake_adapter_uses_visible_candidate_output_only(
    tmp_path: Path,
) -> None:
    """The runner can emit every Invoice field without a source-truth input."""

    fixture = _fixture(tmp_path, engine_kind="docling")
    adapter = _FullInvoiceVisibleLayoutFakeAdapter()

    summary = _run(fixture, adapter)

    output = fixture["output_directory"] / summary["candidate_file"]  # type: ignore[operator]
    bundle = SyntheticPdfCandidateOutputBundleV1.model_validate_json(output.read_bytes())
    assert bundle.parser_revision == SYNTHETIC_PDF_LABEL_NORMALIZER_REVISION
    assert [field.field_name for field in bundle.documents[0].fields] == [
        "exporter_name",
        "importer_name",
        "consignee",
        "load_port",
        "disch_port",
        "vessel",
        "inv_no",
        "inv_date",
        "lc_no",
        "good",
        "unit",
        "qty",
        "unit_price",
        "amount",
        "total",
    ]
    assert adapter.calls == 1


def test_runner_v3_derives_full_docling_markdown_candidate_without_direct_table_cells(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, engine_kind="docling")
    adapter = _FullInvoiceDoclingMarkdownFakeAdapter()

    summary = _run(fixture, adapter)

    output = fixture["output_directory"] / summary["candidate_file"]  # type: ignore[operator]
    bundle = SyntheticPdfCandidateOutputBundleV1.model_validate_json(output.read_bytes())
    assert [field.field_name for field in bundle.documents[0].fields] == [
        "exporter_name",
        "importer_name",
        "consignee",
        "load_port",
        "disch_port",
        "vessel",
        "inv_no",
        "inv_date",
        "lc_no",
        "good",
        "unit",
        "qty",
        "unit_price",
        "amount",
        "total",
    ]
    assert len(bundle.documents[0].table_cells) == 10
    assert adapter.calls == 1


@pytest.mark.parametrize(
    ("document_type", "expected_fields"),
    [
        (
            "bill_of_lading",
            [
                "exporter_name",
                "consignee",
                "load_port",
                "disch_port",
                "final_dest",
                "vessel",
                "voyage",
                "bl_no",
                "container_no",
                "seal_no",
                "good",
                "gross_weight",
                "containers",
            ],
        ),
        (
            "invoice",
            [
                "exporter_name",
                "importer_name",
                "consignee",
                "load_port",
                "disch_port",
                "vessel",
                "inv_no",
                "inv_date",
                "lc_no",
                "good",
                "unit",
                "qty",
                "unit_price",
                "amount",
                "total",
            ],
        ),
        (
            "packing_list",
            [
                "exporter_name",
                "consignee",
                "inv_no",
                "lc_no",
                "good",
                "unit",
                "qty",
                "cartons",
                "net_weight",
                "gross_weight",
                "cbm",
                "net_gross_weight",
            ],
        ),
    ],
)
def test_v3_visible_layout_normalizer_derives_every_rendered_field_from_fake_candidate_output(
    document_type: str, expected_fields: list[str]
) -> None:
    extraction = _full_visible_layout_extraction(document_type)

    fields = normalize_synthetic_pdf_fields(
        document_type=document_type,  # type: ignore[arg-type]
        spans=extraction.spans,
        table_cells=extraction.table_cells,
    )

    assert [field.field_name for field in fields] == expected_fields


def test_v3_visible_layout_normalizer_fails_closed_for_ambiguous_or_missing_table_fields() -> None:
    extraction = _full_visible_layout_extraction("invoice")
    ambiguous_cells = tuple(
        cell
        for cell in extraction.table_cells
        if not (cell.row_index == 0 and cell.column_index == 4)
    ) + (
        CandidateTableCellV1(
            page_number=1,
            table_index=1,
            row_index=0,
            column_index=5,
            text="Amount",
        ),
        CandidateTableCellV1(
            page_number=1,
            table_index=1,
            row_index=1,
            column_index=5,
            text="62.50",
        ),
    )

    fields = normalize_synthetic_pdf_fields(
        document_type="invoice",
        spans=extraction.spans,
        table_cells=ambiguous_cells,
    )

    # All form fields remain direct candidate output, but the table is rejected
    # as ambiguous; no table-derived values are silently filled from elsewhere.
    assert [field.field_name for field in fields] == [
        "exporter_name",
        "importer_name",
        "consignee",
        "load_port",
        "disch_port",
        "vessel",
        "inv_no",
        "inv_date",
        "lc_no",
        "total",
    ]


def test_v3_normalizer_has_only_candidate_span_and_table_inputs() -> None:
    parameters = inspect.signature(normalize_synthetic_pdf_fields).parameters
    assert tuple(parameters) == ("document_type", "spans", "table_cells")
    implementation = inspect.getsource(normalize_synthetic_pdf_fields)
    assert "source_attested" not in implementation
    assert "clean_manifest" not in implementation
    assert "error_labels" not in implementation


@pytest.mark.parametrize(
    ("engine_kind", "configured_directory"),
    [
        ("paddleocr", "models/det"),
        ("docling", "models/docling"),
    ],
)
def test_model_directory_inventory_rejects_an_undeclared_regular_file(
    tmp_path: Path, engine_kind: str, configured_directory: str
) -> None:
    """Provisioning and candidate execution share the same exact-set gate."""

    fixture = _fixture(tmp_path, engine_kind=engine_kind)
    root = fixture["root"]
    configuration_path = fixture["configuration_path"]
    inventory_path = fixture["inventory_path"]
    assert isinstance(root, Path)
    assert isinstance(configuration_path, Path)
    assert isinstance(inventory_path, Path)
    configuration_value = json.loads(configuration_path.read_text(encoding="utf-8"))
    inventory_value = json.loads(inventory_path.read_text(encoding="utf-8"))
    (root / configured_directory / "not-in-inventory.bin").write_bytes(b"untracked")

    with pytest.raises(
        DocumentPreviewRuntimeError, match="document_preview_model_artifact_invalid"
    ):
        write_runtime_sidecars(
            document_preview_root=root,
            configuration_value=configuration_value,
            inventory_value=inventory_value,
            package_version_reader=lambda _name: "test-v1",
        )
    with pytest.raises(SyntheticPdfCandidateRunnerError, match="candidate_model_inventory_invalid"):
        load_unapproved_local_candidate_profile(
            document_preview_root=root,
            configuration_path=configuration_path,
            inventory_path=inventory_path,
            engine_kind=engine_kind,  # type: ignore[arg-type]
            package_version_reader=lambda _name: "test-v1",
        )


def test_model_directory_inventory_rejects_a_missing_declared_file(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, engine_kind="docling")
    root = fixture["root"]
    configuration_path = fixture["configuration_path"]
    inventory_path = fixture["inventory_path"]
    assert isinstance(root, Path)
    assert isinstance(configuration_path, Path)
    assert isinstance(inventory_path, Path)
    (root / "models" / "docling" / "model.bin").unlink()

    with pytest.raises(SyntheticPdfCandidateRunnerError, match="candidate_model_inventory_invalid"):
        load_unapproved_local_candidate_profile(
            document_preview_root=root,
            configuration_path=configuration_path,
            inventory_path=inventory_path,
            engine_kind="docling",
            package_version_reader=lambda _name: "test-v1",
        )


def test_model_directory_inventory_rejects_a_symlink_entry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Exercise the link branch even where Windows disallows test symlinks."""

    fixture = _fixture(tmp_path, engine_kind="docling")
    root = fixture["root"]
    configuration_path = fixture["configuration_path"]
    inventory_path = fixture["inventory_path"]
    assert isinstance(root, Path)
    assert isinstance(configuration_path, Path)
    assert isinstance(inventory_path, Path)
    link_entry = root / "models" / "docling" / "redirected-model.bin"
    link_entry.write_bytes(b"placeholder")
    path_type = type(link_entry)
    original_lstat = path_type.lstat

    def fake_lstat(path: Path) -> object:
        if path == link_entry:
            return SimpleNamespace(st_mode=stat.S_IFLNK, st_file_attributes=0)
        return original_lstat(path)

    monkeypatch.setattr(path_type, "lstat", fake_lstat)

    with pytest.raises(SyntheticPdfCandidateRunnerError, match="candidate_model_inventory_invalid"):
        load_unapproved_local_candidate_profile(
            document_preview_root=root,
            configuration_path=configuration_path,
            inventory_path=inventory_path,
            engine_kind="docling",
            package_version_reader=lambda _name: "test-v1",
        )


def test_candidate_bundle_cannot_be_relabelled_at_evaluation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    summary = _run(fixture, _FakeAdapter())
    output = fixture["output_directory"] / summary["candidate_file"]  # type: ignore[operator]
    bundle = SyntheticPdfCandidateOutputBundleV1.model_validate_json(output.read_bytes())

    with pytest.raises(ValueError, match="provenance does not match"):
        validate_synthetic_candidate_output_provenance(
            candidate_output=bundle,
            packet=fixture["packet"],  # type: ignore[arg-type]
            engine_kind="docling",
            extractor_revision=bundle.extractor_revision,
            configuration_sha256=bundle.configuration_sha256,
            model_inventory_sha256=bundle.model_inventory_sha256,
            parser_revision=bundle.parser_revision,
        )


@pytest.mark.parametrize(
    ("engine_kind", "expected_type"),
    [
        ("pdf_text_layer", "_PdfTextLayerCandidateAdapter"),
        ("paddleocr", "_PaddleOcrCandidateAdapter"),
        ("docling", "_DoclingCandidateAdapter"),
    ],
)
def test_default_factory_has_one_direct_boundary_per_engine(
    tmp_path: Path, engine_kind: str, expected_type: str
) -> None:
    fixture = _fixture(tmp_path, engine_kind=engine_kind)
    from customs_ai_agent.evaluation.synthetic_pdf_candidate_runner import (
        load_unapproved_local_candidate_profile,
    )

    profile = load_unapproved_local_candidate_profile(
        document_preview_root=fixture["root"],  # type: ignore[arg-type]
        configuration_path=fixture["configuration_path"],  # type: ignore[arg-type]
        inventory_path=fixture["inventory_path"],  # type: ignore[arg-type]
        engine_kind=engine_kind,  # type: ignore[arg-type]
        package_version_reader=lambda _name: "test-v1",
    )

    assert type(build_default_synthetic_pdf_candidate_adapter(profile)).__name__ == expected_type


def test_normalizer_uses_only_explicit_labelled_candidate_spans() -> None:
    fields = normalize_synthetic_pdf_fields(
        document_type="packing_list",
        spans=[
            CandidateSpanV1(page_number=1, reading_order=1, text="No. of Invoice: INV-9"),
            CandidateSpanV1(page_number=1, reading_order=2, text="Quantity: 4 PCS"),
            CandidateSpanV1(page_number=1, reading_order=3, text="heading: Containers"),
            CandidateSpanV1(page_number=1, reading_order=4, text="Net/Gross Weight: 3 / 4 KG"),
        ],
    )

    assert [(field.field_name, field.value) for field in fields] == [
        ("inv_no", "INV-9"),
        ("unit", "PCS"),
        ("qty", "4"),
        ("net_gross_weight", "3 / 4 KG"),
    ]


def test_docling_markdown_table_normalizer_keeps_multiple_explicit_tables() -> None:
    cells = normalize_synthetic_pdf_docling_table_cells(
        spans=[
            CandidateSpanV1(
                page_number=1,
                reading_order=1,
                text="Before\n| Item | Qty |\n| --- | ---: |\n| Widget",
            ),
            CandidateSpanV1(
                page_number=1,
                reading_order=2,
                text=(" | 2 |\n| Screw \\| bolt | 3 |\n\n| Country |\n| :--- |\n| KR |\nAfter"),
            ),
        ]
    )

    assert [
        (cell.page_number, cell.table_index, cell.row_index, cell.column_index, cell.text)
        for cell in cells
    ] == [
        (1, 1, 0, 0, "Item"),
        (1, 1, 0, 1, "Qty"),
        (1, 1, 1, 0, "Widget"),
        (1, 1, 1, 1, "2"),
        (1, 1, 2, 0, "Screw | bolt"),
        (1, 1, 2, 1, "3"),
        (1, 2, 0, 0, "Country"),
        (1, 2, 1, 0, "KR"),
    ]


@pytest.mark.parametrize(
    "text",
    [
        "No Markdown table here",
        "| Item | Qty |\n| --- | --- |\n| Widget |",
        "```\n| Item | Qty |\n| --- | --- |\n| Widget | 2 |\n```",
    ],
)
def test_docling_markdown_table_normalizer_rejects_no_or_malformed_table(text: str) -> None:
    assert (
        normalize_synthetic_pdf_docling_table_cells(
            spans=[CandidateSpanV1(page_number=1, reading_order=1, text=text)]
        )
        == []
    )


def test_docling_runner_derives_table_cells_only_from_explicit_markdown(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, engine_kind="docling")
    summary = _run(fixture, _FakeAdapter(include_table_cells=False, markdown_table=True))
    output = fixture["output_directory"] / summary["candidate_file"]  # type: ignore[operator]
    bundle = SyntheticPdfCandidateOutputBundleV1.model_validate_json(output.read_bytes())

    assert [
        (cell.table_index, cell.row_index, cell.column_index, cell.text)
        for cell in bundle.documents[0].table_cells
    ] == [
        (1, 0, 0, "Item"),
        (1, 0, 1, "Qty"),
        (1, 1, 0, "Widget"),
        (1, 1, 1, "2"),
    ]


def test_runner_rejects_source_drift_before_adapter_invocation(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["source_path"].write_bytes(b"changed")  # type: ignore[union-attr]
    adapter = _FakeAdapter()

    with pytest.raises(SyntheticPdfCandidateRunnerError, match="candidate_source_receipt_invalid"):
        _run(fixture, adapter)

    assert adapter.calls == 0


def test_runner_rejects_packet_receipt_and_engine_profile_mismatches(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    receipt = fixture["receipt"]  # type: ignore[assignment]
    source_sha = receipt.assets[0].sha256  # type: ignore[union-attr]
    bad_packet = _packet(
        receipt=receipt,  # type: ignore[arg-type]
        source_sha256=source_sha,
        receipt_digest="0" * 64,
    )
    fixture["packet_path"].write_bytes(  # type: ignore[union-attr]
        canonical_json_file_bytes(bad_packet.model_dump(mode="json"))
    )
    with pytest.raises(SyntheticPdfCandidateRunnerError, match="candidate_packet_receipt_mismatch"):
        _run(fixture, _FakeAdapter())

    profile_mismatch = _fixture(tmp_path / "profile-mismatch", engine_kind="docling")
    profile_mismatch["engine_kind"] = "pdf_text_layer"
    with pytest.raises(SyntheticPdfCandidateRunnerError, match="candidate_engine_profile_mismatch"):
        _run(profile_mismatch, _FakeAdapter())


def test_runner_blocks_network_and_keeps_content_out_of_failure(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)

    with pytest.raises(
        SyntheticPdfCandidateRunnerError, match="candidate_engine_execution_failed"
    ) as error:
        _run(fixture, _FakeAdapter(try_network=True))

    assert "INV-001" not in str(error.value)
    assert not list(fixture["output_directory"].glob("*.json"))  # type: ignore[union-attr]


def test_sealed_packet_is_reserved_before_adapter_factory_and_never_reruns(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, split="sealed")
    adapter = _FakeAdapter()
    factory_calls: list[int] = []

    first = _run(fixture, adapter, factory_calls=factory_calls)
    with pytest.raises(
        SyntheticPdfCandidateRunnerError, match="sealed_candidate_run_already_reserved"
    ):
        _run(fixture, _FakeAdapter(), factory_calls=factory_calls)

    assert first["reservation_file"] is not None
    reservation = (
        fixture["root"] / "candidate-run-reservations" / first["reservation_file"]  # type: ignore[operator]
    )
    assert json.loads(reservation.read_text(encoding="utf-8"))["parser_revision"] == (
        SYNTHETIC_PDF_LABEL_NORMALIZER_REVISION
    )
    assert adapter.calls == 1
    assert len(factory_calls) == 1


def test_runner_has_no_gold_input_and_cli_failure_does_not_echo_paths(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert all(
        "gold" not in name for name in inspect.signature(run_synthetic_pdf_ocr_candidate).parameters
    )
    assert "--gold" not in _parser().format_help()

    result = main(
        [
            "--receipt",
            "untrusted-content-path",
            "--packet",
            "untrusted-content-path",
            "--source-directory",
            "untrusted-content-path",
            "--document-preview-root",
            "untrusted-content-path",
            "--configuration",
            "untrusted-content-path",
            "--inventory",
            "untrusted-content-path",
            "--engine-kind",
            "pdf_text_layer",
            "--output-directory",
            "untrusted-content-path",
        ]
    )

    assert result == 2
    output = capsys.readouterr().out
    assert json.loads(output) == {"status": "failed", "code": "synthetic_pdf_candidate_run_failed"}
    assert "untrusted-content-path" not in output
