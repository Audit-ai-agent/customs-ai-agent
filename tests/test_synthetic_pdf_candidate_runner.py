from __future__ import annotations

import hashlib
import inspect
import json
import socket
import stat
from types import SimpleNamespace
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
import pytest

import customs_ai_agent.evaluation.synthetic_pdf_candidate_runner as candidate_runner_module
import customs_ai_agent.service.document_preview_runtime as runtime_module
from customs_ai_agent.evaluation.document_gold import (
    DatasetAssetReceiptV1,
    DatasetReceiptV1,
    receipt_sha256,
)
from customs_ai_agent.evaluation.document_ocr_quality import (
    CandidateFieldV1,
    CandidateSpanV1,
    CandidateTableCellV1,
    EvaluationPacketV2,
    StratifiedPacketItemV2,
    SyntheticDocumentType,
    SyntheticPdfCandidateOutputBundleV1,
    canonical_json_file_bytes,
    packet_receipt_sha256,
    validate_synthetic_candidate_output_provenance,
)
from customs_ai_agent.evaluation.native_pdf_vector_grid import NativePdfVectorFormRegion
from customs_ai_agent.evaluation.native_pdf_semantic_adapter import (
    NativePdfSemanticAdapter,
    NativePdfSemanticAdapterOutputV1,
)
from customs_ai_agent.evaluation.native_pdf_semantic_model import (
    NativePdfSemanticModelArtifactV1,
    bind_native_pdf_semantic_model_artifact,
)
from customs_ai_agent.evaluation.supervised_template_contract import (
    R1_SUPERVISED_TEMPLATE_DOCUMENT_TYPES,
    R1_SUPERVISED_TEMPLATE_REQUIRED_FIELDS,
    SUPERVISED_TEMPLATE_EXTRACTOR_REVISION,
    SUPERVISED_TEMPLATE_MODEL_ID,
    SUPERVISED_TEMPLATE_MODEL_REVISION,
)
from customs_ai_agent.evaluation.synthetic_pdf_candidate_runner import (
    CandidateExtractionV1,
    EvaluationRuntimeProfileV1,
    SYNTHETIC_PDF_LABEL_NORMALIZER_REVISION,
    SyntheticPdfCandidateAdapter,
    SyntheticPdfCandidateRunnerError,
    build_default_synthetic_pdf_candidate_adapter,
    load_local_synthetic_pdf_evaluation_inputs,
    load_unapproved_local_candidate_profile,
    merge_synthetic_pdf_candidate_fields,
    normalize_synthetic_pdf_docling_table_cells,
    normalize_synthetic_pdf_fields,
    run_synthetic_pdf_ocr_candidate,
    select_synthetic_pdf_candidate_table_cells,
)
from customs_ai_agent.service.document_preview_runtime import (
    DocumentPreviewRuntimeError,
    write_runtime_sidecars,
)
from customs_ai_agent.schemas.document_extraction import (
    BoundingBoxV1,
    DocumentExtractionRequestMetadataV1,
    DocumentPreviewPageV1,
    DocumentSpanV1,
    PreviewInputV1,
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


def _supervised_r1_model_binding():  # type: ignore[no-untyped-def]
    """Build a schema-valid opaque r1 model binding for boundary tests."""

    documents: list[dict[str, object]] = []
    for document_index, document_type in enumerate(R1_SUPERVISED_TEMPLATE_DOCUMENT_TYPES):
        documents.append(
            {
                "document_type": document_type,
                "structural_signature": {
                    "page_count": 1,
                    "minimum_span_count": 1,
                    "minimum_table_cell_count": 1,
                    "required_selectors": [
                        {
                            "source_kind": "span_anchor",
                            "page_number": 1,
                            "anchor_x": 0.1 + document_index / 10,
                            "anchor_y": 0.2,
                            "anchor_tolerance": 0.00001,
                        }
                    ],
                },
                "fields": [
                    {
                        "field_name": field_name,
                        "normalizer": normalizer,
                        "selectors": [
                            {
                                "source_kind": "span_reading_order",
                                "page_number": 1,
                                "reading_order": field_index + 1,
                            }
                        ],
                        "value_transform": "raw-v1",
                    }
                    for field_index, (field_name, normalizer) in enumerate(
                        R1_SUPERVISED_TEMPLATE_REQUIRED_FIELDS[document_type]
                    )
                ],
            }
        )
    artifact = NativePdfSemanticModelArtifactV1.model_validate(
        {
            "schema_version": "1.0",
            "artifact_kind": "native_pdf_semantic_model",
            "model_id": SUPERVISED_TEMPLATE_MODEL_ID,
            "model_revision": SUPERVISED_TEMPLATE_MODEL_REVISION,
            "inference_revision": "native-template-selector-v1",
            "documents": documents,
        }
    )
    artifact_bytes = canonical_json_file_bytes(
        artifact.model_dump(mode="json", warnings=False, exclude_none=True)
    )
    return bind_native_pdf_semantic_model_artifact(
        artifact_bytes=artifact_bytes,
        expected_sha256=_sha(artifact_bytes),
    )


def _supervised_direct_fields(document_type: SyntheticDocumentType) -> tuple[CandidateFieldV1, ...]:
    return tuple(
        CandidateFieldV1(field_name=field_name, value=f"opaque-{field_name}")
        for field_name, _normalizer in R1_SUPERVISED_TEMPLATE_REQUIRED_FIELDS[document_type]
    )


def _semantic_preview_input(content: bytes) -> PreviewInputV1:
    return PreviewInputV1(
        metadata=DocumentExtractionRequestMetadataV1(
            document_asset_id="opaque-asset",
            extraction_request_id="opaque-request",
            source_sha256="0" * 64,
            media_type="application/pdf",
        ),
        byte_size=len(content),
        content=content,
    )


_DOCLING_EXECUTION_PACKAGE_NAMES = (
    "deepsearch-glm",
    "docling",
    "docling-core",
    "docling-ibm-models",
    "docling-parse",
    "numpy",
    "pillow",
    "pypdf",
    "pypdfium2",
    "rtree",
    "safetensors",
    "scipy",
    "torch",
    "torchvision",
    "transformers",
)


_PADDLE_EXECUTION_PACKAGE_NAMES = (
    "albucore",
    "albumentations",
    "beautifulsoup4",
    "cython",
    "fire",
    "fonttools",
    "imgaug",
    "lmdb",
    "numpy",
    "opencv-contrib-python",
    "opencv-python",
    "opencv-python-headless",
    "paddleocr",
    "paddlepaddle",
    "pillow",
    "pyclipper",
    "pypdf",
    "pypdfium2",
    "python-docx",
    "pyyaml",
    "rapidfuzz",
    "requests",
    "scikit-image",
    "shapely",
    "tqdm",
)


@pytest.fixture(autouse=True)
def _isolate_paddle_wheel_record_verifier(monkeypatch: pytest.MonkeyPatch) -> None:
    """Candidate profile fixtures do not install the real Paddle wheel set."""

    monkeypatch.setattr(
        runtime_module,
        "_recorded_cv2_tree_matches_designated_provider",
        lambda: None,
    )


def _profile_sidecars(root: Path, engine_kind: str) -> tuple[Path, Path]:
    if engine_kind == "pdf_text_layer":
        configuration = {
            "schema_version": "1.0",
            "engine_kind": engine_kind,
            "extractor_revision": "pypdf-5.3.1-text-layer-visitor-v2",
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
            "packages": [_package(name) for name in _PADDLE_EXECUTION_PACKAGE_NAMES],
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
                "pdf_backend": "pypdfium2",
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
            "packages": [_package(name) for name in _DOCLING_EXECUTION_PACKAGE_NAMES],
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
            text += (
                "\n\n| Marks/No. | Description of Goods | Quantity | Unit Price | Amount |\n"
                "| --- | --- | ---: | ---: | ---: |\n"
                "| - | Widget | 2 PCS | 1.00 | 2.00 |"
            )
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


@dataclass
class _StaticExtractionFakeAdapter:
    """Return one synthetic adapter output without opening a real PDF."""

    extraction: CandidateExtractionV1
    calls: int = 0

    def extract_pdf(self, _preview_input: object) -> CandidateExtractionV1:
        self.calls += 1
        return self.extraction


def _bbox_span(
    *,
    text: str,
    reading_order: int,
    x: float,
    y: float,
    width: float = 0.10,
    height: float = 0.02,
) -> CandidateSpanV1:
    return CandidateSpanV1(
        page_number=1,
        reading_order=reading_order,
        text=text,
        bbox=BoundingBoxV1(x=x, y=y, width=width, height=height),
    )


def _bbox_grid(
    *,
    prefix: str,
    reading_order_start: int,
    header_y: float,
    body_y: float,
    columns: int = 5,
) -> list[CandidateSpanV1]:
    """Build one opaque public-bbox grid with no schema/header data."""

    spans: list[CandidateSpanV1] = []
    reading_order = reading_order_start
    for row_name, y in (("header", header_y), ("body", body_y)):
        for column_index in range(columns):
            spans.append(
                _bbox_span(
                    text=f"{prefix}-{row_name}-{column_index}",
                    reading_order=reading_order,
                    x=0.05 + (column_index * 0.18),
                    y=y,
                )
            )
            reading_order += 1
    return spans


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
        # This generic engine-shaped output proves the default runner does
        # not reconstruct table cells from Markdown without a local PDF or
        # source-truth input.
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
def test_runner_binds_every_engine_profile_and_keeps_only_verified_table_evidence(
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
    if engine_kind in {"docling", "pdf_text_layer"}:
        # A single unclassified cell cannot be passed through as a synthetic
        # target table.  Docling needs one direct structural target; native
        # text-layer candidates need one projector-produced visitor grid.
        assert bundle.documents[0].table_cells == []
    else:
        assert bundle.documents[0].table_cells[0].text == "Widget"
    assert adapter.calls == 1


def test_docling_runner_preflights_before_loading_a_selected_pdf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, engine_kind="docling")

    class FailingDoclingAdapter:
        def preflight_native_runtime(self) -> None:
            raise runtime_module.DocumentPreviewUnavailable(
                "document_preview_docling_native_runtime_unavailable"
            )

        def extract_pdf(self, _preview_input: object) -> CandidateExtractionV1:
            raise AssertionError("source PDF must not be passed to a failed adapter")

    monkeypatch.setattr(
        candidate_runner_module,
        "_load_selected_pdf",
        lambda **_kwargs: pytest.fail("source PDF must not be loaded before native preflight"),
    )

    with pytest.raises(SyntheticPdfCandidateRunnerError, match="candidate_adapter_unavailable"):
        _run(fixture, FailingDoclingAdapter())  # type: ignore[arg-type]


def test_candidate_profile_rejects_missing_paddle_postprocess_package(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, engine_kind="paddleocr")
    root = fixture["root"]
    configuration_path = fixture["configuration_path"]
    inventory_path = fixture["inventory_path"]
    assert isinstance(root, Path)
    assert isinstance(configuration_path, Path)
    assert isinstance(inventory_path, Path)
    invalid_inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
    invalid_inventory["packages"] = [
        package for package in invalid_inventory["packages"] if package["name"] != "pyclipper"
    ]
    invalid_path = _write_sidecar(root, "model-inventories", invalid_inventory)

    with pytest.raises(SyntheticPdfCandidateRunnerError, match="candidate_runtime_profile_invalid"):
        load_unapproved_local_candidate_profile(
            document_preview_root=root,
            configuration_path=configuration_path,
            inventory_path=invalid_path,
            engine_kind="paddleocr",
            package_version_reader=lambda _name: "test-v1",
        )


def test_runner_v4_full_invoice_fake_adapter_uses_visible_candidate_output_only(
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


def test_runner_does_not_derive_docling_table_cells_from_markdown_spans(
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
        "total",
    ]
    assert bundle.documents[0].table_cells == []
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
def test_v4_visible_layout_normalizer_derives_every_rendered_field_from_fake_candidate_output(
    document_type: str, expected_fields: list[str]
) -> None:
    extraction = _full_visible_layout_extraction(document_type)

    fields = normalize_synthetic_pdf_fields(
        document_type=document_type,  # type: ignore[arg-type]
        spans=extraction.spans,
        table_cells=extraction.table_cells,
    )

    assert [field.field_name for field in fields] == expected_fields


def test_v4_visible_layout_normalizer_fails_closed_for_ambiguous_or_missing_table_fields() -> None:
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


def test_v5_normalizers_have_only_candidate_span_and_table_inputs() -> None:
    field_parameters = inspect.signature(normalize_synthetic_pdf_fields).parameters
    docling_parameters = inspect.signature(normalize_synthetic_pdf_docling_table_cells).parameters
    direct_selector_parameters = inspect.signature(
        select_synthetic_pdf_candidate_table_cells
    ).parameters
    assert tuple(field_parameters) == ("document_type", "spans", "table_cells")
    assert tuple(docling_parameters) == ("document_type", "spans")
    assert tuple(direct_selector_parameters) == ("table_cells",)
    assert all(
        forbidden not in parameter_name
        for parameter_name in (*field_parameters, *docling_parameters, *direct_selector_parameters)
        for forbidden in ("gold", "csv", "source")
    )
    implementation = inspect.getsource(normalize_synthetic_pdf_fields)
    assert "source_attested" not in implementation
    assert "clean_manifest" not in implementation
    assert "error_labels" not in implementation
    selector_implementation = inspect.getsource(select_synthetic_pdf_candidate_table_cells)
    assert "document_type" not in selector_implementation
    assert "_matching_table_header" not in selector_implementation


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


def test_supervised_default_factory_requires_or_accepts_a_bound_model_loader(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path, engine_kind="pdf_text_layer")
    profile = load_unapproved_local_candidate_profile(
        document_preview_root=fixture["root"],  # type: ignore[arg-type]
        configuration_path=fixture["configuration_path"],  # type: ignore[arg-type]
        inventory_path=fixture["inventory_path"],  # type: ignore[arg-type]
        engine_kind="pdf_text_layer",
        package_version_reader=lambda _name: "test-v1",
    )
    supervised_profile = replace(
        profile,
        extractor_revision=SUPERVISED_TEMPLATE_EXTRACTOR_REVISION,
    )

    with pytest.raises(
        SyntheticPdfCandidateRunnerError,
        match="candidate_supervised_model_binding_unavailable",
    ):
        build_default_synthetic_pdf_candidate_adapter(supervised_profile)

    adapter = build_default_synthetic_pdf_candidate_adapter(
        supervised_profile,
        supervised_template_model_loader=lambda _profile: _supervised_r1_model_binding(),
    )

    assert type(adapter).__name__ == "_SupervisedNativePdfSemanticCandidateAdapter"


def test_supervised_semantic_adapter_uses_only_selected_bytes_and_preserves_native_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_fields = _supervised_direct_fields("invoice")
    expected_spans = (
        CandidateSpanV1(
            page_number=1,
            reading_order=1,
            text="opaque-native-span",
            anchor_x=0.1,
            anchor_y=0.2,
        ),
    )
    expected_cells = _table_cells(
        headers=("h0", "h1", "h2", "h3", "h4"),
        values=("v0", "v1", "v2", "v3", "v4"),
    )
    expected_regions = (
        NativePdfVectorFormRegion(
            page_number=1,
            x=0.1,
            y=0.1,
            width=0.3,
            height=0.1,
        ),
    )
    semantic_output = NativePdfSemanticAdapterOutputV1(
        document_type="invoice",
        model_id=SUPERVISED_TEMPLATE_MODEL_ID,
        model_revision=SUPERVISED_TEMPLATE_MODEL_REVISION,
        model_artifact_sha256="0" * 64,
        fields=expected_fields,
        spans=expected_spans,
        table_cells=expected_cells,
        form_regions=expected_regions,
    )
    seen: dict[str, bytes] = {}

    def extract_pdf_bytes(_self: object, pdf_bytes: bytes) -> NativePdfSemanticAdapterOutputV1:
        seen["pdf_bytes"] = pdf_bytes
        return semantic_output

    monkeypatch.setattr(NativePdfSemanticAdapter, "extract_pdf_bytes", extract_pdf_bytes)
    adapter = candidate_runner_module._SupervisedNativePdfSemanticCandidateAdapter(
        model=_supervised_r1_model_binding()
    )
    adapter.preflight_native_runtime()
    extraction = adapter.extract_pdf(_semantic_preview_input(b"selected-pdf-bytes"))

    assert seen == {"pdf_bytes": b"selected-pdf-bytes"}
    assert extraction.spans == expected_spans
    assert extraction.table_cells == expected_cells
    assert extraction.form_regions == expected_regions
    assert extraction.direct_fields == expected_fields
    assert extraction.direct_fields_are_model_bound is True
    assert extraction.table_cells_are_direct is True
    assert extraction.semantic_document_type == "invoice"


@pytest.mark.parametrize(
    ("document_type", "fields", "expected_code"),
    [
        (None, _supervised_direct_fields("invoice"), "candidate_supervised_document_type_invalid"),
        (
            "invoice",
            _supervised_direct_fields("invoice")[:-1],
            "candidate_supervised_fields_invalid",
        ),
        (
            "invoice",
            _supervised_direct_fields("invoice")
            + (CandidateFieldV1(field_name="unexpected", value="opaque"),),
            "candidate_supervised_fields_invalid",
        ),
    ],
)
def test_supervised_semantic_adapter_rejects_invalid_direct_output(
    monkeypatch: pytest.MonkeyPatch,
    document_type: str | None,
    fields: tuple[CandidateFieldV1, ...],
    expected_code: str,
) -> None:
    semantic_output = NativePdfSemanticAdapterOutputV1(
        document_type=document_type,
        model_id=SUPERVISED_TEMPLATE_MODEL_ID,
        model_revision=SUPERVISED_TEMPLATE_MODEL_REVISION,
        model_artifact_sha256="0" * 64,
        fields=fields,
        spans=(),
        table_cells=(),
        form_regions=(),
    )
    monkeypatch.setattr(
        NativePdfSemanticAdapter,
        "extract_pdf_bytes",
        lambda _self, _pdf_bytes: semantic_output,
    )
    adapter = candidate_runner_module._SupervisedNativePdfSemanticCandidateAdapter(
        model=_supervised_r1_model_binding()
    )

    with pytest.raises(SyntheticPdfCandidateRunnerError, match=expected_code):
        adapter.extract_pdf(_semantic_preview_input(b"selected-pdf-bytes"))


def _supervised_test_profile(fixture: dict[str, object]) -> EvaluationRuntimeProfileV1:
    profile = load_unapproved_local_candidate_profile(
        document_preview_root=fixture["root"],  # type: ignore[arg-type]
        configuration_path=fixture["configuration_path"],  # type: ignore[arg-type]
        inventory_path=fixture["inventory_path"],  # type: ignore[arg-type]
        engine_kind="pdf_text_layer",
        package_version_reader=lambda _name: "test-v1",
    )
    return replace(profile, extractor_revision=SUPERVISED_TEMPLATE_EXTRACTOR_REVISION)


def _supervised_direct_extraction(
    *,
    semantic_document_type: SyntheticDocumentType = "invoice",
    fields: tuple[CandidateFieldV1, ...] | None = None,
) -> CandidateExtractionV1:
    return CandidateExtractionV1(
        spans=(
            CandidateSpanV1(
                page_number=1,
                reading_order=1,
                text="opaque-native-span",
                anchor_x=0.1,
                anchor_y=0.2,
            ),
        ),
        table_cells=_table_cells(
            headers=("h0", "h1", "h2", "h3", "h4"),
            values=("v0", "v1", "v2", "v3", "v4"),
        ),
        table_cells_are_direct=True,
        form_regions=(
            NativePdfVectorFormRegion(
                page_number=1,
                x=0.1,
                y=0.1,
                width=0.3,
                height=0.1,
            ),
        ),
        direct_fields=fields if fields is not None else _supervised_direct_fields("invoice"),
        direct_fields_are_model_bound=True,
        semantic_document_type=semantic_document_type,
    )


def test_supervised_runner_uses_only_model_bound_fields_and_direct_native_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _fixture(tmp_path, engine_kind="pdf_text_layer")
    profile = _supervised_test_profile(fixture)
    monkeypatch.setattr(
        candidate_runner_module,
        "load_unapproved_local_candidate_profile",
        lambda **_kwargs: profile,
    )
    for function_name in (
        "normalize_synthetic_pdf_fields",
        "project_native_pdf_vector_form_fields",
        "project_native_pdf_text_layer_fields",
        "project_synthetic_pdf_visitor_table_cells",
    ):
        monkeypatch.setattr(
            candidate_runner_module,
            function_name,
            lambda **_kwargs: pytest.fail(
                "supervised path must not use heuristic field/table projection"
            ),
        )

    direct_fields = _supervised_direct_fields("invoice")
    summary = _run(
        fixture,
        _StaticExtractionFakeAdapter(
            extraction=_supervised_direct_extraction(fields=direct_fields)
        ),
    )
    output = fixture["output_directory"] / summary["candidate_file"]  # type: ignore[operator]
    bundle = SyntheticPdfCandidateOutputBundleV1.model_validate_json(output.read_bytes())

    assert bundle.documents[0].fields == list(direct_fields)
    assert bundle.documents[0].spans == [
        CandidateSpanV1(
            page_number=1,
            reading_order=1,
            text="opaque-native-span",
            anchor_x=0.1,
            anchor_y=0.2,
        )
    ]
    assert len(bundle.documents[0].table_cells) == 10


@pytest.mark.parametrize(
    ("extraction", "expected_code"),
    [
        (
            _supervised_direct_extraction(semantic_document_type="packing_list"),
            "candidate_supervised_evidence_invalid",
        ),
        (
            _supervised_direct_extraction(fields=_supervised_direct_fields("invoice")[:-1]),
            "candidate_supervised_fields_invalid",
        ),
        (
            _supervised_direct_extraction(
                fields=_supervised_direct_fields("invoice")
                + (CandidateFieldV1(field_name="unexpected", value="opaque"),)
            ),
            "candidate_supervised_fields_invalid",
        ),
    ],
)
def test_supervised_runner_rejects_wrong_document_or_field_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    extraction: CandidateExtractionV1,
    expected_code: str,
) -> None:
    fixture = _fixture(tmp_path, engine_kind="pdf_text_layer")
    profile = _supervised_test_profile(fixture)
    monkeypatch.setattr(
        candidate_runner_module,
        "load_unapproved_local_candidate_profile",
        lambda **_kwargs: profile,
    )

    with pytest.raises(SyntheticPdfCandidateRunnerError, match=expected_code):
        _run(fixture, _StaticExtractionFakeAdapter(extraction=extraction))


def test_v4_label_slash_space_matching_does_not_rewrite_candidate_values() -> None:
    fields = normalize_synthetic_pdf_fields(
        document_type="packing_list",
        spans=[
            CandidateSpanV1(page_number=1, reading_order=1, text="No. of L / C: LC / 001"),
            CandidateSpanV1(
                page_number=1,
                reading_order=2,
                text="Net / Gross Weight: 3 / 4 KG",
            ),
        ],
    )

    # Only the labels normalize slash spacing.  The candidate's field values
    # stay untouched for the evaluator's field-owned normalizer.
    assert [(field.field_name, field.value) for field in fields] == [
        ("lc_no", "LC / 001"),
        ("net_gross_weight", "3 / 4 KG"),
    ]


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


def test_v4_packing_list_flattened_item_cell_uses_one_terminal_ascii_quantity_unit() -> None:
    fields = normalize_synthetic_pdf_fields(
        document_type="packing_list",
        spans=(),
        table_cells=_table_cells(
            headers=(
                "Marks / Numbers / PKGS",
                "Goods Description",
                "Net Weight",
                "Gross Weight",
                "Measurement",
            ),
            values=("3 CTNS", "Widget 5 PCS", "10 KG", "12 KG", "0.20 CBM"),
        ),
    )

    assert [(field.field_name, field.value) for field in fields] == [
        ("good", "Widget"),
        ("unit", "PCS"),
        ("qty", "5"),
        ("cartons", "3 CTNS"),
        ("net_weight", "10 KG"),
        ("gross_weight", "12 KG"),
        ("cbm", "0.20 CBM"),
        ("net_gross_weight", "10 KG / 12 KG"),
    ]


@pytest.mark.parametrize("item_cell", ["Widget 5 PCS 2 CTNS", "Widget 5 개"])
def test_v4_packing_list_flattened_item_cell_rejects_ambiguous_or_non_ascii_quantity_unit(
    item_cell: str,
) -> None:
    fields = normalize_synthetic_pdf_fields(
        document_type="packing_list",
        spans=(),
        table_cells=_table_cells(
            headers=(
                "Marks/Numbers/PKGS",
                "Goods Description",
                "Net Weight",
                "Gross Weight",
                "Measurement",
            ),
            values=("3 CTNS", item_cell, "10 KG", "12 KG", "0.20 CBM"),
        ),
    )

    values = {field.field_name: field.value for field in fields}
    assert {"good", "qty", "unit"}.isdisjoint(values)
    assert values["cartons"] == "3 CTNS"
    assert values["net_gross_weight"] == "10 KG / 12 KG"


def test_v4_docling_markdown_table_normalizer_selects_one_complete_target_and_reindexes() -> None:
    cells = normalize_synthetic_pdf_docling_table_cells(
        document_type="invoice",
        spans=[
            CandidateSpanV1(
                page_number=1,
                reading_order=1,
                text="Before\n| Country |\n| --- |\n| KR |",
            ),
            CandidateSpanV1(
                page_number=1,
                reading_order=2,
                text=(
                    "\n\n| Marks/No. | Description of Goods | Quantity | Unit Price | Amount |\n"
                    "| --- | --- | ---: | ---: | ---: |\n"
                    "| - | Widget | 2 PCS | 1.00 | 2.00 |\nAfter"
                ),
            ),
        ],
    )

    assert [
        (cell.page_number, cell.table_index, cell.row_index, cell.column_index, cell.text)
        for cell in cells
    ] == [
        (1, 1, 0, 0, "Marks/No."),
        (1, 1, 0, 1, "Description of Goods"),
        (1, 1, 0, 2, "Quantity"),
        (1, 1, 0, 3, "Unit Price"),
        (1, 1, 0, 4, "Amount"),
        (1, 1, 1, 0, "-"),
        (1, 1, 1, 1, "Widget"),
        (1, 1, 1, 2, "2 PCS"),
        (1, 1, 1, 3, "1.00"),
        (1, 1, 1, 4, "2.00"),
    ]


def test_v4_docling_markdown_table_normalizer_rejects_multiple_complete_targets() -> None:
    cells = normalize_synthetic_pdf_docling_table_cells(
        document_type="invoice",
        spans=[
            CandidateSpanV1(
                page_number=1,
                reading_order=1,
                text=(
                    "| Description of Goods | Quantity | Unit Price | Amount |\n"
                    "| --- | ---: | ---: | ---: |\n"
                    "| Widget | 2 PCS | 1.00 | 2.00 |\n\n"
                    "| Description of Goods | Quantity | Unit Price | Amount |\n"
                    "| --- | ---: | ---: | ---: |\n"
                    "| Spare | 3 PCS | 1.00 | 3.00 |"
                ),
            )
        ],
    )

    assert cells == []


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
            document_type="invoice",
            spans=[CandidateSpanV1(page_number=1, reading_order=1, text=text)],
        )
        == []
    )


def _direct_candidate_table_cells(
    *,
    table_index: int,
    opaque_prefix: str,
    page_number: int = 1,
    columns: int = 5,
    trailing_rows: tuple[tuple[int, ...], ...] = (),
) -> list[CandidateTableCellV1]:
    """Build coordinate-only fixtures; values are not header aliases."""

    cells = [
        CandidateTableCellV1(
            page_number=page_number,
            table_index=table_index,
            row_index=row_index,
            column_index=column_index,
            text=f"{opaque_prefix}-{row_index}-{column_index}",
        )
        for row_index in (0, 1)
        for column_index in range(columns)
    ]
    for row_index, columns in enumerate(trailing_rows, start=2):
        cells.extend(
            CandidateTableCellV1(
                page_number=page_number,
                table_index=table_index,
                row_index=row_index,
                column_index=column_index,
                text=f"{opaque_prefix}-{row_index}-{column_index}",
            )
            for column_index in columns
        )
    return cells


@pytest.mark.parametrize(
    ("document_type", "opaque_prefix", "columns"),
    [
        ("bill_of_lading", "bl", 6),
        ("invoice", "invoice", 5),
        ("packing_list", "packing", 5),
    ],
    ids=["B/L", "Invoice", "Packing-List"],
)
def test_v6_structured_selector_reindexes_one_structural_target_for_every_document_type(
    document_type: str, opaque_prefix: str, columns: int
) -> None:
    """All fixed document forms use the same coordinate-only direct policy."""

    raw_cells = [
        CandidateTableCellV1(
            page_number=1,
            table_index=2,
            row_index=0,
            column_index=0,
            text=f"unrelated-{document_type}",
        ),
        *_direct_candidate_table_cells(table_index=7, opaque_prefix=opaque_prefix, columns=columns),
    ]

    selected = select_synthetic_pdf_candidate_table_cells(table_cells=raw_cells)

    assert len(selected) == columns * 2
    assert {(cell.page_number, cell.table_index) for cell in selected} == {(1, 1)}
    assert {(cell.row_index, cell.column_index) for cell in selected} == {
        (row_index, column_index) for row_index in (0, 1) for column_index in range(columns)
    }
    assert [cell.text for cell in selected] == [
        cell.text for cell in raw_cells if cell.table_index == 7
    ]


def test_v6_structured_selector_allows_trailing_partial_footer_rows() -> None:
    raw_cells = _direct_candidate_table_cells(
        table_index=7,
        opaque_prefix="target",
        trailing_rows=((0, 1), (2,)),
    )

    selected = select_synthetic_pdf_candidate_table_cells(table_cells=raw_cells)

    assert len(selected) == 10
    assert [cell.text for cell in selected] == [cell.text for cell in raw_cells[:10]]


@pytest.mark.parametrize("malformation", ["missing", "duplicate", "row_gap", "seventh_column"])
def test_v6_structured_selector_rejects_missing_duplicate_or_nonrectangular_coordinates(
    malformation: str,
) -> None:
    raw_cells = _direct_candidate_table_cells(table_index=7, opaque_prefix="target")
    if malformation == "missing":
        raw_cells = [cell for cell in raw_cells if (cell.row_index, cell.column_index) != (1, 4)]
    elif malformation == "duplicate":
        raw_cells.append(raw_cells[0])
    elif malformation == "row_gap":
        raw_cells.append(
            CandidateTableCellV1(
                page_number=1,
                table_index=7,
                row_index=3,
                column_index=0,
                text="gap",
            )
        )
    else:
        raw_cells.append(
            CandidateTableCellV1(
                page_number=1,
                table_index=7,
                row_index=1,
                column_index=6,
                text="outside",
            )
        )

    assert select_synthetic_pdf_candidate_table_cells(table_cells=raw_cells) == []


def test_v6_structured_selector_rejects_an_extra_complete_five_column_body_row() -> None:
    raw_cells = _direct_candidate_table_cells(
        table_index=7,
        opaque_prefix="target",
        trailing_rows=(tuple(range(5)),),
    )

    assert select_synthetic_pdf_candidate_table_cells(table_cells=raw_cells) == []


def test_v6_structured_selector_rejects_zero_or_multiple_structural_targets() -> None:
    assert select_synthetic_pdf_candidate_table_cells(table_cells=[]) == []
    assert (
        select_synthetic_pdf_candidate_table_cells(
            table_cells=[
                *_direct_candidate_table_cells(table_index=1, opaque_prefix="first"),
                *_direct_candidate_table_cells(table_index=2, opaque_prefix="second"),
            ]
        )
        == []
    )


def test_docling_runner_never_assigns_table_cells_from_markdown(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path, engine_kind="docling")
    summary = _run(fixture, _FakeAdapter(include_table_cells=False, markdown_table=True))
    output = fixture["output_directory"] / summary["candidate_file"]  # type: ignore[operator]
    bundle = SyntheticPdfCandidateOutputBundleV1.model_validate_json(output.read_bytes())

    assert bundle.documents[0].table_cells == []


def test_docling_runner_routes_raw_structured_cells_through_selection_hook(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, engine_kind="docling")
    seen: dict[str, object] = {}

    def select_target_table(*, table_cells: object) -> list[CandidateTableCellV1]:
        seen["table_cells"] = table_cells
        return [
            CandidateTableCellV1(
                page_number=1,
                table_index=1,
                row_index=0,
                column_index=0,
                text="selected-only",
            )
        ]

    monkeypatch.setattr(
        candidate_runner_module,
        "select_synthetic_pdf_candidate_table_cells",
        select_target_table,
    )

    summary = _run(fixture, _FakeAdapter())
    output = fixture["output_directory"] / summary["candidate_file"]  # type: ignore[operator]
    bundle = SyntheticPdfCandidateOutputBundleV1.model_validate_json(output.read_bytes())

    assert isinstance(seen["table_cells"], tuple)
    assert [cell.text for cell in bundle.documents[0].table_cells] == ["selected-only"]


def test_native_text_layer_runner_routes_only_visitor_spans_through_geometry_projector(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, engine_kind="pdf_text_layer")
    seen: dict[str, object] = {}
    projected = _table_cells(
        headers=("opaque-0", "opaque-1", "opaque-2", "opaque-3", "opaque-4"),
        values=("opaque-5", "opaque-6", "opaque-7", "opaque-8", "opaque-9"),
    )

    def project(*, spans: object) -> list[CandidateTableCellV1]:
        seen["spans"] = spans
        return list(projected)

    def select(*, table_cells: object) -> list[CandidateTableCellV1]:
        seen["table_cells"] = table_cells
        return list(projected)

    monkeypatch.setattr(
        candidate_runner_module,
        "project_synthetic_pdf_visitor_table_cells",
        project,
    )
    monkeypatch.setattr(
        candidate_runner_module,
        "select_synthetic_pdf_candidate_table_cells",
        select,
    )

    summary = _run(fixture, _FakeAdapter(include_table_cells=False))
    output = fixture["output_directory"] / summary["candidate_file"]  # type: ignore[operator]
    bundle = SyntheticPdfCandidateOutputBundleV1.model_validate_json(output.read_bytes())

    assert isinstance(seen["spans"], tuple)
    assert seen["table_cells"] == projected
    assert bundle.documents[0].table_cells == list(projected)


def test_native_text_layer_runner_uses_positioned_field_projection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, engine_kind="pdf_text_layer")
    seen: dict[str, object] = {}

    def project(*, document_type: object, spans: object) -> list[CandidateFieldV1]:
        seen["document_type"] = document_type
        seen["spans"] = spans
        return [CandidateFieldV1(field_name="inv_no", value="INV-001")]

    monkeypatch.setattr(
        candidate_runner_module,
        "project_native_pdf_text_layer_fields",
        project,
    )
    monkeypatch.setattr(
        candidate_runner_module,
        "project_synthetic_pdf_visitor_table_cells",
        lambda *, spans: [],
    )

    summary = _run(fixture, _FakeAdapter(include_table_cells=False))
    output = fixture["output_directory"] / summary["candidate_file"]  # type: ignore[operator]
    bundle = SyntheticPdfCandidateOutputBundleV1.model_validate_json(output.read_bytes())

    assert seen["document_type"] == "invoice"
    assert isinstance(seen["spans"], tuple)
    assert {field.field_name for field in bundle.documents[0].fields} >= {"inv_no"}


def test_native_vector_direct_cells_bypass_visitor_projection_and_add_form_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, engine_kind="pdf_text_layer")
    direct_cells = _table_cells(
        headers=("opaque-0", "opaque-1", "opaque-2", "opaque-3", "opaque-4"),
        values=("opaque-5", "opaque-6", "opaque-7", "opaque-8", "opaque-9"),
    )
    extraction = CandidateExtractionV1(
        spans=(
            CandidateSpanV1(
                page_number=1,
                reading_order=1,
                text="Invoice Number",
                anchor_x=0.10,
                anchor_y=0.10,
            ),
            CandidateSpanV1(
                page_number=1,
                reading_order=2,
                text="INV-001",
                anchor_x=0.50,
                anchor_y=0.10,
            ),
        ),
        table_cells=direct_cells,
        table_cells_are_direct=True,
        form_regions=(
            NativePdfVectorFormRegion(
                page_number=1,
                x=0.05,
                y=0.05,
                width=0.80,
                height=0.20,
            ),
        ),
    )
    monkeypatch.setattr(
        candidate_runner_module,
        "project_synthetic_pdf_visitor_table_cells",
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError("visitor must not run")),
    )

    summary = _run(fixture, _StaticExtractionFakeAdapter(extraction=extraction))
    output = fixture["output_directory"] / summary["candidate_file"]  # type: ignore[operator]
    bundle = SyntheticPdfCandidateOutputBundleV1.model_validate_json(output.read_bytes())

    assert len(bundle.documents[0].table_cells) == 10
    assert [(field.field_name, field.value) for field in bundle.documents[0].fields] == [
        ("inv_no", "INV-001")
    ]


def test_paddle_runner_routes_complete_public_bboxes_through_geometry_projectors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Paddle's public rectangles are evaluation-only geometry evidence."""

    fixture = _fixture(tmp_path, engine_kind="paddleocr")
    extraction = CandidateExtractionV1(
        spans=(
            _bbox_span(text="Invoice Number", reading_order=1, x=0.08, y=0.06, width=0.20),
            _bbox_span(text="INV-001", reading_order=2, x=0.40, y=0.06),
            *_bbox_grid(
                prefix="opaque",
                reading_order_start=3,
                header_y=0.40,
                body_y=0.46,
            ),
        )
    )
    adapter = _StaticExtractionFakeAdapter(extraction=extraction)
    seen: dict[str, object] = {}
    table_projector = candidate_runner_module.project_synthetic_pdf_visitor_table_cells
    field_projector = candidate_runner_module.project_native_pdf_text_layer_fields

    def project_table(*, spans: object) -> list[CandidateTableCellV1]:
        seen["table_spans"] = spans
        return table_projector(spans=spans)  # type: ignore[arg-type]

    def project_fields(*, document_type: object, spans: object) -> list[CandidateFieldV1]:
        seen["field_document_type"] = document_type
        seen["field_spans"] = spans
        return field_projector(  # type: ignore[arg-type]
            document_type=document_type,
            spans=spans,
        )

    monkeypatch.setattr(
        candidate_runner_module, "project_synthetic_pdf_visitor_table_cells", project_table
    )
    monkeypatch.setattr(
        candidate_runner_module, "project_native_pdf_text_layer_fields", project_fields
    )

    summary = _run(fixture, adapter)
    output = fixture["output_directory"] / summary["candidate_file"]  # type: ignore[operator]
    bundle = SyntheticPdfCandidateOutputBundleV1.model_validate_json(output.read_bytes())

    assert isinstance(seen["table_spans"], tuple)
    assert seen["field_document_type"] == "invoice"
    assert isinstance(seen["field_spans"], tuple)
    assert len(bundle.documents[0].table_cells) == 10
    assert [(cell.row_index, cell.column_index) for cell in bundle.documents[0].table_cells] == [
        (row_index, column_index) for row_index in (0, 1) for column_index in range(5)
    ]
    assert [(field.field_name, field.value) for field in bundle.documents[0].fields] == [
        ("inv_no", "INV-001")
    ]
    assert adapter.calls == 1


@pytest.mark.parametrize("geometry", ["missing_bbox", "ambiguous_grid"])
def test_paddle_runner_fails_closed_for_incomplete_or_ambiguous_bbox_geometry(
    tmp_path: Path, geometry: str
) -> None:
    fixture = _fixture(tmp_path, engine_kind="paddleocr")
    spans = _bbox_grid(
        prefix="opaque-first",
        reading_order_start=1,
        header_y=0.40,
        body_y=0.46,
    )
    if geometry == "missing_bbox":
        target = spans[-1]
        spans[-1] = CandidateSpanV1(
            page_number=target.page_number,
            reading_order=target.reading_order,
            text=target.text,
            bbox=None,
        )
    else:
        spans.extend(
            _bbox_grid(
                prefix="opaque-second",
                reading_order_start=20,
                header_y=0.66,
                body_y=0.72,
            )
        )
    adapter = _StaticExtractionFakeAdapter(extraction=CandidateExtractionV1(spans=tuple(spans)))

    summary = _run(fixture, adapter)
    output = fixture["output_directory"] / summary["candidate_file"]  # type: ignore[operator]
    bundle = SyntheticPdfCandidateOutputBundleV1.model_validate_json(output.read_bytes())

    assert bundle.documents[0].table_cells == []
    assert bundle.documents[0].fields == []
    assert adapter.calls == 1


def test_paddle_bbox_geometry_uses_the_existing_conflict_omit_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _fixture(tmp_path, engine_kind="paddleocr")
    adapter = _StaticExtractionFakeAdapter(
        extraction=CandidateExtractionV1(
            spans=(
                _bbox_span(
                    text="Invoice No: VISIBLE-ONLY",
                    reading_order=1,
                    x=0.08,
                    y=0.06,
                    width=0.34,
                ),
            )
        )
    )
    monkeypatch.setattr(
        candidate_runner_module,
        "project_synthetic_pdf_visitor_table_cells",
        lambda *, spans: [],
    )
    monkeypatch.setattr(
        candidate_runner_module,
        "project_native_pdf_text_layer_fields",
        lambda *, document_type, spans: [
            CandidateFieldV1(field_name="inv_no", value="POSITIONED-ONLY")
        ],
    )

    summary = _run(fixture, adapter)
    output = fixture["output_directory"] / summary["candidate_file"]  # type: ignore[operator]
    bundle = SyntheticPdfCandidateOutputBundleV1.model_validate_json(output.read_bytes())

    assert bundle.documents[0].fields == []


def test_paddle_bbox_geometry_does_not_extend_the_public_preview_contract() -> None:
    """Geometry stays in local candidate evaluation; the HTTP wire shape is unchanged."""

    assert "anchor_x" not in DocumentSpanV1.model_fields
    assert "anchor_y" not in DocumentSpanV1.model_fields
    assert "table_cells" not in DocumentPreviewPageV1.model_fields


def test_merge_candidate_fields_keeps_only_equal_or_single_raw_values() -> None:
    merged = merge_synthetic_pdf_candidate_fields(
        [
            CandidateFieldV1(field_name="same", value="value"),
            CandidateFieldV1(field_name="left_only", value="left"),
            CandidateFieldV1(field_name="conflict", value="first"),
        ],
        [
            CandidateFieldV1(field_name="same", value="value"),
            CandidateFieldV1(field_name="right_only", value="right"),
            CandidateFieldV1(field_name="conflict", value="second"),
        ],
    )

    assert [(field.field_name, field.value) for field in merged] == [
        ("same", "value"),
        ("left_only", "left"),
        ("right_only", "right"),
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


def test_local_receipt_loader_accepts_full_synthetic_inventory_and_stays_bounded(
    tmp_path: Path,
) -> None:
    """The real 6,000-PDF receipt is larger than compact sidecars, not unbounded."""

    local_data = tmp_path / "workspace" / "_local" / "data"
    source_sha256 = _sha("full-synthetic-inventory")
    assets = [
        DatasetAssetReceiptV1(
            asset_id=f"asset-{index:04}",
            relative_path=f"clean_sets/set_{index:04}/C{index:04}_INV.pdf",
            media_type="application/pdf",
            byte_size=1,
            sha256=source_sha256,
            source_role="preview_candidate",
        )
        for index in range(6_000)
    ]
    receipt = DatasetReceiptV1(
        dataset_id="synthetic-customs-pdf-2000",
        dataset_revision="20260811-r1",
        received_as_of=date(2026, 8, 11),
        assets=assets,
    )
    packet = _packet(receipt=receipt, source_sha256=source_sha256)
    receipt_path = local_data / "receipts" / "dataset-receipt.json"
    packet_path = local_data / "packets" / "development.json"
    receipt_path.parent.mkdir(parents=True)
    packet_path.parent.mkdir(parents=True)
    receipt_raw = canonical_json_file_bytes(receipt.model_dump(mode="json"))
    receipt_path.write_bytes(receipt_raw)
    packet_path.write_bytes(canonical_json_file_bytes(packet.model_dump(mode="json")))

    assert 1 * 1024 * 1024 < len(receipt_raw) <= 8 * 1024 * 1024
    loaded_receipt, loaded_packet = load_local_synthetic_pdf_evaluation_inputs(
        receipt_path=receipt_path,
        packet_path=packet_path,
    )
    assert receipt_sha256(loaded_receipt) == receipt_sha256(receipt)
    assert loaded_packet.packet_receipt_sha256 == packet.packet_receipt_sha256

    receipt_path.write_bytes(b"x" * (8 * 1024 * 1024 + 1))
    with pytest.raises(SyntheticPdfCandidateRunnerError, match="candidate_receipt_invalid"):
        load_local_synthetic_pdf_evaluation_inputs(
            receipt_path=receipt_path,
            packet_path=packet_path,
        )


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
