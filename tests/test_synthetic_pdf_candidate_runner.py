from __future__ import annotations

import hashlib
import inspect
import json
import socket
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
    SyntheticPdfCandidateRunnerError,
    build_default_synthetic_pdf_candidate_adapter,
    normalize_synthetic_pdf_fields,
    run_synthetic_pdf_ocr_candidate,
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
        for name in ("det", "rec"):
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
            "options": {"artifacts_directory": "models/docling", "ocr_enabled": True},
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

    def extract_pdf(self, _preview_input) -> CandidateExtractionV1:  # type: ignore[no-untyped-def]
        self.calls += 1
        if self.try_network:
            socket.create_connection(("127.0.0.1", 9), timeout=0.01)
        return CandidateExtractionV1(
            spans=(
                CandidateSpanV1(
                    page_number=1,
                    reading_order=1,
                    text="Invoice No: INV-001\nAmount: 10.00\nTOTAL: 10.00",
                ),
            ),
            table_cells=(
                CandidateTableCellV1(
                    page_number=1,
                    table_index=1,
                    row_index=0,
                    column_index=0,
                    text="Widget",
                ),
            ),
        )


def _run(
    fixture: dict[str, object],
    adapter: _FakeAdapter,
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
    assert bundle.parser_revision == "synthetic-pdf-label-normalizer-v1"
    assert [field.field_name for field in bundle.documents[0].fields] == [
        "inv_no",
        "amount",
        "total",
    ]
    assert bundle.documents[0].table_cells[0].text == "Widget"
    assert adapter.calls == 1


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
        ("qty", "4 PCS"),
        ("net_gross_weight", "3 / 4 KG"),
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
