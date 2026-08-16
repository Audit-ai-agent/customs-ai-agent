from __future__ import annotations

import ast
import base64
import hashlib
import inspect
import zlib
from datetime import date
from pathlib import Path

import pytest

from customs_ai_agent.evaluation.document_gold import DatasetAssetReceiptV1, DatasetReceiptV1
from customs_ai_agent.evaluation.document_ocr_quality import (
    EvaluationPacketV2,
    StratifiedPacketItemV2,
    canonical_json_file_bytes,
    packet_receipt_sha256,
    receipt_sha256,
)
from customs_ai_agent.evaluation.source_attested_synthetic_pdf_truth import (
    SourceAttestedSyntheticDatasetScopeV1,
)
from customs_ai_agent.evaluation.supervised_template_training import (
    SupervisedTemplateTrainingError,
    build_source_attested_supervised_template_model_training_receipt,
    build_source_attested_supervised_template_training_labels,
    build_source_attested_supervised_template_training_split,
    read_canonical_source_attested_supervised_template_training_labels,
    read_canonical_source_attested_supervised_template_model_training_receipt,
    read_canonical_source_attested_supervised_template_training_split,
    write_source_attested_supervised_template_training_labels,
    write_source_attested_supervised_template_model_training_receipt,
    write_source_attested_supervised_template_training_split,
)


def _sha(value: bytes | str) -> str:
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _scope(receipt: DatasetReceiptV1, *, sets: int, clean: int, error: int) -> SourceAttestedSyntheticDatasetScopeV1:
    return SourceAttestedSyntheticDatasetScopeV1(
        dataset_id=receipt.dataset_id,
        dataset_revision=receipt.dataset_revision,
        dataset_receipt_sha256=receipt_sha256(receipt),
        total_pdf_assets=clean + error,
        clean_pdf_assets=clean,
        error_pdf_assets=error,
        sets_per_fixture_kind=sets,
        require_complete_canonical_inventory=False,
        require_canonical_packets=False,
    )


def _packet(
    *, receipt: DatasetReceiptV1, split: str, items: list[StratifiedPacketItemV2]
) -> EvaluationPacketV2:
    payload: dict[str, object] = {
        "schema_version": "2.0",
        "packet_kind": "stratified-v2",
        "split": split,
        "dataset_id": receipt.dataset_id,
        "dataset_revision": receipt.dataset_revision,
        "receipt_sha256": receipt_sha256(receipt),
        "error_labels_sha256": _sha("labels"),
        "prepared_as_of": "2026-08-16",
        "items": [item.model_dump(mode="json") for item in items],
        "required_gates": ["field_exact_match", "ocr_cer", "table_cell_f1"],
        "release_thresholds": {
            "field_exact_match_min": 0.99,
            "ocr_cer_max": 0.01,
            "table_cell_f1_min": 0.98,
        },
    }
    payload["packet_receipt_sha256"] = packet_receipt_sha256(payload)
    return EvaluationPacketV2.model_validate(payload)


def _item(asset: DatasetAssetReceiptV1) -> StratifiedPacketItemV2:
    fixture_directory, set_directory, filename = asset.relative_path.split("/")
    set_number = set_directory.removeprefix("set_")
    suffix = filename.removesuffix(".pdf").rsplit("_", maxsplit=1)[1]
    return StratifiedPacketItemV2(
        asset_id=asset.asset_id,
        source_sha256=asset.sha256,
        fixture_kind="clean" if fixture_directory == "clean_sets" else "error",
        document_type={"BL": "bill_of_lading", "INV": "invoice", "PL": "packing_list"}[suffix],
        set_id=("C" if fixture_directory == "clean_sets" else "E") + set_number,
    )


def _split_receipt() -> tuple[DatasetReceiptV1, dict[str, DatasetAssetReceiptV1]]:
    assets: list[DatasetAssetReceiptV1] = []
    for number in range(1, 4):
        for fixture_directory, prefix in (("clean_sets", "C"), ("error_sets", "E")):
            for suffix in ("BL", "INV", "PL"):
                content = f"split-{fixture_directory}-{number}-{suffix}".encode("ascii")
                assets.append(
                    DatasetAssetReceiptV1(
                        asset_id=f"{prefix.lower()}-{number}-{suffix.lower()}",
                        relative_path=(
                            f"{fixture_directory}/set_{number:04d}/{prefix}{number:04d}_{suffix}.pdf"
                        ),
                        media_type="application/pdf",
                        byte_size=len(content),
                        sha256=_sha(content),
                        source_role="preview_candidate",
                    )
                )
    receipt = DatasetReceiptV1(
        dataset_id="synthetic-pdf",
        dataset_revision="source-v1",
        received_as_of=date(2026, 8, 16),
        assets=assets,
    )
    return receipt, {asset.asset_id: asset for asset in assets}


def test_family_safe_split_excludes_clean_error_and_all_document_siblings() -> None:
    receipt, assets = _split_receipt()
    development = _packet(receipt=receipt, split="development", items=[_item(assets["c-1-bl"])])
    sealed = _packet(receipt=receipt, split="sealed", items=[_item(assets["e-2-inv"])])

    split = build_source_attested_supervised_template_training_split(
        receipt=receipt,
        development_packet=development,
        sealed_packet=sealed,
        expected_dataset_scope=_scope(receipt, sets=3, clean=9, error=9),
    )

    assert split.held_out_family_set_numbers == ["0001", "0002"]
    assert split.excluded_family_asset_count == 12
    assert split.training_asset_count == 6
    assert {asset.family_set_number for asset in split.training_assets} == {"0003"}
    assert {(asset.fixture_kind, asset.document_type) for asset in split.training_assets} == {
        ("clean", "bill_of_lading"),
        ("clean", "invoice"),
        ("clean", "packing_list"),
        ("error", "bill_of_lading"),
        ("error", "invoice"),
        ("error", "packing_list"),
    }
    assert split.training_asset_list_sha256


def test_split_refuses_development_sealed_asset_overlap() -> None:
    receipt, assets = _split_receipt()
    item = _item(assets["c-1-bl"])
    development = _packet(receipt=receipt, split="development", items=[item])
    sealed = _packet(receipt=receipt, split="sealed", items=[item])

    with pytest.raises(SupervisedTemplateTrainingError, match="packet_overlap"):
        build_source_attested_supervised_template_training_split(
            receipt=receipt,
            development_packet=development,
            sealed_packet=sealed,
            expected_dataset_scope=_scope(receipt, sets=3, clean=9, error=9),
        )


_HEADER_RECTS = (
    (45.0, 527.8898, 105.0, 16.0),
    (150.0, 527.8898, 170.0, 16.0),
    (320.0, 527.8898, 80.0, 16.0),
    (400.0, 527.8898, 80.0, 16.0),
    (480.0, 527.8898, 70.27559, 16.0),
)
_BODY_RECTS = (
    (45.0, 377.8898, 105.0, 150.0),
    (150.0, 377.8898, 170.0, 150.0),
    (320.0, 377.8898, 80.0, 150.0),
    (400.0, 377.8898, 80.0, 150.0),
    (480.0, 377.8898, 70.27559, 150.0),
)


def _invoice_pdf(*, marker: str, total: str) -> bytes:
    """Create the exact pinned invoice content-stream shape for a local test."""

    rectangles = [(10.0, 100.0, 20.0, 10.0)] * 11
    rectangles.extend(_HEADER_RECTS)
    rectangles.extend(_BODY_RECTS)
    rectangles.extend(((10.0, 120.0, 20.0, 10.0), (10.0, 140.0, 20.0, 10.0)))
    positions = [(20.0, 760.0)] * 41
    for reading_order, position in {
        3: (49.0, 743.8898),
        6: (301.6378, 743.8898),
        7: (301.6378, 734.3898),
        9: (49.0, 673.8898),
        12: (301.6378, 673.8898),
        15: (49.0, 609.8898),
        17: (49.0, 572.8898),
        19: (175.3189, 572.8898),
        24: (49.0, 550.8898),
        32: (48.0, 513.8898),
        33: (48.0, 504.3898),
        34: (48.0, 494.8898),
        35: (153.0, 513.8898),
        36: (358.256, 513.8898),
        37: (453.327, 513.8898),
        38: (494.5246, 513.8898),
        40: (487.1316, 362.8898),
    }.items():
        positions[reading_order - 1] = position
    for offset, rectangle in enumerate(_HEADER_RECTS, start=27):
        positions[offset - 1] = (rectangle[0] + 1.0, rectangle[1] + 1.0)
    lines = ["0.5 w"]
    lines.extend(f"{x} {y} {width} {height} re S" for x, y, width, height in rectangles)
    lines.extend(("BT", "/F1 10 Tf", "12 TL"))
    for reading_order, (x, y) in enumerate(positions, start=1):
        rendered = {
            36: "2 PCS",
            37: "5.00",
            38: "10.00",
            40: total,
        }.get(reading_order, f"{marker}-{reading_order}")
        lines.extend((f"1 0 0 1 {x} {y} Tm", f"({rendered}) Tj"))
    lines.append("ET")
    encoded = base64.a85encode(zlib.compress("\n".join(lines).encode("ascii")), adobe=True)
    return b"".join(
        (
            b"%PDF-1.3\n",
            b"1 0 obj\n<< /Type /Page /MediaBox [0 0 595.2756 841.8898] /Contents 2 0 R >>\nendobj\n",
            b"2 0 obj\n<< /Length ",
            str(len(encoded)).encode("ascii"),
            b" /Filter [/ASCII85Decode /FlateDecode] >>\nstream\n",
            encoded,
            b"\nendstream\nendobj\n%%EOF\n",
        )
    )


def _label_fixture(tmp_path: Path) -> tuple[
    Path, DatasetReceiptV1, EvaluationPacketV2, EvaluationPacketV2, SourceAttestedSyntheticDatasetScopeV1
]:
    source_directory = tmp_path / "workspace" / "_local" / "data" / "synthetic-pdf"
    assets: list[DatasetAssetReceiptV1] = []
    for number in range(1, 4):
        for fixture_directory, prefix, total in (
            ("clean_sets", "C", "10"),
            ("error_sets", "E", "20"),
        ):
            marker = f"private-marker-{fixture_directory}-{number}"
            content = _invoice_pdf(marker=marker, total=total)
            relative_path = f"{fixture_directory}/set_{number:04d}/{prefix}{number:04d}_INV.pdf"
            path = source_directory / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            assets.append(
                DatasetAssetReceiptV1(
                    asset_id=f"{prefix.lower()}-{number}-inv",
                    relative_path=relative_path,
                    media_type="application/pdf",
                    byte_size=len(content),
                    sha256=_sha(content),
                    source_role="preview_candidate",
                )
            )
    receipt = DatasetReceiptV1(
        dataset_id="synthetic-pdf",
        dataset_revision="source-v1",
        received_as_of=date(2026, 8, 16),
        assets=assets,
    )
    assets_by_id = {asset.asset_id: asset for asset in assets}
    development = _packet(receipt=receipt, split="development", items=[_item(assets_by_id["c-1-inv"])])
    sealed = _packet(receipt=receipt, split="sealed", items=[_item(assets_by_id["e-2-inv"])])
    return source_directory, receipt, development, sealed, _scope(receipt, sets=3, clean=3, error=3)


def test_labels_are_pdf_derived_compact_and_immutable(tmp_path: Path) -> None:
    source_directory, receipt, development, sealed, scope = _label_fixture(tmp_path)
    split = build_source_attested_supervised_template_training_split(
        receipt=receipt,
        development_packet=development,
        sealed_packet=sealed,
        expected_dataset_scope=scope,
    )
    labels = build_source_attested_supervised_template_training_labels(
        receipt=receipt,
        development_packet=development,
        sealed_packet=sealed,
        training_split=split,
        source_directory=source_directory,
        expected_dataset_scope=scope,
    )

    assert split.training_asset_count == 2
    assert labels.record_count == 2
    assert labels.field_label_count == 30
    assert {record.asset_id for record in labels.records} == {"c-3-inv", "e-3-inv"}
    encoded = canonical_json_file_bytes(labels.model_dump(mode="json"))
    assert b"private-marker" not in encoded
    preview_root = tmp_path / "workspace" / "_local" / "data" / "document-preview"
    preview_root.mkdir()
    split_path = write_source_attested_supervised_template_training_split(
        split=split, output_directory=preview_root / "supervised-template-training-splits"
    )
    labels_path = write_source_attested_supervised_template_training_labels(
        labels=labels, output_directory=preview_root / "supervised-template-training-labels"
    )
    assert read_canonical_source_attested_supervised_template_training_split(split_path) == split
    assert read_canonical_source_attested_supervised_template_training_labels(labels_path) == labels

    model_receipt = build_source_attested_supervised_template_model_training_receipt(
        training_split=split,
        training_labels=labels,
        model_kind="semantic-template-model",
        model_revision="v1",
        model_artifact_sha256=_sha("model"),
        training_recipe_sha256=_sha("recipe"),
        training_environment_inventory_sha256=_sha("inventory"),
    )
    runtime_safe_payload = model_receipt.model_dump(mode="json")
    assert "training_assets" not in runtime_safe_payload
    assert "records" not in runtime_safe_payload
    assert "source_layout_parser_spec_sha256" not in runtime_safe_payload
    model_receipt_path = write_source_attested_supervised_template_model_training_receipt(
        receipt=model_receipt,
        output_directory=preview_root / "supervised-template-model-training-receipts",
    )
    assert (
        read_canonical_source_attested_supervised_template_model_training_receipt(model_receipt_path)
        == model_receipt
    )


def test_label_compilation_refuses_source_asset_tampering(tmp_path: Path) -> None:
    source_directory, receipt, development, sealed, scope = _label_fixture(tmp_path)
    split = build_source_attested_supervised_template_training_split(
        receipt=receipt,
        development_packet=development,
        sealed_packet=sealed,
        expected_dataset_scope=scope,
    )
    tampered = source_directory / "clean_sets" / "set_0003" / "C0003_INV.pdf"
    tampered.write_bytes(b"tampered")

    with pytest.raises(SupervisedTemplateTrainingError, match="source_asset_invalid"):
        build_source_attested_supervised_template_training_labels(
            receipt=receipt,
            development_packet=development,
            sealed_packet=sealed,
            training_split=split,
            source_directory=source_directory,
            expected_dataset_scope=scope,
        )


def test_runtime_modules_do_not_import_training_only_labels() -> None:
    repository = Path(__file__).resolve().parents[1]
    runtime_modules = tuple((repository / "src" / "customs_ai_agent" / "service").glob("*.py")) + (
        repository / "src" / "customs_ai_agent" / "evaluation" / "synthetic_pdf_candidate_runner.py",
    )
    for path in runtime_modules:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = [
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        ] + [
            node.module or ""
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom)
        ]
        assert all("supervised_template_training" not in name for name in imports), path.name


def test_training_label_compiler_has_no_csv_or_evaluation_pdf_input() -> None:
    parameters = inspect.signature(
        build_source_attested_supervised_template_training_labels
    ).parameters

    assert "clean_manifest_path" not in parameters
    assert "error_labels_path" not in parameters
    assert "development_layout_truth" not in parameters
    assert "sealed_layout_truth" not in parameters
