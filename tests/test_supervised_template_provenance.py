from __future__ import annotations

import ast
import hashlib
import json
import stat
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from types import SimpleNamespace

import pytest

from customs_ai_agent.evaluation.document_gold import DatasetAssetReceiptV1, DatasetReceiptV1
from customs_ai_agent.evaluation.document_ocr_quality import (
    EvaluationPacketV2,
    StratifiedPacketItemV2,
    canonical_json_bytes,
    canonical_json_file_bytes,
    packet_receipt_sha256,
    receipt_sha256,
)
from customs_ai_agent.evaluation.native_pdf_semantic_model import NativePdfSemanticModelArtifactV1
from customs_ai_agent.evaluation.supervised_template_contract import (
    R1_SUPERVISED_TEMPLATE_DOCUMENT_TYPES,
    R1_SUPERVISED_TEMPLATE_REQUIRED_FIELDS,
    SUPERVISED_TEMPLATE_MODEL_ID,
    SUPERVISED_TEMPLATE_MODEL_KIND,
    SUPERVISED_TEMPLATE_MODEL_REVISION,
)
from customs_ai_agent.evaluation import supervised_template_provenance as provenance


def _sha(value: bytes | str) -> str:
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_canonical(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_file_bytes(payload))


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
        "error_labels_sha256": _sha("test-error-labels"),
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
    fixture_kind = "clean" if fixture_directory == "clean_sets" else "error"
    return StratifiedPacketItemV2(
        asset_id=asset.asset_id,
        source_sha256=asset.sha256,
        fixture_kind=fixture_kind,
        document_type={
            "BL": "bill_of_lading",
            "INV": "invoice",
            "PL": "packing_list",
        }[suffix],
        set_id=("C" if fixture_kind == "clean" else "E") + set_number,
    )


def _r1_model_bytes() -> bytes:
    documents: list[dict[str, object]] = []
    for document_index, document_type in enumerate(R1_SUPERVISED_TEMPLATE_DOCUMENT_TYPES):
        fields = [
            {
                "field_name": field_name,
                "normalizer": normalizer,
                "selectors": [
                    {
                        "source_kind": "span_reading_order",
                        "page_number": 1,
                        "reading_order": field_index + 2,
                    }
                ],
                "value_transform": "raw-v1",
            }
            for field_index, (field_name, normalizer) in enumerate(
                R1_SUPERVISED_TEMPLATE_REQUIRED_FIELDS[document_type]
            )
        ]
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
                "fields": fields,
            }
        )
    model = NativePdfSemanticModelArtifactV1.model_validate(
        {
            "schema_version": "1.0",
            "artifact_kind": "native_pdf_semantic_model",
            "model_id": SUPERVISED_TEMPLATE_MODEL_ID,
            "model_revision": SUPERVISED_TEMPLATE_MODEL_REVISION,
            "inference_revision": "native-template-selector-v1",
            "documents": documents,
        }
    )
    return canonical_json_file_bytes(model.model_dump(mode="json", exclude_none=True))


@dataclass(frozen=True)
class _TestChain:
    root: Path
    policy: provenance._R1ProvenancePolicy
    receipt_path: Path
    development_packet_path: Path
    sealed_packet_path: Path
    split_path: Path
    labels_path: Path
    model_receipt_path: Path
    model_path: Path
    model_receipt_sha256: str
    model_sha256: str
    split: provenance._TrainingSplitV1


def _test_chain(tmp_path: Path) -> _TestChain:
    root = tmp_path / "p"
    assets: list[DatasetAssetReceiptV1] = []
    for number in range(1, 4):
        for fixture_directory, prefix in (("clean_sets", "C"), ("error_sets", "E")):
            for suffix in ("BL", "INV", "PL"):
                asset_id = f"{prefix.lower()}-{number}-{suffix.lower()}"
                assets.append(
                    DatasetAssetReceiptV1(
                        asset_id=asset_id,
                        relative_path=(
                            f"{fixture_directory}/set_{number:04d}/{prefix}{number:04d}_{suffix}.pdf"
                        ),
                        media_type="application/pdf",
                        byte_size=1,
                        sha256=_sha(f"metadata-{asset_id}"),
                        source_role="preview_candidate",
                    )
                )
    receipt = DatasetReceiptV1(
        dataset_id="synthetic-test-r1",
        dataset_revision="fixture-v1",
        received_as_of=date(2026, 8, 16),
        assets=assets,
    )
    receipt_digest = receipt_sha256(receipt)
    receipt_path = root / "source-attested-dataset-receipts" / f"{receipt_digest}.json"
    _write_canonical(receipt_path, receipt.model_dump(mode="json"))
    by_id = {asset.asset_id: asset for asset in assets}
    development = _packet(
        receipt=receipt,
        split="development",
        items=[_item(by_id[f"c-1-{suffix.lower()}"]) for suffix in ("BL", "INV", "PL")],
    )
    sealed = _packet(
        receipt=receipt,
        split="sealed",
        items=[_item(by_id[f"e-2-{suffix.lower()}"]) for suffix in ("BL", "INV", "PL")],
    )
    development_path = (
        root / "source-attested-packets" / f"{development.packet_receipt_sha256}.json"
    )
    sealed_path = root / "source-attested-packets" / f"{sealed.packet_receipt_sha256}.json"
    _write_canonical(development_path, development.model_dump(mode="json"))
    _write_canonical(sealed_path, sealed.model_dump(mode="json"))
    provisional_policy = provenance._R1ProvenancePolicy(
        dataset_id=receipt.dataset_id,
        dataset_revision=receipt.dataset_revision,
        dataset_receipt_sha256=receipt_digest,
        development_packet_receipt_sha256=development.packet_receipt_sha256,
        sealed_packet_receipt_sha256=sealed.packet_receipt_sha256,
        training_split_receipt_sha256=_sha("provisional-split"),
        training_labels_receipt_sha256=_sha("provisional-labels"),
        total_pdf_asset_count=18,
        clean_pdf_asset_count=9,
        error_pdf_asset_count=9,
        sets_per_fixture_kind=3,
        development_item_count=3,
        sealed_item_count=3,
        held_out_family_count=2,
        excluded_family_asset_count=12,
        training_asset_count=6,
        training_stratum_count=6,
        training_assets_per_stratum=1,
        field_label_count=80,
    )
    metadata = provenance._validate_r1_dataset_inventory(receipt=receipt, policy=provisional_policy)
    split = provenance._build_rederived_training_split(
        receipt=receipt,
        development_packet=development,
        sealed_packet=sealed,
        metadata_by_asset_id=metadata,
        policy=provisional_policy,
    )
    records: list[dict[str, object]] = []
    for asset in split.training_assets:
        fields = [
            {
                "field_name": field_name,
                "normalizer": normalizer,
                "normalized_value_sha256": _sha(f"test-value:{asset.asset_id}:{field_name}"),
            }
            for field_name, normalizer in R1_SUPERVISED_TEMPLATE_REQUIRED_FIELDS[
                asset.document_type
            ]
        ]
        record: dict[str, object] = {
            "asset_id": asset.asset_id,
            "source_sha256": asset.source_sha256,
            "fixture_kind": asset.fixture_kind,
            "document_type": asset.document_type,
            "family_set_number": asset.family_set_number,
            "content_stream_sha256": _sha(f"content-stream:{asset.asset_id}"),
            "field_labels": fields,
        }
        record["training_label_record_sha256"] = provenance._training_label_record_sha256(record)
        records.append(record)
    labels_payload: dict[str, object] = {
        "schema_version": "1.0",
        "artifact_kind": "source_attested_supervised_template_training_labels",
        "label_kind": "pdf-content-stream-semantic-slot-label-v1",
        "label_compiler_revision": "source-attested-family-safe-training-label-compiler-v1",
        "dataset_id": receipt.dataset_id,
        "dataset_revision": receipt.dataset_revision,
        "dataset_receipt_sha256": receipt_digest,
        "training_split_receipt_sha256": split.training_split_receipt_sha256,
        "record_count": len(records),
        "field_label_count": sum(len(record["field_labels"]) for record in records),
        "records": records,
    }
    labels_payload["training_labels_receipt_sha256"] = provenance._training_labels_sha256(
        labels_payload
    )
    labels = provenance._TrainingLabelsV1.model_validate(labels_payload)
    policy = provenance._R1ProvenancePolicy(
        dataset_id=receipt.dataset_id,
        dataset_revision=receipt.dataset_revision,
        dataset_receipt_sha256=receipt_digest,
        development_packet_receipt_sha256=development.packet_receipt_sha256,
        sealed_packet_receipt_sha256=sealed.packet_receipt_sha256,
        training_split_receipt_sha256=split.training_split_receipt_sha256,
        training_labels_receipt_sha256=labels.training_labels_receipt_sha256,
        total_pdf_asset_count=18,
        clean_pdf_asset_count=9,
        error_pdf_asset_count=9,
        sets_per_fixture_kind=3,
        development_item_count=3,
        sealed_item_count=3,
        held_out_family_count=2,
        excluded_family_asset_count=12,
        training_asset_count=6,
        training_stratum_count=6,
        training_assets_per_stratum=1,
        field_label_count=80,
    )
    split_path = (
        root / "supervised-template-training-splits" / f"{split.training_split_receipt_sha256}.json"
    )
    labels_path = (
        root
        / "supervised-template-training-labels"
        / f"{labels.training_labels_receipt_sha256}.json"
    )
    _write_canonical(split_path, split.model_dump(mode="json"))
    _write_canonical(labels_path, labels.model_dump(mode="json"))
    model_bytes = _r1_model_bytes()
    model_sha256 = _sha(model_bytes)
    model_path = root / "models" / "native-pdf-semantic" / model_sha256 / f"{model_sha256}.json"
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.write_bytes(model_bytes)
    model_receipt_payload: dict[str, object] = {
        "schema_version": "1.0",
        "artifact_kind": "source_attested_supervised_template_model_training_receipt",
        "dataset_id": receipt.dataset_id,
        "dataset_revision": receipt.dataset_revision,
        "dataset_receipt_sha256": receipt_digest,
        "training_split_receipt_sha256": split.training_split_receipt_sha256,
        "training_labels_receipt_sha256": labels.training_labels_receipt_sha256,
        "model_id": SUPERVISED_TEMPLATE_MODEL_ID,
        "model_kind": SUPERVISED_TEMPLATE_MODEL_KIND,
        "model_revision": SUPERVISED_TEMPLATE_MODEL_REVISION,
        "model_artifact_sha256": model_sha256,
        "training_recipe_sha256": _sha("recipe"),
        "training_environment_inventory_sha256": _sha("inventory"),
    }
    model_receipt_payload["model_training_receipt_sha256"] = (
        provenance._model_training_receipt_sha256(model_receipt_payload)
    )
    model_receipt = provenance._ModelTrainingReceiptV1.model_validate(model_receipt_payload)
    model_receipt_path = (
        root
        / "supervised-template-model-training-receipts"
        / f"{model_receipt.model_training_receipt_sha256}.json"
    )
    _write_canonical(model_receipt_path, model_receipt.model_dump(mode="json"))
    return _TestChain(
        root=root,
        policy=policy,
        receipt_path=receipt_path,
        development_packet_path=development_path,
        sealed_packet_path=sealed_path,
        split_path=split_path,
        labels_path=labels_path,
        model_receipt_path=model_receipt_path,
        model_path=model_path,
        model_receipt_sha256=model_receipt.model_training_receipt_sha256,
        model_sha256=model_sha256,
        split=split,
    )


def _load(chain: _TestChain) -> provenance.R1SupervisedTemplateProvenanceBinding:
    return provenance._load_with_policy(
        dataset_receipt_path=chain.receipt_path,
        development_packet_path=chain.development_packet_path,
        sealed_packet_path=chain.sealed_packet_path,
        training_split_path=chain.split_path,
        training_labels_path=chain.labels_path,
        model_training_receipt_path=chain.model_receipt_path,
        model_artifact_path=chain.model_path,
        expected_model_training_receipt_sha256=chain.model_receipt_sha256,
        expected_model_artifact_sha256=chain.model_sha256,
        policy=chain.policy,
    )


def test_loads_only_the_rederived_family_safe_metadata_chain(tmp_path: Path) -> None:
    chain = _test_chain(tmp_path)

    binding = _load(chain)

    assert chain.split.held_out_family_set_numbers == ["0001", "0002"]
    assert chain.split.excluded_family_asset_count == 12
    assert chain.split.training_asset_count == 6
    assert {asset.family_set_number for asset in chain.split.training_assets} == {"0003"}
    assert binding.model.artifact_sha256 == chain.model_sha256
    assert binding.verified_model().artifact.model_id == SUPERVISED_TEMPLATE_MODEL_ID


def test_rejects_a_model_receipt_with_a_different_model_identity(tmp_path: Path) -> None:
    chain = _test_chain(tmp_path)
    payload = json.loads(chain.model_receipt_path.read_text(encoding="utf-8"))
    payload["model_id"] = "another-model"
    payload.pop("model_training_receipt_sha256")
    changed_digest = _sha(canonical_json_bytes(payload))
    payload["model_training_receipt_sha256"] = changed_digest
    changed_path = (
        chain.root / "supervised-template-model-training-receipts" / f"{changed_digest}.json"
    )
    _write_canonical(changed_path, payload)

    with pytest.raises(
        provenance.SupervisedTemplateProvenanceError,
        match="model_receipt_binding_invalid",
    ):
        provenance._load_with_policy(
            dataset_receipt_path=chain.receipt_path,
            development_packet_path=chain.development_packet_path,
            sealed_packet_path=chain.sealed_packet_path,
            training_split_path=chain.split_path,
            training_labels_path=chain.labels_path,
            model_training_receipt_path=changed_path,
            model_artifact_path=chain.model_path,
            expected_model_training_receipt_sha256=changed_digest,
            expected_model_artifact_sha256=chain.model_sha256,
            policy=chain.policy,
        )


def test_rejects_label_artifact_that_carries_a_selector(tmp_path: Path) -> None:
    chain = _test_chain(tmp_path)
    payload = json.loads(chain.labels_path.read_text(encoding="utf-8"))
    records = payload["records"]
    assert isinstance(records, list)
    record = records[0]
    assert isinstance(record, dict)
    record["source_run_reading_orders"] = [1]
    payload.pop("training_labels_receipt_sha256")
    changed_digest = _sha(canonical_json_bytes(payload))
    payload["training_labels_receipt_sha256"] = changed_digest
    changed_path = chain.root / "supervised-template-training-labels" / f"{changed_digest}.json"
    _write_canonical(changed_path, payload)
    changed_policy = replace(chain.policy, training_labels_receipt_sha256=changed_digest)

    with pytest.raises(
        provenance.SupervisedTemplateProvenanceError,
        match="labels_artifact_invalid",
    ):
        provenance._load_with_policy(
            dataset_receipt_path=chain.receipt_path,
            development_packet_path=chain.development_packet_path,
            sealed_packet_path=chain.sealed_packet_path,
            training_split_path=chain.split_path,
            training_labels_path=changed_path,
            model_training_receipt_path=chain.model_receipt_path,
            model_artifact_path=chain.model_path,
            expected_model_training_receipt_sha256=chain.model_receipt_sha256,
            expected_model_artifact_sha256=chain.model_sha256,
            policy=changed_policy,
        )


def test_rejects_a_symlinked_metadata_artifact_before_opening(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    chain = _test_chain(tmp_path)
    original_lstat = Path.lstat

    def symlink_lstat(self: Path) -> SimpleNamespace:
        if self == chain.receipt_path:
            return SimpleNamespace(st_mode=stat.S_IFLNK)
        return original_lstat(self)  # type: ignore[return-value]

    monkeypatch.setattr(Path, "lstat", symlink_lstat)

    with pytest.raises(
        provenance.SupervisedTemplateProvenanceError,
        match="dataset_artifact_invalid",
    ):
        _load(chain)


def test_provenance_module_has_no_source_truth_training_or_csv_import() -> None:
    module_path = Path(provenance.__file__)
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_modules = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ] + [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]

    assert all("source_attested_synthetic_pdf_truth" not in name for name in imported_modules)
    assert all("supervised_template_training" not in name for name in imported_modules)
    assert "csv" not in imported_modules
