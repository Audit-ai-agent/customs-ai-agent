from __future__ import annotations

import ast
import hashlib
import inspect
from pathlib import Path

import pytest

import customs_ai_agent.evaluation.native_pdf_semantic_adapter as semantic_adapter
import customs_ai_agent.evaluation.supervised_template_direct_selector_trainer as trainer
from customs_ai_agent.evaluation.document_ocr_quality import (
    CandidateSpanV1,
    CandidateTableCellV1,
    normalize_field_value,
)
from customs_ai_agent.evaluation.native_pdf_vector_grid import (
    NativePdfVectorGridExtraction,
    NativePdfVectorGridPage,
)
from customs_ai_agent.evaluation.supervised_template_training import (
    SourceAttestedSupervisedTemplateTrainingLabelsV1,
    SourceAttestedSupervisedTemplateTrainingSplitV1,
    SupervisedTemplateTrainingAssetV1,
    SupervisedTemplateTrainingFieldLabelV1,
    SupervisedTemplateTrainingLabelRecordV1,
    training_asset_list_sha256,
    training_label_record_sha256,
    training_labels_receipt_sha256,
    training_split_receipt_sha256,
)


def _sha(value: bytes | str) -> str:
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _field_digest(value: str, normalizer: str) -> str:
    return _sha(normalize_field_value(value, normalizer))


def _field_value(*, document_type: str, field_name: str, normalizer: str, ordinal: int) -> str:
    if document_type == "packing_list" and field_name == "net_weight":
        return "900041"
    if document_type == "packing_list" and field_name == "gross_weight":
        return "900042"
    if normalizer == "numeric-v1":
        return str(900_100 + ordinal)
    if normalizer == "identifier-v1":
        return f"ID{ordinal:04d}"
    return f"TEXT{ordinal:04d}"


def _candidate_extraction(
    document_type: str,
) -> tuple[NativePdfVectorGridExtraction, dict[str, str]]:
    base_x = {"bill_of_lading": 0.10, "invoice": 0.30, "packing_list": 0.50}[document_type]
    values: dict[str, str] = {}
    spans: list[CandidateSpanV1] = []
    for ordinal, (field_name, normalizer) in enumerate(
        trainer._REQUIRED_FIELDS[document_type],
        start=1,  # type: ignore[index]
    ):
        if field_name == "net_gross_weight":
            continue
        value = _field_value(
            document_type=document_type,
            field_name=field_name,
            normalizer=normalizer,
            ordinal=ordinal,
        )
        values[field_name] = value
        spans.append(
            CandidateSpanV1(
                page_number=1,
                reading_order=len(spans) + 1,
                text=value,
                anchor_x=base_x,
                anchor_y=0.05 + (len(spans) * 0.01),
            )
        )
    if document_type == "packing_list":
        values["net_gross_weight"] = f"{values['net_weight']} / {values['gross_weight']}"
    spans.append(
        CandidateSpanV1(
            page_number=1,
            reading_order=len(spans) + 1,
            text="FILLER",
            anchor_x=base_x,
            anchor_y=0.90,
        )
    )
    return (
        NativePdfVectorGridExtraction(
            pages=(NativePdfVectorGridPage(page_number=1, width=100, height=100),),
            spans=tuple(spans),
            table_cells=(
                CandidateTableCellV1(
                    page_number=1,
                    table_index=1,
                    row_index=0,
                    column_index=0,
                    text="GRID",
                ),
            ),
            form_regions=(),
        ),
        values,
    )


def _training_fixture(tmp_path: Path):  # type: ignore[no-untyped-def]
    source_directory = tmp_path / "workspace" / "_local" / "data" / "synthetic-pdf"
    source_directory.mkdir(parents=True)
    assets: list[SupervisedTemplateTrainingAssetV1] = []
    records: list[SupervisedTemplateTrainingLabelRecordV1] = []
    evidence_by_bytes: dict[bytes, NativePdfVectorGridExtraction] = {}
    for fixture_kind in ("clean", "error"):
        for document_type, suffix in (
            ("bill_of_lading", "BL"),
            ("invoice", "INV"),
            ("packing_list", "PL"),
        ):
            payload = f"selected-{fixture_kind}-{document_type}".encode("ascii")
            prefix = "C" if fixture_kind == "clean" else "E"
            path = (
                source_directory
                / ("clean_sets" if fixture_kind == "clean" else "error_sets")
                / "set_0002"
                / f"{prefix}0002_{suffix}.pdf"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(payload)
            asset = SupervisedTemplateTrainingAssetV1(
                asset_id=f"{fixture_kind}-{suffix.casefold()}",
                source_sha256=_sha(payload),
                fixture_kind=fixture_kind,
                document_type=document_type,
                family_set_number="0002",
            )
            assets.append(asset)
            extraction, values = _candidate_extraction(document_type)
            evidence_by_bytes[payload] = extraction
            labels = [
                SupervisedTemplateTrainingFieldLabelV1(
                    field_name=field_name,
                    normalizer=normalizer,
                    normalized_value_sha256=_field_digest(values[field_name], normalizer),
                )
                for field_name, normalizer in trainer._REQUIRED_FIELDS[document_type]  # type: ignore[index]
            ]
            record_payload: dict[str, object] = {
                "asset_id": asset.asset_id,
                "source_sha256": asset.source_sha256,
                "fixture_kind": fixture_kind,
                "document_type": document_type,
                "family_set_number": "0002",
                "content_stream_sha256": _sha(f"stream-{asset.asset_id}"),
                "field_labels": [label.model_dump(mode="json") for label in labels],
            }
            record_payload["training_label_record_sha256"] = training_label_record_sha256(
                record_payload
            )
            records.append(SupervisedTemplateTrainingLabelRecordV1.model_validate(record_payload))
    assets.sort(key=lambda asset: asset.asset_id)
    records.sort(key=lambda record: record.asset_id)
    split_payload: dict[str, object] = {
        "schema_version": "1.0",
        "artifact_kind": "source_attested_supervised_template_training_split",
        "split_revision": "source-attested-family-safe-training-split-v1",
        "dataset_id": "synthetic-pdf",
        "dataset_revision": "r1",
        "dataset_receipt_sha256": _sha("dataset"),
        "development_packet_receipt_sha256": _sha("development"),
        "sealed_packet_receipt_sha256": _sha("sealed"),
        "held_out_family_set_numbers": ["0001"],
        "held_out_family_count": 1,
        "excluded_family_asset_count": 6,
        "training_assets": [asset.model_dump(mode="json") for asset in assets],
        "training_asset_count": len(assets),
        "training_asset_list_sha256": training_asset_list_sha256(assets),
        "training_strata": [
            {"fixture_kind": fixture_kind, "document_type": document_type, "asset_count": 1}
            for fixture_kind in ("clean", "error")
            for document_type in ("bill_of_lading", "invoice", "packing_list")
        ],
    }
    split_payload["training_split_receipt_sha256"] = training_split_receipt_sha256(split_payload)
    split = SourceAttestedSupervisedTemplateTrainingSplitV1.model_validate(split_payload)
    labels_payload: dict[str, object] = {
        "schema_version": "1.0",
        "artifact_kind": "source_attested_supervised_template_training_labels",
        "label_kind": "pdf-content-stream-semantic-slot-label-v1",
        "label_compiler_revision": "source-attested-family-safe-training-label-compiler-v1",
        "dataset_id": split.dataset_id,
        "dataset_revision": split.dataset_revision,
        "dataset_receipt_sha256": split.dataset_receipt_sha256,
        "training_split_receipt_sha256": split.training_split_receipt_sha256,
        "record_count": len(records),
        "field_label_count": sum(len(record.field_labels) for record in records),
        "records": [record.model_dump(mode="json") for record in records],
    }
    labels_payload["training_labels_receipt_sha256"] = training_labels_receipt_sha256(
        labels_payload
    )
    labels = SourceAttestedSupervisedTemplateTrainingLabelsV1.model_validate(labels_payload)
    return source_directory, split, labels, evidence_by_bytes, values


def _allow_fixture_counts(
    monkeypatch: pytest.MonkeyPatch, split: SourceAttestedSupervisedTemplateTrainingSplitV1
) -> None:
    monkeypatch.setattr(trainer, "_EXPECTED_R1_TRAINING_ASSET_COUNT", split.training_asset_count)
    monkeypatch.setattr(trainer, "_EXPECTED_R1_HELD_OUT_FAMILY_COUNT", split.held_out_family_count)
    monkeypatch.setattr(
        trainer,
        "_EXPECTED_R1_EXCLUDED_FAMILY_ASSET_COUNT",
        split.excluded_family_asset_count,
    )
    monkeypatch.setattr(
        trainer,
        "_EXPECTED_R1_TRAINING_STRATA",
        tuple(
            (entry.fixture_kind, entry.document_type, entry.asset_count)
            for entry in split.training_strata
        ),
    )


def test_candidate_native_trainer_discovers_join_and_writes_isolated_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_directory, split, labels, evidence_by_bytes, last_values = _training_fixture(tmp_path)
    monkeypatch.setattr(
        trainer,
        "build_source_attested_supervised_template_training_split",
        lambda **_kwargs: split,
    )
    _allow_fixture_counts(monkeypatch, split)

    def fake_extract(*, pdf_bytes: bytes) -> NativePdfVectorGridExtraction:
        return evidence_by_bytes[pdf_bytes]

    monkeypatch.setattr(trainer, "extract_native_pdf_vector_grid_from_pdf_bytes", fake_extract)
    monkeypatch.setattr(
        semantic_adapter, "extract_native_pdf_vector_grid_from_pdf_bytes", fake_extract
    )
    result = trainer.train_source_attested_supervised_template_direct_selector_model(
        receipt=object(),  # type: ignore[arg-type]
        development_packet=object(),  # type: ignore[arg-type]
        sealed_packet=object(),  # type: ignore[arg-type]
        training_split=split,
        training_labels=labels,
        source_directory=source_directory,
        model_id=trainer.SUPERVISED_TEMPLATE_DIRECT_SELECTOR_MODEL_ID,
        model_revision=trainer.SUPERVISED_TEMPLATE_DIRECT_SELECTOR_MODEL_REVISION,
        training_environment_inventory_sha256=_sha("environment"),
    )

    packing_list = next(
        document for document in result.model.documents if document.document_type == "packing_list"
    )
    net_gross = next(
        field for field in packing_list.fields if field.field_name == "net_gross_weight"
    )
    assert net_gross.value_transform == "join-slash-v1"
    assert len(net_gross.selectors) == 2
    assert (
        result.model_training_receipt.model_id
        == trainer.SUPERVISED_TEMPLATE_DIRECT_SELECTOR_MODEL_ID
    )
    assert result.model_training_receipt.model_artifact_sha256 == result.model_artifact_sha256
    assert all(
        asset.asset_id.encode("utf-8") not in result.model_artifact_bytes
        for asset in split.training_assets
    )
    assert all(
        value.encode("utf-8") not in result.model_artifact_bytes for value in last_values.values()
    )

    root = tmp_path / "workspace" / "_local" / "data" / "document-preview"
    root.mkdir()
    paths = trainer.write_source_attested_supervised_template_direct_selector_training_result(
        result=result,
        recipe_output_directory=root / "supervised-template-training-recipes",
        model_root_directory=root / "models" / "native-pdf-semantic",
        model_training_receipt_output_directory=root / "supervised-template-training-receipts",
    )
    assert paths.model_path.name == f"{result.model_artifact_sha256}.json"
    assert paths.model_path.parent.name == result.model_artifact_sha256
    assert paths.model_path.parent.parent.name == "native-pdf-semantic"
    assert paths.model_training_receipt_path.name == (
        f"{result.model_training_receipt.model_training_receipt_sha256}.json"
    )
    assert (
        trainer.write_source_attested_supervised_template_direct_selector_training_result(
            result=result,
            recipe_output_directory=root / "supervised-template-training-recipes",
            model_root_directory=root / "models" / "native-pdf-semantic",
            model_training_receipt_output_directory=root / "supervised-template-training-receipts",
        )
        == paths
    )


def test_r1_count_guard_runs_before_any_training_pdf_is_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_directory, split, labels, _evidence_by_bytes, _values = _training_fixture(tmp_path)
    monkeypatch.setattr(
        trainer,
        "build_source_attested_supervised_template_training_split",
        lambda **_kwargs: split,
    )
    monkeypatch.setattr(
        trainer,
        "_read_verified_training_pdf",
        lambda **_kwargs: pytest.fail("training PDF was opened before the r1 count guard"),
    )

    with pytest.raises(
        trainer.SupervisedTemplateDirectSelectorTrainerError,
        match="r1_split_counts_invalid",
    ):
        trainer.train_source_attested_supervised_template_direct_selector_model(
            receipt=object(),  # type: ignore[arg-type]
            development_packet=object(),  # type: ignore[arg-type]
            sealed_packet=object(),  # type: ignore[arg-type]
            training_split=split,
            training_labels=labels,
            source_directory=source_directory,
            model_id=trainer.SUPERVISED_TEMPLATE_DIRECT_SELECTOR_MODEL_ID,
            model_revision=trainer.SUPERVISED_TEMPLATE_DIRECT_SELECTOR_MODEL_REVISION,
            training_environment_inventory_sha256=_sha("environment"),
        )


def test_rederived_split_mismatch_rejects_before_any_training_pdf_is_opened(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_directory, split, labels, _evidence_by_bytes, _values = _training_fixture(tmp_path)
    rebuilt = split.model_copy(update={"training_split_receipt_sha256": "0" * 64})
    monkeypatch.setattr(
        trainer,
        "build_source_attested_supervised_template_training_split",
        lambda **_kwargs: rebuilt,
    )
    monkeypatch.setattr(
        trainer,
        "_read_verified_training_pdf",
        lambda **_kwargs: pytest.fail("training PDF was opened before split re-derivation"),
    )

    with pytest.raises(
        trainer.SupervisedTemplateDirectSelectorTrainerError,
        match="split_rederivation_invalid",
    ):
        trainer.train_source_attested_supervised_template_direct_selector_model(
            receipt=object(),  # type: ignore[arg-type]
            development_packet=object(),  # type: ignore[arg-type]
            sealed_packet=object(),  # type: ignore[arg-type]
            training_split=split,
            training_labels=labels,
            source_directory=source_directory,
            model_id=trainer.SUPERVISED_TEMPLATE_DIRECT_SELECTOR_MODEL_ID,
            model_revision=trainer.SUPERVISED_TEMPLATE_DIRECT_SELECTOR_MODEL_REVISION,
            training_environment_inventory_sha256=_sha("environment"),
        )


def test_training_asset_tampering_rejects_before_candidate_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source_directory, split, labels, _evidence_by_bytes, _values = _training_fixture(tmp_path)
    _allow_fixture_counts(monkeypatch, split)
    monkeypatch.setattr(
        trainer,
        "build_source_attested_supervised_template_training_split",
        lambda **_kwargs: split,
    )
    first = split.training_assets[0]
    tampered = trainer._source_pdf_path(source_directory=source_directory, asset=first)
    tampered.write_bytes(b"tampered")
    monkeypatch.setattr(
        trainer,
        "extract_native_pdf_vector_grid_from_pdf_bytes",
        lambda **_kwargs: pytest.fail("tampered PDF reached candidate evidence"),
    )

    with pytest.raises(
        trainer.SupervisedTemplateDirectSelectorTrainerError,
        match="training_pdf_sha256_mismatch",
    ):
        trainer.train_source_attested_supervised_template_direct_selector_model(
            receipt=object(),  # type: ignore[arg-type]
            development_packet=object(),  # type: ignore[arg-type]
            sealed_packet=object(),  # type: ignore[arg-type]
            training_split=split,
            training_labels=labels,
            source_directory=source_directory,
            model_id=trainer.SUPERVISED_TEMPLATE_DIRECT_SELECTOR_MODEL_ID,
            model_revision=trainer.SUPERVISED_TEMPLATE_DIRECT_SELECTOR_MODEL_REVISION,
            training_environment_inventory_sha256=_sha("environment"),
        )


def test_trainer_uses_hash_only_labels_and_no_source_profile_import() -> None:
    module_path = Path(trainer.__file__)
    tree = ast.parse(module_path.read_text(encoding="utf-8"))
    imported_modules = [
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    ] + [node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
    attributes = {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute) and isinstance(node.attr, str)
    }

    assert all("source_attested_synthetic_pdf_truth" not in name for name in imported_modules)
    assert "source_run_reading_orders" not in attributes
    assert "value_transform" not in {
        node.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "label"
    }
    signature = inspect.signature(
        trainer.train_source_attested_supervised_template_direct_selector_model
    )
    assert "clean_manifest_path" not in signature.parameters
    assert "error_labels_path" not in signature.parameters
    assert "development_pdf_path" not in signature.parameters
    assert "sealed_pdf_path" not in signature.parameters
    assert "model_artifact_sha256" not in signature.parameters


def test_model_identity_is_fixed_for_current_r1_scope(tmp_path: Path) -> None:
    source_directory, split, labels, _evidence_by_bytes, _values = _training_fixture(tmp_path)
    with pytest.raises(
        trainer.SupervisedTemplateDirectSelectorTrainerError,
        match="model_identity_invalid",
    ):
        trainer.train_source_attested_supervised_template_direct_selector_model(
            receipt=object(),  # type: ignore[arg-type]
            development_packet=object(),  # type: ignore[arg-type]
            sealed_packet=object(),  # type: ignore[arg-type]
            training_split=split,
            training_labels=labels,
            source_directory=source_directory,
            model_id="different-model",
            model_revision=trainer.SUPERVISED_TEMPLATE_DIRECT_SELECTOR_MODEL_REVISION,
            training_environment_inventory_sha256=_sha("environment"),
        )
