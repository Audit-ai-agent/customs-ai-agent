from __future__ import annotations

import csv
import hashlib
import json
from datetime import date, datetime, timezone
from pathlib import Path

import pytest

from customs_ai_agent.evaluation.document_gold import DatasetAssetReceiptV1, DatasetReceiptV1
from customs_ai_agent.evaluation.document_ocr_quality import (
    CandidateDocumentOutputV1,
    CandidateFieldV1,
    CandidateOutputBundleV1,
    CandidateSpanV1,
    CandidateTableCellV1,
    DocumentExtractionGoldV2,
    EvaluationPacketV2,
    GoldSpanV2,
    GoldTableCellV2,
    OcrEvaluationResultV1,
    build_draft_gold_record,
    build_gold_packet,
    build_stratified_pdf_gold_packets,
    build_synthetic_pdf_draft_gold,
    canonical_json_file_bytes,
    evaluation_result_sha256,
    read_canonical_evaluation_result,
    score_candidate_output,
    select_best_development_result,
    verify_receipt_source_assets,
    write_evaluation_result,
)
from tools.prepare_stratified_synthetic_pdf_gold_packet import prepare_stratified_packets
from tools.seed_synthetic_pdf_ocr_gold import seed_synthetic_pdf_gold


_CLEAN_HEADERS = [
    "set_id",
    "exporter_name",
    "importer_name",
    "consignee",
    "bank",
    "load_port",
    "disch_port",
    "final_dest",
    "vessel",
    "voyage",
    "bl_no",
    "inv_no",
    "inv_date",
    "lc_no",
    "container_no",
    "seal_no",
    "good",
    "unit",
    "qty",
    "unit_price",
    "amount",
    "total",
    "cartons",
    "containers",
    "net_weight",
    "gross_weight",
    "cbm",
]
_ERROR_HEADERS = [
    "set_id",
    "num_errors_in_set",
    "error_type",
    "affected_doc",
    "field",
    "correct_value",
    "wrong_value",
    "note",
]


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _receipt(per_stratum: int = 14) -> DatasetReceiptV1:
    assets: list[DatasetAssetReceiptV1] = []
    for fixture_directory, prefix in (("clean_sets", "C"), ("error_sets", "E")):
        for index in range(1, per_stratum + 1):
            record = f"{prefix}{index:04}"
            for suffix in ("BL", "INV", "PL"):
                relative_path = f"{fixture_directory}/set_{index:04}/{record}_{suffix}.pdf"
                assets.append(
                    DatasetAssetReceiptV1(
                        asset_id=f"asset-{fixture_directory[:1]}-{index:04}-{suffix}",
                        relative_path=relative_path,
                        media_type="application/pdf",
                        byte_size=100 + index,
                        sha256=_sha(relative_path),
                        source_role="preview_candidate",
                    )
                )
    return DatasetReceiptV1(
        dataset_id="synthetic-pdf",
        dataset_revision="human-source-v1",
        received_as_of="2026-08-14",
        assets=assets,
    )


def _write_ground_truth(tmp_path: Path, *, count: int = 14) -> tuple[Path, Path]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    clean_path = tmp_path / "clean_manifest.csv"
    error_path = tmp_path / "error_labels.csv"
    with clean_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_CLEAN_HEADERS)
        writer.writeheader()
        for index in range(1, count + 1):
            row = {field: f"{field}-{index}" for field in _CLEAN_HEADERS}
            row.update(
                {
                    "set_id": f"C{index:04}",
                    "qty": str(index),
                    "unit_price": str(index * 10),
                    "amount": str(index * 20),
                    "total": str(index * 30),
                    "cartons": str(index + 2),
                    "containers": "1",
                    "net_weight": str(index * 2),
                    "gross_weight": str(index * 3),
                    "cbm": str(index * 4),
                }
            )
            writer.writerow(row)
    with error_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_ERROR_HEADERS)
        writer.writeheader()
        for index in range(1, count + 1):
            for affected_doc, field in (
                ("BILL_OF_LADING", "Container No."),
                ("COMMERCIAL_INVOICE", "Amount"),
                ("PACKING_LIST", "Quantity"),
            ):
                writer.writerow(
                    {
                        "set_id": f"E{index:04}",
                        "num_errors_in_set": "3",
                        "error_type": "synthetic",
                        "affected_doc": affected_doc,
                        "field": field,
                        "correct_value": f"correct-{index}",
                        "wrong_value": f"wrong-{index}",
                        "note": "fixture",
                    }
                )
    return clean_path, error_path


def _completed_gold(packet: EvaluationPacketV2, drafts: list[DocumentExtractionGoldV2]):
    approved: list[DocumentExtractionGoldV2] = []
    for index, draft in enumerate(drafts, start=1):
        ready = build_draft_gold_record(
            packet=packet,
            item=next(item for item in packet.items if item.asset_id == draft.asset_id),
            fields=draft.fields,
            spans=[GoldSpanV2(page_number=1, text=f"span {index}", reading_order=1)],
            table_cells=[
                GoldTableCellV2(
                    page_number=1,
                    table_index=1,
                    row_index=0,
                    column_index=0,
                    text=f"cell {index}",
                )
            ],
            source_attestation=draft.source_attestation,
        )
        from customs_ai_agent.evaluation.document_ocr_quality import approve_gold_record

        approved.append(
            approve_gold_record(
                ready,
                approval_id=f"approval-{index}",
                reviewer_id_sha256=_sha("reviewer"),
                approved_at=datetime(2026, 8, 14, tzinfo=timezone.utc),
            )
        )
    return build_gold_packet(packet=packet, records=approved)


def _matching_candidate(packet: EvaluationPacketV2, gold_packet) -> CandidateOutputBundleV1:
    return CandidateOutputBundleV1(
        dataset_id=packet.dataset_id,
        dataset_revision=packet.dataset_revision,
        packet_receipt_sha256=packet.packet_receipt_sha256,
        documents=[
            CandidateDocumentOutputV1(
                asset_id=record.asset_id,
                source_sha256=record.source_sha256,
                fields=[
                    CandidateFieldV1(field_name=field.field_name, value=field.value)
                    for field in record.fields
                ],
                spans=[
                    CandidateSpanV1(
                        page_number=span.page_number,
                        text=span.text,
                        bbox=span.bbox,
                        reading_order=span.reading_order,
                    )
                    for span in record.spans
                ],
                table_cells=[
                    CandidateTableCellV1(
                        page_number=cell.page_number,
                        table_index=cell.table_index,
                        row_index=cell.row_index,
                        column_index=cell.column_index,
                        text=cell.text,
                    )
                    for cell in record.table_cells
                ],
            )
            for record in gold_packet.records
        ],
    )


def test_stratified_v2_packet_is_exact_and_deterministic(tmp_path: Path) -> None:
    receipt = _receipt()
    _, labels = _write_ground_truth(tmp_path)
    development, sealed = build_stratified_pdf_gold_packets(
        receipt,
        prepared_as_of=date(2026, 8, 14),
        error_labels_path=labels,
    )

    assert development.schema_version == sealed.schema_version == "2.0"
    assert development.packet_kind == sealed.packet_kind == "stratified-v2"
    assert len(development.items) == 60
    assert len(sealed.items) == 20
    assert {item.asset_id for item in development.items}.isdisjoint(
        item.asset_id for item in sealed.items
    )
    assert set(
        {
            (item.fixture_kind, item.document_type): sum(
                1
                for selected in development.items
                if (selected.fixture_kind, selected.document_type)
                == (item.fixture_kind, item.document_type)
            )
            for item in development.items
        }.values()
    ) == {10}
    sealed_counts = {
        stratum: sum(
            1 for item in sealed.items if (item.fixture_kind, item.document_type) == stratum
        )
        for stratum in {(item.fixture_kind, item.document_type) for item in sealed.items}
    }
    assert sorted(sealed_counts.values()) == [3, 3, 3, 3, 4, 4]
    again = build_stratified_pdf_gold_packets(
        receipt,
        prepared_as_of=date(2026, 8, 14),
        error_labels_path=labels,
    )
    assert [item.asset_id for item in again[0].items] == [
        item.asset_id for item in development.items
    ]
    assert [item.asset_id for item in again[1].items] == [item.asset_id for item in sealed.items]


def test_stratified_v2_refuses_missing_stratum_or_bad_source_path(tmp_path: Path) -> None:
    _, labels = _write_ground_truth(tmp_path)
    _, underfilled_labels = _write_ground_truth(tmp_path / "underfilled", count=12)
    with pytest.raises(ValueError, match="at least thirteen"):
        build_stratified_pdf_gold_packets(
            _receipt(per_stratum=12),
            prepared_as_of=date(2026, 8, 14),
            error_labels_path=underfilled_labels,
        )
    receipt = _receipt()
    malformed = receipt.model_copy(
        update={
            "assets": [
                receipt.assets[0].model_copy(update={"relative_path": "unexpected.pdf"}),
                *receipt.assets[1:],
            ]
        }
    )
    with pytest.raises(ValueError, match="required clean/error layout"):
        build_stratified_pdf_gold_packets(
            malformed,
            prepared_as_of=date(2026, 8, 14),
            error_labels_path=labels,
        )


def test_error_strata_include_only_actually_labeled_error_documents(tmp_path: Path) -> None:
    receipt = _receipt()
    _, labels = _write_ground_truth(tmp_path)
    rows = list(csv.DictReader(labels.open("r", encoding="utf-8", newline="")))
    with labels.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_ERROR_HEADERS)
        writer.writeheader()
        writer.writerows(row for row in rows if row["set_id"] != "E0014")

    development, sealed = build_stratified_pdf_gold_packets(
        receipt,
        prepared_as_of=date(2026, 8, 14),
        error_labels_path=labels,
    )

    selected_error_ids = {
        item.set_id for item in [*development.items, *sealed.items] if item.fixture_kind == "error"
    }
    assert "E0014" not in selected_error_ids
    assert selected_error_ids <= {f"E{index:04}" for index in range(1, 14)}


def test_source_attested_draft_applies_error_wrong_value_and_requires_review(
    tmp_path: Path,
) -> None:
    receipt = _receipt()
    clean, labels = _write_ground_truth(tmp_path)
    development, _ = build_stratified_pdf_gold_packets(
        receipt,
        prepared_as_of=date(2026, 8, 14),
        error_labels_path=labels,
    )

    gold_packet = build_synthetic_pdf_draft_gold(
        receipt=receipt,
        packet=development,
        clean_manifest_path=clean,
        error_labels_path=labels,
    )

    assert gold_packet.status == "draft"
    assert len(gold_packet.records) == 60
    error_record = next(
        record
        for record in gold_packet.records
        if record.fixture_kind == "error" and record.document_type == "invoice"
    )
    assert error_record.adjudication_status == "draft"
    assert next(
        field for field in error_record.fields if field.field_name == "amount"
    ).value.startswith("wrong-")
    assert (
        error_record.source_attestation.clean_manifest_sha256
        == hashlib.sha256(clean.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="requires reviewed fields, spans, and table cells"):
        from customs_ai_agent.evaluation.document_ocr_quality import approve_gold_record

        approve_gold_record(
            error_record,
            approval_id="approval",
            reviewer_id_sha256=_sha("reviewer"),
            approved_at=datetime(2026, 8, 14, tzinfo=timezone.utc),
        )


def test_approved_gold_scoring_and_sealed_result_are_fail_closed(tmp_path: Path) -> None:
    receipt = _receipt()
    clean, labels = _write_ground_truth(tmp_path)
    _, sealed = build_stratified_pdf_gold_packets(
        receipt,
        prepared_as_of=date(2026, 8, 14),
        error_labels_path=labels,
    )
    drafts = build_synthetic_pdf_draft_gold(
        receipt=receipt,
        packet=sealed,
        clean_manifest_path=clean,
        error_labels_path=labels,
    )
    gold_packet = _completed_gold(sealed, drafts.records)
    candidate = _matching_candidate(sealed, gold_packet)
    result = score_candidate_output(
        receipt=receipt,
        packet=sealed,
        gold_packet=gold_packet,
        candidate_output=candidate,
        engine_kind="pdf_text_layer",
        extractor_revision="native-text-v1",
        configuration_sha256=_sha("config"),
        model_inventory_sha256=_sha("inventory"),
    )

    assert result.packet_kind == "sealed"
    assert result.passed is True
    assert result.metrics.field_exact_match == 1.0
    assert result.metrics.ocr_cer == 0.0
    assert result.metrics.table_cell_f1 == 1.0
    output = tmp_path / "workspace" / "_local" / "data" / "document-preview" / "evaluation-results"
    path = write_evaluation_result(result=result, output_directory=output)
    assert path.name == f"{result.evaluation_result_sha256}.json"
    assert read_canonical_evaluation_result(path) == result
    with pytest.raises(ValueError, match="sealed evaluation already recorded"):
        write_evaluation_result(result=result, output_directory=output)
    path.write_bytes(
        canonical_json_file_bytes(result.model_dump(mode="json", warnings=False)) + b" "
    )
    with pytest.raises(ValueError, match="not canonical"):
        read_canonical_evaluation_result(path)


def test_candidate_and_gold_source_drift_fail_closed(tmp_path: Path) -> None:
    source = tmp_path / "source"
    assets: list[DatasetAssetReceiptV1] = []
    for index in range(1, 2):
        for fixture_directory, prefix in (("clean_sets", "C"), ("error_sets", "E")):
            for suffix in ("BL", "INV", "PL"):
                record = f"{prefix}{index:04}"
                relative = f"{fixture_directory}/set_{index:04}/{record}_{suffix}.pdf"
                path = source / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                content = relative.encode("utf-8")
                path.write_bytes(content)
                assets.append(
                    DatasetAssetReceiptV1(
                        asset_id=f"asset-{fixture_directory[:1]}-{suffix}",
                        relative_path=relative,
                        media_type="application/pdf",
                        byte_size=len(content),
                        sha256=hashlib.sha256(content).hexdigest(),
                        source_role="preview_candidate",
                    )
                )
    receipt = DatasetReceiptV1(
        dataset_id="source-drift",
        dataset_revision="v1",
        received_as_of="2026-08-14",
        assets=assets,
    )
    verify_receipt_source_assets(receipt=receipt, source_directory=source)
    (source / "clean_sets" / "set_0001" / "C0001_BL.pdf").write_bytes(b"changed")
    with pytest.raises(ValueError, match="drifted"):
        verify_receipt_source_assets(receipt=receipt, source_directory=source)


def test_development_selection_uses_fixed_metric_order_and_rejects_no_pass() -> None:
    def result(
        *, engine_kind: str, field: float, cer: float, table: float
    ) -> OcrEvaluationResultV1:
        payload: dict[str, object] = {
            "schema_version": "1.0",
            "dataset_id": "synthetic-pdf",
            "dataset_revision": "human-source-v1",
            "packet_kind": "development",
            "packet_receipt_sha256": _sha("packet"),
            "gold_receipt_sha256": _sha("gold"),
            "engine_kind": engine_kind,
            "extractor_revision": f"{engine_kind}-v1",
            "configuration_sha256": _sha(f"config-{engine_kind}"),
            "model_inventory_sha256": _sha(f"inventory-{engine_kind}"),
            "passed": field >= 0.99 and cer <= 0.01 and table >= 0.98,
            "metrics": {
                "field_exact_match": field,
                "ocr_cer": cer,
                "table_cell_f1": table,
            },
        }
        payload["evaluation_result_sha256"] = evaluation_result_sha256(payload)
        return OcrEvaluationResultV1.model_validate(payload)

    selected = select_best_development_result(
        [
            result(engine_kind="paddleocr", field=0.99, cer=0.01, table=0.98),
            result(engine_kind="docling", field=0.99, cer=0.005, table=0.98),
            result(engine_kind="pdf_text_layer", field=0.995, cer=0.01, table=0.98),
        ]
    )
    assert selected.engine_kind == "pdf_text_layer"
    with pytest.raises(ValueError, match="no development OCR candidate"):
        select_best_development_result(
            [result(engine_kind="docling", field=0.98, cer=0.02, table=0.97)]
        )


def test_packet_and_gold_tools_write_only_local_metadata(tmp_path: Path) -> None:
    receipt = _receipt()
    workspace = tmp_path / "workspace"
    clean, labels = _write_ground_truth(workspace)
    receipt_path = workspace / "receipt.json"
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_bytes(
        canonical_json_file_bytes(receipt.model_dump(mode="json", warnings=False))
    )
    packet_output = workspace / "_local" / "data" / "document-preview" / "gold-packets"
    summary = prepare_stratified_packets(
        receipt_path=receipt_path,
        error_labels_path=labels,
        output_directory=packet_output,
        prepared_as_of=date(2026, 8, 14),
    )
    assert summary["development_count"] == 60
    packet_path = packet_output / "synthetic-pdf-development-stratified-v2.json"
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    assert packet["packet_kind"] == "stratified-v2"
    assert "relative_path" not in json.dumps(packet)

    source = workspace / "source"
    assets: list[DatasetAssetReceiptV1] = []
    for asset in receipt.assets:
        path = source / asset.relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        content = f"{asset.asset_id}:{asset.relative_path}".encode("utf-8")
        path.write_bytes(content)
        assets.append(
            asset.model_copy(
                update={"byte_size": len(content), "sha256": hashlib.sha256(content).hexdigest()}
            )
        )
    physical_receipt = receipt.model_copy(update={"assets": assets})
    receipt_path.write_bytes(
        canonical_json_file_bytes(physical_receipt.model_dump(mode="json", warnings=False))
    )
    packet_path.write_bytes(
        canonical_json_file_bytes(
            build_stratified_pdf_gold_packets(
                physical_receipt,
                prepared_as_of=date(2026, 8, 14),
                error_labels_path=labels,
            )[0].model_dump(mode="json", warnings=False)
        )
    )
    gold_output = workspace / "_local" / "data" / "document-preview" / "gold"
    gold_summary = seed_synthetic_pdf_gold(
        receipt_path=receipt_path,
        packet_path=packet_path,
        source_directory=source,
        clean_manifest_path=clean,
        error_labels_path=labels,
        output_directory=gold_output,
    )
    assert gold_summary["status"] == "draft_gold_created"
    assert len(list(gold_output.glob("*.json"))) == 1
