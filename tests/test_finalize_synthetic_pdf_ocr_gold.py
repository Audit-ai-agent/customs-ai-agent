from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from customs_ai_agent.evaluation.document_gold import (
    DatasetAssetReceiptV1,
    DatasetReceiptV1,
    receipt_sha256,
)
from customs_ai_agent.evaluation.document_ocr_quality import (
    EvaluationPacketV2,
    GoldFieldV2,
    GoldPacketV2,
    SourceGroundTruthAttestationV1,
    StratifiedPacketItemV2,
    build_draft_gold_record,
    build_gold_packet,
    canonical_json_file_bytes,
    ground_truth_receipt_sha256,
    packet_receipt_sha256,
)
from tools.finalize_synthetic_pdf_ocr_gold import (
    GoldFinalizationError,
    finalize_synthetic_pdf_gold,
    main,
)
from tools.review_document_gold import (
    LocalGoldReviewDraftV1,
    LocalReviewFieldV1,
    LocalReviewSpanV1,
    LocalReviewTableCellV1,
)


def _sha256(value: bytes | str) -> str:
    raw = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_file_bytes(value))


def _fixture(tmp_path: Path) -> dict[str, object]:
    data = tmp_path / "workspace" / "_local" / "data"
    source_directory = data / "synthetic-pdf"
    source_path = source_directory / "pdf" / "invoice.pdf"
    source_path.parent.mkdir(parents=True)
    source_bytes = b"%PDF-1.7\n1 0 obj<</Type/Page>>endobj\n%%EOF\n"
    source_path.write_bytes(source_bytes)
    source_sha256 = _sha256(source_bytes)
    receipt = DatasetReceiptV1(
        dataset_id="synthetic-pdf",
        dataset_revision="human-source-v1",
        received_as_of=date(2026, 8, 14),
        assets=[
            DatasetAssetReceiptV1(
                asset_id="asset-001",
                relative_path="pdf/invoice.pdf",
                media_type="application/pdf",
                byte_size=len(source_bytes),
                sha256=source_sha256,
                source_role="preview_candidate",
            )
        ],
    )
    item = StratifiedPacketItemV2(
        asset_id="asset-001",
        source_sha256=source_sha256,
        fixture_kind="clean",
        document_type="invoice",
        set_id="C0001",
    )
    packet_payload: dict[str, object] = {
        "schema_version": "2.0",
        "packet_kind": "stratified-v2",
        "split": "development",
        "dataset_id": receipt.dataset_id,
        "dataset_revision": receipt.dataset_revision,
        "receipt_sha256": receipt_sha256(receipt),
        "error_labels_sha256": _sha256("error-labels"),
        "prepared_as_of": "2026-08-14",
        "items": [item.model_dump(mode="json")],
        "required_gates": ["field_exact_match", "ocr_cer", "table_cell_f1"],
        "release_thresholds": {
            "field_exact_match_min": 0.99,
            "ocr_cer_max": 0.01,
            "table_cell_f1_min": 0.98,
        },
    }
    packet_payload["packet_receipt_sha256"] = packet_receipt_sha256(packet_payload)
    packet = EvaluationPacketV2.model_validate(packet_payload)
    attestation_payload: dict[str, object] = {
        "source_kind": "synthetic_pdf_manifest_v1",
        "clean_manifest_sha256": _sha256("clean-manifest"),
        "error_labels_sha256": _sha256("error-labels"),
        "source_set_id": "C0001",
        "error_label_sha256s": [],
    }
    attestation_payload["ground_truth_receipt_sha256"] = ground_truth_receipt_sha256(
        attestation_payload
    )
    source_attestation = SourceGroundTruthAttestationV1.model_validate(attestation_payload)
    seeded_record = build_draft_gold_record(
        packet=packet,
        item=item,
        fields=[GoldFieldV2(field_name="inv_no", value="INV-001", normalizer="identifier-v1")],
        source_attestation=source_attestation,
    )
    seeded_gold = build_gold_packet(packet=packet, records=[seeded_record])
    receipt_path = data / "receipts" / "dataset-receipt.json"
    packet_path = data / "packets" / "development.json"
    seeded_gold_path = data / "gold" / "seeded-development.json"
    _write_json(receipt_path, receipt.model_dump(mode="json"))
    _write_json(packet_path, packet.model_dump(mode="json"))
    _write_json(seeded_gold_path, seeded_gold.model_dump(mode="json"))
    review_directory = data / "gold-review" / "drafts"
    _write_handoff(
        directory=review_directory,
        receipt=receipt,
        packet=packet,
        source_sha256=source_sha256,
        approval_id="approval-001",
        reviewer_id="reviewer-001",
    )
    return {
        "receipt": receipt,
        "packet": packet,
        "receipt_path": receipt_path,
        "packet_path": packet_path,
        "seeded_gold_path": seeded_gold_path,
        "source_directory": source_directory,
        "source_path": source_path,
        "review_directory": review_directory,
        "output_directory": data / "final-gold",
    }


def _write_handoff(
    *,
    directory: Path,
    receipt: DatasetReceiptV1,
    packet: EvaluationPacketV2,
    source_sha256: str,
    approval_id: str | None,
    reviewer_id: str | None,
) -> Path:
    handoff = LocalGoldReviewDraftV1(
        review_origin="loopback_review_v1",
        dataset_id=receipt.dataset_id,
        dataset_revision=receipt.dataset_revision,
        receipt_sha256=receipt_sha256(receipt),
        packet_receipt_sha256=packet.packet_receipt_sha256,
        packet_kind="stratified-v2",
        packet_split=packet.split,
        asset_id="asset-001",
        source_sha256=source_sha256,
        reviewer_id=reviewer_id,
        approval_id=approval_id,
        review_status="reviewer_submitted",
        fields=[LocalReviewFieldV1(field_name="inv_no", value="INV-001", page_number=1)],
        spans=[LocalReviewSpanV1(page_number=1, text="INVOICE INV-001", reading_order=1)],
        table_cells=[
            LocalReviewTableCellV1(
                page_number=1,
                table_index=1,
                row_index=0,
                column_index=0,
                text="Widget",
            )
        ],
    )
    path = directory / "asset-001.json"
    _write_json(path, handoff.model_dump(mode="json"))
    return path


def _finalize(fixture: dict[str, object]) -> dict[str, object]:
    return finalize_synthetic_pdf_gold(
        receipt_path=fixture["receipt_path"],  # type: ignore[arg-type]
        packet_path=fixture["packet_path"],  # type: ignore[arg-type]
        seeded_gold_path=fixture["seeded_gold_path"],  # type: ignore[arg-type]
        source_directory=fixture["source_directory"],  # type: ignore[arg-type]
        review_draft_directory=fixture["review_directory"],  # type: ignore[arg-type]
        output_directory=fixture["output_directory"],  # type: ignore[arg-type]
        approval_id="approval-001",
        reviewer_id="reviewer-001",
        confirm_human_adjudication=True,
    )


def test_finalizer_creates_complete_human_adjudicated_gold_without_plain_reviewer_id(
    tmp_path: Path,
) -> None:
    fixture = _fixture(tmp_path)

    summary = _finalize(fixture)

    output = fixture["output_directory"] / summary["gold_file"]  # type: ignore[operator]
    final_gold = GoldPacketV2.model_validate_json(output.read_bytes())
    assert summary["status"] == "human_adjudicated_gold_created"
    assert final_gold.status == "human_adjudicated"
    assert final_gold.records[0].reviewer_approval is not None
    assert final_gold.records[0].reviewer_approval.reviewer_id_sha256 == _sha256("reviewer-001")
    assert b"reviewer-001" not in output.read_bytes()
    assert final_gold.records[0].fields[0].normalizer == "identifier-v1"
    assert final_gold.records[0].spans[0].text == "INVOICE INV-001"


def test_finalizer_rejects_missing_handoff_approval_before_gold_is_created(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    handoff = json.loads(
        (fixture["review_directory"] / "asset-001.json").read_text(encoding="utf-8")
    )  # type: ignore[operator]
    handoff["approval_id"] = None
    _write_json(fixture["review_directory"] / "asset-001.json", handoff)  # type: ignore[operator]

    with pytest.raises(GoldFinalizationError, match="review_handoff_invalid"):
        _finalize(fixture)


def test_finalizer_rejects_missing_selected_handoff(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    (fixture["review_directory"] / "asset-001.json").unlink()  # type: ignore[operator]

    with pytest.raises(GoldFinalizationError, match="review_handoff_coverage_incomplete"):
        _finalize(fixture)


def test_finalizer_rejects_source_receipt_drift_before_using_handoff(tmp_path: Path) -> None:
    fixture = _fixture(tmp_path)
    fixture["source_path"].write_bytes(b"%PDF-1.7\nchanged\n%%EOF\n")  # type: ignore[union-attr]

    with pytest.raises(GoldFinalizationError, match="source_receipt_invalid"):
        _finalize(fixture)


def test_finalizer_cli_requires_explicit_human_adjudication_confirmation(
    capsys: pytest.CaptureFixture[str],
) -> None:
    result = main(
        [
            "--receipt",
            "ignored.json",
            "--packet",
            "ignored.json",
            "--seeded-gold",
            "ignored.json",
            "--source-directory",
            "ignored",
            "--review-draft-directory",
            "ignored",
            "--output-directory",
            "ignored",
            "--approval-id",
            "approval-001",
            "--reviewer-id",
            "reviewer-001",
        ]
    )

    assert result == 2
    assert json.loads(capsys.readouterr().out) == {
        "status": "failed",
        "code": "gold_finalization_failed",
    }
