from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from customs_ai_agent.evaluation.document_gold import (
    DatasetAssetReceiptV1,
    DatasetReceiptV1,
    EvaluationPacketItemV1,
    EvaluationPacketV1,
    receipt_sha256,
)
from customs_ai_agent.evaluation.document_ocr_quality import (
    EvaluationPacketV2,
    GoldFieldV2,
    GoldSpanV2,
    GoldTableCellV2,
    SourceGroundTruthAttestationV1,
    approve_gold_record,
    build_draft_gold_record,
    build_gold_packet,
    canonical_json_file_bytes,
    ground_truth_receipt_sha256,
)
from tools.review_document_gold import (
    ReviewContext,
    ReviewToolError,
    _packet_receipt_sha256,
    _validate_loopback_host,
    create_review_app,
)


def _pdf() -> bytes:
    return b"%PDF-1.7\n1 0 obj<</Type /Page>>endobj\n%%EOF\n"


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _v2_packet(receipt: DatasetReceiptV1, source_sha256: str) -> dict[str, object]:
    packet: dict[str, object] = {
        "schema_version": "2.0",
        "packet_kind": "stratified-v2",
        "split": "development",
        "dataset_id": receipt.dataset_id,
        "dataset_revision": receipt.dataset_revision,
        "receipt_sha256": receipt_sha256(receipt),
        "error_labels_sha256": "e" * 64,
        "packet_receipt_sha256": "0" * 64,
        "prepared_as_of": "2026-08-14",
        "items": [
            {
                "asset_id": "asset-001",
                "source_sha256": source_sha256,
                "media_type": "application/pdf",
                "fixture_kind": "clean",
                "document_type": "invoice",
                "set_id": "set-001",
            }
        ],
        "required_gates": ["field_exact_match", "ocr_cer", "table_cell_f1"],
        "release_thresholds": {
            "field_exact_match_min": 0.99,
            "ocr_cer_max": 0.01,
            "table_cell_f1_min": 0.98,
        },
    }
    packet["packet_receipt_sha256"] = _packet_receipt_sha256(packet)
    return packet


def _context(tmp_path: Path) -> tuple[ReviewContext, Path, Path, Path]:
    data = tmp_path / "workspace" / "_local" / "data"
    dataset_root = data / "synthetic-pdf"
    source = dataset_root / "pdf" / "invoice.pdf"
    source.parent.mkdir(parents=True)
    content = _pdf()
    source.write_bytes(content)
    source_sha = hashlib.sha256(content).hexdigest()
    receipt = DatasetReceiptV1(
        dataset_id="synthetic-pdf",
        dataset_revision="human-checked-v1",
        received_as_of="2026-08-14",
        assets=[
            DatasetAssetReceiptV1(
                asset_id="asset-001",
                relative_path="pdf/invoice.pdf",
                media_type="application/pdf",
                byte_size=len(content),
                sha256=source_sha,
                source_role="preview_candidate",
            )
        ],
    )
    receipt_path = data / "receipts" / "synthetic-pdf" / "dataset-receipt.json"
    packet_path = data / "evaluation" / "synthetic-pdf" / "development-stratified-v2.json"
    _write_json(receipt_path, receipt.model_dump(mode="json"))
    _write_json(packet_path, _v2_packet(receipt, source_sha))
    output = data / "gold-review" / "synthetic-pdf"
    return (
        ReviewContext.from_paths(
            receipt_path=receipt_path,
            packet_path=packet_path,
            dataset_root=dataset_root,
            output_directory=output,
        ),
        source,
        packet_path,
        output,
    )


def _submission(source_sha256: str) -> dict[str, object]:
    return {
        "source_sha256": source_sha256,
        "reviewer_id": "reviewer-001",
        "approval_id": "approval-001",
        "review_status": "reviewer_submitted",
        "fields": [{"field_name": "invoice_number", "value": "INV-001", "page_number": 1}],
        "spans": [
            {
                "page_number": 1,
                "text": "INVOICE NO INV-001",
                "reading_order": 1,
                "bbox": {"x": 0.1, "y": 0.1, "width": 0.2, "height": 0.1},
            }
        ],
        "table_cells": [
            {
                "page_number": 1,
                "table_index": 1,
                "row_index": 0,
                "column_index": 0,
                "text": "Widget",
            }
        ],
    }


def _seeded_gold_draft(context: ReviewContext, packet_path: Path, output: Path) -> Path:
    """Create a minimal source-attested v2 draft solely for loopback UI tests."""

    packet = EvaluationPacketV2.model_validate_json(packet_path.read_bytes())
    item = packet.items[0]
    attestation_payload: dict[str, object] = {
        "source_kind": "synthetic_pdf_manifest_v1",
        "clean_manifest_sha256": "c" * 64,
        "error_labels_sha256": "e" * 64,
        "source_set_id": item.set_id,
        "error_label_sha256s": [],
    }
    attestation_payload["ground_truth_receipt_sha256"] = ground_truth_receipt_sha256(
        attestation_payload
    )
    record = build_draft_gold_record(
        packet=packet,
        item=item,
        fields=[GoldFieldV2(field_name="inv_no", value="INV-001", normalizer="identifier-v1")],
        source_attestation=SourceGroundTruthAttestationV1.model_validate(attestation_payload),
    )
    gold = build_gold_packet(packet=packet, records=[record])
    path = output.parent / "seeded-gold.json"
    path.write_bytes(canonical_json_file_bytes(gold.model_dump(mode="json")))
    return path


def test_loopback_review_serves_only_bound_pdf_and_saves_local_intermediate_draft(
    tmp_path: Path,
) -> None:
    context, source, _, output = _context(tmp_path)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    app = create_review_app(context)

    with TestClient(app, raise_server_exceptions=False) as client:
        index = client.get("/")
        items = client.get("/api/items")
        pdf = client.get("/api/items/asset-001/pdf")
        saved = client.post("/api/drafts/asset-001", json=_submission(source_sha))
        handoff = client.get("/api/drafts/asset-001/handoff")

    assert index.status_code == 200
    assert "Gold 승인·OCR 활성화는 수행하지 않습니다" in index.text
    assert items.json() == {
        "status": "ready",
        "packet_kind": "stratified-v2",
        "split": "development",
        "items": [{"asset_id": "asset-001", "source_sha256": source_sha}],
    }
    assert pdf.status_code == 200
    assert pdf.headers["content-type"] == "application/pdf"
    assert pdf.content == _pdf()
    assert saved.json() == {
        "status": "saved",
        "asset_id": "asset-001",
        "source_sha256": source_sha,
        "review_status": "reviewer_submitted",
    }
    stored_path = output / "drafts" / "asset-001.json"
    assert stored_path.is_file()
    stored = json.loads(stored_path.read_text(encoding="utf-8"))
    assert stored["schema_version"] == "local-gold-review-draft-v1"
    assert stored["source_sha256"] == source_sha
    assert stored["fields"][0]["value"] == "INV-001"
    exported = handoff.json()
    assert exported["handoff_status"] == "reviewer_submitted"
    assert exported["approval_id"] == "approval-001"
    assert exported["packet_receipt_sha256"] == context.packet.packet_receipt_sha256
    assert "adjudication_status" not in exported


def test_review_prefills_exact_source_attested_fields_without_approving_gold(
    tmp_path: Path,
) -> None:
    context, source, packet_path, output = _context(tmp_path)
    gold_path = _seeded_gold_draft(context, packet_path, output)
    prefilled_context = ReviewContext.from_paths(
        receipt_path=context.receipt_path,
        packet_path=packet_path,
        dataset_root=context.dataset_root,
        output_directory=output,
        gold_draft_path=gold_path,
    )
    app = create_review_app(prefilled_context)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/drafts/asset-001")

    assert response.status_code == 200
    draft = response.json()["draft"]
    assert draft["review_status"] == "draft"
    assert draft["review_origin"] == "loopback_review_v1"
    assert draft["fields"] == [{"field_name": "inv_no", "value": "INV-001", "page_number": None}]
    assert draft["spans"] == []
    assert not (output / "drafts" / "asset-001.json").exists()


def test_review_rejects_an_approved_or_mixed_gold_packet_as_prefill(tmp_path: Path) -> None:
    context, _, packet_path, output = _context(tmp_path)
    draft_path = _seeded_gold_draft(context, packet_path, output)
    packet = EvaluationPacketV2.model_validate_json(packet_path.read_bytes())
    draft_gold = build_gold_packet(
        packet=packet,
        records=[
            approve_gold_record(
                build_draft_gold_record(
                    packet=packet,
                    item=packet.items[0],
                    fields=[
                        GoldFieldV2(
                            field_name="inv_no", value="INV-001", normalizer="identifier-v1"
                        )
                    ],
                    spans=[GoldSpanV2(page_number=1, text="INV-001", reading_order=1)],
                    table_cells=[
                        GoldTableCellV2(
                            page_number=1,
                            table_index=1,
                            row_index=0,
                            column_index=0,
                            text="Widget",
                        )
                    ],
                    source_attestation=SourceGroundTruthAttestationV1.model_validate(
                        {
                            "source_kind": "synthetic_pdf_manifest_v1",
                            "clean_manifest_sha256": "c" * 64,
                            "error_labels_sha256": "e" * 64,
                            "source_set_id": "set-001",
                            "error_label_sha256s": [],
                            "ground_truth_receipt_sha256": ground_truth_receipt_sha256(
                                {
                                    "source_kind": "synthetic_pdf_manifest_v1",
                                    "clean_manifest_sha256": "c" * 64,
                                    "error_labels_sha256": "e" * 64,
                                    "source_set_id": "set-001",
                                    "error_label_sha256s": [],
                                }
                            ),
                        }
                    ),
                ),
                approval_id="approval-001",
                reviewer_id_sha256=hashlib.sha256(b"reviewer-001").hexdigest(),
                approved_at=datetime(2026, 8, 14, tzinfo=timezone.utc),
            )
        ],
    )
    draft_path.write_bytes(canonical_json_file_bytes(draft_gold.model_dump(mode="json")))

    with pytest.raises(ReviewToolError, match="gold_draft_binding_invalid"):
        ReviewContext.from_paths(
            receipt_path=context.receipt_path,
            packet_path=packet_path,
            dataset_root=context.dataset_root,
            output_directory=output,
            gold_draft_path=draft_path,
        )


def test_review_refuses_source_change_and_never_exposes_source_text_in_error(
    tmp_path: Path,
) -> None:
    context, source, _, _ = _context(tmp_path)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    source.write_bytes(b"%PDF-1.7\nchanged\n%%EOF\n")
    app = create_review_app(context)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/drafts/asset-001", json=_submission(source_sha))

    assert response.status_code == 409
    assert response.json() == {"status": "failed", "code": "source_sha_mismatch"}
    assert "changed" not in response.text


def test_review_fails_closed_when_receipt_or_packet_changes_after_start(tmp_path: Path) -> None:
    context, _, packet_path, _ = _context(tmp_path)
    packet_path.write_text("{}", encoding="utf-8")
    app = create_review_app(context)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.get("/api/items")

    assert response.status_code == 409
    assert response.json() == {"status": "failed", "code": "control_file_changed"}


def test_review_requires_loopback_and_local_data_boundary(tmp_path: Path) -> None:
    context, _, _, _ = _context(tmp_path)
    assert _validate_loopback_host("127.0.0.1") == "127.0.0.1"
    assert _validate_loopback_host("::1") == "::1"
    with pytest.raises(ReviewToolError, match="loopback_host_required"):
        _validate_loopback_host("0.0.0.0")
    with pytest.raises(ReviewToolError, match="loopback_host_required"):
        _validate_loopback_host("localhost")
    with pytest.raises(ReviewToolError, match="local_data_boundary_required"):
        ReviewContext.from_paths(
            receipt_path=context.receipt_path,
            packet_path=context.packet_path,
            dataset_root=tmp_path / "outside",
            output_directory=context.output_directory,
        )


def test_reviewer_submission_cannot_claim_final_gold_status(tmp_path: Path) -> None:
    context, source, _, _ = _context(tmp_path)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    payload = _submission(source_sha)
    payload["adjudication_status"] = "human_adjudicated"
    app = create_review_app(context)

    with TestClient(app, raise_server_exceptions=False) as client:
        response = client.post("/api/drafts/asset-001", json=payload)

    assert response.status_code == 409
    assert response.json() == {"status": "failed", "code": "submission_invalid"}


def test_packet_receipt_or_source_binding_mismatch_refuses_start(tmp_path: Path) -> None:
    context, source, packet_path, output = _context(tmp_path)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet["items"][0]["source_sha256"] = "1" * 64
    packet["packet_receipt_sha256"] = _packet_receipt_sha256(packet)
    _write_json(packet_path, packet)

    with pytest.raises(ReviewToolError, match="packet_asset_mismatch"):
        ReviewContext.from_paths(
            receipt_path=context.receipt_path,
            packet_path=packet_path,
            dataset_root=source.parent.parent,
            output_directory=output,
        )


def test_v2_packet_without_error_label_receipt_is_rejected(tmp_path: Path) -> None:
    context, source, packet_path, output = _context(tmp_path)
    packet = json.loads(packet_path.read_text(encoding="utf-8"))
    packet.pop("error_labels_sha256")
    packet["packet_receipt_sha256"] = _packet_receipt_sha256(packet)
    _write_json(packet_path, packet)

    with pytest.raises(ReviewToolError, match="packet_invalid"):
        ReviewContext.from_paths(
            receipt_path=context.receipt_path,
            packet_path=packet_path,
            dataset_root=source.parent.parent,
            output_directory=output,
        )


def test_legacy_packet_is_read_only_when_its_original_schema_still_valid(tmp_path: Path) -> None:
    context, source, packet_path, output = _context(tmp_path)
    source_sha = hashlib.sha256(source.read_bytes()).hexdigest()
    legacy = EvaluationPacketV1(
        split="development",
        dataset_id=context.receipt.dataset_id,
        dataset_revision=context.receipt.dataset_revision,
        receipt_sha256=receipt_sha256(context.receipt),
        prepared_as_of="2026-08-14",
        items=[
            EvaluationPacketItemV1(
                asset_id="asset-001",
                source_sha256=source_sha,
                media_type="application/pdf",
            )
        ],
        required_gates=["field_exact_match", "ocr_cer", "table_cell_f1"],
    )
    _write_json(packet_path, legacy.model_dump(mode="json"))

    reloaded = ReviewContext.from_paths(
        receipt_path=context.receipt_path,
        packet_path=packet_path,
        dataset_root=source.parent.parent,
        output_directory=output,
    )

    assert reloaded.packet.packet_kind == "legacy-v1"
    assert reloaded.packet.items[0].asset_id == "asset-001"
