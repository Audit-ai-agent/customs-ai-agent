from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

import pytest

from customs_ai_agent.evaluation.document_gold import (
    DatasetAssetReceiptV1,
    DatasetReceiptV1,
    build_draft_gold_packets,
    receipt_sha256,
)
from tools.prepare_document_preview_gold_packet import prepare_packets
from tools.create_document_dataset_receipt import build_receipt, write_immutable_receipt


def _receipt(count: int = 80) -> DatasetReceiptV1:
    return DatasetReceiptV1(
        dataset_id="synthetic-documents",
        dataset_revision="source-v1",
        received_as_of="2026-08-11",
        assets=[
            DatasetAssetReceiptV1(
                asset_id=f"asset-{index:03}",
                relative_path=f"pdf/{index:03}.pdf",
                media_type="application/pdf",
                byte_size=1_000 + index,
                sha256=hashlib.sha256(f"asset-{index}".encode()).hexdigest(),
                triplet_id=f"triplet-{index // 3:03}",
                source_role="preview_candidate",
            )
            for index in range(count)
        ],
    )


def test_draft_gold_packets_are_immutable_non_overlapping_60_plus_20() -> None:
    receipt = _receipt()

    development, sealed = build_draft_gold_packets(receipt, prepared_as_of=date(2026, 8, 11))

    assert development.status == "awaiting_human_adjudication"
    assert sealed.status == "awaiting_human_adjudication"
    assert len(development.items) == 60
    assert len(sealed.items) == 20
    assert {item.asset_id for item in development.items}.isdisjoint(
        item.asset_id for item in sealed.items
    )
    assert development.receipt_sha256 == receipt_sha256(receipt) == sealed.receipt_sha256
    assert development.required_gates == ["field_exact_match", "ocr_cer", "table_cell_f1"]
    assert development.release_thresholds.model_dump() == {
        "field_exact_match_min": 0.99,
        "ocr_cer_max": 0.01,
        "table_cell_f1_min": 0.98,
    }
    # Re-running from exactly the same receipt produces the same sealed split.
    again = build_draft_gold_packets(receipt, prepared_as_of=date(2026, 8, 11))
    assert [item.asset_id for item in again[1].items] == [item.asset_id for item in sealed.items]


def test_draft_gold_packet_refuses_underfilled_or_changed_sample_size() -> None:
    with pytest.raises(ValueError, match="enough immutable assets"):
        build_draft_gold_packets(_receipt(79), prepared_as_of=date(2026, 8, 11))
    with pytest.raises(ValueError, match="exactly 60 development plus 20 sealed"):
        build_draft_gold_packets(_receipt(), prepared_as_of=date(2026, 8, 11), development_count=59)


def test_packet_tool_writes_only_local_draft_metadata(tmp_path: Path) -> None:
    root = tmp_path / "workspace"
    receipt_path = root / "receipt.json"
    output = root / "_local" / "data" / "evaluation" / "source-v1"
    receipt_path.parent.mkdir(parents=True)
    receipt_path.write_text(
        json.dumps(_receipt().model_dump(mode="json"), ensure_ascii=False), encoding="utf-8"
    )

    summary = prepare_packets(
        receipt_path=receipt_path,
        output_directory=output,
        prepared_as_of=date(2026, 8, 11),
    )

    assert summary == {
        "status": "awaiting_human_adjudication",
        "dataset_id": "synthetic-documents",
        "dataset_revision": "source-v1",
        "development_count": 60,
        "sealed_count": 20,
    }
    development = json.loads((output / "document-preview-development-draft.json").read_text())
    sealed = json.loads((output / "document-preview-sealed-draft.json").read_text())
    assert development["status"] == sealed["status"] == "awaiting_human_adjudication"
    assert "relative_path" not in json.dumps(development)
    assert "relative_path" not in json.dumps(sealed)


def test_receipt_tool_hashes_source_in_place_and_writes_only_local_metadata(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    (source / "invoice.pdf").write_bytes(b"%PDF-1.7\n%%EOF\n")
    (source / "triplets.json").write_text("[]", encoding="utf-8")
    output = tmp_path / "workspace" / "_local" / "data" / "receipts" / "source-v1"

    receipt = build_receipt(
        source_directory=source,
        dataset_id="virtual-source",
        dataset_revision="source-v1",
        received_as_of=date(2026, 8, 11),
    )
    summary = write_immutable_receipt(receipt=receipt, output_directory=output)

    assert summary["status"] == "receipt_created"
    stored = json.loads((output / "dataset-receipt.json").read_text(encoding="utf-8"))
    assert {entry["media_type"] for entry in stored["assets"]} == {
        "application/pdf",
        "application/json",
    }
    assert {(entry["media_type"], entry["source_role"]) for entry in stored["assets"]} == {
        ("application/pdf", "preview_candidate"),
        ("application/json", "structured_reference"),
    }
    # Same receipt is idempotent; a changed receipt cannot replace it.
    assert write_immutable_receipt(receipt=receipt, output_directory=output)["asset_count"] == 2
    with pytest.raises(ValueError, match="immutable receipt already exists"):
        write_immutable_receipt(
            receipt=_receipt(),
            output_directory=output,
        )
