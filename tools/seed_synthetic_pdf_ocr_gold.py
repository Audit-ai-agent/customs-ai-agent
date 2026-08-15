"""Convert verified synthetic CSV ground truth into local OCR Gold drafts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from customs_ai_agent.evaluation.document_gold import DatasetReceiptV1
from customs_ai_agent.evaluation.document_ocr_quality import (
    EvaluationPacketV2,
    build_synthetic_pdf_draft_gold,
    canonical_json_file_bytes,
    verify_receipt_source_assets,
)


def _local_data_directory(path: Path) -> Path:
    resolved = path.resolve()
    parts = [part.casefold() for part in resolved.parts]
    if not any(parts[index : index + 2] == ["_local", "data"] for index in range(len(parts) - 1)):
        raise ValueError("output directory must be below the local-only _local/data boundary")
    return resolved


def _write_immutable(path: Path, payload: object) -> None:
    encoded = canonical_json_file_bytes(payload)
    if path.exists():
        if path.read_bytes() != encoded:
            raise ValueError("an immutable local Gold draft already exists with different bytes")
        return
    path.write_bytes(encoded)


def seed_synthetic_pdf_gold(
    *,
    receipt_path: Path,
    packet_path: Path,
    source_directory: Path,
    clean_manifest_path: Path,
    error_labels_path: Path,
    output_directory: Path,
) -> dict[str, object]:
    receipt = DatasetReceiptV1.model_validate_json(receipt_path.read_bytes())
    packet = EvaluationPacketV2.model_validate_json(packet_path.read_bytes())
    verify_receipt_source_assets(receipt=receipt, source_directory=source_directory)
    gold_packet = build_synthetic_pdf_draft_gold(
        receipt=receipt,
        packet=packet,
        clean_manifest_path=clean_manifest_path,
        error_labels_path=error_labels_path,
    )
    output = _local_data_directory(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    _write_immutable(
        output / f"synthetic-pdf-{packet.split}-gold-{gold_packet.gold_receipt_sha256}.json",
        gold_packet.model_dump(mode="json", warnings=False),
    )
    return {
        "status": "draft_gold_created",
        "packet_kind": packet.packet_kind,
        "packet_split": packet.split,
        "dataset_id": packet.dataset_id,
        "dataset_revision": packet.dataset_revision,
        "record_count": len(gold_packet.records),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Seed local-only synthetic PDF field Gold; it does not approve OCR Gold."
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--error-labels", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        summary = seed_synthetic_pdf_gold(
            receipt_path=arguments.receipt,
            packet_path=arguments.packet,
            source_directory=arguments.source_directory,
            clean_manifest_path=arguments.clean_manifest,
            error_labels_path=arguments.error_labels,
            output_directory=arguments.output_directory,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        print(json.dumps({"status": "failed", "code": "gold_seed_failed"}, separators=(",", ":")))
        return 2
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
