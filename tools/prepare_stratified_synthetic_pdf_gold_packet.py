"""Prepare immutable stratified-v2 synthetic PDF OCR Gold packets locally."""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Sequence

from customs_ai_agent.evaluation.document_gold import DatasetReceiptV1
from customs_ai_agent.evaluation.document_ocr_quality import (
    build_stratified_pdf_gold_packets,
    canonical_json_file_bytes,
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
            raise ValueError("an immutable stratified packet already exists with different bytes")
        return
    path.write_bytes(encoded)


def prepare_stratified_packets(
    *,
    receipt_path: Path,
    error_labels_path: Path,
    output_directory: Path,
    prepared_as_of: date,
) -> dict[str, object]:
    receipt = DatasetReceiptV1.model_validate_json(receipt_path.read_bytes())
    development, sealed = build_stratified_pdf_gold_packets(
        receipt,
        prepared_as_of=prepared_as_of,
        error_labels_path=error_labels_path,
    )
    output = _local_data_directory(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    _write_immutable(
        output / "synthetic-pdf-development-stratified-v2.json",
        development.model_dump(mode="json", warnings=False),
    )
    _write_immutable(
        output / "synthetic-pdf-sealed-stratified-v2.json",
        sealed.model_dump(mode="json", warnings=False),
    )
    return {
        "status": "awaiting_human_adjudication",
        "packet_kind": "stratified-v2",
        "dataset_id": receipt.dataset_id,
        "dataset_revision": receipt.dataset_revision,
        "development_count": len(development.items),
        "sealed_count": len(sealed.items),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare local-only stratified synthetic PDF OCR Gold packets."
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--error-labels", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--as-of", required=True, help="Packet preparation date in YYYY-MM-DD format."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        prepared_as_of = date.fromisoformat(arguments.as_of)
        if prepared_as_of.isoformat() != arguments.as_of:
            raise ValueError("--as-of must be an ISO calendar date")
        summary = prepare_stratified_packets(
            receipt_path=arguments.receipt,
            error_labels_path=arguments.error_labels,
            output_directory=arguments.output_directory,
            prepared_as_of=prepared_as_of,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        print(
            json.dumps(
                {"status": "failed", "code": "packet_preparation_failed"}, separators=(",", ":")
            )
        )
        return 2
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
