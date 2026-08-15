"""Choose one measured local OCR candidate from development results only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from customs_ai_agent.evaluation.document_ocr_quality import (
    canonical_json_file_bytes,
    read_canonical_evaluation_result,
    select_best_development_result,
)


def _local_data_directory(path: Path) -> Path:
    resolved = path.resolve()
    parts = [part.casefold() for part in resolved.parts]
    if not any(parts[index : index + 2] == ["_local", "data"] for index in range(len(parts) - 1)):
        raise ValueError(
            "selection output directory must be below the local-only _local/data boundary"
        )
    return resolved


def select_synthetic_pdf_ocr_engine(
    *,
    result_paths: Sequence[Path],
    output_directory: Path,
) -> dict[str, object]:
    selected = select_best_development_result(
        [read_canonical_evaluation_result(path) for path in result_paths]
    )
    output = _local_data_directory(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": "1.0",
        "packet_kind": "development",
        "dataset_id": selected.dataset_id,
        "dataset_revision": selected.dataset_revision,
        "packet_receipt_sha256": selected.packet_receipt_sha256,
        "gold_receipt_sha256": selected.gold_receipt_sha256,
        "selected_evaluation_result_sha256": selected.evaluation_result_sha256,
        "engine_kind": selected.engine_kind,
        "extractor_revision": selected.extractor_revision,
        "configuration_sha256": selected.configuration_sha256,
        "model_inventory_sha256": selected.model_inventory_sha256,
    }
    destination = output / f"development-selection-{selected.evaluation_result_sha256}.json"
    encoded = canonical_json_file_bytes(payload)
    if destination.exists() and destination.read_bytes() != encoded:
        raise ValueError("immutable development selection already exists with different bytes")
    if not destination.exists():
        destination.write_bytes(encoded)
    return {
        "status": "development_candidate_selected",
        "selected_evaluation_result_sha256": selected.evaluation_result_sha256,
        "engine_kind": selected.engine_kind,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Select a passing local development OCR candidate without copying OCR content."
    )
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        summary = select_synthetic_pdf_ocr_engine(
            result_paths=arguments.result,
            output_directory=arguments.output_directory,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        print(
            json.dumps(
                {"status": "failed", "code": "development_selection_failed"}, separators=(",", ":")
            )
        )
        return 2
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
