"""Compile one local source-attested synthetic-PDF layout truth receipt.

This tool never reads an OCR candidate and never copies PDF text to stdout.
It only emits a compact status, hashes, and counts after writing a hash-named
receipt below the local document-preview root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from customs_ai_agent.evaluation.document_gold import DatasetReceiptV1
from customs_ai_agent.evaluation.document_ocr_quality import EvaluationPacketV2
from customs_ai_agent.evaluation.source_attested_synthetic_pdf_truth import (
    SourceAttestedSyntheticPdfTruthError,
    compile_source_attested_synthetic_pdf_layout_truth,
    write_source_attested_synthetic_dataset_receipt,
    write_source_attested_synthetic_layout_truth,
    write_source_attested_synthetic_packet,
)


def _preview_root(path: Path) -> Path:
    try:
        resolved = path.resolve(strict=True)
        if (
            not resolved.is_dir()
            or resolved.is_symlink()
            or tuple(part.casefold() for part in resolved.parts[-3:])
            != ("_local", "data", "document-preview")
        ):
            raise ValueError("document preview root is invalid")
    except OSError as error:
        raise ValueError("document preview root is invalid") from error
    return resolved


def _output_directory(*, root: Path, output_directory: Path | None) -> Path:
    candidate = output_directory or root / "source-layout-truths"
    try:
        resolved = candidate.resolve()
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("source layout truth output directory is invalid") from error
    return resolved


def compile_source_attested_synthetic_pdf_layout(
    *,
    receipt_path: Path,
    packet_path: Path,
    source_directory: Path,
    clean_manifest_path: Path,
    error_labels_path: Path,
    document_preview_root: Path,
    output_directory: Path | None = None,
) -> dict[str, object]:
    root = _preview_root(document_preview_root)
    try:
        receipt = DatasetReceiptV1.model_validate_json(receipt_path.read_bytes())
        packet = EvaluationPacketV2.model_validate_json(packet_path.read_bytes())
    except (OSError, ValueError) as error:
        raise SourceAttestedSyntheticPdfTruthError("source_layout_input_invalid") from error
    write_source_attested_synthetic_dataset_receipt(
        receipt=receipt,
        output_directory=root / "source-attested-dataset-receipts",
    )
    write_source_attested_synthetic_packet(
        packet=packet,
        output_directory=root / "source-attested-packets",
    )
    truth = compile_source_attested_synthetic_pdf_layout_truth(
        receipt=receipt,
        packet=packet,
        source_directory=source_directory,
        clean_manifest_path=clean_manifest_path,
        error_labels_path=error_labels_path,
    )
    destination = write_source_attested_synthetic_layout_truth(
        truth=truth,
        output_directory=_output_directory(root=root, output_directory=output_directory),
    )
    return {
        "status": "source_attested_layout_truth_recorded",
        "packet_kind": packet.split,
        "layout_truth_receipt_sha256": truth.layout_truth_receipt_sha256,
        "record_count": len(truth.records),
        "source_manifest_mismatch_count": truth.source_manifest_mismatch_count,
        "result_file": destination.name,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile local source-attested synthetic-PDF layout truth without printing PDF or text."
        )
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--clean-manifest", type=Path, required=True)
    parser.add_argument("--error-labels", type=Path, required=True)
    parser.add_argument("--document-preview-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        summary = compile_source_attested_synthetic_pdf_layout(
            receipt_path=arguments.receipt,
            packet_path=arguments.packet,
            source_directory=arguments.source_directory,
            clean_manifest_path=arguments.clean_manifest,
            error_labels_path=arguments.error_labels,
            document_preview_root=arguments.document_preview_root,
            output_directory=arguments.output_directory,
        )
    except (SourceAttestedSyntheticPdfTruthError, OSError, ValueError, json.JSONDecodeError):
        print(
            json.dumps(
                {"status": "failed", "code": "source_attested_layout_truth_failed"},
                separators=(",", ":"),
            )
        )
        return 2
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
