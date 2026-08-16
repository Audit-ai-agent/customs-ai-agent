"""Write family-safe, source-PDF training labels without printing document text.

This is an offline training-preparation command only.  It does not provision a
model, run a candidate, select an engine, or touch an activation pointer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from customs_ai_agent.evaluation.document_gold import DatasetReceiptV1
from customs_ai_agent.evaluation.document_ocr_quality import EvaluationPacketV2
from customs_ai_agent.evaluation.source_attested_synthetic_pdf_truth import (
    write_source_attested_synthetic_dataset_receipt,
    write_source_attested_synthetic_packet,
)
from customs_ai_agent.evaluation.supervised_template_training import (
    SupervisedTemplateTrainingError,
    build_source_attested_supervised_template_training_labels,
    build_source_attested_supervised_template_training_split,
    write_source_attested_supervised_template_training_labels,
    write_source_attested_supervised_template_training_split,
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


def _output_directory(*, root: Path, output_directory: Path | None, name: str) -> Path:
    candidate = output_directory or root / name
    try:
        resolved = candidate.resolve()
        resolved.relative_to(root)
    except ValueError as error:
        raise ValueError("supervised training output directory is invalid") from error
    return resolved


def compile_source_attested_supervised_template_training_labels(
    *,
    receipt_path: Path,
    development_packet_path: Path,
    sealed_packet_path: Path,
    source_directory: Path,
    document_preview_root: Path,
    split_output_directory: Path | None = None,
    labels_output_directory: Path | None = None,
) -> dict[str, object]:
    """Build the split and labels from immutable local inputs only."""

    root = _preview_root(document_preview_root)
    try:
        receipt = DatasetReceiptV1.model_validate_json(receipt_path.read_bytes())
        development_packet = EvaluationPacketV2.model_validate_json(
            development_packet_path.read_bytes()
        )
        sealed_packet = EvaluationPacketV2.model_validate_json(sealed_packet_path.read_bytes())
    except (OSError, ValueError) as error:
        raise SupervisedTemplateTrainingError("supervised_training_input_invalid") from error
    # Keep the original immutable inputs in their normal local receipt stores;
    # neither the split nor labels become a runtime candidate or approval.
    write_source_attested_synthetic_dataset_receipt(
        receipt=receipt, output_directory=root / "source-attested-dataset-receipts"
    )
    write_source_attested_synthetic_packet(
        packet=development_packet, output_directory=root / "source-attested-packets"
    )
    write_source_attested_synthetic_packet(
        packet=sealed_packet, output_directory=root / "source-attested-packets"
    )
    split = build_source_attested_supervised_template_training_split(
        receipt=receipt,
        development_packet=development_packet,
        sealed_packet=sealed_packet,
    )
    labels = build_source_attested_supervised_template_training_labels(
        receipt=receipt,
        development_packet=development_packet,
        sealed_packet=sealed_packet,
        training_split=split,
        source_directory=source_directory,
    )
    split_destination = write_source_attested_supervised_template_training_split(
        split=split,
        output_directory=_output_directory(
            root=root,
            output_directory=split_output_directory,
            name="supervised-template-training-splits",
        ),
    )
    labels_destination = write_source_attested_supervised_template_training_labels(
        labels=labels,
        output_directory=_output_directory(
            root=root,
            output_directory=labels_output_directory,
            name="supervised-template-training-labels",
        ),
    )
    return {
        "status": "supervised_template_training_labels_recorded",
        "training_split_receipt_sha256": split.training_split_receipt_sha256,
        "training_labels_receipt_sha256": labels.training_labels_receipt_sha256,
        "training_asset_count": split.training_asset_count,
        "held_out_family_count": split.held_out_family_count,
        "field_label_count": labels.field_label_count,
        "split_file": split_destination.name,
        "labels_file": labels_destination.name,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compile local-only family-safe supervised-template labels without printing PDF text."
        )
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--development-packet", type=Path, required=True)
    parser.add_argument("--sealed-packet", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--document-preview-root", type=Path, required=True)
    parser.add_argument("--split-output-directory", type=Path)
    parser.add_argument("--labels-output-directory", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        summary = compile_source_attested_supervised_template_training_labels(
            receipt_path=arguments.receipt,
            development_packet_path=arguments.development_packet,
            sealed_packet_path=arguments.sealed_packet,
            source_directory=arguments.source_directory,
            document_preview_root=arguments.document_preview_root,
            split_output_directory=arguments.split_output_directory,
            labels_output_directory=arguments.labels_output_directory,
        )
    except (SupervisedTemplateTrainingError, OSError, ValueError, json.JSONDecodeError):
        print(
            json.dumps(
                {"status": "failed", "code": "supervised_template_training_labels_failed"},
                separators=(",", ":"),
            )
        )
        return 2
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
