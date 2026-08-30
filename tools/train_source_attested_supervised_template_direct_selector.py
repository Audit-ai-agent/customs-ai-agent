"""Train one local native-PDF supervised-template selector model.

The command reads development/sealed packet metadata only to re-derive the
family-safe split.  It opens PDF bytes solely through the resulting training
asset list, and writes no candidate/runtime configuration or activation
pointer.  It is not an OCR evaluation or a general OCR training command.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from customs_ai_agent.evaluation.supervised_template_direct_selector_trainer import (
    SUPERVISED_TEMPLATE_DIRECT_SELECTOR_MODEL_ID,
    SUPERVISED_TEMPLATE_DIRECT_SELECTOR_MODEL_REVISION,
    SupervisedTemplateDirectSelectorTrainerError,
    read_canonical_source_attested_dataset_receipt_for_selector_training,
    read_canonical_source_attested_packet_for_selector_training,
    read_canonical_source_attested_supervised_template_training_labels_for_selector_training,
    read_canonical_source_attested_supervised_template_training_split_for_selector_training,
    train_source_attested_supervised_template_direct_selector_model,
    write_source_attested_supervised_template_direct_selector_training_result,
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


def train_source_attested_supervised_template_direct_selector(
    *,
    receipt_path: Path,
    development_packet_path: Path,
    sealed_packet_path: Path,
    training_split_path: Path,
    training_labels_path: Path,
    source_directory: Path,
    document_preview_root: Path,
    training_environment_inventory_sha256: str,
) -> dict[str, object]:
    """Build/replay/write only a fully attested local selector artifact."""

    root = _preview_root(document_preview_root)
    result = train_source_attested_supervised_template_direct_selector_model(
        receipt=read_canonical_source_attested_dataset_receipt_for_selector_training(receipt_path),
        development_packet=read_canonical_source_attested_packet_for_selector_training(
            development_packet_path
        ),
        sealed_packet=read_canonical_source_attested_packet_for_selector_training(
            sealed_packet_path
        ),
        training_split=(
            read_canonical_source_attested_supervised_template_training_split_for_selector_training(
                training_split_path
            )
        ),
        training_labels=(
            read_canonical_source_attested_supervised_template_training_labels_for_selector_training(
                training_labels_path
            )
        ),
        source_directory=source_directory,
        model_id=SUPERVISED_TEMPLATE_DIRECT_SELECTOR_MODEL_ID,
        model_revision=SUPERVISED_TEMPLATE_DIRECT_SELECTOR_MODEL_REVISION,
        training_environment_inventory_sha256=training_environment_inventory_sha256,
    )
    paths = write_source_attested_supervised_template_direct_selector_training_result(
        result=result,
        recipe_output_directory=root / "supervised-template-training-recipes",
        model_root_directory=root / "models" / "native-pdf-semantic",
        model_training_receipt_output_directory=root / "supervised-template-training-receipts",
    )
    return {
        "status": "supervised_template_direct_selector_trained",
        "model_id": SUPERVISED_TEMPLATE_DIRECT_SELECTOR_MODEL_ID,
        "model_revision": SUPERVISED_TEMPLATE_DIRECT_SELECTOR_MODEL_REVISION,
        "model_artifact_sha256": result.model_artifact_sha256,
        "trainer_recipe_sha256": result.recipe.trainer_recipe_sha256,
        "model_training_receipt_sha256": (
            result.model_training_receipt.model_training_receipt_sha256
        ),
        "training_split_receipt_sha256": result.model_training_receipt.training_split_receipt_sha256,
        "training_labels_receipt_sha256": result.model_training_receipt.training_labels_receipt_sha256,
        "training_asset_count": result.training_asset_count,
        "field_label_count": result.field_label_count,
        "recipe_file": paths.recipe_path.name,
        "model_file": paths.model_path.name,
        "model_directory": paths.model_path.parent.name,
        "training_receipt_file": paths.model_training_receipt_path.name,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train local-only native-PDF supervised-template selectors without opening "
            "development or sealed PDFs."
        )
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--development-packet", type=Path, required=True)
    parser.add_argument("--sealed-packet", type=Path, required=True)
    parser.add_argument("--training-split", type=Path, required=True)
    parser.add_argument("--training-labels", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--document-preview-root", type=Path, required=True)
    parser.add_argument("--training-environment-inventory-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        summary = train_source_attested_supervised_template_direct_selector(
            receipt_path=arguments.receipt,
            development_packet_path=arguments.development_packet,
            sealed_packet_path=arguments.sealed_packet,
            training_split_path=arguments.training_split,
            training_labels_path=arguments.training_labels,
            source_directory=arguments.source_directory,
            document_preview_root=arguments.document_preview_root,
            training_environment_inventory_sha256=(arguments.training_environment_inventory_sha256),
        )
    except (SupervisedTemplateDirectSelectorTrainerError, OSError, ValueError):
        print(
            json.dumps(
                {"status": "failed", "code": "supervised_template_direct_selector_training_failed"},
                separators=(",", ":"),
            )
        )
        return 2
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
