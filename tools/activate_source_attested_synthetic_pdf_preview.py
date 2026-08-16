"""Activate a passing supervised native-template compatibility result locally.

This command is the automatic counterpart to the preserved human approval
tool.  It does not download packages or models, read Gold/review material, or
print any PDF/OCR content.  It accepts only a canonical source-attested
sealed result and writes the one active PDF pointer atomically after every
runtime sidecar and local artifact is revalidated.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
from pathlib import Path
from typing import Callable, Sequence

from customs_ai_agent.service.document_preview_runtime import (
    DocumentPreviewRuntimeError,
    parse_source_attested_synthetic_sealed_evaluation_result,
    validate_document_preview_root,
    write_source_attested_synthetic_pdf_preview_pointer,
)


def activate_source_attested_synthetic_runtime(
    *,
    document_preview_root: Path,
    evaluation_result_path: Path,
    configuration_sha256: str,
    model_inventory_sha256: str,
    package_version_reader: Callable[[str], str] = importlib.metadata.version,
) -> str:
    """Read one local source-attested result and atomically activate it."""

    root = validate_document_preview_root(document_preview_root, create=True)
    try:
        if evaluation_result_path.is_symlink():
            raise OSError("evaluation result path is a link")
        source = evaluation_result_path.resolve(strict=True)
        # Source-attested sealed results are deliberately kept apart from the
        # human-Gold evaluator's generic ``evaluation-results`` sidecars.
        # The runtime reloads this same immutable source artifact during its
        # evidence-chain verification, so accepting the generic directory here
        # would produce a pointer that can never be safely resolved.
        expected_directory = (root / "source-attested-evaluation-results").resolve(strict=True)
        if source.parent != expected_directory:
            raise OSError("evaluation result path is outside the source-attested result directory")
        if not source.is_file() or source.is_symlink():
            raise OSError("evaluation result is not a regular file")
        raw = source.read_bytes()
    except (OSError, ValueError) as error:
        raise DocumentPreviewRuntimeError(
            "document_preview_source_attested_activation_input_invalid"
        ) from error
    evaluation = parse_source_attested_synthetic_sealed_evaluation_result(raw)
    if source.name != f"{evaluation.evaluation_result_sha256}.json":
        raise DocumentPreviewRuntimeError(
            "document_preview_source_attested_activation_input_invalid"
        )
    activation_sha256, _activation = write_source_attested_synthetic_pdf_preview_pointer(
        document_preview_root=root,
        evaluation_raw=raw,
        configuration_sha256=configuration_sha256,
        model_inventory_sha256=model_inventory_sha256,
        package_version_reader=package_version_reader,
    )
    return activation_sha256


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Atomically activate one passing source-attested synthetic PDF compatibility result."
        )
    )
    parser.add_argument("--document-preview-root", type=Path, required=True)
    parser.add_argument("--evaluation-result", type=Path, required=True)
    parser.add_argument("--configuration-sha256", required=True)
    parser.add_argument("--model-inventory-sha256", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        activation_sha256 = activate_source_attested_synthetic_runtime(
            document_preview_root=arguments.document_preview_root,
            evaluation_result_path=arguments.evaluation_result,
            configuration_sha256=arguments.configuration_sha256,
            model_inventory_sha256=arguments.model_inventory_sha256,
        )
    except DocumentPreviewRuntimeError as error:
        print(json.dumps({"status": "failed", "code": error.code}, separators=(",", ":")))
        return 2
    print(
        json.dumps(
            {
                "status": "activated",
                "activation_kind": "source_attested_synthetic",
                "evaluation_scope": "synthetic_r1_supervised_native_template_compatibility",
                "media_type": "application/pdf",
                "activation_sha256": activation_sha256,
                "runtime_network": "disabled",
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
