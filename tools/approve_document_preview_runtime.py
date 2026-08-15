"""Atomically activate a human-approved, sealed PDF preview evaluation.

The command is deliberately manual.  It validates the evaluator's canonical
sealed result, the pinned configuration/inventory, installed package versions,
and every local model file before replacing the active pointer.  It never
downloads a dependency or model and never exposes document/OCR content.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from customs_ai_agent.service.document_preview_runtime import (
    DocumentPreviewRuntimeError,
    validate_document_preview_root,
    write_approved_pdf_preview_pointer,
)


def approve_runtime(
    *,
    document_preview_root: Path,
    approval_id: str,
    evaluation_result_path: Path,
    configuration_sha256: str,
    model_inventory_sha256: str,
) -> str:
    root = validate_document_preview_root(document_preview_root, create=True)
    try:
        source = evaluation_result_path.resolve(strict=True)
        source.relative_to(root)
        raw = source.read_bytes()
    except (OSError, ValueError) as error:
        raise DocumentPreviewRuntimeError("document_preview_approval_input_invalid") from error
    approval_sha256, _approval = write_approved_pdf_preview_pointer(
        document_preview_root=root,
        approval_id=approval_id,
        evaluation_raw=raw,
        configuration_sha256=configuration_sha256,
        model_inventory_sha256=model_inventory_sha256,
    )
    return approval_sha256


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Atomically approve one sealed local PDF preview evaluation."
    )
    parser.add_argument("--document-preview-root", type=Path, required=True)
    parser.add_argument("--approval-id", required=True)
    parser.add_argument("--evaluation-result", type=Path, required=True)
    parser.add_argument("--configuration-sha256", required=True)
    parser.add_argument("--model-inventory-sha256", required=True)
    parser.add_argument(
        "--confirm-human-approval",
        action="store_true",
        help="Required acknowledgement that a reviewer approved this sealed result.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if not arguments.confirm_human_approval:
        print(
            json.dumps(
                {"status": "failed", "code": "human_approval_required"}, separators=(",", ":")
            )
        )
        return 2
    try:
        approval_sha256 = approve_runtime(
            document_preview_root=arguments.document_preview_root,
            approval_id=arguments.approval_id,
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
                "status": "approved",
                "approval_sha256": approval_sha256,
                "media_type": "application/pdf",
                "runtime_network": "disabled",
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
