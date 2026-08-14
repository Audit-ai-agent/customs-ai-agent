"""Run one locally provisioned synthetic-PDF OCR candidate, offline only.

The command deliberately has no Gold argument.  It records raw candidate
content only below ``_local/data/document-preview/candidate-outputs`` and does
not create an approval pointer or an HTTP preview result.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence, cast

from customs_ai_agent.evaluation.document_ocr_quality import EngineKind
from customs_ai_agent.evaluation.synthetic_pdf_candidate_runner import (
    SyntheticPdfCandidateRunnerError,
    run_synthetic_pdf_ocr_candidate,
)


_ENGINE_KINDS = ("pdf_text_layer", "paddleocr", "docling")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one offline synthetic-PDF OCR evaluation candidate; "
            "this never approves a preview engine."
        )
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument(
        "--document-preview-root",
        type=Path,
        required=True,
        help="Exact local _local/data/document-preview root containing models and sidecars.",
    )
    parser.add_argument(
        "--configuration",
        type=Path,
        required=True,
        help="Immutable runtime-configurations/<sha256>.json sidecar.",
    )
    parser.add_argument(
        "--inventory",
        type=Path,
        required=True,
        help="Immutable model-inventories/<sha256>.json sidecar.",
    )
    parser.add_argument("--engine-kind", choices=_ENGINE_KINDS, required=True)
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help="A directory below document-preview/candidate-outputs.",
    )
    parser.add_argument(
        "--confirm-local-evaluation",
        action="store_true",
        help="Required acknowledgement that this is a local, unapproved evaluation run.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        summary = run_synthetic_pdf_ocr_candidate(
            receipt_path=arguments.receipt,
            packet_path=arguments.packet,
            source_directory=arguments.source_directory,
            document_preview_root=arguments.document_preview_root,
            configuration_path=arguments.configuration,
            inventory_path=arguments.inventory,
            engine_kind=cast(EngineKind, arguments.engine_kind),
            output_directory=arguments.output_directory,
            confirm_local_evaluation=arguments.confirm_local_evaluation,
        )
    except (SyntheticPdfCandidateRunnerError, OSError, ValueError, json.JSONDecodeError):
        # Do not reveal a PDF path, OCR span, model location, or provider body.
        print(
            json.dumps(
                {"status": "failed", "code": "synthetic_pdf_candidate_run_failed"},
                separators=(",", ":"),
            )
        )
        return 2
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
