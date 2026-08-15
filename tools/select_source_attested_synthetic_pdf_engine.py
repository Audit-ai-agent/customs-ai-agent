"""Record the deterministic three-engine source-attested development selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from customs_ai_agent.evaluation.source_attested_synthetic_pdf_truth import (
    SourceAttestedSyntheticPdfTruthError,
    build_source_attested_synthetic_development_selection,
    read_canonical_source_attested_synthetic_evaluation_result,
    write_source_attested_synthetic_development_selection,
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


def _selection_directory(*, root: Path, output_directory: Path | None) -> Path:
    candidate = output_directory or root / "source-attested-development-selections"
    try:
        resolved = candidate.resolve()
        resolved.relative_to(root / "source-attested-development-selections")
    except ValueError as error:
        raise ValueError("source-attested selection output directory is invalid") from error
    return resolved


def _result_path(*, path: Path, root: Path) -> Path:
    try:
        if path.is_symlink():
            raise ValueError("source-attested result path is invalid")
        resolved = path.resolve(strict=True)
        resolved.relative_to((root / "source-attested-evaluation-results").resolve(strict=True))
        if not resolved.is_file() or resolved.is_symlink():
            raise ValueError("source-attested result path is invalid")
    except (OSError, ValueError) as error:
        raise ValueError("source-attested result path is invalid") from error
    return resolved


def select_source_attested_synthetic_pdf_engine(
    *,
    result_paths: Sequence[Path],
    document_preview_root: Path,
    output_directory: Path | None = None,
) -> dict[str, object]:
    root = _preview_root(document_preview_root)
    selection = build_source_attested_synthetic_development_selection(
        results=[
            read_canonical_source_attested_synthetic_evaluation_result(
                _result_path(path=path, root=root)
            )
            for path in result_paths
        ]
    )
    destination = write_source_attested_synthetic_development_selection(
        selection=selection,
        output_directory=_selection_directory(root=root, output_directory=output_directory),
    )
    return {
        "status": "source_attested_development_candidate_selected",
        "development_selection_receipt_sha256": selection.development_selection_receipt_sha256,
        "selected_engine_kind": selection.selected_engine_kind,
        "selected_evaluation_result_sha256": selection.selected_evaluation_result_sha256,
        "result_file": destination.name,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Select one passing source-attested synthetic-PDF development candidate without "
            "printing OCR content."
        )
    )
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--document-preview-root", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        summary = select_source_attested_synthetic_pdf_engine(
            result_paths=arguments.result,
            document_preview_root=arguments.document_preview_root,
            output_directory=arguments.output_directory,
        )
    except (SourceAttestedSyntheticPdfTruthError, OSError, ValueError, json.JSONDecodeError):
        print(
            json.dumps(
                {"status": "failed", "code": "source_attested_development_selection_failed"},
                separators=(",", ":"),
            )
        )
        return 2
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
