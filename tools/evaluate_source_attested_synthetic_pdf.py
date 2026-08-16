"""Evaluate one local OCR candidate against source-attested synthetic PDF truth.

This is intentionally separate from the reviewed-Gold evaluator.  Its fixed
result scope is ``synthetic_r1_supervised_native_template_compatibility``: it
compares the supervised native-template candidate with the exact final
born-digital synthetic PDF, and it does not assert independent optical-OCR or
real-document quality.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence, cast

from customs_ai_agent.evaluation.document_ocr_quality import (
    EngineKind,
    SyntheticPdfCandidateOutputBundleV1,
    canonical_json_file_bytes,
)
from customs_ai_agent.evaluation.source_attested_synthetic_pdf_truth import (
    SOURCE_ATTESTED_CANDIDATE_PARSER_REVISION,
    SourceAttestedSyntheticPdfTruthError,
    build_source_attested_synthetic_candidate_execution,
    build_source_attested_synthetic_sealed_run,
    read_canonical_source_attested_synthetic_development_selection,
    read_canonical_source_attested_synthetic_layout_truth,
    score_source_attested_synthetic_pdf_candidate_output,
    write_source_attested_synthetic_candidate_bundle,
    write_source_attested_synthetic_candidate_execution,
    write_source_attested_synthetic_dataset_receipt,
    write_source_attested_synthetic_evaluation_result,
    write_source_attested_synthetic_packet,
    write_source_attested_synthetic_sealed_run,
)
from customs_ai_agent.evaluation.synthetic_pdf_candidate_runner import (
    SYNTHETIC_PDF_LABEL_NORMALIZER_REVISION,
    SyntheticPdfCandidateRunnerError,
    load_local_synthetic_pdf_evaluation_inputs,
    load_unapproved_local_candidate_profile,
)


def _path_below(*, path: Path, parent: Path, code: str) -> Path:
    try:
        if path.is_symlink():
            raise ValueError(code)
        resolved = path.resolve(strict=True)
        resolved.relative_to(parent.resolve(strict=True))
        if not resolved.is_file() or resolved.is_symlink():
            raise ValueError(code)
    except (OSError, ValueError) as error:
        raise SyntheticPdfCandidateRunnerError(code) from error
    return resolved


def _candidate_output(
    *, path: Path, root: Path
) -> tuple[SyntheticPdfCandidateOutputBundleV1, bytes]:
    resolved = _path_below(
        path=path, parent=root / "candidate-outputs", code="candidate_output_invalid"
    )
    try:
        raw = resolved.read_bytes()
        candidate = SyntheticPdfCandidateOutputBundleV1.model_validate_json(raw)
    except (OSError, ValueError) as error:
        raise SyntheticPdfCandidateRunnerError("candidate_output_invalid") from error
    if (
        resolved.name != f"{candidate.candidate_bundle_sha256}.json"
        or raw != canonical_json_file_bytes(candidate.model_dump(mode="json", warnings=False))
    ):
        raise SyntheticPdfCandidateRunnerError("candidate_output_invalid")
    return candidate, raw


def _output_directory(*, root: Path, name: str, candidate: Path | None) -> Path:
    value = candidate or root / name
    try:
        resolved = value.resolve()
        resolved.relative_to(root / name)
    except ValueError as error:
        raise SyntheticPdfCandidateRunnerError(
            "source_attested_output_directory_invalid"
        ) from error
    return resolved


def _activate_after_sealed_pass(
    *,
    root: Path,
    evaluation_raw: bytes,
    configuration_sha256: str,
    model_inventory_sha256: str,
) -> str:
    """Load the runtime writer only for the explicit, passing sealed command."""

    try:
        from customs_ai_agent.service.document_preview_runtime import (
            write_source_attested_synthetic_pdf_preview_pointer,
        )
    except ImportError as error:  # pragma: no cover - integration branch owns the writer
        raise SyntheticPdfCandidateRunnerError("source_attested_activation_unavailable") from error
    try:
        activation_sha256, _activation = write_source_attested_synthetic_pdf_preview_pointer(
            document_preview_root=root,
            evaluation_raw=evaluation_raw,
            configuration_sha256=configuration_sha256,
            model_inventory_sha256=model_inventory_sha256,
        )
    except Exception as error:  # noqa: BLE001 - runtime keeps the content-free failure boundary
        raise SyntheticPdfCandidateRunnerError("source_attested_activation_failed") from error
    return activation_sha256


def evaluate_source_attested_synthetic_pdf(
    *,
    receipt_path: Path,
    packet_path: Path,
    layout_truth_path: Path,
    candidate_output_path: Path,
    document_preview_root: Path,
    configuration_path: Path,
    inventory_path: Path,
    engine_kind: EngineKind,
    development_selection_path: Path | None = None,
    output_directory: Path | None = None,
    sealed_run_output_directory: Path | None = None,
    auto_activate_source_attested_synthetic: bool = False,
) -> dict[str, object]:
    # The candidate runner and the source-attested writer deliberately share
    # one fixed rendered-layout parser revision.  Refuse before writing any
    # local evidence if a future runner changes it without a new approval
    # scope, parser spec, and runtime pin.
    if SYNTHETIC_PDF_LABEL_NORMALIZER_REVISION != SOURCE_ATTESTED_CANDIDATE_PARSER_REVISION:
        raise SyntheticPdfCandidateRunnerError("source_attested_candidate_parser_revision_invalid")
    profile = load_unapproved_local_candidate_profile(
        document_preview_root=document_preview_root,
        configuration_path=configuration_path,
        inventory_path=inventory_path,
        engine_kind=engine_kind,
    )
    receipt, packet = load_local_synthetic_pdf_evaluation_inputs(
        receipt_path=receipt_path,
        packet_path=packet_path,
        receipt_code="source_attested_input_invalid",
        packet_code="source_attested_input_invalid",
        validation_code="source_attested_input_invalid",
    )
    write_source_attested_synthetic_dataset_receipt(
        receipt=receipt,
        output_directory=profile.document_preview_root / "source-attested-dataset-receipts",
    )
    write_source_attested_synthetic_packet(
        packet=packet,
        output_directory=profile.document_preview_root / "source-attested-packets",
    )
    layout_truth = read_canonical_source_attested_synthetic_layout_truth(
        _path_below(
            path=layout_truth_path,
            parent=profile.document_preview_root / "source-layout-truths",
            code="source_attested_layout_truth_invalid",
        )
    )
    candidate, candidate_output_raw = _candidate_output(
        path=candidate_output_path, root=profile.document_preview_root
    )
    write_source_attested_synthetic_candidate_bundle(
        candidate_bundle=candidate,
        output_directory=profile.document_preview_root / "source-attested-candidate-bundles",
    )
    candidate_execution = build_source_attested_synthetic_candidate_execution(
        receipt=receipt,
        packet=packet,
        layout_truth=layout_truth,
        candidate_output=candidate,
        candidate_output_raw=candidate_output_raw,
        engine_kind=profile.engine_kind,
        extractor_revision=profile.extractor_revision,
        configuration_sha256=profile.configuration_sha256,
        model_inventory_sha256=profile.model_inventory_sha256,
        parser_revision=SYNTHETIC_PDF_LABEL_NORMALIZER_REVISION,
    )
    write_source_attested_synthetic_candidate_execution(
        execution=candidate_execution,
        output_directory=profile.document_preview_root / "source-attested-candidate-executions",
    )

    development_selection = None
    sealed_run = None
    if packet.split == "sealed":
        if development_selection_path is None:
            raise SyntheticPdfCandidateRunnerError("source_attested_development_selection_required")
        development_selection = read_canonical_source_attested_synthetic_development_selection(
            _path_below(
                path=development_selection_path,
                parent=profile.document_preview_root / "source-attested-development-selections",
                code="source_attested_development_selection_invalid",
            )
        )
        sealed_run = build_source_attested_synthetic_sealed_run(
            packet=packet,
            layout_truth=layout_truth,
            development_selection=development_selection,
            candidate_output=candidate,
            candidate_execution=candidate_execution,
            document_preview_root=profile.document_preview_root,
            engine_kind=profile.engine_kind,
            extractor_revision=profile.extractor_revision,
            configuration_sha256=profile.configuration_sha256,
            model_inventory_sha256=profile.model_inventory_sha256,
            parser_revision=SYNTHETIC_PDF_LABEL_NORMALIZER_REVISION,
        )
        write_source_attested_synthetic_sealed_run(
            sealed_run=sealed_run,
            output_directory=_output_directory(
                root=profile.document_preview_root,
                name="source-attested-sealed-runs",
                candidate=sealed_run_output_directory,
            ),
        )
    elif development_selection_path is not None or auto_activate_source_attested_synthetic:
        raise SyntheticPdfCandidateRunnerError("source_attested_development_mode_invalid")

    result = score_source_attested_synthetic_pdf_candidate_output(
        receipt=receipt,
        packet=packet,
        layout_truth=layout_truth,
        candidate_output_raw=candidate_output_raw,
        candidate_execution=candidate_execution,
        engine_kind=profile.engine_kind,
        extractor_revision=profile.extractor_revision,
        configuration_sha256=profile.configuration_sha256,
        model_inventory_sha256=profile.model_inventory_sha256,
        parser_revision=SYNTHETIC_PDF_LABEL_NORMALIZER_REVISION,
        development_selection=development_selection,
        sealed_run=sealed_run,
    )
    destination = write_source_attested_synthetic_evaluation_result(
        result=result,
        output_directory=_output_directory(
            root=profile.document_preview_root,
            name="source-attested-evaluation-results",
            candidate=output_directory,
        ),
    )
    activation_sha256: str | None = None
    if auto_activate_source_attested_synthetic:
        if packet.split != "sealed" or not result.passed:
            raise SyntheticPdfCandidateRunnerError("source_attested_auto_activation_gate_failed")
        try:
            evaluation_raw = destination.read_bytes()
        except OSError as error:
            raise SyntheticPdfCandidateRunnerError("source_attested_activation_failed") from error
        activation_sha256 = _activate_after_sealed_pass(
            root=profile.document_preview_root,
            evaluation_raw=evaluation_raw,
            configuration_sha256=profile.configuration_sha256,
            model_inventory_sha256=profile.model_inventory_sha256,
        )
    return {
        "status": "source_attested_evaluation_recorded",
        "packet_kind": result.packet_kind,
        "evaluation_result_sha256": result.evaluation_result_sha256,
        "candidate_execution_receipt_sha256": (
            candidate_execution.candidate_execution_receipt_sha256
        ),
        "passed": result.passed,
        "result_file": destination.name,
        "activation_sha256": activation_sha256,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a source-attested synthetic PDF candidate without printing PDF or OCR content."
        )
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--layout-truth", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--document-preview-root", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument(
        "--engine-kind", choices=("pdf_text_layer", "paddleocr", "docling"), required=True
    )
    parser.add_argument("--development-selection", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--sealed-run-output-directory", type=Path)
    parser.add_argument(
        "--auto-activate-source-attested-synthetic",
        action="store_true",
        help=(
            "After one passing sealed source-attested result only, atomically write the local "
            "automatic PDF preview pointer."
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        summary = evaluate_source_attested_synthetic_pdf(
            receipt_path=arguments.receipt,
            packet_path=arguments.packet,
            layout_truth_path=arguments.layout_truth,
            candidate_output_path=arguments.candidate_output,
            document_preview_root=arguments.document_preview_root,
            configuration_path=arguments.configuration,
            inventory_path=arguments.inventory,
            engine_kind=cast(EngineKind, arguments.engine_kind),
            development_selection_path=arguments.development_selection,
            output_directory=arguments.output_directory,
            sealed_run_output_directory=arguments.sealed_run_output_directory,
            auto_activate_source_attested_synthetic=arguments.auto_activate_source_attested_synthetic,
        )
    except (
        SourceAttestedSyntheticPdfTruthError,
        SyntheticPdfCandidateRunnerError,
        OSError,
        ValueError,
        json.JSONDecodeError,
    ):
        print(
            json.dumps(
                {"status": "failed", "code": "source_attested_evaluation_failed"},
                separators=(",", ":"),
            )
        )
        return 2
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
