"""Score a provenance-bound local synthetic PDF OCR candidate.

The evaluator is the only local command that receives both reviewed Gold and
candidate OCR output.  It refuses the legacy unbound candidate shape: engine,
revision, runtime configuration, inventory, and parser identities are checked
against independently verified local sidecars before any Gold comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence, cast

from customs_ai_agent.evaluation.document_gold import DatasetReceiptV1
from customs_ai_agent.evaluation.document_ocr_quality import (
    EngineKind,
    EvaluationPacketV2,
    GoldPacketV2,
    SyntheticPdfCandidateOutputBundleV1,
    canonical_json_file_bytes,
    score_synthetic_pdf_candidate_output,
    write_evaluation_result,
)
from customs_ai_agent.evaluation.synthetic_pdf_candidate_runner import (
    SYNTHETIC_PDF_LABEL_NORMALIZER_REVISION,
    SyntheticPdfCandidateRunnerError,
    load_unapproved_local_candidate_profile,
)


def _read_candidate_output(
    *, candidate_output_path: Path, root: Path
) -> SyntheticPdfCandidateOutputBundleV1:
    try:
        if candidate_output_path.is_symlink():
            raise ValueError("candidate output path is a link")
        resolved = candidate_output_path.resolve(strict=True)
        resolved.relative_to(root / "candidate-outputs")
        raw = resolved.read_bytes()
        candidate = SyntheticPdfCandidateOutputBundleV1.model_validate_json(raw)
    except (OSError, ValueError) as error:
        raise SyntheticPdfCandidateRunnerError("candidate_output_invalid") from error
    if (
        resolved.name != f"{candidate.candidate_bundle_sha256}.json"
        or raw != canonical_json_file_bytes(candidate.model_dump(mode="json", warnings=False))
    ):
        raise SyntheticPdfCandidateRunnerError("candidate_output_invalid")
    return candidate


def _evaluation_output_directory(*, output_directory: Path | None, root: Path) -> Path:
    destination = output_directory or root / "evaluation-results"
    try:
        resolved = destination.resolve()
        resolved.relative_to(root / "evaluation-results")
    except ValueError as error:
        raise SyntheticPdfCandidateRunnerError("evaluation_output_directory_invalid") from error
    return resolved


def evaluate_synthetic_pdf_ocr(
    *,
    receipt_path: Path,
    packet_path: Path,
    gold_packet_path: Path,
    candidate_output_path: Path,
    document_preview_root: Path,
    configuration_path: Path,
    inventory_path: Path,
    engine_kind: EngineKind,
    output_directory: Path | None = None,
) -> dict[str, object]:
    profile = load_unapproved_local_candidate_profile(
        document_preview_root=document_preview_root,
        configuration_path=configuration_path,
        inventory_path=inventory_path,
        engine_kind=engine_kind,
    )
    try:
        receipt = DatasetReceiptV1.model_validate_json(receipt_path.read_bytes())
        packet = EvaluationPacketV2.model_validate_json(packet_path.read_bytes())
        gold_packet = GoldPacketV2.model_validate_json(gold_packet_path.read_bytes())
    except (OSError, ValueError) as error:
        raise SyntheticPdfCandidateRunnerError("ocr_evaluation_input_invalid") from error
    candidate_output = _read_candidate_output(
        candidate_output_path=candidate_output_path, root=profile.document_preview_root
    )
    result = score_synthetic_pdf_candidate_output(
        receipt=receipt,
        packet=packet,
        gold_packet=gold_packet,
        candidate_output=candidate_output,
        engine_kind=profile.engine_kind,
        extractor_revision=profile.extractor_revision,
        configuration_sha256=profile.configuration_sha256,
        model_inventory_sha256=profile.model_inventory_sha256,
        parser_revision=SYNTHETIC_PDF_LABEL_NORMALIZER_REVISION,
    )
    destination = write_evaluation_result(
        result=result,
        output_directory=_evaluation_output_directory(
            output_directory=output_directory, root=profile.document_preview_root
        ),
    )
    return {
        "status": "evaluation_recorded",
        "packet_kind": result.packet_kind,
        "evaluation_result_sha256": result.evaluation_result_sha256,
        "passed": result.passed,
        "result_file": destination.name,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Score a reviewed local synthetic-PDF candidate without printing OCR content."
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--gold-packet", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--document-preview-root", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument(
        "--engine-kind", choices=("pdf_text_layer", "paddleocr", "docling"), required=True
    )
    parser.add_argument("--output-directory", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        summary = evaluate_synthetic_pdf_ocr(
            receipt_path=arguments.receipt,
            packet_path=arguments.packet,
            gold_packet_path=arguments.gold_packet,
            candidate_output_path=arguments.candidate_output,
            document_preview_root=arguments.document_preview_root,
            configuration_path=arguments.configuration,
            inventory_path=arguments.inventory,
            engine_kind=cast(EngineKind, arguments.engine_kind),
            output_directory=arguments.output_directory,
        )
    except (SyntheticPdfCandidateRunnerError, OSError, ValueError, json.JSONDecodeError):
        print(
            json.dumps({"status": "failed", "code": "ocr_evaluation_failed"}, separators=(",", ":"))
        )
        return 2
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
