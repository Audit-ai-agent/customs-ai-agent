"""Score a local synthetic PDF OCR candidate and write its content-free result."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from customs_ai_agent.evaluation.document_gold import DatasetReceiptV1
from customs_ai_agent.evaluation.document_ocr_quality import (
    CandidateOutputBundleV1,
    EvaluationPacketV2,
    GoldPacketV2,
    score_candidate_output,
    write_evaluation_result,
)


def _default_output_directory() -> Path:
    for parent in (Path.cwd().resolve(), *Path.cwd().resolve().parents):
        local_root = parent / "_local"
        if local_root.is_dir():
            return local_root / "data" / "document-preview" / "evaluation-results"
    raise ValueError("workspace local-only data root is unavailable")


def evaluate_synthetic_pdf_ocr(
    *,
    receipt_path: Path,
    packet_path: Path,
    gold_packet_path: Path,
    candidate_output_path: Path,
    engine_kind: str,
    extractor_revision: str,
    configuration_sha256: str,
    model_inventory_sha256: str,
    output_directory: Path | None = None,
) -> dict[str, object]:
    receipt = DatasetReceiptV1.model_validate_json(receipt_path.read_bytes())
    packet = EvaluationPacketV2.model_validate_json(packet_path.read_bytes())
    gold_packet = GoldPacketV2.model_validate_json(gold_packet_path.read_bytes())
    candidate_output = CandidateOutputBundleV1.model_validate_json(candidate_output_path.read_bytes())
    result = score_candidate_output(
        receipt=receipt,
        packet=packet,
        gold_packet=gold_packet,
        candidate_output=candidate_output,
        engine_kind=engine_kind,  # validated by the strict evaluator model
        extractor_revision=extractor_revision,
        configuration_sha256=configuration_sha256,
        model_inventory_sha256=model_inventory_sha256,
    )
    destination = write_evaluation_result(
        result=result,
        output_directory=output_directory or _default_output_directory(),
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
        description="Score an approved local synthetic PDF OCR candidate without printing OCR content."
    )
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--gold-packet", type=Path, required=True)
    parser.add_argument("--candidate-output", type=Path, required=True)
    parser.add_argument("--engine-kind", choices=("pdf_text_layer", "paddleocr", "docling"), required=True)
    parser.add_argument("--extractor-revision", required=True)
    parser.add_argument("--configuration-sha256", required=True)
    parser.add_argument("--model-inventory-sha256", required=True)
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
            engine_kind=arguments.engine_kind,
            extractor_revision=arguments.extractor_revision,
            configuration_sha256=arguments.configuration_sha256,
            model_inventory_sha256=arguments.model_inventory_sha256,
            output_directory=arguments.output_directory,
        )
    except (OSError, ValueError, json.JSONDecodeError):
        print(json.dumps({"status": "failed", "code": "ocr_evaluation_failed"}, separators=(",", ":")))
        return 2
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
