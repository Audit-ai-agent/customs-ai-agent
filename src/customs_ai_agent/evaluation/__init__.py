"""Offline-only evaluation metadata helpers.

The package contains schemas and deterministic packet construction only.  Raw
documents, OCR output, and human Gold remain outside Git under ``_local/data``.
"""

from customs_ai_agent.evaluation.document_gold import (
    DatasetAssetReceiptV1,
    DatasetReceiptV1,
    DocumentExtractionGoldV1,
    EvaluationPacketV1,
    GoldReleaseThresholdsV1,
    build_draft_gold_packets,
    receipt_sha256,
)
from customs_ai_agent.evaluation.document_ocr_quality import (
    CandidateOutputBundleV1,
    DocumentExtractionGoldV2,
    EvaluationPacketV2,
    GoldPacketV2,
    OcrEvaluationResultV1,
    build_stratified_pdf_gold_packets,
    build_synthetic_pdf_draft_gold,
    read_canonical_evaluation_result,
    score_candidate_output,
)

__all__ = [
    "DatasetAssetReceiptV1",
    "DatasetReceiptV1",
    "CandidateOutputBundleV1",
    "DocumentExtractionGoldV1",
    "DocumentExtractionGoldV2",
    "EvaluationPacketV1",
    "EvaluationPacketV2",
    "GoldPacketV2",
    "GoldReleaseThresholdsV1",
    "OcrEvaluationResultV1",
    "build_draft_gold_packets",
    "build_stratified_pdf_gold_packets",
    "build_synthetic_pdf_draft_gold",
    "read_canonical_evaluation_result",
    "receipt_sha256",
    "score_candidate_output",
]
