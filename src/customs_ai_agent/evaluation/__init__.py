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

__all__ = [
    "DatasetAssetReceiptV1",
    "DatasetReceiptV1",
    "DocumentExtractionGoldV1",
    "EvaluationPacketV1",
    "GoldReleaseThresholdsV1",
    "build_draft_gold_packets",
    "receipt_sha256",
]
