"""Canonical OCR model-set identities shared by workers and release checks."""

from __future__ import annotations

from typing import Final

OCR_MODEL_ASSETS_LOCK_SHA256: Final = (
    "6d93a74c6087e5ea3d06c9b069f0266b7a7fb714e07d5c18d473dbace3e9afca"
)

OCR_MODEL_SET_IDS: Final[dict[str, str]] = {
    "paddle": "paddle-customs-layout-table-model-set-v2",
    "docling": "docling-default-image-pipeline-model-set-v1",
}
