"""Canonical OCR model-set identities shared by workers and release checks."""

from __future__ import annotations

from typing import Final

OCR_MODEL_ASSETS_LOCK_SHA256: Final = (
    "acab257b285e1e174d5c039a022cc41d190215fc0e71283dad28470bbf1d7e8d"
)

OCR_MODEL_SET_IDS: Final[dict[str, str]] = {
    "paddle": "paddle-customs-layout-table-model-set-v2",
    "docling": "docling-default-image-pipeline-model-set-v1",
}
