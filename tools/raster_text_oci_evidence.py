"""Shared exact direct-evidence closure for Raster Text OCR OCI export."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path, PurePosixPath
from typing import Any

from customs_ai_agent.service.raster_text_preview_runtime import (
    RasterTextActivationPointerV12,
    RasterTextSealedEvaluationV9,
    _parse_raster_text_sealed_evaluation,
)

_SHA256 = re.compile(r"[a-f0-9]{64}")
_MAX_EVIDENCE_BYTES = 64 * 1024 * 1024
_DIRECT_EVIDENCE_FIELDS = (
    ("selection_receipt_path", "selection_receipt_sha256"),
    ("projection_receipt_path", "projection_receipt_sha256"),
    ("stage1_decision_receipt_path", "stage1_decision_receipt_sha256"),
    ("source_gold_receipt_path", "source_gold_receipt_sha256"),
    ("calibration_receipt_path", "calibration_receipt_sha256"),
    ("candidate_execution_path", "candidate_execution_sha256"),
    ("warm_service_smoke_receipt_path", "warm_service_smoke_receipt_sha256"),
)


class RasterTextOciEvidenceError(RuntimeError):
    pass


def _relative(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value or len(value) > 512 or "\\" in value:
        raise RasterTextOciEvidenceError
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or any(part in {"", "."} for part in relative.parts)
    ):
        raise RasterTextOciEvidenceError
    return relative


def _regular_child(root: Path, relative: PurePosixPath) -> Path:
    current = root
    try:
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise RasterTextOciEvidenceError
        resolved = current.resolve(strict=True)
    except (OSError, RuntimeError):
        raise RasterTextOciEvidenceError from None
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise RasterTextOciEvidenceError
    return resolved


def _read_bound_file(
    *, root: Path, relative: PurePosixPath, expected_sha256: object, budget: list[int]
) -> bytes:
    if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
        raise RasterTextOciEvidenceError
    path = _regular_child(root, relative)
    try:
        size = path.stat().st_size
        if size <= 0 or size > _MAX_EVIDENCE_BYTES - budget[0]:
            raise RasterTextOciEvidenceError
        payload = path.read_bytes()
    except OSError:
        raise RasterTextOciEvidenceError from None
    if len(payload) != size or hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise RasterTextOciEvidenceError
    budget[0] += size
    return payload


def verify_v9_direct_evidence_closure(*, root: Path, pointer: Any) -> set[PurePosixPath]:
    """Read/hash all seven compact direct V9 receipts via Agent's exact parser."""

    try:
        resolved_root = root.resolve(strict=True)
    except OSError:
        raise RasterTextOciEvidenceError from None
    if (
        root.is_symlink()
        or not resolved_root.is_dir()
        or type(pointer) is not RasterTextActivationPointerV12
    ):
        raise RasterTextOciEvidenceError
    budget = [0]
    sealed_relative = _relative(pointer.sealed_evaluation_path)
    sealed_raw = _read_bound_file(
        root=resolved_root,
        relative=sealed_relative,
        expected_sha256=pointer.sealed_evaluation_sha256,
        budget=budget,
    )
    try:
        evaluation = _parse_raster_text_sealed_evaluation(sealed_raw)
    except Exception:
        raise RasterTextOciEvidenceError from None
    if type(evaluation) is not RasterTextSealedEvaluationV9:
        raise RasterTextOciEvidenceError
    opened = {sealed_relative}
    observed: dict[PurePosixPath, str] = {}
    for path_field, sha_field in _DIRECT_EVIDENCE_FIELDS:
        relative = _relative(getattr(evaluation, path_field))
        expected_sha256 = getattr(evaluation, sha_field)
        previous = observed.setdefault(relative, expected_sha256)
        if previous != expected_sha256:
            raise RasterTextOciEvidenceError
        if relative not in opened:
            _read_bound_file(
                root=resolved_root,
                relative=relative,
                expected_sha256=expected_sha256,
                budget=budget,
            )
            opened.add(relative)
    return opened
