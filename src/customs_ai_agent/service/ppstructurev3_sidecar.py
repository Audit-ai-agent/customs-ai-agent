"""Hash-pinnable, offline PP-StructureV3 raster sidecar.

The Agent process deliberately keeps PaddleOCR/PaddleX out of its normal
dependency set.  An operator may instead provision one local executable for
this module, bind its file hash in the approved runtime profile, and hand it
only decoded RGB pages over stdin.  This module is intentionally strict:

* all PP-Structure model directories are explicit and local;
* sockets are denied in the *child process* before PaddleOCR is imported;
* native output is projected into the provider-neutral v2 wire model without
  inventing missing geometry, reading order, fields, or table cells; and
* failures never write OCR text, pixels, paths, or native diagnostics to
  stdout/stderr.

It is not an activation mechanism.  The parent resolver still requires the
hash-bound profile, local package/model inventories, calibration receipt, and
passing sealed evaluation before this command can be selected.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import socket
import struct
import sys
from typing import BinaryIO, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from customs_ai_agent.schemas.document_extraction import BoundingBoxV1
from customs_ai_agent.schemas.raster_document_extraction import (
    NormalizedPointV2,
    NormalizedPolygonV2,
    RasterDocumentPreviewPageV2,
    RasterDocumentSpanV2,
    RasterDocumentType,
    RasterExtractionWarningV2,
    RasterFieldCandidateV2,
    RasterTableCellV2,
    RasterTableRowV2,
    RasterTableV2,
)


_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:-]{0,127}$")
_MAX_CONFIGURATION_BYTES = 1024 * 1024
_MAX_PROTOCOL_HEADER_BYTES = 1024 * 1024
_MAX_RASTER_PIXELS = 40_000_000
_MAX_PAGES = 20
_BATCH_PROTOCOL = "customs-raster-ppstructurev3-batch-v1"
_EVALUATION_BATCH_PROTOCOL = "customs-raster-ppstructurev3-evaluation-batch-v2"
_STRUCTURE_PROBE_PROTOCOL = "customs-raster-ppstructurev3-structure-probe-v1"
# The service protocol deliberately has a different identity from both
# evaluation protocols.  A long-lived OCR process may serve one approved
# profile, but it must never be usable to produce calibration/sealed evidence.
_SERVICE_PROTOCOL = "customs-raster-ppstructurev3-service-v1"
# Evaluation assets are single-page raster files admitted by the same strict
# decoder as the service.  This cap is deliberately separate from the normal
# one-shot protocol: it bounds a single long-lived worker message without
# widening the service request surface.
_MAX_BATCH_REQUEST_BYTES = 128 * 1024 * 1024
_MAX_EVALUATION_BATCH_ITEMS = 4
# Four maximum 216-DPI RGB pages occupy 51.63 MiB in the fixed corpus.  Keep
# a hard 64-MiB raw-pixel ceiling so the parent/child protocol retains headroom
# for its header, BGR copies, model tensors, and the host's one sidecar budget.
_MAX_EVALUATION_BATCH_RAW_BYTES = 64 * 1024 * 1024
# A normal service response remains capped at one MiB.  The evaluation-only
# envelope can carry four independently capped results and has its own bound.
_MAX_EVALUATION_BATCH_RESPONSE_BYTES = (4 * _MAX_CONFIGURATION_BYTES) + (16 * 1024)
_MAX_STRUCTURE_PROBE_TABLES = 100
_MAX_STRUCTURE_PROBE_BYTES = 32 * 1024

_TableProjectionStatus = Literal[
    "not_attempted_unknown_document",
    "document_layout_unsupported",
    "missing_native_table",
    "invalid_native_table",
    "ambiguous_native_table",
    "missing_cell_geometry",
    "invalid_cell_geometry",
    "missing_header_evidence",
    "duplicate_header_evidence",
    "header_grid_incomplete",
    "body_grid_incomplete",
    "missing_body_evidence",
    "duplicate_body_evidence",
    "ambiguous_body_evidence",
    "invalid_projected_geometry",
    "invalid_projected_table",
    "accepted",
]


class PpStructureV3SidecarError(RuntimeError):
    """Content-free error intended to terminate the sidecar without output."""


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


class _SidecarWireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PpStructureV3SidecarConfigurationV1(_SidecarWireModel):
    """Complete PP-StructureV3 model and inference configuration.

    Model names and directories are both required.  PaddleOCR documents that
    an omitted ``*_model_dir`` may download a model, so optional defaults are
    deliberately unavailable in this profile.
    """

    schema_version: Literal["1.0"] = "1.0"
    runtime_api: Literal["paddleocr3-ppstructurev3"] = "paddleocr3-ppstructurev3"
    extractor_revision: str = Field(min_length=1, max_length=256)
    raster_profile_revision: str = Field(min_length=1, max_length=128)

    layout_detection_model_name: str = Field(min_length=1, max_length=128)
    layout_detection_model_directory: str = Field(min_length=1, max_length=240)
    text_detection_model_name: str = Field(min_length=1, max_length=128)
    text_detection_model_directory: str = Field(min_length=1, max_length=240)
    text_recognition_model_name: str = Field(min_length=1, max_length=128)
    text_recognition_model_directory: str = Field(min_length=1, max_length=240)
    table_classification_model_name: str = Field(min_length=1, max_length=128)
    table_classification_model_directory: str = Field(min_length=1, max_length=240)
    wired_table_structure_recognition_model_name: str = Field(min_length=1, max_length=128)
    wired_table_structure_recognition_model_directory: str = Field(min_length=1, max_length=240)
    wireless_table_structure_recognition_model_name: str = Field(min_length=1, max_length=128)
    wireless_table_structure_recognition_model_directory: str = Field(min_length=1, max_length=240)
    wired_table_cells_detection_model_name: str = Field(min_length=1, max_length=128)
    wired_table_cells_detection_model_directory: str = Field(min_length=1, max_length=240)
    wireless_table_cells_detection_model_name: str = Field(min_length=1, max_length=128)
    wireless_table_cells_detection_model_directory: str = Field(min_length=1, max_length=240)

    # The v1 decision deliberately avoids unproven preprocessing and hosted
    # functionality.  Literal values make a changed configuration a different
    # hash-bound artifact rather than a silent runtime variation.
    device: Literal["cpu"] = "cpu"
    # A historical, already hash-bound release may explicitly keep its
    # single-thread configuration. Newly derived releases use the fixed
    # six-thread CPU profile; no environment- or host-dependent value is
    # accepted. Keeping the two exact values readable prevents a source
    # update from invalidating an already approved CPU-1 pointer, while every
    # CPU-6 result still receives a distinct configuration hash, profile
    # revision, calibration, and sealed evaluation.
    cpu_threads: Literal[1, 6]
    # Evaluation-only native batching is explicit profile material.  Missing
    # legacy configurations retain the exact sequential v1 behavior, while a
    # newly derived release must carry ``4`` in its hashed configuration.
    # Service requests remain one-document one-shot regardless of this value.
    evaluation_batch_size: Literal[1, 4] = 1
    # The bundled Windows CPU runtime has an unsafe combination in the
    # upstream PP-StructureV3 defaults: ``limit_type=min`` leaves a 216 DPI
    # page at its full raster size, and the auto-resolved static runtime keeps
    # the new IR enabled.  Both settings are therefore explicit, immutable
    # profile material rather than values inherited from PaddleX YAML.
    #
    # ``max`` bounds the detector's long side to a fixed allow-listed value.
    # 960, 1536, and 2526 remain readable for historical already-hash-bound
    # releases; newly derived candidate releases use 2048. This is not an
    # opportunistic resize at request time: it is part of the signed/config hash-bound
    # candidate and must pass the same sealed Gate as every other provider
    # setting before it can be activated.
    # These fields intentionally have no defaults.  Filling a new engine or
    # detector setting into an old, already-evaluated sidecar JSON would alter
    # its native behavior without changing that JSON's SHA-256, profile, or
    # sealed Gate evidence.  A historical legacy bundle is migrated only by
    # the release builder; normal runtime/approval resolution rejects it.
    inference_engine: Literal["paddle_static"]
    paddle_static_run_mode: Literal["paddle"]
    paddle_static_enable_new_ir: Literal[False]
    paddle_static_enable_cinn: Literal[False]
    text_detection_limit_side_len: Literal[960, 1536, 2048, 2526]
    text_detection_limit_type: Literal["max"]
    # ``Literal`` cannot represent float values in Python's typing rules;
    # these remain required and the validator below pins their exact values.
    text_detection_threshold: float
    text_detection_box_threshold: float
    text_detection_unclip_ratio: float
    enable_hpi: Literal[False] = False
    enable_mkldnn: Literal[False] = False
    use_tensorrt: Literal[False] = False
    use_doc_orientation_classify: Literal[False] = False
    use_doc_unwarping: Literal[False] = False
    use_textline_orientation: Literal[False] = False
    use_seal_recognition: Literal[False] = False
    use_formula_recognition: Literal[False] = False
    use_chart_recognition: Literal[False] = False
    use_region_detection: Literal[False] = False
    use_table_recognition: Literal[True] = True
    use_table_orientation_classify: Literal[False] = False
    use_ocr_results_with_table_cells: Literal[True] = True
    use_e2e_wired_table_rec_model: Literal[False] = False
    use_e2e_wireless_table_rec_model: Literal[False] = False

    @field_validator(
        "layout_detection_model_directory",
        "text_detection_model_directory",
        "text_recognition_model_directory",
        "table_classification_model_directory",
        "wired_table_structure_recognition_model_directory",
        "wireless_table_structure_recognition_model_directory",
        "wired_table_cells_detection_model_directory",
        "wireless_table_cells_detection_model_directory",
    )
    @classmethod
    def _local_relative_directory(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "\\" in value
            or ":" in value
            or any(not part or part == "." for part in path.parts)
        ):
            raise ValueError("model directory must be local and relative")
        return value

    @field_validator(
        "text_detection_threshold",
        "text_detection_box_threshold",
        "text_detection_unclip_ratio",
    )
    @classmethod
    def _fixed_safe_detector_thresholds(cls, value: float, info: object) -> float:
        """Reject unbound detector drift instead of changing native behavior."""

        allowed = {
            "text_detection_threshold": {0.3},
            # Keep the historical 0.6 profile readable while allowing the
            # separately hash-bound 0.4 small-label detection candidate.
            "text_detection_box_threshold": {0.4, 0.6},
            # 1.5 is historical material.  2.0 is the PP-StructureV3
            # documented default and is evaluated only as a distinct profile.
            "text_detection_unclip_ratio": {1.5, 2.0},
        }
        field_name = getattr(info, "field_name", None)
        if field_name not in allowed or value not in allowed[field_name]:
            raise ValueError("detector threshold is not the pinned safe value")
        return value

    def model_directories(self) -> tuple[str, ...]:
        return (
            self.layout_detection_model_directory,
            self.text_detection_model_directory,
            self.text_recognition_model_directory,
            self.table_classification_model_directory,
            self.wired_table_structure_recognition_model_directory,
            self.wireless_table_structure_recognition_model_directory,
            self.wired_table_cells_detection_model_directory,
            self.wireless_table_cells_detection_model_directory,
        )

    @staticmethod
    def _resolve_model_directory(model_root: Path, relative: str) -> Path:
        root = model_root.resolve(strict=True)
        if model_root.is_symlink() or not root.is_dir():
            raise PpStructureV3SidecarError("ppstructurev3_model_root_invalid")
        candidate = (root / Path(*PurePosixPath(relative).parts)).resolve(strict=True)
        try:
            candidate.relative_to(root)
        except ValueError:
            raise PpStructureV3SidecarError("ppstructurev3_model_directory_invalid") from None
        if candidate.is_symlink() or not candidate.is_dir():
            raise PpStructureV3SidecarError("ppstructurev3_model_directory_invalid")
        return candidate

    def validate_model_directories(self, model_root: Path) -> None:
        for directory in self.model_directories():
            self._resolve_model_directory(model_root, directory)

    @staticmethod
    def _native_model_directory(directory: Path) -> str:
        """Return a native-safe model path after the security check.

        Paddle's Windows native inference layer fails to open some otherwise
        valid model files when the absolute path contains non-ASCII characters.
        The profile is deliberately rooted under the project's Korean-named
        directory, so hand the native layer an existing DOS short path instead
        of weakening the path containment check above.  A volume without short
        names cannot safely run this pinned Windows profile and must fail
        closed rather than silently changing the model location.
        """

        return str(_windows_operational_model_path(directory))

    def pipeline_kwargs(self, model_root: Path) -> dict[str, object]:
        """Return the PPStructureV3 *constructor* options explicitly.

        PaddleOCR accepts the table cell handling switches only on
        ``PPStructureV3.predict``.  Keeping these two argument groups separate
        prevents unknown constructor arguments from being passed through to
        PaddleX as silent runtime drift.
        """

        directories = {
            key: self._native_model_directory(self._resolve_model_directory(model_root, value))
            for key, value in (
                ("layout_detection_model_dir", self.layout_detection_model_directory),
                ("text_detection_model_dir", self.text_detection_model_directory),
                ("text_recognition_model_dir", self.text_recognition_model_directory),
                ("table_classification_model_dir", self.table_classification_model_directory),
                (
                    "wired_table_structure_recognition_model_dir",
                    self.wired_table_structure_recognition_model_directory,
                ),
                (
                    "wireless_table_structure_recognition_model_dir",
                    self.wireless_table_structure_recognition_model_directory,
                ),
                (
                    "wired_table_cells_detection_model_dir",
                    self.wired_table_cells_detection_model_directory,
                ),
                (
                    "wireless_table_cells_detection_model_dir",
                    self.wireless_table_cells_detection_model_directory,
                ),
            )
        }
        return {
            "layout_detection_model_name": self.layout_detection_model_name,
            **directories,
            "text_detection_model_name": self.text_detection_model_name,
            "text_recognition_model_name": self.text_recognition_model_name,
            "table_classification_model_name": self.table_classification_model_name,
            "wired_table_structure_recognition_model_name": (
                self.wired_table_structure_recognition_model_name
            ),
            "wireless_table_structure_recognition_model_name": (
                self.wireless_table_structure_recognition_model_name
            ),
            "wired_table_cells_detection_model_name": self.wired_table_cells_detection_model_name,
            "wireless_table_cells_detection_model_name": (
                self.wireless_table_cells_detection_model_name
            ),
            "device": self.device,
            "cpu_threads": self.cpu_threads,
            "engine": self.inference_engine,
            "engine_config": {
                "run_mode": self.paddle_static_run_mode,
                "cpu_threads": self.cpu_threads,
                "enable_new_ir": self.paddle_static_enable_new_ir,
                "enable_cinn": self.paddle_static_enable_cinn,
            },
            "enable_hpi": self.enable_hpi,
            "enable_mkldnn": self.enable_mkldnn,
            "use_tensorrt": self.use_tensorrt,
            "text_det_limit_side_len": self.text_detection_limit_side_len,
            "text_det_limit_type": self.text_detection_limit_type,
            "text_det_thresh": self.text_detection_threshold,
            "text_det_box_thresh": self.text_detection_box_threshold,
            "text_det_unclip_ratio": self.text_detection_unclip_ratio,
            "use_doc_orientation_classify": self.use_doc_orientation_classify,
            "use_doc_unwarping": self.use_doc_unwarping,
            "use_textline_orientation": self.use_textline_orientation,
            "use_seal_recognition": self.use_seal_recognition,
            "use_formula_recognition": self.use_formula_recognition,
            "use_chart_recognition": self.use_chart_recognition,
            "use_region_detection": self.use_region_detection,
            "use_table_recognition": self.use_table_recognition,
        }

    def prediction_kwargs(self) -> dict[str, object]:
        """Return PPStructureV3.predict-only table projection options."""

        return {
            "use_table_orientation_classify": self.use_table_orientation_classify,
            "use_ocr_results_with_table_cells": self.use_ocr_results_with_table_cells,
            "use_e2e_wired_table_rec_model": self.use_e2e_wired_table_rec_model,
            "use_e2e_wireless_table_rec_model": self.use_e2e_wireless_table_rec_model,
        }


class PpStructureV3SidecarResultV1(_SidecarWireModel):
    """Strict compact projection emitted by the locally pinned sidecar."""

    protocol: Literal["customs-raster-ppstructurev3-v1"] = "customs-raster-ppstructurev3-v1"
    source_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    raster_profile_revision: str = Field(min_length=1, max_length=128)
    provider_revision: str = Field(min_length=1, max_length=256)
    document_type: RasterDocumentType = "unknown"
    classification_confidence: float | None = Field(default=None, ge=0, le=1)
    pages: list[RasterDocumentPreviewPageV2] = Field(max_length=_MAX_PAGES)
    field_candidates: list[RasterFieldCandidateV2] = Field(default_factory=list, max_length=256)
    tables: list[RasterTableV2] = Field(default_factory=list, max_length=100)
    warnings: list[RasterExtractionWarningV2] = Field(default_factory=list, max_length=32)
    needs_review_reasons: list[RasterExtractionWarningV2] = Field(
        default_factory=list, max_length=32
    )


class PpStructureV3NativeStructureProbeV1(_SidecarWireModel):
    """Bounded, content-free facts from one native PP-Structure result.

    This is an operator diagnostic for an immutable sidecar bundle. It is
    deliberately not part of the normal OCR result protocol: it never
    serializes recognized text, coordinates, source hashes, native error
    details, arbitrary provider keys, or image bytes. It distinguishes a
    native table-detection absence from a later strict projection rejection.
    """

    page_number: int = Field(ge=1, le=_MAX_PAGES)
    native_output_mapping: bool
    has_overall_ocr_result: bool
    overall_ocr_span_count: int = Field(ge=0, le=100_000)
    has_layout_detection_result: bool
    layout_box_count: int = Field(ge=0, le=100_000)
    layout_table_box_count: int = Field(ge=0, le=100_000)
    has_parsing_result_list: bool
    parsing_block_count: int = Field(ge=0, le=100_000)
    native_table_list_valid: bool
    native_table_count: int = Field(ge=0, le=100_000)
    native_table_mapping_count: int = Field(ge=0, le=_MAX_STRUCTURE_PROBE_TABLES)
    table_cell_box_list_present: list[bool] = Field(max_length=_MAX_STRUCTURE_PROBE_TABLES)
    table_cell_counts: list[int] = Field(max_length=_MAX_STRUCTURE_PROBE_TABLES)
    table_ocr_prediction_present: list[bool] = Field(max_length=_MAX_STRUCTURE_PROBE_TABLES)
    table_ocr_span_counts: list[int] = Field(max_length=_MAX_STRUCTURE_PROBE_TABLES)
    native_table_probe_truncated: bool

    @model_validator(mode="after")
    def _structural_counters_are_self_consistent(
        self,
    ) -> "PpStructureV3NativeStructureProbeV1":
        """Reject a contradictory content-free diagnostic response.

        The parent deliberately receives no OCR text, geometry, source hash,
        or native object.  These relationships are therefore the only way it
        can prove that a bounded probe summary faithfully describes one
        native response rather than a malformed mix of counters.
        """

        if not self.native_output_mapping:
            if any(
                (
                    self.has_overall_ocr_result,
                    self.overall_ocr_span_count,
                    self.has_layout_detection_result,
                    self.layout_box_count,
                    self.layout_table_box_count,
                    self.has_parsing_result_list,
                    self.parsing_block_count,
                    self.native_table_list_valid,
                    self.native_table_count,
                    self.native_table_mapping_count,
                    self.table_cell_box_list_present,
                    self.table_cell_counts,
                    self.table_ocr_prediction_present,
                    self.table_ocr_span_counts,
                    self.native_table_probe_truncated,
                )
            ):
                raise ValueError("native structure probe counters are inconsistent")
            return self
        if (
            (not self.has_overall_ocr_result and self.overall_ocr_span_count != 0)
            or (not self.has_layout_detection_result and self.layout_box_count != 0)
            or (not self.has_layout_detection_result and self.layout_table_box_count != 0)
            or (not self.has_parsing_result_list and self.parsing_block_count != 0)
            or self.layout_table_box_count > self.layout_box_count
        ):
            raise ValueError("native structure probe counters are inconsistent")
        table_lists = (
            self.table_cell_box_list_present,
            self.table_cell_counts,
            self.table_ocr_prediction_present,
            self.table_ocr_span_counts,
        )
        if not self.native_table_list_valid:
            if (
                self.native_table_count != 0
                or self.native_table_mapping_count != 0
                or any(table_lists)
                or self.native_table_probe_truncated
            ):
                raise ValueError("native structure probe table counters are inconsistent")
            return self
        visible_table_count = min(self.native_table_count, _MAX_STRUCTURE_PROBE_TABLES)
        if (
            any(len(values) != visible_table_count for values in table_lists)
            or self.native_table_mapping_count > visible_table_count
            or any(count < 0 for count in self.table_cell_counts)
            or any(count < 0 for count in self.table_ocr_span_counts)
            or self.native_table_probe_truncated
            != (self.native_table_count > _MAX_STRUCTURE_PROBE_TABLES)
        ):
            raise ValueError("native structure probe table counters are inconsistent")
        if any(
            not present and count != 0
            for present, count in zip(
                self.table_cell_box_list_present, self.table_cell_counts, strict=True
            )
        ) or any(
            not present and count != 0
            for present, count in zip(
                self.table_ocr_prediction_present, self.table_ocr_span_counts, strict=True
            )
        ):
            raise ValueError("native structure probe table counters are inconsistent")
        return self


class PpStructureV3StructureProbeResultV1(_SidecarWireModel):
    """Content-free one-shot diagnostic envelope for a request's pages."""

    protocol: Literal["customs-raster-ppstructurev3-structure-probe-v1"] = (
        "customs-raster-ppstructurev3-structure-probe-v1"
    )
    pages: list[PpStructureV3NativeStructureProbeV1] = Field(min_length=1, max_length=_MAX_PAGES)
    table_projection_status: _TableProjectionStatus


class _EvaluationBatchResultItemV2(_SidecarWireModel):
    """One result re-bound to the parent's opaque batch item identifier."""

    item_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER.pattern)
    result: PpStructureV3SidecarResultV1


class _EvaluationBatchResultV2(_SidecarWireModel):
    protocol: Literal["customs-raster-ppstructurev3-evaluation-batch-v2"] = (
        "customs-raster-ppstructurev3-evaluation-batch-v2"
    )
    results: list[_EvaluationBatchResultItemV2] = Field(
        min_length=1, max_length=_MAX_EVALUATION_BATCH_ITEMS
    )

    @model_validator(mode="after")
    def _result_ids_are_unique(self) -> "_EvaluationBatchResultV2":
        item_ids = [item.item_id for item in self.results]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("evaluation batch result identifiers must be unique")
        return self


@dataclass(frozen=True, slots=True)
class _InputPage:
    page_number: int
    width: int
    height: int
    pixels: bytes


class _EvaluationBatchRequestItemV2(_SidecarWireModel):
    """One opaque parent-bound item in the evaluation-only native batch."""

    item_id: str = Field(min_length=1, max_length=128, pattern=_IDENTIFIER.pattern)
    input_length: int = Field(ge=1, le=_MAX_BATCH_REQUEST_BYTES)
    input_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")


class _EvaluationBatchRequestHeaderV2(_SidecarWireModel):
    protocol: Literal["customs-raster-ppstructurev3-evaluation-batch-v2"] = (
        "customs-raster-ppstructurev3-evaluation-batch-v2"
    )
    items: list[_EvaluationBatchRequestItemV2] = Field(
        min_length=1, max_length=_MAX_EVALUATION_BATCH_ITEMS
    )

    @model_validator(mode="after")
    def _items_are_unique_and_bounded(self) -> "_EvaluationBatchRequestHeaderV2":
        item_ids = [item.item_id for item in self.items]
        if len(item_ids) != len(set(item_ids)):
            raise ValueError("evaluation batch item identifiers must be unique")
        if sum(item.input_length for item in self.items) > _MAX_EVALUATION_BATCH_RAW_BYTES:
            raise ValueError("evaluation batch raw payload is too large")
        return self


@dataclass(frozen=True, slots=True)
class _PixelBox:
    left: float
    top: float
    right: float
    bottom: float

    @property
    def width(self) -> float:
        return self.right - self.left

    @property
    def height(self) -> float:
        return self.bottom - self.top

    @property
    def center(self) -> tuple[float, float]:
        return ((self.left + self.right) / 2, (self.top + self.bottom) / 2)

    def contains_center(self, point: tuple[float, float]) -> bool:
        return self.left <= point[0] <= self.right and self.top <= point[1] <= self.bottom


@dataclass(frozen=True, slots=True)
class _RawSpan:
    text: str
    confidence: float
    bbox: BoundingBoxV1
    polygon: NormalizedPolygonV2
    pixel_bbox: _PixelBox
    native_index: int


@contextmanager
def deny_network_connections() -> Iterator[None]:
    """Deny sockets in this child process, including during OCR imports.

    The parent process's thread-local guard does not cross a subprocess
    boundary.  This independent, short-lived patch is therefore mandatory for
    the sidecar itself.  It is restored in unit tests; production sidecars exit
    immediately after one request.
    """

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_create_connection = socket.create_connection

    def _deny_connect(_socket: socket.socket, _address: object) -> None:
        raise OSError("network access denied")

    def _deny_connect_ex(_socket: socket.socket, _address: object) -> int:
        return getattr(socket, "EAI_FAIL", 1)

    def _deny_create_connection(*_args: object, **_kwargs: object) -> socket.socket:
        raise OSError("network access denied")

    setattr(socket.socket, "connect", _deny_connect)
    setattr(socket.socket, "connect_ex", _deny_connect_ex)
    setattr(socket, "create_connection", _deny_create_connection)
    try:
        yield
    finally:
        setattr(socket.socket, "connect", original_connect)
        setattr(socket.socket, "connect_ex", original_connect_ex)
        setattr(socket, "create_connection", original_create_connection)


def _relative_local_file(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or ".." in path.parts or "\\" in value or ":" in value:
        raise PpStructureV3SidecarError("ppstructurev3_configuration_invalid")
    return path


def _read_hashed_file(path: Path, expected_sha256: str, *, maximum_size: int) -> bytes:
    try:
        if (
            _SHA256.fullmatch(expected_sha256) is None
            or path.is_symlink()
            or not path.is_file()
            or path.stat().st_size < 1
            or path.stat().st_size > maximum_size
        ):
            raise PpStructureV3SidecarError("ppstructurev3_configuration_invalid")
        raw = path.read_bytes()
    except PpStructureV3SidecarError:
        raise
    except OSError:
        raise PpStructureV3SidecarError("ppstructurev3_configuration_invalid") from None
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise PpStructureV3SidecarError("ppstructurev3_configuration_drift")
    return raw


def _environment_path(name: str) -> Path:
    value = os.environ.get(name)
    if not value:
        raise PpStructureV3SidecarError("ppstructurev3_configuration_invalid")
    return Path(value)


def _verify_sidecar_module_binding() -> None:
    """Ensure the child imported the same first-party source the profile pinned."""

    expected_sha256 = os.environ.get("CUSTOMS_AGENT_PPSTRUCTURE_SIDECAR_MODULE_SHA256", "")
    module_path = Path(__file__)
    _read_hashed_file(module_path, expected_sha256, maximum_size=8 * 1024 * 1024)


def _verify_child_package_inventory(
    package_inventory_path: Path,
    expected_sha256: str,
    required_packages_raw: str,
) -> None:
    """Bind the sidecar interpreter, not only the parent interpreter, to pins."""

    raw = _read_hashed_file(
        package_inventory_path,
        expected_sha256,
        maximum_size=_MAX_CONFIGURATION_BYTES,
    )
    try:
        payload = json.loads(raw)
        packages = payload["packages"]
        required_packages = json.loads(required_packages_raw)
    except (KeyError, TypeError, json.JSONDecodeError):
        raise PpStructureV3SidecarError("ppstructurev3_package_inventory_invalid") from None
    if (
        not isinstance(packages, list)
        or not isinstance(required_packages, list)
        or not all(isinstance(name, str) and name for name in required_packages)
        or len(required_packages) != len(set(required_packages))
    ):
        raise PpStructureV3SidecarError("ppstructurev3_package_inventory_invalid")
    observed: dict[str, str] = {}
    for item in packages:
        if (
            not isinstance(item, Mapping)
            or not isinstance(item.get("name"), str)
            or not isinstance(item.get("version"), str)
            or not item["name"]
            or not item["version"]
            or item["name"] in observed
        ):
            raise PpStructureV3SidecarError("ppstructurev3_package_inventory_invalid")
        observed[item["name"]] = item["version"]
    for package_name in required_packages:
        try:
            if importlib.metadata.version(package_name) != observed.get(package_name):
                raise PpStructureV3SidecarError("ppstructurev3_package_inventory_drift")
        except importlib.metadata.PackageNotFoundError:
            raise PpStructureV3SidecarError("ppstructurev3_package_inventory_drift") from None


def _sha256_regular_file(path: Path) -> str:
    """Hash a declared local model artifact without loading it all into memory."""

    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size < 1:
            raise PpStructureV3SidecarError("ppstructurev3_model_inventory_invalid")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()
    except PpStructureV3SidecarError:
        raise
    except OSError:
        raise PpStructureV3SidecarError("ppstructurev3_model_inventory_invalid") from None


def _windows_operational_model_path(path: Path) -> Path:
    """Return a DOS alias for deep local model trees after validation.

    The canonical long path remains the security identity.  A release adds a
    content-addressed 64-character directory below the project root, though,
    and Windows recursive traversal can then fail if it uses the resolved long
    spelling.  Reusing the same ASCII alias for hashing and native inference
    keeps those two operations on the exact already-validated directory.
    """

    if os.name != "nt":
        return path
    try:
        import ctypes

        value = str(path)
        kernel32 = getattr(ctypes, "windll").kernel32
        required = kernel32.GetShortPathNameW(value, None, 0)
        if required <= 0:
            raise OSError("short path unavailable")
        buffer = ctypes.create_unicode_buffer(required + 1)
        written = kernel32.GetShortPathNameW(value, buffer, len(buffer))
        short_value = buffer.value if 0 < written < len(buffer) else ""
    except Exception:
        raise PpStructureV3SidecarError("ppstructurev3_model_path_unsupported") from None
    if not short_value or not short_value.isascii():
        raise PpStructureV3SidecarError("ppstructurev3_model_path_unsupported")
    return Path(short_value)


def _verify_child_model_inventory(
    *, model_root: Path, model_inventory_path: Path, expected_sha256: str
) -> None:
    """Recheck all model bytes in the one-request child before Paddle imports.

    The parent checks this inventory when resolving activation.  A sidecar is
    short lived, so repeating it here also rejects a model replacement made
    after service startup and before this request's native runtime is created.
    """

    raw = _read_hashed_file(
        model_inventory_path,
        expected_sha256,
        maximum_size=_MAX_CONFIGURATION_BYTES,
    )
    try:
        payload = json.loads(raw)
        artifacts = payload["artifacts"]
    except (KeyError, TypeError, json.JSONDecodeError):
        raise PpStructureV3SidecarError("ppstructurev3_model_inventory_invalid") from None
    if not isinstance(artifacts, list) or not artifacts:
        raise PpStructureV3SidecarError("ppstructurev3_model_inventory_invalid")
    try:
        root = model_root.resolve(strict=True)
        if model_root.is_symlink() or not root.is_dir():
            raise PpStructureV3SidecarError("ppstructurev3_model_inventory_invalid")
        operational_root = _windows_operational_model_path(root)
    except PpStructureV3SidecarError:
        raise
    except OSError:
        raise PpStructureV3SidecarError("ppstructurev3_model_inventory_invalid") from None
    declared: set[str] = set()
    for item in artifacts:
        if not isinstance(item, Mapping):
            raise PpStructureV3SidecarError("ppstructurev3_model_inventory_invalid")
        relative = item.get("relative_path")
        expected = item.get("sha256")
        if (
            not isinstance(relative, str)
            or not isinstance(expected, str)
            or _SHA256.fullmatch(expected) is None
            or relative in declared
        ):
            raise PpStructureV3SidecarError("ppstructurev3_model_inventory_invalid")
        local = _relative_local_file(relative)
        try:
            candidate = (root / Path(*local.parts)).resolve(strict=True)
            candidate.relative_to(root)
        except (OSError, ValueError):
            raise PpStructureV3SidecarError("ppstructurev3_model_inventory_invalid") from None
        operational_candidate = operational_root / Path(*local.parts)
        if _sha256_regular_file(operational_candidate) != expected:
            raise PpStructureV3SidecarError("ppstructurev3_model_inventory_drift")
        declared.add(relative)
    try:
        actual: set[str] = set()
        for candidate in operational_root.rglob("*"):
            if candidate.is_symlink():
                raise PpStructureV3SidecarError("ppstructurev3_model_inventory_invalid")
            if candidate.is_file():
                actual.add(candidate.relative_to(operational_root).as_posix())
            elif not candidate.is_dir():
                raise PpStructureV3SidecarError("ppstructurev3_model_inventory_invalid")
        if actual != declared:
            raise PpStructureV3SidecarError("ppstructurev3_model_inventory_drift")
    except PpStructureV3SidecarError:
        raise
    except OSError:
        raise PpStructureV3SidecarError("ppstructurev3_model_inventory_invalid") from None


def configuration_from_environment() -> tuple[PpStructureV3SidecarConfigurationV1, Path]:
    """Read the one hash-bound configuration and local model root from env."""

    if os.environ.get("CUSTOMS_AGENT_LOCAL_FILES_ONLY") != "1":
        raise PpStructureV3SidecarError("ppstructurev3_local_only_required")
    _verify_sidecar_module_binding()
    configuration_path = _environment_path("CUSTOMS_AGENT_PPSTRUCTURE_CONFIG_PATH")
    configuration_sha256 = os.environ.get("CUSTOMS_AGENT_PPSTRUCTURE_CONFIG_SHA256", "")
    raw = _read_hashed_file(
        configuration_path,
        configuration_sha256,
        maximum_size=_MAX_CONFIGURATION_BYTES,
    )
    try:
        configuration = PpStructureV3SidecarConfigurationV1.model_validate_json(raw)
    except ValidationError:
        raise PpStructureV3SidecarError("ppstructurev3_configuration_invalid") from None
    model_root = _environment_path("CUSTOMS_AGENT_PPSTRUCTURE_MODEL_ROOT")
    configuration.validate_model_directories(model_root)
    _verify_child_package_inventory(
        _environment_path("CUSTOMS_AGENT_PPSTRUCTURE_PACKAGE_INVENTORY_PATH"),
        os.environ.get("CUSTOMS_AGENT_PPSTRUCTURE_PACKAGE_INVENTORY_SHA256", ""),
        os.environ.get("CUSTOMS_AGENT_PPSTRUCTURE_REQUIRED_PACKAGES", ""),
    )
    _verify_child_model_inventory(
        model_root=model_root,
        model_inventory_path=_environment_path("CUSTOMS_AGENT_PPSTRUCTURE_MODEL_INVENTORY_PATH"),
        expected_sha256=os.environ.get("CUSTOMS_AGENT_PPSTRUCTURE_MODEL_INVENTORY_SHA256", ""),
    )
    return configuration, model_root


def _as_mapping(value: object) -> Mapping[str, object] | None:
    if isinstance(value, Mapping):
        return value
    model_dump = getattr(value, "model_dump", None)
    if callable(model_dump):
        dumped = model_dump()
        return dumped if isinstance(dumped, Mapping) else None
    to_dict = getattr(value, "to_dict", None)
    if callable(to_dict):
        dumped = to_dict()
        return dumped if isinstance(dumped, Mapping) else None
    json_value = getattr(value, "json", None)
    if callable(json_value):
        json_value = json_value()
    if isinstance(json_value, str):
        try:
            parsed = json.loads(json_value)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, Mapping) else None
    return json_value if isinstance(json_value, Mapping) else None


def _as_list(value: object) -> list[object] | None:
    if isinstance(value, (str, bytes, bytearray, Mapping)):
        return None
    to_list = getattr(value, "tolist", None)
    if callable(to_list):
        value = to_list()
    if isinstance(value, Sequence):
        return list(value)
    return None


def summarize_ppstructurev3_native_structure(
    native_result: object,
    *,
    page_number: int,
) -> PpStructureV3NativeStructureProbeV1:
    """Return only bounded structural counters from one native result.

    Keep this separate from ``project_ppstructurev3_predictions`` so normal
    OCR output and its fail-closed semantics do not change. In particular,
    values under ``rec_texts`` are inspected only for their sequence length;
    their content is never copied into the returned diagnostic object.
    """

    mapping = _as_mapping(native_result)
    if mapping is None:
        return PpStructureV3NativeStructureProbeV1(
            page_number=page_number,
            native_output_mapping=False,
            has_overall_ocr_result=False,
            overall_ocr_span_count=0,
            has_layout_detection_result=False,
            layout_box_count=0,
            layout_table_box_count=0,
            has_parsing_result_list=False,
            parsing_block_count=0,
            native_table_list_valid=False,
            native_table_count=0,
            native_table_mapping_count=0,
            table_cell_box_list_present=[],
            table_cell_counts=[],
            table_ocr_prediction_present=[],
            table_ocr_span_counts=[],
            native_table_probe_truncated=False,
        )

    overall = _as_mapping(mapping.get("overall_ocr_res"))
    overall_texts = _as_list(overall.get("rec_texts")) if overall is not None else None
    layout = _as_mapping(mapping.get("layout_det_res"))
    layout_boxes = _as_list(layout.get("boxes")) if layout is not None else None
    layout_table_box_count = 0
    for value in layout_boxes or ():
        box = _as_mapping(value)
        label = box.get("label") if box is not None else None
        if isinstance(label, str) and label.casefold() == "table":
            layout_table_box_count += 1
    parsing = _as_list(mapping.get("parsing_res_list"))
    tables = _as_list(mapping.get("table_res_list"))
    bounded_tables = (tables or [])[:_MAX_STRUCTURE_PROBE_TABLES]
    table_mapping_count = 0
    cell_box_list_present: list[bool] = []
    cell_counts: list[int] = []
    table_ocr_prediction_present: list[bool] = []
    table_ocr_span_counts: list[int] = []
    for value in bounded_tables:
        table = _as_mapping(value)
        if table is None:
            cell_box_list_present.append(False)
            cell_counts.append(0)
            table_ocr_prediction_present.append(False)
            table_ocr_span_counts.append(0)
            continue
        table_mapping_count += 1
        cells = _as_list(table.get("cell_box_list"))
        cell_box_list_present.append(cells is not None)
        cell_counts.append(len(cells) if cells is not None else 0)
        table_ocr = _as_mapping(table.get("table_ocr_pred"))
        table_ocr_prediction_present.append(table_ocr is not None)
        table_ocr_texts = _as_list(table_ocr.get("rec_texts")) if table_ocr is not None else None
        table_ocr_span_counts.append(len(table_ocr_texts) if table_ocr_texts is not None else 0)
    return PpStructureV3NativeStructureProbeV1(
        page_number=page_number,
        native_output_mapping=True,
        has_overall_ocr_result=overall is not None,
        overall_ocr_span_count=len(overall_texts) if overall_texts is not None else 0,
        has_layout_detection_result=layout is not None,
        layout_box_count=len(layout_boxes) if layout_boxes is not None else 0,
        layout_table_box_count=layout_table_box_count,
        has_parsing_result_list=parsing is not None,
        parsing_block_count=len(parsing) if parsing is not None else 0,
        native_table_list_valid=tables is not None,
        native_table_count=len(tables) if tables is not None else 0,
        native_table_mapping_count=table_mapping_count,
        table_cell_box_list_present=cell_box_list_present,
        table_cell_counts=cell_counts,
        table_ocr_prediction_present=table_ocr_prediction_present,
        table_ocr_span_counts=table_ocr_span_counts,
        native_table_probe_truncated=(
            tables is not None and len(tables) > _MAX_STRUCTURE_PROBE_TABLES
        ),
    )


def _finite_number(value: object) -> float | None:
    if isinstance(value, bool):
        return None
    try:
        # ``str`` keeps this boundary tolerant of NumPy scalar values without
        # accepting arbitrary native objects as executable input.
        number = float(str(value))
    except (TypeError, ValueError):
        return None
    return number if number == number and abs(number) != float("inf") else None


def _pixel_box_from_rectangle(value: object, *, width: int, height: int) -> _PixelBox | None:
    values: list[object] | None = None
    mapping = _as_mapping(value)
    if mapping is not None:
        if {"x", "y", "width", "height"} <= set(mapping):
            x = _finite_number(mapping["x"])
            y = _finite_number(mapping["y"])
            box_width = _finite_number(mapping["width"])
            box_height = _finite_number(mapping["height"])
            if x is None or y is None or box_width is None or box_height is None:
                return None
            values = [
                x,
                y,
                x + box_width,
                y + box_height,
            ]
        elif {"left", "top", "right", "bottom"} <= set(mapping):
            values = [mapping["left"], mapping["top"], mapping["right"], mapping["bottom"]]
        elif {"x1", "y1", "x2", "y2"} <= set(mapping):
            values = [mapping["x1"], mapping["y1"], mapping["x2"], mapping["y2"]]
    else:
        values = _as_list(value)
    if values is None or len(values) != 4:
        return None
    numbers = [_finite_number(item) for item in values]
    if any(item is None for item in numbers):
        return None
    left, top, right, bottom = (float(item) for item in numbers if item is not None)
    if left < 0 or top < 0 or right > width or bottom > height or left >= right or top >= bottom:
        return None
    return _PixelBox(left, top, right, bottom)


def _polygon_from_native(
    value: object, *, width: int, height: int
) -> tuple[NormalizedPolygonV2, BoundingBoxV1, _PixelBox] | None:
    points = _as_list(value)
    if points is None or not 4 <= len(points) <= 32:
        return None
    pixel_points: list[tuple[float, float]] = []
    for point in points:
        pair = _as_list(point)
        if pair is None or len(pair) != 2:
            return None
        x, y = (_finite_number(pair[0]), _finite_number(pair[1]))
        if x is None or y is None or x < 0 or y < 0 or x > width or y > height:
            return None
        pixel_points.append((x, y))
    left = min(point[0] for point in pixel_points)
    top = min(point[1] for point in pixel_points)
    right = max(point[0] for point in pixel_points)
    bottom = max(point[1] for point in pixel_points)
    if left >= right or top >= bottom:
        return None
    try:
        polygon = NormalizedPolygonV2(
            points=[NormalizedPointV2(x=x / width, y=y / height) for x, y in pixel_points]
        )
        bbox = BoundingBoxV1(
            x=left / width,
            y=top / height,
            width=(right - left) / width,
            height=(bottom - top) / height,
        )
    except ValidationError:
        return None
    return polygon, bbox, _PixelBox(left, top, right, bottom)


def _native_ocr_spans(
    result: Mapping[str, object], page: _InputPage
) -> tuple[list[_RawSpan], set[str]]:
    overall = _as_mapping(result.get("overall_ocr_res"))
    if overall is None:
        return [], {"ppstructurev3_output_incomplete"}
    texts = _as_list(overall.get("rec_texts"))
    scores = _as_list(overall.get("rec_scores"))
    polygons = _as_list(overall.get("rec_polys"))
    if (
        texts is None
        or scores is None
        or polygons is None
        or not (len(texts) == len(scores) == len(polygons))
    ):
        return [], {"ppstructurev3_output_incomplete"}
    spans: list[_RawSpan] = []
    reasons: set[str] = set()
    for native_index, (text, score, polygon) in enumerate(zip(texts, scores, polygons), start=1):
        if not isinstance(text, str) or not text.strip():
            reasons.add("ppstructurev3_output_incomplete")
            continue
        confidence = _finite_number(score)
        geometry = _polygon_from_native(polygon, width=page.width, height=page.height)
        if confidence is None or not 0 <= confidence <= 1 or geometry is None:
            reasons.add("layout_incomplete")
            continue
        normalized_polygon, bbox, pixel_bbox = geometry
        spans.append(
            _RawSpan(
                text=text.strip(),
                confidence=confidence,
                bbox=bbox,
                polygon=normalized_polygon,
                pixel_bbox=pixel_bbox,
                native_index=native_index,
            )
        )
    return spans, reasons


def _block_orders(result: Mapping[str, object]) -> list[tuple[int, _PixelBox]]:
    parsing = _as_list(result.get("parsing_res_list"))
    if parsing is None:
        return []
    ordered: list[tuple[int, _PixelBox]] = []
    for entry in parsing:
        mapping = _as_mapping(entry)
        if mapping is None:
            continue
        order_value = mapping.get("block_order", mapping.get("reading_order"))
        order = _finite_number(order_value)
        if order is None or not order.is_integer() or order < 0:
            continue
        rectangle = (
            mapping.get("block_bbox")
            if "block_bbox" in mapping
            else mapping.get("block_box", mapping.get("bbox"))
        )
        # Page dimensions are unavailable here.  Native parsing boxes are
        # already bounded by spans below; accepting no block is safer than
        # guessing a normalized geometry conversion.
        raw = _as_list(rectangle)
        if raw is None or len(raw) != 4:
            continue
        values = [_finite_number(item) for item in raw]
        if any(value is None for value in values):
            continue
        left, top, right, bottom = (float(value) for value in values if value is not None)
        if left < right and top < bottom:
            ordered.append((int(order), _PixelBox(left, top, right, bottom)))
    return ordered


def _ordered_spans(
    raw_spans: list[_RawSpan], result: Mapping[str, object], page_number: int
) -> tuple[list[RasterDocumentSpanV2], set[str]]:
    """Keep PP-Structure block order where it covers every OCR span.

    ``rec_texts`` itself has an order, but it is not documented as multi-column
    reading order.  If parsing block order does not fully cover the spans we
    retain the raw OCR order only to make the draft inspectable and explicitly
    send it to review.
    """

    if not raw_spans:
        return [], set()
    block_orders = _block_orders(result)
    assigned: list[tuple[int, int] | None] = []
    for span in raw_spans:
        containing = [
            order for order, box in block_orders if box.contains_center(span.pixel_bbox.center)
        ]
        assigned.append((min(containing), span.native_index) if len(containing) == 1 else None)
    reasons: set[str] = set()
    if all(item is not None for item in assigned):

        def _bound_key(index: int) -> tuple[int, int]:
            value = assigned[index]
            assert value is not None
            return value

        indexes = sorted(
            range(len(raw_spans)),
            key=_bound_key,
        )
    else:
        indexes = list(range(len(raw_spans)))
        reasons.add("provider_reading_order_unavailable")
    output: list[RasterDocumentSpanV2] = []
    for ordinal, index in enumerate(indexes, start=1):
        span = raw_spans[index]
        output.append(
            RasterDocumentSpanV2(
                span_id=f"p{page_number}-s{ordinal}",
                ordinal=ordinal,
                reading_order=ordinal,
                text=span.text,
                bbox=span.bbox,
                polygon=span.polygon,
                confidence=span.confidence,
            )
        )
    return output, reasons


_DOCUMENT_TITLES: tuple[tuple[RasterDocumentType, str], ...] = (
    ("bill_of_lading", "bill of lading"),
    ("commercial_invoice", "commercial invoice"),
    ("packing_list", "packing list"),
)
_DOCUMENT_TITLE_MAX_JOINED_SPANS = 3
_DOCUMENT_TITLE_MAX_HORIZONTAL_GAP = 0.100


def _is_contiguous_title_row(evidence: Sequence[RasterDocumentSpanV2]) -> bool:
    """Require a short, visually contiguous OCR title row.

    This is deliberately not a fuzzy title classifier.  It only allows an
    OCR provider to split one printed supported title over at most three
    consecutive spans on the same visual row.  A disconnected phrase or a
    field label containing title words therefore cannot open the structured
    known-form mapper.
    """

    if not evidence or any(span.bbox is None for span in evidence):
        return False
    page_numbers = {_span_page_number(span) for span in evidence}
    if len(page_numbers) != 1 or None in page_numbers:
        return False
    for previous, current in zip(evidence, evidence[1:]):
        if current.reading_order != previous.reading_order + 1:
            return False
        assert previous.bbox is not None and current.bbox is not None
        prior_box = previous.bbox
        current_box = current.bbox
        if current_box.x + _FIELD_POSITION_EPSILON < prior_box.x:
            return False
        if current_box.x - (prior_box.x + prior_box.width) > _DOCUMENT_TITLE_MAX_HORIZONTAL_GAP:
            return False
        if abs(
            (current_box.y + current_box.height / 2) - (prior_box.y + prior_box.height / 2)
        ) > max(_FIELD_SAME_ROW_Y_DELTA, prior_box.height, current_box.height):
            return False
    return True


def _document_title_confidences(
    *, title: str, spans: Sequence[RasterDocumentSpanV2]
) -> list[float]:
    """Return only exact one-to-three-span instances of one visible title."""

    confidences: list[float] = []
    for start in range(len(spans)):
        evidence: list[RasterDocumentSpanV2] = []
        for end in range(start, min(len(spans), start + _DOCUMENT_TITLE_MAX_JOINED_SPANS)):
            evidence.append(spans[end])
            if not _is_contiguous_title_row(evidence):
                break
            if _normalized_visible_label(" ".join(span.text for span in evidence)) == title:
                confidences.append(min(span.confidence for span in evidence))
    return confidences


def _classify_document(
    spans: Sequence[RasterDocumentSpanV2],
) -> tuple[RasterDocumentType, float | None, set[str]]:
    matches: dict[RasterDocumentType, list[float]] = {}
    for document_type, title in _DOCUMENT_TITLES:
        confidence = _document_title_confidences(title=title, spans=spans)
        if confidence:
            matches[document_type] = confidence
    if len(matches) != 1:
        return "unknown", None, {"document_type_ambiguous" if matches else "document_type_unknown"}
    document_type, confidences = next(iter(matches.items()))
    # The classification confidence is derived only from title OCR confidence,
    # not an invented score from a keyword heuristic.
    return document_type, min(confidences), set()


@dataclass(frozen=True, slots=True)
class _FieldRule:
    """One fixed visible label vocabulary for the source-attested v1 forms.

    These are candidate-side labels only.  They do not read a source PDF,
    manifest, JSON Gold, CSV, or another document.  Their field identifiers
    intentionally match the final-content-stream Gold vocabulary so field EM
    compares the same semantic slot rather than two unrelated UI names.
    """

    field_id: str
    labels: tuple[str, ...]


_FIELD_RULES: dict[RasterDocumentType, tuple[_FieldRule, ...]] = {
    "bill_of_lading": (
        _FieldRule("exporter_name", ("SHIPPER", "EXPORTER")),
        _FieldRule("consignee", ("CONSIGNEE", "TO APPLICANT / CONSIGNEE")),
        _FieldRule("load_port", ("PORT OF LOADING", "LOAD PORT")),
        _FieldRule("disch_port", ("PORT OF DISCHARGE", "DISCHARGE PORT")),
        _FieldRule("final_dest", ("FINAL DESTINATION",)),
        _FieldRule("vessel", ("VESSEL / FLIGHT", "VESSEL/FLIGHT", "VESSEL")),
        _FieldRule("voyage", ("VOYAGE NO", "VOYAGE")),
        _FieldRule("bl_no", ("BILL OF LADING NO", "B/L NO", "BL NO")),
        _FieldRule("container_no", ("CONTAINER NO", "CONTAINER NUMBER")),
        _FieldRule("seal_no", ("SEAL NO", "SEAL NUMBER")),
        _FieldRule("good", ("GOODS DESCRIPTION", "DESCRIPTION OF GOODS")),
        _FieldRule("gross_weight", ("GROSS WEIGHT",)),
        _FieldRule("containers", ("NO OF CONTAINERS", "NUMBER OF CONTAINERS", "CONTAINERS")),
    ),
    "commercial_invoice": (
        _FieldRule("exporter_name", ("EXPORTER", "SELLER", "SHIPPER")),
        _FieldRule("importer_name", ("NOTIFY PARTY", "IMPORTER", "BUYER")),
        _FieldRule("consignee", ("CONSIGNEE", "TO APPLICANT / CONSIGNEE")),
        _FieldRule("load_port", ("PORT OF LOADING", "LOAD PORT")),
        _FieldRule("disch_port", ("PORT OF DISCHARGE", "DISCHARGE PORT")),
        _FieldRule("vessel", ("VESSEL / FLIGHT", "VESSEL/FLIGHT", "VESSEL")),
        _FieldRule("inv_no", ("INVOICE NUMBER", "INVOICE NO", "INV NO")),
        _FieldRule("inv_date", ("INVOICE DATE",)),
        _FieldRule("lc_no", ("LETTER OF CREDIT", "L/C NO", "LC NO")),
        _FieldRule("good", ("GOODS DESCRIPTION", "DESCRIPTION OF GOODS")),
        _FieldRule("unit", ("UNIT",)),
        _FieldRule("qty", ("QUANTITY", "QTY")),
        _FieldRule("unit_price", ("UNIT PRICE",)),
        _FieldRule("amount", ("AMOUNT",)),
        _FieldRule("total", ("TOTAL",)),
    ),
    "packing_list": (
        _FieldRule("exporter_name", ("EXPORTER", "SELLER", "SHIPPER")),
        _FieldRule("consignee", ("CONSIGNEE", "TO APPLICANT / CONSIGNEE")),
        _FieldRule("inv_no", ("NO OF INVOICE", "INVOICE NO", "INV NO")),
        _FieldRule("lc_no", ("NO OF L/C", "LETTER OF CREDIT", "L/C NO", "LC NO")),
        _FieldRule("good", ("GOODS DESCRIPTION", "DESCRIPTION OF GOODS")),
        _FieldRule("unit", ("UNIT",)),
        _FieldRule("qty", ("QUANTITY", "QTY")),
        _FieldRule("cartons", ("NO OF CARTONS / PKGS", "CARTONS", "PKGS")),
        _FieldRule("net_weight", ("NET WEIGHT",)),
        _FieldRule("gross_weight", ("GROSS WEIGHT",)),
        _FieldRule("cbm", ("CBM", "VOLUME", "MEASUREMENT")),
        _FieldRule("net_gross_weight", ("NET/GROSS WEIGHT",)),
    ),
    "unknown": (),
}


_KnownFormFieldAxis = Literal["right", "below"]


@dataclass(frozen=True, slots=True)
class _KnownFormFieldSlot:
    """One candidate-visible label/value slot on a supported fixed form.

    The slot is intentionally much narrower than the legacy generic mapper:
    its value must occupy the first observed right-hand or below-label slot.
    Later OCR spans are outside that slot, even when they happen to remain in
    the old broad geometry window.  This is a fixed-form layout rule, not a
    text-similarity or nearest-neighbour guess.
    """

    field_id: str
    axes: tuple[_KnownFormFieldAxis, ...]


def _known_form_field_slots(
    document_type: RasterDocumentType,
) -> tuple[_KnownFormFieldSlot, ...]:
    """Return only local, visible-label slots for the three v1 forms.

    Keep this separate from the evaluator/source templates.  The field names
    and label vocabularies already exposed by ``_FIELD_RULES`` are the public
    candidate vocabulary; this function merely makes the physical relation
    between each observed label and its immediate value slot explicit.
    """

    # The forms use both label-to-right summary blocks and label-to-below
    # tabular blocks.  Each listed axis is a permitted *fixed slot* relation,
    # not a fallback to an arbitrary nearby OCR span.  When both relations
    # have independently observed values, the result is ambiguous and is
    # deliberately withheld for review.
    return tuple(
        _KnownFormFieldSlot(rule.field_id, ("right", "below"))
        for rule in _FIELD_RULES[document_type]
    )


# Keep this byte-for-byte semantic rule aligned with the existing
# native-PDF candidate projector: quantities split only when the whole visible
# value is a number, or one ASCII number followed by one ASCII unit.  Finding
# a number somewhere inside arbitrary OCR text would manufacture evidence and
# make a failed raster candidate look complete.
_QUANTITY_TOKEN = r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
_QUANTITY_WITH_UNIT = re.compile(rf"^\s*(?P<quantity>{_QUANTITY_TOKEN})\s+(?P<unit>[A-Za-z]+)\s*$")
_QUANTITY_ONLY = re.compile(rf"^\s*(?P<quantity>{_QUANTITY_TOKEN})\s*$")
_UNIT_ONLY = re.compile(r"^\s*(?P<unit>[A-Za-z]+)\s*$")
_SPAN_ID_PAGE_NUMBER = re.compile(r"^p(?P<page>[1-9]\d*)-s\d+$")
_FIELD_SAME_ROW_Y_DELTA = 0.020
_FIELD_MAX_RIGHT_GAP = 0.550
_FIELD_MAX_BELOW_GAP = 0.100
_FIELD_BELOW_X_DELTA = 0.100
_FIELD_POSITION_EPSILON = 0.000_001
# Known-form slots deliberately use a smaller boundary than the historical
# generic mapper.  A later span in a broad text block must never win merely
# because it still fits the generic label/value window.
_KNOWN_FIELD_MAX_RIGHT_GAP = 0.250
_KNOWN_FIELD_MAX_BELOW_GAP = 0.125
_KNOWN_FIELD_SAME_ROW_CENTER_DELTA = 0.020
_KNOWN_FIELD_MIN_COLUMN_OVERLAP = 0.600
# A blank gap larger than this establishes a distinct provider-observed slot.
# Text within one slot is never concatenated in v1: multiple spans mean the
# candidate evidence is not uniquely one field value and must go to review.
_KNOWN_FIELD_SLOT_BOUNDARY_GAP = 0.025


def _normalized_text(value: str) -> str:
    return " ".join(value.split())


def _normalized_visible_label(value: str) -> str:
    """Normalize typography for matching a fixed visible candidate label."""

    normalized = re.sub(r"[^a-z0-9/]+", " ", value.casefold())
    normalized = re.sub(r"\s*/\s*", "/", normalized)
    return " ".join(normalized.split())


def _union_bbox(boxes: Sequence[BoundingBoxV1 | None]) -> BoundingBoxV1 | None:
    if not boxes or any(box is None for box in boxes):
        return None
    concrete = [box for box in boxes if box is not None]
    left = min(box.x for box in concrete)
    top = min(box.y for box in concrete)
    right = max(box.x + box.width for box in concrete)
    bottom = max(box.y + box.height for box in concrete)
    try:
        return BoundingBoxV1(x=left, y=top, width=right - left, height=bottom - top)
    except ValidationError:
        return None


def _is_visible_label(value: str, *, rules: Sequence[_FieldRule]) -> bool:
    normalized = _normalized_visible_label(value)
    return any(
        normalized == _normalized_visible_label(alias) for rule in rules for alias in rule.labels
    )


def _is_inline_visible_field(value: str) -> bool:
    """Keep another explicit inline field from becoming a nearby value."""

    return any(
        _inline_rule_value(rule, value) is not None
        for rules in _FIELD_RULES.values()
        for rule in rules
    )


def _inline_rule_value(rule: _FieldRule, value: str) -> str | None:
    """Read one explicit ``label: value`` OCR span without broad guessing."""

    aliases = "|".join(re.escape(alias) for alias in sorted(rule.labels, key=len, reverse=True))
    match = re.fullmatch(
        rf"\s*(?:{aliases})\s*(?::|#|-)\s*(?P<value>.+?)\s*",
        value,
        re.IGNORECASE,
    )
    return (
        match.group("value").strip() if match is not None and match.group("value").strip() else None
    )


def _adjacent_label_value(label: RasterDocumentSpanV2, value: RasterDocumentSpanV2) -> bool:
    """Require one same-row-right or same-column-below candidate value."""

    label_page = _span_page_number(label)
    value_page = _span_page_number(value)
    if (
        label.bbox is None
        or value.bbox is None
        or label_page is None
        or value_page is None
        or label_page != value_page
        or value.reading_order <= label.reading_order
    ):
        return False
    label_box = label.bbox
    value_box = value.bbox
    same_row = (
        value_box.x + _FIELD_POSITION_EPSILON >= label_box.x + label_box.width
        and value_box.x - (label_box.x + label_box.width) <= _FIELD_MAX_RIGHT_GAP
        and abs((value_box.y + value_box.height / 2) - (label_box.y + label_box.height / 2))
        <= max(_FIELD_SAME_ROW_Y_DELTA, label_box.height, value_box.height)
    )
    below = (
        value_box.y + _FIELD_POSITION_EPSILON >= label_box.y + label_box.height
        and value_box.y - (label_box.y + label_box.height) <= _FIELD_MAX_BELOW_GAP
        and abs(value_box.x - label_box.x) <= _FIELD_BELOW_X_DELTA
    )
    return same_row or below


def _span_page_number(span: RasterDocumentSpanV2) -> int | None:
    """Read the page encoded by this sidecar's locally generated span ID.

    ``RasterDocumentSpanV2`` deliberately keeps page ownership on its parent
    page object.  The sidecar creates IDs as ``p{page}-s{ordinal}``, so this
    small parser retains the no-cross-page field relation without changing the
    public v2 schema.  Unexpected IDs simply cannot form field evidence.
    """

    match = _SPAN_ID_PAGE_NUMBER.fullmatch(span.span_id)
    return int(match.group("page")) if match is not None else None


_FieldMappingStatus = Literal["not_observed", "accepted", "incomplete", "ambiguous"]


@dataclass(frozen=True, slots=True)
class _FieldMappingDecision:
    """One field value decision with its exact observed OCR evidence."""

    status: _FieldMappingStatus
    value: str | None = None
    evidence: tuple[RasterDocumentSpanV2, ...] = ()


def _contiguous_label_evidence(
    rule: _FieldRule,
    spans: Sequence[RasterDocumentSpanV2],
) -> tuple[tuple[RasterDocumentSpanV2, ...], ...]:
    """Find maximal one-to-three span exact visible-label joins.

    PP-Structure may emit a printed label as several immediately adjacent OCR
    spans.  We admit that bounded join only when its normalized rendering is
    exactly one local alias.  A shorter overlapping alias is discarded in
    favour of the maximal observed label, so e.g. a one-word alias cannot
    manufacture a duplicate next to its longer printed form.
    """

    aliases = {_normalized_visible_label(alias) for alias in rule.labels}
    matches: dict[tuple[str, ...], tuple[RasterDocumentSpanV2, ...]] = {}
    for start in range(len(spans)):
        evidence: list[RasterDocumentSpanV2] = []
        for end in range(start, min(len(spans), start + 3)):
            span = spans[end]
            if evidence:
                previous = evidence[-1]
                if (
                    _span_page_number(span) != _span_page_number(previous)
                    or span.reading_order != previous.reading_order + 1
                ):
                    break
            evidence.append(span)
            rendered = _normalized_visible_label(" ".join(item.text for item in evidence))
            if rendered in aliases:
                frozen = tuple(evidence)
                matches[tuple(item.span_id for item in frozen)] = frozen
    all_matches = tuple(matches.values())
    maximal: list[tuple[RasterDocumentSpanV2, ...]] = []
    for matched_evidence in all_matches:
        evidence_ids = {item.span_id for item in matched_evidence}
        if any(evidence_ids < {item.span_id for item in other} for other in all_matches):
            continue
        maximal.append(matched_evidence)
    return tuple(maximal)


def _evidence_bbox(
    evidence: Sequence[RasterDocumentSpanV2],
) -> BoundingBoxV1 | None:
    return _union_bbox([span.bbox for span in evidence])


def _known_field_value_candidates(
    *,
    label_evidence: Sequence[RasterDocumentSpanV2],
    spans: Sequence[RasterDocumentSpanV2],
    label_span_ids: set[str],
) -> list[RasterDocumentSpanV2]:
    """Return non-label spans eligible for a single observed field slot."""

    label_page = _span_page_number(label_evidence[0]) if label_evidence else None
    last_order = label_evidence[-1].reading_order if label_evidence else -1
    return [
        span
        for span in spans
        if span.span_id not in label_span_ids
        and not _is_visible_label(
            span.text,
            rules=(
                _FIELD_RULES["bill_of_lading"]
                + _FIELD_RULES["commercial_invoice"]
                + _FIELD_RULES["packing_list"]
            ),
        )
        and not _is_inline_visible_field(span.text)
        and _span_page_number(span) == label_page
        and span.reading_order > last_order
        and span.bbox is not None
    ]


def _right_hand_field_slot(
    *,
    label_box: BoundingBoxV1,
    candidates: Sequence[RasterDocumentSpanV2],
    label_barriers: Sequence[RasterDocumentSpanV2],
) -> _FieldMappingDecision:
    """Read one bounded, label-free right-hand slot without ranking text.

    A fixed-form slot starts at the observed label's right edge and ends at
    the first visible label barrier or the first blank provider-layout gap.
    Candidates are not ranked by text or confidence.  Every span inside that
    one physical slot is inspected; zero or multiple spans is fail-closed.
    """

    barrier_left = min(
        (
            span.bbox.x
            for span in label_barriers
            if span.bbox is not None
            and span.bbox.x + _FIELD_POSITION_EPSILON >= label_box.x + label_box.width
            and abs((span.bbox.y + span.bbox.height / 2) - (label_box.y + label_box.height / 2))
            <= max(_KNOWN_FIELD_SAME_ROW_CENTER_DELTA, label_box.height, span.bbox.height)
        ),
        default=None,
    )
    eligible = [
        span
        for span in candidates
        if span.bbox is not None
        and span.bbox.x + _FIELD_POSITION_EPSILON >= label_box.x + label_box.width
        and span.bbox.x - (label_box.x + label_box.width) <= _KNOWN_FIELD_MAX_RIGHT_GAP
        and abs((span.bbox.y + span.bbox.height / 2) - (label_box.y + label_box.height / 2))
        <= max(_KNOWN_FIELD_SAME_ROW_CENTER_DELTA, label_box.height, span.bbox.height)
        and (
            barrier_left is None
            or span.bbox.x + span.bbox.width <= barrier_left + _FIELD_POSITION_EPSILON
        )
    ]
    if not eligible:
        return _FieldMappingDecision("incomplete")
    slot: list[RasterDocumentSpanV2] = []
    right_edge = label_box.x + label_box.width
    for span in sorted(eligible, key=lambda item: item.bbox.x if item.bbox is not None else 1.0):
        assert span.bbox is not None
        gap = span.bbox.x - right_edge
        if not slot:
            if gap > _KNOWN_FIELD_MAX_RIGHT_GAP:
                break
        elif gap > _KNOWN_FIELD_SLOT_BOUNDARY_GAP:
            break
        slot.append(span)
        right_edge = max(right_edge, span.bbox.x + span.bbox.width)
    if not slot:
        return _FieldMappingDecision("incomplete")
    if len(slot) != 1:
        return _FieldMappingDecision("ambiguous")
    selected = slot[0]
    return _FieldMappingDecision("accepted", selected.text.strip(), (selected,))


def _below_field_slot(
    *,
    label_box: BoundingBoxV1,
    candidates: Sequence[RasterDocumentSpanV2],
    label_barriers: Sequence[RasterDocumentSpanV2],
) -> _FieldMappingDecision:
    """Read one bounded, label-free below-label slot without ranking text."""

    barrier_top = min(
        (
            span.bbox.y
            for span in label_barriers
            if span.bbox is not None
            and span.bbox.y + _FIELD_POSITION_EPSILON >= label_box.y + label_box.height
            and max(
                0.0,
                min(span.bbox.x + span.bbox.width, label_box.x + label_box.width)
                - max(span.bbox.x, label_box.x),
            )
            >= _KNOWN_FIELD_MIN_COLUMN_OVERLAP * min(span.bbox.width, label_box.width)
        ),
        default=None,
    )
    eligible: list[RasterDocumentSpanV2] = []
    for span in candidates:
        if span.bbox is None:
            continue
        if (
            span.bbox.y + _FIELD_POSITION_EPSILON < label_box.y + label_box.height
            or span.bbox.y - (label_box.y + label_box.height) > _KNOWN_FIELD_MAX_BELOW_GAP
        ):
            continue
        overlap = max(
            0.0,
            min(span.bbox.x + span.bbox.width, label_box.x + label_box.width)
            - max(span.bbox.x, label_box.x),
        )
        if overlap < _KNOWN_FIELD_MIN_COLUMN_OVERLAP * min(span.bbox.width, label_box.width):
            continue
        if (
            barrier_top is not None
            and span.bbox.y + span.bbox.height > barrier_top + _FIELD_POSITION_EPSILON
        ):
            continue
        eligible.append(span)
    if not eligible:
        return _FieldMappingDecision("incomplete")
    slot: list[RasterDocumentSpanV2] = []
    bottom_edge = label_box.y + label_box.height
    for span in sorted(eligible, key=lambda item: item.bbox.y if item.bbox is not None else 1.0):
        assert span.bbox is not None
        gap = span.bbox.y - bottom_edge
        if not slot:
            if gap > _KNOWN_FIELD_MAX_BELOW_GAP:
                break
        elif gap > _KNOWN_FIELD_SLOT_BOUNDARY_GAP:
            break
        slot.append(span)
        bottom_edge = max(bottom_edge, span.bbox.y + span.bbox.height)
    if not slot:
        return _FieldMappingDecision("incomplete")
    if len(slot) != 1:
        return _FieldMappingDecision("ambiguous")
    selected = slot[0]
    return _FieldMappingDecision("accepted", selected.text.strip(), (selected,))


def _known_form_field_decision(
    *,
    rule: _FieldRule,
    slot: _KnownFormFieldSlot,
    spans: Sequence[RasterDocumentSpanV2],
    label_span_ids: set[str],
    label_barriers: Sequence[RasterDocumentSpanV2],
) -> _FieldMappingDecision:
    """Map one known-form field only when one physical slot is unique."""

    inline_matches = [
        (value, (span,))
        for span in spans
        if (value := _inline_rule_value(rule, span.text)) is not None
    ]
    labels = _contiguous_label_evidence(rule, spans)
    if len(inline_matches) > 1 or (inline_matches and labels):
        return _FieldMappingDecision("ambiguous")
    if len(inline_matches) == 1:
        value, evidence = inline_matches[0]
        return _FieldMappingDecision("accepted", value, evidence)
    if not labels:
        return _FieldMappingDecision("not_observed")
    if len(labels) != 1:
        return _FieldMappingDecision("ambiguous")
    label = labels[0]
    label_box = _evidence_bbox(label)
    if label_box is None:
        return _FieldMappingDecision("incomplete")
    candidates = _known_field_value_candidates(
        label_evidence=label,
        spans=spans,
        label_span_ids=label_span_ids,
    )
    decisions: list[_FieldMappingDecision] = []
    for axis in slot.axes:
        decision = (
            _right_hand_field_slot(
                label_box=label_box,
                candidates=candidates,
                label_barriers=label_barriers,
            )
            if axis == "right"
            else _below_field_slot(
                label_box=label_box,
                candidates=candidates,
                label_barriers=label_barriers,
            )
        )
        if decision.status == "ambiguous":
            return decision
        if decision.status == "accepted":
            decisions.append(decision)
    if not decisions:
        return _FieldMappingDecision("incomplete")
    if len(decisions) != 1:
        return _FieldMappingDecision("ambiguous")
    value = decisions[0].value
    value_evidence = decisions[0].evidence
    if value is None or not value:
        return _FieldMappingDecision("incomplete")
    return _FieldMappingDecision("accepted", value, (*label, *value_evidence))


def _generic_field_decision(
    *,
    rule: _FieldRule,
    rules: Sequence[_FieldRule],
    spans: Sequence[RasterDocumentSpanV2],
) -> _FieldMappingDecision:
    """Preserve the historical broad mapper for non-observed known slots."""

    matches: list[tuple[str, tuple[RasterDocumentSpanV2, ...]]] = []
    aliases = {_normalized_visible_label(alias) for alias in rule.labels}
    observed_label = False
    for span in spans:
        inline_value = _inline_rule_value(rule, span.text)
        if inline_value is not None:
            matches.append((inline_value, (span,)))
            observed_label = True
            continue
        if _normalized_visible_label(span.text) not in aliases:
            continue
        observed_label = True
        value_spans = [
            value_span
            for value_span in spans
            if value_span.span_id != span.span_id
            and not _is_visible_label(value_span.text, rules=rules)
            and _adjacent_label_value(span, value_span)
        ]
        if len(value_spans) == 1:
            matches.append((value_spans[0].text.strip(), (span, value_spans[0])))
        elif len(value_spans) > 1:
            return _FieldMappingDecision("ambiguous")
    unique = {
        (value, tuple(item.span_id for item in evidence)): (value, evidence)
        for value, evidence in matches
        if value
    }
    if len(unique) > 1:
        return _FieldMappingDecision("ambiguous")
    if not unique:
        return _FieldMappingDecision("incomplete" if observed_label else "not_observed")
    value, evidence = next(iter(unique.values()))
    return _FieldMappingDecision("accepted", value, evidence)


def _append_projected_field_value(
    *,
    candidates: dict[str, list[tuple[str, tuple[RasterDocumentSpanV2, ...]]]],
    field_id: str,
    value: str,
    evidence: tuple[RasterDocumentSpanV2, ...],
) -> None:
    """Apply only source-template-compatible, visible value transforms."""

    stripped = value.strip()
    if not stripped:
        return
    if field_id == "qty":
        quantity_with_unit = _QUANTITY_WITH_UNIT.fullmatch(stripped)
        if quantity_with_unit is not None:
            candidates.setdefault("qty", []).append(
                (quantity_with_unit.group("quantity"), evidence)
            )
            candidates.setdefault("unit", []).append((quantity_with_unit.group("unit"), evidence))
            return
        quantity_only = _QUANTITY_ONLY.fullmatch(stripped)
        if quantity_only is not None:
            candidates.setdefault("qty", []).append((quantity_only.group("quantity"), evidence))
        return
    if field_id == "unit":
        unit_only = _UNIT_ONLY.fullmatch(stripped)
        if unit_only is not None:
            candidates.setdefault("unit", []).append((unit_only.group("unit"), evidence))
        return
    candidates.setdefault(field_id, []).append((stripped, evidence))


def _field_candidates(
    document_type: RasterDocumentType,
    spans: Sequence[RasterDocumentSpanV2],
) -> tuple[list[RasterFieldCandidateV2], set[str]]:
    rules = _FIELD_RULES[document_type]
    slots = {slot.field_id: slot for slot in _known_form_field_slots(document_type)}
    observed_labels = {
        span.span_id
        for rule in rules
        for evidence in _contiguous_label_evidence(rule, spans)
        for span in evidence
    }
    label_barriers = tuple(span for span in spans if span.span_id in observed_labels)
    projected: dict[str, list[tuple[str, tuple[RasterDocumentSpanV2, ...]]]] = {}
    reasons: set[str] = set()
    for rule in rules:
        known_slot = slots.get(rule.field_id)
        known = (
            _known_form_field_decision(
                rule=rule,
                slot=known_slot,
                spans=spans,
                label_span_ids=observed_labels,
                label_barriers=label_barriers,
            )
            if known_slot is not None
            else _FieldMappingDecision("not_observed")
        )
        if known.status == "accepted":
            value = known.value
            evidence = known.evidence
        elif known.status in {"ambiguous", "incomplete"}:
            reasons.add(
                "field_mapping_ambiguous"
                if known.status == "ambiguous"
                else "field_mapping_incomplete"
            )
            continue
        else:
            # Unknown labels and legacy configurations retain the previous
            # broad mapping behaviour.  Once a known-form label is observed,
            # however, its stricter slot result wins and cannot fall back to a
            # looser nearby-text guess.
            generic = _generic_field_decision(rule=rule, rules=rules, spans=spans)
            if generic.status != "accepted":
                if generic.status == "ambiguous":
                    reasons.add("field_mapping_ambiguous")
                continue
            value = generic.value
            evidence = generic.evidence
        if value is None:
            reasons.add("field_mapping_incomplete")
            continue
        _append_projected_field_value(
            candidates=projected, field_id=rule.field_id, value=value, evidence=evidence
        )
    # Packing List source truth has a direct joined semantic field.  If the
    # visible form exposes only the two unambiguous component values, reproduce
    # the fixed source transform rather than inventing a new semantic value.
    if (
        "net_gross_weight" not in projected
        and len(projected.get("net_weight", ())) == 1
        and len(projected.get("gross_weight", ())) == 1
    ):
        net_value, net_evidence = projected["net_weight"][0]
        gross_value, gross_evidence = projected["gross_weight"][0]
        projected["net_gross_weight"] = [
            (f"{net_value} / {gross_value}", (*net_evidence, *gross_evidence))
        ]

    output: list[RasterFieldCandidateV2] = []
    for rule in rules:
        values = projected.get(rule.field_id, ())
        if len(values) != 1:
            reasons.add(
                "field_mapping_ambiguous" if len(values) > 1 else "field_mapping_incomplete"
            )
            continue
        value, evidence = values[0]
        bbox = _union_bbox([span.bbox for span in evidence])
        if bbox is None:
            reasons.add("layout_incomplete")
            continue
        output.append(
            RasterFieldCandidateV2(
                field_id=rule.field_id,
                value=value,
                confidence=min(span.confidence for span in evidence),
                evidence_span_ids=[span.span_id for span in evidence],
                bbox=bbox,
            )
        )
    return output, reasons


@dataclass(frozen=True, slots=True)
class _TableProjectionAttempt:
    """One deterministic, content-free outcome of a native table projection."""

    table: RasterTableV2 | None
    status: _TableProjectionStatus


@dataclass(frozen=True, slots=True)
class _KnownFormTableSpec:
    """Visible, candidate-only header contract for one supported form."""

    header_aliases: tuple[tuple[str, ...], ...]


# These are local, fixed visible form headers.  They deliberately do not
# consume a PDF content stream, Gold, CSV, JSON fixture, or OCR/table-recognizer
# output.  The dimensions are part of the v1 support boundary: B/L is 2x6 and
# Invoice/Packing List are 2x5 (header plus exactly one body row).
_KNOWN_FORM_TABLE_SPECS: dict[
    Literal["bill_of_lading", "commercial_invoice", "packing_list"], _KnownFormTableSpec
] = {
    "bill_of_lading": _KnownFormTableSpec(
        header_aliases=(
            ("container no", "container number"),
            ("seal no", "seal number"),
            (
                "no & kind of packages/containers",
                "no and kind of packages/containers",
                "no & kind of packages containers",
                "no and kind of packages containers",
                "no kind of packages/containers",
                "no of packages/containers",
                "packages/containers",
            ),
            ("goods description", "description of goods", "description"),
            ("gross weight",),
            ("measurement", "measurement cbm", "cbm", "volume"),
        )
    ),
    "commercial_invoice": _KnownFormTableSpec(
        header_aliases=(
            (
                "marks/no",
                "marks no",
                "marks/nos",
                "marks nos",
                "marks/numbers",
                "marks numbers",
            ),
            ("description of goods", "goods description", "description"),
            ("quantity", "qty", "quantity unit", "quantity/unit"),
            ("unit price",),
            ("amount",),
        )
    ),
    "packing_list": _KnownFormTableSpec(
        header_aliases=(
            (
                "marks/numbers/pkgs",
                "marks numbers pkgs",
                "marks numbers packages",
                "cartons",
            ),
            (
                "goods description",
                "description of goods",
                "goods description/quantity",
                "description of goods/quantity",
                "description",
            ),
            ("net weight",),
            ("gross weight",),
            ("measurement", "measurement cbm", "cbm", "volume"),
        )
    ),
}

_KNOWN_FORM_MIN_AXIS_OVERLAP = 0.80
_KNOWN_FORM_MAX_CELL_OVERLAP = 0.10


@dataclass(frozen=True, slots=True)
class _NativeTableCell:
    """One provider-supplied native cell rectangle, never synthesized."""

    index: int
    pixel_bbox: _PixelBox


@dataclass(frozen=True, slots=True)
class _NativeTableCellEvidence:
    """One observed cell and the exact overall-OCR spans it contains."""

    cell: _NativeTableCell
    spans: tuple[RasterDocumentSpanV2, ...]


@dataclass(frozen=True, slots=True)
class _KnownFormTableTarget:
    """One complete, uniquely evidenced native header/body subgrid."""

    header: tuple[_NativeTableCellEvidence, ...]
    body: tuple[_NativeTableCellEvidence, ...]


def _native_table_cells(
    native_table: Mapping[str, object], *, page: RasterDocumentPreviewPageV2
) -> tuple[list[_NativeTableCell], _TableProjectionStatus | None]:
    """Read actual PP-Structure cell rectangles without table-OCR fallback."""

    values = _as_list(native_table.get("cell_box_list"))
    if values is None or not values:
        return [], "missing_cell_geometry"
    cells: list[_NativeTableCell] = []
    for index, value in enumerate(values):
        box = _pixel_box_from_rectangle(value, width=page.width, height=page.height)
        if box is None:
            return [], "invalid_cell_geometry"
        cells.append(_NativeTableCell(index=index, pixel_bbox=box))
    return cells, None


def _span_center_in_cell(
    *, span: RasterDocumentSpanV2, cell: _NativeTableCell, page: RasterDocumentPreviewPageV2
) -> bool:
    if span.bbox is None:
        return False
    return cell.pixel_bbox.contains_center(
        (
            (span.bbox.x + span.bbox.width / 2) * page.width,
            (span.bbox.y + span.bbox.height / 2) * page.height,
        )
    )


def _spans_in_native_cell(
    *, cell: _NativeTableCell, page: RasterDocumentPreviewPageV2
) -> tuple[RasterDocumentSpanV2, ...]:
    return tuple(
        sorted(
            (span for span in page.spans if _span_center_in_cell(span=span, cell=cell, page=page)),
            key=lambda span: (span.reading_order, span.ordinal, span.span_id),
        )
    )


def _contiguous_header_evidence_matches(
    *, spans: Sequence[RasterDocumentSpanV2], aliases: Sequence[str]
) -> tuple[tuple[RasterDocumentSpanV2, ...], ...]:
    """Match bounded exact joins in one already-established span order."""

    normalized_aliases = {_normalized_visible_label(alias) for alias in aliases}
    matches: dict[tuple[str, ...], tuple[RasterDocumentSpanV2, ...]] = {}
    # A native OCR span can hold the whole visible header.  Some detectors split
    # a header across two or three adjacent spans, so admit only bounded,
    # contiguous exact joins from the same native cell.
    for start in range(len(spans)):
        for end in range(start + 1, min(len(spans), start + 3) + 1):
            evidence = tuple(spans[start:end])
            rendered = _normalized_visible_label(" ".join(span.text for span in evidence))
            if rendered in normalized_aliases:
                matches[tuple(span.span_id for span in evidence)] = evidence
    return tuple(matches.values())


def _one_row_visual_header_order(
    spans: Sequence[RasterDocumentSpanV2],
) -> tuple[RasterDocumentSpanV2, ...] | None:
    """Return a strictly visual order for one native-cell header row only.

    PP-Structure may omit a document-wide reading order.  That must not alter
    the ledger's provider order, but a fixed-form header split across adjacent
    spans in one already-observed native cell can be checked in its unambiguous
    visual order.  This helper neither repairs OCR text nor creates geometry:
    every participating span needs a box, all boxes must form one
    non-overlapping row, and ties/stacked spans are rejected.
    """

    ordered: list[tuple[_PixelBox, RasterDocumentSpanV2]] = []
    for span in spans:
        bbox = span.bbox
        if bbox is None:
            return None
        pixel_box = _PixelBox(
            left=bbox.x,
            top=bbox.y,
            right=bbox.x + bbox.width,
            bottom=bbox.y + bbox.height,
        )
        if pixel_box.width <= 0 or pixel_box.height <= 0:
            return None
        ordered.append((pixel_box, span))
    ordered.sort(
        key=lambda item: (
            item[0].left,
            item[0].top,
            item[0].right,
            item[0].bottom,
            item[1].span_id,
        )
    )
    boxes = [item[0] for item in ordered]
    if not _boxes_form_one_row(boxes):
        return None
    return tuple(item[1] for item in ordered)


def _header_evidence_matches(
    *, spans: Sequence[RasterDocumentSpanV2], aliases: Sequence[str]
) -> tuple[tuple[RasterDocumentSpanV2, ...], ...]:
    """Match exact local aliases from overall OCR spans, never table OCR text.

    The provider order remains the first and preferred evidence order.  Only
    when it yields no exact alias do we inspect the same native-cell spans in
    a proven single visual row.  This narrow fallback avoids treating a
    missing provider reading order as a reason to concatenate arbitrary text.
    """

    provider_matches = _contiguous_header_evidence_matches(spans=spans, aliases=aliases)
    if provider_matches:
        return provider_matches
    visual_spans = _one_row_visual_header_order(spans)
    if visual_spans is None:
        return ()
    return _contiguous_header_evidence_matches(spans=visual_spans, aliases=aliases)


def _boxes_form_one_row(boxes: Sequence[_PixelBox]) -> bool:
    """Require one observed non-overlapping row; do not infer a grid from order."""

    if not boxes:
        return False
    vertical_overlap = min(box.bottom for box in boxes) - max(box.top for box in boxes)
    if vertical_overlap < _KNOWN_FORM_MIN_AXIS_OVERLAP * min(box.height for box in boxes):
        return False
    for prior, current in zip(boxes, boxes[1:]):
        if current.center[0] <= prior.center[0]:
            return False
        horizontal_overlap = max(
            0.0, min(prior.right, current.right) - max(prior.left, current.left)
        )
        if horizontal_overlap > _KNOWN_FORM_MAX_CELL_OVERLAP * min(prior.width, current.width):
            return False
    return True


def _known_form_header(
    *,
    spec: _KnownFormTableSpec,
    cells: Sequence[_NativeTableCell],
    page: RasterDocumentPreviewPageV2,
) -> tuple[tuple[_NativeTableCellEvidence, ...] | None, _TableProjectionStatus]:
    """Bind every header semantic to one and only one cell/span evidence set."""

    candidates: list[list[_NativeTableCellEvidence]] = [[] for _aliases in spec.header_aliases]
    for cell in cells:
        spans = _spans_in_native_cell(cell=cell, page=page)
        semantic_matches: list[tuple[int, tuple[RasterDocumentSpanV2, ...]]] = []
        for column_index, aliases in enumerate(spec.header_aliases):
            matches = _header_evidence_matches(spans=spans, aliases=aliases)
            if len(matches) > 1:
                return None, "duplicate_header_evidence"
            if matches:
                semantic_matches.append((column_index, matches[0]))
        if len(semantic_matches) > 1:
            # A merged/provider-overlapping cell cannot safely be assigned to
            # multiple target columns.
            return None, "duplicate_header_evidence"
        if semantic_matches:
            column_index, evidence = semantic_matches[0]
            candidates[column_index].append(_NativeTableCellEvidence(cell=cell, spans=evidence))
    if any(not values for values in candidates):
        return None, "missing_header_evidence"
    if any(len(values) != 1 for values in candidates):
        return None, "duplicate_header_evidence"
    selected = tuple(values[0] for values in candidates)
    evidence_ids = [span.span_id for item in selected for span in item.spans]
    if len(evidence_ids) != len(set(evidence_ids)):
        return None, "duplicate_header_evidence"
    if not _boxes_form_one_row([item.cell.pixel_bbox for item in selected]):
        return None, "header_grid_incomplete"
    return selected, "accepted"


def _aligned_body_columns(
    *, body: _PixelBox, header: Sequence[_NativeTableCellEvidence]
) -> list[int]:
    """Find native columns whose observed header box aligns with this body cell."""

    aligned: list[int] = []
    header_bottom = max(item.cell.pixel_bbox.bottom for item in header)
    tolerance = max(1.0, 0.20 * min(item.cell.pixel_bbox.height for item in header))
    if body.top + tolerance < header_bottom:
        return aligned
    for index, item in enumerate(header):
        header_box = item.cell.pixel_bbox
        overlap = max(0.0, min(body.right, header_box.right) - max(body.left, header_box.left))
        if overlap < _KNOWN_FORM_MIN_AXIS_OVERLAP * min(body.width, header_box.width):
            continue
        if abs(body.center[0] - header_box.center[0]) > 0.25 * max(body.width, header_box.width):
            continue
        aligned.append(index)
    return aligned


def _body_row_groups(
    cells: Sequence[_NativeTableCell],
) -> list[list[_NativeTableCell]]:
    """Group only vertically-overlapping native candidates into observed rows."""

    groups: list[list[_NativeTableCell]] = []
    for cell in sorted(cells, key=lambda item: (item.pixel_bbox.center[1], item.pixel_bbox.top)):
        if not groups:
            groups.append([cell])
            continue
        prior = groups[-1]
        prior_top = min(item.pixel_bbox.top for item in prior)
        prior_bottom = max(item.pixel_bbox.bottom for item in prior)
        overlap = max(
            0.0, min(cell.pixel_bbox.bottom, prior_bottom) - max(cell.pixel_bbox.top, prior_top)
        )
        if overlap >= _KNOWN_FORM_MIN_AXIS_OVERLAP * min(
            cell.pixel_bbox.height, prior_bottom - prior_top
        ):
            prior.append(cell)
        else:
            groups.append([cell])
    return groups


def _complete_geometry_row(
    *,
    cells: Sequence[_NativeTableCell],
    expected_columns: int,
    page: RasterDocumentPreviewPageV2,
) -> tuple[_NativeTableCellEvidence, ...] | None:
    """Bind one observed full row without using its OCR text as a header key.

    This is intentionally narrower than a generic grid inference: a row must
    contain exactly the known form's number of provider cell boxes, be one
    non-overlapping visual row, and give every cell unique overall-OCR span
    evidence.  It never creates an empty cell or reads ``table_ocr_pred``.
    """

    if len(cells) != expected_columns:
        return None
    ordered = tuple(sorted(cells, key=lambda item: item.pixel_bbox.left))
    if not _boxes_form_one_row([item.pixel_bbox for item in ordered]):
        return None
    evidence = tuple(
        _NativeTableCellEvidence(
            cell=cell,
            spans=_spans_in_native_cell(cell=cell, page=page),
        )
        for cell in ordered
    )
    if any(not item.spans for item in evidence):
        return None
    span_ids = [span.span_id for item in evidence for span in item.spans]
    return evidence if len(span_ids) == len(set(span_ids)) else None


def _geometry_known_form_table_target(
    *,
    spec: _KnownFormTableSpec,
    cells: Sequence[_NativeTableCell],
    page: RasterDocumentPreviewPageV2,
) -> _KnownFormTableTarget | None:
    """Select one unique top-level known-form grid when header OCR is absent.

    A PP-Structure table can retain a valid header/body cell grid while its
    general OCR result misses a printed header alias.  Once the document type
    is already exact-title classified, the grid is still usable only when
    exactly one *adjacent* pair of rows supplies the fixed column count,
    observed cell geometry, one-to-one column alignment, and non-overlapping
    OCR evidence.  Footer fragments, merged rows, duplicate rows, or a
    second complete pair make the fallback unavailable rather than guessed.
    """

    expected_columns = len(spec.header_aliases)
    candidates: list[_KnownFormTableTarget] = []
    groups = _body_row_groups(cells)
    for header_cells, body_cells in zip(groups, groups[1:]):
        header = _complete_geometry_row(
            cells=header_cells,
            expected_columns=expected_columns,
            page=page,
        )
        body = _complete_geometry_row(
            cells=body_cells,
            expected_columns=expected_columns,
            page=page,
        )
        if header is None or body is None:
            continue
        if any(
            tuple(_aligned_body_columns(body=item.cell.pixel_bbox, header=header))
            != (column_index,)
            for column_index, item in enumerate(body)
        ):
            continue
        span_ids = [span.span_id for item in (*header, *body) for span in item.spans]
        if len(span_ids) != len(set(span_ids)):
            continue
        candidates.append(_KnownFormTableTarget(header=header, body=body))
    return candidates[0] if len(candidates) == 1 else None


def _known_form_body(
    *,
    header: tuple[_NativeTableCellEvidence, ...],
    cells: Sequence[_NativeTableCell],
    page: RasterDocumentPreviewPageV2,
) -> tuple[tuple[_NativeTableCellEvidence, ...] | None, _TableProjectionStatus]:
    """Find exactly one fully observed body row below the known header row."""

    header_indexes = {item.cell.index for item in header}
    body_like: list[_NativeTableCell] = []
    aligned_columns: dict[int, tuple[int, ...]] = {}
    for cell in cells:
        if cell.index in header_indexes:
            continue
        columns = tuple(_aligned_body_columns(body=cell.pixel_bbox, header=header))
        if columns:
            body_like.append(cell)
            aligned_columns[cell.index] = columns
    if not body_like:
        return None, "body_grid_incomplete"

    header_evidence_ids = {span.span_id for item in header for span in item.spans}
    targets: list[tuple[_NativeTableCellEvidence, ...]] = []
    observed_body_evidence_failure: _TableProjectionStatus | None = None
    for group in _body_row_groups(body_like):
        by_column: dict[int, list[_NativeTableCell]] = {}
        has_merged_candidate = False
        for cell in group:
            columns = aligned_columns[cell.index]
            if len(columns) != 1:
                has_merged_candidate = True
                continue
            by_column.setdefault(columns[0], []).append(cell)
        if has_merged_candidate:
            # A merged cell that overlaps target columns in the same observed
            # row makes the body geometry ambiguous.  Merged footer cells in a
            # different row remain irrelevant and are deliberately ignored.
            continue
        if set(by_column) != set(range(len(header))) or any(
            len(values) != 1 for values in by_column.values()
        ):
            continue
        ordered_cells = tuple(by_column[index][0] for index in range(len(header)))
        if not _boxes_form_one_row([item.pixel_bbox for item in ordered_cells]):
            continue
        evidence_cells: list[_NativeTableCellEvidence] = []
        for cell in ordered_cells:
            evidence = tuple(
                span
                for span in _spans_in_native_cell(cell=cell, page=page)
                if span.span_id not in header_evidence_ids
            )
            if not evidence:
                observed_body_evidence_failure = "missing_body_evidence"
                break
            evidence_cells.append(_NativeTableCellEvidence(cell=cell, spans=evidence))
        else:
            evidence_ids = [span.span_id for item in evidence_cells for span in item.spans]
            if len(evidence_ids) != len(set(evidence_ids)):
                observed_body_evidence_failure = "duplicate_body_evidence"
                continue
            targets.append(tuple(evidence_cells))
    if len(targets) == 1:
        return targets[0], "accepted"
    if len(targets) > 1:
        return None, "ambiguous_body_evidence"
    if observed_body_evidence_failure is not None:
        return None, observed_body_evidence_failure
    return None, "body_grid_incomplete"


def _known_form_table_target(
    *,
    document_type: RasterDocumentType,
    native_table: Mapping[str, object],
    page: RasterDocumentPreviewPageV2,
) -> tuple[_KnownFormTableTarget | None, _TableProjectionStatus]:
    if document_type == "unknown":
        return None, "not_attempted_unknown_document"
    spec = _KNOWN_FORM_TABLE_SPECS[document_type]
    cells, cell_status = _native_table_cells(native_table, page=page)
    if cell_status is not None:
        return None, cell_status
    header, header_status = _known_form_header(spec=spec, cells=cells, page=page)
    if header is None:
        # General OCR may miss a header alias even though the provider emitted
        # a fully evidenced native grid.  The geometry fallback remains
        # bounded to one exact-title-known form and one unique 2xN pair; it
        # cannot substitute text, fill cells, or choose among alternatives.
        if header_status == "missing_header_evidence":
            geometry_target = _geometry_known_form_table_target(
                spec=spec,
                cells=cells,
                page=page,
            )
            if geometry_target is not None:
                return geometry_target, "accepted"
        return None, header_status
    body, body_status = _known_form_body(header=header, cells=cells, page=page)
    if body is None:
        return None, body_status
    return _KnownFormTableTarget(header=header, body=body), "accepted"


def _project_known_form_table(
    *,
    table_id: str,
    page_number: int,
    page: RasterDocumentPreviewPageV2,
    target: _KnownFormTableTarget,
) -> _TableProjectionAttempt:
    cells: list[RasterTableCellV2] = []
    for row_index, evidence_row in enumerate((target.header, target.body)):
        for column_index, evidence in enumerate(evidence_row):
            box = evidence.cell.pixel_bbox
            try:
                bbox = BoundingBoxV1(
                    x=box.left / page.width,
                    y=box.top / page.height,
                    width=box.width / page.width,
                    height=box.height / page.height,
                )
                cells.append(
                    RasterTableCellV2(
                        cell_id=f"{table_id}-r{row_index}-c{column_index}",
                        row_index=row_index,
                        column_index=column_index,
                        text="\n".join(span.text for span in evidence.spans),
                        bbox=bbox,
                        confidence=min(span.confidence for span in evidence.spans),
                        evidence_span_ids=[span.span_id for span in evidence.spans],
                    )
                )
            except ValidationError:
                return _TableProjectionAttempt(None, "invalid_projected_geometry")
    table_bbox = _union_bbox([cell.bbox for cell in cells])
    if table_bbox is None:
        return _TableProjectionAttempt(None, "invalid_projected_geometry")
    try:
        rows = [
            RasterTableRowV2(
                row_index=row_index,
                cells=[cell for cell in cells if cell.row_index == row_index],
            )
            for row_index in (0, 1)
        ]
        return _TableProjectionAttempt(
            RasterTableV2(
                table_id=table_id,
                page_number=page_number,
                bbox=table_bbox,
                rows=rows,
            ),
            "accepted",
        )
    except ValidationError:
        return _TableProjectionAttempt(None, "invalid_projected_table")


def _known_form_table_attempt(
    *,
    document_type: RasterDocumentType,
    table_id: str,
    page_number: int,
    page: RasterDocumentPreviewPageV2,
    native_table: Mapping[str, object],
) -> _TableProjectionAttempt:
    target, status = _known_form_table_target(
        document_type=document_type,
        native_table=native_table,
        page=page,
    )
    if target is None:
        return _TableProjectionAttempt(None, status)
    return _project_known_form_table(
        table_id=table_id,
        page_number=page_number,
        page=page,
        target=target,
    )


def _observed_table_attempts(
    *,
    document_type: RasterDocumentType,
    native_results: Sequence[Mapping[str, object]],
    pages: Sequence[RasterDocumentPreviewPageV2],
) -> tuple[list[_TableProjectionAttempt], _TableProjectionStatus | None]:
    if document_type == "unknown":
        return [], "not_attempted_unknown_document"
    if len(pages) != 1 or len(native_results) != 1:
        return [], "document_layout_unsupported"
    table_values = _as_list(native_results[0].get("table_res_list"))
    if table_values is None or not table_values:
        return [], "missing_native_table"
    native_tables = [_as_mapping(value) for value in table_values]
    if any(table is None for table in native_tables):
        return [], "invalid_native_table"
    attempts = [
        _known_form_table_attempt(
            document_type=document_type,
            table_id="items",
            page_number=pages[0].page_number,
            page=pages[0],
            native_table=table,
        )
        for table in native_tables
        if table is not None
    ]
    return attempts, None


def _table_projection_status(
    *,
    document_type: RasterDocumentType,
    native_results: Sequence[Mapping[str, object]],
    pages: Sequence[RasterDocumentPreviewPageV2],
) -> _TableProjectionStatus:
    """Return the exact content-free known-form selector outcome."""

    attempts, blocking_status = _observed_table_attempts(
        document_type=document_type,
        native_results=native_results,
        pages=pages,
    )
    if blocking_status is not None:
        return blocking_status
    accepted = [attempt for attempt in attempts if attempt.table is not None]
    if len(accepted) == 1 and all(
        attempt.table is not None or attempt.status == "missing_header_evidence"
        for attempt in attempts
    ):
        return "accepted"
    if len(accepted) > 1:
        return "ambiguous_native_table"
    if len(accepted) == 1:
        # A second native table supplied target-header evidence but could not
        # prove a unique complete body, so the complete target is not unique.
        return "ambiguous_native_table"
    if not attempts:
        return "missing_native_table"
    statuses = {attempt.status for attempt in attempts}
    if len(statuses) == 1:
        return next(iter(statuses))
    if "duplicate_header_evidence" in statuses:
        return "duplicate_header_evidence"
    if "ambiguous_body_evidence" in statuses or "duplicate_body_evidence" in statuses:
        return "ambiguous_body_evidence"
    if "missing_header_evidence" in statuses:
        return "missing_header_evidence"
    return "body_grid_incomplete"


def _table_candidates(
    *,
    document_type: RasterDocumentType,
    native_results: Sequence[Mapping[str, object]],
    pages: Sequence[RasterDocumentPreviewPageV2],
) -> tuple[list[RasterTableV2], set[str]]:
    """Select exactly one complete known-form native header/body subgrid.

    Provider tables often retain footer fragments or merged cells.  Those are
    ignored only when they lie outside one uniquely evidenced target pair; a
    missing, duplicate, or ambiguous header/body target always remains a
    reviewer-visible failure.  No table-OCR text is used to identify or fill a
    selected cell.
    """

    attempts, blocking_status = _observed_table_attempts(
        document_type=document_type,
        native_results=native_results,
        pages=pages,
    )
    if blocking_status is not None:
        return [], {
            "document_layout_unsupported"
            if blocking_status == "document_layout_unsupported"
            else "table_structure_incomplete"
        }
    accepted = [attempt.table for attempt in attempts if attempt.table is not None]
    if len(accepted) == 1 and all(
        attempt.table is not None or attempt.status == "missing_header_evidence"
        for attempt in attempts
    ):
        return [accepted[0]], set()
    if len(accepted) > 1 or len(accepted) == 1:
        return [], {"table_structure_ambiguous"}
    if any(
        attempt.status
        in {"duplicate_header_evidence", "duplicate_body_evidence", "ambiguous_body_evidence"}
        for attempt in attempts
    ):
        return [], {"table_structure_ambiguous"}
    return [], {"table_structure_incomplete"}


def project_ppstructurev3_predictions(
    *,
    source_sha256: str,
    raster_profile_revision: str,
    configuration: PpStructureV3SidecarConfigurationV1,
    pages: Sequence[_InputPage],
    native_results: Sequence[object],
) -> PpStructureV3SidecarResultV1:
    """Project PP-StructureV3 output without using any OCR/VLM Gold or fallback."""

    if (
        _SHA256.fullmatch(source_sha256) is None
        or raster_profile_revision != configuration.raster_profile_revision
        or len(pages) != len(native_results)
        or not pages
    ):
        raise PpStructureV3SidecarError("ppstructurev3_request_binding_invalid")
    native_mappings: list[Mapping[str, object]] = []
    preview_pages: list[RasterDocumentPreviewPageV2] = []
    reasons: set[str] = set()
    for page, native_result in zip(pages, native_results):
        mapping = _as_mapping(native_result)
        if mapping is None:
            mapping = {}
            reasons.add("ppstructurev3_output_incomplete")
        native_mappings.append(mapping)
        raw_spans, page_reasons = _native_ocr_spans(mapping, page)
        page_spans, ordering_reasons = _ordered_spans(raw_spans, mapping, page.page_number)
        reasons.update(page_reasons)
        reasons.update(ordering_reasons)
        preview_pages.append(
            RasterDocumentPreviewPageV2(
                page_number=page.page_number,
                width=page.width,
                height=page.height,
                coordinate_unit="pixels",
                spans=page_spans,
            )
        )
    all_spans = [span for page in preview_pages for span in page.spans]
    document_type, classification_confidence, classification_reasons = _classify_document(all_spans)
    reasons.update(classification_reasons)
    fields: list[RasterFieldCandidateV2] = []
    tables: list[RasterTableV2] = []
    if document_type != "unknown":
        fields, field_reasons = _field_candidates(document_type, all_spans)
        tables, table_reasons = _table_candidates(
            document_type=document_type,
            native_results=native_mappings,
            pages=preview_pages,
        )
        reasons.update(field_reasons)
        reasons.update(table_reasons)
    warning_models = [RasterExtractionWarningV2(code=code) for code in sorted(reasons)]
    return PpStructureV3SidecarResultV1(
        source_sha256=source_sha256,
        raster_profile_revision=raster_profile_revision,
        provider_revision=configuration.extractor_revision,
        document_type=document_type,
        classification_confidence=classification_confidence,
        pages=preview_pages,
        field_candidates=fields,
        tables=tables,
        warnings=[],
        needs_review_reasons=warning_models,
    )


def _read_request(stream: BinaryIO) -> tuple[str, str, list[_InputPage]]:
    header_raw = stream.readline(_MAX_PROTOCOL_HEADER_BYTES + 1)
    if (
        not header_raw
        or len(header_raw) > _MAX_PROTOCOL_HEADER_BYTES
        or not header_raw.endswith(b"\n")
    ):
        raise PpStructureV3SidecarError("ppstructurev3_protocol_invalid")
    try:
        header = json.loads(header_raw)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise PpStructureV3SidecarError("ppstructurev3_protocol_invalid") from None
    if not isinstance(header, Mapping) or set(header) != {
        "protocol",
        "source_sha256",
        "raster_profile_revision",
        "pages",
    }:
        raise PpStructureV3SidecarError("ppstructurev3_protocol_invalid")
    if (
        header.get("protocol") != "customs-raster-ppstructurev3-v1"
        or not isinstance(header.get("source_sha256"), str)
        or _SHA256.fullmatch(header["source_sha256"]) is None
        or not isinstance(header.get("raster_profile_revision"), str)
        or _IDENTIFIER.fullmatch(header["raster_profile_revision"]) is None
    ):
        raise PpStructureV3SidecarError("ppstructurev3_protocol_invalid")
    descriptors = header.get("pages")
    if not isinstance(descriptors, list) or not 1 <= len(descriptors) <= _MAX_PAGES:
        raise PpStructureV3SidecarError("ppstructurev3_protocol_invalid")
    expected_total = 0
    parsed: list[tuple[int, int, int, int, str]] = []
    for expected_number, descriptor in enumerate(descriptors, start=1):
        if not isinstance(descriptor, Mapping) or set(descriptor) != {
            "page_number",
            "width",
            "height",
            "pixel_length",
            "pixel_sha256",
        }:
            raise PpStructureV3SidecarError("ppstructurev3_protocol_invalid")
        page_number = descriptor.get("page_number")
        width = descriptor.get("width")
        height = descriptor.get("height")
        pixel_length = descriptor.get("pixel_length")
        pixel_sha256 = descriptor.get("pixel_sha256")
        if not all(
            isinstance(value, int) and not isinstance(value, bool)
            for value in (page_number, width, height, pixel_length)
        ):
            raise PpStructureV3SidecarError("ppstructurev3_protocol_invalid")
        assert isinstance(page_number, int)
        assert isinstance(width, int)
        assert isinstance(height, int)
        assert isinstance(pixel_length, int)
        if (
            page_number != expected_number
            or width < 1
            or height < 1
            or width * height > _MAX_RASTER_PIXELS
            or pixel_length != width * height * 3
            or not isinstance(pixel_sha256, str)
            or _SHA256.fullmatch(pixel_sha256) is None
        ):
            raise PpStructureV3SidecarError("ppstructurev3_protocol_invalid")
        expected_total += pixel_length
        parsed.append((page_number, width, height, pixel_length, pixel_sha256))
    payload = stream.read(expected_total + 1)
    if len(payload) != expected_total:
        raise PpStructureV3SidecarError("ppstructurev3_protocol_invalid")
    pages: list[_InputPage] = []
    offset = 0
    for page_number, width, height, pixel_length, pixel_sha256 in parsed:
        pixels = payload[offset : offset + pixel_length]
        offset += pixel_length
        if hashlib.sha256(pixels).hexdigest() != pixel_sha256:
            raise PpStructureV3SidecarError("ppstructurev3_protocol_invalid")
        pages.append(_InputPage(page_number, width, height, pixels))
    return header["source_sha256"], header["raster_profile_revision"], pages


def _read_evaluation_batch_request(
    stream: BinaryIO,
) -> list[tuple[str, str, str, list[_InputPage]]]:
    """Read 1--4 independently authenticated single-page evaluation inputs.

    The outer envelope is deliberately evaluation-only.  It carries opaque
    item identifiers so identical raster SHA-256 values from clean/error
    families cannot be confused, while each nested payload remains the exact
    v1 request with its own pixel and source bindings.
    """

    header_raw = stream.readline(_MAX_PROTOCOL_HEADER_BYTES + 1)
    if (
        not header_raw
        or len(header_raw) > _MAX_PROTOCOL_HEADER_BYTES
        or not header_raw.endswith(b"\n")
    ):
        raise PpStructureV3SidecarError("ppstructurev3_protocol_invalid")
    try:
        header = _EvaluationBatchRequestHeaderV2.model_validate_json(header_raw)
    except (ValidationError, UnicodeDecodeError, ValueError):
        raise PpStructureV3SidecarError("ppstructurev3_protocol_invalid") from None
    expected_total = sum(item.input_length for item in header.items)
    if expected_total > _MAX_EVALUATION_BATCH_RAW_BYTES:
        raise PpStructureV3SidecarError("ppstructurev3_protocol_invalid")
    payload = stream.read(expected_total + 1)
    if len(payload) != expected_total:
        raise PpStructureV3SidecarError("ppstructurev3_protocol_invalid")
    parsed: list[tuple[str, str, str, list[_InputPage]]] = []
    offset = 0
    for item in header.items:
        nested = payload[offset : offset + item.input_length]
        offset += item.input_length
        if hashlib.sha256(nested).hexdigest() != item.input_sha256:
            raise PpStructureV3SidecarError("ppstructurev3_protocol_invalid")
        source_sha256, raster_profile_revision, pages = _read_request(io.BytesIO(nested))
        # Raster v1's sealed corpus contains only single-page assets.  Keeping
        # one source document per native list element prevents field/table
        # projection from ever combining unrelated pages as one document.
        if len(pages) != 1:
            raise PpStructureV3SidecarError("ppstructurev3_protocol_invalid")
        parsed.append((item.item_id, source_sha256, raster_profile_revision, pages))
    return parsed


def _native_bgr_image(page: _InputPage) -> object:
    """Copy one RGB request page into a writable BGR native input array."""

    try:
        import numpy as np  # type: ignore[import-not-found]

        rgb = np.frombuffer(page.pixels, dtype=np.uint8).reshape((page.height, page.width, 3))
        return np.ascontiguousarray(rgb[:, :, ::-1])
    except Exception:
        raise PpStructureV3SidecarError("ppstructurev3_runtime_unavailable") from None


def _native_prediction(
    pipeline: object,
    page: _InputPage,
    *,
    prediction_kwargs: Mapping[str, object],
) -> object:
    try:
        # The raster protocol stores RGB bytes. PaddleX treats an ndarray as
        # BGR and may pass it straight to native code, so it needs the copied,
        # writable buffer supplied by the common conversion helper.
        image = _native_bgr_image(page)
        predicted = pipeline.predict(image, **prediction_kwargs)  # type: ignore[attr-defined]
        if isinstance(predicted, Mapping):
            return predicted
        iterator = iter(predicted)
        first = next(iterator)
        try:
            next(iterator)
        except StopIteration:
            return first
        raise PpStructureV3SidecarError("ppstructurev3_output_incomplete")
    except PpStructureV3SidecarError:
        raise
    except Exception:
        raise PpStructureV3SidecarError("ppstructurev3_runtime_unavailable") from None


def _native_predictions(
    pipeline: object,
    pages: Sequence[_InputPage],
    *,
    prediction_kwargs: Mapping[str, object],
) -> list[object]:
    """Run the native list-input API once and require one result per page."""

    if not 1 <= len(pages) <= _MAX_EVALUATION_BATCH_ITEMS:
        raise PpStructureV3SidecarError("ppstructurev3_request_binding_invalid")
    try:
        predicted = pipeline.predict(  # type: ignore[attr-defined]
            [_native_bgr_image(page) for page in pages], **prediction_kwargs
        )
        results: list[object]
        if isinstance(predicted, Mapping):
            results = [predicted]
        else:
            results = list(predicted)
        if len(results) != len(pages):
            raise PpStructureV3SidecarError("ppstructurev3_output_incomplete")
        return results
    except PpStructureV3SidecarError:
        raise
    except Exception:
        raise PpStructureV3SidecarError("ppstructurev3_runtime_unavailable") from None


def _read_exact(stream: BinaryIO, size: int, *, allow_clean_eof: bool = False) -> bytes | None:
    """Read exactly ``size`` bytes without accepting a truncated frame."""

    if size < 1:
        raise PpStructureV3SidecarError("ppstructurev3_protocol_invalid")
    parts: list[bytes] = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            if allow_clean_eof and not parts:
                return None
            raise PpStructureV3SidecarError("ppstructurev3_protocol_invalid")
        parts.append(chunk)
        remaining -= len(chunk)
    return b"".join(parts)


def _read_batch_frame(stream: BinaryIO) -> bytes | None:
    """Read one bounded binary frame, or clean EOF between frames only."""

    raw_length = _read_exact(stream, 8, allow_clean_eof=True)
    if raw_length is None:
        return None
    length = struct.unpack(">Q", raw_length)[0]
    if not 1 <= length <= _MAX_BATCH_REQUEST_BYTES:
        raise PpStructureV3SidecarError("ppstructurev3_protocol_invalid")
    payload = _read_exact(stream, length)
    if payload is None:  # Defensive for the type checker; clean EOF is off.
        raise PpStructureV3SidecarError("ppstructurev3_protocol_invalid")
    return payload


def _write_batch_frame(
    output: BinaryIO, payload: bytes, *, maximum_size: int = _MAX_CONFIGURATION_BYTES
) -> None:
    if not 1 <= len(payload) <= maximum_size:
        raise PpStructureV3SidecarError("ppstructurev3_output_too_large")
    output.write(struct.pack(">Q", len(payload)))
    output.write(payload)
    output.flush()


def _encoded_result(result: PpStructureV3SidecarResultV1) -> bytes:
    encoded = result.model_dump_json(by_alias=True, exclude_none=False).encode("utf-8")
    if len(encoded) > _MAX_CONFIGURATION_BYTES:
        raise PpStructureV3SidecarError("ppstructurev3_output_too_large")
    return encoded


def _encoded_structure_probe(result: PpStructureV3StructureProbeResultV1) -> bytes:
    """Encode one bounded diagnostic without widening OCR-result limits."""

    encoded = result.model_dump_json(by_alias=True, exclude_none=False).encode("utf-8")
    if len(encoded) > _MAX_STRUCTURE_PROBE_BYTES:
        raise PpStructureV3SidecarError("ppstructurev3_output_too_large")
    return encoded


def _encoded_evaluation_batch_result(
    results: Sequence[tuple[str, PpStructureV3SidecarResultV1]],
) -> bytes:
    """Encode a bounded v2 envelope that preserves parent item identities."""

    try:
        envelope = _EvaluationBatchResultV2(
            results=[
                _EvaluationBatchResultItemV2(item_id=item_id, result=result)
                for item_id, result in results
            ]
        )
        encoded = envelope.model_dump_json(by_alias=True, exclude_none=False).encode("utf-8")
    except (ValidationError, ValueError):
        raise PpStructureV3SidecarError("ppstructurev3_output_incomplete") from None
    if len(encoded) > _MAX_EVALUATION_BATCH_RESPONSE_BYTES:
        raise PpStructureV3SidecarError("ppstructurev3_output_too_large")
    return encoded


def run_once_from_stdio(stream: BinaryIO, output: BinaryIO) -> None:
    """Run exactly one request; callers deliberately suppress diagnostics."""

    configuration, model_root = configuration_from_environment()
    source_sha256, raster_profile_revision, pages = _read_request(stream)
    if raster_profile_revision != configuration.raster_profile_revision:
        raise PpStructureV3SidecarError("ppstructurev3_request_binding_invalid")
    with deny_network_connections():
        try:
            from paddleocr import PPStructureV3  # type: ignore[import-not-found]

            pipeline = PPStructureV3(**configuration.pipeline_kwargs(model_root))
        except Exception:
            raise PpStructureV3SidecarError("ppstructurev3_runtime_unavailable") from None
        prediction_kwargs = configuration.prediction_kwargs()
        native_results = [
            _native_prediction(pipeline, page, prediction_kwargs=prediction_kwargs)
            for page in pages
        ]
    result = project_ppstructurev3_predictions(
        source_sha256=source_sha256,
        raster_profile_revision=raster_profile_revision,
        configuration=configuration,
        pages=pages,
        native_results=native_results,
    )
    output.write(_encoded_result(result))


def run_structure_probe_from_stdio(stream: BinaryIO, output: BinaryIO) -> None:
    """Run one explicit content-free native-structure diagnostic request.

    The diagnostic uses the same hash-bound configuration, native pipeline,
    RGB-to-BGR conversion, and network denial as the production one-shot
    request. It intentionally emits only the counters in
    ``PpStructureV3NativeStructureProbeV1`` and one fixed, content-free table
    projection status. It is not routed by the service runtime or the
    candidate evaluator.
    """

    configuration, model_root = configuration_from_environment()
    _source_sha256, raster_profile_revision, pages = _read_request(stream)
    if raster_profile_revision != configuration.raster_profile_revision:
        raise PpStructureV3SidecarError("ppstructurev3_request_binding_invalid")
    with deny_network_connections():
        try:
            from paddleocr import PPStructureV3  # type: ignore[import-not-found]

            pipeline = PPStructureV3(**configuration.pipeline_kwargs(model_root))
        except Exception:
            raise PpStructureV3SidecarError("ppstructurev3_runtime_unavailable") from None
        prediction_kwargs = configuration.prediction_kwargs()
        native_results = [
            _native_prediction(pipeline, page, prediction_kwargs=prediction_kwargs)
            for page in pages
        ]
    projected = project_ppstructurev3_predictions(
        source_sha256=_source_sha256,
        raster_profile_revision=raster_profile_revision,
        configuration=configuration,
        pages=pages,
        native_results=native_results,
    )
    native_mappings = [
        mapping if (mapping := _as_mapping(result)) is not None else {} for result in native_results
    ]
    output.write(
        _encoded_structure_probe(
            PpStructureV3StructureProbeResultV1(
                pages=[
                    summarize_ppstructurev3_native_structure(result, page_number=page.page_number)
                    for page, result in zip(pages, native_results, strict=True)
                ],
                table_projection_status=_table_projection_status(
                    document_type=projected.document_type,
                    native_results=native_mappings,
                    pages=projected.pages,
                ),
            )
        )
    )


def run_batch_from_stdio(stream: BinaryIO, output: BinaryIO) -> None:
    """Run sequential evaluation requests with one verified model pipeline.

    This is deliberately not a service protocol.  The parent opens it only
    for a finite calibration or sealed split, verifies all hash-bound runtime
    material before and after the session, and writes no candidate artifact
    until this worker exits cleanly.  A malformed request or native failure
    terminates the whole worker without emitting error details.
    """

    configuration, model_root = configuration_from_environment()
    with deny_network_connections():
        try:
            from paddleocr import PPStructureV3  # type: ignore[import-not-found]

            pipeline = PPStructureV3(**configuration.pipeline_kwargs(model_root))
        except Exception:
            raise PpStructureV3SidecarError("ppstructurev3_runtime_unavailable") from None
        prediction_kwargs = configuration.prediction_kwargs()
        _write_batch_frame(
            output,
            _canonical_json({"protocol": _BATCH_PROTOCOL, "status": "ready"}),
        )
        while True:
            payload = _read_batch_frame(stream)
            if payload is None:
                return
            source_sha256, raster_profile_revision, pages = _read_request(io.BytesIO(payload))
            if raster_profile_revision != configuration.raster_profile_revision:
                raise PpStructureV3SidecarError("ppstructurev3_request_binding_invalid")
            native_results = [
                _native_prediction(pipeline, page, prediction_kwargs=prediction_kwargs)
                for page in pages
            ]
            result = project_ppstructurev3_predictions(
                source_sha256=source_sha256,
                raster_profile_revision=raster_profile_revision,
                configuration=configuration,
                pages=pages,
                native_results=native_results,
            )
            _write_batch_frame(output, _encoded_result(result))


def run_service_from_stdio(stream: BinaryIO, output: BinaryIO) -> None:
    """Run sequential approved-service requests with one loaded pipeline.

    This has the same tightly framed RGB input and local-only execution
    boundary as the one-shot command, but is intentionally a *different*
    protocol from the finite evaluator batch.  It produces no candidate
    receipt, cannot batch calibration assets, and exists solely so the parent
    can keep one already-approved profile warm for bounded user requests.
    """

    configuration, model_root = configuration_from_environment()
    with deny_network_connections():
        try:
            from paddleocr import PPStructureV3  # type: ignore[import-not-found]

            pipeline = PPStructureV3(**configuration.pipeline_kwargs(model_root))
        except Exception:
            raise PpStructureV3SidecarError("ppstructurev3_runtime_unavailable") from None
        prediction_kwargs = configuration.prediction_kwargs()
        _write_batch_frame(
            output,
            _canonical_json({"protocol": _SERVICE_PROTOCOL, "status": "ready"}),
        )
        while True:
            payload = _read_batch_frame(stream)
            if payload is None:
                return
            source_sha256, raster_profile_revision, pages = _read_request(io.BytesIO(payload))
            if raster_profile_revision != configuration.raster_profile_revision:
                raise PpStructureV3SidecarError("ppstructurev3_request_binding_invalid")
            native_results = [
                _native_prediction(pipeline, page, prediction_kwargs=prediction_kwargs)
                for page in pages
            ]
            result = project_ppstructurev3_predictions(
                source_sha256=source_sha256,
                raster_profile_revision=raster_profile_revision,
                configuration=configuration,
                pages=pages,
                native_results=native_results,
            )
            _write_batch_frame(output, _encoded_result(result))


def run_evaluation_batch_v2_from_stdio(stream: BinaryIO, output: BinaryIO) -> None:
    """Run native list-input batches only for the finite synthetic evaluator.

    This is deliberately separate from both the service one-shot command and
    legacy evaluation batch v1. Every outer item remains an independently
    verified single-page document, and each native prediction is projected
    separately before it re-enters the parent candidate ledger.
    """

    configuration, model_root = configuration_from_environment()
    if configuration.evaluation_batch_size != _MAX_EVALUATION_BATCH_ITEMS:
        raise PpStructureV3SidecarError("ppstructurev3_request_binding_invalid")
    with deny_network_connections():
        try:
            from paddleocr import PPStructureV3  # type: ignore[import-not-found]

            pipeline = PPStructureV3(**configuration.pipeline_kwargs(model_root))
        except Exception:
            raise PpStructureV3SidecarError("ppstructurev3_runtime_unavailable") from None
        prediction_kwargs = configuration.prediction_kwargs()
        _write_batch_frame(
            output,
            _canonical_json({"protocol": _EVALUATION_BATCH_PROTOCOL, "status": "ready"}),
        )
        while True:
            payload = _read_batch_frame(stream)
            if payload is None:
                return
            requests = _read_evaluation_batch_request(io.BytesIO(payload))
            if any(
                raster_profile_revision != configuration.raster_profile_revision
                for _item_id, _source_sha256, raster_profile_revision, _pages in requests
            ):
                raise PpStructureV3SidecarError("ppstructurev3_request_binding_invalid")
            pages = [pages[0] for _item_id, _source_sha256, _revision, pages in requests]
            native_results = _native_predictions(
                pipeline, pages, prediction_kwargs=prediction_kwargs
            )
            results: list[tuple[str, PpStructureV3SidecarResultV1]] = []
            for (
                item_id,
                source_sha256,
                raster_profile_revision,
                request_pages,
            ), native_result in zip(requests, native_results, strict=True):
                results.append(
                    (
                        item_id,
                        project_ppstructurev3_predictions(
                            source_sha256=source_sha256,
                            raster_profile_revision=raster_profile_revision,
                            configuration=configuration,
                            pages=request_pages,
                            native_results=[native_result],
                        ),
                    )
                )
            _write_batch_frame(
                output,
                _encoded_evaluation_batch_result(results),
                maximum_size=_MAX_EVALUATION_BATCH_RESPONSE_BYTES,
            )


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point.  Error details intentionally stay private."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if arguments == ["--customs-raster-ppstructurev3-v1"]:
            run_once_from_stdio(sys.stdin.buffer, sys.stdout.buffer)
        elif arguments == [f"--{_STRUCTURE_PROBE_PROTOCOL}"]:
            run_structure_probe_from_stdio(sys.stdin.buffer, sys.stdout.buffer)
        elif arguments == ["--customs-raster-ppstructurev3-batch-v1"]:
            run_batch_from_stdio(sys.stdin.buffer, sys.stdout.buffer)
        elif arguments == [f"--{_SERVICE_PROTOCOL}"]:
            run_service_from_stdio(sys.stdin.buffer, sys.stdout.buffer)
        elif arguments == ["--customs-raster-ppstructurev3-evaluation-batch-v2"]:
            run_evaluation_batch_v2_from_stdio(sys.stdin.buffer, sys.stdout.buffer)
        else:
            return 2
    except Exception:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover - console-script path
    raise SystemExit(main())


__all__ = [
    "PpStructureV3SidecarConfigurationV1",
    "PpStructureV3SidecarError",
    "PpStructureV3SidecarResultV1",
    "PpStructureV3NativeStructureProbeV1",
    "PpStructureV3StructureProbeResultV1",
    "configuration_from_environment",
    "deny_network_connections",
    "main",
    "project_ppstructurev3_predictions",
    "run_batch_from_stdio",
    "run_evaluation_batch_v2_from_stdio",
    "run_once_from_stdio",
    "run_service_from_stdio",
    "run_structure_probe_from_stdio",
    "summarize_ppstructurev3_native_structure",
]
