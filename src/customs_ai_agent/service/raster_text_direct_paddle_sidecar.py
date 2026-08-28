"""Narrow offline static-inference POC for the raster text sidecar.

This is intentionally *not* selected by the raster profile resolver or an
approval pointer.  It proves that the seven locally-attested text model files
can be used without a high-level OCR wrapper package: detector preprocessing,
DB-style box extraction, perspective crops and CTC decoding live here.

The one-shot wire envelope is kept byte-for-byte compatible with the existing
text sidecar.  A future candidate may copy this source into a hash-bound
sidecar release only after its own complete release/profile verification.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.metadata
import io
import json
import math
import os
from pathlib import Path
import re
import socket
import sys
from typing import Any, BinaryIO, Iterator, Literal, Sequence, cast


_PROTOCOL = "customs-raster-text-sidecar-v1"
_SESSION_PROTOCOL = "customs-raster-text-sidecar-v1-session"
_SERVICE_PROTOCOL = "customs-raster-text-sidecar-v1-service"
_MAX_HEADER_BYTES = 1024 * 1024
_MAX_INPUT_BYTES = 128 * 1024 * 1024
_MAX_RESPONSE_BYTES = 2 * 1024 * 1024
_MODEL_HASH_CHUNK_BYTES = 1024 * 1024
_HEX = frozenset("0123456789abcdef")
_DIRECT_REQUIRED_PACKAGES = (
    "numpy",
    "opencv-contrib-python",
    "paddlepaddle",
    "pyclipper",
)
_DIRECT_MODEL_ARTIFACTS = frozenset(
    {
        "text-det/inference.json",
        "text-det/inference.pdiparams",
        "text-det/inference.yml",
        "text-rec/config.json",
        "text-rec/inference.json",
        "text-rec/inference.pdiparams",
        "text-rec/inference.yml",
    }
)
_DIRECT_MOBILE_MODEL_ARTIFACTS = _DIRECT_MODEL_ARTIFACTS | {"text-det/config.json"}
_SOURCE_ATTESTED_V12_RUNTIME_PROFILE = (
    "12.0",
    "direct_paddle_source_attested_text_harmonic_recall_native960_lowbox_strip2",
    "paddle-inference-static-v1",
)
_PUBLIC_MOBILE_V13_RUNTIME_PROFILE = (
    "13.0",
    "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_semantic_repair",
    "paddle-inference-static-v1",
)
_PUBLIC_MOBILE_V14_RUNTIME_PROFILE = (
    "14.0",
    "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_semantic_repair_v2",
    "paddle-inference-static-v1",
)
_PUBLIC_MOBILE_V15_ONEDNN_RUNTIME_PROFILE = (
    "15.0",
    "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_semantic_repair_v2_onednn",
    "paddle-inference-static-v1",
)
_PUBLIC_MOBILE_V16_CUBIC_RUNTIME_PROFILE = (
    "16.0",
    "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_"
    "semantic_repair_v2_onednn_cubic_recognition",
    "paddle-inference-static-v1",
)
_PUBLIC_MOBILE_V17_PADDED_RUNTIME_PROFILE = (
    "17.0",
    "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_"
    "semantic_repair_v2_onednn_padded_recognition",
    "paddle-inference-static-v1",
)
_PUBLIC_SERVER_RECOGNIZER_V18_RUNTIME_PROFILE = (
    "18.0",
    "direct_paddle_mobile_det_server_rec_harmonic_recall_native960_lowbox_strip2_"
    "semantic_repair_v2_onednn",
    "paddle-inference-static-v1",
)
_PUBLIC_FINETUNED_RECOGNIZER_V19_RUNTIME_PROFILE = (
    "19.0",
    "direct_paddle_mobile_det_public_finetuned_rec_harmonic_recall_native960_"
    "lowbox_strip2_semantic_repair_v2_onednn",
    "paddle-inference-static-v1",
)
_PUBLIC_FINETUNED_RECOGNIZER_V19_VOYAGE_RETRY_RUNTIME_PROFILE = (
    "19.0",
    "direct_paddle_mobile_det_public_finetuned_rec_harmonic_recall_native960_"
    "lowbox_strip2_semantic_repair_v2_voyage_retry_onednn",
    "paddle-inference-static-v1",
)
_PUBLIC_VOYAGE_REPLAY_RECOGNIZER_V20_RUNTIME_PROFILE = (
    "20.0",
    "direct_paddle_mobile_det_public_voyage_replay_rec_harmonic_recall_native960_"
    "lowbox_strip2_semantic_repair_v2_onednn",
    "paddle-inference-static-v1",
)
_ONEDNN_RUNTIME_PROFILES = frozenset(
    {
        _PUBLIC_MOBILE_V15_ONEDNN_RUNTIME_PROFILE,
        _PUBLIC_MOBILE_V16_CUBIC_RUNTIME_PROFILE,
        _PUBLIC_MOBILE_V17_PADDED_RUNTIME_PROFILE,
        _PUBLIC_SERVER_RECOGNIZER_V18_RUNTIME_PROFILE,
        _PUBLIC_FINETUNED_RECOGNIZER_V19_RUNTIME_PROFILE,
        _PUBLIC_FINETUNED_RECOGNIZER_V19_VOYAGE_RETRY_RUNTIME_PROFILE,
        _PUBLIC_VOYAGE_REPLAY_RECOGNIZER_V20_RUNTIME_PROFILE,
    }
)
_MAX_DETECTION_CANDIDATES = 1_000
_RECOGNITION_BATCH_SIZE = 8
_RECOGNITION_MAX_WIDTH = 3_200
_LEGACY_DETECTION_LIMITS = frozenset({1024, 1280, 1536, 2048, 2526})
_DIRECT_V2_DETECTION_LIMIT = 960
_DIRECT_V6_DETECTION_LIMIT = 1024
# These are the exact lower dynamic input bounds declared by both attested
# PP-OCRv5 detector exports (server and mobile) and the shared English mobile
# recognizer.  They are intentionally fixed engine constants: neither a
# document nor profile configuration can turn startup warmup into inference
# on caller data or alter its allocation shape.
_PRE_READY_WARMUP_DETECTOR_SHAPE = (1, 3, 32, 32)
_PRE_READY_WARMUP_RECOGNIZER_SHAPE = (1, 3, 48, 160)
_V15_PRE_READY_WARMUP_DETECTOR_SHAPE = (1, 3, 736, 960)
_V15_PRE_READY_WARMUP_RECOGNIZER_SHAPE = (1, 3, 48, 320)
_V15_WARMUP_SHAPE_POLICY = "detector-736x960-recognizer-48x320-v1"
_STRIP2_MAX_IOU_COMPARISONS = 4_096
_GRID2_MAX_IOU_COMPARISONS = _STRIP2_MAX_IOU_COMPARISONS
_GRID2_HORIZONTAL_OVERLAP_NUMERATOR = 15
_GRID2_HORIZONTAL_OVERLAP_DENOMINATOR = 100
_GRID2_VERTICAL_OVERLAP_NUMERATOR = 1
_GRID2_VERTICAL_OVERLAP_DENOMINATOR = 10
# V11 uses the same fixed grid as V10, but recognizes a box as a possible
# seam fragment only when the already-clipped local quadrilateral reaches an
# *internal* tile boundary.  Half a pixel is deliberately conservative: it
# catches clipping/rounding at that exact boundary without treating ordinary
# nearby text as a seam candidate.  This is an engine semantic, never input.
_GUARDED_GRID2_INTERNAL_EDGE_EPSILON = 0.5
_SUPPORTED_DOCUMENT_HEADINGS = frozenset({"BILL OF LADING", "COMMERCIAL INVOICE", "PACKING LIST"})
_FORM_ORDINAL_WITH_MISSING_LEFT_PARENTHESIS = re.compile(r"(?:[1-9]|[1-9][0-9])\)")
_WEIGHT_AMOUNT_WITH_UNIT = (
    r"(?:0|[1-9][0-9]{0,8})(?:,[0-9]{3})*(?:\.[0-9]{1,6})?"
    r"[ \t]+(?P<__UNIT_NAME__>KG|KGS|LB|LBS|MT|TON|TONS)"
)
_ADJACENT_WEIGHT_AMOUNTS = re.compile(
    r"^(?P<first>"
    + _WEIGHT_AMOUNT_WITH_UNIT.replace("__UNIT_NAME__", "first_unit")
    + r")(?P<gap>[ \t]+)(?P<second>"
    + _WEIGHT_AMOUNT_WITH_UNIT.replace("__UNIT_NAME__", "second_unit")
    + r")$"
)
_SINGLE_WEIGHT_AMOUNT = re.compile(
    r"^" + _WEIGHT_AMOUNT_WITH_UNIT.replace("__UNIT_NAME__", "unit") + r"$"
)
_SEMANTIC_LABEL_WORD = re.compile(r"[A-Za-z]+")
_SEMANTIC_LABEL_VERTICAL_GAP_HEIGHT_MULTIPLIER = 4.0
_BOUNDED_ASCII_FIELD_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,31}")
_VOYAGE_RETRY_MIN_CONFIDENCE_GAIN = 0.02
_VOYAGE_RETRY_CONFIDENCE_ABS_TOLERANCE = 1e-12
_VOYAGE_RETRY_PREPARED_INPUT_ABS_TOLERANCE = 1.0 / 255.0
_VOYAGE_VALUE_VERTICAL_GAP_HEIGHT_MULTIPLIER = 1.5
_SOURCE_ATTESTED_BL_VOYAGE = re.compile(r"^[0-9]{4}[EWNS]$")
_SOURCE_ATTESTED_BL_TRUNCATED_VOYAGE = re.compile(r"^[0-9]{3}[EWNS]$")


class DirectPaddleSidecarError(RuntimeError):
    """Private failure; callers expose only a generic unavailable state."""


@dataclass(frozen=True, slots=True)
class _Page:
    page_number: int
    width: int
    height: int
    pixels: bytes


@dataclass(frozen=True, slots=True)
class DirectPaddleProfile:
    """The small, explicit subset needed by direct text inference."""

    raster_text_profile_revision: str
    extractor_revision: str
    text_detection_model_directory: str
    text_recognition_model_directory: str
    cpu_threads: int
    text_detection_limit_side_len: int
    text_detection_threshold: float
    text_detection_box_threshold: float
    text_detection_unclip_ratio: float
    # Existing V1/V2 test and runtime profiles retain the server-det default.
    # V3's mobile detector has an extra attested config artifact and must use
    # the separate exact inventory set below.
    model_spec: Literal[
        "server_det",
        "mobile_det",
        "mobile_det_trained_rec",
        "mobile_det_server_rec",
        "mobile_det_public_finetuned_rec",
        "mobile_det_public_voyage_replay_rec",
    ] = "server_det"
    # This is derived from the immutable schema/engine tuple below; no
    # profile field or environment value is allowed to tune it at runtime.
    row_band_mode: Literal["legacy_max", "harmonic_v1"] = "legacy_max"
    # This is likewise derived solely from an exact immutable schema/engine
    # tuple.  V9--V12 tiled detector passes are never caller-provided knobs.
    detector_pass_mode: Literal["single", "strip2", "grid2", "guarded_grid2"] = "single"
    # V2--V16 are the bounded service candidates.  Their startup-only native
    # warmup is derived after accepting the immutable schema/engine tuple;
    # V1 POC forms retain their historical behavior.
    pre_ready_warmup: bool = False
    # V13--V16 select two fixed, candidate-independent repair revisions.  The
    # mode is derived from an exact schema/engine tuple and is never a profile
    # or request knob.
    semantic_repair_mode: Literal["none", "supported_docs_v1", "supported_docs_v2"] = "none"
    # V15/V16 expose these bounded native execution controls.  Defaults
    # retain the byte-for-byte V1--V14 path: oneDNN stays disabled, OpenCV is
    # not reconfigured, and recognition continues in batches of eight.
    enable_mkldnn: bool = False
    mkldnn_cache_capacity: int = 0
    recognition_batch_size: int = _RECOGNITION_BATCH_SIZE
    opencv_threads: int = 0
    warmup_shape_policy: str = "legacy-minimal-v1"
    onednn_model_policy: str = "disabled"
    # V16 changes only the recognizer resize interpolation.  Both values are
    # derived from an exact immutable tuple; callers cannot tune crop geometry
    # or interpolation per request.
    recognition_crop_policy: Literal[
        "detector-quad-v1",
        "horizontal-height-0p08-max4-v1",
        "horizontal-height-0p32-max8-v1",
    ] = "detector-quad-v1"
    recognition_resize_interpolation: Literal["linear-v1", "cubic-v1"] = "linear-v1"
    # The multilingual server export appends one decoder space class even
    # though its manifest character list already contains a space.  This
    # policy is derived only from V18's exact tuple; historical profiles keep
    # their existing optional-space contract.
    ctc_label_policy: Literal["optional-space-v1", "append-space-v1"] = "optional-space-v1"
    # The V19 voyage-retry discriminator alone enables one bounded secondary
    # recognizer call.  This is derived from the exact immutable engine tuple,
    # never from a request, document profile, image type or environment knob.
    voyage_retry_mode: Literal["none", "bill-of-lading-padded-v1"] = "none"


@dataclass(frozen=True, slots=True)
class _Line:
    text: str
    confidence: float
    polygon: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class _StripDetectedBox:
    """One tiled detector result mapped into page coordinates.

    V9/V10 intentionally ignore ``touches_internal_tile_edge``.  V11 alone
    sets and consumes it, so its guarded seam rule cannot alter a historical
    candidate's ordering or NMS semantics.
    """

    box: Any
    score: float
    tile_index: int
    ordinal: int
    touches_internal_tile_edge: bool = False


@dataclass(frozen=True, slots=True)
class _Grid2Tile:
    """One exact V10/V11 tile in deterministic TL/TR/BL/BR order."""

    ordinal: int
    x_start: int
    x_end: int
    y_start: int
    y_end: int


@dataclass(slots=True)
class _NativeEngine:
    detector: Any
    recognizer: Any
    cv2: Any
    np: Any
    pyclipper: Any
    profile: DirectPaddleProfile
    characters: tuple[str, ...]


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "")
    if not value:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_environment_invalid")
    return value


def _read_regular_file(path: Path, *, limit: int) -> bytes:
    try:
        if path.is_symlink() or not path.is_file():
            raise DirectPaddleSidecarError("direct_paddle_sidecar_material_invalid")
        value = path.read_bytes()
    except DirectPaddleSidecarError:
        raise
    except OSError:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_material_invalid") from None
    if not value or len(value) > limit:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_material_invalid")
    return value


def _strict_sha256(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in _HEX for character in value)
    ):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_configuration_invalid")
    return value


def _required_string(value: object) -> str:
    if not isinstance(value, str) or not value or len(value) > 256:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_configuration_invalid")
    return value


def _profile_from_configuration(raw: bytes) -> DirectPaddleProfile:
    """Read the attested direct-text fields, never a wrapper configuration."""

    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_configuration_invalid") from None
    if not isinstance(value, dict):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_configuration_invalid")
    runtime_profile = cast(
        tuple[str, str, str],
        (value.get("schema_version"), value.get("engine_kind"), value.get("runtime_api")),
    )
    native_tuning_fields = {
        "enable_mkldnn",
        "mkldnn_cache_capacity",
        "recognition_batch_size",
        "opencv_threads",
        "warmup_shape_policy",
        "onednn_model_policy",
    }
    recognition_policy_fields = {
        "recognition_crop_policy",
        "recognition_resize_interpolation",
    }
    ctc_policy_fields = {"ctc_label_policy"}
    if runtime_profile == _PUBLIC_MOBILE_V15_ONEDNN_RUNTIME_PROFILE:
        if not native_tuning_fields <= set(value):
            raise DirectPaddleSidecarError("direct_paddle_sidecar_configuration_invalid")
        if recognition_policy_fields & set(value):
            raise DirectPaddleSidecarError("direct_paddle_sidecar_configuration_invalid")
    elif runtime_profile == _PUBLIC_MOBILE_V16_CUBIC_RUNTIME_PROFILE:
        if not (native_tuning_fields | recognition_policy_fields) <= set(value):
            raise DirectPaddleSidecarError("direct_paddle_sidecar_configuration_invalid")
    elif runtime_profile == _PUBLIC_MOBILE_V17_PADDED_RUNTIME_PROFILE:
        if not (native_tuning_fields | recognition_policy_fields) <= set(value):
            raise DirectPaddleSidecarError("direct_paddle_sidecar_configuration_invalid")
    elif runtime_profile == _PUBLIC_SERVER_RECOGNIZER_V18_RUNTIME_PROFILE:
        if not (native_tuning_fields | recognition_policy_fields | ctc_policy_fields) <= set(value):
            raise DirectPaddleSidecarError("direct_paddle_sidecar_configuration_invalid")
    elif runtime_profile in {
        _PUBLIC_FINETUNED_RECOGNIZER_V19_RUNTIME_PROFILE,
        _PUBLIC_FINETUNED_RECOGNIZER_V19_VOYAGE_RETRY_RUNTIME_PROFILE,
    }:
        if not (native_tuning_fields | ctc_policy_fields) <= set(value):
            raise DirectPaddleSidecarError("direct_paddle_sidecar_configuration_invalid")
        if recognition_policy_fields & set(value):
            raise DirectPaddleSidecarError("direct_paddle_sidecar_configuration_invalid")
    elif runtime_profile == _PUBLIC_VOYAGE_REPLAY_RECOGNIZER_V20_RUNTIME_PROFILE:
        if not (native_tuning_fields | recognition_policy_fields | ctc_policy_fields) <= set(value):
            raise DirectPaddleSidecarError("direct_paddle_sidecar_configuration_invalid")
    elif (native_tuning_fields | recognition_policy_fields | ctc_policy_fields) & set(value):
        # Historical profiles must never acquire V15/V16 native semantics by
        # adding otherwise well-typed fields to their immutable tuple.
        raise DirectPaddleSidecarError("direct_paddle_sidecar_configuration_invalid")
    # Row ordering is an engine semantic, not a caller-provided knob.  In
    # particular, a V3 profile must reject V4-only material rather than
    # silently accepting it and changing its historical output order.
    if {
        "row_band_mode",
        "detector_pass_mode",
        "strip_count",
        "strip_overlap",
        "strip_nms_iou",
        "grid_columns",
        "grid_rows",
        "grid_overlap_x",
        "grid_overlap_y",
        "grid_nms_iou",
        "guarded_grid_internal_edge_epsilon",
        "guarded_grid_nms_iou",
        "guarded_grid_intersection_over_min_area",
        "semantic_repair_mode",
        "voyage_retry_mode",
    } & set(value):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_configuration_invalid")
    required = {
        "schema_version",
        "engine_kind",
        "runtime_api",
        "extractor_revision",
        "raster_text_profile_revision",
        "text_detection_model_name",
        "text_detection_model_directory",
        "text_recognition_model_name",
        "text_recognition_model_directory",
        "cpu_threads",
        "text_detection_limit_side_len",
        "text_detection_limit_type",
        "text_detection_threshold",
        "text_detection_box_threshold",
        "text_detection_unclip_ratio",
    }
    if not required <= set(value):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_configuration_invalid")
    numeric_keys = [
        "cpu_threads",
        "text_detection_limit_side_len",
        "text_detection_threshold",
        "text_detection_box_threshold",
        "text_detection_unclip_ratio",
    ]
    if runtime_profile in _ONEDNN_RUNTIME_PROFILES:
        numeric_keys.extend(("mkldnn_cache_capacity", "recognition_batch_size", "opencv_threads"))
    numeric = {key: value[key] for key in numeric_keys}
    if any(
        isinstance(item, bool) or not isinstance(item, (int, float)) for item in numeric.values()
    ):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_configuration_invalid")
    # Each static pair has an explicit profile discriminator.  In
    # particular, a V3--V12 mobile detector cannot be loaded through a V2
    # server profile (or vice versa) merely because their tensor interfaces
    # look similar.  Only exact V4/V5/V6/V7/V8 tuples receive harmonic row
    # bands; V8 retains V5 semantics while varying only its box threshold.
    accepted_runtime_profiles = {
        ("1.0", "paddleocr3_direct_text", "paddleocr3-paddlex-3.2"): (
            "PP-OCRv5_server_det",
            "en_PP-OCRv5_mobile_rec",
            "server_det",
            "legacy_max",
        ),
        ("1.0", "paddle_static_text_poc", "paddle-inference-static-v1"): (
            "PP-OCRv5_server_det",
            "en_PP-OCRv5_mobile_rec",
            "server_det",
            "legacy_max",
        ),
        ("2.0", "direct_paddle_text", "paddle-inference-static-v1"): (
            "PP-OCRv5_server_det",
            "en_PP-OCRv5_mobile_rec",
            "server_det",
            "legacy_max",
        ),
        ("3.0", "direct_paddle_mobile_text", "paddle-inference-static-v1"): (
            "PP-OCRv5_mobile_det",
            "en_PP-OCRv5_mobile_rec",
            "mobile_det",
            "legacy_max",
        ),
        ("4.0", "direct_paddle_mobile_text_harmonic", "paddle-inference-static-v1"): (
            "PP-OCRv5_mobile_det",
            "en_PP-OCRv5_mobile_rec",
            "mobile_det",
            "harmonic_v1",
        ),
        (
            "5.0",
            "direct_paddle_mobile_text_harmonic_recall",
            "paddle-inference-static-v1",
        ): (
            "PP-OCRv5_mobile_det",
            "en_PP-OCRv5_mobile_rec",
            "mobile_det",
            "harmonic_v1",
        ),
        # V6 deliberately retains V5's harmonic reading order and .4 box
        # threshold.  Its only candidate-level inference difference is the
        # 960 -> 1024 detector input limit checked below.
        ("6.0", "direct_paddle_mobile_text_1024", "paddle-inference-static-v1"): (
            "PP-OCRv5_mobile_det",
            "en_PP-OCRv5_mobile_rec",
            "mobile_det",
            "harmonic_v1",
        ),
        # V7 retains V6's model pair, 1024-side detector input, and harmonic
        # row order.  Only its .3 detector box threshold is separately
        # allowed below; no earlier profile can inherit that value.
        (
            "7.0",
            "direct_paddle_mobile_text_harmonic_recall_1024_lowbox",
            "paddle-inference-static-v1",
        ): (
            "PP-OCRv5_mobile_det",
            "en_PP-OCRv5_mobile_rec",
            "mobile_det",
            "harmonic_v1",
        ),
        # V8 returns to V5's native 960-side detector input and permits the
        # same fixed .3 box threshold only under this separate tuple.
        (
            "8.0",
            "direct_paddle_mobile_text_harmonic_recall_native960_lowbox",
            "paddle-inference-static-v1",
        ): (
            "PP-OCRv5_mobile_det",
            "en_PP-OCRv5_mobile_rec",
            "mobile_det",
            "harmonic_v1",
        ),
        # V9 is V8's exact native-960 low-box tuple plus a fixed internal
        # two-strip detector pass.  The pass mode is derived below, never
        # accepted from configuration material.
        (
            "9.0",
            "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2",
            "paddle-inference-static-v1",
        ): (
            "PP-OCRv5_mobile_det",
            "en_PP-OCRv5_mobile_rec",
            "mobile_det",
            "harmonic_v1",
        ),
        # V10 is V8's exact native-960 low-box tuple plus an internal fixed
        # four-tile detector pass.  Its grid geometry is not configuration.
        (
            "10.0",
            "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_grid2",
            "paddle-inference-static-v1",
        ): (
            "PP-OCRv5_mobile_det",
            "en_PP-OCRv5_mobile_rec",
            "mobile_det",
            "harmonic_v1",
        ),
        # V11 retains V10's exact four tiles, but its guarded cross-tile NMS
        # is selected by this distinct immutable engine tuple alone.
        (
            "11.0",
            "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_guarded_grid2",
            "paddle-inference-static-v1",
        ): (
            "PP-OCRv5_mobile_det",
            "en_PP-OCRv5_mobile_rec",
            "mobile_det",
            "harmonic_v1",
        ),
        # V12 changes only the recognizer provenance/model bytes relative to
        # V9.  Its distinct immutable tuple keeps the source-attested
        # recognizer from being loaded through any historical mobile profile
        # while retaining V9's exact strip2 detector and reading-order
        # semantics.
        _SOURCE_ATTESTED_V12_RUNTIME_PROFILE: (
            "PP-OCRv5_mobile_det",
            "source_attested_tiny_crnn_bigru_ctc_v1",
            "mobile_det_trained_rec",
            "harmonic_v1",
        ),
        # V13 uses the exact public V9 detector/recognizer and strip2
        # inference path.  Only its bounded semantic repair is new, selected
        # by this immutable tuple rather than caller configuration.
        _PUBLIC_MOBILE_V13_RUNTIME_PROFILE: (
            "PP-OCRv5_mobile_det",
            "en_PP-OCRv5_mobile_rec",
            "mobile_det",
            "harmonic_v1",
        ),
        # V14 keeps V13's exact public OCR tuple and adds one separately
        # fingerprinted, text-only closed-shape repair revision.
        _PUBLIC_MOBILE_V14_RUNTIME_PROFILE: (
            "PP-OCRv5_mobile_det",
            "en_PP-OCRv5_mobile_rec",
            "mobile_det",
            "harmonic_v1",
        ),
        # V15 preserves V14's OCR and semantic-repair behavior while moving
        # only native CPU execution controls into a separately fingerprinted
        # candidate.  Its bounded fields are validated below before oneDNN
        # can be enabled.
        _PUBLIC_MOBILE_V15_ONEDNN_RUNTIME_PROFILE: (
            "PP-OCRv5_mobile_det",
            "en_PP-OCRv5_mobile_rec",
            "mobile_det",
            "harmonic_v1",
        ),
        # V16 keeps V15's models, detector, semantic repair and oneDNN
        # controls.  Its sole OCR change is a fingerprint-bound cubic resize
        # when a detected crop is normalized to recognizer height 48.
        _PUBLIC_MOBILE_V16_CUBIC_RUNTIME_PROFILE: (
            "PP-OCRv5_mobile_det",
            "en_PP-OCRv5_mobile_rec",
            "mobile_det",
            "harmonic_v1",
        ),
        # V17 keeps the linear recognizer resize and expands only the left and
        # right detector-quad edges by one bounded, immutable source-pixel
        # policy before the existing perspective warp.
        _PUBLIC_MOBILE_V17_PADDED_RUNTIME_PROFILE: (
            "PP-OCRv5_mobile_det",
            "en_PP-OCRv5_mobile_rec",
            "mobile_det",
            "harmonic_v1",
        ),
        # V18 retains V15's fast detector, strip2 pass, ordering, semantic
        # repair, and recognizer-only oneDNN controls.  Only the separately
        # attested recognizer bytes and their exact CTC alphabet contract
        # change under this discriminator.
        _PUBLIC_SERVER_RECOGNIZER_V18_RUNTIME_PROFILE: (
            "PP-OCRv5_mobile_det",
            "PP-OCRv5_server_rec",
            "mobile_det_server_rec",
            "harmonic_v1",
        ),
        # V19 preserves V15's measured fast path.  Only the recognizer bytes
        # change, and their warm-start receipt fixes the 438-class public
        # English CTC mapping verified again during startup below.
        _PUBLIC_FINETUNED_RECOGNIZER_V19_RUNTIME_PROFILE: (
            "PP-OCRv5_mobile_det",
            "source_attested_en_PP-OCRv5_mobile_rec_head_v2",
            "mobile_det_public_finetuned_rec",
            "harmonic_v1",
        ),
        # V19b leaves V19's model bytes, detector and ordinary recognition
        # path unchanged.  Its distinct discriminator alone permits one
        # conservative, geometry-bound padded retry for a B/L voyage value.
        _PUBLIC_FINETUNED_RECOGNIZER_V19_VOYAGE_RETRY_RUNTIME_PROFILE: (
            "PP-OCRv5_mobile_det",
            "source_attested_en_PP-OCRv5_mobile_rec_head_v2",
            "mobile_det_public_finetuned_rec",
            "harmonic_v1",
        ),
        # V20 changes only the source-attested recognizer bytes and applies
        # one fixed horizontal margin to every detector line before its one
        # ordinary recognition pass.  It has no document-specific retry.
        _PUBLIC_VOYAGE_REPLAY_RECOGNIZER_V20_RUNTIME_PROFILE: (
            "PP-OCRv5_mobile_det",
            "source_attested_en_PP-OCRv5_mobile_rec_voyage_replay_head_v3",
            "mobile_det_public_voyage_replay_rec",
            "harmonic_v1",
        ),
    }
    detection_limit = int(numeric["text_detection_limit_side_len"])
    model_specification = accepted_runtime_profiles.get(runtime_profile)
    if (
        model_specification is None
        or value.get("text_detection_model_name") != model_specification[0]
        or value.get("text_recognition_model_name") != model_specification[1]
        or int(numeric["cpu_threads"]) not in {1, 2, 4, 6}
        # V2 owns the customer-facing direct static engine and is kept below
        # the measured 30-second CPU lane.  Older unselected POC forms keep
        # their historical bounds only so their immutable test fixtures do
        # not acquire a new runtime meaning.
        or (
            detection_limit != _DIRECT_V2_DETECTION_LIMIT
            if runtime_profile
            in {
                ("2.0", "direct_paddle_text", "paddle-inference-static-v1"),
                ("3.0", "direct_paddle_mobile_text", "paddle-inference-static-v1"),
                (
                    "4.0",
                    "direct_paddle_mobile_text_harmonic",
                    "paddle-inference-static-v1",
                ),
                (
                    "5.0",
                    "direct_paddle_mobile_text_harmonic_recall",
                    "paddle-inference-static-v1",
                ),
                (
                    "8.0",
                    "direct_paddle_mobile_text_harmonic_recall_native960_lowbox",
                    "paddle-inference-static-v1",
                ),
                (
                    "9.0",
                    "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2",
                    "paddle-inference-static-v1",
                ),
                (
                    "10.0",
                    "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_grid2",
                    "paddle-inference-static-v1",
                ),
                (
                    "11.0",
                    "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_guarded_grid2",
                    "paddle-inference-static-v1",
                ),
                _SOURCE_ATTESTED_V12_RUNTIME_PROFILE,
                _PUBLIC_MOBILE_V13_RUNTIME_PROFILE,
                _PUBLIC_MOBILE_V14_RUNTIME_PROFILE,
                _PUBLIC_MOBILE_V15_ONEDNN_RUNTIME_PROFILE,
                _PUBLIC_MOBILE_V16_CUBIC_RUNTIME_PROFILE,
                _PUBLIC_MOBILE_V17_PADDED_RUNTIME_PROFILE,
                _PUBLIC_SERVER_RECOGNIZER_V18_RUNTIME_PROFILE,
                _PUBLIC_FINETUNED_RECOGNIZER_V19_RUNTIME_PROFILE,
                _PUBLIC_FINETUNED_RECOGNIZER_V19_VOYAGE_RETRY_RUNTIME_PROFILE,
                _PUBLIC_VOYAGE_REPLAY_RECOGNIZER_V20_RUNTIME_PROFILE,
            }
            else (
                detection_limit != _DIRECT_V6_DETECTION_LIMIT
                if runtime_profile
                in {
                    ("6.0", "direct_paddle_mobile_text_1024", "paddle-inference-static-v1"),
                    (
                        "7.0",
                        "direct_paddle_mobile_text_harmonic_recall_1024_lowbox",
                        "paddle-inference-static-v1",
                    ),
                }
                else detection_limit not in _LEGACY_DETECTION_LIMITS
            )
        )
        or (
            runtime_profile
            == ("4.0", "direct_paddle_mobile_text_harmonic", "paddle-inference-static-v1")
            and (
                int(numeric["cpu_threads"]) != 6
                or value.get("sidecar_timeout_seconds") != 30
                or float(numeric["text_detection_box_threshold"]) != 0.6
                or float(numeric["text_detection_unclip_ratio"]) != 1.5
            )
        )
        or (
            runtime_profile
            == ("6.0", "direct_paddle_mobile_text_1024", "paddle-inference-static-v1")
            and (
                int(numeric["cpu_threads"]) != 6
                or value.get("sidecar_timeout_seconds") != 30
                or float(numeric["text_detection_box_threshold"]) != 0.4
                or float(numeric["text_detection_unclip_ratio"]) != 1.5
            )
        )
        or (
            runtime_profile
            == (
                "7.0",
                "direct_paddle_mobile_text_harmonic_recall_1024_lowbox",
                "paddle-inference-static-v1",
            )
            and (
                int(numeric["cpu_threads"]) != 6
                or value.get("sidecar_timeout_seconds") != 30
                or float(numeric["text_detection_box_threshold"]) != 0.3
                or float(numeric["text_detection_unclip_ratio"]) != 1.5
            )
        )
        or (
            runtime_profile == _SOURCE_ATTESTED_V12_RUNTIME_PROFILE
            and (
                int(numeric["cpu_threads"]) != 6
                or value.get("sidecar_timeout_seconds") != 30
                or value.get("text_detection_model_directory") != "text-det"
                or value.get("text_recognition_model_directory") != "text-rec"
                or float(numeric["text_detection_box_threshold"]) != 0.3
                or float(numeric["text_detection_unclip_ratio"]) != 1.5
            )
        )
        or (
            runtime_profile
            in {_PUBLIC_MOBILE_V13_RUNTIME_PROFILE, _PUBLIC_MOBILE_V14_RUNTIME_PROFILE}
            and (
                int(numeric["cpu_threads"]) != 6
                or value.get("sidecar_timeout_seconds") != 30
                or float(numeric["text_detection_box_threshold"]) != 0.3
                or float(numeric["text_detection_unclip_ratio"]) != 1.5
            )
        )
        or (
            runtime_profile in _ONEDNN_RUNTIME_PROFILES
            and (
                int(numeric["cpu_threads"]) not in {4, 6}
                or value.get("sidecar_timeout_seconds") != 30
                or float(numeric["text_detection_box_threshold"]) != 0.3
                or float(numeric["text_detection_unclip_ratio"]) != 1.5
                or value.get("enable_mkldnn") is not True
                or int(numeric["mkldnn_cache_capacity"]) != 10
                or int(numeric["recognition_batch_size"]) not in {8, 16}
                or int(numeric["opencv_threads"]) != 1
                or value.get("warmup_shape_policy") != _V15_WARMUP_SHAPE_POLICY
                or value.get("onednn_model_policy") != "recognizer-only-v1"
                or (
                    runtime_profile == _PUBLIC_SERVER_RECOGNIZER_V18_RUNTIME_PROFILE
                    and (
                        value.get("recognition_crop_policy") != "detector-quad-v1"
                        or value.get("recognition_resize_interpolation") != "linear-v1"
                        or value.get("ctc_label_policy") != "append-space-v1"
                        or int(numeric["recognition_batch_size"]) != 8
                    )
                )
                or (
                    runtime_profile
                    in {
                        _PUBLIC_FINETUNED_RECOGNIZER_V19_RUNTIME_PROFILE,
                        _PUBLIC_FINETUNED_RECOGNIZER_V19_VOYAGE_RETRY_RUNTIME_PROFILE,
                    }
                    and (
                        value.get("ctc_label_policy") != "optional-space-v1"
                        or int(numeric["recognition_batch_size"]) != 8
                    )
                )
                or (
                    runtime_profile == _PUBLIC_VOYAGE_REPLAY_RECOGNIZER_V20_RUNTIME_PROFILE
                    and (
                        value.get("ctc_label_policy") != "optional-space-v1"
                        or int(numeric["recognition_batch_size"]) != 8
                        or value.get("recognition_crop_policy") != "horizontal-height-0p32-max8-v1"
                        or value.get("recognition_resize_interpolation") != "linear-v1"
                    )
                )
                or (
                    runtime_profile == _PUBLIC_MOBILE_V16_CUBIC_RUNTIME_PROFILE
                    and (
                        value.get("recognition_crop_policy") != "detector-quad-v1"
                        or value.get("recognition_resize_interpolation") != "cubic-v1"
                    )
                )
                or (
                    runtime_profile == _PUBLIC_MOBILE_V17_PADDED_RUNTIME_PROFILE
                    and (
                        value.get("recognition_crop_policy") != "horizontal-height-0p08-max4-v1"
                        or value.get("recognition_resize_interpolation") != "linear-v1"
                    )
                )
            )
        )
        or (
            runtime_profile
            == (
                "10.0",
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_grid2",
                "paddle-inference-static-v1",
            )
            and (
                int(numeric["cpu_threads"]) != 6
                or value.get("sidecar_timeout_seconds") != 30
                or float(numeric["text_detection_box_threshold"]) != 0.3
                or float(numeric["text_detection_unclip_ratio"]) != 1.5
            )
        )
        or (
            runtime_profile
            == (
                "11.0",
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_guarded_grid2",
                "paddle-inference-static-v1",
            )
            and (
                int(numeric["cpu_threads"]) != 6
                or value.get("sidecar_timeout_seconds") != 30
                or float(numeric["text_detection_box_threshold"]) != 0.3
                or float(numeric["text_detection_unclip_ratio"]) != 1.5
            )
        )
        or (
            runtime_profile
            == (
                "9.0",
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2",
                "paddle-inference-static-v1",
            )
            and (
                int(numeric["cpu_threads"]) != 6
                or value.get("sidecar_timeout_seconds") != 30
                or float(numeric["text_detection_box_threshold"]) != 0.3
                or float(numeric["text_detection_unclip_ratio"]) != 1.5
            )
        )
        or (
            runtime_profile
            == (
                "8.0",
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox",
                "paddle-inference-static-v1",
            )
            and (
                int(numeric["cpu_threads"]) != 6
                or value.get("sidecar_timeout_seconds") != 30
                or float(numeric["text_detection_box_threshold"]) != 0.3
                or float(numeric["text_detection_unclip_ratio"]) != 1.5
            )
        )
        or (
            runtime_profile
            == (
                "5.0",
                "direct_paddle_mobile_text_harmonic_recall",
                "paddle-inference-static-v1",
            )
            and (
                int(numeric["cpu_threads"]) != 6
                or value.get("sidecar_timeout_seconds") != 30
                or float(numeric["text_detection_box_threshold"]) != 0.4
                or float(numeric["text_detection_unclip_ratio"]) != 1.5
            )
        )
        or value.get("text_detection_limit_type") != "max"
        or float(numeric["text_detection_threshold"]) != 0.3
        # The lower .3 value is a V7--V12-only semantics boundary.  Do not
        # expand the historical global accepted set, or an older schema could
        # silently change its detector recall behavior.
        or (
            runtime_profile
            in {
                (
                    "7.0",
                    "direct_paddle_mobile_text_harmonic_recall_1024_lowbox",
                    "paddle-inference-static-v1",
                ),
                (
                    "8.0",
                    "direct_paddle_mobile_text_harmonic_recall_native960_lowbox",
                    "paddle-inference-static-v1",
                ),
                (
                    "9.0",
                    "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2",
                    "paddle-inference-static-v1",
                ),
                (
                    "10.0",
                    "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_grid2",
                    "paddle-inference-static-v1",
                ),
                (
                    "11.0",
                    "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_guarded_grid2",
                    "paddle-inference-static-v1",
                ),
                _SOURCE_ATTESTED_V12_RUNTIME_PROFILE,
                _PUBLIC_MOBILE_V13_RUNTIME_PROFILE,
                _PUBLIC_MOBILE_V14_RUNTIME_PROFILE,
                _PUBLIC_MOBILE_V15_ONEDNN_RUNTIME_PROFILE,
                _PUBLIC_MOBILE_V16_CUBIC_RUNTIME_PROFILE,
                _PUBLIC_MOBILE_V17_PADDED_RUNTIME_PROFILE,
                _PUBLIC_SERVER_RECOGNIZER_V18_RUNTIME_PROFILE,
                _PUBLIC_FINETUNED_RECOGNIZER_V19_RUNTIME_PROFILE,
                _PUBLIC_FINETUNED_RECOGNIZER_V19_VOYAGE_RETRY_RUNTIME_PROFILE,
                _PUBLIC_VOYAGE_REPLAY_RECOGNIZER_V20_RUNTIME_PROFILE,
            }
            and float(numeric["text_detection_box_threshold"]) != 0.3
        )
        or (
            runtime_profile
            not in {
                (
                    "7.0",
                    "direct_paddle_mobile_text_harmonic_recall_1024_lowbox",
                    "paddle-inference-static-v1",
                ),
                (
                    "8.0",
                    "direct_paddle_mobile_text_harmonic_recall_native960_lowbox",
                    "paddle-inference-static-v1",
                ),
                (
                    "9.0",
                    "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2",
                    "paddle-inference-static-v1",
                ),
                (
                    "10.0",
                    "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_grid2",
                    "paddle-inference-static-v1",
                ),
                (
                    "11.0",
                    "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_guarded_grid2",
                    "paddle-inference-static-v1",
                ),
                _SOURCE_ATTESTED_V12_RUNTIME_PROFILE,
                _PUBLIC_MOBILE_V13_RUNTIME_PROFILE,
                _PUBLIC_MOBILE_V14_RUNTIME_PROFILE,
                _PUBLIC_MOBILE_V15_ONEDNN_RUNTIME_PROFILE,
                _PUBLIC_MOBILE_V16_CUBIC_RUNTIME_PROFILE,
                _PUBLIC_MOBILE_V17_PADDED_RUNTIME_PROFILE,
                _PUBLIC_SERVER_RECOGNIZER_V18_RUNTIME_PROFILE,
                _PUBLIC_FINETUNED_RECOGNIZER_V19_RUNTIME_PROFILE,
                _PUBLIC_FINETUNED_RECOGNIZER_V19_VOYAGE_RETRY_RUNTIME_PROFILE,
                _PUBLIC_VOYAGE_REPLAY_RECOGNIZER_V20_RUNTIME_PROFILE,
            }
            and float(numeric["text_detection_box_threshold"]) not in {0.4, 0.6}
        )
        or float(numeric["text_detection_unclip_ratio"]) not in {1.5, 2.0}
    ):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_configuration_invalid")
    revision = _required_string(value["raster_text_profile_revision"])
    if not revision.startswith("synthetic-raster-text-v1:") or len(revision) != 49:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_configuration_invalid")
    return DirectPaddleProfile(
        raster_text_profile_revision=revision,
        extractor_revision=_required_string(value["extractor_revision"]),
        text_detection_model_directory=_required_string(value["text_detection_model_directory"]),
        text_recognition_model_directory=_required_string(
            value["text_recognition_model_directory"]
        ),
        cpu_threads=int(numeric["cpu_threads"]),
        text_detection_limit_side_len=detection_limit,
        text_detection_threshold=float(numeric["text_detection_threshold"]),
        text_detection_box_threshold=float(numeric["text_detection_box_threshold"]),
        text_detection_unclip_ratio=float(numeric["text_detection_unclip_ratio"]),
        model_spec=cast(
            Literal[
                "server_det",
                "mobile_det",
                "mobile_det_trained_rec",
                "mobile_det_server_rec",
                "mobile_det_public_finetuned_rec",
                "mobile_det_public_voyage_replay_rec",
            ],
            model_specification[2],
        ),
        row_band_mode=cast(Literal["legacy_max", "harmonic_v1"], model_specification[3]),
        detector_pass_mode=(
            "strip2"
            if runtime_profile
            in {
                (
                    "9.0",
                    "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2",
                    "paddle-inference-static-v1",
                ),
                _SOURCE_ATTESTED_V12_RUNTIME_PROFILE,
                _PUBLIC_MOBILE_V13_RUNTIME_PROFILE,
                _PUBLIC_MOBILE_V14_RUNTIME_PROFILE,
                _PUBLIC_MOBILE_V15_ONEDNN_RUNTIME_PROFILE,
                _PUBLIC_MOBILE_V16_CUBIC_RUNTIME_PROFILE,
                _PUBLIC_MOBILE_V17_PADDED_RUNTIME_PROFILE,
                _PUBLIC_SERVER_RECOGNIZER_V18_RUNTIME_PROFILE,
                _PUBLIC_FINETUNED_RECOGNIZER_V19_RUNTIME_PROFILE,
                _PUBLIC_FINETUNED_RECOGNIZER_V19_VOYAGE_RETRY_RUNTIME_PROFILE,
                _PUBLIC_VOYAGE_REPLAY_RECOGNIZER_V20_RUNTIME_PROFILE,
            }
            else "grid2"
            if runtime_profile
            == (
                "10.0",
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_grid2",
                "paddle-inference-static-v1",
            )
            else "guarded_grid2"
            if runtime_profile
            == (
                "11.0",
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_guarded_grid2",
                "paddle-inference-static-v1",
            )
            else "single"
        ),
        # This is reached only after ``runtime_profile`` matched one of the
        # exact tuples above.  Therefore the schema check cannot be supplied
        # by a caller as a runtime tuning knob: it means exactly V2--V12.
        pre_ready_warmup=runtime_profile[0]
        in {
            "2.0",
            "3.0",
            "4.0",
            "5.0",
            "6.0",
            "7.0",
            "8.0",
            "9.0",
            "10.0",
            "11.0",
            "12.0",
            "13.0",
            "14.0",
            "15.0",
            "16.0",
            "17.0",
            "18.0",
            "19.0",
            "20.0",
        },
        semantic_repair_mode=(
            "supported_docs_v2"
            if runtime_profile
            in {
                _PUBLIC_MOBILE_V14_RUNTIME_PROFILE,
                _PUBLIC_MOBILE_V15_ONEDNN_RUNTIME_PROFILE,
                _PUBLIC_MOBILE_V16_CUBIC_RUNTIME_PROFILE,
                _PUBLIC_MOBILE_V17_PADDED_RUNTIME_PROFILE,
                _PUBLIC_SERVER_RECOGNIZER_V18_RUNTIME_PROFILE,
                _PUBLIC_FINETUNED_RECOGNIZER_V19_RUNTIME_PROFILE,
                _PUBLIC_FINETUNED_RECOGNIZER_V19_VOYAGE_RETRY_RUNTIME_PROFILE,
                _PUBLIC_VOYAGE_REPLAY_RECOGNIZER_V20_RUNTIME_PROFILE,
            }
            else "supported_docs_v1"
            if runtime_profile == _PUBLIC_MOBILE_V13_RUNTIME_PROFILE
            else "none"
        ),
        enable_mkldnn=runtime_profile in _ONEDNN_RUNTIME_PROFILES,
        mkldnn_cache_capacity=(
            int(numeric["mkldnn_cache_capacity"])
            if runtime_profile in _ONEDNN_RUNTIME_PROFILES
            else 0
        ),
        recognition_batch_size=(
            int(numeric["recognition_batch_size"])
            if runtime_profile in _ONEDNN_RUNTIME_PROFILES
            else _RECOGNITION_BATCH_SIZE
        ),
        opencv_threads=(
            int(numeric["opencv_threads"]) if runtime_profile in _ONEDNN_RUNTIME_PROFILES else 0
        ),
        warmup_shape_policy=(
            _V15_WARMUP_SHAPE_POLICY
            if runtime_profile in _ONEDNN_RUNTIME_PROFILES
            else "legacy-minimal-v1"
        ),
        onednn_model_policy=(
            "recognizer-only-v1" if runtime_profile in _ONEDNN_RUNTIME_PROFILES else "disabled"
        ),
        recognition_crop_policy=(
            "horizontal-height-0p32-max8-v1"
            if runtime_profile == _PUBLIC_VOYAGE_REPLAY_RECOGNIZER_V20_RUNTIME_PROFILE
            else "horizontal-height-0p08-max4-v1"
            if runtime_profile == _PUBLIC_MOBILE_V17_PADDED_RUNTIME_PROFILE
            else "detector-quad-v1"
        ),
        recognition_resize_interpolation=(
            "cubic-v1"
            if runtime_profile == _PUBLIC_MOBILE_V16_CUBIC_RUNTIME_PROFILE
            else "linear-v1"
        ),
        ctc_label_policy=(
            "append-space-v1"
            if runtime_profile == _PUBLIC_SERVER_RECOGNIZER_V18_RUNTIME_PROFILE
            else "optional-space-v1"
        ),
        voyage_retry_mode=(
            "bill-of-lading-padded-v1"
            if runtime_profile == _PUBLIC_FINETUNED_RECOGNIZER_V19_VOYAGE_RETRY_RUNTIME_PROFILE
            else "none"
        ),
    )


def _verify_package_inventory(raw: bytes, *, expected_sha256: str) -> None:
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_package_drift")
    try:
        packages = json.loads(raw)["packages"]
    except (KeyError, TypeError, json.JSONDecodeError):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_package_invalid") from None
    if not isinstance(packages, list):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_package_invalid")
    observed: dict[str, str] = {}
    for item in packages:
        if (
            not isinstance(item, dict)
            or not isinstance(item.get("name"), str)
            or not isinstance(item.get("version"), str)
            or not item["name"]
            or not item["version"]
            or item["name"] in observed
        ):
            raise DirectPaddleSidecarError("direct_paddle_sidecar_package_invalid")
        observed[item["name"]] = item["version"]
    for package in _DIRECT_REQUIRED_PACKAGES:
        try:
            if importlib.metadata.version(package) != observed.get(package):
                raise DirectPaddleSidecarError("direct_paddle_sidecar_package_drift")
        except importlib.metadata.PackageNotFoundError:
            raise DirectPaddleSidecarError("direct_paddle_sidecar_package_drift") from None


def _sha256_model_artifact(path: Path) -> str:
    try:
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(_MODEL_HASH_CHUNK_BYTES):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_model_drift") from None


def _verify_model_inventory(
    raw: bytes,
    *,
    expected_sha256: str,
    model_root: Path,
    model_spec: Literal[
        "server_det",
        "mobile_det",
        "mobile_det_trained_rec",
        "mobile_det_server_rec",
        "mobile_det_public_finetuned_rec",
        "mobile_det_public_voyage_replay_rec",
    ] = "server_det",
) -> None:
    """Bind exactly one separately attested direct detector/recognizer set."""

    if model_spec not in {
        "server_det",
        "mobile_det",
        "mobile_det_trained_rec",
        "mobile_det_server_rec",
        "mobile_det_public_finetuned_rec",
        "mobile_det_public_voyage_replay_rec",
    }:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_model_invalid")
    expected_artifacts = (
        _DIRECT_MODEL_ARTIFACTS if model_spec == "server_det" else _DIRECT_MOBILE_MODEL_ARTIFACTS
    )

    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_model_drift")
    try:
        artifacts = json.loads(raw)["artifacts"]
        root = model_root.resolve(strict=True)
    except (KeyError, TypeError, OSError, json.JSONDecodeError):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_model_invalid") from None
    if model_root.is_symlink() or not root.is_dir() or not isinstance(artifacts, list):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_model_invalid")
    expected: set[str] = set()
    for item in artifacts:
        relative = item.get("relative_path") if isinstance(item, dict) else None
        digest = item.get("sha256") if isinstance(item, dict) else None
        if not isinstance(relative, str) or not isinstance(digest, str):
            raise DirectPaddleSidecarError("direct_paddle_sidecar_model_invalid")
        canonical_relative = relative.replace("\\", "/")
        if canonical_relative in expected or canonical_relative not in expected_artifacts:
            raise DirectPaddleSidecarError("direct_paddle_sidecar_model_invalid")
        try:
            candidate = (root / canonical_relative).resolve(strict=True)
            candidate.relative_to(root)
        except (OSError, ValueError):
            raise DirectPaddleSidecarError("direct_paddle_sidecar_model_invalid") from None
        if (
            candidate.is_symlink()
            or not candidate.is_file()
            or _sha256_model_artifact(candidate) != _strict_sha256(digest)
        ):
            raise DirectPaddleSidecarError("direct_paddle_sidecar_model_drift")
        expected.add(canonical_relative)
    if expected != expected_artifacts:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_model_invalid")
    actual: set[str] = set()
    try:
        for candidate in root.rglob("*"):
            if candidate.is_symlink():
                raise DirectPaddleSidecarError("direct_paddle_sidecar_model_invalid")
            if candidate.is_file():
                actual.add(candidate.relative_to(root).as_posix())
            elif not candidate.is_dir():
                raise DirectPaddleSidecarError("direct_paddle_sidecar_model_invalid")
    except DirectPaddleSidecarError:
        raise
    except (OSError, ValueError):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_model_invalid") from None
    if actual != expected_artifacts:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_model_drift")


def configuration_from_environment() -> tuple[DirectPaddleProfile, Path]:
    """Verify a direct-only local input set before native imports.

    These variables intentionally use a distinct prefix.  Adding this POC
    cannot change the established raster text runtime's environment contract.
    """

    prefix = "CUSTOMS_AGENT_RASTER_TEXT_DIRECT_PADDLE_"
    configuration_path = Path(_required_environment(prefix + "CONFIG_PATH"))
    configuration_raw = _read_regular_file(configuration_path, limit=_MAX_HEADER_BYTES)
    if hashlib.sha256(configuration_raw).hexdigest() != _strict_sha256(
        _required_environment(prefix + "CONFIG_SHA256")
    ):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_configuration_drift")
    profile = _profile_from_configuration(configuration_raw)
    if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != _strict_sha256(
        _required_environment(prefix + "MODULE_SHA256")
    ):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_module_drift")
    try:
        required = json.loads(_required_environment(prefix + "REQUIRED_PACKAGES"))
    except json.JSONDecodeError:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_package_invalid") from None
    if not isinstance(required, list) or tuple(required) != _DIRECT_REQUIRED_PACKAGES:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_package_invalid")
    _verify_package_inventory(
        _read_regular_file(
            Path(_required_environment(prefix + "PACKAGE_INVENTORY_PATH")), limit=_MAX_HEADER_BYTES
        ),
        expected_sha256=_strict_sha256(_required_environment(prefix + "PACKAGE_INVENTORY_SHA256")),
    )
    model_root = Path(_required_environment(prefix + "MODEL_ROOT"))
    _verify_model_inventory(
        _read_regular_file(
            Path(_required_environment(prefix + "MODEL_INVENTORY_PATH")), limit=_MAX_HEADER_BYTES
        ),
        expected_sha256=_strict_sha256(_required_environment(prefix + "MODEL_INVENTORY_SHA256")),
        model_root=model_root,
        model_spec=profile.model_spec,
    )
    return profile, model_root


@contextmanager
def deny_network_connections() -> Iterator[None]:
    """Deny DNS and peer creation before native libraries are imported."""

    original_socket = socket.socket
    original_create_connection = socket.create_connection
    original_getaddrinfo = socket.getaddrinfo
    original_gethostbyname = socket.gethostbyname
    original_gethostbyname_ex = socket.gethostbyname_ex

    def _blocked(*_args: object, **_kwargs: object) -> object:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_network_forbidden")

    class _BlockedSocket(original_socket):  # type: ignore[misc,valid-type]
        def connect(self, *_args: object, **_kwargs: object) -> None:
            _blocked()

        def connect_ex(self, *_args: object, **_kwargs: object) -> int:
            _blocked()
            raise AssertionError("unreachable")

        def sendto(self, *_args: object, **_kwargs: object) -> int:
            _blocked()
            raise AssertionError("unreachable")

    socket.socket = _BlockedSocket  # type: ignore[assignment,misc]
    socket.create_connection = _blocked  # type: ignore[assignment]
    socket.getaddrinfo = _blocked  # type: ignore[assignment]
    socket.gethostbyname = _blocked  # type: ignore[assignment]
    socket.gethostbyname_ex = _blocked  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket = original_socket  # type: ignore[assignment,misc]
        socket.create_connection = original_create_connection  # type: ignore[assignment]
        socket.getaddrinfo = original_getaddrinfo  # type: ignore[assignment]
        socket.gethostbyname = original_gethostbyname  # type: ignore[assignment]
        socket.gethostbyname_ex = original_gethostbyname_ex  # type: ignore[assignment]


def _read_request(stream: BinaryIO) -> tuple[str, str, tuple[_Page, ...]]:
    header = stream.readline(_MAX_HEADER_BYTES + 1)
    if not header or len(header) > _MAX_HEADER_BYTES or not header.endswith(b"\n"):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_input_invalid")
    try:
        value = json.loads(header)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_input_invalid") from None
    if not isinstance(value, dict) or value.get("protocol") != _PROTOCOL:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_input_invalid")
    source_sha256 = value.get("source_sha256")
    revision = value.get("raster_text_profile_revision")
    page_values = value.get("pages")
    if (
        not isinstance(source_sha256, str)
        or len(source_sha256) != 64
        or any(character not in _HEX for character in source_sha256)
        or not isinstance(revision, str)
        or not isinstance(page_values, list)
        or not 1 <= len(page_values) <= 20
    ):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_input_invalid")
    dimensions: list[tuple[int, int, int]] = []
    for index, item in enumerate(page_values, start=1):
        if not isinstance(item, dict):
            raise DirectPaddleSidecarError("direct_paddle_sidecar_input_invalid")
        page_number, width, height = item.get("page_number"), item.get("width"), item.get("height")
        if (
            page_number != index
            or isinstance(width, bool)
            or isinstance(height, bool)
            or not isinstance(width, int)
            or not isinstance(height, int)
            or width < 1
            or height < 1
            or width * height > 40_000_000
        ):
            raise DirectPaddleSidecarError("direct_paddle_sidecar_input_invalid")
        dimensions.append((page_number, width, height))
    size = sum(width * height * 3 for _page, width, height in dimensions)
    if size > _MAX_INPUT_BYTES:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_input_invalid")
    pixels = stream.read(size)
    if len(pixels) != size or stream.read(1):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_input_invalid")
    offset = 0
    pages: list[_Page] = []
    for page_number, width, height in dimensions:
        length = width * height * 3
        pages.append(_Page(page_number, width, height, pixels[offset : offset + length]))
        offset += length
    return source_sha256, revision, tuple(pages)


def _native_model_directory(*, model_root: Path, relative_directory: str) -> str:
    try:
        root = model_root.resolve(strict=True)
        directory = (root / relative_directory).resolve(strict=True)
        directory.relative_to(root)
        if directory.is_symlink() or not directory.is_dir():
            raise OSError
    except (OSError, ValueError):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_model_invalid") from None
    if sys.platform != "win32":
        return str(directory)
    try:
        import ctypes

        required = ctypes.windll.kernel32.GetShortPathNameW(str(directory), None, 0)
        if not isinstance(required, int) or required < 1:
            raise OSError
        buffer = ctypes.create_unicode_buffer(required + 1)
        written = ctypes.windll.kernel32.GetShortPathNameW(str(directory), buffer, len(buffer))
        if not isinstance(written, int) or written < 1 or written >= len(buffer):
            raise OSError
        if not buffer.value or not buffer.value.isascii():
            raise OSError
        return buffer.value
    except Exception:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_model_invalid") from None


def _native_dependencies() -> tuple[Any, Any, Any, Any]:
    """Import only the narrow local static-inference dependencies."""

    try:
        import cv2  # type: ignore[import-not-found]
        import numpy as np  # type: ignore[import-not-found]
        import paddle  # type: ignore[import-not-found]
        import pyclipper  # type: ignore[import-not-found]
    except Exception:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_runtime_unavailable") from None
    return cv2, np, paddle, pyclipper


def _load_characters(
    *, model_root: Path, relative_directory: str, model_spec: str
) -> tuple[str, ...]:
    try:
        raw = _read_regular_file(
            model_root / relative_directory / "config.json", limit=_MAX_HEADER_BYTES
        )
        value = json.loads(raw.decode("utf-8"))
        characters = value["PostProcess"]["character_dict"]
    except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_model_invalid") from None
    expected_count = (
        18_383
        if model_spec == "mobile_det_server_rec"
        else 436
        if model_spec
        in {
            "mobile_det_public_finetuned_rec",
            "mobile_det_public_voyage_replay_rec",
        }
        else None
    )
    if (
        not isinstance(characters, list)
        or not 1 <= len(characters) <= (18_383 if expected_count == 18_383 else 512)
        or (expected_count is not None and len(characters) != expected_count)
        or any(not isinstance(character, str) or not character for character in characters)
        or len(set(characters)) != len(characters)
    ):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_model_invalid")
    return tuple(characters)


def _new_predictor(
    *,
    paddle: Any,
    directory: str,
    cpu_threads: int,
    enable_mkldnn: bool = False,
    mkldnn_cache_capacity: int = 0,
    verify_mkldnn_disabled: bool = False,
) -> Any:
    try:
        config = paddle.inference.Config(  # type: ignore[attr-defined]
            str(Path(directory) / "inference.json"), str(Path(directory) / "inference.pdiparams")
        )
        config.disable_glog_info()
        config.disable_gpu()
    except Exception:
        raise DirectPaddleSidecarError(
            "direct_paddle_sidecar_predictor_config_unavailable"
        ) from None
    if enable_mkldnn:
        core = getattr(getattr(paddle, "base", None), "core", None)
        compiled_with_mkldnn = getattr(core, "is_compiled_with_mkldnn", None)
        enable = getattr(config, "enable_mkldnn", None)
        set_cache_capacity = getattr(config, "set_mkldnn_cache_capacity", None)
        enabled = getattr(config, "mkldnn_enabled", None)
        try:
            if (
                not callable(compiled_with_mkldnn)
                or not bool(compiled_with_mkldnn())
                or not callable(enable)
                or not callable(set_cache_capacity)
                or not callable(enabled)
                or mkldnn_cache_capacity != 10
            ):
                raise ValueError
            enable()
            set_cache_capacity(mkldnn_cache_capacity)
            if not bool(enabled()):
                raise ValueError
        except Exception:
            raise DirectPaddleSidecarError(
                "direct_paddle_sidecar_onednn_config_unavailable"
            ) from None
    else:
        try:
            if mkldnn_cache_capacity != 0:
                raise ValueError
            disable_mkldnn = getattr(config, "disable_mkldnn", None)
            if not callable(disable_mkldnn):
                raise ValueError
            disable_mkldnn()
            if verify_mkldnn_disabled:
                enabled = getattr(config, "mkldnn_enabled", None)
                if not callable(enabled) or bool(enabled()):
                    raise ValueError
        except Exception:
            raise DirectPaddleSidecarError("direct_paddle_sidecar_runtime_unavailable") from None
    try:
        # Paddle 3.3.1's JSON static predictor rejects memory_optimize_pass
        # for this exported seven-file model.  Do not enable an optional pass
        # that changes the native graph; the pinned model itself is the only
        # optimization boundary for this minimal direct POC.
        config.set_cpu_math_library_num_threads(cpu_threads)
        return paddle.inference.create_predictor(config)  # type: ignore[attr-defined]
    except Exception:
        raise DirectPaddleSidecarError(
            "direct_paddle_sidecar_onednn_predictor_unavailable"
            if enable_mkldnn
            else "direct_paddle_sidecar_runtime_unavailable"
        ) from None


def _engine(profile: DirectPaddleProfile, model_root: Path) -> _NativeEngine:
    cv2, np, paddle, pyclipper = _native_dependencies()
    if profile.opencv_threads:
        set_opencv_threads = getattr(cv2, "setNumThreads", None)
        get_opencv_threads = getattr(cv2, "getNumThreads", None)
        if not callable(set_opencv_threads) or not callable(get_opencv_threads):
            raise DirectPaddleSidecarError("direct_paddle_sidecar_runtime_unavailable")
        try:
            set_opencv_threads(profile.opencv_threads)
            if get_opencv_threads() != profile.opencv_threads:
                raise ValueError
        except Exception:
            raise DirectPaddleSidecarError(
                "direct_paddle_sidecar_opencv_threads_unavailable"
            ) from None
    detection_directory = _native_model_directory(
        model_root=model_root, relative_directory=profile.text_detection_model_directory
    )
    recognition_directory = _native_model_directory(
        model_root=model_root, relative_directory=profile.text_recognition_model_directory
    )
    return _NativeEngine(
        detector=_new_predictor(
            paddle=paddle,
            directory=detection_directory,
            cpu_threads=profile.cpu_threads,
            enable_mkldnn=(
                profile.enable_mkldnn and profile.onednn_model_policy != "recognizer-only-v1"
            ),
            mkldnn_cache_capacity=(
                profile.mkldnn_cache_capacity
                if profile.enable_mkldnn and profile.onednn_model_policy != "recognizer-only-v1"
                else 0
            ),
            verify_mkldnn_disabled=profile.onednn_model_policy == "recognizer-only-v1",
        ),
        recognizer=_new_predictor(
            paddle=paddle,
            directory=recognition_directory,
            cpu_threads=profile.cpu_threads,
            enable_mkldnn=profile.enable_mkldnn,
            mkldnn_cache_capacity=profile.mkldnn_cache_capacity,
        ),
        cv2=cv2,
        np=np,
        pyclipper=pyclipper,
        profile=profile,
        characters=_load_characters(
            model_root=model_root,
            relative_directory=profile.text_recognition_model_directory,
            model_spec=profile.model_spec,
        ),
    )


def _run_predictor(*, predictor: Any, batch: Any, np: Any) -> Any:
    try:
        input_names = list(predictor.get_input_names())  # type: ignore[attr-defined]
        output_names = list(predictor.get_output_names())  # type: ignore[attr-defined]
        if len(input_names) != 1 or len(output_names) != 1:
            raise ValueError
        input_handle = predictor.get_input_handle(input_names[0])  # type: ignore[attr-defined]
        input_handle.reshape(list(batch.shape))
        input_handle.copy_from_cpu(batch)
        predictor.run()  # type: ignore[attr-defined]
        output = predictor.get_output_handle(output_names[0]).copy_to_cpu()  # type: ignore[attr-defined]
        result = np.asarray(output)
        if not result.size or not bool(np.isfinite(result).all()):
            raise ValueError
        return result
    except Exception:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid") from None


def _warm_engine(*, engine: _NativeEngine) -> None:
    """Force the two native static graphs ready without accepting document data.

    The fixed zero tensors use the lower dynamic shapes declared in the
    attested PP-OCRv5 exports.  This allocates kernels and native buffers
    during the separately bounded service startup phase, never during a
    document request.  Any native/output error is fatal before ``ready`` is
    emitted so a partially initialized sidecar cannot serve OCR.
    """

    if engine.profile.warmup_shape_policy == _V15_WARMUP_SHAPE_POLICY:
        detector_shape = _V15_PRE_READY_WARMUP_DETECTOR_SHAPE
        recognizer_shape = _V15_PRE_READY_WARMUP_RECOGNIZER_SHAPE
    elif engine.profile.warmup_shape_policy == "legacy-minimal-v1":
        detector_shape = _PRE_READY_WARMUP_DETECTOR_SHAPE
        recognizer_shape = _PRE_READY_WARMUP_RECOGNIZER_SHAPE
    else:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_configuration_invalid")
    detector_batch = engine.np.zeros(detector_shape, dtype="float32")
    recognizer_batch = engine.np.zeros(recognizer_shape, dtype="float32")
    try:
        _run_predictor(predictor=engine.detector, batch=detector_batch, np=engine.np)
    except DirectPaddleSidecarError:
        if engine.profile.enable_mkldnn:
            raise DirectPaddleSidecarError(
                "direct_paddle_sidecar_onednn_detector_warmup_unavailable"
            ) from None
        raise
    try:
        recognizer_output = _run_predictor(
            predictor=engine.recognizer, batch=recognizer_batch, np=engine.np
        )
        exact_output_contract = engine.profile.model_spec in {
            "mobile_det_server_rec",
            "mobile_det_public_finetuned_rec",
            "mobile_det_public_voyage_replay_rec",
        }
        if exact_output_contract and (
            recognizer_output.ndim != 3
            or recognizer_output.shape[0] != 1
            or recognizer_output.shape[2] != len(engine.characters) + 2
        ):
            raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
    except DirectPaddleSidecarError:
        if engine.profile.enable_mkldnn:
            raise DirectPaddleSidecarError(
                "direct_paddle_sidecar_onednn_recognizer_warmup_unavailable"
            ) from None
        raise
    except Exception:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_runtime_unavailable") from None


def _detector_resize_shape(*, height: int, width: int, limit: int) -> tuple[int, int]:
    if (
        height < 1
        or width < 1
        or limit not in (_DIRECT_V2_DETECTION_LIMIT, *_LEGACY_DETECTION_LIMITS)
    ):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_input_invalid")
    ratio = min(1.0, float(limit) / float(max(height, width)))
    resized_height = max(32, int(round((height * ratio) / 32.0) * 32))
    resized_width = max(32, int(round((width * ratio) / 32.0) * 32))
    return min(resized_height, limit), min(resized_width, limit)


def _detector_input(*, image: Any, engine: _NativeEngine) -> tuple[Any, float, float]:
    np = engine.np
    cv2 = engine.cv2
    try:
        height, width = image.shape[:2]
        resized_height, resized_width = _detector_resize_shape(
            height=int(height), width=int(width), limit=engine.profile.text_detection_limit_side_len
        )
        bgr = image[:, :, ::-1]
        resized = cv2.resize(bgr, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
        normalized = resized.astype("float32") / np.float32(255.0)
        mean = np.asarray((0.485, 0.456, 0.406), dtype="float32")
        standard_deviation = np.asarray((0.229, 0.224, 0.225), dtype="float32")
        normalized = (normalized - mean) / standard_deviation
        batch = np.ascontiguousarray(normalized.transpose((2, 0, 1))[None, :, :, :])
        # A DB head may return a full-size map or a downsampled map.  Its
        # coordinates always span this resized page, so the postprocessor
        # uses the original page dimensions and the observed map dimensions
        # rather than assuming a fixed decoder stride.
        return batch, float(width), float(height)
    except DirectPaddleSidecarError:
        raise
    except Exception:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_input_invalid") from None


def _probability_map(*, output: Any, np: Any) -> Any:
    try:
        value = np.asarray(output)
        if value.ndim == 4 and tuple(value.shape[:2]) == (1, 1):
            value = value[0, 0]
        elif value.ndim == 3 and value.shape[0] == 1:
            value = value[0]
        else:
            raise ValueError
        if value.ndim != 2 or value.shape[0] < 2 or value.shape[1] < 2:
            raise ValueError
        if (
            not bool(np.isfinite(value).all())
            or float(value.min()) < -1e-6
            or float(value.max()) > 1.000001
        ):
            raise ValueError
        return np.clip(value.astype("float32"), 0.0, 1.0)
    except Exception:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid") from None


def _polygon_signed_area(points: Sequence[tuple[float, float]]) -> float:
    """Return a signed shoelace area for one already-validated polygon."""

    if len(points) < 3:
        return 0.0
    return 0.5 * sum(
        point[0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * point[1]
        for index, point in enumerate(points)
    )


def _ordered_quad(box: Any, *, np: Any) -> Any:
    """Return top-left, top-right, bottom-right, bottom-left for one rectangle."""

    try:
        points = np.asarray(box, dtype="float32")
        if tuple(points.shape) != (4, 2) or not bool(np.isfinite(points).all()):
            raise ValueError
        by_y = points[np.argsort(points[:, 1])]
        top = by_y[:2][np.argsort(by_y[:2, 0])]
        bottom = by_y[2:][np.argsort(by_y[2:, 0])]
        result = np.asarray((top[0], top[1], bottom[1], bottom[0]), dtype="float32")
        # Opposite edges of every proper rectangle are parallel.  Testing
        # their determinant would therefore reject all ordinary OCR boxes;
        # validate the quadrilateral's actual enclosed area instead.
        coordinates = tuple((float(point[0]), float(point[1])) for point in result)
        if abs(_polygon_signed_area(coordinates)) < 1e-4:
            raise ValueError
        return result
    except Exception:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid") from None


def _polygon_score(*, probability: Any, contour: Any, engine: _NativeEngine) -> float:
    try:
        mask = engine.np.zeros(probability.shape, dtype="uint8")
        points = engine.np.round(contour).astype("int32")
        engine.cv2.fillPoly(mask, [points], 1)
        score = float(engine.cv2.mean(probability, mask=mask)[0])
        if not math.isfinite(score):
            raise ValueError
        return score
    except Exception:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid") from None


def _unclip(*, box: Any, engine: _NativeEngine) -> Any | None:
    """Use the local polygon offset dependency rather than a heuristic scale."""

    try:
        contour = engine.np.asarray(box, dtype="float32")
        area = abs(float(engine.cv2.contourArea(contour)))
        perimeter = float(engine.cv2.arcLength(contour, True))
        if area <= 0 or perimeter <= 0:
            return None
        distance = area * engine.profile.text_detection_unclip_ratio / perimeter
        offset = engine.pyclipper.PyclipperOffset()
        offset.AddPath(
            engine.np.round(contour).astype("int32").tolist(),
            engine.pyclipper.JT_ROUND,
            engine.pyclipper.ET_CLOSEDPOLYGON,
        )
        candidates = offset.Execute(distance)
        if not candidates:
            return None
        candidate = max(
            candidates, key=lambda item: abs(float(engine.cv2.contourArea(engine.np.asarray(item))))
        )
        expanded = engine.np.asarray(candidate, dtype="float32")
        if expanded.ndim != 2 or expanded.shape[0] < 3:
            return None
        return engine.cv2.boxPoints(engine.cv2.minAreaRect(expanded))
    except Exception:
        return None


def _detected_boxes(
    *,
    probability: Any,
    original_width: float,
    original_height: float,
    engine: _NativeEngine,
) -> list[Any]:
    try:
        bitmap = (probability > engine.profile.text_detection_threshold).astype("uint8") * 255
        contours = engine.cv2.findContours(
            bitmap, engine.cv2.RETR_LIST, engine.cv2.CHAIN_APPROX_SIMPLE
        )[-2]
    except Exception:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid") from None
    boxes: list[Any] = []
    for contour in contours[:_MAX_DETECTION_CANDIDATES]:
        try:
            if contour is None or len(contour) < 3:
                continue
            initial = engine.cv2.boxPoints(engine.cv2.minAreaRect(contour))
            width = float(engine.np.linalg.norm(initial[0] - initial[1]))
            height = float(engine.np.linalg.norm(initial[1] - initial[2]))
            if min(width, height) < 3.0:
                continue
            if (
                _polygon_score(probability=probability, contour=contour, engine=engine)
                < engine.profile.text_detection_box_threshold
            ):
                continue
            expanded = _unclip(box=initial, engine=engine)
            if expanded is None:
                continue
            box = _ordered_quad(expanded, np=engine.np)
            width = float(engine.np.linalg.norm(box[0] - box[1]))
            height = float(engine.np.linalg.norm(box[1] - box[2]))
            if min(width, height) < 3.0:
                continue
            scale_x = original_width / float(probability.shape[1])
            scale_y = original_height / float(probability.shape[0])
            box[:, 0] = engine.np.clip(box[:, 0] * scale_x, 0.0, original_width - 1.0)
            box[:, 1] = engine.np.clip(box[:, 1] * scale_y, 0.0, original_height - 1.0)
            boxes.append(box)
        except DirectPaddleSidecarError:
            raise
        except Exception:
            continue
    return boxes


def _detected_scored_boxes(
    *,
    probability: Any,
    original_width: float,
    original_height: float,
    tile_index: int,
    engine: _NativeEngine,
) -> list[_StripDetectedBox]:
    """Return V9--V12 tile boxes with score needed for deterministic NMS.

    The legacy ``_detected_boxes`` result stays intentionally untouched for
    V1--V8.  V9--V12 alone need the pre-unclip contour score as deterministic
    cross-tile NMS priority, so candidate metadata stays local to those exact
    engine tuples.
    """

    if (
        isinstance(tile_index, bool)
        or not isinstance(tile_index, int)
        or tile_index not in {0, 1, 2, 3}
    ):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
    try:
        bitmap = (probability > engine.profile.text_detection_threshold).astype("uint8") * 255
        contours = engine.cv2.findContours(
            bitmap, engine.cv2.RETR_LIST, engine.cv2.CHAIN_APPROX_SIMPLE
        )[-2]
    except Exception:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid") from None
    boxes: list[_StripDetectedBox] = []
    for ordinal, contour in enumerate(contours[:_MAX_DETECTION_CANDIDATES]):
        try:
            if contour is None or len(contour) < 3:
                continue
            initial = engine.cv2.boxPoints(engine.cv2.minAreaRect(contour))
            width = float(engine.np.linalg.norm(initial[0] - initial[1]))
            height = float(engine.np.linalg.norm(initial[1] - initial[2]))
            if min(width, height) < 3.0:
                continue
            score = _polygon_score(probability=probability, contour=contour, engine=engine)
            if score < engine.profile.text_detection_box_threshold:
                continue
            expanded = _unclip(box=initial, engine=engine)
            if expanded is None:
                continue
            box = _ordered_quad(expanded, np=engine.np)
            width = float(engine.np.linalg.norm(box[0] - box[1]))
            height = float(engine.np.linalg.norm(box[1] - box[2]))
            if min(width, height) < 3.0:
                continue
            scale_x = original_width / float(probability.shape[1])
            scale_y = original_height / float(probability.shape[0])
            box[:, 0] = engine.np.clip(box[:, 0] * scale_x, 0.0, original_width - 1.0)
            box[:, 1] = engine.np.clip(box[:, 1] * scale_y, 0.0, original_height - 1.0)
            boxes.append(
                _StripDetectedBox(box=box, score=score, tile_index=tile_index, ordinal=ordinal)
            )
        except DirectPaddleSidecarError:
            raise
        except Exception:
            continue
    return boxes


def _strip2_horizontal_tiles(height: int) -> tuple[tuple[int, int], tuple[int, int]]:
    """Return V9's exact overlapping top/bottom detector strips."""

    if isinstance(height, bool) or not isinstance(height, int) or height < 1:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_input_invalid")
    overlap = math.ceil(height / 10.0)
    top_end = math.ceil((height + overlap) / 2.0)
    bottom_start = top_end - overlap
    if not (0 <= bottom_start < top_end <= height):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_input_invalid")
    return ((0, top_end), (bottom_start, height))


def _offset_quad_points_y(*, box: Any, y_offset: int) -> tuple[tuple[float, float], ...]:
    """Map one validated V9 tile-local quad into page coordinates."""

    if isinstance(y_offset, bool) or not isinstance(y_offset, int) or y_offset < 0:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
    return tuple((x, y + float(y_offset)) for x, y in _quad_points(box))


def _offset_detected_box_y(*, box: Any, y_offset: int, engine: _NativeEngine) -> Any:
    """Convert one globally mapped V9 quad back to the native array shape."""

    try:
        shifted = engine.np.asarray(
            _offset_quad_points_y(box=box, y_offset=y_offset), dtype="float32"
        )
        if tuple(shifted.shape) != (4, 2) or not bool(engine.np.isfinite(shifted).all()):
            raise ValueError
        return shifted
    except Exception:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid") from None


def _quad_points(box: Any) -> tuple[tuple[float, float], ...]:
    """Normalize an internal rotated quadrilateral for deterministic geometry."""

    try:
        points = tuple((float(point[0]), float(point[1])) for point in box)
    except (IndexError, TypeError, ValueError):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid") from None
    if (
        len(points) != 4
        or not all(math.isfinite(coordinate) for point in points for coordinate in point)
        or abs(_polygon_signed_area(points)) < 1e-6
    ):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
    return points


def _counterclockwise_polygon(
    points: tuple[tuple[float, float], ...],
) -> tuple[tuple[float, float], ...]:
    return points if _polygon_signed_area(points) > 0 else tuple(reversed(points))


def _edge_cross(
    start: tuple[float, float], end: tuple[float, float], point: tuple[float, float]
) -> float:
    return (end[0] - start[0]) * (point[1] - start[1]) - (end[1] - start[1]) * (point[0] - start[0])


def _edge_intersection(
    segment_start: tuple[float, float],
    segment_end: tuple[float, float],
    clip_start: tuple[float, float],
    clip_end: tuple[float, float],
) -> tuple[float, float]:
    """Intersect a subject segment with one clip edge without cv2 variation."""

    denominator = _edge_cross(
        (0.0, 0.0),
        (segment_end[0] - segment_start[0], segment_end[1] - segment_start[1]),
        (clip_end[0] - clip_start[0], clip_end[1] - clip_start[1]),
    )
    if abs(denominator) < 1e-12:
        return segment_end
    relative = (clip_start[0] - segment_start[0], clip_start[1] - segment_start[1])
    clip_direction = (clip_end[0] - clip_start[0], clip_end[1] - clip_start[1])
    factor = _edge_cross((0.0, 0.0), relative, clip_direction) / denominator
    return (
        segment_start[0] + factor * (segment_end[0] - segment_start[0]),
        segment_start[1] + factor * (segment_end[1] - segment_start[1]),
    )


def _convex_intersection_area(
    first: tuple[tuple[float, float], ...], second: tuple[tuple[float, float], ...]
) -> float:
    """Compute convex quad overlap with a deterministic Sutherland--Hodgman clip."""

    subject = list(_counterclockwise_polygon(first))
    clip = _counterclockwise_polygon(second)
    for index, clip_start in enumerate(clip):
        clip_end = clip[(index + 1) % len(clip)]
        if not subject:
            return 0.0
        output: list[tuple[float, float]] = []
        previous = subject[-1]
        previous_inside = _edge_cross(clip_start, clip_end, previous) >= -1e-9
        for current in subject:
            current_inside = _edge_cross(clip_start, clip_end, current) >= -1e-9
            if current_inside != previous_inside:
                output.append(_edge_intersection(previous, current, clip_start, clip_end))
            if current_inside:
                output.append(current)
            previous = current
            previous_inside = current_inside
        subject = output
    if len(subject) < 3:
        return 0.0
    area = abs(_polygon_signed_area(tuple(subject)))
    return area if math.isfinite(area) else 0.0


def _rotated_quad_iou(first: Any, second: Any) -> float:
    """Return finite IoU for two proper rotated detector quadrilaterals."""

    first_points = _quad_points(first)
    second_points = _quad_points(second)
    first_area = abs(_polygon_signed_area(first_points))
    second_area = abs(_polygon_signed_area(second_points))
    intersection = _convex_intersection_area(first_points, second_points)
    union = first_area + second_area - intersection
    if union <= 0 or not math.isfinite(union):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
    return min(1.0, max(0.0, intersection / union))


def _strip2_cross_tile_nms(
    candidates: Sequence[_StripDetectedBox], *, overlap_start: float, overlap_end: float
) -> list[Any]:
    """Deduplicate only V9 cross-strip rotated quads at fixed IoU .50.

    The two detector tiles can only produce a duplicate inside their fixed
    shared horizontal strip.  Bounding expensive polygon clips to boxes that
    both reach that band and vertically overlap keeps V9 in its request lane;
    a fixed comparison cap fails closed instead of silently changing output.
    """

    if (
        not math.isfinite(overlap_start)
        or not math.isfinite(overlap_end)
        or overlap_start < 0
        or overlap_start >= overlap_end
    ):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
    ranked: list[tuple[_StripDetectedBox, float, tuple[float, ...], float, float]] = []
    for candidate in candidates:
        if (
            isinstance(candidate.tile_index, bool)
            or not isinstance(candidate.tile_index, int)
            or candidate.tile_index not in {0, 1}
            or isinstance(candidate.ordinal, bool)
            or not isinstance(candidate.ordinal, int)
            or candidate.ordinal < 0
            or isinstance(candidate.score, bool)
            or not isinstance(candidate.score, (int, float))
            or not math.isfinite(candidate.score)
        ):
            raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
        points = _quad_points(candidate.box)
        vertical_start = min(point[1] for point in points)
        vertical_end = max(point[1] for point in points)
        ranked.append(
            (
                candidate,
                abs(_polygon_signed_area(points)),
                tuple(coordinate for point in points for coordinate in point),
                vertical_start,
                vertical_end,
            )
        )
    ranked.sort(
        key=lambda item: (
            -item[0].score,
            -item[1],
            item[0].tile_index,
            item[0].ordinal,
            item[2],
        )
    )
    kept: list[tuple[_StripDetectedBox, float, float]] = []
    comparisons = 0
    for candidate, _area, _coordinates, vertical_start, vertical_end in ranked:
        candidate_reaches_overlap = max(vertical_start, overlap_start) < min(
            vertical_end, overlap_end
        )
        suppressed = False
        for selected, selected_start, selected_end in kept:
            if (
                candidate.tile_index == selected.tile_index
                or not candidate_reaches_overlap
                or max(selected_start, overlap_start) >= min(selected_end, overlap_end)
                or max(vertical_start, selected_start) >= min(vertical_end, selected_end)
            ):
                continue
            if comparisons >= _STRIP2_MAX_IOU_COMPARISONS:
                raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
            comparisons += 1
            if _rotated_quad_iou(candidate.box, selected.box) >= 0.50:
                suppressed = True
                break
        if suppressed:
            continue
        kept.append((candidate, vertical_start, vertical_end))
    return [candidate.box for candidate, _start, _end in kept]


def _strip2_detected_boxes(*, image: Any, engine: _NativeEngine) -> list[Any]:
    """Detect V9's two RGB strips, map them globally, then cross-tile NMS."""

    try:
        height = int(image.shape[0])
        tiles = _strip2_horizontal_tiles(height)
        candidates: list[_StripDetectedBox] = []
        for tile_index, (start, end) in enumerate(tiles):
            # ``image`` is the original page RGB array.  ``_detector_input``
            # retains its existing RGB-to-BGR preprocessing after this exact
            # tile selection; recognition below receives the full BGR page.
            tile = image[start:end, :, :]
            detector_input, tile_width, tile_height = _detector_input(image=tile, engine=engine)
            probability = _probability_map(
                output=_run_predictor(
                    predictor=engine.detector, batch=detector_input, np=engine.np
                ),
                np=engine.np,
            )
            for candidate in _detected_scored_boxes(
                probability=probability,
                original_width=tile_width,
                original_height=tile_height,
                tile_index=tile_index,
                engine=engine,
            ):
                candidates.append(
                    _StripDetectedBox(
                        box=_offset_detected_box_y(
                            box=candidate.box, y_offset=start, engine=engine
                        ),
                        score=candidate.score,
                        tile_index=candidate.tile_index,
                        ordinal=candidate.ordinal,
                    )
                )
        return _strip2_cross_tile_nms(
            candidates, overlap_start=float(tiles[1][0]), overlap_end=float(tiles[0][1])
        )
    except DirectPaddleSidecarError:
        raise
    except Exception:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_execution_failed") from None


def _grid2_axis_tiles(
    *, dimension: int, overlap_numerator: int, overlap_denominator: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Build one fixed V10/V11 overlapping axis without runtime input."""

    if (
        isinstance(dimension, bool)
        or not isinstance(dimension, int)
        or dimension < 1
        or isinstance(overlap_numerator, bool)
        or not isinstance(overlap_numerator, int)
        or overlap_numerator < 1
        or isinstance(overlap_denominator, bool)
        or not isinstance(overlap_denominator, int)
        or overlap_denominator < overlap_numerator
    ):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_input_invalid")
    overlap = (dimension * overlap_numerator + overlap_denominator - 1) // overlap_denominator
    first_end = (dimension + overlap + 1) // 2
    second_start = first_end - overlap
    if not (0 <= second_start < first_end <= dimension):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_input_invalid")
    return ((0, first_end), (second_start, dimension))


def _grid2_tiles(width: int, height: int) -> tuple[_Grid2Tile, _Grid2Tile, _Grid2Tile, _Grid2Tile]:
    """Return exact V10/V11 TL/TR/BL/BR tiles with 15% x and 10% y overlap."""

    left, right = _grid2_axis_tiles(
        dimension=width,
        overlap_numerator=_GRID2_HORIZONTAL_OVERLAP_NUMERATOR,
        overlap_denominator=_GRID2_HORIZONTAL_OVERLAP_DENOMINATOR,
    )
    top, bottom = _grid2_axis_tiles(
        dimension=height,
        overlap_numerator=_GRID2_VERTICAL_OVERLAP_NUMERATOR,
        overlap_denominator=_GRID2_VERTICAL_OVERLAP_DENOMINATOR,
    )
    return (
        _Grid2Tile(ordinal=0, x_start=left[0], x_end=left[1], y_start=top[0], y_end=top[1]),
        _Grid2Tile(ordinal=1, x_start=right[0], x_end=right[1], y_start=top[0], y_end=top[1]),
        _Grid2Tile(ordinal=2, x_start=left[0], x_end=left[1], y_start=bottom[0], y_end=bottom[1]),
        _Grid2Tile(ordinal=3, x_start=right[0], x_end=right[1], y_start=bottom[0], y_end=bottom[1]),
    )


def _offset_quad_points_xy(
    *, box: Any, x_offset: int, y_offset: int
) -> tuple[tuple[float, float], ...]:
    """Map one validated V10/V11 tile-local quad into page coordinates."""

    if (
        isinstance(x_offset, bool)
        or not isinstance(x_offset, int)
        or x_offset < 0
        or isinstance(y_offset, bool)
        or not isinstance(y_offset, int)
        or y_offset < 0
    ):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
    return tuple((x + float(x_offset), y + float(y_offset)) for x, y in _quad_points(box))


def _offset_detected_box_xy(
    *, box: Any, x_offset: int, y_offset: int, engine: _NativeEngine
) -> Any:
    """Convert one globally mapped V10/V11 quad back to the native array shape."""

    try:
        shifted = engine.np.asarray(
            _offset_quad_points_xy(box=box, x_offset=x_offset, y_offset=y_offset),
            dtype="float32",
        )
        if tuple(shifted.shape) != (4, 2) or not bool(engine.np.isfinite(shifted).all()):
            raise ValueError
        return shifted
    except Exception:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid") from None


def _grid2_tile_overlap(
    first: _Grid2Tile, second: _Grid2Tile
) -> tuple[float, float, float, float] | None:
    """Return one pair's shared global rectangle, if it exists."""

    for tile in (first, second):
        if (
            isinstance(tile.ordinal, bool)
            or not isinstance(tile.ordinal, int)
            or tile.ordinal not in {0, 1, 2, 3}
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in (tile.x_start, tile.x_end, tile.y_start, tile.y_end)
            )
            or tile.x_start < 0
            or tile.y_start < 0
            or tile.x_start >= tile.x_end
            or tile.y_start >= tile.y_end
        ):
            raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
    x_start = max(first.x_start, second.x_start)
    x_end = min(first.x_end, second.x_end)
    y_start = max(first.y_start, second.y_start)
    y_end = min(first.y_end, second.y_end)
    if x_start >= x_end or y_start >= y_end:
        return None
    return (float(x_start), float(x_end), float(y_start), float(y_end))


def _quad_bounds(points: tuple[tuple[float, float], ...]) -> tuple[float, float, float, float]:
    return (
        min(point[0] for point in points),
        max(point[0] for point in points),
        min(point[1] for point in points),
        max(point[1] for point in points),
    )


def _bounds_intersect(
    first: tuple[float, float, float, float], second: tuple[float, float, float, float]
) -> bool:
    return max(first[0], second[0]) < min(first[1], second[1]) and max(first[2], second[2]) < min(
        first[3], second[3]
    )


def _grid2_cross_tile_nms(
    candidates: Sequence[_StripDetectedBox], *, tiles: Sequence[_Grid2Tile]
) -> list[Any]:
    """Deduplicate V10 only where two tiles and both boxes share an overlap.

    IoU clipping is deliberately deferred until the tiles overlap, both global
    quads reach that exact pair-overlap rectangle, and their bounding boxes
    overlap.  The fixed cap then fail-closes rather than silently changing a
    document result under an adversarially dense detector output.
    """

    if len(tiles) != 4 or tuple(tile.ordinal for tile in tiles) != (0, 1, 2, 3):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
    tiles_by_ordinal = {tile.ordinal: tile for tile in tiles}
    if len(tiles_by_ordinal) != 4:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
    ranked: list[
        tuple[_StripDetectedBox, float, tuple[float, ...], tuple[float, float, float, float]]
    ] = []
    for candidate in candidates:
        if (
            isinstance(candidate.tile_index, bool)
            or not isinstance(candidate.tile_index, int)
            or candidate.tile_index not in tiles_by_ordinal
            or isinstance(candidate.ordinal, bool)
            or not isinstance(candidate.ordinal, int)
            or candidate.ordinal < 0
            or isinstance(candidate.score, bool)
            or not isinstance(candidate.score, (int, float))
            or not math.isfinite(candidate.score)
        ):
            raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
        points = _quad_points(candidate.box)
        ranked.append(
            (
                candidate,
                abs(_polygon_signed_area(points)),
                tuple(coordinate for point in points for coordinate in point),
                _quad_bounds(points),
            )
        )
    ranked.sort(
        key=lambda item: (
            -item[0].score,
            -item[1],
            item[0].tile_index,
            item[0].ordinal,
            item[2],
        )
    )
    kept: list[tuple[_StripDetectedBox, tuple[float, float, float, float]]] = []
    overlaps: dict[tuple[int, int], tuple[float, float, float, float] | None] = {}
    comparisons = 0
    for candidate, _area, _coordinates, bounds in ranked:
        suppressed = False
        for selected, selected_bounds in kept:
            if candidate.tile_index == selected.tile_index:
                continue
            key = (
                min(candidate.tile_index, selected.tile_index),
                max(candidate.tile_index, selected.tile_index),
            )
            pair_overlap = overlaps.get(key)
            if key not in overlaps:
                pair_overlap = _grid2_tile_overlap(
                    tiles_by_ordinal[key[0]], tiles_by_ordinal[key[1]]
                )
                overlaps[key] = pair_overlap
            if (
                pair_overlap is None
                or not _bounds_intersect(bounds, pair_overlap)
                or not _bounds_intersect(selected_bounds, pair_overlap)
                or not _bounds_intersect(bounds, selected_bounds)
            ):
                continue
            if comparisons >= _GRID2_MAX_IOU_COMPARISONS:
                raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
            comparisons += 1
            if _rotated_quad_iou(candidate.box, selected.box) >= 0.50:
                suppressed = True
                break
        if not suppressed:
            kept.append((candidate, bounds))
    return [candidate.box for candidate, _bounds in kept]


def _guarded_grid2_touches_internal_tile_edge(
    *, box: Any, tile: _Grid2Tile, page_width: int, page_height: int
) -> bool:
    """Return whether one clipped V11 local quad reaches an internal seam.

    This intentionally examines the detector's *local, already clipped* box
    before it is offset into page coordinates.  A tile's page-facing edges
    (for example TL's left/top) are never seams, even when a clipped box
    reaches them.  Coordinates more than the fixed half-pixel tolerance
    outside their local clipped range are invalid rather than guessed.
    """

    if (
        isinstance(page_width, bool)
        or not isinstance(page_width, int)
        or page_width < 1
        or isinstance(page_height, bool)
        or not isinstance(page_height, int)
        or page_height < 1
        or isinstance(tile.ordinal, bool)
        or not isinstance(tile.ordinal, int)
        or tile.ordinal not in {0, 1, 2, 3}
        or any(
            isinstance(value, bool) or not isinstance(value, int)
            for value in (tile.x_start, tile.x_end, tile.y_start, tile.y_end)
        )
        or tile.x_start < 0
        or tile.y_start < 0
        or tile.x_start >= tile.x_end
        or tile.y_start >= tile.y_end
        or tile.x_end > page_width
        or tile.y_end > page_height
    ):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
    points = _quad_points(box)
    local_right = float(tile.x_end - tile.x_start - 1)
    local_bottom = float(tile.y_end - tile.y_start - 1)
    epsilon = _GUARDED_GRID2_INTERNAL_EDGE_EPSILON
    if (
        local_right < 0
        or local_bottom < 0
        or any(
            x < -epsilon or x > local_right + epsilon or y < -epsilon or y > local_bottom + epsilon
            for x, y in points
        )
    ):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
    min_x, max_x, min_y, max_y = _quad_bounds(points)
    return bool(
        (tile.x_start > 0 and min_x <= epsilon)
        or (tile.x_end < page_width and max_x >= local_right - epsilon)
        or (tile.y_start > 0 and min_y <= epsilon)
        or (tile.y_end < page_height and max_y >= local_bottom - epsilon)
    )


def _rotated_quad_overlap_metrics(first: Any, second: Any) -> tuple[float, float]:
    """Return IoU and intersection/min-area for V11's fixed seam rule."""

    first_points = _quad_points(first)
    second_points = _quad_points(second)
    first_area = abs(_polygon_signed_area(first_points))
    second_area = abs(_polygon_signed_area(second_points))
    intersection = _convex_intersection_area(first_points, second_points)
    union = first_area + second_area - intersection
    minimum_area = min(first_area, second_area)
    if (
        union <= 0
        or minimum_area <= 0
        or not math.isfinite(union)
        or not math.isfinite(minimum_area)
        or not math.isfinite(intersection)
    ):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
    return (
        min(1.0, max(0.0, intersection / union)),
        min(1.0, max(0.0, intersection / minimum_area)),
    )


def _guarded_grid2_cross_tile_nms(
    candidates: Sequence[_StripDetectedBox], *, tiles: Sequence[_Grid2Tile]
) -> list[Any]:
    """Keep V11 seam fragments unless an intact cross-tile alternative wins.

    Unlike V10's symmetric deduplication, V11 never suppresses a non-edge
    candidate and never lets an edge candidate suppress another edge
    candidate.  Non-edge candidates always rank first.  An edge candidate is
    removed only when a previously kept non-edge candidate from another
    overlapping tile has enough geometry evidence: normal rotated IoU .50 or
    containment (intersection/min-area) .80.  The same fixed 4,096 comparison
    cap still fail-closes dense output.
    """

    if len(tiles) != 4 or tuple(tile.ordinal for tile in tiles) != (0, 1, 2, 3):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
    tiles_by_ordinal = {tile.ordinal: tile for tile in tiles}
    if len(tiles_by_ordinal) != 4:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
    ranked: list[
        tuple[_StripDetectedBox, float, tuple[float, ...], tuple[float, float, float, float]]
    ] = []
    for candidate in candidates:
        if (
            isinstance(candidate.tile_index, bool)
            or not isinstance(candidate.tile_index, int)
            or candidate.tile_index not in tiles_by_ordinal
            or isinstance(candidate.ordinal, bool)
            or not isinstance(candidate.ordinal, int)
            or candidate.ordinal < 0
            or isinstance(candidate.score, bool)
            or not isinstance(candidate.score, (int, float))
            or not math.isfinite(candidate.score)
            or not isinstance(candidate.touches_internal_tile_edge, bool)
        ):
            raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
        points = _quad_points(candidate.box)
        ranked.append(
            (
                candidate,
                abs(_polygon_signed_area(points)),
                tuple(coordinate for point in points for coordinate in point),
                _quad_bounds(points),
            )
        )
    ranked.sort(
        key=lambda item: (
            item[0].touches_internal_tile_edge,
            -item[0].score,
            -item[1],
            item[0].tile_index,
            item[0].ordinal,
            item[2],
        )
    )
    kept: list[tuple[_StripDetectedBox, tuple[float, float, float, float]]] = []
    overlaps: dict[tuple[int, int], tuple[float, float, float, float] | None] = {}
    comparisons = 0
    for candidate, _area, _coordinates, bounds in ranked:
        # Only a seam-marked candidate may be suppressed.  The ordering above
        # guarantees every relevant intact alternative was already selected.
        suppressed = False
        if candidate.touches_internal_tile_edge:
            for selected, selected_bounds in kept:
                if (
                    selected.touches_internal_tile_edge
                    or candidate.tile_index == selected.tile_index
                ):
                    continue
                key = (
                    min(candidate.tile_index, selected.tile_index),
                    max(candidate.tile_index, selected.tile_index),
                )
                pair_overlap = overlaps.get(key)
                if key not in overlaps:
                    pair_overlap = _grid2_tile_overlap(
                        tiles_by_ordinal[key[0]], tiles_by_ordinal[key[1]]
                    )
                    overlaps[key] = pair_overlap
                if (
                    pair_overlap is None
                    or not _bounds_intersect(bounds, pair_overlap)
                    or not _bounds_intersect(selected_bounds, pair_overlap)
                    or not _bounds_intersect(bounds, selected_bounds)
                ):
                    continue
                if comparisons >= _GRID2_MAX_IOU_COMPARISONS:
                    raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
                comparisons += 1
                iou, intersection_over_min_area = _rotated_quad_overlap_metrics(
                    candidate.box, selected.box
                )
                if iou >= 0.50 or intersection_over_min_area >= 0.80:
                    suppressed = True
                    break
        if not suppressed:
            kept.append((candidate, bounds))
    return [candidate.box for candidate, _bounds in kept]


def _guarded_grid2_detected_boxes(*, image: Any, engine: _NativeEngine) -> list[Any]:
    """Detect V11's fixed grid and retain only explicitly justified seams."""

    try:
        height, width = (int(value) for value in image.shape[:2])
        tiles = _grid2_tiles(width, height)
        candidates: list[_StripDetectedBox] = []
        for tile in tiles:
            local_image = image[tile.y_start : tile.y_end, tile.x_start : tile.x_end, :]
            detector_input, local_width, local_height = _detector_input(
                image=local_image, engine=engine
            )
            if local_width != float(tile.x_end - tile.x_start) or local_height != float(
                tile.y_end - tile.y_start
            ):
                raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
            probability = _probability_map(
                output=_run_predictor(
                    predictor=engine.detector, batch=detector_input, np=engine.np
                ),
                np=engine.np,
            )
            for candidate in _detected_scored_boxes(
                probability=probability,
                original_width=local_width,
                original_height=local_height,
                tile_index=tile.ordinal,
                engine=engine,
            ):
                touches_internal_tile_edge = _guarded_grid2_touches_internal_tile_edge(
                    box=candidate.box, tile=tile, page_width=width, page_height=height
                )
                candidates.append(
                    _StripDetectedBox(
                        box=_offset_detected_box_xy(
                            box=candidate.box,
                            x_offset=tile.x_start,
                            y_offset=tile.y_start,
                            engine=engine,
                        ),
                        score=candidate.score,
                        tile_index=candidate.tile_index,
                        ordinal=candidate.ordinal,
                        touches_internal_tile_edge=touches_internal_tile_edge,
                    )
                )
        return _guarded_grid2_cross_tile_nms(candidates, tiles=tiles)
    except DirectPaddleSidecarError:
        raise
    except Exception:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_execution_failed") from None


def _grid2_detected_boxes(*, image: Any, engine: _NativeEngine) -> list[Any]:
    """Detect V10's fixed RGB grid, map boxes globally, and pair-NMS them."""

    try:
        height, width = (int(value) for value in image.shape[:2])
        tiles = _grid2_tiles(width, height)
        candidates: list[_StripDetectedBox] = []
        for tile in tiles:
            # ``image`` remains original page RGB until existing detector
            # preprocessing flips each exact local tile to BGR.  Recognition
            # below still receives one full original BGR page.
            local_image = image[tile.y_start : tile.y_end, tile.x_start : tile.x_end, :]
            detector_input, local_width, local_height = _detector_input(
                image=local_image, engine=engine
            )
            probability = _probability_map(
                output=_run_predictor(
                    predictor=engine.detector, batch=detector_input, np=engine.np
                ),
                np=engine.np,
            )
            for candidate in _detected_scored_boxes(
                probability=probability,
                original_width=local_width,
                original_height=local_height,
                tile_index=tile.ordinal,
                engine=engine,
            ):
                candidates.append(
                    _StripDetectedBox(
                        box=_offset_detected_box_xy(
                            box=candidate.box,
                            x_offset=tile.x_start,
                            y_offset=tile.y_start,
                            engine=engine,
                        ),
                        score=candidate.score,
                        tile_index=candidate.tile_index,
                        ordinal=candidate.ordinal,
                    )
                )
        return _grid2_cross_tile_nms(candidates, tiles=tiles)
    except DirectPaddleSidecarError:
        raise
    except Exception:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_execution_failed") from None


def _recognition_quad(
    *,
    image: Any,
    ordered: Any,
    engine: _NativeEngine,
    policy_override: Literal[
        "detector-quad-v1",
        "horizontal-height-0p08-max4-v1",
        "horizontal-height-0p16-max8-v1",
        "horizontal-height-0p32-max8-v1",
    ]
    | None = None,
) -> Any | None:
    """Return the immutable recognizer crop quad for one accepted profile."""

    policy = engine.profile.recognition_crop_policy if policy_override is None else policy_override
    if policy == "detector-quad-v1":
        return ordered
    if policy not in {
        "horizontal-height-0p08-max4-v1",
        "horizontal-height-0p16-max8-v1",
        "horizontal-height-0p32-max8-v1",
    }:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_configuration_invalid")
    try:
        image_height, image_width = image.shape[:2]
        if image_height < 2 or image_width < 2 or not bool(engine.np.isfinite(ordered).all()):
            return None
        top = ordered[1] - ordered[0]
        bottom = ordered[2] - ordered[3]
        top_length = float(engine.np.linalg.norm(top))
        bottom_length = float(engine.np.linalg.norm(bottom))
        left_height = float(engine.np.linalg.norm(ordered[0] - ordered[3]))
        right_height = float(engine.np.linalg.norm(ordered[1] - ordered[2]))
        if min(top_length, bottom_length, left_height, right_height) <= 0:
            return None
        horizontal = top / top_length + bottom / bottom_length
        horizontal_length = float(engine.np.linalg.norm(horizontal))
        if not math.isfinite(horizontal_length) or horizontal_length <= 0:
            return None
        horizontal = horizontal / horizontal_length
        average_height = (left_height + right_height) / 2.0
        if policy == "horizontal-height-0p08-max4-v1":
            margin = min(4.0, max(1.0, average_height * 0.08))
        elif policy == "horizontal-height-0p16-max8-v1":
            margin = min(8.0, max(1.0, average_height * 0.16))
        else:
            # V20 uses one integer source-pixel margin for every line.  The
            # ceil and 2..8 clamp are part of its immutable crop contract.
            margin = float(min(8, max(2, math.ceil(average_height * 0.32))))
        if not math.isfinite(margin):
            return None
        expanded = ordered.astype("float32", copy=True)
        expanded[0] -= horizontal * margin
        expanded[3] -= horizontal * margin
        expanded[1] += horizontal * margin
        expanded[2] += horizontal * margin
        expanded[:, 0] = engine.np.clip(expanded[:, 0], 0.0, float(image_width - 1))
        expanded[:, 1] = engine.np.clip(expanded[:, 1], 0.0, float(image_height - 1))
        if not bool(engine.np.isfinite(expanded).all()):
            return None
        return expanded
    except DirectPaddleSidecarError:
        raise
    except Exception:
        return None


def _crop_quad(
    *,
    image: Any,
    box: Any,
    engine: _NativeEngine,
    policy_override: Literal[
        "detector-quad-v1",
        "horizontal-height-0p08-max4-v1",
        "horizontal-height-0p16-max8-v1",
        "horizontal-height-0p32-max8-v1",
    ]
    | None = None,
) -> Any | None:
    try:
        ordered = _ordered_quad(box, np=engine.np)
        ordered = _recognition_quad(
            image=image,
            ordered=ordered,
            engine=engine,
            policy_override=policy_override,
        )
        if ordered is None:
            return None
        width = int(
            max(
                engine.np.linalg.norm(ordered[0] - ordered[1]),
                engine.np.linalg.norm(ordered[2] - ordered[3]),
            )
        )
        height = int(
            max(
                engine.np.linalg.norm(ordered[0] - ordered[3]),
                engine.np.linalg.norm(ordered[1] - ordered[2]),
            )
        )
        if width < 2 or height < 2:
            return None
        target = engine.np.asarray(
            ((0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1)), dtype="float32"
        )
        matrix = engine.cv2.getPerspectiveTransform(ordered, target)
        crop = engine.cv2.warpPerspective(
            image,
            matrix,
            (width, height),
            flags=engine.cv2.INTER_CUBIC,
            borderMode=engine.cv2.BORDER_REPLICATE,
        )
        if crop is None or crop.size == 0:
            return None
        if crop.shape[0] >= crop.shape[1] * 1.5:
            crop = engine.cv2.rotate(crop, engine.cv2.ROTATE_90_CLOCKWISE)
        return crop
    except DirectPaddleSidecarError:
        raise
    except Exception:
        return None


def _recognition_input(*, crop: Any, engine: _NativeEngine) -> Any:
    try:
        height, width = crop.shape[:2]
        if height < 1 or width < 1:
            raise ValueError
        resized_width = min(
            _RECOGNITION_MAX_WIDTH,
            max(1, int(math.ceil(48.0 * float(width) / float(height)))),
        )
        if engine.profile.recognition_crop_policy not in {
            "detector-quad-v1",
            "horizontal-height-0p08-max4-v1",
            "horizontal-height-0p32-max8-v1",
        }:
            raise ValueError
        interpolation = (
            engine.cv2.INTER_CUBIC
            if engine.profile.recognition_resize_interpolation == "cubic-v1"
            else engine.cv2.INTER_LINEAR
            if engine.profile.recognition_resize_interpolation == "linear-v1"
            else None
        )
        if interpolation is None:
            raise ValueError
        resized = engine.cv2.resize(crop, (resized_width, 48), interpolation=interpolation)
        normalized = (
            resized.astype("float32") / engine.np.float32(255.0) - engine.np.float32(0.5)
        ) / engine.np.float32(0.5)
        channel_first = normalized.transpose((2, 0, 1))
        return engine.np.ascontiguousarray(channel_first)
    except Exception:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid") from None


def _sequence(value: object) -> list[object]:
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return list(value)
    tolist = getattr(value, "tolist", None)
    if not callable(tolist):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
    try:
        converted = tolist()
    except Exception:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid") from None
    if not isinstance(converted, Sequence) or isinstance(converted, (str, bytes, bytearray)):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
    return list(converted)


def _ctc_decode_v18_ndarray(
    probabilities: Any,
    *,
    characters: tuple[str, ...],
    np: Any,
) -> tuple[str, float]:
    """Decode the V18 server recognizer without materializing Python rows."""

    try:
        if probabilities.ndim != 2:
            raise ValueError
        timestep_count, class_count = probabilities.shape
        if (
            not isinstance(timestep_count, int)
            or not 1 <= timestep_count <= 1_024
            or not isinstance(class_count, int)
            or class_count != len(characters) + 2
            or np.issubdtype(probabilities.dtype, np.bool_)
            or not (
                np.issubdtype(probabilities.dtype, np.integer)
                or np.issubdtype(probabilities.dtype, np.floating)
            )
        ):
            raise ValueError

        # Keep the model's ordinary float32 tensor in place.  Only logit rows
        # need a float64 copy for the stable softmax denominator; exported
        # softmax rows therefore avoid both the old 735k-element Python list
        # and an unnecessary full-tensor copy.
        values = np.asarray(probabilities)
        if not bool(np.isfinite(values).all()):
            raise ValueError
        row_minimums = values.min(axis=1)
        row_maximums = values.max(axis=1)
        with np.errstate(over="ignore", invalid="ignore"):
            row_sums = values.sum(axis=1, dtype=np.float64)
        probability_rows = (
            (row_minimums >= 0.0) & (row_maximums <= 1.0) & (row_sums >= 0.95) & (row_sums <= 1.05)
        )

        best = values.argmax(axis=1)
        best_scores = np.empty(timestep_count, dtype=np.float64)
        probability_indices = np.flatnonzero(probability_rows)
        if probability_indices.size:
            best_scores[probability_indices] = values[
                probability_indices, best[probability_indices]
            ]

        logit_indices = np.flatnonzero(~probability_rows)
        if logit_indices.size:
            logit_values = np.asarray(values[logit_indices], dtype=np.float64)
            maxima = logit_values.max(axis=1)
            with np.errstate(over="ignore", under="ignore", invalid="ignore"):
                totals = np.exp(logit_values - maxima[:, None]).sum(axis=1, dtype=np.float64)
            if not bool(np.isfinite(totals).all()) or bool((totals <= 0.0).any()):
                raise ValueError
            best_scores[logit_indices] = 1.0 / totals

        previous = np.empty_like(best)
        previous[0] = -1
        previous[1:] = best[:-1]
        retained = np.flatnonzero((best != 0) & (best != previous))
        text: list[str] = []
        score_total = 0.0
        for position in retained:
            label_index = int(best[position])
            text.append(characters[label_index - 1] if label_index <= len(characters) else " ")
            score_total += float(best_scores[position])

        decoded = "".join(text).strip()
        if len(decoded) > 8_000:
            raise ValueError
        return decoded, (score_total / int(retained.size) if retained.size else 0.0)
    except DirectPaddleSidecarError:
        raise
    except Exception:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid") from None


def _ctc_decode(
    probabilities: object,
    *,
    characters: tuple[str, ...],
    label_policy: Literal["optional-space-v1", "append-space-v1"] = "optional-space-v1",
) -> tuple[str, float]:
    """Decode one CTC sequence, including the model's implicit space class."""

    # ``append-space-v1`` uniquely identifies the V18 PP-OCRv5 server
    # recognizer.  Its 18,385-class native output is prohibitively expensive
    # to turn into nested Python lists.  Restrict the vectorized path to an
    # actual NumPy ndarray; all prior profiles and test doubles retain the
    # original scalar decoder below byte-for-byte.
    np = sys.modules.get("numpy")
    if (
        label_policy == "append-space-v1"
        and np is not None
        and isinstance(probabilities, np.ndarray)
    ):
        return _ctc_decode_v18_ndarray(probabilities, characters=characters, np=np)

    timesteps = _sequence(probabilities)
    if not timesteps or len(timesteps) > 1_024:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
    text: list[str] = []
    scores: list[float] = []
    previous = -1
    labels: tuple[str, ...] | None = None
    for step in timesteps:
        values = _sequence(step)
        if labels is None:
            if label_policy == "append-space-v1" and len(values) == len(characters) + 2:
                # PP-OCRv5_server_rec is exported with BaseRecLabelDecode's
                # default ``use_space_char=True``.  PaddleX therefore appends
                # one final space class even though the pinned multilingual
                # list already contains a space.  Preserve both indices as
                # the same output character and keep blank at zero.
                labels = ("", *characters, " ")
            elif label_policy == "optional-space-v1" and len(values) == len(characters) + 1:
                labels = ("", *characters)
            elif (
                label_policy == "optional-space-v1"
                and len(values) == len(characters) + 2
                and " " not in characters
            ):
                # The pinned v5 English recognizer's export includes the
                # decoder's default optional space token after the manifest
                # dictionary.  Only this exact surplus is accepted.
                labels = ("", *characters, " ")
            else:
                raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
        if len(values) != len(labels):
            raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
        numeric: list[float] = []
        for value in values:
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
            ):
                raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
            numeric.append(float(value))
        # Exports normally end in softmax.  Supporting a finite logit tensor
        # makes this POC reject neither compatible Paddle static export form.
        if all(0.0 <= value <= 1.0 for value in numeric) and 0.95 <= sum(numeric) <= 1.05:
            normalized = numeric
        else:
            maximum = max(numeric)
            exponents = [math.exp(value - maximum) for value in numeric]
            total = sum(exponents)
            if not math.isfinite(total) or total <= 0:
                raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
            normalized = [value / total for value in exponents]
        best = max(range(len(normalized)), key=lambda index: normalized[index])
        if best != 0 and best != previous:
            text.append(labels[best])
            scores.append(normalized[best])
        previous = best
    decoded = "".join(text).strip()
    if len(decoded) > 8_000:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
    return decoded, (sum(scores) / len(scores) if scores else 0.0)


def _recognize_crops(*, image: Any, boxes: Sequence[Any], engine: _NativeEngine) -> list[_Line]:
    pending: list[tuple[Any, Any]] = []
    for box in boxes:
        crop = _crop_quad(image=image, box=box, engine=engine)
        if crop is not None:
            pending.append((box, crop))
    prepared = [(box, _recognition_input(crop=crop, engine=engine)) for box, crop in pending]
    # Dynamic-width recognition avoids truncating a long invoice line.  Width
    # ordering only reduces zero padding; document order is reconstructed from
    # geometry after decoding.
    prepared.sort(key=lambda item: int(item[1].shape[2]))
    output: list[_Line] = []
    for start in range(0, len(prepared), engine.profile.recognition_batch_size):
        batch_items = prepared[start : start + engine.profile.recognition_batch_size]
        try:
            batch_width = max(int(item.shape[2]) for _box, item in batch_items)
            batch = engine.np.zeros((len(batch_items), 3, 48, batch_width), dtype="float32")
            for index, (_box, item) in enumerate(batch_items):
                batch[index, :, :, : item.shape[2]] = item
            recognition = _run_predictor(predictor=engine.recognizer, batch=batch, np=engine.np)
            if recognition.ndim != 3 or recognition.shape[0] != len(batch_items):
                raise ValueError
        except DirectPaddleSidecarError:
            raise
        except Exception:
            raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid") from None
        for index, (box, _input) in enumerate(batch_items):
            text, confidence = _ctc_decode(
                recognition[index],
                characters=engine.characters,
                label_policy=engine.profile.ctc_label_policy,
            )
            if text:
                polygon = tuple((float(point[0]), float(point[1])) for point in box)
                output.append(_Line(text=text, confidence=confidence, polygon=polygon))
    return output


def _recognize_padded_voyage_consensus(
    *, image: Any, line: _Line, engine: _NativeEngine
) -> tuple[_Line, _Line] | None:
    """Read one observed line at two fixed margins in one predictor batch.

    Detection is deliberately not repeated.  The original detector polygon is
    expanded only on its horizontal axis by two fixed source-pixel policies.
    Both crops are sent through one recognizer call as a batch of two; the
    caller accepts a change only when the independently prepared crops agree.
    """

    try:
        box = engine.np.asarray(line.polygon, dtype="float32")
        prepared_inputs: list[Any] = []
        for policy in (
            "horizontal-height-0p08-max4-v1",
            "horizontal-height-0p16-max8-v1",
        ):
            crop = _crop_quad(
                image=image,
                box=box,
                engine=engine,
                policy_override=policy,
            )
            if crop is None:
                return None
            prepared = _recognition_input(crop=crop, engine=engine)
            prepared_width = int(prepared.shape[2])
            if prepared_width < 1 or prepared_width > _RECOGNITION_MAX_WIDTH:
                raise ValueError
            prepared_inputs.append(prepared)
        first_input, second_input = prepared_inputs
        if first_input.shape == second_input.shape and bool(
            engine.np.allclose(
                first_input,
                second_input,
                rtol=0.0,
                atol=_VOYAGE_RETRY_PREPARED_INPUT_ABS_TOLERANCE,
            )
        ):
            # Clipping or integer warp rounding can collapse the two nominal
            # margins into the same recognizer input.  One observation must
            # never be counted twice as independent consensus.
            return None
        batch_width = max(int(prepared.shape[2]) for prepared in prepared_inputs)
        batch = engine.np.zeros((2, 3, 48, batch_width), dtype="float32")
        for index, prepared in enumerate(prepared_inputs):
            prepared_width = int(prepared.shape[2])
            batch[index, :, :, :prepared_width] = prepared
        recognition = _run_predictor(predictor=engine.recognizer, batch=batch, np=engine.np)
        if recognition.ndim != 3 or recognition.shape[0] != 2:
            raise ValueError
        candidates: list[_Line] = []
        for index in range(2):
            text, confidence = _ctc_decode(
                recognition[index],
                characters=engine.characters,
                label_policy=engine.profile.ctc_label_policy,
            )
            if not text:
                return None
            candidates.append(_Line(text=text, confidence=confidence, polygon=line.polygon))
        return candidates[0], candidates[1]
    except DirectPaddleSidecarError:
        raise
    except Exception:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid") from None


def _ordered_lines(
    lines: Sequence[_Line], *, row_band_mode: Literal["legacy_max", "harmonic_v1"]
) -> list[_Line]:
    """Order decoded lines using the profile-bound legacy or harmonic band."""

    if row_band_mode not in {"legacy_max", "harmonic_v1"}:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")

    positioned: list[tuple[_Line, float, float, float, int]] = []
    for ordinal, line in enumerate(lines):
        if len(line.polygon) < 3:
            raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
        xs = [point[0] for point in line.polygon]
        ys = [point[1] for point in line.polygon]
        if not all(math.isfinite(value) for value in (*xs, *ys)):
            raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
        top, bottom = min(ys), max(ys)
        height = bottom - top
        if height <= 0:
            raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
        positioned.append((line, (top + bottom) / 2.0, min(xs), height, ordinal))
    positioned.sort(key=lambda item: (item[1], item[2], item[4]))
    rows: list[list[tuple[_Line, float, float, float, int]]] = []
    row_center: list[float] = []
    row_height: list[float] = []
    for item in positioned:
        _line, center, _left, height, _ordinal = item
        if row_band_mode == "legacy_max":
            row_band = 0.6 * max(height, row_height[-1]) if rows else 0.0
        elif rows:
            denominator = height + row_height[-1]
            if not math.isfinite(denominator) or denominator <= 0:
                raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
            row_band = 0.6 * (2.0 * height * row_height[-1] / denominator)
        else:
            row_band = 0.0
        if rows and abs(center - row_center[-1]) <= row_band:
            rows[-1].append(item)
            row_center[-1] = sum(entry[1] for entry in rows[-1]) / len(rows[-1])
            row_height[-1] = sum(entry[3] for entry in rows[-1]) / len(rows[-1])
        else:
            rows.append([item])
            row_center.append(center)
            row_height.append(height)
    return [
        line
        for row in rows
        for line, _center, _left, _height, _ordinal in sorted(
            row, key=lambda item: (item[2], item[1], item[4])
        )
    ]


def _line_bounds(line: _Line) -> tuple[float, float, float, float]:
    """Return already-validated line bounds without discarding its polygon."""

    xs = [point[0] for point in line.polygon]
    ys = [point[1] for point in line.polygon]
    if len(xs) < 3 or not all(math.isfinite(value) for value in (*xs, *ys)):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
    left, top, right, bottom = min(xs), min(ys), max(xs), max(ys)
    if right <= left or bottom <= top:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
    return left, top, right, bottom


def _normalized_voyage_label(text: str) -> str:
    """Normalize only visible label typography, never a field value."""

    return " ".join(re.sub(r"[^A-Za-z0-9]+", " ", text).casefold().split())


def _unique_bill_of_lading_voyage_value_line(lines: Sequence[_Line]) -> int | None:
    """Find one geometry-bound B/L voyage value without template coordinates.

    This consumes OCR text and detector polygons only.  It requires one exact
    case-insensitive title, one visible VOYAGE/VOYAGE NO label, and one bounded
    ASCII identifier directly below and in the same observed column.  Any
    duplicate, malformed polygon, distant row or competing value fails closed.
    """

    titles = [line for line in lines if line.text.casefold() == "bill of lading"]
    labels = [
        (index, line)
        for index, line in enumerate(lines)
        if _normalized_voyage_label(line.text) in {"voyage", "voyage no"}
    ]
    if len(titles) != 1 or len(labels) != 1:
        return None
    label_index, label = labels[0]
    try:
        label_left, label_top, label_right, label_bottom = _line_bounds(label)
        label_height = label_bottom - label_top
        candidates: list[int] = []
        for index, line in enumerate(lines):
            if index == label_index or _BOUNDED_ASCII_FIELD_IDENTIFIER.fullmatch(line.text) is None:
                continue
            value_left, value_top, value_right, value_bottom = _line_bounds(line)
            value_height = value_bottom - value_top
            label_center_y = (label_top + label_bottom) / 2.0
            value_center_y = (value_top + value_bottom) / 2.0
            value_center_x = (value_left + value_right) / 2.0
            vertical_gap = max(0.0, value_top - label_bottom)
            if (
                value_center_y > label_center_y
                and label_left <= value_center_x <= label_right
                and min(value_right, label_right) > max(value_left, label_left)
                and vertical_gap
                <= _VOYAGE_VALUE_VERTICAL_GAP_HEIGHT_MULTIPLIER * max(label_height, value_height)
            ):
                candidates.append(index)
    except DirectPaddleSidecarError:
        return None
    return candidates[0] if len(candidates) == 1 else None


def _is_single_ascii_identifier_insertion(*, original: str, candidate: str) -> bool:
    """Return whether candidate is exactly one insertion into one identifier."""

    if (
        _BOUNDED_ASCII_FIELD_IDENTIFIER.fullmatch(original) is None
        or _BOUNDED_ASCII_FIELD_IDENTIFIER.fullmatch(candidate) is None
    ):
        return False
    original_folded = original.upper()
    candidate_folded = candidate.upper()
    if len(candidate_folded) != len(original_folded) + 1:
        return False
    return any(
        candidate_folded[:index] + candidate_folded[index + 1 :] == original_folded
        for index in range(len(candidate_folded))
    )


def _is_single_ascii_identifier_deletion(*, original: str, candidate: str) -> bool:
    """Return whether candidate removes only the leading identifier character."""

    if (
        _BOUNDED_ASCII_FIELD_IDENTIFIER.fullmatch(original) is None
        or _BOUNDED_ASCII_FIELD_IDENTIFIER.fullmatch(candidate) is None
    ):
        return False
    original_folded = original.upper()
    candidate_folded = candidate.upper()
    return (
        len(original_folded) == len(candidate_folded) + 1
        and original_folded[1:] == candidate_folded
    )


def _has_voyage_retry_confidence_gain(*, original: float, candidate: float) -> bool:
    """Compare the fixed gain without rejecting its float representation boundary."""

    difference = candidate - original
    return difference > _VOYAGE_RETRY_MIN_CONFIDENCE_GAIN or math.isclose(
        difference,
        _VOYAGE_RETRY_MIN_CONFIDENCE_GAIN,
        rel_tol=0.0,
        abs_tol=_VOYAGE_RETRY_CONFIDENCE_ABS_TOLERANCE,
    )


def _voyage_retry_consensus_improves_line(
    *, original: _Line, candidates: tuple[_Line, _Line]
) -> bool:
    """Accept only one bounded edit on which both fixed padded crops agree."""

    first, second = candidates
    if not all(
        math.isfinite(value) and 0.0 <= value <= 1.0
        for value in (original.confidence, first.confidence, second.confidence)
    ):
        return False
    if first.text != second.text:
        return False
    candidate = first.text
    prefix_insertion = (
        _SOURCE_ATTESTED_BL_TRUNCATED_VOYAGE.fullmatch(original.text.upper()) is not None
        and _SOURCE_ATTESTED_BL_VOYAGE.fullmatch(candidate.upper()) is not None
        and len(candidate) == len(original.text) + 1
        and candidate[1:].casefold() == original.text.casefold()
        and _is_single_ascii_identifier_insertion(original=original.text, candidate=candidate)
    )
    confident_deletion = (
        _is_single_ascii_identifier_deletion(original=original.text, candidate=candidate)
        and _SOURCE_ATTESTED_BL_VOYAGE.fullmatch(candidate.upper()) is not None
        and all(
            _has_voyage_retry_confidence_gain(
                original=original.confidence,
                candidate=item.confidence,
            )
            for item in candidates
        )
    )
    return prefix_insertion or confident_deletion


def _retry_bill_of_lading_voyage_line(
    *, image: Any, lines: Sequence[_Line], engine: _NativeEngine
) -> list[_Line]:
    """Apply at most one optional padded retry and otherwise preserve OCR."""

    original = list(lines)
    target_index = _unique_bill_of_lading_voyage_value_line(original)
    if target_index is None:
        return original
    try:
        candidates = _recognize_padded_voyage_consensus(
            image=image,
            line=original[target_index],
            engine=engine,
        )
    except DirectPaddleSidecarError:
        return original
    if candidates is None or not _voyage_retry_consensus_improves_line(
        original=original[target_index], candidates=candidates
    ):
        return original
    repaired = list(original)
    repaired[target_index] = candidates[0]
    return repaired


def _same_semantic_row(first: _Line, second: _Line) -> bool:
    """Use the fixed harmonic row band for a conservative same-row check."""

    _first_left, first_top, _first_right, first_bottom = _line_bounds(first)
    _second_left, second_top, _second_right, second_bottom = _line_bounds(second)
    first_height = first_bottom - first_top
    second_height = second_bottom - second_top
    denominator = first_height + second_height
    if denominator <= 0 or not math.isfinite(denominator):
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
    harmonic_height = 2.0 * first_height * second_height / denominator
    return (
        abs(((first_top + first_bottom) / 2.0) - ((second_top + second_bottom) / 2.0))
        <= 0.6 * harmonic_height
    )


def _label_is_vertically_near_value(*, label: _Line, value: _Line) -> bool:
    """Accept a value below its label only within the fixed four-height bound."""

    _label_left, label_top, _label_right, label_bottom = _line_bounds(label)
    _value_left, value_top, _value_right, value_bottom = _line_bounds(value)
    label_height = label_bottom - label_top
    value_height = value_bottom - value_top
    if (value_top + value_bottom) / 2.0 <= (label_top + label_bottom) / 2.0:
        return False
    vertical_gap = max(0.0, value_top - label_bottom)
    return vertical_gap <= _SEMANTIC_LABEL_VERTICAL_GAP_HEIGHT_MULTIPLIER * max(
        label_height, value_height
    )


def _value_x_corresponds_to_label(*, value: _Line, label: _Line, other_label: _Line) -> bool:
    """Require overlap and an unambiguous nearest label on the x axis."""

    value_left, _value_top, value_right, _value_bottom = _line_bounds(value)
    label_left, _label_top, label_right, _label_bottom = _line_bounds(label)
    other_left, _other_top, other_right, _other_bottom = _line_bounds(other_label)
    value_center = (value_left + value_right) / 2.0
    label_center = (label_left + label_right) / 2.0
    other_center = (other_left + other_right) / 2.0
    return (
        min(value_right, label_right) > max(value_left, label_left)
        and label_left <= value_center <= label_right
        and abs(value_center - label_center) < abs(value_center - other_center)
    )


def _lines_overlap_x(first: _Line, second: _Line) -> bool:
    first_left, _first_top, first_right, _first_bottom = _line_bounds(first)
    second_left, _second_top, second_right, _second_bottom = _line_bounds(second)
    return min(first_right, second_right) > max(first_left, second_left)


def _semantic_label_words(line: _Line) -> set[str]:
    return {word.upper() for word in _SEMANTIC_LABEL_WORD.findall(line.text)}


def _unique_split_weight_quartet(lines: Sequence[_Line]) -> tuple[int, int] | None:
    """Find one split NET/GROSS label plus two separate value-line quartet."""

    net_labels: list[tuple[int, _Line]] = []
    gross_labels: list[tuple[int, _Line]] = []
    for index, line in enumerate(lines):
        words = _semantic_label_words(line)
        if "NET" in words and "GROSS" not in words:
            net_labels.append((index, line))
        elif "GROSS" in words and "NET" not in words:
            gross_labels.append((index, line))

    amount_pairs: list[tuple[int, int, _Line, _Line]] = []
    for first_index in range(len(lines) - 1):
        second_index = first_index + 1
        first = _SINGLE_WEIGHT_AMOUNT.fullmatch(lines[first_index].text)
        second = _SINGLE_WEIGHT_AMOUNT.fullmatch(lines[second_index].text)
        if (
            first is None
            or second is None
            or first.group("unit") != second.group("unit")
            or not _same_semantic_row(lines[first_index], lines[second_index])
        ):
            continue
        first_left, _first_top, first_right, _first_bottom = _line_bounds(lines[first_index])
        second_left, _second_top, second_right, _second_bottom = _line_bounds(lines[second_index])
        if (first_left + first_right) / 2.0 >= (second_left + second_right) / 2.0:
            continue
        amount_pairs.append((first_index, second_index, lines[first_index], lines[second_index]))

    quartets: set[tuple[int, int, int, int]] = set()
    for net_index, net_label in net_labels:
        for gross_index, gross_label in gross_labels:
            net_left, _net_top, net_right, _net_bottom = _line_bounds(net_label)
            gross_left, _gross_top, gross_right, _gross_bottom = _line_bounds(gross_label)
            if (
                not _same_semantic_row(net_label, gross_label)
                or (net_left + net_right) / 2.0 >= (gross_left + gross_right) / 2.0
            ):
                continue
            for first_index, second_index, first_amount, second_amount in amount_pairs:
                if (
                    _label_is_vertically_near_value(label=net_label, value=first_amount)
                    and _label_is_vertically_near_value(label=gross_label, value=second_amount)
                    and _value_x_corresponds_to_label(
                        value=first_amount, label=net_label, other_label=gross_label
                    )
                    and _value_x_corresponds_to_label(
                        value=second_amount, label=gross_label, other_label=net_label
                    )
                ):
                    nearby_label_indexes = {
                        index
                        for index, line in enumerate(lines)
                        if {"NET", "GROSS"} & _semantic_label_words(line)
                        and (
                            (
                                _lines_overlap_x(line, first_amount)
                                and _label_is_vertically_near_value(label=line, value=first_amount)
                            )
                            or (
                                _lines_overlap_x(line, second_amount)
                                and _label_is_vertically_near_value(label=line, value=second_amount)
                            )
                        )
                    }
                    if nearby_label_indexes == {net_index, gross_index}:
                        quartets.add((net_index, gross_index, first_index, second_index))
    if len(quartets) != 1:
        return None
    _net_index, _gross_index, first_index, second_index = next(iter(quartets))
    return first_index, second_index


def _has_unambiguous_nearby_net_gross_labels(*, candidate: _Line, lines: Sequence[_Line]) -> bool:
    """Require one unambiguous combined or split geometry-bound label pair."""

    candidate_left, candidate_top, candidate_right, candidate_bottom = _line_bounds(candidate)
    candidate_height = candidate_bottom - candidate_top
    combined_labels = 0
    net_only_labels = 0
    gross_only_labels = 0
    for line in lines:
        if line is candidate:
            continue
        left, top, right, bottom = _line_bounds(line)
        # NET/GROSS labels for one summary value line sit above, below, or on
        # the same horizontal band.  Requiring x overlap and a four-line-
        # height vertical bound avoids borrowing a distant page label or a
        # table heading from another section.
        if right <= candidate_left or left >= candidate_right:
            continue
        vertical_gap = max(0.0, candidate_top - bottom, top - candidate_bottom)
        if vertical_gap > _SEMANTIC_LABEL_VERTICAL_GAP_HEIGHT_MULTIPLIER * max(
            candidate_height, bottom - top
        ):
            continue
        words = _semantic_label_words(line)
        if {"NET", "GROSS"} <= words:
            combined_labels += 1
        elif "NET" in words:
            net_only_labels += 1
        elif "GROSS" in words:
            gross_only_labels += 1
    return (combined_labels == 1 and net_only_labels == 0 and gross_only_labels == 0) or (
        combined_labels == 0 and net_only_labels == 1 and gross_only_labels == 1
    )


def _semantic_repair_supported_document_lines(lines: Sequence[_Line]) -> list[_Line]:
    """Apply two conservative, source-independent punctuation repairs.

    A repair is considered only after an exact supported-document heading is
    recognized.  Standalone numeric form ordinals are the only safe missing
    left-parenthesis shape: a list/item line such as ``1) WIDGET`` is left
    untouched.  The slash repair is narrower still: it applies only to an
    exact Packing List page.  It preserves the existing one-line amount path
    and also accepts exactly one geometry-bound quartet of split NET/GROSS
    labels plus two consecutive, same-row, same-unit amount lines.  Duplicate
    or mixed evidence stays unmodified.  No field value or template answer is
    stored.
    """

    headings = {line.text for line in lines if line.text in _SUPPORTED_DOCUMENT_HEADINGS}
    # More than one exact supported title on one page is an ambiguous
    # classification.  It is safer to preserve every recognized character.
    if len(headings) != 1:
        return list(lines)

    repaired: list[_Line] = []
    packing_list = "PACKING LIST" in headings
    packing_amount_candidates: dict[int, re.Match[str]] = {}
    split_weight_pair: tuple[int, int] | None = None
    if packing_list:
        for index, line in enumerate(lines):
            amounts = _ADJACENT_WEIGHT_AMOUNTS.fullmatch(line.text)
            if amounts is not None and amounts.group("first_unit") == amounts.group("second_unit"):
                packing_amount_candidates[index] = amounts
        # Multiple matching value rows can be an item table rather than one
        # form summary.  The page must contain exactly one candidate even if
        # only one happens to sit near a label; otherwise none is modified.
        if len(packing_amount_candidates) != 1:
            packing_amount_candidates.clear()
        elif not _has_unambiguous_nearby_net_gross_labels(
            candidate=lines[next(iter(packing_amount_candidates))], lines=lines
        ):
            packing_amount_candidates.clear()
        split_weight_pair = _unique_split_weight_quartet(lines)
        # Two simultaneously valid repair shapes are ambiguous.  Preserve
        # both rather than guessing which occurrence owns the separator.
        if packing_amount_candidates and split_weight_pair is not None:
            packing_amount_candidates.clear()
            split_weight_pair = None

    for index, line in enumerate(lines):
        text = line.text
        if _FORM_ORDINAL_WITH_MISSING_LEFT_PARENTHESIS.fullmatch(text):
            text = f"({text}"
        elif index in packing_amount_candidates:
            amounts = packing_amount_candidates[index]
            text = amounts.group("first") + " / " + amounts.group("second")
        elif split_weight_pair is not None and index == split_weight_pair[0]:
            text = f"{text} /"
        repaired.append(
            line
            if text == line.text
            else _Line(text=text, confidence=line.confidence, polygon=line.polygon)
        )
    return repaired


def semantic_repair_v14_packing_carton_texts(texts: Sequence[str]) -> tuple[str, ...]:
    """Restore one closed Packing List carton-count separator.

    This pure delta consumes only the ordered text emitted after the complete
    V13 repair.  It deliberately has no access to polygons, confidence,
    source documents, Gold values or template data.  Exactly one occurrence
    of the exact supported heading must be ``PACKING LIST``, and exactly one
    complete line must be ``C/T:`` immediately followed by one to three ASCII
    digits.  Every ambiguous or non-closed shape is returned byte-for-byte.
    """

    original = tuple(texts)
    # Keep every semantic literal inside this function.  Its source digest
    # can therefore bind the complete derived-evidence transform without a
    # mutable lookup table or an independently hashed regex constant.
    headings = tuple(
        text
        for text in original
        if text in ("BILL OF LADING", "COMMERCIAL INVOICE", "PACKING LIST")
    )
    if headings != ("PACKING LIST",):
        return original
    candidates = tuple(
        (index, match)
        for index, text in enumerate(original)
        if (match := re.fullmatch(r"C/T:(?P<count>[0-9]{1,3})", text)) is not None
    )
    if len(candidates) != 1:
        return original
    index, match = candidates[0]
    repaired = list(original)
    repaired[index] = f"C/T: {match.group('count')}"
    return tuple(repaired)


def _semantic_repair_supported_document_lines_v2(lines: Sequence[_Line]) -> list[_Line]:
    """Apply V13 repairs, then V14's independent text-only delta."""

    v13_repaired = _semantic_repair_supported_document_lines(lines)
    repaired_texts = semantic_repair_v14_packing_carton_texts(
        tuple(line.text for line in v13_repaired)
    )
    return [
        line
        if text == line.text
        else _Line(text=text, confidence=line.confidence, polygon=line.polygon)
        for line, text in zip(v13_repaired, repaired_texts, strict=True)
    ]


def _predict_pages(engine: _NativeEngine, pages: Sequence[_Page]) -> list[dict[str, object]]:
    try:
        output: list[dict[str, object]] = []
        for page in pages:
            image = engine.np.frombuffer(page.pixels, dtype="uint8").reshape(
                (page.height, page.width, 3)
            )
            detector_pass_mode = getattr(engine.profile, "detector_pass_mode", "single")
            if detector_pass_mode == "guarded_grid2":
                boxes = _guarded_grid2_detected_boxes(image=image, engine=engine)
            elif detector_pass_mode == "grid2":
                boxes = _grid2_detected_boxes(image=image, engine=engine)
            elif detector_pass_mode == "strip2":
                boxes = _strip2_detected_boxes(image=image, engine=engine)
            elif detector_pass_mode == "single":
                detector_input, original_width, original_height = _detector_input(
                    image=image, engine=engine
                )
                probability = _probability_map(
                    output=_run_predictor(
                        predictor=engine.detector, batch=detector_input, np=engine.np
                    ),
                    np=engine.np,
                )
                boxes = _detected_boxes(
                    probability=probability,
                    original_width=original_width,
                    original_height=original_height,
                    engine=engine,
                )
            else:
                raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
            lines = _ordered_lines(
                _recognize_crops(image=image[:, :, ::-1], boxes=boxes, engine=engine),
                row_band_mode=engine.profile.row_band_mode,
            )
            voyage_retry_mode = getattr(engine.profile, "voyage_retry_mode", "none")
            if voyage_retry_mode == "bill-of-lading-padded-v1":
                lines = _retry_bill_of_lading_voyage_line(
                    image=image[:, :, ::-1],
                    lines=lines,
                    engine=engine,
                )
            elif voyage_retry_mode != "none":
                raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
            semantic_repair_mode = getattr(engine.profile, "semantic_repair_mode", "none")
            if semantic_repair_mode == "supported_docs_v1":
                lines = _semantic_repair_supported_document_lines(lines)
            elif semantic_repair_mode == "supported_docs_v2":
                lines = _semantic_repair_supported_document_lines_v2(lines)
            elif semantic_repair_mode != "none":
                raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
            if lines:
                output.append(
                    {
                        "page_number": page.page_number,
                        "text": "\n".join(line.text for line in lines),
                        "confidence": min(line.confidence for line in lines),
                    }
                )
        return output
    except DirectPaddleSidecarError:
        raise
    except Exception:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_execution_failed") from None


def _response_bytes(
    *,
    profile: DirectPaddleProfile,
    source_sha256: str,
    revision: str,
    text_pages: list[dict[str, object]],
) -> bytes:
    encoded = json.dumps(
        {
            "protocol": _PROTOCOL,
            "source_sha256": source_sha256,
            "raster_text_profile_revision": revision,
            "provider_revision": profile.extractor_revision,
            "pages": text_pages,
            "warnings": [],
            "needs_review_reasons": [],
        },
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if not encoded or len(encoded) > _MAX_RESPONSE_BYTES:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
    return encoded


def _ready_bytes(*, profile: DirectPaddleProfile, protocol: str) -> bytes:
    if protocol not in {_SESSION_PROTOCOL, _SERVICE_PROTOCOL}:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
    encoded = json.dumps(
        {
            "protocol": protocol,
            "status": "ready",
            "raster_text_profile_revision": profile.raster_text_profile_revision,
            "provider_revision": profile.extractor_revision,
        },
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    if not encoded or len(encoded) > _MAX_RESPONSE_BYTES:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
    return encoded


def _read_framed_request(stream: BinaryIO) -> bytes | None:
    header = stream.read(8)
    if not header:
        return None
    if len(header) != 8:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_input_invalid")
    length = int.from_bytes(header, byteorder="big", signed=False)
    if not 1 <= length <= _MAX_INPUT_BYTES:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_input_invalid")
    payload = stream.read(length)
    if len(payload) != length:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_input_invalid")
    return payload


def _write_frame(stream: BinaryIO, payload: bytes) -> None:
    if not payload or len(payload) > _MAX_RESPONSE_BYTES:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_output_invalid")
    stream.write(len(payload).to_bytes(8, byteorder="big", signed=False))
    stream.write(payload)
    stream.flush()


def _verify_live_materials(*, profile: DirectPaddleProfile, model_root: Path) -> None:
    """Rehash the direct local input binding around every warm prediction."""

    current_profile, current_root = configuration_from_environment()
    try:
        if current_profile != profile or current_root.resolve(strict=True) != model_root.resolve(
            strict=True
        ):
            raise DirectPaddleSidecarError("direct_paddle_sidecar_binding_invalid")
    except DirectPaddleSidecarError:
        raise
    except OSError:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_model_invalid") from None


def _run_framed_from_stdio(*, stream: BinaryIO, output: BinaryIO, ready_protocol: str) -> None:
    """Serve compatible evaluation/service frames without changing selection wiring."""

    profile, model_root = configuration_from_environment()
    with deny_network_connections():
        engine = _engine(profile, model_root)
        _verify_live_materials(profile=profile, model_root=model_root)
        if profile.pre_ready_warmup:
            _warm_engine(engine=engine)
            # A modification while native allocations were created must not
            # become a ready service merely because the initial rehash passed.
            _verify_live_materials(profile=profile, model_root=model_root)
        _write_frame(output, _ready_bytes(profile=profile, protocol=ready_protocol))
        while True:
            payload = _read_framed_request(stream)
            if payload is None:
                return
            _verify_live_materials(profile=profile, model_root=model_root)
            source_sha256, revision, pages = _read_request(io.BytesIO(payload))
            if revision != profile.raster_text_profile_revision:
                raise DirectPaddleSidecarError("direct_paddle_sidecar_binding_invalid")
            text_pages = _predict_pages(engine, pages)
            _verify_live_materials(profile=profile, model_root=model_root)
            _write_frame(
                output,
                _response_bytes(
                    profile=profile,
                    source_sha256=source_sha256,
                    revision=revision,
                    text_pages=text_pages,
                ),
            )


def run_once_from_stdio(stream: BinaryIO, output: BinaryIO) -> None:
    profile, model_root = configuration_from_environment()
    source_sha256, revision, pages = _read_request(stream)
    if revision != profile.raster_text_profile_revision:
        raise DirectPaddleSidecarError("direct_paddle_sidecar_binding_invalid")
    with deny_network_connections():
        engine = _engine(profile, model_root)
        text_pages = _predict_pages(engine, pages)
    output.write(
        _response_bytes(
            profile=profile,
            source_sha256=source_sha256,
            revision=revision,
            text_pages=text_pages,
        )
    )


def run_session_from_stdio(stream: BinaryIO, output: BinaryIO) -> None:
    _run_framed_from_stdio(stream=stream, output=output, ready_protocol=_SESSION_PROTOCOL)


def run_service_from_stdio(stream: BinaryIO, output: BinaryIO) -> None:
    _run_framed_from_stdio(stream=stream, output=output, ready_protocol=_SERVICE_PROTOCOL)


def main(argv: Sequence[str] | None = None) -> int:
    arguments = list(sys.argv[1:] if argv is None else argv)
    try:
        if arguments == [f"--{_PROTOCOL}"]:
            run_once_from_stdio(sys.stdin.buffer, sys.stdout.buffer)
        elif arguments == [f"--{_SESSION_PROTOCOL}"]:
            run_session_from_stdio(sys.stdin.buffer, sys.stdout.buffer)
        elif arguments == [f"--{_SERVICE_PROTOCOL}"]:
            run_service_from_stdio(sys.stdin.buffer, sys.stdout.buffer)
        else:
            return 2
    except DirectPaddleSidecarError as exc:
        # Content-free exit stages allow a local bundle smoke to distinguish
        # oneDNN configuration, predictor construction and later warmup
        # failures without ever serializing a native diagnostic or OCR text.
        return {
            "direct_paddle_sidecar_predictor_config_unavailable": 71,
            "direct_paddle_sidecar_onednn_config_unavailable": 72,
            "direct_paddle_sidecar_onednn_predictor_unavailable": 73,
            "direct_paddle_sidecar_opencv_threads_unavailable": 74,
            "direct_paddle_sidecar_onednn_detector_warmup_unavailable": 75,
            "direct_paddle_sidecar_onednn_recognizer_warmup_unavailable": 76,
        }.get(str(exc), 1)
    except Exception:
        return 1
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())


__all__ = [
    "DirectPaddleProfile",
    "DirectPaddleSidecarError",
    "configuration_from_environment",
    "deny_network_connections",
    "main",
    "run_once_from_stdio",
    "run_service_from_stdio",
    "run_session_from_stdio",
    "semantic_repair_v14_packing_carton_texts",
]
