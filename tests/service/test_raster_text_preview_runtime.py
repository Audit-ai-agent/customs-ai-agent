"""Focused lifecycle tests for the persistent direct-text service child."""

from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import subprocess
import struct
import threading
import time
from types import SimpleNamespace
from typing import Callable, cast

import pytest

import customs_ai_agent.service.raster_text_preview_runtime as raster_text_runtime
from customs_ai_agent.service.raster_document_extraction import RasterizedPageV2
from customs_ai_agent.service.raster_text_extraction import RasterTextPreviewUnavailable
from customs_ai_agent.service.raster_text_preview_runtime import (
    PaddleOcrTextCommandProvider,
    PaddleOcrTextEvaluationSession,
    PaddleOcrTextServiceSession,
    RasterTextActivationPointerV1,
    RasterTextActivationPointerV2,
    RasterTextActivationPointerV3,
    RasterTextActivationPointerV4,
    RasterTextCalibrationReceiptV1,
    RasterTextCalibrationReceiptV2,
    RasterTextCalibrationReceiptV3,
    RasterTextCalibrationReceiptV4,
    RasterTextOfflineProfileV1,
    RasterTextOfflineProfileV2,
    RasterTextOfflineProfileV3,
    RasterTextOfflineProfileV4,
    RasterTextOfflineProfileV5,
    RasterTextOfflineProfileV6,
    RasterTextOfflineProfileV7,
    RasterTextOfflineProfileV8,
    RasterTextOfflineProfileV9,
    RasterTextOfflineProfileV10,
    RasterTextOfflineProfileV11,
    RasterTextOfflineProfileV12,
    RasterTextOfflineProfileV13,
    RasterTextOfflineProfileV14,
    RasterTextRuntimeError,
    RasterTextSealedEvaluationV1,
    RasterTextSealedEvaluationV2,
    RasterTextSealedEvaluationV3,
    RasterTextSealedEvaluationV4,
    RasterTextSealedProfileScoreV1,
    RasterTextSealedProfileScoreV2,
    RasterTextSealedProfileScoreV3,
    RasterTextSealedProfileScoreV4,
    RasterTextSidecarResultV1,
    RasterTextWarmServiceSmokeReceiptV1,
)
from customs_ai_agent.schemas.raster_text_extraction import RasterTextPageV1


_REVISION = "synthetic-raster-text-v1:" + ("a" * 24)
_SOURCE = "b" * 64


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _canonical_pointer(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _passing_score_payloads() -> list[dict[str, object]]:
    return [
        {
            "document_type": document_type,
            "profile_id": profile_id,
            "character_error_rate": 0.01,
            "critical_value_exact_match": 1.0,
        }
        for document_type in ("bill_of_lading", "commercial_invoice", "packing_list")
        for profile_id in ("png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75")
    ]


def _semantic_evidence_models() -> tuple[
    RasterTextActivationPointerV2,
    RasterTextCalibrationReceiptV2,
    RasterTextSealedEvaluationV2,
]:
    pointer = RasterTextActivationPointerV2(
        metric_revision="source-attested-semantic-text-v2",
        raster_text_profile_revision=_REVISION,
        runtime_configuration_path="profiles/final.json",
        runtime_configuration_sha256="c" * 64,
        sealed_evaluation_path="semantic-sealed-evaluations-v2/final.json",
        sealed_evaluation_sha256="d" * 64,
    )
    calibration = RasterTextCalibrationReceiptV2(
        metric_revision="source-attested-semantic-text-v2",
        raster_text_profile_revision=_REVISION,
        runtime_configuration_sha256="c" * 64,
        calibration_raster_text_profile_revision=_REVISION,
        calibration_runtime_configuration_sha256="c" * 64,
        extractor_revision="direct-text-test-v2",
        source_gold_receipt_path="evidence/calibration-source.json",
        source_gold_receipt_sha256="e" * 64,
        candidate_execution_path="evidence/calibration-candidate.json",
        candidate_execution_sha256="f" * 64,
        minimum_page_confidence=0.5,
        latency_p95_seconds=1.0,
    )
    evaluation = RasterTextSealedEvaluationV2(
        metric_revision="source-attested-semantic-text-v2",
        raster_text_profile_revision=_REVISION,
        runtime_configuration_sha256="c" * 64,
        extractor_revision="direct-text-test-v2",
        source_gold_receipt_path="evidence/sealed-source.json",
        source_gold_receipt_sha256="1" * 64,
        calibration_receipt_path="evidence/calibration.json",
        calibration_receipt_sha256="2" * 64,
        candidate_execution_path="evidence/sealed-candidate.json",
        candidate_execution_sha256="3" * 64,
        warm_service_smoke_receipt_path="evidence/warm.json",
        warm_service_smoke_receipt_sha256="4" * 64,
        latency_p95_seconds=1.0,
        service_warm_latency_p95_seconds=1.0,
        minimum_page_confidence=0.5,
        scores=[
            RasterTextSealedProfileScoreV2.model_validate(value)
            for value in _passing_score_payloads()
        ],
    )
    return pointer, calibration, evaluation


def _semantic_v3_evidence_models() -> tuple[
    RasterTextActivationPointerV3,
    RasterTextCalibrationReceiptV3,
    RasterTextSealedEvaluationV3,
]:
    pointer_v2, calibration_v2, evaluation_v2 = _semantic_evidence_models()
    pointer_values = pointer_v2.model_dump(mode="json", warnings=False)
    pointer_values.update(
        schema_version="3.0",
        metric_revision="source-attested-semantic-text-v3",
    )
    calibration_values = calibration_v2.model_dump(mode="json", warnings=False)
    calibration_values.update(
        schema_version="3.0",
        metric_revision="source-attested-semantic-text-v3",
    )
    evaluation_values = evaluation_v2.model_dump(mode="json", warnings=False)
    evaluation_values.update(
        schema_version="3.0",
        metric_revision="source-attested-semantic-text-v3",
        scores=[
            RasterTextSealedProfileScoreV3.model_validate(value).model_dump(
                mode="json", warnings=False
            )
            for value in _passing_score_payloads()
        ],
    )
    return (
        RasterTextActivationPointerV3.model_validate(pointer_values),
        RasterTextCalibrationReceiptV3.model_validate(calibration_values),
        RasterTextSealedEvaluationV3.model_validate(evaluation_values),
    )


def _fresh_v4_evidence_models() -> tuple[
    RasterTextActivationPointerV4,
    RasterTextCalibrationReceiptV4,
    RasterTextSealedEvaluationV4,
]:
    pointer = RasterTextActivationPointerV4(
        metric_revision="source-attested-semantic-text-v3",
        evidence_revision="source-attested-fresh-holdout-v1",
        raster_text_profile_revision=_REVISION,
        runtime_configuration_path="profiles/final-v14.json",
        runtime_configuration_sha256="c" * 64,
        sealed_evaluation_path="fresh-sealed-evaluations-v4/final.json",
        sealed_evaluation_sha256="d" * 64,
    )
    calibration = RasterTextCalibrationReceiptV4(
        metric_revision="source-attested-semantic-text-v3",
        evidence_revision="source-attested-fresh-holdout-v1",
        raster_text_profile_revision=_REVISION,
        runtime_configuration_sha256="c" * 64,
        calibration_raster_text_profile_revision=_REVISION,
        calibration_runtime_configuration_path="profiles/calibration-v14.json",
        calibration_runtime_configuration_sha256="5" * 64,
        extractor_revision="direct-text-test-v14",
        selection_receipt_path="fresh-scope-selections-v1/selection.json",
        selection_receipt_sha256="6" * 64,
        source_gold_receipt_path="fresh-source-gold-receipts/fresh-calibration/source.json",
        source_gold_receipt_sha256="7" * 64,
        candidate_execution_path="fresh-candidate-executions/fresh-calibration/execution.json",
        candidate_execution_sha256="8" * 64,
        minimum_page_confidence=0.5,
        latency_p95_seconds=1.0,
    )
    evaluation = RasterTextSealedEvaluationV4(
        metric_revision="source-attested-semantic-text-v3",
        evidence_revision="source-attested-fresh-holdout-v1",
        raster_text_profile_revision=_REVISION,
        runtime_configuration_sha256="c" * 64,
        extractor_revision="direct-text-test-v14",
        selection_receipt_path=calibration.selection_receipt_path,
        selection_receipt_sha256=calibration.selection_receipt_sha256,
        source_gold_receipt_path="fresh-source-gold-receipts/fresh-sealed/source.json",
        source_gold_receipt_sha256="9" * 64,
        calibration_receipt_path="fresh-calibration-receipts-v4/calibration.json",
        calibration_receipt_sha256="a" * 64,
        candidate_execution_path="fresh-candidate-executions/fresh-sealed/execution.json",
        candidate_execution_sha256="b" * 64,
        warm_service_smoke_receipt_path="evidence/warm.json",
        warm_service_smoke_receipt_sha256="e" * 64,
        latency_p95_seconds=1.0,
        service_warm_latency_p95_seconds=1.0,
        minimum_page_confidence=0.5,
        scores=[
            RasterTextSealedProfileScoreV4.model_validate(value)
            for value in _passing_score_payloads()
        ],
    )
    return pointer, calibration, evaluation


def _frame(payload: bytes) -> bytes:
    return struct.pack(">Q", len(payload)) + payload


def _profile() -> RasterTextOfflineProfileV1:
    # The service session only reads these already profile-bound values.  A
    # simple typed test double avoids creating an on-disk model bundle.
    return cast(
        RasterTextOfflineProfileV1,
        SimpleNamespace(
            raster_text_profile_revision=_REVISION,
            extractor_revision="direct-text-test-v1",
            minimum_page_confidence=0.5,
            sidecar_timeout_seconds=30,
        ),
    )


def _v2_profile() -> RasterTextOfflineProfileV2:
    """Small typed V2 double for lifecycle-only startup-bound tests."""

    return RasterTextOfflineProfileV2.model_construct(
        schema_version="2.0",
        engine_kind="direct_paddle_text",
        runtime_api="paddle-inference-static-v1",
        raster_text_profile_revision=_REVISION,
        extractor_revision="direct-text-test-v2",
        minimum_page_confidence=0.5,
        sidecar_timeout_seconds=30,
    )


def _v3_profile() -> RasterTextOfflineProfileV3:
    """Small typed V3 double for launch-boundary tests only."""

    return RasterTextOfflineProfileV3.model_construct(
        schema_version="3.0",
        engine_kind="direct_paddle_mobile_text",
        runtime_api="paddle-inference-static-v1",
        raster_text_profile_revision=_REVISION,
        extractor_revision="direct-text-test-v3",
        minimum_page_confidence=0.5,
        sidecar_timeout_seconds=30,
        bundle_root_relative_path=("raster-text-direct-bundles-v1/releases/" + "d" * 64),
        mobile_model_bundle_root_relative_path=(
            "raster-text-direct-mobile-bundles-v1/releases/" + "e" * 64
        ),
    )


def _v4_profile() -> RasterTextOfflineProfileV4:
    """Small typed V4 double for the bounded harmonic launch path."""

    return RasterTextOfflineProfileV4.model_construct(
        schema_version="4.0",
        engine_kind="direct_paddle_mobile_text_harmonic",
        runtime_api="paddle-inference-static-v1",
        raster_text_profile_revision=_REVISION,
        extractor_revision="direct-text-test-v4-harmonic",
        minimum_page_confidence=0.5,
        sidecar_timeout_seconds=30,
        cpu_threads=6,
        text_detection_limit_side_len=960,
        bundle_root_relative_path=("raster-text-direct-bundles-v1/releases/" + "d" * 64),
        mobile_model_bundle_root_relative_path=(
            "raster-text-direct-mobile-bundles-v1/releases/" + "e" * 64
        ),
    )


def _v5_profile() -> RasterTextOfflineProfileV5:
    """Small typed V5 double for the bounded harmonic-recall launch path."""

    return RasterTextOfflineProfileV5.model_construct(
        schema_version="5.0",
        engine_kind="direct_paddle_mobile_text_harmonic_recall",
        runtime_api="paddle-inference-static-v1",
        raster_text_profile_revision=_REVISION,
        extractor_revision="direct-text-test-v5-harmonic-recall",
        minimum_page_confidence=0.5,
        sidecar_timeout_seconds=30,
        cpu_threads=6,
        text_detection_limit_side_len=960,
        text_detection_box_threshold=0.4,
        bundle_root_relative_path=("raster-text-direct-bundles-v1/releases/" + "d" * 64),
        mobile_model_bundle_root_relative_path=(
            "raster-text-direct-mobile-bundles-v1/releases/" + "e" * 64
        ),
    )


def _v6_profile() -> RasterTextOfflineProfileV6:
    """Small typed V6 double for the bounded 1024-side launch path."""

    return RasterTextOfflineProfileV6.model_construct(
        schema_version="6.0",
        engine_kind="direct_paddle_mobile_text_1024",
        runtime_api="paddle-inference-static-v1",
        raster_text_profile_revision=_REVISION,
        extractor_revision="direct-text-test-v6-1024",
        minimum_page_confidence=0.5,
        sidecar_timeout_seconds=30,
        cpu_threads=6,
        text_detection_limit_side_len=1024,
        text_detection_box_threshold=0.4,
        bundle_root_relative_path=("raster-text-direct-bundles-v1/releases/" + "d" * 64),
        mobile_model_bundle_root_relative_path=(
            "raster-text-direct-mobile-bundles-v1/releases/" + "e" * 64
        ),
    )


def _v7_profile() -> RasterTextOfflineProfileV7:
    """Small typed V7 double for the bounded 1024-side low-box launch path."""

    return RasterTextOfflineProfileV7.model_construct(
        schema_version="7.0",
        engine_kind="direct_paddle_mobile_text_harmonic_recall_1024_lowbox",
        runtime_api="paddle-inference-static-v1",
        raster_text_profile_revision=_REVISION,
        extractor_revision="direct-text-test-v7-1024-lowbox",
        minimum_page_confidence=0.5,
        sidecar_timeout_seconds=30,
        cpu_threads=6,
        text_detection_limit_side_len=1024,
        text_detection_box_threshold=0.3,
        bundle_root_relative_path=("raster-text-direct-bundles-v1/releases/" + "d" * 64),
        mobile_model_bundle_root_relative_path=(
            "raster-text-direct-mobile-bundles-v1/releases/" + "e" * 64
        ),
    )


def _v8_profile() -> RasterTextOfflineProfileV8:
    """Small typed V8 double for the bounded native-960 low-box path."""

    return RasterTextOfflineProfileV8.model_construct(
        schema_version="8.0",
        engine_kind="direct_paddle_mobile_text_harmonic_recall_native960_lowbox",
        runtime_api="paddle-inference-static-v1",
        raster_text_profile_revision=_REVISION,
        extractor_revision="direct-text-test-v8-native960-lowbox",
        minimum_page_confidence=0.5,
        sidecar_timeout_seconds=30,
        cpu_threads=6,
        text_detection_limit_side_len=960,
        text_detection_box_threshold=0.3,
        bundle_root_relative_path=("raster-text-direct-bundles-v1/releases/" + "d" * 64),
        mobile_model_bundle_root_relative_path=(
            "raster-text-direct-mobile-bundles-v1/releases/" + "e" * 64
        ),
    )


def _v9_profile() -> RasterTextOfflineProfileV9:
    """Small typed V9 double for the bounded native-960 strip2 path."""

    return RasterTextOfflineProfileV9.model_construct(
        schema_version="9.0",
        engine_kind="direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2",
        runtime_api="paddle-inference-static-v1",
        raster_text_profile_revision=_REVISION,
        extractor_revision="direct-text-test-v9-native960-lowbox-strip2",
        minimum_page_confidence=0.5,
        sidecar_timeout_seconds=30,
        cpu_threads=6,
        text_detection_limit_side_len=960,
        text_detection_box_threshold=0.3,
        bundle_root_relative_path=("raster-text-direct-bundles-v1/releases/" + "d" * 64),
        mobile_model_bundle_root_relative_path=(
            "raster-text-direct-mobile-bundles-v1/releases/" + "e" * 64
        ),
    )


def _v10_profile() -> RasterTextOfflineProfileV10:
    """Small typed V10 double for the bounded native-960 grid2 path."""

    return RasterTextOfflineProfileV10.model_construct(
        schema_version="10.0",
        engine_kind="direct_paddle_mobile_text_harmonic_recall_native960_lowbox_grid2",
        runtime_api="paddle-inference-static-v1",
        raster_text_profile_revision=_REVISION,
        extractor_revision="direct-text-test-v10-native960-lowbox-grid2",
        minimum_page_confidence=0.5,
        sidecar_timeout_seconds=30,
        cpu_threads=6,
        text_detection_limit_side_len=960,
        text_detection_box_threshold=0.3,
        bundle_root_relative_path=("raster-text-direct-bundles-v1/releases/" + "d" * 64),
        mobile_model_bundle_root_relative_path=(
            "raster-text-direct-mobile-bundles-v1/releases/" + "e" * 64
        ),
    )


def _v11_profile() -> RasterTextOfflineProfileV11:
    """Small typed V11 double for the bounded guarded-grid launch path."""

    return RasterTextOfflineProfileV11.model_construct(
        schema_version="11.0",
        engine_kind="direct_paddle_mobile_text_harmonic_recall_native960_lowbox_guarded_grid2",
        runtime_api="paddle-inference-static-v1",
        raster_text_profile_revision=_REVISION,
        extractor_revision="direct-text-test-v11-native960-lowbox-guarded-grid2",
        minimum_page_confidence=0.5,
        sidecar_timeout_seconds=30,
        cpu_threads=6,
        text_detection_limit_side_len=960,
        text_detection_box_threshold=0.3,
        bundle_root_relative_path=("raster-text-direct-bundles-v1/releases/" + "d" * 64),
        mobile_model_bundle_root_relative_path=(
            "raster-text-direct-mobile-bundles-v1/releases/" + "e" * 64
        ),
    )


def _v12_profile() -> RasterTextOfflineProfileV12:
    sidecar_sha = "d" * 64
    combined_sha = "e" * 64
    sidecar_root = f"raster-text-direct-bundles-v1/releases/{sidecar_sha}"
    combined_root = f"raster-text-trained-paddle-bundles-v1/releases/{combined_sha}"
    values: dict[str, object] = {
        "schema_version": "12.0",
        "engine_kind": (
            "direct_paddle_source_attested_text_harmonic_recall_native960_lowbox_strip2"
        ),
        "runtime_api": "paddle-inference-static-v1",
        "extractor_revision": "direct-text-test-v12-source-attested-strip2",
        "bundle_root_relative_path": sidecar_root,
        "bundle_manifest_sha256": sidecar_sha,
        "combined_model_bundle_root_relative_path": combined_root,
        "combined_model_bundle_manifest_sha256": combined_sha,
        "source_detector_bundle_manifest_sha256": "1" * 64,
        "source_detector_model_inventory_sha256": "2" * 64,
        "source_detector_model_snapshot_sha256": "3" * 64,
        "source_recognizer_release_sha256": "4" * 64,
        "source_recognizer_receipt_sha256": "4" * 64,
        "source_recognizer_training_intent_sha256": "5" * 64,
        "source_recognizer_character_dictionary_sha256": "6" * 64,
        "source_recognizer_package_inventory_sha256": "7" * 64,
        "pdfium_version": "4.30.0",
        "raster_dpi": 216,
        "sidecar_timeout_seconds": 30,
        "cpu_threads": 6,
        "text_detection_limit_side_len": 960,
        "text_detection_limit_type": "max",
        "text_detection_threshold": 0.3,
        "text_detection_box_threshold": 0.3,
        "text_detection_unclip_ratio": 1.5,
        "minimum_page_confidence": 0.5,
        "parent_renderer_package_inventory_path": (
            f"{sidecar_root}/inventories/parent-renderer.json"
        ),
        "parent_renderer_package_inventory_sha256": "8" * 64,
        "parent_renderer_required_packages": ["pypdfium2", "Pillow"],
        "sidecar_interpreter_path": f"{sidecar_root}/sidecar/interpreter/python.exe",
        "sidecar_interpreter_sha256": "9" * 64,
        "sidecar_interpreter_root": f"{sidecar_root}/sidecar/interpreter",
        "sidecar_interpreter_root_tree_sha256": "a" * 64,
        "sidecar_module_path": (
            f"{sidecar_root}/sidecar/module-root/customs_ai_agent/service/"
            "raster_text_direct_paddle_sidecar.py"
        ),
        "sidecar_module_sha256": "b" * 64,
        "sidecar_module_root": f"{sidecar_root}/sidecar/module-root",
        "sidecar_module_root_tree_sha256": "c" * 64,
        "sidecar_package_inventory_path": (
            f"{sidecar_root}/inventories/direct-paddle-packages.json"
        ),
        "sidecar_package_inventory_sha256": "f" * 64,
        "sidecar_required_packages": [
            "numpy",
            "opencv-contrib-python",
            "paddlepaddle",
            "pyclipper",
        ],
        "local_model_root": f"{combined_root}/models",
        "model_inventory_path": (f"{combined_root}/inventories/direct-paddle-trained-models.json"),
        "model_inventory_sha256": "0" * 64,
        "text_detection_model_name": "PP-OCRv5_mobile_det",
        "text_detection_model_directory": "text-det",
        "text_recognition_model_name": "source_attested_tiny_crnn_bigru_ctc_v1",
        "text_recognition_model_directory": "text-rec",
    }
    fingerprint = raster_text_runtime.raster_text_profile_fingerprint_sha256(values)
    return RasterTextOfflineProfileV12.model_validate(
        {
            **values,
            "profile_fingerprint_sha256": fingerprint,
            "raster_text_profile_revision": f"synthetic-raster-text-v1:{fingerprint[:24]}",
        }
    )


def _v13_profile() -> RasterTextOfflineProfileV13:
    """Full public-mobile V13 profile with V9's material and runtime bounds."""

    values = _v12_profile().model_dump(
        mode="json",
        warnings=False,
        exclude={"profile_fingerprint_sha256", "raster_text_profile_revision"},
    )
    for key in (
        "combined_model_bundle_root_relative_path",
        "combined_model_bundle_manifest_sha256",
        "source_detector_bundle_manifest_sha256",
        "source_detector_model_inventory_sha256",
        "source_detector_model_snapshot_sha256",
        "source_recognizer_release_sha256",
        "source_recognizer_receipt_sha256",
        "source_recognizer_training_intent_sha256",
        "source_recognizer_character_dictionary_sha256",
        "source_recognizer_package_inventory_sha256",
    ):
        values.pop(key)
    mobile_sha = "e" * 64
    source_sha = "1" * 64
    mobile_root = f"raster-text-direct-mobile-bundles-v1/releases/{mobile_sha}"
    values.update(
        {
            "schema_version": "13.0",
            "engine_kind": (
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_semantic_repair"
            ),
            "extractor_revision": "direct-text-test-v13-public-semantic-repair",
            "mobile_model_bundle_root_relative_path": mobile_root,
            "mobile_model_bundle_manifest_sha256": mobile_sha,
            "source_model_inventory_relative_path": f"model-inventories/{source_sha}.json",
            "source_model_inventory_sha256": source_sha,
            "local_model_root": f"{mobile_root}/models",
            "model_inventory_path": (f"{mobile_root}/inventories/direct-paddle-mobile-models.json"),
            "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
        }
    )
    fingerprint = raster_text_runtime.raster_text_profile_fingerprint_sha256(values)
    return RasterTextOfflineProfileV13.model_validate(
        {
            **values,
            "profile_fingerprint_sha256": fingerprint,
            "raster_text_profile_revision": f"synthetic-raster-text-v1:{fingerprint[:24]}",
        }
    )


def _v14_profile(
    *, minimum_page_confidence: float | None = None
) -> RasterTextOfflineProfileV14:
    """Full V14 profile with V13 OCR material and distinct repair identity."""

    values = _v13_profile().model_dump(
        mode="json",
        warnings=False,
        exclude={"profile_fingerprint_sha256", "raster_text_profile_revision"},
    )
    values.update(
        {
            "schema_version": "14.0",
            "engine_kind": (
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_"
                "strip2_semantic_repair_v2"
            ),
            "extractor_revision": "direct-text-test-v14-public-semantic-repair-v2",
        }
    )
    if minimum_page_confidence is not None:
        values["minimum_page_confidence"] = minimum_page_confidence
    fingerprint = raster_text_runtime.raster_text_profile_fingerprint_sha256(values)
    return RasterTextOfflineProfileV14.model_validate(
        {
            **values,
            "profile_fingerprint_sha256": fingerprint,
            "raster_text_profile_revision": f"synthetic-raster-text-v1:{fingerprint[:24]}",
        }
    )


def _provider(
    profile: RasterTextOfflineProfileV1
    | RasterTextOfflineProfileV2
    | RasterTextOfflineProfileV3
    | RasterTextOfflineProfileV4
    | RasterTextOfflineProfileV5
    | RasterTextOfflineProfileV6
    | RasterTextOfflineProfileV7
    | RasterTextOfflineProfileV8
    | RasterTextOfflineProfileV9
    | RasterTextOfflineProfileV10
    | RasterTextOfflineProfileV11
    | RasterTextOfflineProfileV12
    | RasterTextOfflineProfileV13
    | RasterTextOfflineProfileV14
    | None = None,
) -> PaddleOcrTextCommandProvider:
    return PaddleOcrTextCommandProvider(
        profile=_profile() if profile is None else profile,
        interpreter=Path("sidecar-python"),
        interpreter_root=Path("sidecar-root"),
        module=Path("sidecar-module.py"),
        module_root=Path("module-root"),
        configuration_path=Path("profile.json"),
        configuration_sha256="c" * 64,
        sidecar_inventory=Path("packages.json"),
        model_root=Path("models"),
        model_inventory=Path("models.json"),
    )


def test_v2_bundle_leaf_uses_validated_short_root_without_generic_leaf_resolve(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A deep V2 sidecar module must not re-enter the generic Windows resolver."""

    manifest_sha256 = "d" * 64
    bundle_relative = f"raster-text-direct-bundles-v1/releases/{manifest_sha256}"
    module_relative = (
        f"{bundle_relative}/sidecar/module-root/customs_ai_agent/service/"
        "raster_text_direct_paddle_sidecar.py"
    )
    profile = cast(
        RasterTextOfflineProfileV2,
        SimpleNamespace(bundle_root_relative_path=bundle_relative),
    )
    # The canonical document-preview spelling models the actual Korean,
    # deeply nested workspace condition.  It is intentionally never opened:
    # the validated release-root mock below returns its usable short alias.
    canonical_root = Path("C:/") / ("한글-긴-작업경로-" * 24)
    assert len(str(canonical_root.joinpath(*module_relative.split("/")))) > 260
    short_bundle_root = tmp_path / "BUNDLE~1"
    short_module = short_bundle_root / "sidecar/module-root/customs_ai_agent/service"
    short_module.mkdir(parents=True)
    expected = short_module / "raster_text_direct_paddle_sidecar.py"
    expected.write_text("# sidecar\n", encoding="utf-8")
    bundle_calls: list[str] = []

    def validated_bundle_root(root: Path, value: str) -> Path:
        assert root == canonical_root
        bundle_calls.append(value)
        return short_bundle_root

    monkeypatch.setattr(raster_text_runtime, "_non_symlink_sidecar_path", validated_bundle_root)
    monkeypatch.setattr(
        raster_text_runtime,
        "_sidecar_path",
        lambda *_args: pytest.fail("a V2 deep leaf must not use the generic resolver"),
    )

    resolved = raster_text_runtime._v2_bundle_sidecar_path(  # noqa: SLF001 - path boundary
        root=canonical_root,
        profile=profile,
        value=module_relative,
    )

    assert resolved == expected
    assert resolved.is_file()
    assert bundle_calls == [bundle_relative]

    # A profile material outside the immutable release cannot obtain the
    # short alias, even if a caller supplies a syntactically safe path.
    with pytest.raises(RasterTextRuntimeError, match="runtime_material_invalid"):
        raster_text_runtime._v2_bundle_sidecar_path(  # noqa: SLF001 - path boundary
            root=canonical_root,
            profile=profile,
            value="unrelated/material.py",
        )

    original_is_symlink = Path.is_symlink

    def inserted_link(path: Path) -> bool:
        if path.name == "inserted-link":
            return True
        return original_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", inserted_link)
    with pytest.raises(RasterTextRuntimeError, match="runtime_material_invalid"):
        raster_text_runtime._v2_bundle_sidecar_path(  # noqa: SLF001 - path boundary
            root=canonical_root,
            profile=profile,
            value=f"{bundle_relative}/inserted-link/material.py",
        )


def test_profile_material_path_preserves_v1_generic_resolver(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The direct short-root workaround is not reachable from the V1 profile."""

    expected = tmp_path / "legacy-material"
    calls: list[tuple[Path, str]] = []

    def generic(root: Path, value: str) -> Path:
        calls.append((root, value))
        return expected

    monkeypatch.setattr(raster_text_runtime, "_sidecar_path", generic)
    monkeypatch.setattr(
        raster_text_runtime,
        "_v2_bundle_sidecar_path",
        lambda **_kwargs: pytest.fail("V1 must retain the generic resolver"),
    )
    root = tmp_path / "root"

    assert (
        raster_text_runtime._profile_sidecar_path(  # noqa: SLF001 - path boundary
            root=root,
            profile=_profile(),
            value="legacy/material.json",
        )
        == expected
    )
    assert calls == [(root, "legacy/material.json")]


def test_v3_uses_the_mobile_bundle_root_and_keeps_the_v2_request_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """V3 has a second root, while V2's 30-second request lane stays intact."""

    profile = _v3_profile()
    mobile_root = tmp_path / "MOBILE~1"
    expected = mobile_root / "models" / "text-det"
    expected.mkdir(parents=True)
    calls: list[str] = []

    def resolved_root(_root: Path, value: str) -> Path:
        calls.append(value)
        assert value == profile.mobile_model_bundle_root_relative_path
        return mobile_root

    monkeypatch.setattr(raster_text_runtime, "_non_symlink_sidecar_path", resolved_root)
    resolved = raster_text_runtime._v2_bundle_sidecar_path(  # noqa: SLF001 - boundary test
        root=tmp_path / "document-preview",
        profile=profile,
        value=f"{profile.mobile_model_bundle_root_relative_path}/models/text-det",
    )

    assert resolved == expected
    assert calls == [profile.mobile_model_bundle_root_relative_path]
    assert raster_text_runtime.raster_text_service_startup_timeout_seconds(_v2_profile()) == 60
    assert raster_text_runtime.raster_text_service_startup_timeout_seconds(profile) == 60


def test_v4_uses_the_same_two_root_boundary_and_keeps_the_30_60_bounds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile = _v4_profile()
    mobile_root = tmp_path / "MOBILE~1"
    expected = mobile_root / "models" / "text-rec"
    expected.mkdir(parents=True)
    calls: list[str] = []

    def resolved_root(_root: Path, value: str) -> Path:
        calls.append(value)
        assert value == profile.mobile_model_bundle_root_relative_path
        return mobile_root

    monkeypatch.setattr(raster_text_runtime, "_non_symlink_sidecar_path", resolved_root)
    resolved = raster_text_runtime._v2_bundle_sidecar_path(  # noqa: SLF001 - boundary test
        root=tmp_path / "document-preview",
        profile=profile,
        value=f"{profile.mobile_model_bundle_root_relative_path}/models/text-rec",
    )

    assert resolved == expected
    assert calls == [profile.mobile_model_bundle_root_relative_path]
    assert profile.sidecar_timeout_seconds == 30
    assert raster_text_runtime.raster_text_service_startup_timeout_seconds(profile) == 60
    assert (
        raster_text_runtime._sidecar_bootstrap_for_profile(profile)  # noqa: SLF001
        == raster_text_runtime._DIRECT_PADDLE_V2_ISOLATED_SIDECAR_BOOTSTRAP  # noqa: SLF001
    )


def test_v5_uses_the_same_two_root_boundary_and_keeps_the_30_60_bounds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile = _v5_profile()
    mobile_root = tmp_path / "MOBILE~1"
    expected = mobile_root / "models" / "text-det"
    expected.mkdir(parents=True)
    calls: list[str] = []

    def resolved_root(_root: Path, value: str) -> Path:
        calls.append(value)
        assert value == profile.mobile_model_bundle_root_relative_path
        return mobile_root

    monkeypatch.setattr(raster_text_runtime, "_non_symlink_sidecar_path", resolved_root)
    resolved = raster_text_runtime._v2_bundle_sidecar_path(  # noqa: SLF001 - boundary test
        root=tmp_path / "document-preview",
        profile=profile,
        value=f"{profile.mobile_model_bundle_root_relative_path}/models/text-det",
    )

    assert resolved == expected
    assert calls == [profile.mobile_model_bundle_root_relative_path]
    assert profile.sidecar_timeout_seconds == 30
    assert raster_text_runtime.raster_text_service_startup_timeout_seconds(profile) == 60
    assert (
        raster_text_runtime._sidecar_bootstrap_for_profile(profile)  # noqa: SLF001
        == raster_text_runtime._DIRECT_PADDLE_V2_ISOLATED_SIDECAR_BOOTSTRAP  # noqa: SLF001
    )


def test_v6_uses_the_same_two_root_boundary_and_keeps_the_30_60_bounds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile = _v6_profile()
    mobile_root = tmp_path / "MOBILE~1"
    expected = mobile_root / "models" / "text-det"
    expected.mkdir(parents=True)
    calls: list[str] = []

    def resolved_root(_root: Path, value: str) -> Path:
        calls.append(value)
        assert value == profile.mobile_model_bundle_root_relative_path
        return mobile_root

    monkeypatch.setattr(raster_text_runtime, "_non_symlink_sidecar_path", resolved_root)
    resolved = raster_text_runtime._v2_bundle_sidecar_path(  # noqa: SLF001 - boundary test
        root=tmp_path / "document-preview",
        profile=profile,
        value=f"{profile.mobile_model_bundle_root_relative_path}/models/text-det",
    )

    assert resolved == expected
    assert calls == [profile.mobile_model_bundle_root_relative_path]
    assert profile.sidecar_timeout_seconds == 30
    assert profile.text_detection_limit_side_len == 1024
    assert profile.text_detection_box_threshold == 0.4
    assert raster_text_runtime.raster_text_service_startup_timeout_seconds(profile) == 60
    assert (
        raster_text_runtime._sidecar_bootstrap_for_profile(profile)  # noqa: SLF001
        == raster_text_runtime._DIRECT_PADDLE_V2_ISOLATED_SIDECAR_BOOTSTRAP  # noqa: SLF001
    )


def test_v7_uses_the_same_two_root_boundary_and_keeps_the_30_60_bounds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile = _v7_profile()
    mobile_root = tmp_path / "MOBILE~1"
    expected = mobile_root / "models" / "text-det"
    expected.mkdir(parents=True)
    calls: list[str] = []

    def resolved_root(_root: Path, value: str) -> Path:
        calls.append(value)
        assert value == profile.mobile_model_bundle_root_relative_path
        return mobile_root

    monkeypatch.setattr(raster_text_runtime, "_non_symlink_sidecar_path", resolved_root)
    resolved = raster_text_runtime._v2_bundle_sidecar_path(  # noqa: SLF001 - boundary test
        root=tmp_path / "document-preview",
        profile=profile,
        value=f"{profile.mobile_model_bundle_root_relative_path}/models/text-det",
    )

    assert resolved == expected
    assert calls == [profile.mobile_model_bundle_root_relative_path]
    assert profile.sidecar_timeout_seconds == 30
    assert profile.text_detection_limit_side_len == 1024
    assert profile.text_detection_box_threshold == 0.3
    assert raster_text_runtime.raster_text_service_startup_timeout_seconds(profile) == 60
    assert (
        raster_text_runtime._sidecar_bootstrap_for_profile(profile)  # noqa: SLF001
        == raster_text_runtime._DIRECT_PADDLE_V2_ISOLATED_SIDECAR_BOOTSTRAP  # noqa: SLF001
    )


def test_v8_uses_the_same_two_root_boundary_and_keeps_the_30_60_bounds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile = _v8_profile()
    mobile_root = tmp_path / "MOBILE~1"
    expected = mobile_root / "models" / "text-rec"
    expected.mkdir(parents=True)
    calls: list[str] = []

    def resolved_root(_root: Path, value: str) -> Path:
        calls.append(value)
        assert value == profile.mobile_model_bundle_root_relative_path
        return mobile_root

    monkeypatch.setattr(raster_text_runtime, "_non_symlink_sidecar_path", resolved_root)
    resolved = raster_text_runtime._v2_bundle_sidecar_path(  # noqa: SLF001 - boundary test
        root=tmp_path / "document-preview",
        profile=profile,
        value=f"{profile.mobile_model_bundle_root_relative_path}/models/text-rec",
    )

    assert resolved == expected
    assert calls == [profile.mobile_model_bundle_root_relative_path]
    assert profile.sidecar_timeout_seconds == 30
    assert profile.text_detection_limit_side_len == 960
    assert profile.text_detection_box_threshold == 0.3
    assert raster_text_runtime.raster_text_service_startup_timeout_seconds(profile) == 60
    assert (
        raster_text_runtime._sidecar_bootstrap_for_profile(profile)  # noqa: SLF001
        == raster_text_runtime._DIRECT_PADDLE_V2_ISOLATED_SIDECAR_BOOTSTRAP  # noqa: SLF001
    )


def test_v9_uses_the_same_two_root_boundary_and_keeps_the_30_60_bounds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile = _v9_profile()
    mobile_root = tmp_path / "MOBILE~1"
    expected = mobile_root / "models" / "text-det"
    expected.mkdir(parents=True)
    calls: list[str] = []

    def resolved_root(_root: Path, value: str) -> Path:
        calls.append(value)
        assert value == profile.mobile_model_bundle_root_relative_path
        return mobile_root

    monkeypatch.setattr(raster_text_runtime, "_non_symlink_sidecar_path", resolved_root)
    resolved = raster_text_runtime._v2_bundle_sidecar_path(  # noqa: SLF001 - boundary test
        root=tmp_path / "document-preview",
        profile=profile,
        value=f"{profile.mobile_model_bundle_root_relative_path}/models/text-det",
    )

    assert resolved == expected
    assert calls == [profile.mobile_model_bundle_root_relative_path]
    assert profile.sidecar_timeout_seconds == 30
    assert profile.text_detection_limit_side_len == 960
    assert profile.text_detection_box_threshold == 0.3
    assert raster_text_runtime.raster_text_service_startup_timeout_seconds(profile) == 60
    assert (
        raster_text_runtime._sidecar_bootstrap_for_profile(profile)  # noqa: SLF001
        == raster_text_runtime._DIRECT_PADDLE_V2_ISOLATED_SIDECAR_BOOTSTRAP  # noqa: SLF001
    )


def test_v10_uses_the_same_two_root_boundary_and_keeps_the_30_60_bounds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile = _v10_profile()
    mobile_root = tmp_path / "MOBILE~1"
    expected = mobile_root / "models" / "text-det"
    expected.mkdir(parents=True)
    calls: list[str] = []

    def resolved_root(_root: Path, value: str) -> Path:
        calls.append(value)
        assert value == profile.mobile_model_bundle_root_relative_path
        return mobile_root

    monkeypatch.setattr(raster_text_runtime, "_non_symlink_sidecar_path", resolved_root)
    resolved = raster_text_runtime._v2_bundle_sidecar_path(  # noqa: SLF001 - boundary test
        root=tmp_path / "document-preview",
        profile=profile,
        value=f"{profile.mobile_model_bundle_root_relative_path}/models/text-det",
    )

    assert resolved == expected
    assert calls == [profile.mobile_model_bundle_root_relative_path]
    assert profile.sidecar_timeout_seconds == 30
    assert profile.text_detection_limit_side_len == 960
    assert profile.text_detection_box_threshold == 0.3
    assert raster_text_runtime.raster_text_service_startup_timeout_seconds(profile) == 60
    assert (
        raster_text_runtime._sidecar_bootstrap_for_profile(profile)  # noqa: SLF001
        == raster_text_runtime._DIRECT_PADDLE_V2_ISOLATED_SIDECAR_BOOTSTRAP  # noqa: SLF001
    )


def test_v11_uses_the_same_two_root_boundary_and_keeps_the_30_60_bounds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile = _v11_profile()
    mobile_root = tmp_path / "MOBILE~1"
    expected = mobile_root / "models" / "text-det"
    expected.mkdir(parents=True)
    calls: list[str] = []

    def resolved_root(_root: Path, value: str) -> Path:
        calls.append(value)
        assert value == profile.mobile_model_bundle_root_relative_path
        return mobile_root

    monkeypatch.setattr(raster_text_runtime, "_non_symlink_sidecar_path", resolved_root)
    resolved = raster_text_runtime._v2_bundle_sidecar_path(  # noqa: SLF001 - boundary test
        root=tmp_path / "document-preview",
        profile=profile,
        value=f"{profile.mobile_model_bundle_root_relative_path}/models/text-det",
    )

    assert resolved == expected
    assert calls == [profile.mobile_model_bundle_root_relative_path]
    assert profile.sidecar_timeout_seconds == 30
    assert profile.text_detection_limit_side_len == 960
    assert profile.text_detection_box_threshold == 0.3
    assert raster_text_runtime.raster_text_service_startup_timeout_seconds(profile) == 60
    assert (
        raster_text_runtime._sidecar_bootstrap_for_profile(profile)  # noqa: SLF001
        == raster_text_runtime._DIRECT_PADDLE_V2_ISOLATED_SIDECAR_BOOTSTRAP  # noqa: SLF001
    )


def test_v12_uses_only_the_combined_model_root_and_keeps_v9_bounds(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile = _v12_profile()
    combined_root = tmp_path / "TRAINED~1"
    expected = combined_root / "models" / "text-rec"
    expected.mkdir(parents=True)
    calls: list[str] = []

    def resolved_root(_root: Path, value: str) -> Path:
        calls.append(value)
        assert value == profile.combined_model_bundle_root_relative_path
        return combined_root

    monkeypatch.setattr(raster_text_runtime, "_non_symlink_sidecar_path", resolved_root)
    resolved = raster_text_runtime._v2_bundle_sidecar_path(  # noqa: SLF001
        root=tmp_path / "document-preview",
        profile=profile,
        value=f"{profile.combined_model_bundle_root_relative_path}/models/text-rec",
    )

    assert resolved == expected
    assert calls == [profile.combined_model_bundle_root_relative_path]
    assert profile.sidecar_timeout_seconds == 30
    assert profile.text_detection_limit_side_len == 960
    assert profile.text_detection_box_threshold == 0.3
    assert profile.text_recognition_model_name == "source_attested_tiny_crnn_bigru_ctc_v1"
    assert raster_text_runtime.raster_text_service_startup_timeout_seconds(profile) == 60
    assert (
        raster_text_runtime._sidecar_bootstrap_for_profile(profile)  # noqa: SLF001
        == raster_text_runtime._DIRECT_PADDLE_V2_ISOLATED_SIDECAR_BOOTSTRAP  # noqa: SLF001
    )
    with pytest.raises(RasterTextRuntimeError, match="runtime_material_invalid"):
        raster_text_runtime._v2_bundle_sidecar_path(  # noqa: SLF001
            root=tmp_path / "document-preview",
            profile=profile,
            value=f"raster-text-direct-mobile-bundles-v1/releases/{'f' * 64}/models/text-rec",
        )


def test_v12_parser_rejects_unknown_and_mixed_profile_tuples() -> None:
    profile = _v12_profile()
    raw = _canonical(profile.model_dump(mode="json", warnings=False))

    assert raster_text_runtime.parse_raster_text_offline_profile(raw) == profile
    for schema_version, engine_kind in (
        (
            "12.0",
            "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2",
        ),
        (
            "9.0",
            "direct_paddle_source_attested_text_harmonic_recall_native960_lowbox_strip2",
        ),
        (
            "14.0",
            "direct_paddle_source_attested_text_harmonic_recall_native960_lowbox_strip2",
        ),
    ):
        values = profile.model_dump(mode="json", warnings=False)
        values.update({"schema_version": schema_version, "engine_kind": engine_kind})
        with pytest.raises(ValueError):
            raster_text_runtime.parse_raster_text_offline_profile(_canonical(values))


def test_v13_parser_and_runtime_keep_exact_public_v9_material_without_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile = _v13_profile()
    raw = _canonical(profile.model_dump(mode="json", warnings=False))

    assert raster_text_runtime.parse_raster_text_offline_profile(raw) == profile
    assert profile.text_detection_model_name == "PP-OCRv5_mobile_det"
    assert profile.text_recognition_model_name == "en_PP-OCRv5_mobile_rec"
    assert profile.sidecar_timeout_seconds == 30
    assert profile.cpu_threads == 6
    assert profile.text_detection_limit_side_len == 960
    assert profile.text_detection_threshold == 0.3
    assert profile.text_detection_box_threshold == 0.3
    assert profile.text_detection_unclip_ratio == 1.5
    assert raster_text_runtime.raster_text_service_startup_timeout_seconds(profile) == 60
    assert (
        raster_text_runtime._sidecar_bootstrap_for_profile(profile)  # noqa: SLF001
        == raster_text_runtime._DIRECT_PADDLE_V2_ISOLATED_SIDECAR_BOOTSTRAP  # noqa: SLF001
    )

    mobile_root = tmp_path / "MOBILE~1"
    expected = mobile_root / "models" / "text-rec"
    expected.mkdir(parents=True)
    resolved_roots: list[str] = []

    def resolve_root(_root: Path, value: str) -> Path:
        resolved_roots.append(value)
        assert value == profile.mobile_model_bundle_root_relative_path
        return mobile_root

    monkeypatch.setattr(raster_text_runtime, "_non_symlink_sidecar_path", resolve_root)
    resolved = raster_text_runtime._v2_bundle_sidecar_path(  # noqa: SLF001
        root=tmp_path / "document-preview",
        profile=profile,
        value=f"{profile.mobile_model_bundle_root_relative_path}/models/text-rec",
    )
    assert resolved == expected
    assert resolved_roots == [profile.mobile_model_bundle_root_relative_path]

    for schema_version, engine_kind in (
        (
            "9.0",
            "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_semantic_repair",
        ),
        (
            "13.0",
            "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2",
        ),
        (
            "12.0",
            "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_semantic_repair",
        ),
    ):
        values = profile.model_dump(mode="json", warnings=False)
        values.update({"schema_version": schema_version, "engine_kind": engine_kind})
        with pytest.raises(ValueError):
            raster_text_runtime.parse_raster_text_offline_profile(_canonical(values))


def test_v14_parser_runtime_and_fingerprint_are_distinct_from_v13(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile = _v14_profile()
    v13 = _v13_profile()
    raw = _canonical(profile.model_dump(mode="json", warnings=False))

    assert raster_text_runtime.parse_raster_text_offline_profile(raw) == profile
    assert profile.raster_text_profile_revision != v13.raster_text_profile_revision
    assert profile.profile_fingerprint_sha256 != v13.profile_fingerprint_sha256
    assert profile.text_detection_model_name == v13.text_detection_model_name
    assert profile.text_recognition_model_name == v13.text_recognition_model_name
    assert profile.mobile_model_bundle_manifest_sha256 == v13.mobile_model_bundle_manifest_sha256
    assert profile.bundle_manifest_sha256 == v13.bundle_manifest_sha256
    assert profile.sidecar_timeout_seconds == 30
    assert raster_text_runtime.raster_text_service_startup_timeout_seconds(profile) == 60
    assert (
        raster_text_runtime._sidecar_bootstrap_for_profile(profile)  # noqa: SLF001
        == raster_text_runtime._DIRECT_PADDLE_V2_ISOLATED_SIDECAR_BOOTSTRAP  # noqa: SLF001
    )

    mobile_root = tmp_path / "MOBILE~1"
    expected = mobile_root / "models" / "text-rec"
    expected.mkdir(parents=True)
    monkeypatch.setattr(
        raster_text_runtime,
        "_non_symlink_sidecar_path",
        lambda _root, value: (
            mobile_root
            if value == profile.mobile_model_bundle_root_relative_path
            else pytest.fail("unexpected release root")
        ),
    )
    assert (
        raster_text_runtime._v2_bundle_sidecar_path(  # noqa: SLF001
            root=tmp_path / "document-preview",
            profile=profile,
            value=f"{profile.mobile_model_bundle_root_relative_path}/models/text-rec",
        )
        == expected
    )

    for schema_version, engine_kind in (
        (
            "13.0",
            "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_"
            "strip2_semantic_repair_v2",
        ),
        (
            "14.0",
            "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_"
            "strip2_semantic_repair",
        ),
        (
            "9.0",
            "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_"
            "strip2_semantic_repair_v2",
        ),
    ):
        values = profile.model_dump(mode="json", warnings=False)
        values.update({"schema_version": schema_version, "engine_kind": engine_kind})
        with pytest.raises(ValueError):
            raster_text_runtime.parse_raster_text_offline_profile(_canonical(values))


def test_activation_pointer_parser_preserves_v1_and_requires_v2_metric() -> None:
    pointer_v2, _calibration, _evaluation = _semantic_evidence_models()
    v2_raw = _canonical_pointer(pointer_v2.model_dump(mode="json", warnings=False))

    assert raster_text_runtime.parse_raster_text_activation_pointer(v2_raw) == pointer_v2
    v1_values = pointer_v2.model_dump(mode="json", warnings=False)
    v1_values.pop("metric_revision")
    v1_values["schema_version"] = "1.0"
    pointer_v1 = RasterTextActivationPointerV1.model_validate(v1_values)
    assert (
        raster_text_runtime.parse_raster_text_activation_pointer(_canonical_pointer(v1_values))
        == pointer_v1
    )

    missing_metric = pointer_v2.model_dump(mode="json", warnings=False)
    missing_metric.pop("metric_revision")
    with pytest.raises(RasterTextRuntimeError, match="activation_invalid"):
        raster_text_runtime.parse_raster_text_activation_pointer(_canonical_pointer(missing_metric))

    pointer_v3, _calibration_v3, _evaluation_v3 = _semantic_v3_evidence_models()
    assert raster_text_runtime.parse_raster_text_activation_pointer(
        _canonical_pointer(pointer_v3.model_dump(mode="json", warnings=False))
    ) == pointer_v3
    v3_missing_metric = pointer_v3.model_dump(mode="json", warnings=False)
    v3_missing_metric.pop("metric_revision")
    with pytest.raises(RasterTextRuntimeError, match="activation_invalid"):
        raster_text_runtime.parse_raster_text_activation_pointer(
            _canonical_pointer(v3_missing_metric)
        )

    pointer_v4, _calibration_v4, _evaluation_v4 = _fresh_v4_evidence_models()
    assert raster_text_runtime.parse_raster_text_activation_pointer(
        _canonical_pointer(pointer_v4.model_dump(mode="json", warnings=False))
    ) == pointer_v4
    for missing_field in ("metric_revision", "evidence_revision"):
        missing_binding = pointer_v4.model_dump(mode="json", warnings=False)
        missing_binding.pop(missing_field)
        with pytest.raises(RasterTextRuntimeError, match="activation_invalid"):
            raster_text_runtime.parse_raster_text_activation_pointer(
                _canonical_pointer(missing_binding)
            )


def test_runtime_semantic_evidence_models_match_writer_schema() -> None:
    from customs_ai_agent.evaluation.synthetic_raster_text_activation import (
        RasterTextCalibrationReceiptV2 as EvaluationCalibrationV2,
    )
    from customs_ai_agent.evaluation.synthetic_raster_text_activation import (
        RasterTextSealedEvaluationV2 as EvaluationSealedV2,
    )
    from customs_ai_agent.evaluation.synthetic_raster_text_activation import (
        RasterTextCalibrationReceiptV3 as EvaluationCalibrationV3,
    )
    from customs_ai_agent.evaluation.synthetic_raster_text_activation import (
        RasterTextSealedEvaluationV3 as EvaluationSealedV3,
    )
    from customs_ai_agent.evaluation.synthetic_raster_text_activation import (
        RasterTextCalibrationReceiptV4 as EvaluationCalibrationV4,
    )
    from customs_ai_agent.evaluation.synthetic_raster_text_activation import (
        RasterTextSealedEvaluationV4 as EvaluationSealedV4,
    )

    _pointer, calibration, evaluation = _semantic_evidence_models()
    assert EvaluationCalibrationV2.model_validate(
        calibration.model_dump(mode="json", warnings=False)
    ).model_dump(mode="json", warnings=False) == calibration.model_dump(mode="json", warnings=False)
    assert EvaluationSealedV2.model_validate(
        evaluation.model_dump(mode="json", warnings=False)
    ).model_dump(mode="json", warnings=False) == evaluation.model_dump(mode="json", warnings=False)
    for parser, artifact in (
        (raster_text_runtime._parse_raster_text_calibration_receipt, calibration),  # noqa: SLF001
        (raster_text_runtime._parse_raster_text_sealed_evaluation, evaluation),  # noqa: SLF001
    ):
        missing_metric = artifact.model_dump(mode="json", warnings=False)
        missing_metric.pop("metric_revision")
        with pytest.raises(RasterTextRuntimeError, match="evaluation_invalid"):
            parser(_canonical(missing_metric))

    _pointer_v4, calibration_v4, evaluation_v4 = _fresh_v4_evidence_models()
    assert EvaluationCalibrationV4.model_validate(
        calibration_v4.model_dump(mode="json", warnings=False)
    ) == calibration_v4
    assert EvaluationSealedV4.model_validate(
        evaluation_v4.model_dump(mode="json", warnings=False)
    ) == evaluation_v4
    for parser, artifact in (
        (raster_text_runtime._parse_raster_text_calibration_receipt, calibration_v4),  # noqa: SLF001
        (raster_text_runtime._parse_raster_text_sealed_evaluation, evaluation_v4),  # noqa: SLF001
    ):
        for missing_field in ("metric_revision", "evidence_revision"):
            missing_binding = artifact.model_dump(mode="json", warnings=False)
            missing_binding.pop(missing_field)
            with pytest.raises(RasterTextRuntimeError, match="evaluation_invalid"):
                parser(_canonical(missing_binding))

    _pointer_v3, calibration_v3, evaluation_v3 = _semantic_v3_evidence_models()
    assert EvaluationCalibrationV3.model_validate(
        calibration_v3.model_dump(mode="json", warnings=False)
    ).model_dump(mode="json", warnings=False) == calibration_v3.model_dump(
        mode="json", warnings=False
    )
    assert EvaluationSealedV3.model_validate(
        evaluation_v3.model_dump(mode="json", warnings=False)
    ).model_dump(mode="json", warnings=False) == evaluation_v3.model_dump(
        mode="json", warnings=False
    )
    for parser, artifact in (
        (raster_text_runtime._parse_raster_text_calibration_receipt, calibration_v3),  # noqa: SLF001
        (raster_text_runtime._parse_raster_text_sealed_evaluation, evaluation_v3),  # noqa: SLF001
    ):
        missing_metric = artifact.model_dump(mode="json", warnings=False)
        missing_metric.pop("metric_revision")
        with pytest.raises(RasterTextRuntimeError, match="evaluation_invalid"):
            parser(_canonical(missing_metric))


def test_metric_binding_accepts_each_generation_and_rejects_mixing() -> None:
    pointer_v2, calibration_v2, evaluation_v2 = _semantic_evidence_models()
    pointer_v3, calibration_v3, evaluation_v3 = _semantic_v3_evidence_models()
    pointer_v4, calibration_v4, evaluation_v4 = _fresh_v4_evidence_models()
    pointer_v1_values = pointer_v2.model_dump(mode="json", warnings=False)
    pointer_v1_values.pop("metric_revision")
    pointer_v1_values["schema_version"] = "1.0"
    pointer_v1 = RasterTextActivationPointerV1.model_validate(pointer_v1_values)
    calibration_v1_values = calibration_v2.model_dump(mode="json", warnings=False)
    calibration_v1_values.pop("artifact_kind")
    calibration_v1_values.pop("metric_revision")
    calibration_v1_values["schema_version"] = "1.0"
    calibration_v1 = RasterTextCalibrationReceiptV1.model_validate(calibration_v1_values)
    evaluation_v1_values = evaluation_v2.model_dump(mode="json", warnings=False)
    evaluation_v1_values.pop("artifact_kind")
    evaluation_v1_values.pop("metric_revision")
    evaluation_v1_values["schema_version"] = "1.0"
    evaluation_v1_values["scores"] = [
        RasterTextSealedProfileScoreV1.model_validate(value).model_dump(mode="json", warnings=False)
        for value in _passing_score_payloads()
    ]
    evaluation_v1 = RasterTextSealedEvaluationV1.model_validate(evaluation_v1_values)

    assert (
        raster_text_runtime._bound_raster_text_metric_revision(  # noqa: SLF001
            pointer=pointer_v1,
            evaluation=evaluation_v1,
            calibration=calibration_v1,
        )
        is None
    )
    assert (
        raster_text_runtime._bound_raster_text_metric_revision(  # noqa: SLF001
            pointer=pointer_v2,
            evaluation=evaluation_v2,
            calibration=calibration_v2,
        )
        == "source-attested-semantic-text-v2"
    )
    assert (
        raster_text_runtime._bound_raster_text_metric_revision(  # noqa: SLF001
            pointer=pointer_v3,
            evaluation=evaluation_v3,
            calibration=calibration_v3,
        )
        == "source-attested-semantic-text-v3"
    )
    assert (
        raster_text_runtime._bound_raster_text_metric_revision(  # noqa: SLF001
            pointer=pointer_v4,
            evaluation=evaluation_v4,
            calibration=calibration_v4,
        )
        == "source-attested-semantic-text-v3"
    )
    for pointer, evaluation, calibration in (
        (pointer_v1, evaluation_v2, calibration_v2),
        (pointer_v2, evaluation_v1, calibration_v1),
        (pointer_v2, evaluation_v2, calibration_v1),
        (pointer_v2, evaluation_v3, calibration_v3),
        (pointer_v3, evaluation_v2, calibration_v2),
        (pointer_v3, evaluation_v3, calibration_v2),
        (pointer_v3, evaluation_v4, calibration_v4),
        (pointer_v4, evaluation_v3, calibration_v3),
        (pointer_v4, evaluation_v4, calibration_v3),
        (pointer_v4, evaluation_v3, calibration_v4),
    ):
        with pytest.raises(RasterTextRuntimeError, match="evaluation_invalid"):
            raster_text_runtime._bound_raster_text_metric_revision(  # noqa: SLF001
                pointer=pointer,
                evaluation=evaluation,
                calibration=calibration,
            )


@pytest.mark.parametrize(
    "mismatch",
    (
        None,
        "calibration_profile_floor",
        "calibration_latency",
        "sealed_latency",
        "activated_floor",
        "warm_latency",
    ),
)
def test_fresh_v4_chain_binds_floors_and_latencies_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mismatch: str | None,
) -> None:
    from customs_ai_agent.evaluation import synthetic_raster_text_activation

    root = tmp_path / "document-preview"
    selection_relative = "fresh-scope-selections-v1/selection.json"
    calibration_configuration_relative = "profiles/calibration-v14.json"
    selection_raw = b"fresh-selection"
    selection_path = root / "fresh-scope-selections-v1" / "selection.json"
    selection_path.parent.mkdir(parents=True)
    selection_path.write_bytes(selection_raw)
    selection_sha256 = hashlib.sha256(selection_raw).hexdigest()

    final_profile = _v14_profile(minimum_page_confidence=0.5)
    calibration_profile = _v14_profile(
        minimum_page_confidence=0.1 if mismatch == "calibration_profile_floor" else 0.0
    )
    calibration_configuration_raw = _canonical_pointer(
        calibration_profile.model_dump(mode="json", warnings=False)
    )
    calibration_configuration_path = root / "profiles" / "calibration-v14.json"
    calibration_configuration_path.parent.mkdir(parents=True)
    calibration_configuration_path.write_bytes(calibration_configuration_raw)
    calibration_configuration_sha256 = hashlib.sha256(
        calibration_configuration_raw
    ).hexdigest()
    final_runtime_sha256 = "c" * 64
    pointer, calibration, evaluation = _fresh_v4_evidence_models()
    pointer = pointer.model_copy(
        update={
            "raster_text_profile_revision": final_profile.raster_text_profile_revision,
            "runtime_configuration_sha256": final_runtime_sha256,
        }
    )
    calibration = calibration.model_copy(
        update={
            "raster_text_profile_revision": final_profile.raster_text_profile_revision,
            "runtime_configuration_sha256": final_runtime_sha256,
            "calibration_raster_text_profile_revision": (
                calibration_profile.raster_text_profile_revision
            ),
            "calibration_runtime_configuration_path": calibration_configuration_relative,
            "calibration_runtime_configuration_sha256": calibration_configuration_sha256,
            "extractor_revision": final_profile.extractor_revision,
            "selection_receipt_path": selection_relative,
            "selection_receipt_sha256": selection_sha256,
            "latency_p95_seconds": 2.0 if mismatch == "calibration_latency" else 1.0,
        }
    )
    evaluation = evaluation.model_copy(
        update={
            "raster_text_profile_revision": final_profile.raster_text_profile_revision,
            "runtime_configuration_sha256": final_runtime_sha256,
            "extractor_revision": final_profile.extractor_revision,
            "selection_receipt_path": selection_relative,
            "selection_receipt_sha256": selection_sha256,
            "latency_p95_seconds": 2.0 if mismatch == "sealed_latency" else 1.0,
            "minimum_page_confidence": 0.4 if mismatch == "activated_floor" else 0.5,
            "service_warm_latency_p95_seconds": 2.0 if mismatch == "warm_latency" else 1.0,
        }
    )
    warm = RasterTextWarmServiceSmokeReceiptV1(
        artifact_kind="synthetic_raster_text_warm_service_smoke",
        dataset_id="synthetic-raster-text-v1",
        scope="smoke",
        source_truth_kind="pdf_content_stream_text_v1",
        execution_mode="persistent_service_warm",
        service_protocol="customs-raster-text-sidecar-v1-service",
        raster_text_profile_revision=final_profile.raster_text_profile_revision,
        runtime_configuration_sha256=final_runtime_sha256,
        extractor_revision=final_profile.extractor_revision,
        source_gold_receipt_path="evidence/smoke-source.json",
        source_gold_receipt_sha256="1" * 64,
        candidate_execution_path="evidence/smoke-execution.json",
        candidate_execution_sha256="2" * 64,
        candidate_bundle_sha256="3" * 64,
        equality_mode="exact_canonical_result_v1",
        equality_set_sha256="4" * 64,
        asset_count=24,
        startup_latency_seconds=1.0,
        startup_timeout_seconds=1,
        warm_latency_p95_seconds=1.0,
        request_timeout_seconds=30,
    )
    warm_raw = _canonical(warm.model_dump(mode="json", warnings=False))

    def fresh_artifacts(*, scope: str) -> tuple[object, object, object, list[object]]:
        is_sealed = scope == "fresh-sealed"
        source_path = (
            evaluation.source_gold_receipt_path
            if is_sealed
            else calibration.source_gold_receipt_path
        )
        source_sha256 = (
            evaluation.source_gold_receipt_sha256
            if is_sealed
            else calibration.source_gold_receipt_sha256
        )
        bound_profile = final_profile if is_sealed else calibration_profile
        bound_runtime_sha256 = (
            final_runtime_sha256 if is_sealed else calibration_configuration_sha256
        )
        return (
            SimpleNamespace(
                scope=scope,
                selection_receipt_relative_path=selection_relative,
                selection_receipt_sha256=selection_sha256,
            ),
            SimpleNamespace(
                scope=scope,
                selection_receipt_relative_path=selection_relative,
                selection_receipt_sha256=selection_sha256,
                source_gold_receipt_relative_path=source_path,
                source_gold_receipt_sha256=source_sha256,
                raster_text_profile_revision=bound_profile.raster_text_profile_revision,
                runtime_configuration_sha256=bound_runtime_sha256,
                extractor_revision=bound_profile.extractor_revision,
                latency_p95_seconds=1.0,
            ),
            SimpleNamespace(
                scope=scope,
                selection_receipt_relative_path=selection_relative,
                selection_receipt_sha256=selection_sha256,
                source_gold_receipt_sha256=source_sha256,
                raster_text_profile_revision=bound_profile.raster_text_profile_revision,
                runtime_configuration_sha256=bound_runtime_sha256,
                extractor_revision=bound_profile.extractor_revision,
            ),
            [SimpleNamespace(scope=scope)],
        )

    monkeypatch.setattr(
        synthetic_raster_text_activation,
        "verify_source_attested_fresh_raster_text_evidence",
        lambda **kwargs: fresh_artifacts(scope=cast(str, kwargs["scope"])),
    )
    monkeypatch.setattr(
        raster_text_runtime,
        "_recalculate_source_attested_raster_text_quality",
        lambda **_kwargs: (evaluation.scores, 0.5, True),
    )

    verify = lambda: raster_text_runtime._verify_fresh_v4_evidence_chain(  # noqa: E731, SLF001
        root=root,
        pointer=pointer,
        evaluation=evaluation,
        calibration=calibration,
        profile=final_profile,
        warm_service_smoke_raw=warm_raw,
    )
    if mismatch is None:
        verify()
    else:
        with pytest.raises(RasterTextRuntimeError, match="evaluation_invalid"):
            verify()


def test_semantic_metric_recalculation_uses_only_v2_evaluator_and_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from customs_ai_agent.evaluation import synthetic_raster_text
    from customs_ai_agent.evaluation import synthetic_raster_text_activation

    calls: list[object] = []
    scores = [SimpleNamespace(marker="semantic-v2")]

    def semantic_evaluate(cases: object) -> list[SimpleNamespace]:
        calls.append(("semantic", cases))
        return scores

    def semantic_gate(values: object) -> bool:
        calls.append(("semantic-gate", values))
        return True

    monkeypatch.setattr(
        synthetic_raster_text,
        "evaluate_synthetic_raster_semantic_text_gates_v2",
        semantic_evaluate,
    )
    monkeypatch.setattr(
        synthetic_raster_text,
        "synthetic_raster_semantic_text_gate_passes_v2",
        semantic_gate,
    )
    monkeypatch.setattr(
        synthetic_raster_text,
        "evaluate_synthetic_raster_text_gates",
        lambda _cases: pytest.fail("semantic evidence must not enter the V1 evaluator"),
    )
    monkeypatch.setattr(
        synthetic_raster_text,
        "synthetic_raster_text_gate_passes",
        lambda _scores: pytest.fail("semantic evidence must not enter the V1 gate"),
    )

    def floor(cases: object, *, metric_revision: str | None = None) -> float:
        calls.append(("floor", cases, metric_revision))
        return 0.5

    monkeypatch.setattr(synthetic_raster_text_activation, "_calibrated_floor", floor)
    observed_scores, observed_floor, passed = (
        raster_text_runtime._recalculate_source_attested_raster_text_quality(  # noqa: SLF001
            sealed_cases=["sealed"],
            calibration_cases=["calibration"],
            metric_revision="source-attested-semantic-text-v2",
        )
    )

    assert observed_scores == scores
    assert observed_floor == 0.5
    assert passed is True
    assert calls == [
        ("semantic", ["sealed"]),
        ("floor", ["calibration"], "source-attested-semantic-text-v2"),
        ("semantic-gate", scores),
    ]


def test_semantic_metric_recalculation_uses_only_v3_evaluator_and_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from customs_ai_agent.evaluation import synthetic_raster_text
    from customs_ai_agent.evaluation import synthetic_raster_text_activation

    calls: list[object] = []
    scores = [SimpleNamespace(marker="semantic-v3")]

    def semantic_evaluate(cases: object) -> list[SimpleNamespace]:
        calls.append(("semantic-v3", cases))
        return scores

    def semantic_gate(values: object) -> bool:
        calls.append(("semantic-v3-gate", values))
        return True

    monkeypatch.setattr(
        synthetic_raster_text,
        "evaluate_synthetic_raster_semantic_text_gates_v3",
        semantic_evaluate,
    )
    monkeypatch.setattr(
        synthetic_raster_text,
        "synthetic_raster_semantic_text_gate_passes_v3",
        semantic_gate,
    )
    monkeypatch.setattr(
        synthetic_raster_text,
        "evaluate_synthetic_raster_semantic_text_gates_v2",
        lambda _cases: pytest.fail("V3 evidence must not enter the V2 evaluator"),
    )
    monkeypatch.setattr(
        synthetic_raster_text,
        "evaluate_synthetic_raster_text_gates",
        lambda _cases: pytest.fail("V3 evidence must not enter the V1 evaluator"),
    )

    def floor(cases: object, *, metric_revision: str | None = None) -> float:
        calls.append(("floor", cases, metric_revision))
        return 0.5

    monkeypatch.setattr(synthetic_raster_text_activation, "_calibrated_floor", floor)
    observed_scores, observed_floor, passed = (
        raster_text_runtime._recalculate_source_attested_raster_text_quality(  # noqa: SLF001
            sealed_cases=["sealed"],
            calibration_cases=["calibration"],
            metric_revision="source-attested-semantic-text-v3",
        )
    )

    assert observed_scores == scores
    assert observed_floor == 0.5
    assert passed is True
    assert calls == [
        ("semantic-v3", ["sealed"]),
        ("floor", ["calibration"], "source-attested-semantic-text-v3"),
        ("semantic-v3-gate", scores),
    ]


def test_pointer_writer_derives_v2_metric_and_preserves_existing_pointer_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    configuration_relative = "profiles/final.json"
    evaluation_relative = "semantic-sealed-evaluations-v2/final.json"
    configuration_path = root / "profiles" / "final.json"
    evaluation_path = root / "semantic-sealed-evaluations-v2" / "final.json"
    configuration_path.parent.mkdir(parents=True)
    evaluation_path.parent.mkdir(parents=True)
    configuration_raw = b"profile"
    configuration_path.write_bytes(configuration_raw)
    configuration_sha256 = hashlib.sha256(configuration_raw).hexdigest()
    _pointer, _calibration, evaluation = _semantic_evidence_models()
    evaluation = evaluation.model_copy(
        update={"runtime_configuration_sha256": configuration_sha256}
    )
    evaluation_path.write_bytes(_canonical(evaluation.model_dump(mode="json", warnings=False)))
    profile = SimpleNamespace(
        raster_text_profile_revision=_REVISION,
        extractor_revision="direct-text-test-v2",
    )
    observed: list[RasterTextActivationPointerV2] = []
    monkeypatch.setattr(
        raster_text_runtime,
        "parse_raster_text_offline_profile",
        lambda _raw: profile,
    )

    def accept_pointer(**kwargs: object) -> SimpleNamespace:
        parsed = raster_text_runtime.parse_raster_text_activation_pointer(
            cast(bytes, kwargs["approval_raw"])
        )
        assert isinstance(parsed, RasterTextActivationPointerV2)
        observed.append(parsed)
        return SimpleNamespace()

    monkeypatch.setattr(raster_text_runtime, "_load_binding_from_pointer", accept_pointer)
    _digest, pointer = raster_text_runtime.write_source_attested_synthetic_raster_text_pointer(
        document_preview_root=root,
        runtime_configuration_relative_path=configuration_relative,
        sealed_evaluation_relative_path=evaluation_relative,
    )
    destination = root / "approvals" / "raster-text-active.json"
    assert isinstance(pointer, RasterTextActivationPointerV2)
    assert pointer.metric_revision == "source-attested-semantic-text-v2"
    assert observed == [pointer]
    assert (
        raster_text_runtime.parse_raster_text_activation_pointer(destination.read_bytes())
        == pointer
    )

    previous = destination.read_bytes()

    def reject_pointer(**_kwargs: object) -> None:
        raise RasterTextRuntimeError("raster_text_evaluation_invalid")

    monkeypatch.setattr(raster_text_runtime, "_load_binding_from_pointer", reject_pointer)
    with pytest.raises(RasterTextRuntimeError, match="evaluation_invalid"):
        raster_text_runtime.write_source_attested_synthetic_raster_text_pointer(
            document_preview_root=root,
            runtime_configuration_relative_path=configuration_relative,
            sealed_evaluation_relative_path=evaluation_relative,
        )
    assert destination.read_bytes() == previous


def test_pointer_writer_derives_v3_schema_and_metric(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    configuration_relative = "profiles/final.json"
    evaluation_relative = "semantic-sealed-evaluations-v3/final.json"
    configuration_path = root / "profiles" / "final.json"
    evaluation_path = root / "semantic-sealed-evaluations-v3" / "final.json"
    configuration_path.parent.mkdir(parents=True)
    evaluation_path.parent.mkdir(parents=True)
    configuration_raw = b"profile-v3"
    configuration_path.write_bytes(configuration_raw)
    configuration_sha256 = hashlib.sha256(configuration_raw).hexdigest()
    _pointer, _calibration, evaluation = _semantic_v3_evidence_models()
    evaluation = evaluation.model_copy(
        update={"runtime_configuration_sha256": configuration_sha256}
    )
    evaluation_path.write_bytes(_canonical(evaluation.model_dump(mode="json", warnings=False)))
    profile = SimpleNamespace(
        raster_text_profile_revision=_REVISION,
        extractor_revision="direct-text-test-v2",
    )
    observed: list[RasterTextActivationPointerV3] = []
    monkeypatch.setattr(
        raster_text_runtime,
        "parse_raster_text_offline_profile",
        lambda _raw: profile,
    )

    def accept_pointer(**kwargs: object) -> SimpleNamespace:
        parsed = raster_text_runtime.parse_raster_text_activation_pointer(
            cast(bytes, kwargs["approval_raw"])
        )
        assert isinstance(parsed, RasterTextActivationPointerV3)
        observed.append(parsed)
        return SimpleNamespace()

    monkeypatch.setattr(raster_text_runtime, "_load_binding_from_pointer", accept_pointer)
    _digest, pointer = raster_text_runtime.write_source_attested_synthetic_raster_text_pointer(
        document_preview_root=root,
        runtime_configuration_relative_path=configuration_relative,
        sealed_evaluation_relative_path=evaluation_relative,
    )

    assert isinstance(pointer, RasterTextActivationPointerV3)
    assert pointer.schema_version == "3.0"
    assert pointer.metric_revision == "source-attested-semantic-text-v3"
    assert observed == [pointer]


def test_pointer_writer_derives_fresh_v4_binding_and_preserves_pointer_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    configuration_relative = "profiles/final-v14.json"
    evaluation_relative = "fresh-sealed-evaluations-v4/final.json"
    configuration_path = root / "profiles" / "final-v14.json"
    evaluation_path = root / "fresh-sealed-evaluations-v4" / "final.json"
    configuration_path.parent.mkdir(parents=True)
    evaluation_path.parent.mkdir(parents=True)
    configuration_raw = b"profile-v14"
    configuration_path.write_bytes(configuration_raw)
    configuration_sha256 = hashlib.sha256(configuration_raw).hexdigest()
    _pointer, _calibration, evaluation = _fresh_v4_evidence_models()
    evaluation = evaluation.model_copy(
        update={"runtime_configuration_sha256": configuration_sha256}
    )
    evaluation_path.write_bytes(_canonical(evaluation.model_dump(mode="json", warnings=False)))
    profile = SimpleNamespace(
        raster_text_profile_revision=_REVISION,
        extractor_revision="direct-text-test-v14",
    )
    observed: list[RasterTextActivationPointerV4] = []
    monkeypatch.setattr(
        raster_text_runtime,
        "parse_raster_text_offline_profile",
        lambda _raw: profile,
    )

    def accept_pointer(**kwargs: object) -> SimpleNamespace:
        parsed = raster_text_runtime.parse_raster_text_activation_pointer(
            cast(bytes, kwargs["approval_raw"])
        )
        assert isinstance(parsed, RasterTextActivationPointerV4)
        observed.append(parsed)
        return SimpleNamespace()

    monkeypatch.setattr(raster_text_runtime, "_load_binding_from_pointer", accept_pointer)
    _digest, pointer = raster_text_runtime.write_source_attested_synthetic_raster_text_pointer(
        document_preview_root=root,
        runtime_configuration_relative_path=configuration_relative,
        sealed_evaluation_relative_path=evaluation_relative,
    )

    destination = root / "approvals" / "raster-text-active.json"
    assert isinstance(pointer, RasterTextActivationPointerV4)
    assert pointer.schema_version == "4.0"
    assert pointer.metric_revision == "source-attested-semantic-text-v3"
    assert pointer.evidence_revision == "source-attested-fresh-holdout-v1"
    assert observed == [pointer]
    previous = destination.read_bytes()

    monkeypatch.setattr(
        raster_text_runtime,
        "_load_binding_from_pointer",
        lambda **_kwargs: (_ for _ in ()).throw(
            RasterTextRuntimeError("raster_text_evaluation_invalid")
        ),
    )
    with pytest.raises(RasterTextRuntimeError, match="evaluation_invalid"):
        raster_text_runtime.write_source_attested_synthetic_raster_text_pointer(
            document_preview_root=root,
            runtime_configuration_relative_path=configuration_relative,
            sealed_evaluation_relative_path=evaluation_relative,
        )
    assert destination.read_bytes() == previous


def test_v12_sidecar_environment_uses_only_the_direct_combined_model_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile = _v12_profile()
    monkeypatch.setattr(
        raster_text_runtime,
        "_isolated_v2_sidecar_environment",
        lambda **_kwargs: {"CUSTOMS_AGENT_PPSTRUCTURE_STALE": "1"},
    )
    model_root = tmp_path / "combined-models"
    model_inventory = tmp_path / "combined-inventory.json"

    environment = raster_text_runtime._sidecar_environment(  # noqa: SLF001
        interpreter=tmp_path / "python.exe",
        module_root=tmp_path / "module-root",
        runtime_home=tmp_path / "runtime-home",
        configuration_path=tmp_path / "profile.json",
        configuration_sha256="a" * 64,
        profile=profile,
        model_root=model_root,
        model_inventory=model_inventory,
        sidecar_inventory=tmp_path / "packages.json",
    )

    prefix = "CUSTOMS_AGENT_RASTER_TEXT_DIRECT_PADDLE_"
    assert environment[prefix + "MODEL_ROOT"] == str(model_root)
    assert environment[prefix + "MODEL_INVENTORY_PATH"] == str(model_inventory)
    assert environment["PYTHONDONTWRITEBYTECODE"] == "1"
    assert environment["PYTHONPYCACHEPREFIX"] == str(tmp_path / "runtime-home" / "pycache")
    assert "CUSTOMS_AGENT_RASTER_TEXT_MODEL_ROOT" not in environment
    assert not any(key.startswith("CUSTOMS_AGENT_PPSTRUCTURE_") for key in environment)
    assert not any("RECOGNIZER" in key for key in environment)


def test_v12_cached_binding_reverifies_combined_bundle_without_mobile_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile = _v12_profile()
    configuration_raw = _canonical(profile.model_dump(mode="json", warnings=False))
    configuration_path = tmp_path / "profile.json"
    pointer = raster_text_runtime.RasterTextActivationPointerV1(
        raster_text_profile_revision=profile.raster_text_profile_revision,
        runtime_configuration_path="profiles/v12.json",
        runtime_configuration_sha256=hashlib.sha256(configuration_raw).hexdigest(),
        sealed_evaluation_path="evaluations/v12.json",
        sealed_evaluation_sha256="f" * 64,
    )
    pointer_raw = _canonical_pointer(pointer.model_dump(mode="json", warnings=False))
    verified: list[RasterTextOfflineProfileV12] = []
    paths = {
        profile.sidecar_interpreter_root: tmp_path / "interpreter-root",
        profile.sidecar_interpreter_path: tmp_path / "python.exe",
        profile.sidecar_module_root: tmp_path / "module-root",
        profile.sidecar_module_path: tmp_path / "sidecar.py",
    }

    monkeypatch.setattr(
        raster_text_runtime,
        "_non_symlink_sidecar_path",
        lambda _root, _value: configuration_path,
    )
    monkeypatch.setattr(
        raster_text_runtime,
        "_read_local_file",
        lambda _path, **_kwargs: configuration_raw,
    )
    monkeypatch.setattr(
        raster_text_runtime,
        "_verify_v12_trained_model_bundle",
        lambda **kwargs: verified.append(kwargs["profile"]),
    )
    monkeypatch.setattr(
        raster_text_runtime,
        "_verify_v3_mobile_model_bundle",
        lambda **_kwargs: pytest.fail("V12 must not enter the V3--V11 verifier"),
    )
    monkeypatch.setattr(
        raster_text_runtime,
        "_v2_bundle_sidecar_path",
        lambda **kwargs: paths[kwargs["value"]],
    )
    monkeypatch.setattr(
        raster_text_runtime,
        "_validate_profile_controlled_child_file",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        raster_text_runtime,
        "_sha256_regular_file",
        lambda path: (
            profile.sidecar_interpreter_sha256
            if path == paths[profile.sidecar_interpreter_path]
            else profile.sidecar_module_sha256
        ),
    )

    revision = raster_text_runtime._validate_cached_raster_text_binding_materials(  # noqa: SLF001
        root=tmp_path,
        approval_path=tmp_path / "raster-text-active.json",
        approval_raw=pointer_raw,
        approval_sha256=hashlib.sha256(pointer_raw).hexdigest(),
    )

    assert revision == profile.raster_text_profile_revision
    assert verified == [profile]


def test_v12_full_material_loader_uses_only_the_combined_bundle_verifier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    inventory_raw = b"combined-inventory"
    values = _v12_profile().model_dump(
        mode="json",
        warnings=False,
        exclude={"profile_fingerprint_sha256", "raster_text_profile_revision"},
    )
    values["model_inventory_sha256"] = hashlib.sha256(inventory_raw).hexdigest()
    fingerprint = raster_text_runtime.raster_text_profile_fingerprint_sha256(values)
    profile = RasterTextOfflineProfileV12.model_validate(
        {
            **values,
            "profile_fingerprint_sha256": fingerprint,
            "raster_text_profile_revision": f"synthetic-raster-text-v1:{fingerprint[:24]}",
        }
    )
    configuration_raw = _canonical(profile.model_dump(mode="json", warnings=False))
    paths = {
        profile.parent_renderer_package_inventory_path: tmp_path / "parent.json",
        profile.sidecar_interpreter_root: tmp_path / "interpreter-root",
        profile.sidecar_interpreter_path: tmp_path / "python.exe",
        profile.sidecar_module_root: tmp_path / "module-root",
        profile.sidecar_module_path: tmp_path / "sidecar.py",
        profile.sidecar_package_inventory_path: tmp_path / "packages.json",
        profile.local_model_root: tmp_path / "models",
        profile.model_inventory_path: tmp_path / "models.json",
    }
    verified: list[RasterTextOfflineProfileV12] = []

    monkeypatch.setattr(
        raster_text_runtime,
        "_verify_v12_trained_model_bundle",
        lambda **kwargs: verified.append(kwargs["profile"]),
    )
    monkeypatch.setattr(
        raster_text_runtime,
        "_verify_v3_mobile_model_bundle",
        lambda **_kwargs: pytest.fail("V12 must not enter the V3--V11 verifier"),
    )
    monkeypatch.setattr(
        raster_text_runtime,
        "_profile_sidecar_path",
        lambda **kwargs: paths[kwargs["value"]],
    )
    monkeypatch.setattr(
        raster_text_runtime,
        "_read_local_file",
        lambda path, **_kwargs: (
            inventory_raw if path == paths[profile.model_inventory_path] else b"attested-material"
        ),
    )
    monkeypatch.setattr(raster_text_runtime, "_verify_package_inventory", lambda **_kwargs: None)
    monkeypatch.setattr(
        raster_text_runtime,
        "_verify_model_inventory",
        lambda **_kwargs: pytest.fail("combined loader already verified the V12 model tree"),
    )
    monkeypatch.setattr(
        raster_text_runtime,
        "_validate_profile_controlled_child_file",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        raster_text_runtime,
        "_sha256_regular_file",
        lambda path: (
            profile.sidecar_interpreter_sha256
            if path == paths[profile.sidecar_interpreter_path]
            else profile.sidecar_module_sha256
        ),
    )
    monkeypatch.setattr(
        raster_text_runtime,
        "_controlled_directory_tree_sha256",
        lambda _path: profile.sidecar_interpreter_root_tree_sha256,
    )
    monkeypatch.setattr(
        raster_text_runtime,
        "_controlled_module_root_tree_sha256",
        lambda _path: profile.sidecar_module_root_tree_sha256,
    )
    monkeypatch.setattr(
        raster_text_runtime,
        "_v2_bundle_sidecar_path",
        lambda **kwargs: paths[profile.local_model_root] / PurePosixPath(kwargs["value"]).name,
    )
    for path in (
        paths[profile.sidecar_interpreter_root],
        paths[profile.sidecar_module_root],
        paths[profile.local_model_root] / "text-det",
        paths[profile.local_model_root] / "text-rec",
    ):
        path.mkdir(parents=True, exist_ok=True)

    loaded = raster_text_runtime._load_profile_materials(  # noqa: SLF001
        root=tmp_path,
        configuration_path=tmp_path / "profile.json",
        configuration_raw=configuration_raw,
        configuration_sha256=hashlib.sha256(configuration_raw).hexdigest(),
    )

    assert loaded[0] == profile
    assert verified == [profile]


def _ready(*, provider_revision: str = "direct-text-test-v1") -> bytes:
    return json.dumps(
        {
            "protocol": "customs-raster-text-sidecar-v1-service",
            "status": "ready",
            "raster_text_profile_revision": _REVISION,
            "provider_revision": provider_revision,
        },
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _evaluation_ready(*, provider_revision: str = "direct-text-test-v1") -> bytes:
    return json.dumps(
        {
            "protocol": "customs-raster-text-sidecar-v1-session",
            "status": "ready",
            "raster_text_profile_revision": _REVISION,
            "provider_revision": provider_revision,
        },
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _result(*, provider_revision: str = "direct-text-test-v1") -> bytes:
    return (
        RasterTextSidecarResultV1(
            source_sha256=_SOURCE,
            raster_text_profile_revision=_REVISION,
            provider_revision=provider_revision,
            pages=[RasterTextPageV1(page_number=1, text="READY", confidence=0.9)],
        )
        .model_dump_json()
        .encode("utf-8")
    )


def _page() -> RasterizedPageV2:
    return RasterizedPageV2(page_number=1, width=1, height=1, pixels=b"\x00\x00\x00")


class _CountingInput(io.BytesIO):
    def __init__(self) -> None:
        super().__init__()
        self.writes = 0

    def write(self, value: bytes) -> int:
        self.writes += 1
        return super().write(value)


class _FakeProcess:
    def __init__(self, stdout: io.BytesIO) -> None:
        self.stdin = _CountingInput()
        self.stdout = stdout
        self._alive = True
        self.terminated = 0
        self.killed = 0
        self.waited = 0

    def poll(self) -> int | None:
        return None if self._alive else 0

    def terminate(self) -> None:
        self.terminated += 1
        self._alive = False

    def kill(self) -> None:
        self.killed += 1
        self._alive = False

    def wait(self, *, timeout: int) -> int:
        self.waited += 1
        return 0


def _session(
    process: _FakeProcess, cleaned: list[str], *, timeout: int = 1
) -> PaddleOcrTextServiceSession:
    return PaddleOcrTextServiceSession(
        provider=_provider(),
        process=cast(object, process),
        cleanup_home=lambda: cleaned.append("cleaned"),
        request_timeout_seconds=timeout,
        startup_timeout_seconds=timeout,
    )


def test_service_session_verifies_ready_reuses_process_and_reaps_on_close() -> None:
    process = _FakeProcess(io.BytesIO(_frame(_ready()) + _frame(_result()) + _frame(_result())))
    cleaned: list[str] = []
    session = _session(process, cleaned)

    session.verify_ready()
    assert (
        session.extract_text(
            (_page(),), source_sha256=_SOURCE, raster_text_profile_revision=_REVISION
        ).source_sha256
        == _SOURCE
    )
    assert (
        session.extract_text(
            (_page(),), source_sha256=_SOURCE, raster_text_profile_revision=_REVISION
        ).source_sha256
        == _SOURCE
    )
    # Each serial request emits exactly one length header and one payload to
    # the same process; no fresh process/model initialization occurs.
    assert process.stdin.writes == 4

    session.close()

    assert process.terminated == 1
    assert process.waited == 1
    assert cleaned == ["cleaned"]


def test_service_session_serializes_concurrent_requests() -> None:
    class _GateStream(io.BytesIO):
        def __init__(self, payload: bytes, *, gate_at: int) -> None:
            super().__init__(payload)
            self._gate_at = gate_at
            self.entered = threading.Event()
            self.release = threading.Event()

        def read(self, size: int = -1) -> bytes:
            if self.tell() >= self._gate_at and not self.release.is_set():
                self.entered.set()
                self.release.wait(timeout=3)
            return super().read(size)

        def close(self) -> None:
            self.release.set()
            super().close()

    ready_frame = _frame(_ready())
    stream = _GateStream(
        ready_frame + _frame(_result()) + _frame(_result()), gate_at=len(ready_frame)
    )
    process = _FakeProcess(stream)
    session = _session(process, [])
    session.verify_ready()

    failures: list[BaseException] = []

    def _run() -> None:
        try:
            session.extract_text(
                (_page(),), source_sha256=_SOURCE, raster_text_profile_revision=_REVISION
            )
        except BaseException as error:
            failures.append(error)

    first = threading.Thread(target=_run)
    second = threading.Thread(target=_run)
    first.start()
    assert stream.entered.wait(timeout=1)
    second.start()
    # The second thread cannot write until the first native request releases
    # the one service lane, so its bytes cannot interleave with the first.
    assert process.stdin.writes == 2
    stream.release.set()
    first.join(timeout=2)
    second.join(timeout=2)

    assert not first.is_alive()
    assert not second.is_alive()
    assert not failures
    assert process.stdin.writes == 4
    session.close()


def test_service_session_nonblocking_liveness_distinguishes_busy_from_dead() -> None:
    process = _FakeProcess(io.BytesIO(_frame(_ready())))
    session = _session(process, [])
    session.verify_ready()
    entered = threading.Event()
    release = threading.Event()

    def hold_request_lane() -> None:
        with session._lock:  # noqa: SLF001 - exact native-lane regression
            entered.set()
            assert release.wait(timeout=2)

    holder = threading.Thread(target=hold_request_lane)
    holder.start()
    assert entered.wait(timeout=1)
    started = time.monotonic()
    session.verify_runtime_state_nonblocking()
    assert time.monotonic() - started < 0.1
    release.set()
    holder.join(timeout=1)
    assert not holder.is_alive()

    process._alive = False  # noqa: SLF001 - simulate an idle child exit
    with pytest.raises(RasterTextRuntimeError, match="raster_text_service_unavailable"):
        session.verify_runtime_state_nonblocking()


def test_service_session_timeout_reaps_child_and_hides_native_failure() -> None:
    class _BlockingStream(io.BytesIO):
        def __init__(self, payload: bytes, *, block_at: int) -> None:
            super().__init__(payload)
            self._block_at = block_at
            self.entered = threading.Event()
            self.release = threading.Event()

        def read(self, size: int = -1) -> bytes:
            if self.tell() >= self._block_at and not self.release.is_set():
                self.entered.set()
                self.release.wait(timeout=10)
                return b""
            return super().read(size)

        def close(self) -> None:
            self.release.set()
            super().close()

    ready_frame = _frame(_ready())
    stream = _BlockingStream(ready_frame, block_at=len(ready_frame))
    process = _FakeProcess(stream)
    cleanup_started = threading.Event()
    cleanup_release = threading.Event()
    cleanup_finished = threading.Event()

    def cleanup_home() -> None:
        cleanup_started.set()
        cleanup_release.wait(timeout=2)
        cleanup_finished.set()

    session = PaddleOcrTextServiceSession(
        provider=_provider(),
        process=cast(object, process),
        cleanup_home=cleanup_home,
        request_timeout_seconds=1,
        startup_timeout_seconds=1,
    )
    session.verify_ready()

    try:
        started = time.monotonic()
        with pytest.raises(
            RasterTextPreviewUnavailable, match="raster text service is unavailable"
        ):
            session.extract_text(
                (_page(),), source_sha256=_SOURCE, raster_text_profile_revision=_REVISION
            )
        elapsed = time.monotonic() - started

        assert stream.entered.is_set()
        assert process.terminated == 1
        assert process.waited == 1
        # Cleanup is scheduled after the deadline-owned retirement, not run
        # synchronously by the error-path abort.  The request can return even
        # while an intentionally blocked private-home cleanup remains pending.
        assert cleanup_started.wait(timeout=0.2)
        assert not cleanup_finished.is_set()
        session.abort()
        assert not cleanup_finished.is_set()
        assert elapsed < 1.5
    finally:
        cleanup_release.set()
    assert cleanup_finished.wait(timeout=0.5)


def test_service_request_deadline_includes_a_blocked_pipe_write() -> None:
    class _BlockingWriteInput(_CountingInput):
        def __init__(self) -> None:
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()
            self.finished = threading.Event()

        def write(self, value: bytes) -> int:
            self.entered.set()
            self.release.wait(timeout=10)
            self.finished.set()
            return super().write(value)

        def close(self) -> None:
            self.release.set()
            super().close()

    process = _FakeProcess(io.BytesIO(_frame(_ready())))
    blocked_stdin = _BlockingWriteInput()
    process.stdin = blocked_stdin
    session = _session(process, [], timeout=1)
    session.verify_ready()

    started = time.monotonic()
    with pytest.raises(RasterTextPreviewUnavailable, match="raster text service is unavailable"):
        session.extract_text(
            (_page(),), source_sha256=_SOURCE, raster_text_profile_revision=_REVISION
        )
    elapsed = time.monotonic() - started

    assert blocked_stdin.entered.is_set()
    assert blocked_stdin.finished.wait(timeout=0.1)
    assert process.terminated == 1
    # The request exchanges and cleanup share the one-second test budget;
    # this would be about three seconds with a separate write + reap timeout.
    assert elapsed < 1.5


def test_deadline_cleanup_thread_start_and_late_atexit_failure_keep_module_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    process = _FakeProcess(io.BytesIO())
    cleaned: list[str] = []
    session = _session(process, cleaned)

    # The one module-level exit handler was installed during import.  A later
    # per-session registration failure must be irrelevant: the deadline path
    # has no per-session bound atexit callback to add.
    assert raster_text_runtime._DEFERRED_HOME_CLEANUP_HANDLER_REGISTERED  # noqa: SLF001
    monkeypatch.setattr(
        raster_text_runtime.atexit,
        "register",
        lambda _callback: (_ for _ in ()).throw(RuntimeError("private")),
    )

    def fail_thread_start(_thread: threading.Thread) -> None:
        raise RuntimeError("private")

    monkeypatch.setattr(
        raster_text_runtime.threading.Thread,
        "start",
        fail_thread_start,
    )

    session._discard(deadline=time.monotonic() + 1)  # noqa: SLF001 - lifecycle regression
    token = session._home_cleanup_token  # noqa: SLF001 - lifecycle regression
    assert token is not None
    with raster_text_runtime._DEFERRED_HOME_CLEANUP_REGISTRY_LOCK:  # noqa: SLF001
        assert (
            raster_text_runtime._DEFERRED_HOME_CLEANUP_REGISTRY[token]  # noqa: SLF001
            is session._home_cleanup_state  # noqa: SLF001 - lifecycle regression
        )
    # The public error path invokes abort/close after the deadline-owned
    # discard.  A daemon-start failure must preserve deferred ownership rather
    # than deleting the isolated home synchronously here.
    session.abort()
    session.close()

    assert cleaned == []
    # Simulate the already-registered process-exit handler.  It owns only the
    # detached cleanup state, so this works despite the late atexit failure.
    raster_text_runtime._run_registered_deferred_home_cleanups()  # noqa: SLF001
    assert cleaned == ["cleaned"]
    with raster_text_runtime._DEFERRED_HOME_CLEANUP_REGISTRY_LOCK:  # noqa: SLF001
        assert token not in raster_text_runtime._DEFERRED_HOME_CLEANUP_REGISTRY  # noqa: SLF001


def test_deferred_cleanup_retries_a_transient_failure_without_marking_clean() -> None:
    process = _FakeProcess(io.BytesIO())
    attempts: list[str] = []
    cleaned = threading.Event()

    def cleanup_home() -> None:
        if not attempts:
            attempts.append("failed")
            raise OSError("private")
        attempts.append("cleaned")
        cleaned.set()

    session = PaddleOcrTextServiceSession(
        provider=_provider(),
        process=cast(object, process),
        cleanup_home=cleanup_home,
        request_timeout_seconds=1,
        startup_timeout_seconds=1,
    )
    session._discard(deadline=time.monotonic() + 1)  # noqa: SLF001 - lifecycle regression

    assert cleaned.wait(timeout=1)
    assert attempts == ["failed", "cleaned"]
    # A successful worker removes its exact detached registry entry.  No
    # bound session callback remains retained by atexit after completion.
    assert session._home_cleanup_token is None  # noqa: SLF001 - lifecycle regression
    with raster_text_runtime._DEFERRED_HOME_CLEANUP_REGISTRY_LOCK:  # noqa: SLF001
        assert (
            session._home_cleanup_state  # noqa: SLF001 - lifecycle regression
            not in raster_text_runtime._DEFERRED_HOME_CLEANUP_REGISTRY.values()  # noqa: SLF001
        )
    session.abort()
    assert attempts == ["failed", "cleaned"]


def test_deferred_cleanup_exhaustion_keeps_module_exit_fallback() -> None:
    process = _FakeProcess(io.BytesIO())
    attempts: list[str] = []
    worker_exhausted = threading.Event()

    def cleanup_home() -> None:
        attempts.append("attempt")
        if len(attempts) <= 3:
            if len(attempts) == 3:
                worker_exhausted.set()
            raise OSError("private")

    session = PaddleOcrTextServiceSession(
        provider=_provider(),
        process=cast(object, process),
        cleanup_home=cleanup_home,
        request_timeout_seconds=1,
        startup_timeout_seconds=1,
    )
    session._discard(deadline=time.monotonic() + 1)  # noqa: SLF001 - lifecycle regression
    token = session._home_cleanup_token  # noqa: SLF001 - lifecycle regression
    assert token is not None

    assert worker_exhausted.wait(timeout=1)
    with raster_text_runtime._DEFERRED_HOME_CLEANUP_REGISTRY_LOCK:  # noqa: SLF001
        assert (
            raster_text_runtime._DEFERRED_HOME_CLEANUP_REGISTRY[token]  # noqa: SLF001
            is session._home_cleanup_state  # noqa: SLF001 - lifecycle regression
        )

    # The fourth attempt belongs to the module exit fallback, not the bounded
    # worker.  It proves exhausted retries leave recoverable detached state.
    raster_text_runtime._run_registered_deferred_home_cleanups()  # noqa: SLF001
    assert attempts == ["attempt", "attempt", "attempt", "attempt"]
    with raster_text_runtime._DEFERRED_HOME_CLEANUP_REGISTRY_LOCK:  # noqa: SLF001
        assert token not in raster_text_runtime._DEFERRED_HOME_CLEANUP_REGISTRY  # noqa: SLF001


def test_service_session_child_exit_is_unavailable_and_retires_binding() -> None:
    process = _FakeProcess(io.BytesIO(_frame(_ready())))
    cleaned: list[str] = []
    session = _session(process, cleaned)
    session.verify_ready()
    process._alive = False

    with pytest.raises(RasterTextPreviewUnavailable, match="raster text service is unavailable"):
        session.extract_text(
            (_page(),), source_sha256=_SOURCE, raster_text_profile_revision=_REVISION
        )

    assert cleaned == ["cleaned"]


def test_provider_starts_service_protocol_after_full_parent_verify(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider = _provider()
    process = _FakeProcess(io.BytesIO(_frame(_ready())))
    verified: list[str] = []
    cleaned: list[str] = []
    launch: list[tuple[list[str], dict[str, object]]] = []

    def _verify_parent(_self: PaddleOcrTextCommandProvider) -> None:
        verified.append("full")

    def _popen(command: list[str], **kwargs: object) -> _FakeProcess:
        launch.append((command, kwargs))
        return process

    monkeypatch.setattr(PaddleOcrTextCommandProvider, "_verify_parent_materials", _verify_parent)
    monkeypatch.setattr(
        raster_text_runtime,
        "_new_isolated_v2_sidecar_home",
        lambda: (tmp_path, lambda: cleaned.append("cleaned")),
    )
    monkeypatch.setattr(raster_text_runtime, "_sidecar_environment", lambda **_kwargs: {})
    monkeypatch.setattr(raster_text_runtime.subprocess, "Popen", _popen)

    session = provider.start_service_session()

    assert verified == ["full"]
    assert launch[0][0][-1] == "--customs-raster-text-sidecar-v1-service"
    assert launch[0][1]["bufsize"] == 0
    # A cold native load gets a setup budget independent from the 30-second
    # post-ready request bound in the profile.
    assert session.startup_timeout_seconds == 55
    session.close()
    assert cleaned == ["cleaned"]


def test_v2_one_shot_launches_direct_short_spelling_bootstrap(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The non-persistent direct V2 path cannot fall back to V1 bootstrap."""

    profile = _v2_profile()
    provider = _provider(profile)
    launches: list[list[str]] = []

    class _Home:
        def __enter__(self) -> Path:
            return tmp_path

        def __exit__(self, *_args: object) -> None:
            return None

    def run(command: list[str], **_kwargs: object) -> SimpleNamespace:
        launches.append(command)
        return SimpleNamespace(
            returncode=0,
            stdout=_result(provider_revision=profile.extractor_revision),
        )

    monkeypatch.setattr(
        PaddleOcrTextCommandProvider, "_verify_parent_materials", lambda _self: None
    )
    monkeypatch.setattr(raster_text_runtime, "_isolated_v2_sidecar_home", _Home)
    monkeypatch.setattr(raster_text_runtime, "_sidecar_environment", lambda **_kwargs: {})
    monkeypatch.setattr(raster_text_runtime.subprocess, "run", run)

    assert (
        provider.extract_text(
            (_page(),), source_sha256=_SOURCE, raster_text_profile_revision=_REVISION
        ).source_sha256
        == _SOURCE
    )
    assert launches[0][4] == raster_text_runtime._DIRECT_PADDLE_V2_ISOLATED_SIDECAR_BOOTSTRAP  # noqa: SLF001


def test_direct_v2_bootstrap_keeps_command_on_its_operational_spelling() -> None:
    """Freeze the Windows-safe child boundary without weakening its checks."""

    bootstrap = raster_text_runtime._DIRECT_PADDLE_V2_ISOLATED_SIDECAR_BOOTSTRAP  # noqa: SLF001

    # It must remain executable as a ``python -c`` payload, and the deep
    # command leaf must never enter a long-path ``realpath`` comparison.
    compile(bootstrap, "<direct-paddle-v2-bootstrap>", "exec")
    assert "command_path=os.path.realpath" not in bootstrap
    assert "no_links(command_argument)" in bootstrap
    assert "lexical_inside(module_argument,command_argument)" in bootstrap
    # The standard-library boundary remains independently canonicalized.
    assert "os.path.realpath(sys.executable)" in bootstrap
    assert "os.path.realpath(path)" in bootstrap


@pytest.mark.parametrize(
    ("profile_factory", "expected_startup_timeout"),
    [(_profile, 55), (_v2_profile, 60)],
)
def test_provider_selects_profile_bound_cold_startup_without_relaxing_request_bound(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    profile_factory: Callable[[], RasterTextOfflineProfileV1 | RasterTextOfflineProfileV2],
    expected_startup_timeout: int,
) -> None:
    """Only V2 receives the 60-second ready budget; both remain at 30/request."""

    profile = profile_factory()
    provider = _provider(profile)
    service_process = _FakeProcess(
        io.BytesIO(_frame(_ready(provider_revision=profile.extractor_revision)))
    )
    evaluation_process = _FakeProcess(
        io.BytesIO(_frame(_evaluation_ready(provider_revision=profile.extractor_revision)))
    )
    processes = [service_process, evaluation_process]
    launches: list[list[str]] = []

    class _Home:
        def __enter__(self) -> Path:
            return tmp_path

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(
        PaddleOcrTextCommandProvider, "_verify_parent_materials", lambda _self: None
    )
    monkeypatch.setattr(
        PaddleOcrTextCommandProvider,
        "_verify_parent_binding_materials",
        lambda _self: None,
    )
    monkeypatch.setattr(
        raster_text_runtime,
        "_new_isolated_v2_sidecar_home",
        lambda: (tmp_path, lambda: None),
    )
    monkeypatch.setattr(raster_text_runtime, "_isolated_v2_sidecar_home", _Home)
    monkeypatch.setattr(raster_text_runtime, "_sidecar_environment", lambda **_kwargs: {})

    def popen(command: list[str], **_kwargs: object) -> _FakeProcess:
        launches.append(command)
        return processes.pop(0)

    monkeypatch.setattr(raster_text_runtime.subprocess, "Popen", popen)

    service = provider.start_service_session()
    assert service.startup_timeout_seconds == expected_startup_timeout
    assert service.request_timeout_seconds == 30
    service.close()

    with provider.open_evaluation_session() as evaluation:
        assert evaluation.startup_timeout_seconds == expected_startup_timeout
        assert evaluation.request_timeout_seconds == 30

    expected_bootstrap = (
        raster_text_runtime._DIRECT_PADDLE_V2_ISOLATED_SIDECAR_BOOTSTRAP  # noqa: SLF001
        if isinstance(profile, RasterTextOfflineProfileV2)
        else raster_text_runtime._ISOLATED_SIDECAR_BOOTSTRAP  # noqa: SLF001
    )
    assert [command[4] for command in launches] == [expected_bootstrap, expected_bootstrap]


@pytest.mark.parametrize(
    ("profile_factory", "expected_startup_timeout"),
    [(_profile, 55), (_v2_profile, 60)],
)
def test_evaluation_session_default_startup_bound_follows_provider_profile(
    profile_factory: Callable[[], RasterTextOfflineProfileV1 | RasterTextOfflineProfileV2],
    expected_startup_timeout: int,
) -> None:
    """The generic session class accepts 60, but derives its default safely."""

    profile = profile_factory()
    session = PaddleOcrTextEvaluationSession(
        provider=_provider(profile),
        process=cast(subprocess.Popen[bytes], _FakeProcess(io.BytesIO())),
    )

    assert session.startup_timeout_seconds == expected_startup_timeout
    assert session.request_timeout_seconds == 30
    session.close()

    with pytest.raises(RasterTextRuntimeError, match="raster_text_evaluation_session_invalid"):
        PaddleOcrTextEvaluationSession(
            provider=_provider(_v2_profile()),
            process=cast(subprocess.Popen[bytes], _FakeProcess(io.BytesIO())),
            request_timeout_seconds=31,
            startup_timeout_seconds=60,
        )


def test_service_session_rejects_wrong_ready_binding_and_reaps() -> None:
    wrong_ready = json.dumps(
        {
            "protocol": "customs-raster-text-sidecar-v1-service",
            "status": "ready",
            "raster_text_profile_revision": _REVISION,
            "provider_revision": "different-provider",
        },
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    process = _FakeProcess(io.BytesIO(_frame(wrong_ready)))
    cleaned: list[str] = []
    session = _session(process, cleaned)

    with pytest.raises(RasterTextRuntimeError, match="raster_text_service_output_invalid"):
        session.verify_ready()

    assert process.terminated == 1
    assert cleaned == ["cleaned"]


def test_evaluation_session_consumes_ready_before_warm_requests_and_skips_full_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider()
    process = _FakeProcess(
        io.BytesIO(_frame(_evaluation_ready()) + _frame(_result()) + _frame(_result()))
    )
    full_checks: list[str] = []
    binding_checks: list[str] = []

    def full_tree(_self: PaddleOcrTextCommandProvider) -> None:
        full_checks.append("full")

    def binding_only(_self: PaddleOcrTextCommandProvider) -> None:
        binding_checks.append("cheap")

    monkeypatch.setattr(PaddleOcrTextCommandProvider, "_verify_parent_materials", full_tree)
    monkeypatch.setattr(
        PaddleOcrTextCommandProvider,
        "_verify_parent_binding_materials",
        binding_only,
    )
    session = PaddleOcrTextEvaluationSession(
        provider=provider,
        process=cast(object, process),
        request_timeout_seconds=30,
        startup_timeout_seconds=55,
    )

    # The first readable frame is content-free readiness, so model import/load
    # is outside the first candidate's 30-second request budget.
    session.verify_ready()
    assert (
        session.extract_text(
            (_page(),), source_sha256=_SOURCE, raster_text_profile_revision=_REVISION
        ).source_sha256
        == _SOURCE
    )
    assert (
        session.extract_text(
            (_page(),), source_sha256=_SOURCE, raster_text_profile_revision=_REVISION
        ).source_sha256
        == _SOURCE
    )

    assert session.startup_timeout_seconds == 55
    assert session.request_timeout_seconds == 30
    assert full_checks == []
    assert binding_checks == ["cheap", "cheap"]
    assert process.stdin.writes == 4
    session.close()


def test_provider_opens_evaluation_session_after_one_full_verify_and_ready(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider = _provider()
    process = _FakeProcess(io.BytesIO(_frame(_evaluation_ready()) + _frame(_result())))
    full_checks: list[str] = []
    binding_checks: list[str] = []
    launch: list[tuple[list[str], dict[str, object]]] = []

    def full_tree(_self: PaddleOcrTextCommandProvider) -> None:
        full_checks.append("full")

    def binding_only(_self: PaddleOcrTextCommandProvider) -> None:
        binding_checks.append("cheap")

    class _Home:
        def __enter__(self) -> Path:
            return tmp_path

        def __exit__(self, *_args: object) -> None:
            return None

    def popen(command: list[str], **kwargs: object) -> _FakeProcess:
        launch.append((command, kwargs))
        return process

    monkeypatch.setattr(PaddleOcrTextCommandProvider, "_verify_parent_materials", full_tree)
    monkeypatch.setattr(
        PaddleOcrTextCommandProvider,
        "_verify_parent_binding_materials",
        binding_only,
    )
    monkeypatch.setattr(raster_text_runtime, "_isolated_v2_sidecar_home", _Home)
    monkeypatch.setattr(raster_text_runtime, "_sidecar_environment", lambda **_kwargs: {})
    monkeypatch.setattr(raster_text_runtime.subprocess, "Popen", popen)

    with provider.open_evaluation_session() as session:
        assert session.startup_timeout_seconds == 55
        assert session.request_timeout_seconds == 30
        assert (
            session.extract_text(
                (_page(),), source_sha256=_SOURCE, raster_text_profile_revision=_REVISION
            ).source_sha256
            == _SOURCE
        )

    assert full_checks == ["full"]
    assert binding_checks == ["cheap"]
    assert launch[0][0][-1] == "--customs-raster-text-sidecar-v1-session"
    assert launch[0][1]["bufsize"] == 0
    assert process.stdin.writes == 2


def test_provider_reuses_just_verified_parent_materials_for_evaluation_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    provider = _provider()
    process = _FakeProcess(io.BytesIO(_frame(_evaluation_ready()) + _frame(_result())))
    full_checks: list[str] = []
    binding_checks: list[str] = []

    def full_tree(_self: PaddleOcrTextCommandProvider) -> None:
        full_checks.append("full")

    def binding_only(_self: PaddleOcrTextCommandProvider) -> None:
        binding_checks.append("cheap")

    class _Home:
        def __enter__(self) -> Path:
            return tmp_path

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(PaddleOcrTextCommandProvider, "_verify_parent_materials", full_tree)
    monkeypatch.setattr(
        PaddleOcrTextCommandProvider,
        "_verify_parent_binding_materials",
        binding_only,
    )
    monkeypatch.setattr(raster_text_runtime, "_isolated_v2_sidecar_home", _Home)
    monkeypatch.setattr(raster_text_runtime, "_sidecar_environment", lambda **_kwargs: {})
    monkeypatch.setattr(raster_text_runtime.subprocess, "Popen", lambda *_args, **_kwargs: process)

    with provider.open_evaluation_session(parent_materials_verified=True) as session:
        assert (
            session.extract_text(
                (_page(),), source_sha256=_SOURCE, raster_text_profile_revision=_REVISION
            ).source_sha256
            == _SOURCE
        )

    # The caller's immediately preceding full verification is reused.  One
    # cheap check happens at launch and one before the evaluated document.
    assert full_checks == []
    assert binding_checks == ["cheap", "cheap"]


def test_evaluation_session_fails_closed_on_parent_binding_drift(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    provider = _provider()
    process = _FakeProcess(io.BytesIO(_frame(_evaluation_ready())))

    def changed_binding(_self: PaddleOcrTextCommandProvider) -> None:
        raise RasterTextRuntimeError("raster_text_sidecar_drift")

    monkeypatch.setattr(
        PaddleOcrTextCommandProvider,
        "_verify_parent_binding_materials",
        changed_binding,
    )
    session = PaddleOcrTextEvaluationSession(
        provider=provider,
        process=cast(object, process),
        request_timeout_seconds=30,
        startup_timeout_seconds=55,
    )
    session.verify_ready()

    with pytest.raises(
        RasterTextPreviewUnavailable, match="raster text evaluation sidecar is unavailable"
    ):
        session.extract_text(
            (_page(),), source_sha256=_SOURCE, raster_text_profile_revision=_REVISION
        )

    # No document frame is accepted after local binding drift.  The child is
    # retired, so a later candidate cannot reuse an invalid warm engine.
    assert process.stdin.writes == 0
    assert process.waited == 1


def test_evaluation_timeout_closes_stdout_and_joins_blocked_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BlockingReadStream(io.BytesIO):
        def __init__(self, payload: bytes) -> None:
            super().__init__(payload)
            self.entered = threading.Event()
            self.release = threading.Event()
            self.finished = threading.Event()

        def read(self, size: int = -1) -> bytes:
            if self.tell() >= len(_frame(_evaluation_ready())) and not self.release.is_set():
                self.entered.set()
                self.release.wait(timeout=10)
                self.finished.set()
                return b""
            return super().read(size)

        def close(self) -> None:
            self.release.set()
            super().close()

    provider = _provider()
    stream = _BlockingReadStream(_frame(_evaluation_ready()))
    process = _FakeProcess(stream)
    monkeypatch.setattr(
        PaddleOcrTextCommandProvider,
        "_verify_parent_binding_materials",
        lambda _self: None,
    )
    session = PaddleOcrTextEvaluationSession(
        provider=provider,
        process=cast(object, process),
        request_timeout_seconds=1,
        startup_timeout_seconds=1,
    )
    session.verify_ready()

    started = time.monotonic()
    with pytest.raises(
        RasterTextPreviewUnavailable, match="raster text evaluation sidecar is unavailable"
    ):
        session.extract_text(
            (_page(),), source_sha256=_SOURCE, raster_text_profile_revision=_REVISION
        )
    elapsed = time.monotonic() - started

    assert stream.entered.is_set()
    assert stream.finished.wait(timeout=0.1)
    assert stream.closed
    assert process.terminated == 1
    assert elapsed < 1.5


def test_evaluation_request_deadline_includes_a_blocked_pipe_write(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BlockingWriteInput(_CountingInput):
        def __init__(self) -> None:
            super().__init__()
            self.entered = threading.Event()
            self.release = threading.Event()
            self.finished = threading.Event()

        def write(self, value: bytes) -> int:
            self.entered.set()
            self.release.wait(timeout=10)
            self.finished.set()
            return super().write(value)

        def close(self) -> None:
            self.release.set()
            super().close()

    provider = _provider()
    process = _FakeProcess(io.BytesIO(_frame(_evaluation_ready())))
    blocked_stdin = _BlockingWriteInput()
    process.stdin = blocked_stdin
    monkeypatch.setattr(
        PaddleOcrTextCommandProvider,
        "_verify_parent_binding_materials",
        lambda _self: None,
    )
    session = PaddleOcrTextEvaluationSession(
        provider=provider,
        process=cast(object, process),
        request_timeout_seconds=1,
        startup_timeout_seconds=1,
    )
    session.verify_ready()

    started = time.monotonic()
    with pytest.raises(
        RasterTextPreviewUnavailable, match="raster text evaluation sidecar is unavailable"
    ):
        session.extract_text(
            (_page(),), source_sha256=_SOURCE, raster_text_profile_revision=_REVISION
        )
    elapsed = time.monotonic() - started

    assert blocked_stdin.entered.is_set()
    assert blocked_stdin.finished.wait(timeout=0.1)
    assert process.terminated == 1
    assert elapsed < 1.5
