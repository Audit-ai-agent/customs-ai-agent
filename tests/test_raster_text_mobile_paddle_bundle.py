"""Focused tests for the isolated PP-OCRv5 mobile-detector candidate."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast

import pytest

import customs_ai_agent.evaluation.raster_text_mobile_paddle_bundle as mobile_bundle
import customs_ai_agent.service.raster_text_preview_runtime as raster_text_runtime
from customs_ai_agent.evaluation.raster_text_direct_paddle_bundle import (
    VerifiedRasterTextDirectPaddleBundleV1,
)
from customs_ai_agent.service.raster_text_preview_runtime import (
    RasterTextOfflineProfileV3,
    RasterTextOfflineProfileV4,
    RasterTextOfflineProfileV5,
    RasterTextOfflineProfileV6,
    RasterTextOfflineProfileV7,
    RasterTextOfflineProfileV8,
    RasterTextOfflineProfileV9,
    RasterTextOfflineProfileV10,
    RasterTextOfflineProfileV11,
    RasterTextOfflineProfileV13,
    RasterTextOfflineProfileV14,
    RasterTextOfflineProfileV15,
    RasterTextOfflineProfileV16,
    RasterTextOfflineProfileV17,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def _make_mobile_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    detector_name: str = "PP-OCRv5_mobile_det",
) -> Path:
    """Create a tiny hash-bound local inventory; no model is executed."""

    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    for index, (relative, _target) in enumerate(mobile_bundle._MOBILE_MODEL_ARTIFACTS, start=1):
        if relative.endswith("det/config.json"):
            raw = _canonical({"Global": {"model_name": detector_name}})
        elif relative.endswith("rec/config.json"):
            raw = _canonical({"Global": {"model_name": "en_PP-OCRv5_mobile_rec"}})
        else:
            raw = f"mobile-model-{index}".encode("ascii")
        path = root.joinpath(*relative.split("/"))
        _write(path, raw)
        rows.append(
            {
                "artifact_id": f"artifact-{index}",
                "byte_size": len(raw),
                "license_id": "Apache-2.0",
                "relative_path": relative,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "source_uri": f"https://example.invalid/{index}",
            }
        )
    inventory = {
        "schema_version": "1.0",
        "engine_kind": "paddleocr",
        "runtime_api": "paddleocr3-paddlex-3.2",
        "packages": [],
        "artifacts": rows,
    }
    # Match the fixed, pre-existing PaddleOCR3 source inventory format.  Its
    # source-attestation SHA deliberately includes one trailing LF.
    inventory_raw = _canonical(inventory) + b"\n"
    inventory_sha256 = hashlib.sha256(inventory_raw).hexdigest()
    _write(root / "model-inventories" / f"{inventory_sha256}.json", inventory_raw)
    monkeypatch.setattr(
        mobile_bundle,
        "_MOBILE_SOURCE_INVENTORY_RELATIVE_PATH",
        f"model-inventories/{inventory_sha256}.json",
    )
    monkeypatch.setattr(mobile_bundle, "_MOBILE_SOURCE_INVENTORY_SHA256", inventory_sha256)
    return root


def _v3_payload() -> dict[str, object]:
    sidecar_sha = "a" * 64
    mobile_sha = "b" * 64
    source_sha = "c" * 64
    sidecar_root = f"raster-text-direct-bundles-v1/releases/{sidecar_sha}"
    mobile_root = f"raster-text-direct-mobile-bundles-v1/releases/{mobile_sha}"
    return {
        "schema_version": "3.0",
        "engine_kind": "direct_paddle_mobile_text",
        "runtime_api": "paddle-inference-static-v1",
        "extractor_revision": "direct-paddle-mobile-test-v3",
        "bundle_root_relative_path": sidecar_root,
        "bundle_manifest_sha256": sidecar_sha,
        "mobile_model_bundle_root_relative_path": mobile_root,
        "mobile_model_bundle_manifest_sha256": mobile_sha,
        "source_model_inventory_relative_path": f"model-inventories/{source_sha}.json",
        "source_model_inventory_sha256": source_sha,
        "pdfium_version": "4.30.0",
        "raster_dpi": 216,
        "sidecar_timeout_seconds": 30,
        "cpu_threads": 6,
        "text_detection_limit_side_len": 960,
        "text_detection_limit_type": "max",
        "text_detection_threshold": 0.3,
        "text_detection_box_threshold": 0.6,
        "text_detection_unclip_ratio": 1.5,
        "minimum_page_confidence": 0.5,
        "parent_renderer_package_inventory_path": f"{sidecar_root}/inventories/parent-renderer.json",
        "parent_renderer_package_inventory_sha256": "d" * 64,
        "parent_renderer_required_packages": ["pypdfium2", "Pillow"],
        "sidecar_interpreter_path": f"{sidecar_root}/sidecar/interpreter/python.exe",
        "sidecar_interpreter_sha256": "e" * 64,
        "sidecar_interpreter_root": f"{sidecar_root}/sidecar/interpreter",
        "sidecar_interpreter_root_tree_sha256": "f" * 64,
        "sidecar_module_path": (
            f"{sidecar_root}/sidecar/module-root/customs_ai_agent/service/"
            "raster_text_direct_paddle_sidecar.py"
        ),
        "sidecar_module_sha256": "0" * 64,
        "sidecar_module_root": f"{sidecar_root}/sidecar/module-root",
        "sidecar_module_root_tree_sha256": "1" * 64,
        "sidecar_package_inventory_path": f"{sidecar_root}/inventories/direct-paddle-packages.json",
        "sidecar_package_inventory_sha256": "2" * 64,
        "sidecar_required_packages": [
            "numpy",
            "opencv-contrib-python",
            "paddlepaddle",
            "pyclipper",
        ],
        "local_model_root": f"{mobile_root}/models",
        "model_inventory_path": f"{mobile_root}/inventories/direct-paddle-mobile-models.json",
        "model_inventory_sha256": "3" * 64,
        "text_detection_model_name": "PP-OCRv5_mobile_det",
        "text_detection_model_directory": "text-det",
        "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
        "text_recognition_model_directory": "text-rec",
    }


def _v4_payload() -> dict[str, object]:
    values = _v3_payload()
    values.update(
        {
            "schema_version": "4.0",
            "engine_kind": "direct_paddle_mobile_text_harmonic",
            "extractor_revision": "direct-paddle-mobile-test-v4-harmonic",
        }
    )
    return values


def _v5_payload() -> dict[str, object]:
    values = _v4_payload()
    values.update(
        {
            "schema_version": "5.0",
            "engine_kind": "direct_paddle_mobile_text_harmonic_recall",
            "extractor_revision": "direct-paddle-mobile-test-v5-harmonic-recall",
            "text_detection_box_threshold": 0.4,
        }
    )
    return values


def _v6_payload() -> dict[str, object]:
    """V6 is V5 with only the immutable detector input scale changed."""

    values = _v5_payload()
    values.update(
        {
            "schema_version": "6.0",
            "engine_kind": "direct_paddle_mobile_text_1024",
            "extractor_revision": "direct-paddle-mobile-test-v6-1024",
            "text_detection_limit_side_len": 1024,
        }
    )
    return values


def _v7_payload() -> dict[str, object]:
    """V7 is V6 with only the immutable detector box threshold lowered."""

    values = _v6_payload()
    values.update(
        {
            "schema_version": "7.0",
            "engine_kind": "direct_paddle_mobile_text_harmonic_recall_1024_lowbox",
            "extractor_revision": "direct-paddle-mobile-test-v7-1024-lowbox",
            "text_detection_box_threshold": 0.3,
        }
    )
    return values


def _v8_payload() -> dict[str, object]:
    """V8 is V5 with only the immutable detector box threshold lowered."""

    values = _v5_payload()
    values.update(
        {
            "schema_version": "8.0",
            "engine_kind": "direct_paddle_mobile_text_harmonic_recall_native960_lowbox",
            "extractor_revision": "direct-paddle-mobile-test-v8-native960-lowbox",
            "text_detection_box_threshold": 0.3,
        }
    )
    return values


def _v9_payload() -> dict[str, object]:
    """V9 is V8 plus only the immutable exact strip2 engine tuple."""

    values = _v8_payload()
    values.update(
        {
            "schema_version": "9.0",
            "engine_kind": "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2",
            "extractor_revision": "direct-paddle-mobile-test-v9-native960-lowbox-strip2",
        }
    )
    return values


def _v10_payload() -> dict[str, object]:
    """V10 is V8 plus only the immutable exact grid2 engine tuple."""

    values = _v8_payload()
    values.update(
        {
            "schema_version": "10.0",
            "engine_kind": "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_grid2",
            "extractor_revision": "direct-paddle-mobile-test-v10-native960-lowbox-grid2",
        }
    )
    return values


def _v11_payload() -> dict[str, object]:
    """V11 is V10 plus only its immutable guarded-grid engine tuple."""

    values = _v10_payload()
    values.update(
        {
            "schema_version": "11.0",
            "engine_kind": (
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_guarded_grid2"
            ),
            "extractor_revision": "direct-paddle-mobile-test-v11-native960-lowbox-guarded-grid2",
        }
    )
    return values


def _v13_payload() -> dict[str, object]:
    """V13 is the public V9 tuple plus fixed semantic repair semantics."""

    values = _v9_payload()
    values.update(
        {
            "schema_version": "13.0",
            "engine_kind": (
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_semantic_repair"
            ),
            "extractor_revision": (
                "direct-paddle-mobile-test-v13-native960-lowbox-strip2-semantic-repair"
            ),
        }
    )
    return values


def _v14_payload() -> dict[str, object]:
    """V14 is V13 plus only the second semantic-repair identity."""

    values = _v13_payload()
    values.update(
        {
            "schema_version": "14.0",
            "engine_kind": (
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_"
                "strip2_semantic_repair_v2"
            ),
            "extractor_revision": (
                "direct-paddle-mobile-test-v14-native960-lowbox-strip2-semantic-repair-v2"
            ),
        }
    )
    return values


def _v15_payload() -> dict[str, object]:
    """V15 preserves V14 OCR semantics and binds native CPU controls."""

    values = _v14_payload()
    values.update(
        {
            "schema_version": "15.0",
            "engine_kind": (
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_"
                "strip2_semantic_repair_v2_onednn"
            ),
            "extractor_revision": (
                "direct-paddle-mobile-test-v15-native960-lowbox-strip2-semantic-repair-v2-onednn"
            ),
            "enable_mkldnn": True,
            "mkldnn_cache_capacity": 10,
            "recognition_batch_size": 8,
            "opencv_threads": 1,
            "warmup_shape_policy": "detector-736x960-recognizer-48x320-v1",
            "onednn_model_policy": "recognizer-only-v1",
        }
    )
    return values


def _validated_v3_profile(values: dict[str, object]) -> RasterTextOfflineProfileV3:
    fingerprint = raster_text_runtime.raster_text_profile_fingerprint_sha256(values)
    return RasterTextOfflineProfileV3.model_validate(
        {
            **values,
            "profile_fingerprint_sha256": fingerprint,
            "raster_text_profile_revision": f"synthetic-raster-text-v1:{fingerprint[:24]}",
        }
    )


def _validated_v4_profile(values: dict[str, object]) -> RasterTextOfflineProfileV4:
    fingerprint = raster_text_runtime.raster_text_profile_fingerprint_sha256(values)
    return RasterTextOfflineProfileV4.model_validate(
        {
            **values,
            "profile_fingerprint_sha256": fingerprint,
            "raster_text_profile_revision": f"synthetic-raster-text-v1:{fingerprint[:24]}",
        }
    )


def _validated_v5_profile(values: dict[str, object]) -> RasterTextOfflineProfileV5:
    fingerprint = raster_text_runtime.raster_text_profile_fingerprint_sha256(values)
    return RasterTextOfflineProfileV5.model_validate(
        {
            **values,
            "profile_fingerprint_sha256": fingerprint,
            "raster_text_profile_revision": f"synthetic-raster-text-v1:{fingerprint[:24]}",
        }
    )


def _validated_v6_profile(values: dict[str, object]) -> RasterTextOfflineProfileV6:
    fingerprint = raster_text_runtime.raster_text_profile_fingerprint_sha256(values)
    return RasterTextOfflineProfileV6.model_validate(
        {
            **values,
            "profile_fingerprint_sha256": fingerprint,
            "raster_text_profile_revision": f"synthetic-raster-text-v1:{fingerprint[:24]}",
        }
    )


def _validated_v7_profile(values: dict[str, object]) -> RasterTextOfflineProfileV7:
    fingerprint = raster_text_runtime.raster_text_profile_fingerprint_sha256(values)
    return RasterTextOfflineProfileV7.model_validate(
        {
            **values,
            "profile_fingerprint_sha256": fingerprint,
            "raster_text_profile_revision": f"synthetic-raster-text-v1:{fingerprint[:24]}",
        }
    )


def _validated_v8_profile(values: dict[str, object]) -> RasterTextOfflineProfileV8:
    fingerprint = raster_text_runtime.raster_text_profile_fingerprint_sha256(values)
    return RasterTextOfflineProfileV8.model_validate(
        {
            **values,
            "profile_fingerprint_sha256": fingerprint,
            "raster_text_profile_revision": f"synthetic-raster-text-v1:{fingerprint[:24]}",
        }
    )


def _validated_v9_profile(values: dict[str, object]) -> RasterTextOfflineProfileV9:
    fingerprint = raster_text_runtime.raster_text_profile_fingerprint_sha256(values)
    return RasterTextOfflineProfileV9.model_validate(
        {
            **values,
            "profile_fingerprint_sha256": fingerprint,
            "raster_text_profile_revision": f"synthetic-raster-text-v1:{fingerprint[:24]}",
        }
    )


def _validated_v10_profile(values: dict[str, object]) -> RasterTextOfflineProfileV10:
    fingerprint = raster_text_runtime.raster_text_profile_fingerprint_sha256(values)
    return RasterTextOfflineProfileV10.model_validate(
        {
            **values,
            "profile_fingerprint_sha256": fingerprint,
            "raster_text_profile_revision": f"synthetic-raster-text-v1:{fingerprint[:24]}",
        }
    )


def _validated_v11_profile(values: dict[str, object]) -> RasterTextOfflineProfileV11:
    fingerprint = raster_text_runtime.raster_text_profile_fingerprint_sha256(values)
    return RasterTextOfflineProfileV11.model_validate(
        {
            **values,
            "profile_fingerprint_sha256": fingerprint,
            "raster_text_profile_revision": f"synthetic-raster-text-v1:{fingerprint[:24]}",
        }
    )


def _validated_v13_profile(values: dict[str, object]) -> RasterTextOfflineProfileV13:
    fingerprint = raster_text_runtime.raster_text_profile_fingerprint_sha256(values)
    return RasterTextOfflineProfileV13.model_validate(
        {
            **values,
            "profile_fingerprint_sha256": fingerprint,
            "raster_text_profile_revision": f"synthetic-raster-text-v1:{fingerprint[:24]}",
        }
    )


def _validated_v14_profile(values: dict[str, object]) -> RasterTextOfflineProfileV14:
    fingerprint = raster_text_runtime.raster_text_profile_fingerprint_sha256(values)
    return RasterTextOfflineProfileV14.model_validate(
        {
            **values,
            "profile_fingerprint_sha256": fingerprint,
            "raster_text_profile_revision": f"synthetic-raster-text-v1:{fingerprint[:24]}",
        }
    )


def _validated_v15_profile(values: dict[str, object]) -> RasterTextOfflineProfileV15:
    fingerprint = raster_text_runtime.raster_text_profile_fingerprint_sha256(values)
    return RasterTextOfflineProfileV15.model_validate(
        {
            **values,
            "profile_fingerprint_sha256": fingerprint,
            "raster_text_profile_revision": f"synthetic-raster-text-v1:{fingerprint[:24]}",
        }
    )


def _v16_payload() -> dict[str, object]:
    values = _v15_payload()
    values.update(
        {
            "schema_version": "16.0",
            "engine_kind": (
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_"
                "strip2_semantic_repair_v2_onednn_cubic_recognition"
            ),
            "extractor_revision": (
                "direct-paddle-mobile-test-v16-native960-lowbox-strip2-"
                "semantic-repair-v2-onednn-cubic-recognition"
            ),
            "recognition_crop_policy": "detector-quad-v1",
            "recognition_resize_interpolation": "cubic-v1",
        }
    )
    return values


def _validated_v16_profile(values: dict[str, object]) -> RasterTextOfflineProfileV16:
    fingerprint = raster_text_runtime.raster_text_profile_fingerprint_sha256(values)
    return RasterTextOfflineProfileV16.model_validate(
        {
            **values,
            "profile_fingerprint_sha256": fingerprint,
            "raster_text_profile_revision": f"synthetic-raster-text-v1:{fingerprint[:24]}",
        }
    )


def _v17_payload() -> dict[str, object]:
    values = _v15_payload()
    values.update(
        {
            "schema_version": "17.0",
            "engine_kind": (
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_"
                "strip2_semantic_repair_v2_onednn_padded_recognition"
            ),
            "extractor_revision": (
                "direct-paddle-mobile-test-v17-native960-lowbox-strip2-"
                "semantic-repair-v2-onednn-padded-recognition"
            ),
            "recognition_crop_policy": "horizontal-height-0p08-max4-v1",
            "recognition_resize_interpolation": "linear-v1",
        }
    )
    return values


def _validated_v17_profile(values: dict[str, object]) -> RasterTextOfflineProfileV17:
    fingerprint = raster_text_runtime.raster_text_profile_fingerprint_sha256(values)
    return RasterTextOfflineProfileV17.model_validate(
        {
            **values,
            "profile_fingerprint_sha256": fingerprint,
            "raster_text_profile_revision": f"synthetic-raster-text-v1:{fingerprint[:24]}",
        }
    )


def _prepared_mobile_materials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    candidate: Literal[
        "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10", "v11", "v13", "v14", "v15", "v16", "v17"
    ],
) -> tuple[
    Path,
    (
        RasterTextOfflineProfileV3
        | RasterTextOfflineProfileV4
        | RasterTextOfflineProfileV5
        | RasterTextOfflineProfileV6
        | RasterTextOfflineProfileV7
        | RasterTextOfflineProfileV8
        | RasterTextOfflineProfileV9
        | RasterTextOfflineProfileV10
        | RasterTextOfflineProfileV11
        | RasterTextOfflineProfileV13
        | RasterTextOfflineProfileV14
        | RasterTextOfflineProfileV15
        | RasterTextOfflineProfileV16
        | RasterTextOfflineProfileV17
    ),
]:
    root = _make_mobile_source(tmp_path, monkeypatch)
    model_bundle = mobile_bundle.materialize_verified_raster_text_direct_mobile_paddle_bundle_v1(
        document_preview_root=root
    )
    sidecar_sha = "a" * 64
    sidecar_root = f"raster-text-direct-bundles-v1/releases/{sidecar_sha}"
    manifest = SimpleNamespace(
        extractor_revision="direct-paddle-test-v2",
        raster_dpi=216,
        parent_renderer_package_inventory_relative_path="inventories/parent-renderer.json",
        parent_renderer_required_packages=["pypdfium2", "Pillow"],
        sidecar_interpreter_relative_path="sidecar/interpreter/python.exe",
        sidecar_interpreter_root_relative_path="sidecar/interpreter",
        sidecar_interpreter_root_tree_sha256="1" * 64,
        sidecar_module_relative_path=(
            "sidecar/module-root/customs_ai_agent/service/raster_text_direct_paddle_sidecar.py"
        ),
        sidecar_module_root_relative_path="sidecar/module-root",
        sidecar_module_root_tree_sha256="2" * 64,
        sidecar_package_inventory_relative_path="inventories/direct-paddle-packages.json",
        sidecar_required_packages=[
            "numpy",
            "opencv-contrib-python",
            "paddlepaddle",
            "pyclipper",
        ],
    )

    def bundle_relative_path(relative: str) -> str:
        return f"{sidecar_root}/{relative}"

    def file_sha256(relative: str) -> str:
        return hashlib.sha256(relative.encode("utf-8")).hexdigest()

    sidecar_bundle = SimpleNamespace(
        document_preview_root=root,
        manifest_sha256=sidecar_sha,
        bundle_root_relative_path=sidecar_root,
        manifest=manifest,
        bundle_relative_path=bundle_relative_path,
        file_sha256=file_sha256,
    )
    monkeypatch.setattr(
        mobile_bundle,
        "load_verified_raster_text_direct_paddle_bundle_v1",
        lambda **_kwargs: sidecar_bundle,
    )
    # This focused test supplies only the verified-bundle fields consumed by
    # the builder.  Keep the runtime double, but make its intended contract
    # explicit to mypy at the typed builder boundary.
    typed_sidecar_bundle = cast(VerifiedRasterTextDirectPaddleBundleV1, sidecar_bundle)
    if candidate == "v4":
        prepared_v4 = mobile_bundle.prepare_raster_text_direct_mobile_runtime_profile_v4(
            typed_sidecar_bundle,
            model_bundle,
            pdfium_version="4.30.0",
            minimum_page_confidence=0.42,
        )
        raw = (root / prepared_v4.runtime_configuration_relative_path).read_bytes()
        return root, RasterTextOfflineProfileV4.model_validate_json(raw)
    elif candidate == "v5":
        prepared_v5 = mobile_bundle.prepare_raster_text_direct_mobile_runtime_profile_v5(
            typed_sidecar_bundle,
            model_bundle,
            pdfium_version="4.30.0",
            minimum_page_confidence=0.42,
        )
        raw = (root / prepared_v5.runtime_configuration_relative_path).read_bytes()
        return root, RasterTextOfflineProfileV5.model_validate_json(raw)
    elif candidate == "v6":
        prepared_v6 = mobile_bundle.prepare_raster_text_direct_mobile_runtime_profile_v6(
            typed_sidecar_bundle,
            model_bundle,
            pdfium_version="4.30.0",
            minimum_page_confidence=0.42,
        )
        raw = (root / prepared_v6.runtime_configuration_relative_path).read_bytes()
        return root, RasterTextOfflineProfileV6.model_validate_json(raw)
    elif candidate == "v7":
        prepared_v7 = mobile_bundle.prepare_raster_text_direct_mobile_runtime_profile_v7(
            typed_sidecar_bundle,
            model_bundle,
            pdfium_version="4.30.0",
            minimum_page_confidence=0.42,
        )
        raw = (root / prepared_v7.runtime_configuration_relative_path).read_bytes()
        return root, RasterTextOfflineProfileV7.model_validate_json(raw)
    elif candidate == "v8":
        prepared_v8 = mobile_bundle.prepare_raster_text_direct_mobile_runtime_profile_v8(
            typed_sidecar_bundle,
            model_bundle,
            pdfium_version="4.30.0",
            minimum_page_confidence=0.42,
        )
        raw = (root / prepared_v8.runtime_configuration_relative_path).read_bytes()
        return root, RasterTextOfflineProfileV8.model_validate_json(raw)
    elif candidate == "v9":
        prepared_v9 = mobile_bundle.prepare_raster_text_direct_mobile_runtime_profile_v9(
            typed_sidecar_bundle,
            model_bundle,
            pdfium_version="4.30.0",
            minimum_page_confidence=0.42,
        )
        raw = (root / prepared_v9.runtime_configuration_relative_path).read_bytes()
        return root, RasterTextOfflineProfileV9.model_validate_json(raw)
    elif candidate == "v10":
        prepared_v10 = mobile_bundle.prepare_raster_text_direct_mobile_runtime_profile_v10(
            typed_sidecar_bundle,
            model_bundle,
            pdfium_version="4.30.0",
            minimum_page_confidence=0.42,
        )
        raw = (root / prepared_v10.runtime_configuration_relative_path).read_bytes()
        return root, RasterTextOfflineProfileV10.model_validate_json(raw)
    elif candidate == "v11":
        prepared_v11 = mobile_bundle.prepare_raster_text_direct_mobile_runtime_profile_v11(
            typed_sidecar_bundle,
            model_bundle,
            pdfium_version="4.30.0",
            minimum_page_confidence=0.42,
        )
        raw = (root / prepared_v11.runtime_configuration_relative_path).read_bytes()
        return root, RasterTextOfflineProfileV11.model_validate_json(raw)
    elif candidate == "v13":
        prepared_v13 = mobile_bundle.prepare_raster_text_direct_mobile_runtime_profile_v13(
            typed_sidecar_bundle,
            model_bundle,
            pdfium_version="4.30.0",
            minimum_page_confidence=0.42,
        )
        raw = (root / prepared_v13.runtime_configuration_relative_path).read_bytes()
        return root, RasterTextOfflineProfileV13.model_validate_json(raw)
    elif candidate == "v14":
        prepared_v14 = mobile_bundle.prepare_raster_text_direct_mobile_runtime_profile_v14(
            typed_sidecar_bundle,
            model_bundle,
            pdfium_version="4.30.0",
            minimum_page_confidence=0.42,
        )
        raw = (root / prepared_v14.runtime_configuration_relative_path).read_bytes()
        return root, RasterTextOfflineProfileV14.model_validate_json(raw)
    elif candidate == "v15":
        prepared_v15 = mobile_bundle.prepare_raster_text_direct_mobile_runtime_profile_v15(
            typed_sidecar_bundle,
            model_bundle,
            pdfium_version="4.30.0",
            minimum_page_confidence=0.42,
        )
        raw = (root / prepared_v15.runtime_configuration_relative_path).read_bytes()
        return root, RasterTextOfflineProfileV15.model_validate_json(raw)
    elif candidate == "v16":
        prepared_v16 = mobile_bundle.prepare_raster_text_direct_mobile_runtime_profile_v16(
            typed_sidecar_bundle,
            model_bundle,
            pdfium_version="4.30.0",
            minimum_page_confidence=0.42,
        )
        raw = (root / prepared_v16.runtime_configuration_relative_path).read_bytes()
        return root, RasterTextOfflineProfileV16.model_validate_json(raw)
    elif candidate == "v17":
        prepared_v17 = mobile_bundle.prepare_raster_text_direct_mobile_runtime_profile_v17(
            typed_sidecar_bundle,
            model_bundle,
            pdfium_version="4.30.0",
            minimum_page_confidence=0.42,
        )
        raw = (root / prepared_v17.runtime_configuration_relative_path).read_bytes()
        return root, RasterTextOfflineProfileV17.model_validate_json(raw)
    else:
        prepared_v3 = mobile_bundle.prepare_raster_text_direct_mobile_runtime_profile_v3(
            typed_sidecar_bundle,
            model_bundle,
            pdfium_version="4.30.0",
            minimum_page_confidence=0.42,
        )
        raw = (root / prepared_v3.runtime_configuration_relative_path).read_bytes()
        return root, RasterTextOfflineProfileV3.model_validate_json(raw)


def _prepared_v3_materials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, RasterTextOfflineProfileV3]:
    root, profile = _prepared_mobile_materials(tmp_path, monkeypatch, candidate="v3")
    assert isinstance(profile, RasterTextOfflineProfileV3)
    return root, cast(RasterTextOfflineProfileV3, profile)


def _prepared_v4_materials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, RasterTextOfflineProfileV4]:
    root, profile = _prepared_mobile_materials(tmp_path, monkeypatch, candidate="v4")
    assert isinstance(profile, RasterTextOfflineProfileV4)
    return root, cast(RasterTextOfflineProfileV4, profile)


def _prepared_v5_materials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, RasterTextOfflineProfileV5]:
    root, profile = _prepared_mobile_materials(tmp_path, monkeypatch, candidate="v5")
    assert isinstance(profile, RasterTextOfflineProfileV5)
    return root, cast(RasterTextOfflineProfileV5, profile)


def _prepared_v6_materials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, RasterTextOfflineProfileV6]:
    root, profile = _prepared_mobile_materials(tmp_path, monkeypatch, candidate="v6")
    assert isinstance(profile, RasterTextOfflineProfileV6)
    return root, cast(RasterTextOfflineProfileV6, profile)


def _prepared_v7_materials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, RasterTextOfflineProfileV7]:
    root, profile = _prepared_mobile_materials(tmp_path, monkeypatch, candidate="v7")
    assert isinstance(profile, RasterTextOfflineProfileV7)
    return root, cast(RasterTextOfflineProfileV7, profile)


def _prepared_v8_materials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, RasterTextOfflineProfileV8]:
    root, profile = _prepared_mobile_materials(tmp_path, monkeypatch, candidate="v8")
    assert isinstance(profile, RasterTextOfflineProfileV8)
    return root, cast(RasterTextOfflineProfileV8, profile)


def _prepared_v9_materials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, RasterTextOfflineProfileV9]:
    root, profile = _prepared_mobile_materials(tmp_path, monkeypatch, candidate="v9")
    assert isinstance(profile, RasterTextOfflineProfileV9)
    return root, cast(RasterTextOfflineProfileV9, profile)


def _prepared_v10_materials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, RasterTextOfflineProfileV10]:
    root, profile = _prepared_mobile_materials(tmp_path, monkeypatch, candidate="v10")
    assert isinstance(profile, RasterTextOfflineProfileV10)
    return root, cast(RasterTextOfflineProfileV10, profile)


def _prepared_v11_materials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, RasterTextOfflineProfileV11]:
    root, profile = _prepared_mobile_materials(tmp_path, monkeypatch, candidate="v11")
    assert isinstance(profile, RasterTextOfflineProfileV11)
    return root, cast(RasterTextOfflineProfileV11, profile)


def _prepared_v13_materials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, RasterTextOfflineProfileV13]:
    root, profile = _prepared_mobile_materials(tmp_path, monkeypatch, candidate="v13")
    assert isinstance(profile, RasterTextOfflineProfileV13)
    return root, cast(RasterTextOfflineProfileV13, profile)


def _prepared_v14_materials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, RasterTextOfflineProfileV14]:
    root, profile = _prepared_mobile_materials(tmp_path, monkeypatch, candidate="v14")
    assert isinstance(profile, RasterTextOfflineProfileV14)
    return root, cast(RasterTextOfflineProfileV14, profile)


def _prepared_v15_materials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, RasterTextOfflineProfileV15]:
    root, profile = _prepared_mobile_materials(tmp_path, monkeypatch, candidate="v15")
    assert isinstance(profile, RasterTextOfflineProfileV15)
    return root, cast(RasterTextOfflineProfileV15, profile)


def _prepared_v16_materials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, RasterTextOfflineProfileV16]:
    root, profile = _prepared_mobile_materials(tmp_path, monkeypatch, candidate="v16")
    assert isinstance(profile, RasterTextOfflineProfileV16)
    return root, cast(RasterTextOfflineProfileV16, profile)


def _prepared_v17_materials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[Path, RasterTextOfflineProfileV17]:
    root, profile = _prepared_mobile_materials(tmp_path, monkeypatch, candidate="v17")
    assert isinstance(profile, RasterTextOfflineProfileV17)
    return root, cast(RasterTextOfflineProfileV17, profile)


def _profile_for_changed_manifest(
    profile: RasterTextOfflineProfileV3, manifest_path: Path
) -> RasterTextOfflineProfileV3:
    """Target verifier content after intentionally changing the manifest bytes."""

    return profile.model_copy(
        update={
            "mobile_model_bundle_manifest_sha256": hashlib.sha256(
                manifest_path.read_bytes()
            ).hexdigest()
        }
    )


def test_materializes_and_reloads_the_separate_mobile_model_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_mobile_source(tmp_path, monkeypatch)

    first = mobile_bundle.materialize_verified_raster_text_direct_mobile_paddle_bundle_v1(
        document_preview_root=root
    )
    second = mobile_bundle.materialize_verified_raster_text_direct_mobile_paddle_bundle_v1(
        document_preview_root=root
    )
    loaded = mobile_bundle.load_verified_raster_text_direct_mobile_paddle_bundle_v1(
        document_preview_root=root, manifest_sha256=first.manifest_sha256
    )

    assert first.manifest_sha256 == second.manifest_sha256 == loaded.manifest_sha256
    assert first.bundle_root_relative_path.startswith(
        "raster-text-direct-mobile-bundles-v1/releases/"
    )
    assert first.manifest.text_detection_model_name == "PP-OCRv5_mobile_det"
    assert (first.bundle_root / "models" / "text-det" / "config.json").is_file()
    assert (first.bundle_root / "models" / "text-rec" / "inference.pdiparams").is_file()


def test_source_rejects_a_hash_consistent_but_mismatched_detector_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_mobile_source(tmp_path, monkeypatch, detector_name="PP-OCRv5_server_det")

    with pytest.raises(mobile_bundle.RasterTextMobilePaddleBundleError) as raised:
        mobile_bundle.verify_local_paddleocr3_mobile_model_source_v1(document_preview_root=root)

    assert raised.value.code == "mobile_paddle_bundle_source_model_invalid"
    assert not (root / "raster-text-direct-mobile-bundles-v1" / "releases").exists()


def test_mobile_bundle_tamper_fails_closed_on_explicit_reload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _make_mobile_source(tmp_path, monkeypatch)
    bundle = mobile_bundle.materialize_verified_raster_text_direct_mobile_paddle_bundle_v1(
        document_preview_root=root
    )
    (bundle.bundle_root / "models" / "text-det" / "inference.json").write_bytes(b"tampered")

    with pytest.raises(mobile_bundle.RasterTextMobilePaddleBundleError) as raised:
        mobile_bundle.load_verified_raster_text_direct_mobile_paddle_bundle_v1(
            document_preview_root=root, manifest_sha256=bundle.manifest_sha256
        )

    assert raised.value.code == "mobile_paddle_bundle_manifest_invalid"


def test_v3_profile_fingerprint_rejects_an_unbound_mobile_bundle_or_detector() -> None:
    values = _v3_payload()
    profile = _validated_v3_profile(values)
    assert profile.text_detection_model_name == "PP-OCRv5_mobile_det"

    wrong_root = {
        **values,
        "mobile_model_bundle_root_relative_path": values["bundle_root_relative_path"],
    }
    with pytest.raises(ValueError, match="mobile bundle binding"):
        _validated_v3_profile(wrong_root)

    wrong_detector = {**values, "text_detection_model_name": "PP-OCRv5_server_det"}
    with pytest.raises(ValueError):
        _validated_v3_profile(wrong_detector)

    wrong_packages = {**values, "sidecar_required_packages": ["paddleocr"]}
    with pytest.raises(ValueError, match="required packages"):
        _validated_v3_profile(wrong_packages)


def test_v3_rejects_harmonic_properties_and_v4_rejects_any_other_mobile_tuple() -> None:
    """The candidate mode is schema-bound, never a mutable V3 option."""

    with pytest.raises(ValueError):
        _validated_v3_profile({**_v3_payload(), "row_band_mode": "harmonic_v1"})
    with pytest.raises(ValueError):
        _validated_v3_profile(
            {
                **_v3_payload(),
                "engine_kind": "direct_paddle_mobile_text_harmonic",
            }
        )

    values = _v4_payload()
    profile = _validated_v4_profile(values)
    assert profile.schema_version == "4.0"
    assert profile.engine_kind == "direct_paddle_mobile_text_harmonic"
    assert profile.cpu_threads == 6
    assert profile.sidecar_timeout_seconds == 30
    assert profile.text_detection_limit_side_len == 960

    for key, invalid in (
        ("text_detection_limit_side_len", 1280),
        ("cpu_threads", 4),
        ("engine_kind", "direct_paddle_mobile_text"),
        ("text_detection_model_name", "PP-OCRv5_server_det"),
        ("text_recognition_model_name", "en_PP-OCRv5_server_rec"),
    ):
        with pytest.raises(ValueError):
            _validated_v4_profile({**values, key: invalid})


def test_v5_recall_profile_is_fixed_and_old_mobile_profiles_reject_its_tuple() -> None:
    values = _v5_payload()
    profile = _validated_v5_profile(values)

    assert profile.schema_version == "5.0"
    assert profile.engine_kind == "direct_paddle_mobile_text_harmonic_recall"
    assert profile.cpu_threads == 6
    assert profile.sidecar_timeout_seconds == 30
    assert profile.text_detection_limit_side_len == 960
    assert profile.text_detection_threshold == 0.3
    assert profile.text_detection_box_threshold == 0.4
    assert profile.text_detection_unclip_ratio == 1.5

    for key, invalid in (
        ("text_detection_limit_side_len", 1280),
        ("cpu_threads", 4),
        ("engine_kind", "direct_paddle_mobile_text_harmonic"),
        ("text_detection_model_name", "PP-OCRv5_server_det"),
        ("text_recognition_model_name", "en_PP-OCRv5_server_rec"),
        ("text_detection_box_threshold", 0.6),
    ):
        with pytest.raises(ValueError):
            _validated_v5_profile({**values, key: invalid})

    # Earlier schemas cannot reinterpret the V5 engine or its lower box
    # threshold as a V3/V4 configuration.
    with pytest.raises(ValueError):
        _validated_v3_profile(values)
    with pytest.raises(ValueError):
        _validated_v4_profile(values)


def test_v6_1024_profile_keeps_v5_semantics_and_rejects_any_other_tuple() -> None:
    values = _v6_payload()
    profile = _validated_v6_profile(values)

    assert profile.schema_version == "6.0"
    assert profile.engine_kind == "direct_paddle_mobile_text_1024"
    assert profile.cpu_threads == 6
    assert profile.sidecar_timeout_seconds == 30
    assert profile.text_detection_limit_side_len == 1024
    assert profile.text_detection_threshold == 0.3
    assert profile.text_detection_box_threshold == 0.4
    assert profile.text_detection_unclip_ratio == 1.5

    # The new schema is intentionally V5-equivalent except for detector
    # scale: changing any inherited setting is a distinct candidate.
    for key, invalid in (
        ("text_detection_limit_side_len", 960),
        ("text_detection_limit_side_len", 1280),
        ("cpu_threads", 4),
        ("engine_kind", "direct_paddle_mobile_text_harmonic_recall"),
        ("text_detection_model_name", "PP-OCRv5_server_det"),
        ("text_recognition_model_name", "en_PP-OCRv5_server_rec"),
        ("text_detection_box_threshold", 0.6),
        ("text_detection_unclip_ratio", 2.0),
    ):
        with pytest.raises(ValueError):
            _validated_v6_profile({**values, key: invalid})

    # Existing mobile schemas cannot reinterpret a V6 profile, even though
    # V5 shares its detector/recognizer and lower threshold.
    for validator in (_validated_v3_profile, _validated_v4_profile, _validated_v5_profile):
        with pytest.raises(ValueError):
            validator(values)


def test_v7_low_box_profile_changes_only_v6_box_threshold() -> None:
    values = _v7_payload()
    profile = _validated_v7_profile(values)

    assert profile.schema_version == "7.0"
    assert profile.engine_kind == "direct_paddle_mobile_text_harmonic_recall_1024_lowbox"
    assert profile.cpu_threads == 6
    assert profile.sidecar_timeout_seconds == 30
    assert profile.text_detection_limit_side_len == 1024
    assert profile.text_detection_threshold == 0.3
    assert profile.text_detection_box_threshold == 0.3
    assert profile.text_detection_unclip_ratio == 1.5

    # V7 differs from V6 at exactly this one bound.  Any model, input size,
    # runtime budget, or V6's .4 box threshold is a different candidate.
    for key, invalid in (
        ("text_detection_limit_side_len", 960),
        ("text_detection_limit_side_len", 1280),
        ("cpu_threads", 4),
        ("engine_kind", "direct_paddle_mobile_text_1024"),
        ("text_detection_model_name", "PP-OCRv5_server_det"),
        ("text_recognition_model_name", "en_PP-OCRv5_server_rec"),
        ("text_detection_threshold", 0.4),
        ("text_detection_box_threshold", 0.4),
        ("text_detection_unclip_ratio", 2.0),
    ):
        with pytest.raises(ValueError):
            _validated_v7_profile({**values, key: invalid})

    # V3--V6 cannot reinterpret the lower .3 threshold by borrowing the V7
    # payload; V1/V2 are covered at the direct-sidecar tuple boundary.
    for validator in (
        _validated_v3_profile,
        _validated_v4_profile,
        _validated_v5_profile,
        _validated_v6_profile,
    ):
        with pytest.raises(ValueError):
            validator(values)


def test_v8_native_960_low_box_profile_changes_only_v5_box_threshold() -> None:
    values = _v8_payload()
    profile = _validated_v8_profile(values)

    assert profile.schema_version == "8.0"
    assert profile.engine_kind == "direct_paddle_mobile_text_harmonic_recall_native960_lowbox"
    assert profile.cpu_threads == 6
    assert profile.sidecar_timeout_seconds == 30
    assert profile.text_detection_limit_side_len == 960
    assert profile.text_detection_threshold == 0.3
    assert profile.text_detection_box_threshold == 0.3
    assert profile.text_detection_unclip_ratio == 1.5

    # V8 is V5-native-960 plus its one lower box bound.  V7's 1024 scale is
    # deliberately not accepted through this candidate.
    for key, invalid in (
        ("text_detection_limit_side_len", 1024),
        ("text_detection_limit_side_len", 1280),
        ("cpu_threads", 4),
        ("engine_kind", "direct_paddle_mobile_text_harmonic_recall"),
        ("text_detection_model_name", "PP-OCRv5_server_det"),
        ("text_recognition_model_name", "en_PP-OCRv5_server_rec"),
        ("text_detection_threshold", 0.4),
        ("text_detection_box_threshold", 0.4),
        ("text_detection_unclip_ratio", 2.0),
    ):
        with pytest.raises(ValueError):
            _validated_v8_profile({**values, key: invalid})

    # Existing mobile schemas cannot reinterpret V8's native-960 low-box
    # tuple.  V1/V2 are covered at the direct-sidecar allowlist boundary.
    for validator in (
        _validated_v3_profile,
        _validated_v4_profile,
        _validated_v5_profile,
        _validated_v6_profile,
        _validated_v7_profile,
    ):
        with pytest.raises(ValueError):
            validator(values)


def test_v9_strip2_profile_changes_only_v8_engine_tuple() -> None:
    values = _v9_payload()
    profile = _validated_v9_profile(values)

    assert profile.schema_version == "9.0"
    assert (
        profile.engine_kind == "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2"
    )
    assert profile.cpu_threads == 6
    assert profile.sidecar_timeout_seconds == 30
    assert profile.text_detection_limit_side_len == 960
    assert profile.text_detection_threshold == 0.3
    assert profile.text_detection_box_threshold == 0.3
    assert profile.text_detection_unclip_ratio == 1.5

    # The profile contains no tile/NMS input.  Its only V9-specific material
    # is the exact immutable engine discriminator.
    for key, invalid in (
        ("text_detection_limit_side_len", 1024),
        ("cpu_threads", 4),
        ("engine_kind", "direct_paddle_mobile_text_harmonic_recall_native960_lowbox"),
        ("text_detection_model_name", "PP-OCRv5_server_det"),
        ("text_recognition_model_name", "en_PP-OCRv5_server_rec"),
        ("text_detection_threshold", 0.4),
        ("text_detection_box_threshold", 0.4),
        ("text_detection_unclip_ratio", 2.0),
    ):
        with pytest.raises(ValueError):
            _validated_v9_profile({**values, key: invalid})

    # V3--V8 parsers cannot reinterpret the V9 payload through their own
    # exact schemas.  V1/V2 remain rejected in the direct-sidecar allowlist.
    for validator in (
        _validated_v3_profile,
        _validated_v4_profile,
        _validated_v5_profile,
        _validated_v6_profile,
        _validated_v7_profile,
        _validated_v8_profile,
    ):
        with pytest.raises(ValueError):
            validator(values)


def test_v10_grid2_profile_changes_only_v8_engine_tuple() -> None:
    values = _v10_payload()
    profile = _validated_v10_profile(values)

    assert profile.schema_version == "10.0"
    assert profile.engine_kind == "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_grid2"
    assert profile.cpu_threads == 6
    assert profile.sidecar_timeout_seconds == 30
    assert profile.text_detection_limit_side_len == 960
    assert profile.text_detection_threshold == 0.3
    assert profile.text_detection_box_threshold == 0.3
    assert profile.text_detection_unclip_ratio == 1.5

    for key, invalid in (
        ("text_detection_limit_side_len", 1024),
        ("cpu_threads", 4),
        ("engine_kind", "direct_paddle_mobile_text_harmonic_recall_native960_lowbox"),
        ("text_detection_model_name", "PP-OCRv5_server_det"),
        ("text_recognition_model_name", "en_PP-OCRv5_server_rec"),
        ("text_detection_threshold", 0.4),
        ("text_detection_box_threshold", 0.4),
        ("text_detection_unclip_ratio", 2.0),
    ):
        with pytest.raises(ValueError):
            _validated_v10_profile({**values, key: invalid})

    # V3--V9 cannot reinterpret the V10 payload.  Tiling is an exact
    # sidecar engine semantic rather than a field inherited by old schemas.
    for validator in (
        _validated_v3_profile,
        _validated_v4_profile,
        _validated_v5_profile,
        _validated_v6_profile,
        _validated_v7_profile,
        _validated_v8_profile,
        _validated_v9_profile,
    ):
        with pytest.raises(ValueError):
            validator(values)


def test_v11_guarded_grid2_profile_changes_only_the_immutable_engine_tuple() -> None:
    values = _v11_payload()
    profile = _validated_v11_profile(values)

    assert profile.schema_version == "11.0"
    assert (
        profile.engine_kind
        == "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_guarded_grid2"
    )
    assert profile.cpu_threads == 6
    assert profile.sidecar_timeout_seconds == 30
    assert profile.text_detection_limit_side_len == 960
    assert profile.text_detection_threshold == 0.3
    assert profile.text_detection_box_threshold == 0.3
    assert profile.text_detection_unclip_ratio == 1.5

    for key, invalid in (
        ("text_detection_limit_side_len", 1024),
        ("cpu_threads", 4),
        ("engine_kind", "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_grid2"),
        ("text_detection_model_name", "PP-OCRv5_server_det"),
        ("text_recognition_model_name", "en_PP-OCRv5_server_rec"),
        ("text_detection_threshold", 0.4),
        ("text_detection_box_threshold", 0.4),
        ("text_detection_unclip_ratio", 2.0),
    ):
        with pytest.raises(ValueError):
            _validated_v11_profile({**values, key: invalid})

    # V3--V10 cannot reinterpret the guarded seam behavior through an older
    # profile schema; V11 has no configurable tile/NMS fields of its own.
    for validator in (
        _validated_v3_profile,
        _validated_v4_profile,
        _validated_v5_profile,
        _validated_v6_profile,
        _validated_v7_profile,
        _validated_v8_profile,
        _validated_v9_profile,
        _validated_v10_profile,
    ):
        with pytest.raises(ValueError):
            validator(values)


def test_v13_semantic_profile_changes_only_the_public_v9_engine_tuple() -> None:
    values = _v13_payload()
    profile = _validated_v13_profile(values)

    assert profile.schema_version == "13.0"
    assert (
        profile.engine_kind == "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_"
        "strip2_semantic_repair"
    )
    assert profile.text_detection_model_name == "PP-OCRv5_mobile_det"
    assert profile.text_recognition_model_name == "en_PP-OCRv5_mobile_rec"
    assert profile.cpu_threads == 6
    assert profile.sidecar_timeout_seconds == 30
    assert profile.text_detection_limit_side_len == 960
    assert profile.text_detection_threshold == 0.3
    assert profile.text_detection_box_threshold == 0.3
    assert profile.text_detection_unclip_ratio == 1.5

    for key, invalid in (
        ("text_detection_limit_side_len", 1024),
        ("cpu_threads", 4),
        ("engine_kind", "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2"),
        ("text_detection_model_name", "PP-OCRv5_server_det"),
        ("text_recognition_model_name", "source_attested_tiny_crnn_bigru_ctc_v1"),
        ("text_detection_threshold", 0.4),
        ("text_detection_box_threshold", 0.4),
        ("text_detection_unclip_ratio", 2.0),
    ):
        with pytest.raises(ValueError):
            _validated_v13_profile({**values, key: invalid})

    for validator in (
        _validated_v3_profile,
        _validated_v4_profile,
        _validated_v5_profile,
        _validated_v6_profile,
        _validated_v7_profile,
        _validated_v8_profile,
        _validated_v9_profile,
        _validated_v10_profile,
        _validated_v11_profile,
    ):
        with pytest.raises(ValueError):
            validator(values)


def test_v14_semantic_profile_is_distinct_and_reuses_exact_v13_ocr_material() -> None:
    values = _v14_payload()
    profile = _validated_v14_profile(values)
    v13 = _validated_v13_profile(_v13_payload())

    assert profile.schema_version == "14.0"
    assert (
        profile.engine_kind == "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_"
        "strip2_semantic_repair_v2"
    )
    assert profile.raster_text_profile_revision != v13.raster_text_profile_revision
    assert profile.profile_fingerprint_sha256 != v13.profile_fingerprint_sha256
    for field in (
        "bundle_root_relative_path",
        "bundle_manifest_sha256",
        "mobile_model_bundle_root_relative_path",
        "mobile_model_bundle_manifest_sha256",
        "source_model_inventory_relative_path",
        "source_model_inventory_sha256",
        "text_detection_model_name",
        "text_recognition_model_name",
        "cpu_threads",
        "sidecar_timeout_seconds",
        "text_detection_limit_side_len",
        "text_detection_threshold",
        "text_detection_box_threshold",
        "text_detection_unclip_ratio",
    ):
        assert getattr(profile, field) == getattr(v13, field)

    for key, invalid in (
        ("schema_version", "13.0"),
        (
            "engine_kind",
            "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_semantic_repair",
        ),
        ("text_detection_limit_side_len", 1024),
        ("cpu_threads", 4),
        ("text_detection_box_threshold", 0.4),
    ):
        with pytest.raises(ValueError):
            _validated_v14_profile({**values, key: invalid})

    with pytest.raises(ValueError):
        _validated_v13_profile(values)


def test_v3_profile_preparation_binds_the_mobile_bundle_to_the_sidecar_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, profile = _prepared_v3_materials(tmp_path, monkeypatch)

    assert profile.mobile_model_bundle_root_relative_path.startswith(
        "raster-text-direct-mobile-bundles-v1/releases/"
    )
    assert profile.bundle_manifest_sha256 == "a" * 64
    assert profile.sidecar_timeout_seconds == 30
    assert profile.text_detection_limit_side_len == 960
    assert profile.minimum_page_confidence == 0.42


def test_v4_profile_preparation_keeps_the_two_attested_roots_and_mobile_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, profile = _prepared_v4_materials(tmp_path, monkeypatch)

    assert profile.schema_version == "4.0"
    assert profile.engine_kind == "direct_paddle_mobile_text_harmonic"
    assert profile.bundle_root_relative_path.startswith("raster-text-direct-bundles-v1/releases/")
    assert profile.mobile_model_bundle_root_relative_path.startswith(
        "raster-text-direct-mobile-bundles-v1/releases/"
    )
    assert profile.bundle_root_relative_path != profile.mobile_model_bundle_root_relative_path
    assert profile.sidecar_timeout_seconds == 30
    assert profile.cpu_threads == 6
    assert profile.text_detection_limit_side_len == 960
    assert (
        raster_text_runtime.parse_raster_text_offline_profile(
            profile.model_dump_json().encode("utf-8")
        )
        == profile
    )
    # V4 deliberately invokes the same exhaustive mobile-model material
    # verifier as V3; it does not fall back to an unrelated model root.
    raster_text_runtime._verify_v3_mobile_model_bundle(  # noqa: SLF001 - material boundary
        root=root,
        profile=profile,
    )


def test_v5_profile_preparation_keeps_two_roots_and_recall_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, profile = _prepared_v5_materials(tmp_path, monkeypatch)

    assert profile.schema_version == "5.0"
    assert profile.engine_kind == "direct_paddle_mobile_text_harmonic_recall"
    assert profile.bundle_root_relative_path.startswith("raster-text-direct-bundles-v1/releases/")
    assert profile.mobile_model_bundle_root_relative_path.startswith(
        "raster-text-direct-mobile-bundles-v1/releases/"
    )
    assert profile.bundle_root_relative_path != profile.mobile_model_bundle_root_relative_path
    assert profile.sidecar_timeout_seconds == 30
    assert profile.cpu_threads == 6
    assert profile.text_detection_limit_side_len == 960
    assert profile.text_detection_box_threshold == 0.4
    assert (
        raster_text_runtime.parse_raster_text_offline_profile(
            profile.model_dump_json().encode("utf-8")
        )
        == profile
    )
    raster_text_runtime._verify_v3_mobile_model_bundle(  # noqa: SLF001 - material boundary
        root=root,
        profile=profile,
    )


def test_v6_profile_preparation_keeps_two_roots_and_changes_only_detector_scale(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, profile = _prepared_v6_materials(tmp_path, monkeypatch)

    assert profile.schema_version == "6.0"
    assert profile.engine_kind == "direct_paddle_mobile_text_1024"
    assert profile.bundle_root_relative_path.startswith("raster-text-direct-bundles-v1/releases/")
    assert profile.mobile_model_bundle_root_relative_path.startswith(
        "raster-text-direct-mobile-bundles-v1/releases/"
    )
    assert profile.bundle_root_relative_path != profile.mobile_model_bundle_root_relative_path
    assert profile.sidecar_timeout_seconds == 30
    assert profile.cpu_threads == 6
    assert profile.text_detection_limit_side_len == 1024
    assert profile.text_detection_box_threshold == 0.4
    assert (
        raster_text_runtime.parse_raster_text_offline_profile(
            profile.model_dump_json().encode("utf-8")
        )
        == profile
    )
    raster_text_runtime._verify_v3_mobile_model_bundle(  # noqa: SLF001 - material boundary
        root=root,
        profile=profile,
    )


def test_v7_profile_preparation_keeps_two_roots_and_changes_only_box_threshold(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, profile = _prepared_v7_materials(tmp_path, monkeypatch)

    assert profile.schema_version == "7.0"
    assert profile.engine_kind == "direct_paddle_mobile_text_harmonic_recall_1024_lowbox"
    assert profile.bundle_root_relative_path.startswith("raster-text-direct-bundles-v1/releases/")
    assert profile.mobile_model_bundle_root_relative_path.startswith(
        "raster-text-direct-mobile-bundles-v1/releases/"
    )
    assert profile.bundle_root_relative_path != profile.mobile_model_bundle_root_relative_path
    assert profile.sidecar_timeout_seconds == 30
    assert profile.cpu_threads == 6
    assert profile.text_detection_limit_side_len == 1024
    assert profile.text_detection_box_threshold == 0.3
    assert (
        raster_text_runtime.parse_raster_text_offline_profile(
            profile.model_dump_json().encode("utf-8")
        )
        == profile
    )
    raster_text_runtime._verify_v3_mobile_model_bundle(  # noqa: SLF001 - material boundary
        root=root,
        profile=profile,
    )


def test_v8_profile_preparation_keeps_two_roots_and_native_960_low_box(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, profile = _prepared_v8_materials(tmp_path, monkeypatch)

    assert profile.schema_version == "8.0"
    assert profile.engine_kind == "direct_paddle_mobile_text_harmonic_recall_native960_lowbox"
    assert profile.bundle_root_relative_path.startswith("raster-text-direct-bundles-v1/releases/")
    assert profile.mobile_model_bundle_root_relative_path.startswith(
        "raster-text-direct-mobile-bundles-v1/releases/"
    )
    assert profile.bundle_root_relative_path != profile.mobile_model_bundle_root_relative_path
    assert profile.sidecar_timeout_seconds == 30
    assert profile.cpu_threads == 6
    assert profile.text_detection_limit_side_len == 960
    assert profile.text_detection_box_threshold == 0.3
    assert (
        raster_text_runtime.parse_raster_text_offline_profile(
            profile.model_dump_json().encode("utf-8")
        )
        == profile
    )
    raster_text_runtime._verify_v3_mobile_model_bundle(  # noqa: SLF001 - material boundary
        root=root,
        profile=profile,
    )


def test_v9_profile_preparation_keeps_two_roots_and_native_960_strip2_tuple(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, profile = _prepared_v9_materials(tmp_path, monkeypatch)

    assert profile.schema_version == "9.0"
    assert (
        profile.engine_kind == "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2"
    )
    assert profile.bundle_root_relative_path.startswith("raster-text-direct-bundles-v1/releases/")
    assert profile.mobile_model_bundle_root_relative_path.startswith(
        "raster-text-direct-mobile-bundles-v1/releases/"
    )
    assert profile.bundle_root_relative_path != profile.mobile_model_bundle_root_relative_path
    assert profile.sidecar_timeout_seconds == 30
    assert profile.cpu_threads == 6
    assert profile.text_detection_limit_side_len == 960
    assert profile.text_detection_box_threshold == 0.3
    assert (
        raster_text_runtime.parse_raster_text_offline_profile(
            profile.model_dump_json().encode("utf-8")
        )
        == profile
    )
    raster_text_runtime._verify_v3_mobile_model_bundle(  # noqa: SLF001 - material boundary
        root=root,
        profile=profile,
    )


def test_v10_profile_preparation_keeps_two_roots_and_native_960_grid2_tuple(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, profile = _prepared_v10_materials(tmp_path, monkeypatch)

    assert profile.schema_version == "10.0"
    assert profile.engine_kind == "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_grid2"
    assert profile.bundle_root_relative_path.startswith("raster-text-direct-bundles-v1/releases/")
    assert profile.mobile_model_bundle_root_relative_path.startswith(
        "raster-text-direct-mobile-bundles-v1/releases/"
    )
    assert profile.bundle_root_relative_path != profile.mobile_model_bundle_root_relative_path
    assert profile.sidecar_timeout_seconds == 30
    assert profile.cpu_threads == 6
    assert profile.text_detection_limit_side_len == 960
    assert profile.text_detection_box_threshold == 0.3
    assert (
        raster_text_runtime.parse_raster_text_offline_profile(
            profile.model_dump_json().encode("utf-8")
        )
        == profile
    )
    raster_text_runtime._verify_v3_mobile_model_bundle(  # noqa: SLF001 - material boundary
        root=root,
        profile=profile,
    )


def test_v11_profile_preparation_keeps_two_roots_and_native_960_guarded_grid2_tuple(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, profile = _prepared_v11_materials(tmp_path, monkeypatch)

    assert profile.schema_version == "11.0"
    assert (
        profile.engine_kind
        == "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_guarded_grid2"
    )
    assert profile.bundle_root_relative_path.startswith("raster-text-direct-bundles-v1/releases/")
    assert profile.mobile_model_bundle_root_relative_path.startswith(
        "raster-text-direct-mobile-bundles-v1/releases/"
    )
    assert profile.bundle_root_relative_path != profile.mobile_model_bundle_root_relative_path
    assert profile.sidecar_timeout_seconds == 30
    assert profile.cpu_threads == 6
    assert profile.text_detection_limit_side_len == 960
    assert profile.text_detection_box_threshold == 0.3
    assert (
        raster_text_runtime.parse_raster_text_offline_profile(
            profile.model_dump_json().encode("utf-8")
        )
        == profile
    )
    raster_text_runtime._verify_v3_mobile_model_bundle(  # noqa: SLF001 - material boundary
        root=root,
        profile=profile,
    )


def test_v13_profile_preparation_reuses_public_v9_material_and_exact_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, profile = _prepared_v13_materials(tmp_path, monkeypatch)

    assert profile.schema_version == "13.0"
    assert (
        profile.engine_kind == "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_"
        "strip2_semantic_repair"
    )
    assert profile.bundle_root_relative_path.startswith("raster-text-direct-bundles-v1/releases/")
    assert profile.mobile_model_bundle_root_relative_path.startswith(
        "raster-text-direct-mobile-bundles-v1/releases/"
    )
    assert profile.bundle_root_relative_path != profile.mobile_model_bundle_root_relative_path
    assert profile.text_detection_model_name == "PP-OCRv5_mobile_det"
    assert profile.text_recognition_model_name == "en_PP-OCRv5_mobile_rec"
    assert profile.sidecar_timeout_seconds == 30
    assert profile.cpu_threads == 6
    assert profile.text_detection_limit_side_len == 960
    assert profile.text_detection_threshold == 0.3
    assert profile.text_detection_box_threshold == 0.3
    assert profile.text_detection_unclip_ratio == 1.5
    assert (
        raster_text_runtime.parse_raster_text_offline_profile(
            profile.model_dump_json().encode("utf-8")
        )
        == profile
    )
    raster_text_runtime._verify_v3_mobile_model_bundle(  # noqa: SLF001 - material boundary
        root=root,
        profile=profile,
    )


def test_v14_profile_preparation_is_separately_fingerprinted_with_v13_bounds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, profile = _prepared_v14_materials(tmp_path, monkeypatch)

    assert profile.schema_version == "14.0"
    assert (
        profile.engine_kind == "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_"
        "strip2_semantic_repair_v2"
    )
    assert profile.text_detection_model_name == "PP-OCRv5_mobile_det"
    assert profile.text_recognition_model_name == "en_PP-OCRv5_mobile_rec"
    assert profile.sidecar_timeout_seconds == 30
    assert profile.cpu_threads == 6
    assert profile.text_detection_limit_side_len == 960
    assert profile.text_detection_threshold == 0.3
    assert profile.text_detection_box_threshold == 0.3
    assert profile.text_detection_unclip_ratio == 1.5
    assert (
        raster_text_runtime.parse_raster_text_offline_profile(
            profile.model_dump_json().encode("utf-8")
        )
        == profile
    )
    raster_text_runtime._verify_v3_mobile_model_bundle(  # noqa: SLF001 - material boundary
        root=root,
        profile=profile,
    )


@pytest.mark.parametrize("missing", [False, True], ids=["mismatch", "missing"])
def test_runtime_rejects_a_missing_or_mismatched_source_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing: bool
) -> None:
    root, profile = _prepared_v3_materials(tmp_path, monkeypatch)
    manifest_path = root / profile.mobile_model_bundle_root_relative_path / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    if missing:
        manifest.pop("source_model_snapshot_sha256")
    else:
        manifest["source_model_snapshot_sha256"] = "0" * 64
    manifest_path.write_bytes(_canonical(manifest))

    with pytest.raises(
        raster_text_runtime.RasterTextRuntimeError, match="mobile_model_bundle_invalid"
    ):
        raster_text_runtime._verify_v3_mobile_model_bundle(  # noqa: SLF001 - material boundary
            root=root,
            profile=_profile_for_changed_manifest(profile, manifest_path),
        )


def test_runtime_rejects_a_model_tree_or_manifest_record_mismatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, profile = _prepared_v3_materials(tmp_path, monkeypatch)
    manifest_path = root / profile.mobile_model_bundle_root_relative_path / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["model_root_tree_sha256"] = "0" * 64
    manifest_path.write_bytes(_canonical(manifest))

    with pytest.raises(
        raster_text_runtime.RasterTextRuntimeError, match="mobile_model_bundle_invalid"
    ):
        raster_text_runtime._verify_v3_mobile_model_bundle(  # noqa: SLF001 - material boundary
            root=root,
            profile=_profile_for_changed_manifest(profile, manifest_path),
        )

    # A fresh material set avoids conflating the tree fingerprint failure
    # with an independently malformed manifest record.
    root, profile = _prepared_v3_materials(tmp_path / "record", monkeypatch)
    manifest_path = root / profile.mobile_model_bundle_root_relative_path / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    model_record = next(
        item for item in manifest["files"] if item["path"] == "models/text-det/config.json"
    )
    model_record["byte_size"] += 1
    manifest_path.write_bytes(_canonical(manifest))

    with pytest.raises(
        raster_text_runtime.RasterTextRuntimeError, match="mobile_model_bundle_invalid"
    ):
        raster_text_runtime._verify_v3_mobile_model_bundle(  # noqa: SLF001 - material boundary
            root=root,
            profile=_profile_for_changed_manifest(profile, manifest_path),
        )


def test_runtime_rejects_an_extra_mobile_model_file_and_source_inventory_drift(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, profile = _prepared_v3_materials(tmp_path, monkeypatch)
    model_root = raster_text_runtime._v2_bundle_sidecar_path(  # noqa: SLF001 - short-root boundary
        root=root,
        profile=profile,
        value=profile.local_model_root,
    )
    extra = model_root / "text-det" / "unexpected.bin"
    _write(extra, b"unexpected")

    with pytest.raises(
        raster_text_runtime.RasterTextRuntimeError, match="mobile_model_bundle_invalid"
    ):
        raster_text_runtime._verify_v3_mobile_model_bundle(  # noqa: SLF001 - material boundary
            root=root,
            profile=profile,
        )

    root, profile = _prepared_v3_materials(tmp_path / "source", monkeypatch)
    source_path = root / profile.source_model_inventory_relative_path
    source_path.write_bytes(_canonical({"artifacts": []}))
    with pytest.raises(
        raster_text_runtime.RasterTextRuntimeError, match="mobile_model_bundle_invalid"
    ):
        raster_text_runtime._verify_v3_mobile_model_bundle(  # noqa: SLF001 - material boundary
            root=root,
            profile=profile,
        )


def test_v15_onednn_controls_are_bounded_and_part_of_the_profile_fingerprint() -> None:
    default = _validated_v15_profile(_v15_payload())
    cpu4 = _validated_v15_profile({**_v15_payload(), "cpu_threads": 4})
    batch16 = _validated_v15_profile({**_v15_payload(), "recognition_batch_size": 16})

    assert default.enable_mkldnn
    assert default.mkldnn_cache_capacity == 10
    assert default.cpu_threads == 6
    assert default.recognition_batch_size == 8
    assert default.opencv_threads == 1
    assert default.warmup_shape_policy == "detector-736x960-recognizer-48x320-v1"
    assert default.onednn_model_policy == "recognizer-only-v1"
    assert (
        len(
            {
                default.profile_fingerprint_sha256,
                cpu4.profile_fingerprint_sha256,
                batch16.profile_fingerprint_sha256,
            }
        )
        == 3
    )
    assert not isinstance(default, RasterTextOfflineProfileV14)

    for key, invalid in (
        ("enable_mkldnn", False),
        ("mkldnn_cache_capacity", 0),
        ("cpu_threads", 8),
        ("recognition_batch_size", 4),
        ("opencv_threads", 0),
    ):
        with pytest.raises(ValueError):
            _validated_v15_profile({**_v15_payload(), key: invalid})


def test_v15_preparer_emits_the_fixed_default_native_controls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, profile = _prepared_v15_materials(tmp_path, monkeypatch)

    assert profile.cpu_threads == 6
    assert profile.recognition_batch_size == 8
    assert profile.opencv_threads == 1
    assert profile.enable_mkldnn
    assert profile.mkldnn_cache_capacity == 10
    assert profile.warmup_shape_policy == "detector-736x960-recognizer-48x320-v1"
    assert profile.onednn_model_policy == "recognizer-only-v1"
    assert (
        raster_text_runtime.parse_raster_text_offline_profile(
            (
                root
                / "raster-text-runtime-profiles-v1"
                / profile.profile_fingerprint_sha256[:24]
                / "profile.json"
            ).read_bytes()
        )
        == profile
    )


def test_v16_recognition_policies_are_fixed_and_fingerprinted() -> None:
    profile = _validated_v16_profile(_v16_payload())

    assert profile.recognition_crop_policy == "detector-quad-v1"
    assert profile.recognition_resize_interpolation == "cubic-v1"
    assert profile.cpu_threads == 6
    assert profile.recognition_batch_size == 8
    assert not isinstance(profile, RasterTextOfflineProfileV15)
    assert (
        raster_text_runtime.parse_raster_text_offline_profile(
            json.dumps(
                profile.model_dump(mode="json"), separators=(",", ":"), sort_keys=True
            ).encode()
        )
        == profile
    )

    for key, invalid in (
        ("recognition_crop_policy", "expanded-v1"),
        ("recognition_resize_interpolation", "linear-v1"),
    ):
        with pytest.raises(ValueError):
            _validated_v16_profile({**_v16_payload(), key: invalid})

    assert (
        profile.profile_fingerprint_sha256
        != _validated_v15_profile(_v15_payload()).profile_fingerprint_sha256
    )


def test_v16_preparer_emits_only_the_cubic_recognition_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _root, profile = _prepared_v16_materials(tmp_path, monkeypatch)

    assert profile.recognition_crop_policy == "detector-quad-v1"
    assert profile.recognition_resize_interpolation == "cubic-v1"
    assert profile.enable_mkldnn
    assert profile.mkldnn_cache_capacity == 10
    assert profile.cpu_threads == 6
    assert profile.recognition_batch_size == 8
    assert profile.opencv_threads == 1
    assert profile.warmup_shape_policy == "detector-736x960-recognizer-48x320-v1"
    assert profile.onednn_model_policy == "recognizer-only-v1"


def test_v17_padding_policy_is_fixed_and_fingerprinted() -> None:
    profile = _validated_v17_profile(_v17_payload())

    assert profile.recognition_crop_policy == "horizontal-height-0p08-max4-v1"
    assert profile.recognition_resize_interpolation == "linear-v1"
    assert profile.cpu_threads == 6
    assert profile.recognition_batch_size == 8
    assert not isinstance(profile, (RasterTextOfflineProfileV15, RasterTextOfflineProfileV16))
    assert (
        raster_text_runtime.parse_raster_text_offline_profile(
            json.dumps(
                profile.model_dump(mode="json"), separators=(",", ":"), sort_keys=True
            ).encode()
        )
        == profile
    )

    for key, invalid in (
        ("recognition_crop_policy", "detector-quad-v1"),
        ("recognition_resize_interpolation", "cubic-v1"),
    ):
        with pytest.raises(ValueError):
            _validated_v17_profile({**_v17_payload(), key: invalid})

    fingerprints = {
        _validated_v15_profile(_v15_payload()).profile_fingerprint_sha256,
        _validated_v16_profile(_v16_payload()).profile_fingerprint_sha256,
        profile.profile_fingerprint_sha256,
    }
    assert len(fingerprints) == 3


def test_v17_preparer_emits_only_the_padded_recognition_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _root, profile = _prepared_v17_materials(tmp_path, monkeypatch)

    assert profile.recognition_crop_policy == "horizontal-height-0p08-max4-v1"
    assert profile.recognition_resize_interpolation == "linear-v1"
    assert profile.enable_mkldnn
    assert profile.mkldnn_cache_capacity == 10
    assert profile.cpu_threads == 6
    assert profile.recognition_batch_size == 8
    assert profile.opencv_threads == 1
    assert profile.warmup_shape_policy == "detector-736x960-recognizer-48x320-v1"
    assert profile.onednn_model_policy == "recognizer-only-v1"
