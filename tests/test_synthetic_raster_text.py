from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from itertools import product
import socket
from types import SimpleNamespace
from typing import cast

import pytest

from customs_ai_agent.evaluation import synthetic_raster_text_activation as raster_text_activation
from customs_ai_agent.evaluation import synthetic_raster_text as raster_text_evaluation
from customs_ai_agent.evaluation.raster_offline_bundle import _RASTER_TEXT_CPU_THREADS
from customs_ai_agent.evaluation.synthetic_raster import (
    RasterGoldArtifactV1,
    RasterGoldFieldV1,
    RasterGoldSpanV1,
)
from customs_ai_agent.evaluation.synthetic_raster_text import (
    RasterTextEvaluationCaseV1,
    RasterTextGateScoreV1,
    RasterTextGoldArtifactV1,
    evaluate_synthetic_raster_text_case,
    source_only_text_gold_from_raster_gold,
)
from customs_ai_agent.evaluation.synthetic_raster_text_activation import (
    RasterTextSmokeQualityScreenV1,
    SyntheticRasterTextActivationError,
    auto_activate_synthetic_raster_text_v1,
)
from customs_ai_agent.schemas.document_extraction import BoundingBoxV1
from customs_ai_agent.schemas.raster_text_extraction import (
    RasterTextExtractionResultV1,
    RasterTextInputV1,
    RasterTextPageV1,
    RasterTextPreviewCandidateV1,
    canonical_raster_text,
    canonical_raster_text_sha256,
)
from customs_ai_agent.service.raster_text_sidecar import _prediction_lines, deny_network_connections
from customs_ai_agent.service.document_input import DocumentStructure
import customs_ai_agent.service.raster_text_preview_runtime as raster_text_runtime
from customs_ai_agent.service.raster_text_extraction import (
    LocalRasterTextPreviewEngine,
    RasterTextPreviewUnavailable,
)
from customs_ai_agent.service.raster_text_preview_runtime import (
    AutoActivatingSourceAttestedRasterTextPreviewRuntime,
    PaddleOcrTextCommandProvider,
    RasterTextCalibrationReceiptV1,
    RasterTextOfflineProfileV1,
    RasterTextOfflineProfileV2,
    RasterTextOfflineProfileV3,
    RasterTextOfflineProfileV10,
    RasterTextOfflineProfileV11,
    RasterTextOfflineProfileV12,
    RasterTextOfflineProfileV13,
    RasterTextOfflineProfileV14,
    RasterTextOfflineProfileV15,
    RasterTextRuntimeError,
    RasterTextSealedEvaluationV1,
    RasterTextSealedProfileScoreV1,
    RasterTextWarmServiceSmokeReceiptV1,
    SourceAttestedRasterTextPreviewBindingV1,
    parse_raster_text_activation_pointer,
    parse_raster_text_offline_profile,
)


_SHA = "a" * 64


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=True, allow_nan=False, separators=(",", ":"), sort_keys=True
        ).encode("utf-8")
        + b"\n"
    )


@dataclass(frozen=True)
class _CachedTextRuntimeFixture:
    root: Path
    pointer_path: Path
    pointer_raw: bytes
    configuration_path: Path
    profile: RasterTextOfflineProfileV1
    interpreter: Path
    interpreter_root: Path
    module: Path
    module_root: Path
    sidecar_inventory: Path
    model_root: Path
    model_inventory: Path


def _cached_text_runtime_materials(tmp_path: Path) -> _CachedTextRuntimeFixture:
    """Create only the parent-visible files needed by cached-runtime tests."""

    # The immutable production release path includes a 64-character manifest
    # segment.  Preserve that shape while using the Win32 extended namespace
    # for pytest's already-deep temporary directory.
    temporary_root = Path("\\\\?\\" + str(tmp_path)) if os.name == "nt" else tmp_path
    root = temporary_root / "_local" / "data" / "document-preview"
    manifest_sha256 = "b" * 64
    bundle_relative = f"raster-offline-bundles-v1/releases/{manifest_sha256}"
    bundle = root.joinpath(*bundle_relative.split("/"))
    interpreter_root = bundle / "sidecar" / "interpreter"
    interpreter = interpreter_root / "python.exe"
    module_root = bundle / "sidecar" / "module-root"
    module = module_root / "customs_ai_agent" / "service" / "raster_text_sidecar.py"
    parent_inventory = bundle / "inventories" / "parent-renderer.json"
    sidecar_inventory = bundle / "inventories" / "sidecar-packages.json"
    model_root = bundle / "models"
    model_inventory = bundle / "inventories" / "models.json"
    configuration_path = root / "raster-text-runtime-profiles-v1" / "test" / "profile.json"
    for directory in (
        interpreter_root,
        module.parent,
        parent_inventory.parent,
        model_root / "text-det",
        model_root / "text-rec",
        configuration_path.parent,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    interpreter.write_bytes(b"isolated interpreter")
    module.write_bytes(b"bound raster text sidecar")
    parent_inventory.write_bytes(b"parent inventory")
    sidecar_inventory.write_bytes(b"sidecar inventory")
    model_inventory.write_bytes(b"model inventory")

    def relative(path: Path) -> str:
        return path.relative_to(root).as_posix()

    profile_values: dict[str, object] = {
        "schema_version": "1.0",
        "engine_kind": "paddleocr3_direct_text",
        "runtime_api": "paddleocr3-paddlex-3.2",
        "extractor_revision": "direct-text-test-v1",
        "bundle_root_relative_path": bundle_relative,
        "bundle_manifest_sha256": manifest_sha256,
        "pdfium_version": "4.30.0",
        "raster_dpi": 216,
        "sidecar_timeout_seconds": 30,
        "cpu_threads": 6,
        "text_detection_limit_side_len": 2048,
        "text_detection_limit_type": "max",
        "text_detection_threshold": 0.3,
        "text_detection_box_threshold": 0.6,
        "text_detection_unclip_ratio": 1.5,
        "minimum_page_confidence": 0.5,
        "parent_renderer_package_inventory_path": relative(parent_inventory),
        "parent_renderer_package_inventory_sha256": hashlib.sha256(
            parent_inventory.read_bytes()
        ).hexdigest(),
        "parent_renderer_required_packages": ["pypdfium2", "Pillow"],
        "sidecar_interpreter_path": relative(interpreter),
        "sidecar_interpreter_sha256": hashlib.sha256(interpreter.read_bytes()).hexdigest(),
        "sidecar_interpreter_root": relative(interpreter_root),
        "sidecar_interpreter_root_tree_sha256": raster_text_runtime._controlled_directory_tree_sha256(
            interpreter_root
        ),
        "sidecar_module_path": relative(module),
        "sidecar_module_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
        "sidecar_module_root": relative(module_root),
        "sidecar_module_root_tree_sha256": raster_text_runtime._controlled_module_root_tree_sha256(
            module_root
        ),
        "sidecar_package_inventory_path": relative(sidecar_inventory),
        "sidecar_package_inventory_sha256": hashlib.sha256(
            sidecar_inventory.read_bytes()
        ).hexdigest(),
        "sidecar_required_packages": ["paddleocr", "paddlex", "paddlepaddle"],
        "local_model_root": relative(model_root),
        "model_inventory_path": relative(model_inventory),
        "model_inventory_sha256": hashlib.sha256(model_inventory.read_bytes()).hexdigest(),
        "text_detection_model_name": "PP-OCRv5_server_det",
        "text_detection_model_directory": "text-det",
        "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
        "text_recognition_model_directory": "text-rec",
    }
    fingerprint = raster_text_runtime.raster_text_profile_fingerprint_sha256(profile_values)
    profile = RasterTextOfflineProfileV1.model_validate(
        {
            **profile_values,
            "profile_fingerprint_sha256": fingerprint,
            "raster_text_profile_revision": f"synthetic-raster-text-v1:{fingerprint[:24]}",
        }
    )
    # Match the immutable profile writer rather than Pydantic's declaration
    # order serializer: the runtime rejects omitted/default-filled legacy
    # fields by requiring these exact canonical bytes.
    configuration_raw = raster_text_activation._canonical_json(  # noqa: SLF001
        profile.model_dump(mode="json", warnings=False)
    )
    configuration_path.write_bytes(configuration_raw)
    pointer_raw = _canonical_bytes(
        {
            "schema_version": "1.0",
            "activation_kind": "source_attested_synthetic_raster_text",
            "status": "activated",
            "dataset_id": "synthetic-raster-text-v1",
            "raster_text_profile_revision": profile.raster_text_profile_revision,
            "runtime_configuration_path": relative(configuration_path),
            "runtime_configuration_sha256": hashlib.sha256(configuration_raw).hexdigest(),
            "sealed_evaluation_path": "synthetic-raster-text-v1/sealed/test.json",
            "sealed_evaluation_sha256": "c" * 64,
        }
    )
    pointer_path = root / "approvals" / "raster-text-active.json"
    pointer_path.parent.mkdir(parents=True)
    pointer_path.write_bytes(pointer_raw)
    return _CachedTextRuntimeFixture(
        root=root,
        pointer_path=pointer_path,
        pointer_raw=pointer_raw,
        configuration_path=configuration_path,
        profile=profile,
        interpreter=interpreter,
        interpreter_root=interpreter_root,
        module=module,
        module_root=module_root,
        sidecar_inventory=sidecar_inventory,
        model_root=model_root,
        model_inventory=model_inventory,
    )


def _cached_text_pointer_raw(
    fixture: _CachedTextRuntimeFixture,
    *,
    revision: str,
    sealed_evaluation_sha256: str = "c" * 64,
) -> bytes:
    return _canonical_bytes(
        {
            "schema_version": "1.0",
            "activation_kind": "source_attested_synthetic_raster_text",
            "status": "activated",
            "dataset_id": "synthetic-raster-text-v1",
            "raster_text_profile_revision": revision,
            "runtime_configuration_path": fixture.configuration_path.relative_to(
                fixture.root
            ).as_posix(),
            "runtime_configuration_sha256": hashlib.sha256(
                fixture.configuration_path.read_bytes()
            ).hexdigest(),
            "sealed_evaluation_path": "synthetic-raster-text-v1/sealed/test.json",
            "sealed_evaluation_sha256": sealed_evaluation_sha256,
        }
    )


def _direct_v2_profile() -> RasterTextOfflineProfileV2:
    """Build an in-memory V2 profile for parser/environment contract tests."""

    manifest_sha256 = "d" * 64
    bundle = f"raster-text-direct-bundles-v1/releases/{manifest_sha256}"
    values: dict[str, object] = {
        "schema_version": "2.0",
        "engine_kind": "direct_paddle_text",
        "runtime_api": "paddle-inference-static-v1",
        "extractor_revision": "direct-paddle-test-v2",
        "bundle_root_relative_path": bundle,
        "bundle_manifest_sha256": manifest_sha256,
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
        "parent_renderer_package_inventory_path": f"{bundle}/inventories/parent-renderer.json",
        "parent_renderer_package_inventory_sha256": "1" * 64,
        "parent_renderer_required_packages": ["pypdfium2", "Pillow"],
        "sidecar_interpreter_path": f"{bundle}/sidecar/interpreter/python.exe",
        "sidecar_interpreter_sha256": "2" * 64,
        "sidecar_interpreter_root": f"{bundle}/sidecar/interpreter",
        "sidecar_interpreter_root_tree_sha256": "3" * 64,
        "sidecar_module_path": (
            f"{bundle}/sidecar/module-root/customs_ai_agent/service/"
            "raster_text_direct_paddle_sidecar.py"
        ),
        "sidecar_module_sha256": "4" * 64,
        "sidecar_module_root": f"{bundle}/sidecar/module-root",
        "sidecar_module_root_tree_sha256": "5" * 64,
        "sidecar_package_inventory_path": f"{bundle}/inventories/direct-paddle-packages.json",
        "sidecar_package_inventory_sha256": "6" * 64,
        "sidecar_required_packages": [
            "numpy",
            "opencv-contrib-python",
            "paddlepaddle",
            "pyclipper",
        ],
        "local_model_root": f"{bundle}/models",
        "model_inventory_path": f"{bundle}/inventories/direct-paddle-models.json",
        "model_inventory_sha256": "7" * 64,
        "text_detection_model_name": "PP-OCRv5_server_det",
        "text_detection_model_directory": "text-det",
        "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
        "text_recognition_model_directory": "text-rec",
    }
    fingerprint = raster_text_runtime.raster_text_profile_fingerprint_sha256(values)
    return RasterTextOfflineProfileV2.model_validate(
        {
            **values,
            "profile_fingerprint_sha256": fingerprint,
            "raster_text_profile_revision": f"synthetic-raster-text-v1:{fingerprint[:24]}",
        }
    )


def _direct_v3_profile() -> RasterTextOfflineProfileV3:
    """Build the V3 mobile binding used by activation-only tests."""

    values = _direct_v2_profile().model_dump(
        mode="json",
        warnings=False,
        exclude={"profile_fingerprint_sha256", "raster_text_profile_revision"},
    )
    mobile_manifest_sha256 = "e" * 64
    mobile_bundle = f"raster-text-direct-mobile-bundles-v1/releases/{mobile_manifest_sha256}"
    values.update(
        {
            "schema_version": "3.0",
            "engine_kind": "direct_paddle_mobile_text",
            "extractor_revision": "direct-paddle-mobile-test-v3",
            "mobile_model_bundle_root_relative_path": mobile_bundle,
            "mobile_model_bundle_manifest_sha256": mobile_manifest_sha256,
            "source_model_inventory_relative_path": f"model-inventories/{'f' * 64}.json",
            "source_model_inventory_sha256": "f" * 64,
            "local_model_root": f"{mobile_bundle}/models",
            "model_inventory_path": f"{mobile_bundle}/inventories/direct-paddle-mobile-models.json",
            "model_inventory_sha256": "8" * 64,
            "text_detection_model_name": "PP-OCRv5_mobile_det",
        }
    )
    fingerprint = raster_text_runtime.raster_text_profile_fingerprint_sha256(values)
    return RasterTextOfflineProfileV3.model_validate(
        {
            **values,
            "profile_fingerprint_sha256": fingerprint,
            "raster_text_profile_revision": f"synthetic-raster-text-v1:{fingerprint[:24]}",
        }
    )


def _direct_v11_profile() -> RasterTextOfflineProfileV11:
    """Build the V11 guarded-grid binding used by activation-only tests."""

    values = _direct_v3_profile().model_dump(
        mode="json",
        warnings=False,
        exclude={"profile_fingerprint_sha256", "raster_text_profile_revision"},
    )
    values.update(
        {
            "schema_version": "11.0",
            "engine_kind": (
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_guarded_grid2"
            ),
            "extractor_revision": "direct-paddle-mobile-test-v11",
            "text_detection_box_threshold": 0.3,
        }
    )
    fingerprint = raster_text_runtime.raster_text_profile_fingerprint_sha256(values)
    return RasterTextOfflineProfileV11.model_validate(
        {
            **values,
            "profile_fingerprint_sha256": fingerprint,
            "raster_text_profile_revision": f"synthetic-raster-text-v1:{fingerprint[:24]}",
        }
    )


def _direct_v10_profile() -> RasterTextOfflineProfileV10:
    """Build the closest earlier profile to prove V11's exact selector."""

    values = _direct_v11_profile().model_dump(
        mode="json",
        warnings=False,
        exclude={"profile_fingerprint_sha256", "raster_text_profile_revision"},
    )
    values.update(
        {
            "schema_version": "10.0",
            "engine_kind": "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_grid2",
            "extractor_revision": "direct-paddle-mobile-test-v10",
        }
    )
    fingerprint = raster_text_runtime.raster_text_profile_fingerprint_sha256(values)
    return RasterTextOfflineProfileV10.model_validate(
        {
            **values,
            "profile_fingerprint_sha256": fingerprint,
            "raster_text_profile_revision": f"synthetic-raster-text-v1:{fingerprint[:24]}",
        }
    )


def _direct_v12_profile() -> RasterTextOfflineProfileV12:
    """Build the source-attested trained recognizer activation binding."""

    values = _direct_v2_profile().model_dump(
        mode="json",
        warnings=False,
        exclude={"profile_fingerprint_sha256", "raster_text_profile_revision"},
    )
    combined_manifest_sha256 = "8" * 64
    combined_bundle = f"raster-text-trained-paddle-bundles-v1/releases/{combined_manifest_sha256}"
    values.update(
        {
            "schema_version": "12.0",
            "engine_kind": (
                "direct_paddle_source_attested_text_harmonic_recall_native960_lowbox_strip2"
            ),
            "extractor_revision": "direct-paddle-source-attested-test-v12",
            "combined_model_bundle_root_relative_path": combined_bundle,
            "combined_model_bundle_manifest_sha256": combined_manifest_sha256,
            "source_detector_bundle_manifest_sha256": "9" * 64,
            "source_detector_model_inventory_sha256": "a" * 64,
            "source_detector_model_snapshot_sha256": "b" * 64,
            "source_recognizer_release_sha256": "c" * 64,
            "source_recognizer_receipt_sha256": "c" * 64,
            "source_recognizer_training_intent_sha256": "d" * 64,
            "source_recognizer_character_dictionary_sha256": "e" * 64,
            "source_recognizer_package_inventory_sha256": "f" * 64,
            "text_detection_box_threshold": 0.3,
            "local_model_root": f"{combined_bundle}/models",
            "model_inventory_path": (
                f"{combined_bundle}/inventories/direct-paddle-trained-models.json"
            ),
            "model_inventory_sha256": "0" * 64,
            "text_detection_model_name": "PP-OCRv5_mobile_det",
            "text_recognition_model_name": "source_attested_tiny_crnn_bigru_ctc_v1",
        }
    )
    fingerprint = raster_text_runtime.raster_text_profile_fingerprint_sha256(values)
    return RasterTextOfflineProfileV12.model_validate(
        {
            **values,
            "profile_fingerprint_sha256": fingerprint,
            "raster_text_profile_revision": f"synthetic-raster-text-v1:{fingerprint[:24]}",
        }
    )


def _direct_v13_profile() -> RasterTextOfflineProfileV13:
    """Build the public-mobile semantic-repair activation binding."""

    values = _direct_v3_profile().model_dump(
        mode="json",
        warnings=False,
        exclude={"profile_fingerprint_sha256", "raster_text_profile_revision"},
    )
    values.update(
        {
            "schema_version": "13.0",
            "engine_kind": (
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_semantic_repair"
            ),
            "extractor_revision": "direct-paddle-mobile-test-v13-semantic-repair",
            "text_detection_box_threshold": 0.3,
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


def _direct_v14_profile() -> RasterTextOfflineProfileV14:
    """Build the fresh-holdout V14 semantic-repair activation binding."""

    values = _direct_v3_profile().model_dump(
        mode="json",
        warnings=False,
        exclude={"profile_fingerprint_sha256", "raster_text_profile_revision"},
    )
    values.update(
        {
            "schema_version": "14.0",
            "engine_kind": (
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_"
                "semantic_repair_v2"
            ),
            "extractor_revision": "direct-paddle-mobile-test-v14-semantic-repair-v2",
            "text_detection_box_threshold": 0.3,
        }
    )
    fingerprint = raster_text_runtime.raster_text_profile_fingerprint_sha256(values)
    return RasterTextOfflineProfileV14.model_validate(
        {
            **values,
            "profile_fingerprint_sha256": fingerprint,
            "raster_text_profile_revision": f"synthetic-raster-text-v1:{fingerprint[:24]}",
        }
    )


def _direct_v15_profile() -> RasterTextOfflineProfileV15:
    """Build the staged-holdout V15 oneDNN activation binding."""

    values = _direct_v3_profile().model_dump(
        mode="json",
        warnings=False,
        exclude={"profile_fingerprint_sha256", "raster_text_profile_revision"},
    )
    values.update(
        {
            "schema_version": "15.0",
            "engine_kind": (
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_"
                "semantic_repair_v2_onednn"
            ),
            "extractor_revision": "direct-paddle-mobile-test-v15-onednn",
            "text_detection_box_threshold": 0.3,
            "enable_mkldnn": True,
            "mkldnn_cache_capacity": 10,
            "recognition_batch_size": 8,
            "opencv_threads": 1,
            "warmup_shape_policy": "detector-736x960-recognizer-48x320-v1",
            "onednn_model_policy": "recognizer-only-v1",
        }
    )
    fingerprint = raster_text_runtime.raster_text_profile_fingerprint_sha256(values)
    return RasterTextOfflineProfileV15.model_validate(
        {
            **values,
            "profile_fingerprint_sha256": fingerprint,
            "raster_text_profile_revision": f"synthetic-raster-text-v1:{fingerprint[:24]}",
        }
    )


def test_v2_direct_profile_uses_its_own_parser_and_child_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile = _direct_v2_profile()
    raw = raster_text_activation._canonical_json(  # noqa: SLF001 - writer encoding
        profile.model_dump(mode="json", warnings=False)
    )
    assert parse_raster_text_offline_profile(raw) == profile

    monkeypatch.setattr(
        raster_text_runtime,
        "_isolated_v2_sidecar_environment",
        lambda **_kwargs: {"CUSTOMS_AGENT_PPSTRUCTURE_CONFIG_PATH": "must-be-removed"},
    )
    environment = raster_text_runtime._sidecar_environment(  # noqa: SLF001 - contract boundary
        interpreter=tmp_path / "python.exe",
        module_root=tmp_path / "module-root",
        runtime_home=tmp_path,
        configuration_path=tmp_path / "profile.json",
        configuration_sha256="8" * 64,
        profile=profile,
        model_root=tmp_path / "models",
        model_inventory=tmp_path / "models.json",
        sidecar_inventory=tmp_path / "packages.json",
    )
    assert "CUSTOMS_AGENT_PPSTRUCTURE_CONFIG_PATH" not in environment
    assert environment["CUSTOMS_AGENT_RASTER_TEXT_DIRECT_PADDLE_REQUIRED_PACKAGES"] == (
        '["numpy","opencv-contrib-python","paddlepaddle","pyclipper"]'
    )

    # The V2 immutable material itself, not just the provider's runtime
    # clamp, must refuse a longer customer OCR budget.
    invalid = profile.model_dump(mode="json", warnings=False)
    invalid["sidecar_timeout_seconds"] = 31
    fingerprint_input = {
        key: value
        for key, value in invalid.items()
        if key not in {"profile_fingerprint_sha256", "raster_text_profile_revision"}
    }
    fingerprint = raster_text_runtime.raster_text_profile_fingerprint_sha256(fingerprint_input)
    invalid["profile_fingerprint_sha256"] = fingerprint
    invalid["raster_text_profile_revision"] = f"synthetic-raster-text-v1:{fingerprint[:24]}"
    with pytest.raises(ValueError):
        RasterTextOfflineProfileV2.model_validate(invalid)
    assert "CUSTOMS_AGENT_RASTER_TEXT_CONFIG_PATH" not in environment


def test_pointer_and_profile_reject_missing_default_discriminators(
    tmp_path: Path,
) -> None:
    """Legacy JSON may not become active through Pydantic default filling."""

    fixture = _cached_text_runtime_materials(tmp_path)
    pointer_value = json.loads(fixture.pointer_raw)
    pointer_value.pop("activation_kind")
    legacy_pointer = _canonical_bytes(pointer_value)
    fixture.pointer_path.write_bytes(legacy_pointer)
    with pytest.raises(RasterTextRuntimeError, match="activation_invalid"):
        parse_raster_text_activation_pointer(legacy_pointer)
    assert fixture.pointer_path.read_bytes() == legacy_pointer

    profile_value = json.loads(fixture.configuration_path.read_bytes())
    profile_value.pop("engine_kind")
    legacy_profile = raster_text_activation._canonical_json(  # noqa: SLF001
        profile_value
    )
    with pytest.raises(RasterTextRuntimeError, match="configuration_invalid"):
        raster_text_runtime._load_profile_materials(  # noqa: SLF001 - loader boundary
            root=fixture.root,
            configuration_path=fixture.configuration_path,
            configuration_raw=legacy_profile,
            configuration_sha256=hashlib.sha256(legacy_profile).hexdigest(),
        )


def _result(text: str) -> RasterTextExtractionResultV1:
    pages = [RasterTextPageV1(page_number=1, text=text, confidence=0.9)]
    return RasterTextExtractionResultV1(
        document_asset_id="asset-1",
        extraction_request_id="request-1",
        raster_text_profile_revision="synthetic-raster-text-v1:0123456789abcdef01234567",
        status="preview_ready",
        media_type="image/png",
        byte_size=1,
        source_sha256=_SHA,
        canonical_text=canonical_raster_text(pages),
        canonical_text_sha256=canonical_raster_text_sha256(pages),
        extractor_revision="direct-text-test-v1",
        pages=pages,
        warnings=[],
        needs_review_reasons=[],
    )


def test_candidate_observation_rejects_elapsed_over_warm_request_budget() -> None:
    with pytest.raises(ValueError):
        raster_text_activation.RasterTextCandidateObservationV1(
            asset_id="asset-1",
            profile_id="png-216",
            elapsed_seconds=30.001,
            candidate=_result("TEXT"),
        )


def test_candidate_runner_reuses_the_resolver_parent_attestation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The session must not repeat the resolver's just-finished full-tree scan."""

    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    revision = "synthetic-raster-text-v1:0123456789abcdef01234567"
    starts: list[bool] = []

    class Session:
        def __enter__(self) -> object:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    class Provider:
        def open_evaluation_session(self, *, parent_materials_verified: bool) -> Session:
            starts.append(parent_materials_verified)
            return Session()

    profile = SimpleNamespace(
        raster_text_profile_revision=revision,
        extractor_revision="direct-text-test-v1",
    )
    resolution = SimpleNamespace(
        status="available",
        engine=SimpleNamespace(input_adapter=object(), provider=Provider()),
        profile=profile,
        runtime_configuration_sha256="b" * 64,
    )
    asset = SimpleNamespace(
        asset_id="asset-1",
        profile_id="png-216",
        media_type="image/png",
        image_sha256=_SHA,
        gold_sha256="d" * 64,
        byte_size=1,
    )
    candidate = RasterTextExtractionResultV1(
        document_asset_id="asset-1",
        extraction_request_id=f"raster-text-png-216-{_SHA[:24]}",
        raster_text_profile_revision=revision,
        status="preview_ready",
        media_type="image/png",
        byte_size=1,
        source_sha256=_SHA,
        canonical_text="TEXT",
        canonical_text_sha256=hashlib.sha256(b"TEXT").hexdigest(),
        extractor_revision="direct-text-test-v1",
        pages=[RasterTextPageV1(page_number=1, text="TEXT", confidence=0.9)],
        warnings=[],
        needs_review_reasons=[],
    )
    ticks = iter((1.0, 2.0))

    monkeypatch.setattr(
        raster_text_activation,
        "_load_dataset_receipt",
        lambda *_args, **_kwargs: (SimpleNamespace(pdfium_version="4.30.0"), "a" * 64),
    )
    monkeypatch.setattr(
        raster_text_activation,
        "write_source_attested_raster_text_gold_receipt",
        lambda **_kwargs: ("source.json", "c" * 64),
    )
    monkeypatch.setattr(
        raster_text_activation,
        "resolve_paddleocr3_raster_text_candidate",
        lambda **_kwargs: resolution,
    )
    monkeypatch.setattr(raster_text_activation, "_scope_assets", lambda *_args: [asset])
    monkeypatch.setattr(
        raster_text_activation, "_verified_asset", lambda *_args, **_kwargs: b"image"
    )
    monkeypatch.setattr(
        raster_text_activation, "_raster_text_input_for_asset", lambda **_kwargs: object()
    )
    monkeypatch.setattr(
        raster_text_activation,
        "inspect_raster_document_structure",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        raster_text_activation,
        "extract_raster_text_preview",
        lambda *_args, **_kwargs: candidate,
    )
    monkeypatch.setattr(raster_text_activation.time, "monotonic", lambda: next(ticks))

    raster_text_activation.run_paddleocr3_synthetic_raster_text_candidates(
        document_preview_root=root,
        runtime_configuration_relative_path="profiles/profile.json",
        synthetic_raster_receipt_relative_path="synthetic-raster-v1/receipt.json",
        scope="calibration",
    )

    assert starts == [True]


def test_text_gold_uses_pdfium_visual_geometry_order_not_content_stream_order() -> None:
    box_top = BoundingBoxV1(x=0.1, y=0.1, width=0.2, height=0.03)
    box_bottom = BoundingBoxV1(x=0.1, y=0.8, width=0.2, height=0.03)
    gold = RasterGoldArtifactV1(
        asset_id="asset-1",
        family_id="family-1",
        document_type="commercial_invoice",
        source_pdf_sha256=_SHA,
        profile_id="png-216",
        # The final PDF emitted the lower line first.  Raster OCR observes
        # visual order, so a source-only text Gold must not preserve this
        # content-stream ordinal.
        spans=[
            RasterGoldSpanV1(
                asset_id="asset-1",
                profile_id="png-216",
                page_number=1,
                span_id="bottom",
                text="BOTTOM",
                bbox=box_bottom,
                reading_order=1,
                source_pdf_sha256=_SHA,
            ),
            RasterGoldSpanV1(
                asset_id="asset-1",
                profile_id="png-216",
                page_number=1,
                span_id="top",
                text="TOP",
                bbox=box_top,
                reading_order=2,
                source_pdf_sha256=_SHA,
            ),
        ],
        fields=[
            RasterGoldFieldV1(
                asset_id="asset-1",
                profile_id="png-216",
                field_id="invoice_number",
                value="TOP",
                bbox=box_top,
                evidence_span_ids=["top"],
                source_pdf_sha256=_SHA,
            )
        ],
        table_cells=[],
    )
    assert source_only_text_gold_from_raster_gold(gold).canonical_text == "TOP\nBOTTOM"


def test_critical_values_require_exact_non_overlapping_occurrences() -> None:
    gold = RasterTextGoldArtifactV1(
        asset_id="asset-1",
        family_id="family-1",
        document_type="commercial_invoice",
        profile_id="png-216",
        source_pdf_sha256=_SHA,
        source_raster_gold_sha256=_SHA,
        canonical_text="AB AB",
        critical_values=["AB", "AB"],
    )
    counts = evaluate_synthetic_raster_text_case(
        RasterTextEvaluationCaseV1(
            asset_id="asset-1",
            document_type="commercial_invoice",
            profile_id="png-216",
            image_sha256=_SHA,
            gold=gold,
            candidate=_result("AB ABX"),
        )
    )
    assert (counts.critical_matches, counts.critical_total) == (1, 2)


def test_sidecar_prediction_order_uses_the_same_visual_row_banding_as_gold() -> None:
    # Raw detector order is lower-row, upper-right, upper-left.  The direct
    # OCR sidecar must emit the same physical row/left-to-right order as the
    # source-PDFium Gold compiler rather than preserving that batch order.
    lines = _prediction_lines(
        [
            {
                "rec_texts": ["BOTTOM", "RIGHT", "LEFT"],
                "rec_scores": [0.9, 0.9, 0.9],
                "rec_polys": [
                    [[0, 90], [10, 90], [10, 100], [0, 100]],
                    [[50, 10], [60, 10], [60, 20], [50, 20]],
                    [[0, 12], [10, 12], [10, 22], [0, 22]],
                ],
            }
        ]
    )
    assert [text for text, _score in lines] == ["LEFT", "RIGHT", "BOTTOM"]


def test_auto_activation_missing_bundle_is_content_free_and_keeps_pointer_absent(
    tmp_path: Path,
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    with pytest.raises(SyntheticRasterTextActivationError) as captured:
        auto_activate_synthetic_raster_text_v1(document_preview_root=root)
    assert captured.value.code == "offline_bundle_unavailable"
    assert not (root / "approvals" / "raster-text-active.json").exists()


def test_auto_activation_missing_public_mobile_bundle_keeps_existing_pointer_byte_identical(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    pointer = root / "approvals" / "raster-text-active.json"
    pointer.parent.mkdir(parents=True)
    prior = b'{"previous":"pointer"}\n'
    pointer.write_bytes(prior)
    sidecar = SimpleNamespace(document_preview_root=root)
    monkeypatch.setattr(
        raster_text_activation,
        "materialize_verified_raster_text_direct_paddle_bundle_v1",
        lambda **_kwargs: sidecar,
    )

    def missing_mobile(**_kwargs: object) -> None:
        raise raster_text_activation.RasterTextMobilePaddleBundleError(
            "mobile_paddle_bundle_path_invalid"
        )

    monkeypatch.setattr(
        raster_text_activation,
        "materialize_verified_raster_text_direct_mobile_paddle_bundle_v1",
        missing_mobile,
    )

    with pytest.raises(SyntheticRasterTextActivationError) as captured:
        auto_activate_synthetic_raster_text_v1(document_preview_root=root)

    assert captured.value.code == "offline_bundle_unavailable"
    assert pointer.read_bytes() == prior


def test_auto_activation_v15_orders_staged_evidence_before_schema5_pointer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    # A completed output follows the smoke-first reuse path.  The runner is
    # mocked here; this test asserts that the expensive compiler replay is
    # deferred until the smoke screen has passed.
    (root / "synthetic-raster-v1").mkdir()
    bundle = SimpleNamespace(
        document_preview_root=root,
        manifest_sha256="a" * 64,
        bundle_root_relative_path="raster-text-direct-bundles-v1/releases/" + "a" * 64,
    )
    mobile_bundle = SimpleNamespace(
        document_preview_root=root,
        manifest_sha256="b" * 64,
        bundle_root_relative_path=("raster-text-direct-mobile-bundles-v1/releases/" + "b" * 64),
    )
    inputs = SimpleNamespace(
        build_inputs_relative_path="synthetic-raster-build/inputs.json",
        renderer_configuration=SimpleNamespace(pdfium_version="4.30.0"),
    )
    calibration_profile = SimpleNamespace(
        runtime_configuration_relative_path="profiles/calibration.json",
        runtime_configuration_sha256="b" * 64,
    )
    final_profile_info = SimpleNamespace(
        runtime_configuration_relative_path="profiles/final.json",
        runtime_configuration_sha256="c" * 64,
    )
    final_profile = _direct_v15_profile()
    calls: list[str] = []

    monkeypatch.setattr(
        raster_text_activation,
        "materialize_verified_raster_text_direct_paddle_bundle_v1",
        lambda **_kwargs: bundle,
    )
    monkeypatch.setattr(
        raster_text_activation,
        "load_verified_raster_text_direct_paddle_bundle_v1",
        lambda **_kwargs: calls.append("rehash:sidecar") or bundle,
    )
    monkeypatch.setattr(
        raster_text_activation,
        "materialize_verified_raster_text_direct_mobile_paddle_bundle_v1",
        lambda **_kwargs: mobile_bundle,
    )
    monkeypatch.setattr(
        raster_text_activation,
        "load_verified_raster_text_direct_mobile_paddle_bundle_v1",
        lambda **_kwargs: calls.append("rehash:mobile") or mobile_bundle,
    )
    monkeypatch.setattr(
        raster_text_activation,
        "ensure_fixed_synthetic_raster_build_inputs",
        lambda **_kwargs: inputs,
    )
    monkeypatch.setattr(
        raster_text_activation,
        "compile_synthetic_raster_dataset",
        lambda **_kwargs: calls.append("full-replay"),
    )
    profile_calls = iter((calibration_profile, final_profile_info))
    prepared: list[tuple[object, object, float, int, int, int]] = []

    def prepare_v15(sidecar: object, mobile: object, **kwargs: object) -> SimpleNamespace:
        prepared.append(
            (
                sidecar,
                mobile,
                cast(float, kwargs["minimum_page_confidence"]),
                cast(int, kwargs["cpu_threads"]),
                cast(int, kwargs["recognition_batch_size"]),
                cast(int, kwargs["opencv_threads"]),
            )
        )
        return next(profile_calls)

    monkeypatch.setattr(
        raster_text_activation,
        "prepare_raster_text_direct_mobile_runtime_profile_v15",
        prepare_v15,
    )
    monkeypatch.setattr(
        raster_text_activation,
        "prepare_raster_text_direct_mobile_runtime_profile_v13",
        lambda *_args, **_kwargs: pytest.fail("V15 activation must not prepare V13"),
        raising=False,
    )
    monkeypatch.setattr(raster_text_activation, "_read", lambda *_args, **_kwargs: b"{}")
    monkeypatch.setattr(
        raster_text_activation,
        "parse_raster_text_offline_profile",
        lambda _raw: final_profile,
    )

    def candidate_run(**kwargs: object) -> SimpleNamespace:
        scope = cast(str, kwargs["scope"])
        config = cast(str, kwargs["runtime_configuration_relative_path"])
        if scope == "smoke" and config == "profiles/calibration.json":
            assert kwargs["preflight_smoke_only"] is True
            assert kwargs["build_inputs"] is inputs
        elif scope == "fresh-calibration":
            assert kwargs["fresh_projection_kind"] == "calibration-10-family-v1"
        elif scope == "fresh-sealed":
            if kwargs.get("parent_stage1_decision_relative_path") is None:
                assert kwargs["fresh_projection_kind"] == "sealed-stage1-50-family-v1"
            else:
                assert kwargs["fresh_projection_kind"] == "sealed-stage2-100-family-v1"
                assert (
                    kwargs["parent_stage1_decision_relative_path"]
                    == "stages/decision.json"
                )
        calls.append(f"candidate:{scope}:{config}")
        return SimpleNamespace(
            source_gold_receipt_relative_path=f"gold/{scope}-{config}",
            source_gold_receipt_sha256="e" * 64,
            candidate_execution_relative_path=f"execution/{scope}-{config}",
            candidate_execution_sha256="f" * 64,
            latency_p95_seconds=9.0,
        )

    monkeypatch.setattr(
        raster_text_activation, "run_paddleocr3_synthetic_raster_text_candidates", candidate_run
    )
    monkeypatch.setattr(
        raster_text_activation,
        "write_fresh_raster_text_scope_selection_receipt",
        lambda **_kwargs: None,
    )

    def smoke_screen(**kwargs: object) -> tuple[str, str, bool]:
        candidate_path = cast(str, kwargs["candidate_execution_relative_path"])
        calls.append(f"screen:{candidate_path}")
        return "screen/receipt.json", "a" * 64, True

    monkeypatch.setattr(
        raster_text_activation,
        "write_raster_semantic_text_smoke_quality_screen_v3",
        smoke_screen,
    )
    monkeypatch.setattr(
        raster_text_activation,
        "verify_source_attested_fresh_raster_text_evidence",
        lambda **_kwargs: (None, None, None, []),
    )

    def calibrated_floor(_cases: object, **kwargs: object) -> float:
        assert kwargs == {
            "metric_revision": raster_text_evaluation.RASTER_TEXT_SEMANTIC_METRIC_REVISION_V3
        }
        return 0.5

    monkeypatch.setattr(raster_text_activation, "_calibrated_floor", calibrated_floor)

    def warm_smoke(**kwargs: object) -> tuple[str, str]:
        calls.append(f"warm:{kwargs['runtime_configuration_relative_path']}")
        return "warm/receipt.json", "b" * 64

    monkeypatch.setattr(raster_text_activation, "run_warm_service_raster_text_smoke", warm_smoke)
    monkeypatch.setattr(
        raster_text_activation,
        "_require_auto_activation_warm_latency",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        raster_text_activation,
        "write_raster_semantic_text_fresh_calibration_receipt_v5",
        lambda **_kwargs: ("calibration/receipt.json", "c" * 64),
    )
    monkeypatch.setattr(
        raster_text_activation,
        "write_fresh_raster_text_sealed_stage1_decision",
        lambda **_kwargs: ("stages/decision.json", "9" * 64, "continue"),
    )

    def sealed_write(**kwargs: object) -> tuple[str, str]:
        calls.append(f"sealed:{kwargs['warm_service_smoke_receipt_relative_path']}")
        assert kwargs["stage1_decision_receipt_relative_path"] == "stages/decision.json"
        return "sealed/result.json", "d" * 64

    monkeypatch.setattr(
        raster_text_activation,
        "write_raster_semantic_text_fresh_sealed_evaluation_v5",
        sealed_write,
    )

    def rehash_dataset(*_args: object, **_kwargs: object) -> tuple[SimpleNamespace, str]:
        calls.append("rehash:dataset")
        return (
            SimpleNamespace(build_inputs_relative_path=inputs.build_inputs_relative_path),
            "a" * 64,
        )

    monkeypatch.setattr(
        raster_text_activation,
        "_load_fresh_dataset_receipts",
        lambda *_args, **_kwargs: (*rehash_dataset(), None, "f" * 64),
    )

    def pointer_write(**kwargs: object) -> tuple[str, SimpleNamespace]:
        calls.append(f"pointer:{kwargs['sealed_evaluation_relative_path']}")
        return "e" * 64, SimpleNamespace(
            raster_text_profile_revision=final_profile.raster_text_profile_revision
        )

    monkeypatch.setattr(
        raster_text_activation, "write_source_attested_synthetic_raster_text_pointer", pointer_write
    )

    for forbidden_writer in (
        "write_raster_text_smoke_quality_screen",
        "write_raster_text_calibration_receipt",
        "write_raster_text_sealed_evaluation",
        "write_raster_semantic_text_smoke_quality_screen_v2",
        "write_raster_semantic_text_calibration_receipt_v2",
        "write_raster_semantic_text_sealed_evaluation_v2",
        "write_raster_semantic_text_calibration_receipt_v3",
        "write_raster_semantic_text_sealed_evaluation_v3",
    ):
        monkeypatch.setattr(
            raster_text_activation,
            forbidden_writer,
            lambda **_kwargs: pytest.fail("V15 activation must use fresh semantic metric V3"),
        )

    result = auto_activate_synthetic_raster_text_v1(document_preview_root=root)

    assert result.activation_pointer_sha256 == "e" * 64
    assert prepared == [
        (bundle, mobile_bundle, 0.0, 6, 8, 1),
        (bundle, mobile_bundle, 0.5, 6, 8, 1),
    ]
    assert hasattr(raster_text_activation, "prepare_raster_text_direct_mobile_runtime_profile_v15")
    for removed_v12_entrypoint in (
        "_discover_verified_trained_recognizer_release",
        "prepare_raster_text_trained_runtime_profile_v12",
        "materialize_verified_raster_text_trained_paddle_bundle_v1",
        "load_verified_raster_text_trained_paddle_bundle_v1",
    ):
        assert not hasattr(raster_text_activation, removed_v12_entrypoint)
    assert calls == [
        "candidate:smoke:profiles/calibration.json",
        "screen:execution/smoke-profiles/calibration.json",
        "full-replay",
        "candidate:fresh-calibration:profiles/calibration.json",
        "candidate:smoke:profiles/final.json",
        "screen:execution/smoke-profiles/final.json",
        "warm:profiles/final.json",
        "candidate:fresh-sealed:profiles/final.json",
        "candidate:fresh-sealed:profiles/final.json",
        "sealed:warm/receipt.json",
        "rehash:sidecar",
        "rehash:mobile",
        "rehash:dataset",
        "pointer:sealed/result.json",
    ]


def test_auto_activation_preflight_smoke_gate_failure_stops_before_calibration_and_keeps_pointer(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The zero-floor V15 semantic/latency screen is an immutable rejection."""

    root = tmp_path / "_local" / "data" / "document-preview"
    pointer_path = root / "approvals" / "raster-text-active.json"
    pointer_path.parent.mkdir(parents=True)
    prior_pointer = b'{"previous":"pointer"}\n'
    pointer_path.write_bytes(prior_pointer)
    bundle = SimpleNamespace(
        document_preview_root=root,
        manifest_sha256="a" * 64,
        bundle_root_relative_path="raster-text-direct-bundles-v1/releases/" + "a" * 64,
    )
    mobile_bundle = SimpleNamespace(
        document_preview_root=root,
        manifest_sha256="b" * 64,
        bundle_root_relative_path=("raster-text-direct-mobile-bundles-v1/releases/" + "b" * 64),
    )
    inputs = SimpleNamespace(
        build_inputs_relative_path="synthetic-raster-build/inputs.json",
        renderer_configuration=SimpleNamespace(pdfium_version="4.30.0"),
    )
    calibration_profile = SimpleNamespace(
        runtime_configuration_relative_path="profiles/calibration.json",
        runtime_configuration_sha256="c" * 64,
    )
    prepared: list[float] = []
    candidate_scopes: list[str] = []

    monkeypatch.setattr(
        raster_text_activation,
        "materialize_verified_raster_text_direct_paddle_bundle_v1",
        lambda **_kwargs: bundle,
    )
    monkeypatch.setattr(
        raster_text_activation,
        "materialize_verified_raster_text_direct_mobile_paddle_bundle_v1",
        lambda **_kwargs: mobile_bundle,
    )
    monkeypatch.setattr(
        raster_text_activation,
        "ensure_fixed_synthetic_raster_build_inputs",
        lambda **_kwargs: inputs,
    )
    monkeypatch.setattr(
        raster_text_activation, "compile_synthetic_raster_dataset", lambda **_kwargs: None
    )

    def prepare_v15(_sidecar: object, _mobile: object, **kwargs: object) -> SimpleNamespace:
        prepared.append(cast(float, kwargs["minimum_page_confidence"]))
        assert kwargs["cpu_threads"] == 6
        assert kwargs["recognition_batch_size"] == 8
        assert kwargs["opencv_threads"] == 1
        return calibration_profile

    monkeypatch.setattr(
        raster_text_activation,
        "prepare_raster_text_direct_mobile_runtime_profile_v15",
        prepare_v15,
    )

    def candidate_run(**kwargs: object) -> SimpleNamespace:
        scope = cast(str, kwargs["scope"])
        candidate_scopes.append(scope)
        return SimpleNamespace(
            source_gold_receipt_relative_path=f"gold/{scope}.json",
            candidate_execution_relative_path=f"execution/{scope}.json",
            latency_p95_seconds=9.0,
        )

    monkeypatch.setattr(
        raster_text_activation,
        "run_paddleocr3_synthetic_raster_text_candidates",
        candidate_run,
    )
    monkeypatch.setattr(
        raster_text_activation,
        "write_raster_semantic_text_smoke_quality_screen_v3",
        lambda **_kwargs: ("screen/preflight.json", "d" * 64, False),
    )
    monkeypatch.setattr(
        raster_text_activation,
        "write_source_attested_synthetic_raster_text_pointer",
        lambda **_kwargs: pytest.fail("preflight failure must not write the pointer"),
    )

    with pytest.raises(SyntheticRasterTextActivationError) as captured:
        auto_activate_synthetic_raster_text_v1(document_preview_root=root)

    assert captured.value.code == "raster_text_smoke_gate_failed"
    assert prepared == [0.0]
    assert candidate_scopes == ["smoke"]
    assert pointer_path.read_bytes() == prior_pointer


def test_auto_activation_final_smoke_gate_failure_stops_before_warm_or_sealed(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The final V15 semantic/latency smoke must pass independently of preflight."""

    root = tmp_path / "_local" / "data" / "document-preview"
    pointer_path = root / "approvals" / "raster-text-active.json"
    pointer_path.parent.mkdir(parents=True)
    prior_pointer = b'{"previous":"pointer"}\n'
    pointer_path.write_bytes(prior_pointer)
    bundle = SimpleNamespace(
        document_preview_root=root,
        manifest_sha256="a" * 64,
        bundle_root_relative_path="raster-text-direct-bundles-v1/releases/" + "a" * 64,
    )
    mobile_bundle = SimpleNamespace(
        document_preview_root=root,
        manifest_sha256="b" * 64,
        bundle_root_relative_path=("raster-text-direct-mobile-bundles-v1/releases/" + "b" * 64),
    )
    inputs = SimpleNamespace(
        build_inputs_relative_path="synthetic-raster-build/inputs.json",
        renderer_configuration=SimpleNamespace(pdfium_version="4.30.0"),
    )
    profile_calls = iter(
        (
            SimpleNamespace(
                runtime_configuration_relative_path="profiles/calibration.json",
                runtime_configuration_sha256="c" * 64,
            ),
            SimpleNamespace(
                runtime_configuration_relative_path="profiles/final.json",
                runtime_configuration_sha256="d" * 64,
            ),
        )
    )
    final_profile = _direct_v15_profile()
    prepared: list[float] = []
    candidate_scopes: list[str] = []
    screened_paths: list[str] = []

    monkeypatch.setattr(
        raster_text_activation,
        "materialize_verified_raster_text_direct_paddle_bundle_v1",
        lambda **_kwargs: bundle,
    )
    monkeypatch.setattr(
        raster_text_activation,
        "materialize_verified_raster_text_direct_mobile_paddle_bundle_v1",
        lambda **_kwargs: mobile_bundle,
    )
    monkeypatch.setattr(
        raster_text_activation,
        "ensure_fixed_synthetic_raster_build_inputs",
        lambda **_kwargs: inputs,
    )
    monkeypatch.setattr(
        raster_text_activation, "compile_synthetic_raster_dataset", lambda **_kwargs: None
    )

    def prepare_v15(_sidecar: object, _mobile: object, **kwargs: object) -> SimpleNamespace:
        prepared.append(cast(float, kwargs["minimum_page_confidence"]))
        assert kwargs["cpu_threads"] == 6
        assert kwargs["recognition_batch_size"] == 8
        assert kwargs["opencv_threads"] == 1
        return next(profile_calls)

    monkeypatch.setattr(
        raster_text_activation,
        "prepare_raster_text_direct_mobile_runtime_profile_v15",
        prepare_v15,
    )
    monkeypatch.setattr(raster_text_activation, "_read", lambda *_args, **_kwargs: b"{}")
    monkeypatch.setattr(
        raster_text_activation,
        "parse_raster_text_offline_profile",
        lambda _raw: final_profile,
    )

    def candidate_run(**kwargs: object) -> SimpleNamespace:
        scope = cast(str, kwargs["scope"])
        configuration = cast(str, kwargs["runtime_configuration_relative_path"])
        candidate_scopes.append(f"{scope}:{configuration}")
        return SimpleNamespace(
            source_gold_receipt_relative_path=f"gold/{scope}-{configuration}",
            candidate_execution_relative_path=f"execution/{scope}-{configuration}",
            latency_p95_seconds=9.0,
        )

    monkeypatch.setattr(
        raster_text_activation,
        "run_paddleocr3_synthetic_raster_text_candidates",
        candidate_run,
    )
    monkeypatch.setattr(
        raster_text_activation,
        "write_fresh_raster_text_scope_selection_receipt",
        lambda **_kwargs: None,
    )

    def smoke_screen(**kwargs: object) -> tuple[str, str, bool]:
        candidate_path = cast(str, kwargs["candidate_execution_relative_path"])
        screened_paths.append(candidate_path)
        return "screen/receipt.json", "e" * 64, len(screened_paths) == 1

    monkeypatch.setattr(
        raster_text_activation,
        "write_raster_semantic_text_smoke_quality_screen_v3",
        smoke_screen,
    )
    monkeypatch.setattr(
        raster_text_activation,
        "verify_source_attested_fresh_raster_text_evidence",
        lambda **_kwargs: (None, None, None, []),
    )
    monkeypatch.setattr(
        raster_text_activation,
        "_calibrated_floor",
        lambda _cases, **_kwargs: 0.5,
    )
    monkeypatch.setattr(
        raster_text_activation,
        "run_warm_service_raster_text_smoke",
        lambda **_kwargs: pytest.fail("final smoke failure must stop before warm smoke"),
    )
    monkeypatch.setattr(
        raster_text_activation,
        "write_raster_semantic_text_fresh_sealed_evaluation_v5",
        lambda **_kwargs: pytest.fail("final smoke failure must stop before sealed evaluation"),
    )
    monkeypatch.setattr(
        raster_text_activation,
        "write_source_attested_synthetic_raster_text_pointer",
        lambda **_kwargs: pytest.fail("final smoke failure must not write the pointer"),
    )

    with pytest.raises(SyntheticRasterTextActivationError) as captured:
        auto_activate_synthetic_raster_text_v1(document_preview_root=root)

    assert captured.value.code == "raster_text_smoke_gate_failed"
    assert prepared == [0.0, 0.5]
    assert candidate_scopes == [
        "smoke:profiles/calibration.json",
        "fresh-calibration:profiles/calibration.json",
        "smoke:profiles/final.json",
    ]
    assert screened_paths == [
        "execution/smoke-profiles/calibration.json",
        "execution/smoke-profiles/final.json",
    ]
    assert pointer_path.read_bytes() == prior_pointer


def test_auto_activation_sealed_gate_failure_keeps_existing_pointer_byte_identical(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A failed sealed Gate must not touch an already-active text pointer."""

    root = tmp_path / "_local" / "data" / "document-preview"
    pointer_path = root / "approvals" / "raster-text-active.json"
    pointer_path.parent.mkdir(parents=True)
    prior_pointer = b'{"previous":"pointer"}\n'
    pointer_path.write_bytes(prior_pointer)
    bundle = SimpleNamespace(
        document_preview_root=root,
        manifest_sha256="a" * 64,
        bundle_root_relative_path="raster-text-direct-bundles-v1/releases/" + "a" * 64,
    )
    mobile_bundle = SimpleNamespace(
        document_preview_root=root,
        manifest_sha256="b" * 64,
        bundle_root_relative_path=("raster-text-direct-mobile-bundles-v1/releases/" + "b" * 64),
    )
    inputs = SimpleNamespace(
        build_inputs_relative_path="synthetic-raster-build/inputs.json",
        renderer_configuration=SimpleNamespace(pdfium_version="4.30.0"),
    )
    calibration_profile = SimpleNamespace(
        runtime_configuration_relative_path="profiles/calibration.json",
        runtime_configuration_sha256="b" * 64,
    )
    final_profile_info = SimpleNamespace(
        runtime_configuration_relative_path="profiles/final.json",
        runtime_configuration_sha256="c" * 64,
    )
    final_profile = _direct_v15_profile()
    calls: list[str] = []
    pointer_writer_called = False

    monkeypatch.setattr(
        raster_text_activation,
        "materialize_verified_raster_text_direct_paddle_bundle_v1",
        lambda **_kwargs: bundle,
    )
    monkeypatch.setattr(
        raster_text_activation,
        "materialize_verified_raster_text_direct_mobile_paddle_bundle_v1",
        lambda **_kwargs: mobile_bundle,
    )
    monkeypatch.setattr(
        raster_text_activation,
        "ensure_fixed_synthetic_raster_build_inputs",
        lambda **_kwargs: inputs,
    )
    monkeypatch.setattr(
        raster_text_activation, "compile_synthetic_raster_dataset", lambda **_kwargs: None
    )
    profile_calls = iter((calibration_profile, final_profile_info))
    monkeypatch.setattr(
        raster_text_activation,
        "prepare_raster_text_direct_mobile_runtime_profile_v15",
        lambda *_args, **_kwargs: next(profile_calls),
    )
    monkeypatch.setattr(raster_text_activation, "_read", lambda *_args, **_kwargs: b"{}")
    monkeypatch.setattr(
        raster_text_activation,
        "parse_raster_text_offline_profile",
        lambda _raw: final_profile,
    )

    def candidate_run(**kwargs: object) -> SimpleNamespace:
        scope = cast(str, kwargs["scope"])
        calls.append(f"candidate:{scope}")
        return SimpleNamespace(
            source_gold_receipt_relative_path=f"gold/{scope}.json",
            candidate_execution_relative_path=f"execution/{scope}.json",
            latency_p95_seconds=9.0,
        )

    monkeypatch.setattr(
        raster_text_activation, "run_paddleocr3_synthetic_raster_text_candidates", candidate_run
    )
    monkeypatch.setattr(
        raster_text_activation,
        "write_fresh_raster_text_scope_selection_receipt",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        raster_text_activation,
        "write_raster_semantic_text_smoke_quality_screen_v3",
        lambda **_kwargs: ("screen/receipt.json", "a" * 64, True),
    )
    monkeypatch.setattr(
        raster_text_activation,
        "verify_source_attested_fresh_raster_text_evidence",
        lambda **_kwargs: (None, None, None, []),
    )
    monkeypatch.setattr(
        raster_text_activation,
        "_calibrated_floor",
        lambda _cases, **_kwargs: 0.5,
    )

    def warm_smoke(**kwargs: object) -> tuple[str, str]:
        calls.append(f"warm:{kwargs['runtime_configuration_relative_path']}")
        return "warm/receipt.json", "e" * 64

    monkeypatch.setattr(raster_text_activation, "run_warm_service_raster_text_smoke", warm_smoke)
    monkeypatch.setattr(
        raster_text_activation,
        "_require_auto_activation_warm_latency",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        raster_text_activation,
        "write_raster_semantic_text_fresh_calibration_receipt_v5",
        lambda **_kwargs: ("calibration/receipt.json", "f" * 64),
    )
    monkeypatch.setattr(
        raster_text_activation,
        "write_fresh_raster_text_sealed_stage1_decision",
        lambda **_kwargs: ("stages/decision.json", "9" * 64, "early_pass"),
    )

    def sealed_gate_failure(**kwargs: object) -> tuple[str, str]:
        calls.append(f"sealed:{kwargs['warm_service_smoke_receipt_relative_path']}")
        raise SyntheticRasterTextActivationError("sealed_raster_text_gate_failed")

    monkeypatch.setattr(
        raster_text_activation,
        "write_raster_semantic_text_fresh_sealed_evaluation_v5",
        sealed_gate_failure,
    )

    def pointer_writer(**_kwargs: object) -> tuple[str, SimpleNamespace]:
        nonlocal pointer_writer_called
        pointer_writer_called = True
        raise AssertionError("a failed sealed Gate must not invoke the pointer writer")

    monkeypatch.setattr(
        raster_text_activation,
        "write_source_attested_synthetic_raster_text_pointer",
        pointer_writer,
    )

    with pytest.raises(SyntheticRasterTextActivationError) as captured:
        auto_activate_synthetic_raster_text_v1(document_preview_root=root)

    assert captured.value.code == "sealed_raster_text_gate_failed"
    assert calls == [
        "candidate:smoke",
        "candidate:fresh-calibration",
        "candidate:smoke",
        "warm:profiles/final.json",
        "candidate:fresh-sealed",
        "sealed:warm/receipt.json",
    ]
    assert not pointer_writer_called
    assert pointer_path.read_bytes() == prior_pointer


def test_auto_activation_rejects_a_v12_final_profile_without_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The trained V12 candidate must never become an activation fallback."""

    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    sidecar_bundle = SimpleNamespace(
        document_preview_root=root,
        manifest_sha256="a" * 64,
        bundle_root_relative_path="raster-text-direct-bundles-v1/releases/" + "a" * 64,
    )
    mobile_bundle = SimpleNamespace(
        document_preview_root=root,
        manifest_sha256="b" * 64,
        bundle_root_relative_path=("raster-text-direct-mobile-bundles-v1/releases/" + "b" * 64),
    )
    inputs = SimpleNamespace(
        build_inputs_relative_path="synthetic-raster-build/inputs.json",
        renderer_configuration=SimpleNamespace(pdfium_version="4.30.0"),
    )
    profile_calls = iter(
        (
            SimpleNamespace(
                runtime_configuration_relative_path="profiles/calibration.json",
                runtime_configuration_sha256="c" * 64,
            ),
            SimpleNamespace(
                runtime_configuration_relative_path="profiles/final.json",
                runtime_configuration_sha256="d" * 64,
            ),
        )
    )
    candidate_scopes: list[str] = []

    monkeypatch.setattr(
        raster_text_activation,
        "materialize_verified_raster_text_direct_paddle_bundle_v1",
        lambda **_kwargs: sidecar_bundle,
    )
    monkeypatch.setattr(
        raster_text_activation,
        "materialize_verified_raster_text_direct_mobile_paddle_bundle_v1",
        lambda **_kwargs: mobile_bundle,
    )
    monkeypatch.setattr(
        raster_text_activation,
        "ensure_fixed_synthetic_raster_build_inputs",
        lambda **_kwargs: inputs,
    )
    monkeypatch.setattr(
        raster_text_activation, "compile_synthetic_raster_dataset", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        raster_text_activation,
        "prepare_raster_text_direct_mobile_runtime_profile_v15",
        lambda *_args, **_kwargs: next(profile_calls),
    )
    monkeypatch.setattr(raster_text_activation, "_read", lambda *_args, **_kwargs: b"{}")
    monkeypatch.setattr(
        raster_text_activation,
        "parse_raster_text_offline_profile",
        lambda _raw: _direct_v12_profile(),
    )

    def candidate_run(**kwargs: object) -> SimpleNamespace:
        scope = cast(str, kwargs["scope"])
        candidate_scopes.append(scope)
        return SimpleNamespace(
            source_gold_receipt_relative_path=f"gold/{scope}.json",
            candidate_execution_relative_path=f"execution/{scope}.json",
            latency_p95_seconds=9.0,
        )

    monkeypatch.setattr(
        raster_text_activation,
        "run_paddleocr3_synthetic_raster_text_candidates",
        candidate_run,
    )
    monkeypatch.setattr(
        raster_text_activation,
        "write_fresh_raster_text_scope_selection_receipt",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        raster_text_activation,
        "write_raster_semantic_text_smoke_quality_screen_v3",
        lambda **_kwargs: ("screen/receipt.json", "a" * 64, True),
    )
    monkeypatch.setattr(
        raster_text_activation,
        "verify_source_attested_fresh_raster_text_evidence",
        lambda **_kwargs: (None, None, None, []),
    )
    monkeypatch.setattr(
        raster_text_activation,
        "_calibrated_floor",
        lambda _cases, **_kwargs: 0.5,
    )

    with pytest.raises(SyntheticRasterTextActivationError) as captured:
        auto_activate_synthetic_raster_text_v1(document_preview_root=root)

    assert captured.value.code == "raster_text_profile_invalid"
    assert candidate_scopes == ["smoke", "fresh-calibration"]


def test_dataset_receipt_uses_the_compiler_sized_read_limit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The fixed 24,000-asset receipt is larger than compact evidence files."""

    observed: list[tuple[int, str]] = []

    def _fake_read(_root: Path, _relative: str, *, limit: int, code: str) -> bytes:
        observed.append((limit, code))
        return b"{}"

    monkeypatch.setattr(raster_text_activation, "_read", _fake_read)
    with pytest.raises(SyntheticRasterTextActivationError, match="dataset_receipt_invalid"):
        raster_text_activation._load_dataset_receipt(
            tmp_path,
            "synthetic-raster-v1/synthetic-raster-receipt.json",
            scope="smoke",
        )
    assert observed == [(32 * 1024 * 1024, "raster_text_dataset_receipt_invalid")]


def test_direct_text_profile_uses_the_fixed_six_core_budget() -> None:
    # The release template is already six-core.  The generated text profile
    # must retain that fixed, hash-bound setting rather than quietly use the
    # old two-thread default, which exceeds the serving deadline on CPU.
    assert _RASTER_TEXT_CPU_THREADS == 6


def test_auto_activation_v15_native_controls_are_single_fixed_defaults() -> None:
    assert raster_text_activation._AUTO_ACTIVATION_V15_CPU_THREADS == 6  # noqa: SLF001
    assert (  # noqa: SLF001
        raster_text_activation._AUTO_ACTIVATION_V15_RECOGNITION_BATCH_SIZE == 8
    )
    assert raster_text_activation._AUTO_ACTIVATION_V15_OPENCV_THREADS == 1  # noqa: SLF001
    assert (  # noqa: SLF001
        raster_text_activation._AUTO_ACTIVATION_LATENCY_P95_LIMIT_SECONDS == 10.0
    )


def test_auto_activation_candidate_latency_gate_is_fail_closed() -> None:
    base = dict(
        source_gold_receipt_relative_path="gold.json",
        source_gold_receipt_sha256="a" * 64,
        candidate_execution_relative_path="execution.json",
        candidate_execution_sha256="b" * 64,
        raster_text_profile_revision="synthetic-raster-text-v1:" + "c" * 24,
        runtime_configuration_sha256="d" * 64,
    )
    raster_text_activation._require_auto_activation_candidate_latency(  # noqa: SLF001
        raster_text_activation.RasterTextCandidateRunV1(
            **base,
            latency_p95_seconds=10.0,
        )
    )
    with pytest.raises(SyntheticRasterTextActivationError) as captured:
        raster_text_activation._require_auto_activation_candidate_latency(  # noqa: SLF001
            raster_text_activation.RasterTextCandidateRunV1(
                **base,
                latency_p95_seconds=10.0001,
            )
        )
    assert captured.value.code == "raster_text_latency_gate_failed"


def test_auto_activation_warm_latency_gate_reopens_bound_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile = _direct_v15_profile()
    monkeypatch.setattr(
        raster_text_activation,
        "_load_verified_final_warm_smoke",
        lambda **_kwargs: (SimpleNamespace(warm_latency_p95_seconds=10.0001), b"receipt"),
    )
    with pytest.raises(SyntheticRasterTextActivationError) as captured:
        raster_text_activation._require_auto_activation_warm_latency(  # noqa: SLF001
            root=tmp_path,
            relative_path="warm.json",
            profile=profile,
            runtime_configuration_sha256="e" * 64,
        )
    assert captured.value.code == "raster_text_latency_gate_failed"


def test_network_guard_keeps_socket_a_type_but_blocks_connections() -> None:
    with deny_network_connections():
        probe = socket.socket()
        try:
            assert isinstance(probe, socket.socket)
            with pytest.raises(Exception, match="network_forbidden"):
                probe.connect(("127.0.0.1", 9))
            with pytest.raises(Exception, match="network_forbidden"):
                socket.create_connection(("127.0.0.1", 9))
        finally:
            probe.close()


def test_activation_network_guard_blocks_dns_and_datagrams() -> None:
    with raster_text_activation._deny_network():  # noqa: SLF001 - activation boundary
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            assert isinstance(probe, socket.socket)
            with pytest.raises(SyntheticRasterTextActivationError, match="network_forbidden"):
                socket.getaddrinfo("localhost", 443)
            with pytest.raises(SyntheticRasterTextActivationError, match="network_forbidden"):
                probe.sendto(b"x", ("127.0.0.1", 9))
        finally:
            probe.close()


def test_cached_auto_status_avoids_full_sidecar_tree_hashes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = _cached_text_runtime_materials(tmp_path)
    engine = cast(LocalRasterTextPreviewEngine, object())

    def loader(**_kwargs: object) -> tuple[LocalRasterTextPreviewEngine, str]:
        return engine, fixture.profile.raster_text_profile_revision

    runtime = AutoActivatingSourceAttestedRasterTextPreviewRuntime(
        approval_path=fixture.pointer_path,
        binding_loader=loader,
    )
    # Status never starts the expensive native child.  The lifecycle worker
    # calls prewarm first, then subsequent status reads stay on the cheap
    # pointer/direct-material path.
    assert runtime.status().availability == "unavailable"
    assert runtime.prewarm().availability == "available"

    def unexpected_tree_hash(_path: Path) -> str:
        raise AssertionError("cached status must not hash a complete sidecar tree")

    def unexpected_inventory(**_kwargs: object) -> None:
        raise AssertionError("cached status must not reverify the parent package inventory")

    monkeypatch.setattr(
        raster_text_runtime, "_controlled_directory_tree_sha256", unexpected_tree_hash
    )
    monkeypatch.setattr(
        raster_text_runtime, "_controlled_module_root_tree_sha256", unexpected_tree_hash
    )
    monkeypatch.setattr(
        raster_text_runtime,
        "_verify_package_inventory",
        unexpected_inventory,
    )
    status = runtime.status()
    assert status.availability == "available"
    assert status.raster_text_profile_revision == fixture.profile.raster_text_profile_revision


def test_cached_auto_extract_keeps_one_full_parent_material_verify(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    fixture = _cached_text_runtime_materials(tmp_path)
    provider = PaddleOcrTextCommandProvider(
        profile=fixture.profile,
        interpreter=fixture.interpreter,
        interpreter_root=fixture.interpreter_root,
        module=fixture.module,
        module_root=fixture.module_root,
        configuration_path=fixture.configuration_path,
        configuration_sha256=hashlib.sha256(fixture.configuration_path.read_bytes()).hexdigest(),
        sidecar_inventory=fixture.sidecar_inventory,
        model_root=fixture.model_root,
        model_inventory=fixture.model_inventory,
    )
    candidate = RasterTextPreviewCandidateV1(
        status="preview_ready",
        extractor_revision="direct-text-test-v1",
        pages=[RasterTextPageV1(page_number=1, text="ready")],
    )

    class Engine:
        def extract(
            self, _raster_input: object, _structure: object
        ) -> RasterTextPreviewCandidateV1:
            # This is the same parent pre-launch verifier the concrete
            # provider uses immediately before one-shot OCR execution.
            provider._verify_parent_materials()
            return candidate

    runtime = AutoActivatingSourceAttestedRasterTextPreviewRuntime(
        approval_path=fixture.pointer_path,
        binding_loader=lambda **_kwargs: (
            cast(LocalRasterTextPreviewEngine, Engine()),
            fixture.profile.raster_text_profile_revision,
        ),
    )
    assert runtime.prewarm().availability == "available"
    checks: list[str] = []
    original_interpreter_tree = raster_text_runtime._controlled_directory_tree_sha256
    original_module_tree = raster_text_runtime._controlled_module_root_tree_sha256

    def count_interpreter_tree(path: Path) -> str:
        checks.append("interpreter")
        return original_interpreter_tree(path)

    def count_module_tree(path: Path) -> str:
        checks.append("module")
        return original_module_tree(path)

    monkeypatch.setattr(
        raster_text_runtime, "_controlled_directory_tree_sha256", count_interpreter_tree
    )
    monkeypatch.setattr(
        raster_text_runtime, "_controlled_module_root_tree_sha256", count_module_tree
    )
    raster_input = cast(
        RasterTextInputV1,
        SimpleNamespace(
            metadata=SimpleNamespace(
                raster_text_profile_revision=fixture.profile.raster_text_profile_revision
            )
        ),
    )
    assert runtime.extract(raster_input, cast(DocumentStructure, object())) == candidate
    # The cached validator performs no full tree walk.  The one pair below
    # comes only from the real provider's pre-launch material verifier.
    assert checks == ["interpreter", "module"]


@pytest.mark.parametrize("drift", ("pointer", "configuration", "interpreter", "module"))
def test_cached_auto_runtime_fails_closed_for_pointer_and_direct_material_drift(
    tmp_path: Path, drift: str
) -> None:
    fixture = _cached_text_runtime_materials(tmp_path)
    engine = cast(LocalRasterTextPreviewEngine, object())

    def loader(**kwargs: object) -> tuple[LocalRasterTextPreviewEngine, str]:
        pointer = parse_raster_text_activation_pointer(cast(bytes, kwargs["approval_raw"]))
        return engine, pointer.raster_text_profile_revision

    runtime = AutoActivatingSourceAttestedRasterTextPreviewRuntime(
        approval_path=fixture.pointer_path,
        binding_loader=loader,
    )
    assert runtime.prewarm().availability == "available"
    if drift == "pointer":
        fixture.pointer_path.write_bytes(b"{}\n")
    elif drift == "configuration":
        fixture.configuration_path.write_bytes(b"changed configuration")
    elif drift == "interpreter":
        fixture.interpreter.write_bytes(b"changed interpreter")
    else:
        fixture.module.write_bytes(b"changed module")

    status = runtime.status()
    assert status.availability == "unavailable"
    assert status.raster_text_profile_revision is None


def test_cached_auto_runtime_rejects_pointer_replaced_during_fast_validation(
    tmp_path: Path,
) -> None:
    fixture = _cached_text_runtime_materials(tmp_path)
    first_revision = fixture.profile.raster_text_profile_revision
    second_revision = "synthetic-raster-text-v1:" + ("d" * 24)
    loader_revisions: list[str] = []
    changed = False
    engine = cast(LocalRasterTextPreviewEngine, object())

    def loader(**kwargs: object) -> tuple[LocalRasterTextPreviewEngine, str]:
        pointer = parse_raster_text_activation_pointer(cast(bytes, kwargs["approval_raw"]))
        loader_revisions.append(pointer.raster_text_profile_revision)
        return engine, pointer.raster_text_profile_revision

    def validator(**kwargs: object) -> str:
        nonlocal changed
        pointer = parse_raster_text_activation_pointer(cast(bytes, kwargs["approval_raw"]))
        if not changed:
            changed = True
            fixture.pointer_path.write_bytes(
                _cached_text_pointer_raw(fixture, revision=second_revision)
            )
        return pointer.raster_text_profile_revision

    runtime = AutoActivatingSourceAttestedRasterTextPreviewRuntime(
        approval_path=fixture.pointer_path,
        binding_loader=loader,
        binding_material_validator=validator,
    )
    assert runtime.prewarm().raster_text_profile_revision == first_revision
    # A replacement during the lightweight drift check cannot return a stale
    # first profile.  The next background prewarm is allowed to load the new
    # pointer fresh without a user OCR request paying a cold-start cost.
    assert runtime.status().availability == "unavailable"
    assert runtime.prewarm().raster_text_profile_revision == second_revision
    assert loader_revisions == [first_revision, second_revision]


def test_auto_runtime_retires_warm_service_when_pointer_changes(tmp_path: Path) -> None:
    fixture = _cached_text_runtime_materials(tmp_path)
    closed: list[str] = []
    engine = cast(LocalRasterTextPreviewEngine, object())

    class Service:
        def verify_runtime_state(self) -> None:
            return None

        def close(self) -> None:
            closed.append("closed")

    service = Service()

    def loader(**kwargs: object) -> SourceAttestedRasterTextPreviewBindingV1:
        raw = cast(bytes, kwargs["approval_raw"])
        pointer = parse_raster_text_activation_pointer(raw)
        return SourceAttestedRasterTextPreviewBindingV1(
            root=fixture.root,
            approval_path=fixture.pointer_path,
            approval_sha256=hashlib.sha256(raw).hexdigest(),
            pointer=pointer,
            profile=fixture.profile,
            engine=engine,
            runtime_verifier=service.verify_runtime_state,
            runtime_closer=service.close,
        )

    runtime = AutoActivatingSourceAttestedRasterTextPreviewRuntime(
        approval_path=fixture.pointer_path,
        binding_loader=loader,
    )
    assert runtime.prewarm().availability == "available"

    fixture.pointer_path.write_bytes(
        _cached_text_pointer_raw(
            fixture,
            revision=fixture.profile.raster_text_profile_revision,
            sealed_evaluation_sha256="d" * 64,
        )
    )

    assert runtime.status().availability == "unavailable"
    assert closed == ["closed"]
    runtime.close()
    assert closed == ["closed"]


def test_auto_runtime_suppresses_ocr_result_after_pointer_change(tmp_path: Path) -> None:
    fixture = _cached_text_runtime_materials(tmp_path)
    closed: list[str] = []
    candidate = RasterTextPreviewCandidateV1(
        status="preview_ready",
        extractor_revision="direct-text-test-v1",
        pages=[RasterTextPageV1(page_number=1, text="ready")],
    )

    class Service:
        def verify_runtime_state(self) -> None:
            return None

        def close(self) -> None:
            closed.append("closed")

    class Engine:
        def extract(
            self, _raster_input: object, _structure: object
        ) -> RasterTextPreviewCandidateV1:
            fixture.pointer_path.write_bytes(
                _cached_text_pointer_raw(
                    fixture,
                    revision=fixture.profile.raster_text_profile_revision,
                    sealed_evaluation_sha256="d" * 64,
                )
            )
            return candidate

    service = Service()

    def loader(**kwargs: object) -> SourceAttestedRasterTextPreviewBindingV1:
        raw = cast(bytes, kwargs["approval_raw"])
        pointer = parse_raster_text_activation_pointer(raw)
        return SourceAttestedRasterTextPreviewBindingV1(
            root=fixture.root,
            approval_path=fixture.pointer_path,
            approval_sha256=hashlib.sha256(raw).hexdigest(),
            pointer=pointer,
            profile=fixture.profile,
            engine=cast(LocalRasterTextPreviewEngine, Engine()),
            runtime_verifier=service.verify_runtime_state,
            runtime_closer=service.close,
        )

    runtime = AutoActivatingSourceAttestedRasterTextPreviewRuntime(
        approval_path=fixture.pointer_path,
        binding_loader=loader,
    )
    assert runtime.prewarm().availability == "available"
    raster_input = cast(
        RasterTextInputV1,
        SimpleNamespace(
            metadata=SimpleNamespace(
                raster_text_profile_revision=fixture.profile.raster_text_profile_revision
            )
        ),
    )

    with pytest.raises(RasterTextPreviewUnavailable, match="raster text preview is unavailable"):
        runtime.extract(raster_input, cast(DocumentStructure, object()))

    assert closed == ["closed"]


def test_sealed_evidence_requires_and_binds_warm_service_smoke() -> None:
    scores = [
        RasterTextSealedProfileScoreV1(
            document_type=document_type,
            profile_id=profile_id,
            character_error_rate=0.01,
            critical_value_exact_match=1.0,
        )
        for document_type, profile_id in product(
            ("bill_of_lading", "commercial_invoice", "packing_list"),
            ("png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75"),
        )
    ]
    artifact = RasterTextSealedEvaluationV1(
        raster_text_profile_revision="synthetic-raster-text-v1:0123456789abcdef01234567",
        runtime_configuration_sha256=_SHA,
        extractor_revision="direct-text-test-v1",
        source_gold_receipt_path="synthetic-raster-text-v1/source-gold-receipts/sealed/a.json",
        source_gold_receipt_sha256=_SHA,
        calibration_receipt_path="synthetic-raster-text-v1/calibration-receipts/a.json",
        calibration_receipt_sha256=_SHA,
        candidate_execution_path="synthetic-raster-text-v1/candidate-executions/sealed/a.json",
        candidate_execution_sha256=_SHA,
        warm_service_smoke_receipt_path="synthetic-raster-text-v1/warm-service-smoke/a.json",
        warm_service_smoke_receipt_sha256=_SHA,
        latency_p95_seconds=1,
        service_warm_latency_p95_seconds=1,
        minimum_page_confidence=0.8,
        scores=scores,
    )
    assert artifact.warm_service_smoke_receipt_sha256 == _SHA


def test_pointer_loader_recomputes_sealed_gate_and_calibration_floor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Self-claimed passing scores/floors never activate a crafted pointer."""

    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    revision = "synthetic-raster-text-v1:0123456789abcdef01234567"
    configuration_raw = b"configuration"
    configuration_sha = hashlib.sha256(configuration_raw).hexdigest()
    profile = SimpleNamespace(
        raster_text_profile_revision=revision,
        extractor_revision="direct-text-test-v1",
        minimum_page_confidence=0.5,
        pdfium_version="4.30.0",
        raster_dpi=216,
    )
    scores = [
        RasterTextSealedProfileScoreV1(
            document_type=document_type,
            profile_id=profile_id,
            character_error_rate=0.01,
            critical_value_exact_match=1.0,
        )
        for document_type, profile_id in product(
            ("bill_of_lading", "commercial_invoice", "packing_list"),
            ("png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75"),
        )
    ]

    def write(relative: str, raw: bytes) -> str:
        path = root.joinpath(*relative.split("/"))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(raw)
        return relative

    write("profiles/final.json", configuration_raw)
    sealed_source = write("evidence/sealed-source.json", b"sealed-source")
    sealed_execution = write("evidence/sealed-execution.json", b"sealed-execution")
    calibration_source = write("evidence/calibration-source.json", b"calibration-source")
    calibration_execution = write("evidence/calibration-execution.json", b"calibration-execution")
    smoke_source = write("evidence/smoke-source.json", b"smoke-source")
    smoke_execution = write("evidence/smoke-execution.json", b"smoke-execution")
    calibration = RasterTextCalibrationReceiptV1(
        raster_text_profile_revision=revision,
        runtime_configuration_sha256=configuration_sha,
        calibration_raster_text_profile_revision=revision,
        calibration_runtime_configuration_sha256=configuration_sha,
        extractor_revision=profile.extractor_revision,
        source_gold_receipt_path=calibration_source,
        source_gold_receipt_sha256=hashlib.sha256(
            (root / calibration_source).read_bytes()
        ).hexdigest(),
        candidate_execution_path=calibration_execution,
        candidate_execution_sha256=hashlib.sha256(
            (root / calibration_execution).read_bytes()
        ).hexdigest(),
        minimum_page_confidence=0.5,
        latency_p95_seconds=1.0,
    )
    calibration_path = write(
        "evidence/calibration.json",
        raster_text_activation._canonical_json(  # noqa: SLF001 - exact writer encoding
            calibration.model_dump(mode="json", warnings=False)
        ),
    )
    warm = RasterTextWarmServiceSmokeReceiptV1(
        artifact_kind="synthetic_raster_text_warm_service_smoke",
        dataset_id="synthetic-raster-text-v1",
        scope="smoke",
        source_truth_kind="pdf_content_stream_text_v1",
        execution_mode="persistent_service_warm",
        service_protocol="customs-raster-text-sidecar-v1-service",
        raster_text_profile_revision=revision,
        runtime_configuration_sha256=configuration_sha,
        extractor_revision=profile.extractor_revision,
        source_gold_receipt_path=smoke_source,
        source_gold_receipt_sha256=hashlib.sha256((root / smoke_source).read_bytes()).hexdigest(),
        candidate_execution_path=smoke_execution,
        candidate_execution_sha256=hashlib.sha256(
            (root / smoke_execution).read_bytes()
        ).hexdigest(),
        candidate_bundle_sha256="a" * 64,
        equality_mode="exact_canonical_result_v1",
        equality_set_sha256="b" * 64,
        asset_count=24,
        startup_latency_seconds=1.0,
        startup_timeout_seconds=55,
        warm_latency_p95_seconds=1.0,
        request_timeout_seconds=30,
    )
    warm_path = write(
        "evidence/warm.json",
        raster_text_activation._canonical_json(  # noqa: SLF001 - exact writer encoding
            warm.model_dump(mode="json", warnings=False)
        ),
    )
    evaluation = RasterTextSealedEvaluationV1(
        raster_text_profile_revision=revision,
        runtime_configuration_sha256=configuration_sha,
        extractor_revision=profile.extractor_revision,
        source_gold_receipt_path=sealed_source,
        source_gold_receipt_sha256=hashlib.sha256((root / sealed_source).read_bytes()).hexdigest(),
        calibration_receipt_path=calibration_path,
        calibration_receipt_sha256=hashlib.sha256(
            (root / calibration_path).read_bytes()
        ).hexdigest(),
        candidate_execution_path=sealed_execution,
        candidate_execution_sha256=hashlib.sha256(
            (root / sealed_execution).read_bytes()
        ).hexdigest(),
        warm_service_smoke_receipt_path=warm_path,
        warm_service_smoke_receipt_sha256=hashlib.sha256(
            (root / warm_path).read_bytes()
        ).hexdigest(),
        latency_p95_seconds=1.0,
        service_warm_latency_p95_seconds=1.0,
        minimum_page_confidence=0.5,
        scores=scores,
    )
    evaluation_path = write(
        "evidence/sealed.json",
        raster_text_activation._canonical_json(  # noqa: SLF001 - exact writer encoding
            evaluation.model_dump(mode="json", warnings=False)
        ),
    )
    pointer_raw = _canonical_bytes(
        {
            "schema_version": "1.0",
            "activation_kind": "source_attested_synthetic_raster_text",
            "status": "activated",
            "dataset_id": "synthetic-raster-text-v1",
            "raster_text_profile_revision": revision,
            "runtime_configuration_path": "profiles/final.json",
            "runtime_configuration_sha256": configuration_sha,
            "sealed_evaluation_path": evaluation_path,
            "sealed_evaluation_sha256": hashlib.sha256(
                (root / evaluation_path).read_bytes()
            ).hexdigest(),
        }
    )
    calls: list[str] = []

    def verified_chain(**kwargs: object) -> tuple[object, object, object, list[object]]:
        scope = cast(str, kwargs["scope"])
        calls.append(scope)
        execution = SimpleNamespace(
            scope=scope,
            raster_text_profile_revision=revision,
            runtime_configuration_sha256=configuration_sha,
        )
        return (
            SimpleNamespace(scope=scope),
            execution,
            SimpleNamespace(scope=scope),
            [SimpleNamespace()],
        )

    monkeypatch.setattr(
        raster_text_activation, "verify_source_attested_raster_text_evidence", verified_chain
    )
    monkeypatch.setattr(raster_text_activation, "_calibrated_floor", lambda _cases: 0.5)
    monkeypatch.setattr(
        raster_text_evaluation, "evaluate_synthetic_raster_text_gates", lambda _cases: scores
    )
    monkeypatch.setattr(
        raster_text_evaluation, "synthetic_raster_text_gate_passes", lambda _scores: True
    )
    monkeypatch.setattr(
        raster_text_runtime, "_verify_warm_service_smoke_evidence", lambda **_kwargs: None
    )
    monkeypatch.setattr(
        raster_text_runtime,
        "_load_profile_materials",
        lambda **_kwargs: (
            profile,
            Path("interpreter"),
            Path("interpreter-root"),
            Path("module"),
            Path("module-root"),
            Path("inventory"),
            Path("models"),
            Path("model-inventory"),
        ),
    )

    loaded = raster_text_runtime._load_binding_from_pointer(  # noqa: SLF001 - loader boundary
        root=root,
        approval_path=root / "approvals" / "raster-text-active.json",
        approval_raw=pointer_raw,
        approval_sha256=hashlib.sha256(pointer_raw).hexdigest(),
        start_service_session=False,
    )
    assert loaded.raster_text_profile_revision == revision
    assert calls == ["sealed", "calibration"]

    altered_scores = [
        RasterTextSealedProfileScoreV1(**{**scores[0].model_dump(), "character_error_rate": 0.02}),
        *scores[1:],
    ]
    monkeypatch.setattr(
        raster_text_evaluation,
        "evaluate_synthetic_raster_text_gates",
        lambda _cases: altered_scores,
    )
    with pytest.raises(RasterTextRuntimeError, match="evaluation_invalid"):
        raster_text_runtime._load_binding_from_pointer(  # noqa: SLF001 - loader boundary
            root=root,
            approval_path=root / "approvals" / "raster-text-active.json",
            approval_raw=pointer_raw,
            approval_sha256=hashlib.sha256(pointer_raw).hexdigest(),
            start_service_session=False,
        )

    destination = root / "approvals" / "raster-text-active.json"
    destination.parent.mkdir(parents=True)
    previous_pointer = b"previous-pointer"
    destination.write_bytes(previous_pointer)
    monkeypatch.setattr(
        raster_text_runtime,
        "parse_raster_text_offline_profile",
        lambda _raw: profile,
    )
    with pytest.raises(RasterTextRuntimeError, match="evaluation_invalid"):
        raster_text_runtime.write_source_attested_synthetic_raster_text_pointer(
            document_preview_root=root,
            runtime_configuration_relative_path="profiles/final.json",
            sealed_evaluation_relative_path=evaluation_path,
        )
    assert destination.read_bytes() == previous_pointer

    monkeypatch.setattr(
        raster_text_evaluation, "evaluate_synthetic_raster_text_gates", lambda _cases: scores
    )
    monkeypatch.setattr(raster_text_activation, "_calibrated_floor", lambda _cases: 0.49)
    with pytest.raises(RasterTextRuntimeError, match="evaluation_invalid"):
        raster_text_runtime._load_binding_from_pointer(  # noqa: SLF001 - loader boundary
            root=root,
            approval_path=root / "approvals" / "raster-text-active.json",
            approval_raw=pointer_raw,
            approval_sha256=hashlib.sha256(pointer_raw).hexdigest(),
            start_service_session=False,
        )


def test_warm_service_smoke_receipt_binds_exact_result_set_without_ocr_text() -> None:
    artifact = RasterTextWarmServiceSmokeReceiptV1(
        artifact_kind="synthetic_raster_text_warm_service_smoke",
        dataset_id="synthetic-raster-text-v1",
        scope="smoke",
        source_truth_kind="pdf_content_stream_text_v1",
        execution_mode="persistent_service_warm",
        service_protocol="customs-raster-text-sidecar-v1-service",
        raster_text_profile_revision="synthetic-raster-text-v1:0123456789abcdef01234567",
        runtime_configuration_sha256=_SHA,
        extractor_revision="direct-text-test-v1",
        source_gold_receipt_path="synthetic-raster-text-v1/source-gold-receipts/smoke/a.json",
        source_gold_receipt_sha256=_SHA,
        candidate_execution_path="synthetic-raster-text-v1/candidate-executions/smoke/a.json",
        candidate_execution_sha256=_SHA,
        candidate_bundle_sha256=_SHA,
        equality_mode="exact_canonical_result_v1",
        equality_set_sha256="b" * 64,
        asset_count=24,
        startup_latency_seconds=2,
        startup_timeout_seconds=55,
        warm_latency_p95_seconds=1,
        request_timeout_seconds=30,
    )
    payload = artifact.model_dump(mode="json")
    assert payload["execution_mode"] == "persistent_service_warm"
    assert "canonical_text" not in payload
    # V2's separately bounded native-ready path can take 60 seconds, while
    # the persisted warm request lane remains fixed at 30 seconds.
    v2_ready_payload = {
        **payload,
        "startup_latency_seconds": 60,
        "startup_timeout_seconds": 60,
    }
    v2_ready = RasterTextWarmServiceSmokeReceiptV1.model_validate(v2_ready_payload)
    assert v2_ready.startup_timeout_seconds == 60
    assert v2_ready.request_timeout_seconds == 30
    for key in (
        "artifact_kind",
        "dataset_id",
        "scope",
        "source_truth_kind",
        "execution_mode",
        "service_protocol",
        "equality_mode",
        "asset_count",
    ):
        legacy = dict(payload)
        legacy.pop(key)
        with pytest.raises(ValueError):
            RasterTextWarmServiceSmokeReceiptV1.model_validate(legacy)
    for key, value in (
        ("artifact_kind", "synthetic_raster_text_cold_service_smoke"),
        ("execution_mode", "one_shot_cold"),
        ("service_protocol", "customs-raster-text-sidecar-v1"),
    ):
        legacy = {**payload, key: value}
        with pytest.raises(ValueError):
            RasterTextWarmServiceSmokeReceiptV1.model_validate(legacy)


def test_exact_result_set_hash_rejects_a_changed_request_or_text() -> None:
    result = _result("same")
    original = raster_text_runtime.raster_text_exact_result_set_sha256(
        [("asset-1", "png-216", result)]
    )
    request_changed = result.model_copy(update={"extraction_request_id": "request-2"})
    text_changed = _result("different")
    assert original != raster_text_runtime.raster_text_exact_result_set_sha256(
        [("asset-1", "png-216", request_changed)]
    )
    assert original != raster_text_runtime.raster_text_exact_result_set_sha256(
        [("asset-1", "png-216", text_changed)]
    )


def test_warm_smoke_uses_service_session_and_rejects_nonidentical_candidate(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Activation evidence may not substitute the evaluator or a pointer.

    This isolates the bounded 24-result comparison without creating OCR text
    artifacts on disk.  The receipt writer receives only a digest-bearing
    model, never the individual candidate values.
    """

    revision = "synthetic-raster-text-v1:0123456789abcdef01234567"
    configuration_sha = "b" * 64
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    assets = [
        SimpleNamespace(
            asset_id=f"asset-{index:02d}",
            profile_id="png-216",
            image_sha256=f"{index:064x}",
            media_type="image/png",
        )
        for index in range(24)
    ]

    def candidate_for(
        asset: SimpleNamespace, *, text: str = "same"
    ) -> RasterTextExtractionResultV1:
        pages = [RasterTextPageV1(page_number=1, text=text, confidence=0.9)]
        return RasterTextExtractionResultV1(
            document_asset_id=asset.asset_id,
            extraction_request_id=f"raster-text-{asset.profile_id}-{asset.image_sha256[:24]}",
            raster_text_profile_revision=revision,
            status="preview_ready",
            media_type="image/png",
            byte_size=5,
            source_sha256=asset.image_sha256,
            canonical_text=canonical_raster_text(pages),
            canonical_text_sha256=canonical_raster_text_sha256(pages),
            extractor_revision="direct-text-test-v1",
            pages=pages,
            warnings=[],
            needs_review_reasons=[],
        )

    candidates = [
        SimpleNamespace(
            asset_id=asset.asset_id,
            profile_id=asset.profile_id,
            candidate=candidate_for(asset),
        )
        for asset in assets
    ]
    source = SimpleNamespace(scope="smoke", synthetic_raster_receipt_sha256="a" * 64)
    execution = SimpleNamespace(
        scope="smoke",
        synthetic_raster_receipt_sha256="a" * 64,
        runtime_configuration_sha256=configuration_sha,
        raster_text_profile_revision=revision,
        extractor_revision="direct-text-test-v1",
        candidate_count=24,
        candidate_bundle_sha256="c" * 64,
    )
    bundle = SimpleNamespace(
        scope="smoke",
        synthetic_raster_receipt_sha256="a" * 64,
        runtime_configuration_sha256=configuration_sha,
        raster_text_profile_revision=revision,
        extractor_revision="direct-text-test-v1",
        candidates=candidates,
    )

    class Session:
        request_timeout_seconds = 30
        startup_timeout_seconds = 55

        def __init__(self) -> None:
            self.closed = False

        def close(self) -> None:
            self.closed = True

    session = Session()
    starts: list[bool] = []
    parent_verifications: list[str] = []

    class Provider:
        def _verify_parent_materials(self) -> None:
            parent_verifications.append("full")

        def start_service_session(self, *, parent_materials_verified: bool) -> Session:
            starts.append(parent_materials_verified)
            return session

    provider = Provider()
    resolution = SimpleNamespace(
        status="available",
        engine=SimpleNamespace(input_adapter=object(), provider=provider),
        profile=SimpleNamespace(
            raster_text_profile_revision=revision,
            extractor_revision="direct-text-test-v1",
            sidecar_timeout_seconds=30,
        ),
        runtime_configuration_sha256=configuration_sha,
    )
    receipt = SimpleNamespace(pdfium_version="4.30.0")
    captured: list[dict[str, object]] = []
    ticks = iter(range(1, 200))

    monkeypatch.setattr(
        raster_text_activation,
        "_load_dataset_receipt",
        lambda *_args, **_kwargs: (receipt, "a" * 64),
    )
    monkeypatch.setattr(
        raster_text_activation,
        "_load_source_gold_cases",
        lambda **_kwargs: (source, execution, bundle, []),
    )
    monkeypatch.setattr(
        raster_text_activation,
        "verify_source_attested_raster_text_evidence",
        lambda **_kwargs: (source, execution, bundle, []),
    )
    monkeypatch.setattr(raster_text_activation, "_scope_assets", lambda *_args: assets)
    monkeypatch.setattr(
        raster_text_activation, "_verified_asset", lambda *_args, **_kwargs: b"image"
    )
    monkeypatch.setattr(raster_text_activation, "_read", lambda *_args, **_kwargs: b"receipt")
    monkeypatch.setattr(
        raster_text_activation,
        "resolve_paddleocr3_raster_text_candidate",
        lambda **_kwargs: resolution,
    )
    monkeypatch.setattr(
        raster_text_activation,
        "inspect_raster_document_structure",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(raster_text_activation.time, "monotonic", lambda: float(next(ticks)))

    def service_result(
        engine: object, *, raster_input: RasterTextInputV1, **_kwargs: object
    ) -> RasterTextExtractionResultV1:
        assert getattr(engine, "provider") is session
        return next(
            item.candidate
            for item in candidates
            if item.asset_id == raster_input.metadata.document_asset_id
        )

    monkeypatch.setattr(raster_text_activation, "extract_raster_text_preview", service_result)

    def write_receipt(**kwargs: object) -> tuple[str, str]:
        captured.append(cast(dict[str, object], kwargs["value"]))
        relative_path = cast(str, kwargs["relative_path"])
        return relative_path, Path(relative_path).stem

    monkeypatch.setattr(
        raster_text_activation, "write_or_verify_canonical_local_json", write_receipt
    )
    relative, digest = raster_text_activation.run_warm_service_raster_text_smoke(
        document_preview_root=root,
        runtime_configuration_relative_path="profiles/final.json",
        synthetic_raster_receipt_relative_path="synthetic-raster-v1/receipt.json",
        source_gold_receipt_relative_path="synthetic-raster-text-v1/source-gold-receipts/smoke/a.json",
        candidate_execution_relative_path="synthetic-raster-text-v1/candidate-executions/smoke/a.json",
    )

    assert starts == [True]
    assert parent_verifications == ["full"]
    assert session.closed
    assert len(digest) == 64
    assert relative.startswith("synthetic-raster-text-v1/warm-service-smoke/")
    assert captured[0]["equality_mode"] == "exact_canonical_result_v1"
    assert "canonical_text" not in captured[0]

    def changed_service_result(
        engine: object, *, raster_input: RasterTextInputV1, **_kwargs: object
    ) -> RasterTextExtractionResultV1:
        assert getattr(engine, "provider") is session
        return candidate_for(assets[0], text="changed")

    monkeypatch.setattr(
        raster_text_activation, "extract_raster_text_preview", changed_service_result
    )
    session.closed = False
    with pytest.raises(SyntheticRasterTextActivationError, match="warm_smoke_failed"):
        raster_text_activation.run_warm_service_raster_text_smoke(
            document_preview_root=root,
            runtime_configuration_relative_path="profiles/final.json",
            synthetic_raster_receipt_relative_path="synthetic-raster-v1/receipt.json",
            source_gold_receipt_relative_path="synthetic-raster-text-v1/source-gold-receipts/smoke/a.json",
            candidate_execution_relative_path="synthetic-raster-text-v1/candidate-executions/smoke/a.json",
        )
    assert session.closed


def test_source_attested_evidence_rejects_a_binding_not_in_the_fixed_receipt(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A compact candidate receipt cannot substitute a different raster asset."""

    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    revision = "synthetic-raster-text-v1:0123456789abcdef01234567"
    configuration_sha = "b" * 64
    asset = SimpleNamespace(
        asset_id="asset-1",
        profile_id="png-216",
        family_id="family-1",
        document_type="commercial_invoice",
        source_pdf_sha256="c" * 64,
        image_sha256="d" * 64,
        gold_relative_path="synthetic-raster-v1/gold/asset-1.json",
        gold_sha256="e" * 64,
        media_type="image/png",
        byte_size=5,
    )
    pages = [RasterTextPageV1(page_number=1, text="same", confidence=0.9)]
    candidate = RasterTextExtractionResultV1(
        document_asset_id=asset.asset_id,
        extraction_request_id=f"raster-text-{asset.profile_id}-{asset.image_sha256[:24]}",
        raster_text_profile_revision=revision,
        status="preview_ready",
        media_type=asset.media_type,
        byte_size=asset.byte_size,
        source_sha256=asset.image_sha256,
        canonical_text=canonical_raster_text(pages),
        canonical_text_sha256=canonical_raster_text_sha256(pages),
        extractor_revision="direct-text-test-v1",
        pages=pages,
        warnings=[],
        needs_review_reasons=[],
    )
    binding = SimpleNamespace(
        asset_id=asset.asset_id,
        profile_id=asset.profile_id,
        family_id=asset.family_id,
        document_type=asset.document_type,
        source_pdf_sha256=asset.source_pdf_sha256,
        image_sha256=asset.image_sha256,
        source_raster_gold_relative_path=asset.gold_relative_path,
        source_raster_gold_sha256=asset.gold_sha256,
    )
    source = SimpleNamespace(
        scope="smoke", synthetic_raster_receipt_sha256="a" * 64, bindings=[binding]
    )
    execution = SimpleNamespace(
        scope="smoke",
        synthetic_raster_receipt_sha256="a" * 64,
        runtime_configuration_sha256=configuration_sha,
        raster_text_profile_revision=revision,
        extractor_revision="direct-text-test-v1",
        candidate_count=1,
    )
    bundle = SimpleNamespace(
        scope="smoke",
        synthetic_raster_receipt_sha256="a" * 64,
        runtime_configuration_sha256=configuration_sha,
        raster_text_profile_revision=revision,
        extractor_revision="direct-text-test-v1",
        runner_implementation_sha256=(
            raster_text_activation._runner_implementation_sha256()  # noqa: SLF001
        ),
        candidates=[
            SimpleNamespace(
                asset_id=asset.asset_id, profile_id=asset.profile_id, candidate=candidate
            )
        ],
    )
    monkeypatch.setattr(
        raster_text_activation,
        "_load_dataset_receipt",
        lambda *_args, **_kwargs: (SimpleNamespace(), "a" * 64),
    )
    monkeypatch.setattr(
        raster_text_activation,
        "_load_source_gold_cases",
        lambda **_kwargs: (source, execution, bundle, [SimpleNamespace()]),
    )
    monkeypatch.setattr(raster_text_activation, "_scope_assets", lambda *_args: [asset])

    assert (
        raster_text_activation.verify_source_attested_raster_text_evidence(
            document_preview_root=root,
            source_gold_receipt_relative_path="source.json",
            candidate_execution_relative_path="execution.json",
            scope="smoke",
            expected_raster_text_profile_revision=revision,
            expected_runtime_configuration_sha256=configuration_sha,
            expected_extractor_revision="direct-text-test-v1",
        )[0]
        is source
    )

    binding.image_sha256 = "f" * 64
    with pytest.raises(SyntheticRasterTextActivationError, match="evaluation_invalid"):
        raster_text_activation.verify_source_attested_raster_text_evidence(
            document_preview_root=root,
            source_gold_receipt_relative_path="source.json",
            candidate_execution_relative_path="execution.json",
            scope="smoke",
            expected_raster_text_profile_revision=revision,
            expected_runtime_configuration_sha256=configuration_sha,
            expected_extractor_revision="direct-text-test-v1",
        )

    binding.image_sha256 = asset.image_sha256
    bundle.runner_implementation_sha256 = "0" * 64
    with pytest.raises(SyntheticRasterTextActivationError, match="evaluation_invalid"):
        raster_text_activation.verify_source_attested_raster_text_evidence(
            document_preview_root=root,
            source_gold_receipt_relative_path="source.json",
            candidate_execution_relative_path="execution.json",
            scope="smoke",
            expected_raster_text_profile_revision=revision,
            expected_runtime_configuration_sha256=configuration_sha,
            expected_extractor_revision="direct-text-test-v1",
        )

    bundle.runner_implementation_sha256 = (
        raster_text_activation._runner_implementation_sha256()  # noqa: SLF001
    )
    evaluator_path = dict(raster_text_activation._RUNNER_IMPLEMENTATION_FILES)[  # noqa: SLF001
        "evaluation/synthetic_raster_text.py"
    ]
    original_read_bytes = Path.read_bytes

    def drifted_read_bytes(path: Path) -> bytes:
        raw = original_read_bytes(path)
        if path.resolve() == evaluator_path.resolve():
            return raw + b"\n# test-only evaluator drift\n"
        return raw

    monkeypatch.setattr(Path, "read_bytes", drifted_read_bytes)
    with pytest.raises(SyntheticRasterTextActivationError, match="evaluation_invalid"):
        raster_text_activation.verify_source_attested_raster_text_evidence(
            document_preview_root=root,
            source_gold_receipt_relative_path="source.json",
            candidate_execution_relative_path="execution.json",
            scope="smoke",
            expected_raster_text_profile_revision=revision,
            expected_runtime_configuration_sha256=configuration_sha,
            expected_extractor_revision="direct-text-test-v1",
        )


@pytest.mark.parametrize(
    "relative_path",
    [
        "evaluation/synthetic_raster_text_activation.py",
        "evaluation/synthetic_raster_text.py",
        "evaluation/synthetic_raster.py",
        "evaluation/source_attested_synthetic_pdf_truth.py",
        "evaluation/raster_offline_bundle.py",
        "evaluation/raster_text_direct_paddle_bundle.py",
        "evaluation/raster_text_mobile_paddle_bundle.py",
        "service/raster_text_extraction.py",
        "service/raster_text_preview_runtime.py",
        "service/raster_text_direct_paddle_sidecar.py",
        "service/raster_document_preview_runtime.py",
        "service/document_input.py",
        "schemas/raster_text_extraction.py",
    ],
)
def test_runner_implementation_hash_binds_every_fixed_decision_dependency(
    monkeypatch: pytest.MonkeyPatch, relative_path: str
) -> None:
    baseline = raster_text_activation._runner_implementation_sha256()  # noqa: SLF001
    target = dict(raster_text_activation._RUNNER_IMPLEMENTATION_FILES)[  # noqa: SLF001
        relative_path
    ]
    original_read_bytes = Path.read_bytes

    def drifted_read_bytes(path: Path) -> bytes:
        raw = original_read_bytes(path)
        if path.resolve() == target.resolve():
            return raw + b"\n# test-only dependency drift\n"
        return raw

    monkeypatch.setattr(Path, "read_bytes", drifted_read_bytes)
    assert raster_text_activation._runner_implementation_sha256() != baseline  # noqa: SLF001


def test_smoke_quality_screen_can_reject_but_cannot_claim_a_relaxed_pass() -> None:
    scores = [
        RasterTextGateScoreV1(
            document_type=document_type,
            profile_id=profile_id,
            character_error_rate=0.01,
            critical_value_exact_match=1.0,
        )
        for document_type, profile_id in product(
            ("bill_of_lading", "commercial_invoice", "packing_list"),
            ("png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75"),
        )
    ]
    scores[0] = RasterTextGateScoreV1(
        document_type=scores[0].document_type,
        profile_id=scores[0].profile_id,
        character_error_rate=0.03,
        critical_value_exact_match=1.0,
    )
    shared = {
        "source_gold_receipt_path": "synthetic-raster-text-v1/source-gold-receipts/smoke/a.json",
        "source_gold_receipt_sha256": _SHA,
        "candidate_execution_path": "synthetic-raster-text-v1/candidate-executions/smoke/a.json",
        "candidate_execution_sha256": _SHA,
        "raster_text_profile_revision": "synthetic-raster-text-v1:0123456789abcdef01234567",
        "runtime_configuration_sha256": _SHA,
        "extractor_revision": "direct-text-test-v1",
        "scores": scores,
    }
    rejected = RasterTextSmokeQualityScreenV1(**shared, gate_passed=False)
    assert not rejected.gate_passed
    with pytest.raises(ValueError, match="smoke quality result"):
        RasterTextSmokeQualityScreenV1(**shared, gate_passed=True)
