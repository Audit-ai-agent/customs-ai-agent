"""V20 voyage-replay runtime and activation-pointer isolation checks."""

from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, Literal

from pydantic import ValidationError
import pytest

from customs_ai_agent.evaluation import (
    raster_text_public_voyage_replay_paddle_bundle as voyage_bundle,
)
from customs_ai_agent.evaluation import source_attested_bl_repair_audit as audit
from customs_ai_agent.evaluation import (
    source_attested_public_raster_text_voyage_replay_corpus as replay,
)
from customs_ai_agent.service import raster_text_preview_runtime as runtime


_SHA_A = "a" * 64
_SHA_B = "b" * 64
_SHA_C = "c" * 64
_SHA_D = "d" * 64
_SHA_E = "e" * 64
_SHA_F = "f" * 64
_PROFILE_PREFIX = "synthetic-raster-text-v1:"
_V20_ENGINE = (
    "direct_paddle_mobile_det_public_voyage_replay_rec_harmonic_recall_native960_"
    "lowbox_strip2_semantic_repair_v2_onednn"
)
_V19B_ENGINE = (
    "direct_paddle_mobile_det_public_finetuned_rec_harmonic_recall_native960_"
    "lowbox_strip2_semantic_repair_v2_voyage_retry_onednn"
)
_EVIDENCE_REVISION = "source-attested-public-finetuned-staged-train-holdout-v2"
_AUDIT_RECEIPT_PATH = "synthetic-bl-repair-audit-v1/receipt.json"


def _manifest(
    split: Literal["train", "development"],
) -> replay.VoyageReplayCorpusManifestV1:
    train = split == "train"
    voyage_count = 500 if train else 200
    general_count = 250 if train else 200
    lane_counts: dict[str, int] = {
        str(recipe.recipe_id): (63 if index < 4 else 62) if train else 25
        for index, recipe in enumerate(replay.VOYAGE_REPLAY_RECIPES_V1)
    }
    all_fields = {
        "bill_of_lading:voyage": voyage_count,
        "commercial_invoice:invoice_number": general_count,
    }
    document_counts = {
        "bill_of_lading": general_count - 2,
        "commercial_invoice": 1,
        "packing_list": 1,
    }
    profile_counts = {
        "png-216": general_count // 3 + (1 if general_count % 3 else 0),
        "jpeg-216-q75": general_count // 3 + (1 if general_count % 3 == 2 else 0),
        "jpeg-150-q75": general_count // 3,
    }
    selection_path = f"synthetic-raster-text-v1/fresh-scope-selections/{_SHA_C}.json"
    return replay.VoyageReplayCorpusManifestV1(
        split=split,
        purpose="optimizer" if train else "validation-only",
        source_receipt_sha256=_SHA_A,
        raster_receipt_sha256=_SHA_B,
        selection_receipt_relative_path=selection_path,
        selection_receipt_sha256=_SHA_C,
        parent_selection_receipt_sha256=_SHA_D,
        selected_family_count=500 if train else 100,
        selected_family_ids_sha256=_SHA_A,
        excluded_family_ids_sha256=_SHA_B,
        independent_audit_recipe_sha256=_SHA_C,
        independent_audit_family_ids_sha256=_SHA_D,
        independent_audit_source_sha256s_sha256=_SHA_E,
        voyage_crop_count=voyage_count,
        general_crop_count=general_count,
        total_crop_count=voyage_count + general_count,
        voyage_recipe_set_sha256=replay._RECIPE_SET_SHA256,
        voyage_recipe_sha256s=replay.voyage_replay_recipe_sha256s_v1(),
        voyage_recipe_lane_counts=lane_counts,
        voyage_sample_ids_sha256=_SHA_A,
        voyage_label_sha256s_sha256=_SHA_B,
        voyage_label_pattern_sha256=replay._VOYAGE_PATTERN_SHA256,
        general_sample_ids_sha256=_SHA_C,
        general_field_ids_sha256=_SHA_D,
        general_field_crop_counts={"commercial_invoice:invoice_number": general_count},
        all_field_key_ids_sha256=replay._hash_list(sorted(all_fields)),
        all_field_key_crop_counts=all_fields,
        general_document_type_crop_counts=document_counts,
        general_profile_crop_counts=profile_counts,
    )


def _manifest_sha256(manifest: replay.VoyageReplayCorpusManifestV1) -> str:
    return hashlib.sha256(
        replay.canonical_json_bytes(manifest.model_dump(mode="json", warnings=False))
    ).hexdigest()


def _profile_payload() -> dict[str, object]:
    sidecar_root = f"raster-text-direct-bundles-v1/releases/{_SHA_A}"
    model_root = f"raster-text-public-voyage-replay-paddle-bundles-v1/releases/{_SHA_B}"
    train = _manifest("train")
    development = _manifest("development")
    payload: dict[str, object] = {
        "schema_version": "20.0",
        "engine_kind": _V20_ENGINE,
        "runtime_api": "paddle-inference-static-v1",
        "extractor_revision": "v20-voyage-replay-test",
        "bundle_root_relative_path": sidecar_root,
        "bundle_manifest_sha256": _SHA_A,
        "mobile_model_bundle_root_relative_path": model_root,
        "mobile_model_bundle_manifest_sha256": _SHA_B,
        "source_model_inventory_relative_path": (
            f"{model_root}/provenance/public-voyage-replay-recognizer-training-receipt.json"
        ),
        "source_model_inventory_sha256": _SHA_C,
        "pdfium_version": "4.30.0",
        "raster_dpi": 216,
        "sidecar_timeout_seconds": 30,
        "cpu_threads": 4,
        "text_detection_limit_side_len": 960,
        "text_detection_limit_type": "max",
        "text_detection_threshold": 0.3,
        "text_detection_box_threshold": 0.3,
        "text_detection_unclip_ratio": 1.5,
        "minimum_page_confidence": 0.42,
        "parent_renderer_package_inventory_path": f"{sidecar_root}/parent/packages.json",
        "parent_renderer_package_inventory_sha256": _SHA_D,
        "parent_renderer_required_packages": ["pypdfium2", "Pillow"],
        "sidecar_interpreter_path": f"{sidecar_root}/sidecar/python/python.exe",
        "sidecar_interpreter_sha256": _SHA_E,
        "sidecar_interpreter_root": f"{sidecar_root}/sidecar/python",
        "sidecar_interpreter_root_tree_sha256": _SHA_F,
        "sidecar_module_path": f"{sidecar_root}/sidecar/module/runtime.py",
        "sidecar_module_sha256": _SHA_A,
        "sidecar_module_root": f"{sidecar_root}/sidecar/module",
        "sidecar_module_root_tree_sha256": _SHA_B,
        "sidecar_package_inventory_path": f"{sidecar_root}/sidecar/packages.json",
        "sidecar_package_inventory_sha256": _SHA_C,
        "sidecar_required_packages": [
            "numpy",
            "opencv-contrib-python",
            "paddlepaddle",
            "pyclipper",
        ],
        "local_model_root": f"{model_root}/models",
        "model_inventory_path": (
            f"{model_root}/inventories/direct-paddle-public-voyage-replay-models.json"
        ),
        "model_inventory_sha256": _SHA_D,
        "text_detection_model_name": "PP-OCRv5_mobile_det",
        "text_detection_model_directory": "text-det",
        "text_recognition_model_name": (
            "source_attested_en_PP-OCRv5_mobile_rec_voyage_replay_head_v3"
        ),
        "text_recognition_model_directory": "text-rec",
        "enable_mkldnn": True,
        "mkldnn_cache_capacity": 10,
        "recognition_batch_size": 8,
        "opencv_threads": 1,
        "warmup_shape_policy": "detector-736x960-recognizer-48x320-v1",
        "onednn_model_policy": "recognizer-only-v1",
        "ctc_label_policy": "optional-space-v1",
        "recognition_crop_policy": "horizontal-height-0p32-max8-v1",
        "recognition_resize_interpolation": "linear-v1",
        "source_recognizer_release_sha256": _SHA_A,
        "source_recognizer_receipt_sha256": _SHA_B,
        "source_recognizer_training_intent_sha256": _SHA_C,
        "source_recognizer_selection_receipt_relative_path": (
            f"synthetic-raster-text-v1/fresh-scope-selections/{_SHA_C}.json"
        ),
        "source_recognizer_selection_receipt_sha256": _SHA_C,
        "source_recognizer_parent_selection_receipt_sha256": _SHA_D,
        "source_recognizer_train_corpus_manifest": train.model_dump(mode="json"),
        "source_recognizer_development_corpus_manifest": development.model_dump(mode="json"),
        "source_recognizer_train_manifest_sha256": _manifest_sha256(train),
        "source_recognizer_development_manifest_sha256": _manifest_sha256(development),
        "source_recognizer_parent_release_sha256": _SHA_D,
        "source_recognizer_parent_receipt_sha256": _SHA_E,
        "source_recognizer_voyage_recipe_set_sha256": replay._RECIPE_SET_SHA256,
        "source_recognizer_base_dictionary_sha256": _SHA_F,
        "source_recognizer_output_class_count": 438,
    }
    return _rebind_profile(payload)


def _rebind_profile(payload: dict[str, object]) -> dict[str, object]:
    material = {
        key: value
        for key, value in payload.items()
        if key not in {"profile_fingerprint_sha256", "raster_text_profile_revision"}
    }
    fingerprint = runtime.raster_text_profile_fingerprint_sha256(material)
    payload["profile_fingerprint_sha256"] = fingerprint
    payload["raster_text_profile_revision"] = _PROFILE_PREFIX + fingerprint[:24]
    return payload


def _v9_pointer_fields() -> dict[str, object]:
    return {
        "metric_revision": "source-attested-semantic-text-v3",
        "evidence_revision": _EVIDENCE_REVISION,
        "latency_gate_revision": "warm-p95-10s-v1",
        "engine_kind": _V20_ENGINE,
        "raster_text_profile_revision": _PROFILE_PREFIX + "1" * 24,
        "runtime_configuration_path": "runtime/v20.json",
        "runtime_configuration_sha256": _SHA_A,
        "sealed_evaluation_path": "evaluation/v7.json",
        "sealed_evaluation_sha256": _SHA_B,
        "source_recognizer_release_sha256": _SHA_A,
        "source_recognizer_receipt_sha256": _SHA_B,
        "source_recognizer_training_intent_sha256": _SHA_C,
        "source_recognizer_selection_receipt_path": (
            f"synthetic-raster-text-v1/fresh-scope-selections/{_SHA_C}.json"
        ),
        "source_recognizer_selection_receipt_sha256": _SHA_C,
        "source_recognizer_parent_selection_receipt_sha256": _SHA_D,
        "source_recognizer_parent_release_sha256": _SHA_D,
        "source_recognizer_parent_receipt_sha256": _SHA_E,
        "source_recognizer_voyage_recipe_set_sha256": replay._RECIPE_SET_SHA256,
        "source_recognizer_train_manifest_sha256": _SHA_E,
        "source_recognizer_development_manifest_sha256": _SHA_F,
        "repair_audit_revision": "source-attested-independent-bl-repair-audit-v1",
        "repair_audit_dataset_receipt_path": _AUDIT_RECEIPT_PATH,
        "repair_audit_dataset_receipt_sha256": _SHA_C,
        "repair_audit_evaluation_path": (f"synthetic-bl-repair-audit-v1/evaluations/{_SHA_D}.json"),
        "repair_audit_evaluation_sha256": _SHA_D,
    }


def test_v20_profile_is_canonical_and_keeps_v19_wire_schema_unchanged() -> None:
    payload = _profile_payload()
    profile = runtime.RasterTextOfflineProfileV20.model_validate(payload)
    raw = runtime._canonical_evaluation_json_bytes(profile.model_dump(mode="json", warnings=False))

    parsed = runtime.parse_raster_text_offline_profile(raw)

    assert parsed == profile
    assert profile.schema_version == "20.0"
    assert profile.engine_kind == _V20_ENGINE
    assert profile.recognition_crop_policy == "horizontal-height-0p32-max8-v1"
    assert profile.recognition_batch_size == 8
    assert runtime.raster_text_service_startup_timeout_seconds(profile) == 60
    assert (
        runtime._sidecar_bootstrap_for_profile(profile)
        == runtime._DIRECT_PADDLE_V2_ISOLATED_SIDECAR_BOOTSTRAP
    )
    assert "source_recognizer_train_corpus_provenance" not in profile.model_dump(mode="json")
    assert runtime.RasterTextOfflineProfileV19.model_fields["schema_version"].default == "19.0"


def test_v20_provider_uses_only_the_direct_paddle_environment(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile = runtime.RasterTextOfflineProfileV20.model_validate(_profile_payload())
    monkeypatch.setattr(
        runtime,
        "_isolated_v2_sidecar_environment",
        lambda **_kwargs: {
            "PATH": "pinned",
            "CUSTOMS_AGENT_PPSTRUCTURE_CONFIG_PATH": "must-be-removed",
        },
    )

    environment = runtime._sidecar_environment(
        interpreter=tmp_path / "python.exe",
        module_root=tmp_path / "module",
        runtime_home=tmp_path / "home",
        configuration_path=tmp_path / "profile.json",
        configuration_sha256=_SHA_A,
        profile=profile,
        model_root=tmp_path / "models",
        model_inventory=tmp_path / "models.json",
        sidecar_inventory=tmp_path / "packages.json",
    )

    prefix = "CUSTOMS_AGENT_RASTER_TEXT_DIRECT_PADDLE_"
    assert environment[prefix + "CONFIG_SHA256"] == _SHA_A
    assert environment[prefix + "MODEL_INVENTORY_SHA256"] == profile.model_inventory_sha256
    assert not any(key.startswith("CUSTOMS_AGENT_PPSTRUCTURE_") for key in environment)


def _wrong_model(payload: dict[str, object]) -> None:
    payload["text_recognition_model_name"] = "source_attested_en_PP-OCRv5_mobile_rec_head_v2"


def _wrong_crop(payload: dict[str, object]) -> None:
    payload["recognition_crop_policy"] = "detector-quad-v1"


def _wrong_recipe(payload: dict[str, object]) -> None:
    payload["source_recognizer_voyage_recipe_set_sha256"] = _SHA_A


def _wrong_manifest_hash(payload: dict[str, object]) -> None:
    payload["source_recognizer_train_manifest_sha256"] = "0" * 64


@pytest.mark.parametrize(
    "mutate",
    [_wrong_model, _wrong_crop, _wrong_recipe, _wrong_manifest_hash],
    ids=["model", "crop", "recipe", "manifest"],
)
def test_v20_rejects_tuple_or_provenance_drift_with_rebound_fingerprint(
    mutate: Callable[[dict[str, object]], None],
) -> None:
    payload = _profile_payload()
    mutate(payload)
    _rebind_profile(payload)

    with pytest.raises(ValidationError):
        runtime.RasterTextOfflineProfileV20.model_validate(payload)


def test_v20_dispatches_only_to_the_voyage_replay_bundle_verifier(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile = runtime.RasterTextOfflineProfileV20.model_construct()
    calls: list[str] = []
    monkeypatch.setattr(
        runtime,
        "_verify_v20_public_voyage_replay_model_bundle",
        lambda **_kwargs: calls.append("voyage-replay"),
    )
    monkeypatch.setattr(
        runtime,
        "_verify_v19_public_finetuned_model_bundle",
        lambda **_kwargs: calls.append("v19"),
    )
    monkeypatch.setattr(
        runtime,
        "_verify_v3_mobile_model_bundle",
        lambda **_kwargs: calls.append("mobile"),
    )

    runtime._verify_profile_model_bundle(root=tmp_path, profile=profile)

    assert calls == ["voyage-replay"]


def test_v20_bundle_verifier_checks_every_public_profile_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile = runtime.RasterTextOfflineProfileV20.model_validate(_profile_payload())
    train = profile.source_recognizer_train_corpus_manifest
    development = profile.source_recognizer_development_corpus_manifest
    manifest = SimpleNamespace(
        local_model_root_relative_path="models",
        model_inventory_relative_path="inventories/direct-paddle-public-voyage-replay-models.json",
        recognizer_receipt_provenance_relative_path=(
            "provenance/public-voyage-replay-recognizer-training-receipt.json"
        ),
        source_recognizer_release_sha256=profile.source_recognizer_release_sha256,
        source_recognizer_receipt_sha256=profile.source_recognizer_receipt_sha256,
        source_recognizer_training_intent_sha256=(profile.source_recognizer_training_intent_sha256),
        source_recognizer_parent_release_sha256=profile.source_recognizer_parent_release_sha256,
        source_recognizer_parent_receipt_sha256=profile.source_recognizer_parent_receipt_sha256,
        source_recognizer_train_manifest_sha256=profile.source_recognizer_train_manifest_sha256,
        source_recognizer_development_manifest_sha256=(
            profile.source_recognizer_development_manifest_sha256
        ),
        source_recognizer_train_corpus_manifest=train,
        source_recognizer_development_corpus_manifest=development,
        source_recognizer_selection_receipt_relative_path=(
            profile.source_recognizer_selection_receipt_relative_path
        ),
        source_recognizer_selection_receipt_sha256=(
            profile.source_recognizer_selection_receipt_sha256
        ),
        source_recognizer_parent_selection_receipt_sha256=(
            profile.source_recognizer_parent_selection_receipt_sha256
        ),
        source_recognizer_voyage_recipe_set_sha256=(
            profile.source_recognizer_voyage_recipe_set_sha256
        ),
        source_recognizer_base_dictionary_sha256=(profile.source_recognizer_base_dictionary_sha256),
        source_recognizer_output_class_count=438,
        text_detection_model_name=profile.text_detection_model_name,
        text_recognition_model_name=profile.text_recognition_model_name,
    )

    class _Bundle:
        bundle_root_relative_path = profile.mobile_model_bundle_root_relative_path

        def __init__(self) -> None:
            self.manifest = manifest

        def bundle_relative_path(self, value: str) -> str:
            return f"{self.bundle_root_relative_path}/{value}"

        def file_sha256(self, value: str) -> str:
            if value == manifest.model_inventory_relative_path:
                return profile.model_inventory_sha256
            return profile.source_model_inventory_sha256

    monkeypatch.setattr(
        voyage_bundle,
        "load_verified_raster_text_public_voyage_replay_paddle_bundle_v1",
        lambda **_kwargs: _Bundle(),
    )

    runtime._verify_v20_public_voyage_replay_model_bundle(root=tmp_path, profile=profile)

    manifest.source_recognizer_parent_release_sha256 = "0" * 64
    with pytest.raises(
        runtime.RasterTextRuntimeError,
        match="raster_text_public_voyage_replay_model_bundle_invalid",
    ):
        runtime._verify_v20_public_voyage_replay_model_bundle(root=tmp_path, profile=profile)


def test_v9_pointer_is_canonical_and_cannot_cross_bind_v19_or_v8() -> None:
    pointer = runtime.RasterTextActivationPointerV9.model_validate(_v9_pointer_fields())
    raw = runtime._canonical_json_bytes(pointer.model_dump(mode="json", warnings=False))

    parsed = runtime.parse_raster_text_activation_pointer(raw)

    assert isinstance(parsed, runtime.RasterTextActivationPointerV9)
    invalid = pointer.model_dump(mode="json", warnings=False)
    invalid["engine_kind"] = _V19B_ENGINE
    with pytest.raises(runtime.RasterTextRuntimeError, match="raster_text_activation_invalid"):
        runtime.parse_raster_text_activation_pointer(runtime._canonical_json_bytes(invalid))
    invalid = pointer.model_dump(mode="json", warnings=False)
    del invalid["source_recognizer_train_manifest_sha256"]
    with pytest.raises(runtime.RasterTextRuntimeError, match="raster_text_activation_invalid"):
        runtime.parse_raster_text_activation_pointer(runtime._canonical_json_bytes(invalid))

    profile_v20 = runtime.RasterTextOfflineProfileV20.model_construct(engine_kind=_V20_ENGINE)
    runtime._verify_activation_pointer_engine_binding(pointer=pointer, profile=profile_v20)
    pointer_v8 = runtime.RasterTextActivationPointerV8.model_construct(engine_kind=_V19B_ENGINE)
    with pytest.raises(runtime.RasterTextRuntimeError, match="raster_text_profile_mismatch"):
        runtime._verify_activation_pointer_engine_binding(pointer=pointer_v8, profile=profile_v20)
    profile_v19b = runtime.RasterTextOfflineProfileV19.model_construct(engine_kind=_V19B_ENGINE)
    with pytest.raises(runtime.RasterTextRuntimeError, match="raster_text_profile_mismatch"):
        runtime._verify_activation_pointer_engine_binding(pointer=pointer, profile=profile_v19b)


def test_v20_evidence_adds_parent_recipe_and_manifest_binding(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    profile = runtime.RasterTextOfflineProfileV20.model_validate(_profile_payload())
    pointer = runtime.RasterTextActivationPointerV9.model_construct(
        source_recognizer_parent_release_sha256=profile.source_recognizer_parent_release_sha256,
        source_recognizer_parent_receipt_sha256=profile.source_recognizer_parent_receipt_sha256,
        source_recognizer_voyage_recipe_set_sha256=(
            profile.source_recognizer_voyage_recipe_set_sha256
        ),
        source_recognizer_train_manifest_sha256=(profile.source_recognizer_train_manifest_sha256),
        source_recognizer_development_manifest_sha256=(
            profile.source_recognizer_development_manifest_sha256
        ),
    )
    monkeypatch.setattr(runtime, "_verify_v19_finetune_evidence_provenance", lambda **_kwargs: None)
    evaluation = runtime.RasterTextSealedEvaluationV7.model_construct()
    calibration = runtime.RasterTextCalibrationReceiptV7.model_construct()

    runtime._verify_v20_voyage_replay_evidence_provenance(
        root=tmp_path,
        pointer=pointer,
        evaluation=evaluation,
        calibration=calibration,
        profile=profile,
        calibration_profile=profile,
        selection_raw=b"selection",
    )

    drifted = pointer.model_copy(update={"source_recognizer_voyage_recipe_set_sha256": "0" * 64})
    with pytest.raises(runtime.RasterTextRuntimeError, match="raster_text_evaluation_invalid"):
        runtime._verify_v20_voyage_replay_evidence_provenance(
            root=tmp_path,
            pointer=drifted,
            evaluation=evaluation,
            calibration=calibration,
            profile=profile,
            calibration_profile=profile,
            selection_raw=b"selection",
        )


def test_pointer_writer_derives_v9_only_with_the_complete_independent_audit(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    profile = runtime.RasterTextOfflineProfileV20.model_validate(_profile_payload())
    configuration_raw = b"v20-configuration"
    sealed_raw = b"sealed-v7"
    receipt_raw = b"audit-receipt"
    audit_evaluation_raw = b"audit-evaluation"
    configuration_sha256 = hashlib.sha256(configuration_raw).hexdigest()
    receipt_sha256 = hashlib.sha256(receipt_raw).hexdigest()
    audit_evaluation_sha256 = hashlib.sha256(audit_evaluation_raw).hexdigest()
    audit_evaluation_path = (
        f"synthetic-bl-repair-audit-v1/evaluations/{audit_evaluation_sha256}.json"
    )
    evaluation = runtime.RasterTextSealedEvaluationV7.model_construct(
        raster_text_profile_revision=profile.raster_text_profile_revision,
        extractor_revision=profile.extractor_revision,
        runtime_configuration_sha256=configuration_sha256,
        metric_revision="source-attested-semantic-text-v3",
        evidence_revision=_EVIDENCE_REVISION,
        latency_gate_revision="warm-p95-10s-v1",
    )
    repair_evaluation = SimpleNamespace(
        dataset_receipt_sha256=receipt_sha256,
        extractor_revision=profile.extractor_revision,
        gate_passed=True,
        scores=[SimpleNamespace(gate_passed=True)],
    )

    class _ReceiptParser:
        @classmethod
        def model_validate_json(cls, raw: bytes) -> SimpleNamespace:
            assert raw == receipt_raw
            return SimpleNamespace()

    def read_material(path: Path, *, limit: int) -> bytes:
        assert limit in {runtime._MAX_CONFIGURATION_BYTES, runtime._MAX_EVIDENCE_BYTES}
        if path.name == "v20.json":
            return configuration_raw
        if path.name == "v7.json":
            return sealed_raw
        if path.name == "receipt.json":
            return receipt_raw
        return audit_evaluation_raw

    monkeypatch.setattr(runtime, "_read_local_file", read_material)
    monkeypatch.setattr(
        runtime, "_sidecar_path", lambda base, value: base.joinpath(*value.split("/"))
    )
    monkeypatch.setattr(runtime, "parse_raster_text_offline_profile", lambda _raw: profile)
    monkeypatch.setattr(runtime, "_parse_raster_text_sealed_evaluation", lambda _raw: evaluation)
    monkeypatch.setattr(runtime, "_load_binding_from_pointer", lambda **_kwargs: None)
    monkeypatch.setattr(runtime, "_atomic_replace_raster_text_pointer", lambda **_kwargs: None)
    monkeypatch.setattr(audit, "BlRepairAuditReceiptV1", _ReceiptParser)
    monkeypatch.setattr(audit, "bl_repair_audit_receipt_sha256", lambda _value: receipt_sha256)
    monkeypatch.setattr(
        audit,
        "bl_repair_audit_evaluation_sha256",
        lambda _value: audit_evaluation_sha256,
    )
    monkeypatch.setattr(
        audit, "verify_bl_repair_audit_evaluation", lambda **_kwargs: repair_evaluation
    )

    _digest, pointer = runtime.write_source_attested_synthetic_raster_text_pointer(
        document_preview_root=root,
        runtime_configuration_relative_path="runtime/v20.json",
        sealed_evaluation_relative_path="evaluation/v7.json",
        repair_audit_dataset_receipt_relative_path=_AUDIT_RECEIPT_PATH,
        repair_audit_evaluation_relative_path=audit_evaluation_path,
    )

    assert isinstance(pointer, runtime.RasterTextActivationPointerV9)
    assert pointer.engine_kind == _V20_ENGINE
    assert (
        pointer.source_recognizer_parent_release_sha256
        == profile.source_recognizer_parent_release_sha256
    )
    assert (
        pointer.source_recognizer_train_manifest_sha256
        == profile.source_recognizer_train_manifest_sha256
    )

    with pytest.raises(
        runtime.RasterTextRuntimeError, match="raster_text_activation_arguments_invalid"
    ):
        runtime.write_source_attested_synthetic_raster_text_pointer(
            document_preview_root=root,
            runtime_configuration_relative_path="runtime/v20.json",
            sealed_evaluation_relative_path="evaluation/v7.json",
        )
