"""Focused tests for the immutable public-fine-tuned Paddle bundle."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from customs_ai_agent.evaluation import raster_text_mobile_paddle_bundle as mobile_bundle
from customs_ai_agent.evaluation import (
    raster_text_public_finetuned_paddle_bundle as public_bundle,
)
from customs_ai_agent.evaluation import (
    source_attested_public_raster_text_recognizer_finetuning as fine_tune,
)
from customs_ai_agent.evaluation.raster_text_public_finetuned_paddle_bundle import (
    RasterTextPublicFineTunedPaddleBundleError,
    load_verified_raster_text_public_finetuned_paddle_bundle_v1,
    materialize_verified_raster_text_public_finetuned_paddle_bundle_v1,
    prepare_raster_text_public_finetuned_one_pass_runtime_profile_v19,
    prepare_raster_text_public_finetuned_runtime_profile_v19,
)


_OUTPUT_MODEL_NAME = "source_attested_en_PP-OCRv5_mobile_rec_head_v2"
_COLLECTION = "raster-text-public-finetuned-paddle-bundles-v1"
_INVENTORY = "inventories/direct-paddle-public-finetuned-models.json"
_MODEL_FILENAMES = (
    "config.json",
    "inference.json",
    "inference.pdiparams",
    "inference.yml",
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def _preview_root(tmp_path: Path) -> Path:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    return root


def _base_characters() -> tuple[str, ...]:
    # The public recognizer contract requires 436 unique, single-code-point
    # entries and reserves ASCII space as its implicit final CTC class.
    return tuple(chr(0x1000 + index) for index in range(436))


def _make_mobile_bundle(
    root: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    characters: tuple[str, ...],
) -> mobile_bundle.VerifiedRasterTextDirectMobilePaddleBundleV1:
    rows: list[dict[str, object]] = []
    for index, (relative, _target) in enumerate(mobile_bundle._MOBILE_MODEL_ARTIFACTS, start=1):
        if relative.endswith("det/config.json"):
            raw = _canonical({"Global": {"model_name": "PP-OCRv5_mobile_det"}})
        elif relative.endswith("rec/config.json"):
            raw = _canonical(
                {
                    "Global": {"model_name": "en_PP-OCRv5_mobile_rec"},
                    "PostProcess": {
                        "character_dict": list(characters),
                        "name": "CTCLabelDecode",
                    },
                }
            )
        else:
            raw = f"public-mobile-model-{index}".encode("ascii")
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
    inventory_raw = (
        _canonical(
            {
                "schema_version": "1.0",
                "engine_kind": "paddleocr",
                "runtime_api": "paddleocr3-paddlex-3.2",
                "packages": [],
                "artifacts": rows,
            }
        )
        + b"\n"
    )
    inventory_sha = hashlib.sha256(inventory_raw).hexdigest()
    _write(root / "model-inventories" / f"{inventory_sha}.json", inventory_raw)
    monkeypatch.setattr(
        mobile_bundle,
        "_MOBILE_SOURCE_INVENTORY_RELATIVE_PATH",
        f"model-inventories/{inventory_sha}.json",
    )
    monkeypatch.setattr(mobile_bundle, "_MOBILE_SOURCE_INVENTORY_SHA256", inventory_sha)
    return mobile_bundle.materialize_verified_raster_text_direct_mobile_paddle_bundle_v1(
        document_preview_root=root
    )


def _metric(crop_count: int, cer: float, exact: float) -> dict[str, object]:
    return {
        "crop_count": crop_count,
        "reference_character_count": crop_count * 5,
        "semantic_cer": cer,
        "semantic_crop_exact_match": exact,
    }


def _fine_tune_receipt_payload_v1(
    stage: Path,
    *,
    mobile: mobile_bundle.VerifiedRasterTextDirectMobilePaddleBundleV1,
    dictionary_sha256: str,
) -> dict[str, object]:
    artifacts = {
        relative: fine_tune.sha256_file(stage / relative) for relative in fine_tune._MODEL_FILES
    }
    candidate_metrics = {
        profile: _metric(10, 0.09, 0.81) for profile in fine_tune._TRAINING_PROFILES
    }
    baseline_metrics = {
        profile: _metric(10, 0.10, 0.80) for profile in fine_tune._TRAINING_PROFILES
    }
    crop_manifests = {
        split: fine_tune.PublicFineTuneCropManifestV1(
            split=split,
            source_crop_manifest_sha256=("b" if split == "train" else "c") * 64,
            source_receipt_sha256="d" * 64,
            raster_receipt_sha256="e" * 64,
            family_count=3,
            source_crop_count=30,
            usable_crop_count=30,
            skipped_decorative_crop_count=0,
            semantic_crop_list_sha256="f" * 64,
            skipped_sample_list_sha256="1" * 64,
            base_dictionary_sha256=dictionary_sha256,
            source_profile_crop_counts={profile: 10 for profile in fine_tune._TRAINING_PROFILES},
            usable_profile_crop_counts={profile: 10 for profile in fine_tune._TRAINING_PROFILES},
            scan_duplicate_pair_count=1,
            scan_duplicate_pixel_list_sha256="2" * 64,
        )
        for split in ("train", "development")
    }
    return {
        "artifact_kind": "source_attested_public_recognizer_finetune_receipt",
        "trainer_revision": "public-paddle-jit-ctc-domain-finetune-v1",
        "model_revision": "en-ppocrv5-mobile-rec-source-attested-v1",
        "semantic_label_revision": "source-attested-ocr-crop-semantic-label-v1",
        "stage": "overfit",
        "tuning_scope": "head",
        "training_intent_sha256": "a" * 64,
        "train_manifest_sha256": fine_tune._manifest_sha256(crop_manifests["train"]),
        "development_manifest_sha256": fine_tune._manifest_sha256(crop_manifests["development"]),
        "train_crop_manifest": crop_manifests["train"].model_dump(mode="json"),
        "development_crop_manifest": crop_manifests["development"].model_dump(mode="json"),
        "direct_runtime_bundle_manifest_sha256": "d" * 64,
        "direct_runtime_package_inventory_sha256": "e" * 64,
        "public_mobile_model_bundle_manifest_sha256": mobile.manifest_sha256,
        "public_mobile_source_inventory_sha256": (mobile.manifest.source_model_inventory_sha256),
        "public_mobile_source_snapshot_sha256": (mobile.manifest.source_model_snapshot_sha256),
        "public_recognizer_artifact_sha256s": {
            relative: mobile.file_sha256(relative) for relative in fine_tune._BASE_RECOGNIZER_FILES
        },
        "base_dictionary_sha256": dictionary_sha256,
        "base_character_count": 436,
        "output_class_count": 438,
        "ctc_blank_index": 0,
        "dictionary_first_class_index": 1,
        "implicit_space_class_index": 437,
        "form_marker_policy": "remove-black-square-checkbox-and-underscore-run-ge-2-v1",
        "whitespace_policy": ("nfc-map-control-whitespace-one-for-one-preserve-ascii-spaces-v1"),
        "training_module_sha256": "8" * 64,
        "balanced_corpus_module_sha256": "9" * 64,
        "source_crop_base_module_sha256": "0" * 64,
        "head_tensor_contract_sha256": fine_tune.sha256_bytes(
            fine_tune.canonical_json_bytes(fine_tune._EXPECTED_HEAD_SHAPES)
        ),
        "trainable_parameter_name_list_sha256": fine_tune.sha256_bytes(
            fine_tune.canonical_json_bytes(sorted(fine_tune._EXPECTED_HEAD_SHAPES))
        ),
        "parent_python_revision": (
            f"{fine_tune.sys.version_info.major}.{fine_tune.sys.version_info.minor}"
        ),
        "seed_sha256": "1" * 64,
        "batch_size": 32,
        "maximum_epochs": 6,
        "completed_epochs": 2,
        "best_epoch": 1,
        "early_stopped": True,
        "learning_rate": 1.0e-4,
        "trainable_tensor_count": 2,
        "trainable_parameter_count": 52_998,
        "development_semantic_cer": 0.09,
        "development_semantic_crop_exact_match": 0.81,
        "baseline_development_semantic_cer": 0.10,
        "baseline_development_semantic_crop_exact_match": 0.80,
        "baseline_development_by_profile": baseline_metrics,
        "best_development_by_profile": candidate_metrics,
        "improvement_contract": "overall-cer-then-crop-exact-strict-v1",
        "evaluation_contract": "semantic_crop_cer_and_exact_without_critical_fields_v1",
        "initialization": "pinned_public_inference_graph_warm_start",
        "framework_license": "Apache-2.0",
        "static_roundtrip_verified": True,
        "model_artifact_sha256s": artifacts,
    }


def _fine_tune_receipt_payload_v2(
    stage: Path,
    *,
    mobile: mobile_bundle.VerifiedRasterTextDirectMobilePaddleBundleV1,
    dictionary_sha256: str,
) -> dict[str, object]:
    payload = _fine_tune_receipt_payload_v1(
        stage, mobile=mobile, dictionary_sha256=dictionary_sha256
    )
    selection_sha = "6" * 64
    parent_selection_sha = "7" * 64
    selection_path = f"synthetic-raster-text-v1/fresh-scope-selections/{selection_sha}.json"
    profile_counts = {
        "train": {
            "png-216": 334,
            "jpeg-216-q75": 333,
            "jpeg-150-q75": 333,
        },
        "development": {
            "png-216": 68,
            "jpeg-216-q75": 66,
            "jpeg-150-q75": 66,
        },
    }
    crop_manifests: dict[str, fine_tune.PublicFineTuneCropManifestV2] = {}
    for split in ("train", "development"):
        family_count = 500 if split == "train" else 100
        crop_count = family_count * 2
        first_third = crop_count // 3
        critical_field_crop_counts = {
            "consignee": first_third,
            "invoice_number": first_third,
            "total_packages": crop_count - (2 * first_third),
        }
        document_type_crop_counts = {
            "bill_of_lading": first_third,
            "commercial_invoice": first_third,
            "packing_list": crop_count - (2 * first_third),
        }
        crop_manifests[split] = fine_tune.PublicFineTuneCropManifestV2(
            split=split,
            source_crop_manifest_sha256=("b" if split == "train" else "c") * 64,
            source_receipt_sha256="d" * 64,
            raster_receipt_sha256="e" * 64,
            selection_receipt_relative_path=selection_path,
            selection_receipt_sha256=selection_sha,
            parent_selection_receipt_sha256=parent_selection_sha,
            selected_family_ids_sha256=("8" if split == "train" else "9") * 64,
            excluded_family_ids_sha256="0" * 64,
            family_count=family_count,
            source_crop_count=crop_count,
            usable_crop_count=crop_count,
            semantic_crop_list_sha256=("1" if split == "train" else "2") * 64,
            skipped_sample_list_sha256="3" * 64,
            base_dictionary_sha256=dictionary_sha256,
            source_profile_crop_counts=profile_counts[split],
            usable_profile_crop_counts=profile_counts[split],
            critical_field_crop_count=crop_count,
            critical_field_ids_sha256=fine_tune.sha256_bytes(
                fine_tune.canonical_json_bytes(sorted(critical_field_crop_counts))
            ),
            critical_field_crop_counts=critical_field_crop_counts,
            document_type_crop_counts=document_type_crop_counts,
            source_label_length_p50=8,
            source_label_length_p95=16,
        )
    candidate_metrics = {
        profile: _metric(count, 0.09, 0.81)
        for profile, count in profile_counts["development"].items()
    }
    baseline_metrics = {
        profile: _metric(count, 0.10, 0.80)
        for profile, count in profile_counts["development"].items()
    }
    payload.update(
        {
            "trainer_revision": "public-paddle-jit-ctc-leakage-safe-head-v2",
            "model_revision": "en-ppocrv5-mobile-rec-source-attested-head-v2",
            "stage": "leakage-safe-v2",
            "tuning_scope": "head",
            "train_manifest_sha256": fine_tune._manifest_sha256(crop_manifests["train"]),
            "development_manifest_sha256": fine_tune._manifest_sha256(
                crop_manifests["development"]
            ),
            "train_crop_manifest": crop_manifests["train"].model_dump(mode="json"),
            "development_crop_manifest": crop_manifests["development"].model_dump(mode="json"),
            "selection_receipt_relative_path": selection_path,
            "selection_receipt_sha256": selection_sha,
            "parent_selection_receipt_sha256": parent_selection_sha,
            "overlap_verifier_revision": "selection-v2-all-holdouts-zero-overlap-v1",
            "train_holdout_overlap_count": 0,
            "maximum_epochs": 4,
            "completed_epochs": 2,
            "best_epoch": 1,
            "baseline_development_by_profile": baseline_metrics,
            "best_development_by_profile": candidate_metrics,
        }
    )
    return payload


def _make_recognizer_release(
    root: Path,
    *,
    mobile: mobile_bundle.VerifiedRasterTextDirectMobilePaddleBundleV1,
    characters: tuple[str, ...],
    legacy_v1: bool = False,
) -> fine_tune.VerifiedPublicFineTunedRecognizerReleaseV1:
    stage = fine_tune.create_public_finetune_stage_v1(root)
    (stage / "text-rec").mkdir()
    configuration = {
        "Global": {"model_name": _OUTPUT_MODEL_NAME},
        "PostProcess": {
            "character_dict": list(characters),
            "name": "CTCLabelDecode",
        },
        "PreProcess": {},
    }
    _write(
        stage / "text-rec" / "config.json",
        fine_tune.canonical_json_file_bytes(configuration),
    )
    _write(stage / "text-rec" / "inference.json", b"fine-tuned-inference-graph")
    _write(stage / "text-rec" / "inference.pdiparams", b"fine-tuned-parameters")
    _write(
        stage / "text-rec" / "inference.yml",
        f"Global:\n  model_name: {_OUTPUT_MODEL_NAME}\n".encode("utf-8"),
    )
    dictionary_sha = fine_tune.sha256_bytes(fine_tune.canonical_json_bytes(list(characters)))
    if legacy_v1:
        return fine_tune.publish_public_finetuned_recognizer_release_v1(
            root,
            staged_release=stage,
            receipt_payload=_fine_tune_receipt_payload_v1(
                stage, mobile=mobile, dictionary_sha256=dictionary_sha
            ),
        )
    return fine_tune.publish_public_finetuned_recognizer_release_v2(
        root,
        staged_release=stage,
        receipt_payload=_fine_tune_receipt_payload_v2(
            stage, mobile=mobile, dictionary_sha256=dictionary_sha
        ),
    )


@dataclass(frozen=True)
class _Sources:
    root: Path
    mobile: mobile_bundle.VerifiedRasterTextDirectMobilePaddleBundleV1
    recognizer: fine_tune.VerifiedPublicFineTunedRecognizerReleaseV1


@pytest.fixture
def sources(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> _Sources:
    root = _preview_root(tmp_path)
    characters = _base_characters()
    mobile = _make_mobile_bundle(root, monkeypatch, characters=characters)
    recognizer = _make_recognizer_release(root, mobile=mobile, characters=characters)
    return _Sources(root=root, mobile=mobile, recognizer=recognizer)


def _materialize(sources: _Sources) -> Any:
    return materialize_verified_raster_text_public_finetuned_paddle_bundle_v1(
        mobile_bundle=sources.mobile,
        recognizer_release=sources.recognizer,
    )


def test_materialize_load_and_idempotency_preserve_the_exact_combined_tree(
    sources: _Sources,
) -> None:
    assert isinstance(sources.recognizer.receipt, fine_tune.PublicFineTuneReceiptV2)
    assert sources.recognizer.receipt.stage == "leakage-safe-v2"
    assert sources.recognizer.receipt.maximum_epochs == 4
    assert sources.recognizer.receipt.train_crop_manifest.family_count == 500
    assert sources.recognizer.receipt.train_crop_manifest.usable_crop_count == 1_000
    assert sources.recognizer.receipt.development_crop_manifest.family_count == 100
    assert sources.recognizer.receipt.development_crop_manifest.usable_crop_count == 200
    assert sources.recognizer.receipt.train_holdout_overlap_count == 0
    first = _materialize(sources)
    second = _materialize(sources)
    loaded = load_verified_raster_text_public_finetuned_paddle_bundle_v1(
        document_preview_root=sources.root,
        manifest_sha256=first.manifest_sha256,
    )

    assert first.manifest_sha256 == second.manifest_sha256 == loaded.manifest_sha256
    assert os.path.samefile(first.bundle_root, second.bundle_root)
    assert loaded.bundle_root_relative_path == (f"{_COLLECTION}/releases/{loaded.manifest_sha256}")
    assert loaded.manifest.bundle_id == (
        "raster_text_public_finetuned_paddle_combined_model_bundle_v1"
    )
    assert loaded.manifest.source_mobile_bundle_manifest_sha256 == sources.mobile.manifest_sha256
    records = {item.path: item for item in loaded.manifest.files}
    detector_snapshot = hashlib.sha256(
        _canonical(
            {
                "artifacts": [
                    {
                        "byte_size": records[f"models/text-det/{name}"].byte_size,
                        "relative_path": f"models/text-det/{name}",
                        "sha256": records[f"models/text-det/{name}"].sha256,
                    }
                    for name in _MODEL_FILENAMES
                ]
            }
        )
    ).hexdigest()
    assert loaded.manifest.source_detector_model_snapshot_sha256 == detector_snapshot
    assert loaded.manifest.source_recognizer_release_sha256 == sources.recognizer.release_sha256
    assert (
        loaded.manifest.source_recognizer_receipt_sha256
        == sources.recognizer.receipt.receipt_sha256
    )
    assert loaded.manifest.source_recognizer_training_intent_sha256 == "a" * 64
    assert (
        loaded.manifest.source_recognizer_base_dictionary_sha256
        == sources.recognizer.receipt.base_dictionary_sha256
    )
    assert loaded.manifest.source_recognizer_output_class_count == 438
    assert (
        loaded.manifest.source_recognizer_train_corpus_provenance.critical_field_crop_count == 1_000
    )
    assert (
        loaded.manifest.source_recognizer_development_corpus_provenance.critical_field_crop_count
        == 200
    )
    assert loaded.manifest.source_recognizer_train_corpus_provenance.source_label_length_p95 == 16
    assert loaded.manifest.text_detection_model_name == "PP-OCRv5_mobile_det"
    assert loaded.manifest.text_recognition_model_name == _OUTPUT_MODEL_NAME
    assert loaded.manifest.local_model_root_relative_path == "models"
    assert loaded.manifest.model_inventory_relative_path == _INVENTORY

    actual = sorted(
        path.relative_to(loaded.bundle_root).as_posix()
        for path in loaded.bundle_root.rglob("*")
        if path.is_file()
    )
    expected = sorted(
        {
            "manifest.json",
            _INVENTORY,
            *(f"models/text-det/{name}" for name in _MODEL_FILENAMES),
            *(f"models/text-rec/{name}" for name in _MODEL_FILENAMES),
            "provenance/mobile-detector-bundle-manifest.json",
            "provenance/mobile-detector-model-inventory.json",
            "provenance/public-finetuned-recognizer-release-manifest.json",
            "provenance/public-finetuned-recognizer-training-receipt.json",
        }
    )
    assert actual == expected
    assert (
        loaded.bundle_root.joinpath("provenance/mobile-detector-bundle-manifest.json").read_bytes()
        == (sources.mobile.bundle_root / "manifest.json").read_bytes()
    )
    assert (
        loaded.bundle_root.joinpath(
            "provenance/public-finetuned-recognizer-training-receipt.json"
        ).read_bytes()
        == (sources.recognizer.release_path / "training-receipt.json").read_bytes()
    )
    assert not list(sources.root.glob(f".{_COLLECTION}.staging-*"))


def test_materializer_rejects_a_legacy_v1_overfit_recognizer_release(
    sources: _Sources,
) -> None:
    legacy_release = _make_recognizer_release(
        sources.root,
        mobile=sources.mobile,
        characters=_base_characters(),
        legacy_v1=True,
    )
    assert isinstance(legacy_release.receipt, fine_tune.PublicFineTuneReceiptV1)
    assert not isinstance(legacy_release.receipt, fine_tune.PublicFineTuneReceiptV2)
    assert legacy_release.receipt.stage == "overfit"

    with pytest.raises(RasterTextPublicFineTunedPaddleBundleError) as raised:
        materialize_verified_raster_text_public_finetuned_paddle_bundle_v1(
            mobile_bundle=sources.mobile,
            recognizer_release=legacy_release,
        )

    assert raised.value.code == "public_finetuned_paddle_bundle_source_invalid"


def test_materializer_rejects_recognizer_source_drift_before_release_reuse(
    sources: _Sources,
) -> None:
    _materialize(sources)
    sources.recognizer.file_paths["text-rec/inference.pdiparams"].write_bytes(b"drift")

    with pytest.raises(RasterTextPublicFineTunedPaddleBundleError) as raised:
        _materialize(sources)

    assert raised.value.code == "public_finetuned_paddle_bundle_source_drift"


def test_materializer_rejects_sources_from_different_preview_roots(
    sources: _Sources,
    tmp_path: Path,
) -> None:
    foreign_root = _preview_root(tmp_path / "foreign")
    foreign_release = _make_recognizer_release(
        foreign_root,
        mobile=sources.mobile,
        characters=_base_characters(),
    )

    with pytest.raises(RasterTextPublicFineTunedPaddleBundleError) as raised:
        materialize_verified_raster_text_public_finetuned_paddle_bundle_v1(
            mobile_bundle=sources.mobile,
            recognizer_release=foreign_release,
        )

    assert raised.value.code == "public_finetuned_paddle_bundle_source_binding_invalid"


def test_loader_rejects_model_drift(sources: _Sources) -> None:
    bundle = _materialize(sources)
    (bundle.bundle_root / "models" / "text-rec" / "inference.json").write_bytes(b"drift")
    with pytest.raises(RasterTextPublicFineTunedPaddleBundleError) as raised:
        load_verified_raster_text_public_finetuned_paddle_bundle_v1(
            document_preview_root=sources.root,
            manifest_sha256=bundle.manifest_sha256,
        )
    assert raised.value.code in {
        "public_finetuned_paddle_bundle_bundle_drift",
        "public_finetuned_paddle_bundle_manifest_invalid",
    }


def test_loader_rejects_an_extra_file(sources: _Sources) -> None:
    bundle = _materialize(sources)
    (bundle.bundle_root / "unexpected.bin").write_bytes(b"extra")

    with pytest.raises(RasterTextPublicFineTunedPaddleBundleError) as raised:
        load_verified_raster_text_public_finetuned_paddle_bundle_v1(
            document_preview_root=sources.root,
            manifest_sha256=bundle.manifest_sha256,
        )

    assert raised.value.code == "public_finetuned_paddle_bundle_manifest_invalid"


@dataclass(frozen=True)
class _FakeSidecar:
    document_preview_root: Path
    bundle_root_relative_path: str
    manifest_sha256: str
    manifest: Any
    hashes: dict[str, str]

    def bundle_relative_path(self, path: str) -> str:
        return f"{self.bundle_root_relative_path}/{path}"

    def file_sha256(self, path: str) -> str:
        return self.hashes[path]


def _fake_sidecar(root: Path) -> _FakeSidecar:
    manifest_sha = "8" * 64
    bundle_root = f"raster-text-direct-bundles-v1/releases/{manifest_sha}"
    paths = {
        "inventories/parent-renderer.json": "1" * 64,
        "sidecar/interpreter/python.exe": "2" * 64,
        "sidecar/module-root/customs_ai_agent/service/raster_text_direct_paddle_sidecar.py": "3"
        * 64,
        "inventories/direct-paddle-packages.json": "4" * 64,
    }
    manifest = SimpleNamespace(
        extractor_revision="direct-paddle-test-v2",
        raster_dpi=216,
        parent_renderer_package_inventory_relative_path=("inventories/parent-renderer.json"),
        parent_renderer_required_packages=("pypdfium2", "Pillow"),
        sidecar_interpreter_relative_path="sidecar/interpreter/python.exe",
        sidecar_interpreter_root_relative_path="sidecar/interpreter",
        sidecar_interpreter_root_tree_sha256="5" * 64,
        sidecar_module_relative_path=(
            "sidecar/module-root/customs_ai_agent/service/raster_text_direct_paddle_sidecar.py"
        ),
        sidecar_module_root_relative_path="sidecar/module-root",
        sidecar_module_root_tree_sha256="6" * 64,
        sidecar_package_inventory_relative_path=("inventories/direct-paddle-packages.json"),
        sidecar_required_packages=(
            "numpy",
            "opencv-contrib-python",
            "paddlepaddle",
            "pyclipper",
        ),
    )
    return _FakeSidecar(root, bundle_root, manifest_sha, manifest, paths)


def test_v19_profile_is_exact_content_addressed_and_rejects_invalid_cpu_controls(
    sources: _Sources,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle = _materialize(sources)
    sidecar = _fake_sidecar(sources.root)

    def load_sidecar(*, document_preview_root: Path, manifest_sha256: str) -> _FakeSidecar:
        assert document_preview_root.resolve() == sources.root.resolve()
        assert manifest_sha256 == sidecar.manifest_sha256
        return sidecar

    monkeypatch.setattr(
        public_bundle,
        "load_verified_raster_text_direct_paddle_bundle_v1",
        load_sidecar,
    )

    prepared = prepare_raster_text_public_finetuned_runtime_profile_v19(
        sidecar_bundle=sidecar,  # type: ignore[arg-type]
        model_bundle=bundle,
        pdfium_version="4.30.0",
        minimum_page_confidence=0.73,
        cpu_threads=6,
    )
    again = prepare_raster_text_public_finetuned_runtime_profile_v19(
        sidecar_bundle=sidecar,  # type: ignore[arg-type]
        model_bundle=bundle,
        pdfium_version="4.30.0",
        minimum_page_confidence=0.73,
        cpu_threads=6,
    )
    profile_path = sources.root.joinpath(*prepared.runtime_configuration_relative_path.split("/"))
    raw = profile_path.read_bytes()
    profile = json.loads(raw)

    assert prepared == again
    assert hashlib.sha256(raw).hexdigest() == prepared.runtime_configuration_sha256
    assert profile["schema_version"] == "19.0"
    assert profile["engine_kind"] == (
        "direct_paddle_mobile_det_public_finetuned_rec_harmonic_recall_native960_"
        "lowbox_strip2_semantic_repair_v2_voyage_retry_onednn"
    )
    assert profile["extractor_revision"] == (
        "direct-paddle-test-v2:mobile-det-v19b-public-finetuned-rec-native960-"
        "lowbox-strip2-semantic-repair-v2-voyage-retry-onednn"
    )
    assert profile["text_detection_model_name"] == "PP-OCRv5_mobile_det"
    assert profile["text_recognition_model_name"] == _OUTPUT_MODEL_NAME
    assert profile["ctc_label_policy"] == "optional-space-v1"
    assert profile["local_model_root"] == f"{bundle.bundle_root_relative_path}/models"
    assert profile["model_inventory_path"] == (f"{bundle.bundle_root_relative_path}/{_INVENTORY}")
    assert profile["mobile_model_bundle_manifest_sha256"] == bundle.manifest_sha256
    assert profile["source_recognizer_selection_receipt_relative_path"] == (
        sources.recognizer.receipt.selection_receipt_relative_path
    )
    assert profile["source_recognizer_selection_receipt_sha256"] == (
        sources.recognizer.receipt.selection_receipt_sha256
    )
    assert profile["source_recognizer_parent_selection_receipt_sha256"] == (
        sources.recognizer.receipt.parent_selection_receipt_sha256
    )
    assert profile["source_recognizer_train_corpus_provenance"] == (
        bundle.manifest.source_recognizer_train_corpus_provenance.model_dump(mode="json")
    )
    assert profile["source_recognizer_development_corpus_provenance"] == (
        bundle.manifest.source_recognizer_development_corpus_provenance.model_dump(mode="json")
    )
    assert profile["cpu_threads"] == 6
    assert profile["recognition_batch_size"] == 8
    assert profile["opencv_threads"] == 1
    assert profile["enable_mkldnn"] is True
    assert profile["onednn_model_policy"] == "recognizer-only-v1"
    assert profile["sidecar_timeout_seconds"] == 30
    assert profile["minimum_page_confidence"] == 0.73
    assert prepared.raster_text_profile_revision == (
        f"synthetic-raster-text-v1:{profile['profile_fingerprint_sha256'][:24]}"
    )

    one_pass = prepare_raster_text_public_finetuned_one_pass_runtime_profile_v19(
        sidecar_bundle=sidecar,  # type: ignore[arg-type]
        model_bundle=bundle,
        pdfium_version="4.30.0",
        minimum_page_confidence=0.73,
        cpu_threads=6,
    )
    one_pass_raw = sources.root.joinpath(
        *one_pass.runtime_configuration_relative_path.split("/")
    ).read_bytes()
    one_pass_profile = json.loads(one_pass_raw)
    assert one_pass_profile["engine_kind"] == (
        "direct_paddle_mobile_det_public_finetuned_rec_harmonic_recall_native960_"
        "lowbox_strip2_semantic_repair_v2_onednn"
    )
    assert one_pass_profile["extractor_revision"] == (
        "direct-paddle-test-v2:mobile-det-v19-public-finetuned-rec-native960-"
        "lowbox-strip2-semantic-repair-v2-onednn"
    )
    ignored = {
        "engine_kind",
        "extractor_revision",
        "profile_fingerprint_sha256",
        "raster_text_profile_revision",
    }
    assert {key: value for key, value in one_pass_profile.items() if key not in ignored} == {
        key: value for key, value in profile.items() if key not in ignored
    }
    assert one_pass.runtime_configuration_sha256 != prepared.runtime_configuration_sha256

    with pytest.raises(RasterTextPublicFineTunedPaddleBundleError) as raised:
        prepare_raster_text_public_finetuned_runtime_profile_v19(
            sidecar_bundle=sidecar,  # type: ignore[arg-type]
            model_bundle=bundle,
            pdfium_version="4.30.0",
            minimum_page_confidence=0.73,
            cpu_threads=8,
        )
    assert raised.value.code == "public_finetuned_paddle_bundle_runtime_profile_invalid"
