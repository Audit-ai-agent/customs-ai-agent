"""Focused tests for the immutable Voyage-replay Paddle bundle."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from typing import Any, Literal

import pytest

from customs_ai_agent.evaluation import raster_text_mobile_paddle_bundle as mobile_bundle
from customs_ai_agent.evaluation import (
    raster_text_public_voyage_replay_paddle_bundle as voyage_bundle,
)
from customs_ai_agent.evaluation import (
    source_attested_public_raster_text_recognizer_voyage_replay as replay,
)
from customs_ai_agent.evaluation import (
    source_attested_public_raster_text_voyage_replay_corpus as replay_corpus,
)
from customs_ai_agent.evaluation.raster_text_public_voyage_replay_paddle_bundle import (
    RasterTextPublicVoyageReplayPaddleBundleError,
    load_verified_raster_text_public_voyage_replay_paddle_bundle_v1,
    materialize_verified_raster_text_public_voyage_replay_paddle_bundle_v1,
    prepare_raster_text_public_voyage_replay_runtime_profile_v20,
)
from customs_ai_agent.evaluation.source_attested_raster_text_recognizer_training import (
    canonical_json_bytes,
    canonical_json_file_bytes,
)
from customs_ai_agent.service.raster_text_direct_paddle_sidecar import (
    DirectPaddleSidecarError,
    _profile_from_configuration,
)
from customs_ai_agent.service.raster_text_preview_runtime import RasterTextOfflineProfileV20


_OUTPUT_MODEL_NAME = "source_attested_en_PP-OCRv5_mobile_rec_voyage_replay_head_v3"
_COLLECTION = "raster-text-public-voyage-replay-paddle-bundles-v1"
_INVENTORY = "inventories/direct-paddle-public-voyage-replay-models.json"
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


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def _preview_root(tmp_path: Path) -> Path:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    return root


def _base_characters() -> tuple[str, ...]:
    return tuple(chr(0x1000 + index) for index in range(436))


def _make_mobile_bundle(
    root: Path,
) -> mobile_bundle.VerifiedRasterTextDirectMobilePaddleBundleV1:
    bundle_root = root / "test-mobile-source"
    inventory_relative = "inventories/direct-paddle-mobile-models.json"
    _write(bundle_root / inventory_relative, _canonical({"artifacts": [], "schema_version": "1.0"}))
    records: list[mobile_bundle.DirectPaddleBundleFileV1] = []
    for relative in (
        inventory_relative,
        *(
            f"models/{directory}/{name}"
            for directory in ("text-det", "text-rec")
            for name in _MODEL_FILENAMES
        ),
    ):
        path = bundle_root.joinpath(*relative.split("/"))
        if not path.exists():
            _write(path, f"mobile:{relative}".encode("utf-8"))
        raw = path.read_bytes()
        records.append(
            mobile_bundle.DirectPaddleBundleFileV1(
                path=relative,
                sha256=_sha(raw),
                byte_size=len(raw),
            )
        )
    manifest = mobile_bundle.RasterTextMobilePaddleBundleManifestV1(
        schema_version="1.0",
        bundle_id="raster_text_direct_paddle_mobile_model_bundle_v1",
        source_model_inventory_relative_path=f"model-inventories/{'1' * 64}.json",
        source_model_inventory_sha256="1" * 64,
        source_model_snapshot_sha256="2" * 64,
        text_detection_model_name="PP-OCRv5_mobile_det",
        text_recognition_model_name="en_PP-OCRv5_mobile_rec",
        local_model_root_relative_path="models",
        model_inventory_relative_path=inventory_relative,
        model_root_tree_sha256="3" * 64,
        files=sorted(records, key=lambda item: item.path),
    )
    raw = _canonical(manifest.model_dump(mode="json", warnings=False))
    _write(bundle_root / "manifest.json", raw)
    digest = _sha(raw)
    return mobile_bundle.VerifiedRasterTextDirectMobilePaddleBundleV1(
        document_preview_root=root,
        bundle_root=bundle_root,
        bundle_root_relative_path="test-mobile-source",
        manifest=manifest,
        manifest_sha256=digest,
        file_sha256_by_relative_path={item.path: item.sha256 for item in manifest.files},
    )


def _corpus_manifest(
    split: Literal["train", "development"],
) -> replay_corpus.VoyageReplayCorpusManifestV1:
    train = split == "train"
    selection_sha = "4" * 64
    recipe_sha256s = replay_corpus.voyage_replay_recipe_sha256s_v1()
    recipe_ids = tuple(recipe_sha256s)
    lane_counts = (
        {recipe_id: 63 if index < 4 else 62 for index, recipe_id in enumerate(recipe_ids)}
        if train
        else {recipe_id: 25 for recipe_id in recipe_ids}
    )
    general_counts = (
        {"consignee": 84, "invoice_number": 83, "total_packages": 83}
        if train
        else {"consignee": 67, "invoice_number": 67, "total_packages": 66}
    )
    all_counts = {
        "bill_of_lading:voyage": 500 if train else 200,
        "bill_of_lading:consignee": general_counts["consignee"],
        "commercial_invoice:invoice_number": general_counts["invoice_number"],
        "packing_list:total_packages": general_counts["total_packages"],
    }
    document_counts = {
        "bill_of_lading": general_counts["consignee"],
        "commercial_invoice": general_counts["invoice_number"],
        "packing_list": general_counts["total_packages"],
    }
    profile_counts = (
        {"png-216": 84, "jpeg-216-q75": 83, "jpeg-150-q75": 83}
        if train
        else {"png-216": 67, "jpeg-216-q75": 67, "jpeg-150-q75": 66}
    )
    return replay_corpus.VoyageReplayCorpusManifestV1(
        split=split,
        purpose="optimizer" if train else "validation-only",
        source_receipt_sha256="5" * 64,
        raster_receipt_sha256="6" * 64,
        selection_receipt_relative_path=(
            f"synthetic-raster-text-v1/fresh-scope-selections/{selection_sha}.json"
        ),
        selection_receipt_sha256=selection_sha,
        parent_selection_receipt_sha256="7" * 64,
        selected_family_count=500 if train else 100,
        selected_family_ids_sha256=("8" if train else "9") * 64,
        excluded_family_ids_sha256="a" * 64,
        independent_audit_recipe_sha256="b" * 64,
        independent_audit_family_ids_sha256="c" * 64,
        independent_audit_source_sha256s_sha256="d" * 64,
        voyage_crop_count=500 if train else 200,
        general_crop_count=250 if train else 200,
        total_crop_count=750 if train else 400,
        voyage_recipe_set_sha256=_sha(canonical_json_bytes(recipe_sha256s)),
        voyage_recipe_sha256s=recipe_sha256s,
        voyage_recipe_lane_counts=lane_counts,
        voyage_sample_ids_sha256="e" * 64,
        voyage_label_sha256s_sha256="f" * 64,
        voyage_label_pattern_sha256=_sha(b"^[0-9]{4}[EWNS]$"),
        general_sample_ids_sha256="0" * 64,
        general_field_ids_sha256="1" * 64,
        general_field_crop_counts=general_counts,
        all_field_key_ids_sha256=_sha(canonical_json_bytes(sorted(all_counts))),
        all_field_key_crop_counts=all_counts,
        general_document_type_crop_counts=document_counts,
        general_profile_crop_counts=profile_counts,
    )


def _metric(crop_count: int) -> replay.ReplayMetricV1:
    return replay.ReplayMetricV1(
        crop_count=crop_count,
        reference_character_count=crop_count * 5,
        character_error_count=0,
        exact_match_count=crop_count,
        character_error_rate=0.0,
        exact_match=1.0,
    )


def _development_evidence() -> replay.VoyageReplayDevelopmentEvidenceV1:
    recipe_ids = tuple(replay_corpus.voyage_replay_recipe_sha256s_v1())
    return replay.VoyageReplayDevelopmentEvidenceV1(
        voyage_overall=_metric(200),
        voyage_by_lane={recipe_id: _metric(25) for recipe_id in recipe_ids},
        general_overall=_metric(200),
        general_by_profile={
            "png-216": _metric(67),
            "jpeg-216-q75": _metric(67),
            "jpeg-150-q75": _metric(66),
        },
        general_by_field_key={"invoice_number": _metric(100), "total_packages": _metric(100)},
    )


def _make_recognizer_release(
    root: Path,
    *,
    mobile: mobile_bundle.VerifiedRasterTextDirectMobilePaddleBundleV1,
    characters: tuple[str, ...],
) -> replay.VerifiedVoyageReplayReleaseV1:
    collection = root / "raster-text-public-voyage-replay-recognizers-v1"
    stage = collection / ".staging-test"
    (stage / "text-rec").mkdir(parents=True)
    configuration = {
        "Global": {"model_name": _OUTPUT_MODEL_NAME},
        "PostProcess": {
            "character_dict": list(characters),
            "name": "CTCLabelDecode",
        },
        "PreProcess": {},
    }
    _write(stage / "text-rec" / "config.json", canonical_json_file_bytes(configuration))
    _write(stage / "text-rec" / "inference.json", b"voyage-replay-static-graph")
    _write(stage / "text-rec" / "inference.pdiparams", b"voyage-replay-head-parameters")
    _write(
        stage / "text-rec" / "inference.yml",
        f"Global:\n  model_name: {_OUTPUT_MODEL_NAME}\n".encode("utf-8"),
    )
    train = _corpus_manifest("train")
    development = _corpus_manifest("development")
    artifacts = {relative: replay.sha256_file(stage / relative) for relative in replay._MODEL_FILES}
    evidence = _development_evidence()
    receipt_payload: dict[str, object] = {
        "artifact_kind": "source_attested_public_recognizer_voyage_replay_receipt",
        "trainer_revision": "public-paddle-jit-ctc-leakage-safe-voyage-replay-head-v3",
        "model_revision": "en-ppocrv5-mobile-rec-source-attested-voyage-replay-head-v3",
        "output_model_name": _OUTPUT_MODEL_NAME,
        "stage": "voyage-replay-v3",
        "tuning_scope": "ctc_head_only",
        "training_intent_sha256": "2" * 64,
        "parent_release_sha256": replay._PARENT_RELEASE_SHA256,
        "parent_receipt_sha256": replay._PARENT_RECEIPT_SHA256,
        "parent_model_artifact_sha256s": {relative: "3" * 64 for relative in replay._MODEL_FILES},
        "train_manifest_sha256": _sha(
            canonical_json_bytes(train.model_dump(mode="json", warnings=False))
        ),
        "development_manifest_sha256": _sha(
            canonical_json_bytes(development.model_dump(mode="json", warnings=False))
        ),
        "train_corpus_manifest": train.model_dump(mode="json", warnings=False),
        "development_corpus_manifest": development.model_dump(mode="json", warnings=False),
        "direct_runtime_bundle_manifest_sha256": "4" * 64,
        "direct_runtime_package_inventory_sha256": "5" * 64,
        "public_mobile_model_bundle_manifest_sha256": mobile.manifest_sha256,
        "base_dictionary_sha256": _sha(canonical_json_bytes(list(characters))),
        "output_class_count": 438,
        "parent_python_revision": f"{sys.version_info.major}.{sys.version_info.minor}",
        "training_module_sha256": "6" * 64,
        "corpus_module_sha256": "7" * 64,
        "source_crop_base_module_sha256": "8" * 64,
        "seed_sha256": "9" * 64,
        "batch_size": 30,
        "voyage_per_batch": 20,
        "general_per_batch": 10,
        "batches_per_epoch": 25,
        "maximum_epochs": 2,
        "completed_epochs": 1,
        "published_epoch": 1,
        "earliest_passing_checkpoint": True,
        "learning_rate": 5.0e-5,
        "weight_decay": 1.0e-5,
        "gradient_clip_global_norm": 1.0,
        "optimizer": "AdamW",
        "prepared_crop_cache_revision": "exact-width-normalized-lru-v1",
        "prepared_crop_cache_limit_bytes": 536_870_912,
        "trainable_tensor_count": 2,
        "trainable_parameter_count": 52_998,
        "trainable_parameter_name_list_sha256": "a" * 64,
        "parent_baseline_development": evidence.model_dump(mode="json", warnings=False),
        "published_checkpoint_development": evidence.model_dump(mode="json", warnings=False),
        "framework_license": "Apache-2.0",
        "initialization": "verified_v2_release_warm_start",
        "static_roundtrip_verified": True,
        "model_artifact_sha256s": artifacts,
    }
    return replay.publish_voyage_replay_release_v1(
        root,
        staged_release=stage,
        receipt_payload=receipt_payload,
    )


@dataclass(frozen=True)
class _Sources:
    root: Path
    mobile: mobile_bundle.VerifiedRasterTextDirectMobilePaddleBundleV1
    recognizer: replay.VerifiedVoyageReplayReleaseV1


@pytest.fixture
def sources(
    tmp_path_factory: pytest.TempPathFactory,
    monkeypatch: pytest.MonkeyPatch,
) -> _Sources:
    root = _preview_root(tmp_path_factory.mktemp("vr"))
    characters = _base_characters()
    mobile = _make_mobile_bundle(root)

    def load_mobile(
        *, document_preview_root: Path, manifest_sha256: str
    ) -> mobile_bundle.VerifiedRasterTextDirectMobilePaddleBundleV1:
        assert document_preview_root.resolve() == root.resolve()
        if manifest_sha256 != mobile.manifest_sha256:
            raise mobile_bundle.RasterTextMobilePaddleBundleError("test_mobile_drift")
        return mobile

    monkeypatch.setattr(
        voyage_bundle,
        "load_verified_raster_text_direct_mobile_paddle_bundle_v1",
        load_mobile,
    )
    recognizer = _make_recognizer_release(root, mobile=mobile, characters=characters)
    return _Sources(root=root, mobile=mobile, recognizer=recognizer)


def _materialize(
    sources: _Sources,
) -> voyage_bundle.VerifiedRasterTextPublicVoyageReplayPaddleBundleV1:
    return materialize_verified_raster_text_public_voyage_replay_paddle_bundle_v1(
        mobile_bundle=sources.mobile,
        recognizer_release=sources.recognizer,
    )


def test_materialize_load_and_idempotency_bind_the_complete_v3_provenance(
    sources: _Sources,
) -> None:
    first = _materialize(sources)
    second = _materialize(sources)
    loaded = load_verified_raster_text_public_voyage_replay_paddle_bundle_v1(
        sources.root, first.manifest_sha256
    )

    assert first.manifest_sha256 == second.manifest_sha256 == loaded.manifest_sha256
    assert first.bundle_root.samefile(second.bundle_root)
    assert loaded.bundle_root_relative_path == (f"{_COLLECTION}/releases/{loaded.manifest_sha256}")
    manifest = loaded.manifest
    receipt = sources.recognizer.receipt
    assert manifest.source_mobile_bundle_manifest_sha256 == sources.mobile.manifest_sha256
    assert manifest.source_recognizer_release_sha256 == sources.recognizer.release_sha256
    assert manifest.source_recognizer_receipt_sha256 == receipt.receipt_sha256
    assert manifest.source_recognizer_training_intent_sha256 == receipt.training_intent_sha256
    assert manifest.source_recognizer_parent_release_sha256 == receipt.parent_release_sha256
    assert manifest.source_recognizer_parent_receipt_sha256 == receipt.parent_receipt_sha256
    assert (
        manifest.source_recognizer_parent_model_artifact_sha256s
        == receipt.parent_model_artifact_sha256s
    )
    assert manifest.source_recognizer_train_corpus_manifest == receipt.train_corpus_manifest
    assert (
        manifest.source_recognizer_development_corpus_manifest
        == receipt.development_corpus_manifest
    )
    assert (
        manifest.source_recognizer_voyage_recipe_set_sha256
        == receipt.train_corpus_manifest.voyage_recipe_set_sha256
    )
    assert manifest.source_recognizer_trainer_revision == (
        "public-paddle-jit-ctc-leakage-safe-voyage-replay-head-v3"
    )
    assert manifest.source_recognizer_model_revision == (
        "en-ppocrv5-mobile-rec-source-attested-voyage-replay-head-v3"
    )
    assert manifest.source_recognizer_output_class_count == 438
    assert manifest.text_detection_model_name == "PP-OCRv5_mobile_det"
    assert manifest.text_recognition_model_name == _OUTPUT_MODEL_NAME

    actual = {
        path.relative_to(loaded.bundle_root).as_posix()
        for path in loaded.bundle_root.rglob("*")
        if path.is_file()
    }
    assert actual == {
        "manifest.json",
        _INVENTORY,
        *(f"models/text-det/{name}" for name in _MODEL_FILENAMES),
        *(f"models/text-rec/{name}" for name in _MODEL_FILENAMES),
        "provenance/mobile-detector-bundle-manifest.json",
        "provenance/mobile-detector-model-inventory.json",
        "provenance/public-voyage-replay-recognizer-release-manifest.json",
        "provenance/public-voyage-replay-recognizer-training-receipt.json",
    }
    assert (
        loaded.bundle_root.joinpath(
            "provenance/public-voyage-replay-recognizer-training-receipt.json"
        ).read_bytes()
        == sources.recognizer.file_paths["training-receipt.json"].read_bytes()
    )
    assert not list((sources.root / _COLLECTION).glob(".staging-*"))


def test_loader_rejects_byte_drift_and_unexpected_tree_entries(sources: _Sources) -> None:
    bundle = _materialize(sources)
    (bundle.bundle_root / "models" / "text-rec" / "inference.pdiparams").write_bytes(b"drift")
    with pytest.raises(RasterTextPublicVoyageReplayPaddleBundleError) as raised:
        load_verified_raster_text_public_voyage_replay_paddle_bundle_v1(
            sources.root, bundle.manifest_sha256
        )
    assert raised.value.code == "public_voyage_replay_paddle_bundle_bundle_drift"

    # Recreate the pristine release under a new fixture-owned source tree.
    # A separate assertion below covers exact-tree rejection without mutating
    # any receipt or model hash.


def test_loader_rejects_an_extra_file_and_manifest_parent_drift(sources: _Sources) -> None:
    bundle = _materialize(sources)
    (bundle.bundle_root / "unexpected.bin").write_bytes(b"extra")
    with pytest.raises(RasterTextPublicVoyageReplayPaddleBundleError) as raised:
        load_verified_raster_text_public_voyage_replay_paddle_bundle_v1(
            sources.root, bundle.manifest_sha256
        )
    assert raised.value.code == "public_voyage_replay_paddle_bundle_manifest_invalid"

    payload = bundle.manifest.model_dump(mode="json", warnings=False)
    payload["source_recognizer_parent_receipt_sha256"] = "0" * 64
    with pytest.raises(ValueError):
        voyage_bundle.RasterTextPublicVoyageReplayPaddleBundleManifestV1.model_validate(payload)


def test_materializer_reopens_the_owner_release_and_rejects_source_drift(
    sources: _Sources,
) -> None:
    sources.recognizer.file_paths["text-rec/inference.json"].write_bytes(b"source-drift")
    with pytest.raises(RasterTextPublicVoyageReplayPaddleBundleError) as raised:
        _materialize(sources)
    assert raised.value.code == "public_voyage_replay_paddle_bundle_source_drift"


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
        "sidecar/module-root/customs_ai_agent/service/raster_text_direct_paddle_sidecar.py": (
            "3" * 64
        ),
        "inventories/direct-paddle-packages.json": "4" * 64,
    }
    manifest = SimpleNamespace(
        extractor_revision="direct-paddle-test-v2",
        raster_dpi=216,
        parent_renderer_package_inventory_relative_path="inventories/parent-renderer.json",
        parent_renderer_required_packages=("pypdfium2", "Pillow"),
        sidecar_interpreter_relative_path="sidecar/interpreter/python.exe",
        sidecar_interpreter_root_relative_path="sidecar/interpreter",
        sidecar_interpreter_root_tree_sha256="5" * 64,
        sidecar_module_relative_path=(
            "sidecar/module-root/customs_ai_agent/service/raster_text_direct_paddle_sidecar.py"
        ),
        sidecar_module_root_relative_path="sidecar/module-root",
        sidecar_module_root_tree_sha256="6" * 64,
        sidecar_package_inventory_relative_path="inventories/direct-paddle-packages.json",
        sidecar_required_packages=(
            "numpy",
            "opencv-contrib-python",
            "paddlepaddle",
            "pyclipper",
        ),
    )
    return _FakeSidecar(root, bundle_root, manifest_sha, manifest, paths)


def test_v20_profile_is_exact_idempotent_and_sidecar_parseable(
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
        voyage_bundle,
        "load_verified_raster_text_direct_paddle_bundle_v1",
        load_sidecar,
    )
    prepared = prepare_raster_text_public_voyage_replay_runtime_profile_v20(
        sidecar_bundle=sidecar,  # type: ignore[arg-type]
        model_bundle=bundle,
        pdfium_version="4.30.0",
        minimum_page_confidence=0.73,
        cpu_threads=6,
    )
    again = prepare_raster_text_public_voyage_replay_runtime_profile_v20(
        sidecar_bundle=sidecar,  # type: ignore[arg-type]
        model_bundle=bundle,
        pdfium_version="4.30.0",
        minimum_page_confidence=0.73,
        cpu_threads=6,
    )
    profile_path = sources.root.joinpath(*prepared.runtime_configuration_relative_path.split("/"))
    raw = profile_path.read_bytes()
    value = json.loads(raw)
    profile = RasterTextOfflineProfileV20.model_validate(value)
    direct = _profile_from_configuration(raw)

    assert prepared == again
    assert _sha(raw) == prepared.runtime_configuration_sha256
    assert profile.schema_version == "20.0"
    assert profile.engine_kind == (
        "direct_paddle_mobile_det_public_voyage_replay_rec_harmonic_recall_"
        "native960_lowbox_strip2_semantic_repair_v2_onednn"
    )
    assert profile.text_detection_model_name == "PP-OCRv5_mobile_det"
    assert profile.text_recognition_model_name == _OUTPUT_MODEL_NAME
    assert profile.recognition_crop_policy == "horizontal-height-0p32-max8-v1"
    assert profile.recognition_resize_interpolation == "linear-v1"
    assert profile.recognition_batch_size == 8
    assert profile.enable_mkldnn is True
    assert profile.onednn_model_policy == "recognizer-only-v1"
    assert profile.sidecar_timeout_seconds == 30
    assert profile.source_recognizer_parent_release_sha256 == (
        sources.recognizer.receipt.parent_release_sha256
    )
    assert profile.source_recognizer_parent_receipt_sha256 == (
        sources.recognizer.receipt.parent_receipt_sha256
    )
    assert profile.source_recognizer_train_corpus_manifest == (
        sources.recognizer.receipt.train_corpus_manifest
    )
    assert profile.source_recognizer_development_corpus_manifest == (
        sources.recognizer.receipt.development_corpus_manifest
    )
    assert direct.model_spec == "mobile_det_public_voyage_replay_rec"
    assert direct.detector_pass_mode == "strip2"
    assert direct.semantic_repair_mode == "supported_docs_v2"
    assert direct.recognition_crop_policy == "horizontal-height-0p32-max8-v1"
    assert direct.recognition_batch_size == 8
    assert direct.voyage_retry_mode == "none"
    assert "voyage_retry_mode" not in value

    bad = dict(value)
    bad["engine_kind"] = (
        "direct_paddle_mobile_det_public_finetuned_rec_harmonic_recall_native960_"
        "lowbox_strip2_semantic_repair_v2_voyage_retry_onednn"
    )
    with pytest.raises(ValueError):
        RasterTextOfflineProfileV20.model_validate(bad)
    with pytest.raises(DirectPaddleSidecarError):
        _profile_from_configuration(_canonical(bad))

    with pytest.raises(RasterTextPublicVoyageReplayPaddleBundleError) as raised:
        prepare_raster_text_public_voyage_replay_runtime_profile_v20(
            sidecar_bundle=sidecar,  # type: ignore[arg-type]
            model_bundle=bundle,
            pdfium_version="4.30.0",
            minimum_page_confidence=0.73,
            cpu_threads=8,
        )
    assert raised.value.code == "public_voyage_replay_paddle_bundle_runtime_profile_invalid"
