"""Focused tests for the trained-recognizer combined Paddle bundle."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest

import customs_ai_agent.evaluation.raster_text_mobile_paddle_bundle as mobile_bundle
import customs_ai_agent.evaluation.raster_text_trained_paddle_bundle as trained_bundle
import customs_ai_agent.service.raster_text_preview_runtime as raster_text_runtime
from customs_ai_agent.evaluation.raster_text_direct_paddle_bundle import (
    VerifiedRasterTextDirectPaddleBundleV1,
)
from customs_ai_agent.evaluation.source_attested_raster_text_recognizer_training import (
    VerifiedRecognizerRelease,
    canonical_json_bytes,
    canonical_json_file_bytes,
    create_recognizer_training_stage,
    publish_source_attested_raster_text_recognizer_release,
)


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _write(path: Path, raw: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)


def _make_mobile_detector_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> mobile_bundle.VerifiedRasterTextDirectMobilePaddleBundleV1:
    """Create a tiny, locally attested source fixture without running a model."""

    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    artifacts: list[dict[str, object]] = []
    for index, (source_relative, _target_relative) in enumerate(
        mobile_bundle._MOBILE_MODEL_ARTIFACTS, start=1
    ):
        if source_relative.endswith("det/config.json"):
            raw = _canonical({"Global": {"model_name": "PP-OCRv5_mobile_det"}})
        elif source_relative.endswith("rec/config.json"):
            raw = _canonical({"Global": {"model_name": "en_PP-OCRv5_mobile_rec"}})
        else:
            raw = f"mobile-model-{index}".encode("ascii")
        _write(root.joinpath(*source_relative.split("/")), raw)
        artifacts.append(
            {
                "artifact_id": f"artifact-{index}",
                "byte_size": len(raw),
                "license_id": "Apache-2.0",
                "relative_path": source_relative,
                "sha256": hashlib.sha256(raw).hexdigest(),
                "source_uri": f"https://example.invalid/{index}",
            }
        )
    inventory = {
        "schema_version": "1.0",
        "engine_kind": "paddleocr",
        "runtime_api": "paddleocr3-paddlex-3.2",
        "packages": [],
        "artifacts": artifacts,
    }
    inventory_raw = _canonical(inventory) + b"\n"
    inventory_sha256 = hashlib.sha256(inventory_raw).hexdigest()
    inventory_relative = f"model-inventories/{inventory_sha256}.json"
    _write(root.joinpath(*inventory_relative.split("/")), inventory_raw)
    monkeypatch.setattr(mobile_bundle, "_MOBILE_SOURCE_INVENTORY_RELATIVE_PATH", inventory_relative)
    monkeypatch.setattr(mobile_bundle, "_MOBILE_SOURCE_INVENTORY_SHA256", inventory_sha256)
    return mobile_bundle.materialize_verified_raster_text_direct_mobile_paddle_bundle_v1(
        document_preview_root=root
    )


def _make_trained_recognizer_release(
    root: Path,
    *,
    model_name: str = "source_attested_tiny_crnn_bigru_ctc_v1",
) -> VerifiedRecognizerRelease:
    characters = [" ", "A", "B", "1", "2"]
    stage = create_recognizer_training_stage(root)
    text_rec = stage / "text-rec"
    text_rec.mkdir()
    config = {
        "Global": {"model_name": model_name},
        "PostProcess": {"character_dict": characters, "name": "CTCLabelDecode"},
        "PreProcess": {
            "input_shape": [3, 48, -1],
            "normalization": "rgb-minus-half-v1",
        },
    }
    _write(text_rec / "config.json", canonical_json_file_bytes(config))
    _write(text_rec / "inference.json", b'{"graph":"static"}\n')
    _write(text_rec / "inference.pdiparams", b"trained-parameter-bytes")
    _write(
        text_rec / "inference.yml",
        (
            "Global:\n"
            "  model_name: source_attested_tiny_crnn_bigru_ctc_v1\n"
            "  model_type: rec\n"
            "PostProcess:\n"
            "  name: CTCLabelDecode\n"
        ).encode("utf-8"),
    )
    receipt = {
        "schema_version": "1.0",
        "artifact_kind": "source_attested_raster_text_recognizer_training_receipt",
        "trainer_revision": "source-attested-tiny-crnn-bigru-ctc-v1",
        "stage": "full",
        "training_intent_sha256": "1" * 64,
        "train_manifest_sha256": "2" * 64,
        "development_manifest_sha256": "3" * 64,
        "character_dictionary_sha256": hashlib.sha256(canonical_json_bytes(characters)).hexdigest(),
        "package_inventory_sha256": "5" * 64,
        "training_runtime_bundle_manifest_sha256": "6" * 64,
        "parent_python_revision": "3.11",
        "training_module_sha256": "7" * 64,
        "seed_sha256": "8" * 64,
        "batch_size": 8,
        "maximum_epochs": 10,
        "include_merged_lines": True,
        "initialization": "deterministic_random_no_pretrained_checkpoint",
        "framework_license": "Apache-2.0",
        "model_revision": "tiny-ds-cnn-bigru-ctc-v1",
        "completed_epochs": 2,
        "best_epoch": 1,
        "early_stopped": False,
        "development_cer": 0.01,
        "development_exact_match": 0.99,
    }
    return publish_source_attested_raster_text_recognizer_release(
        root, staged_release=stage, receipt_payload=receipt
    )


def _materials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[
    Path,
    mobile_bundle.VerifiedRasterTextDirectMobilePaddleBundleV1,
    VerifiedRecognizerRelease,
]:
    detector = _make_mobile_detector_bundle(tmp_path, monkeypatch)
    recognizer = _make_trained_recognizer_release(detector.document_preview_root)
    return detector.document_preview_root, detector, recognizer


def _materialize(
    root: Path,
    detector: mobile_bundle.VerifiedRasterTextDirectMobilePaddleBundleV1,
    recognizer: VerifiedRecognizerRelease,
) -> trained_bundle.VerifiedRasterTextTrainedPaddleBundleV1:
    return trained_bundle.materialize_verified_raster_text_trained_paddle_bundle_v1(
        document_preview_root=root,
        detector_bundle_manifest_sha256=detector.manifest_sha256,
        recognizer_release_sha256=recognizer.release_sha256,
    )


def _fake_sidecar_bundle(
    root: Path, monkeypatch: pytest.MonkeyPatch
) -> VerifiedRasterTextDirectPaddleBundleV1:
    manifest_sha256 = "9" * 64
    relative_root = f"raster-text-direct-bundles-v1/releases/{manifest_sha256}"
    hashes = {
        "inventories/parent-renderer.json": "a" * 64,
        "sidecar/interpreter/python.exe": "b" * 64,
        "sidecar/module-root/customs_ai_agent/service/raster_text_direct_paddle_sidecar.py": (
            "c" * 64
        ),
        "inventories/direct-paddle-packages.json": "d" * 64,
    }
    manifest = SimpleNamespace(
        extractor_revision="direct-paddle-test-v2",
        raster_dpi=216,
        parent_renderer_package_inventory_relative_path="inventories/parent-renderer.json",
        parent_renderer_required_packages=["pypdfium2", "Pillow"],
        sidecar_interpreter_relative_path="sidecar/interpreter/python.exe",
        sidecar_interpreter_root_relative_path="sidecar/interpreter",
        sidecar_interpreter_root_tree_sha256="e" * 64,
        sidecar_module_relative_path=(
            "sidecar/module-root/customs_ai_agent/service/raster_text_direct_paddle_sidecar.py"
        ),
        sidecar_module_root_relative_path="sidecar/module-root",
        sidecar_module_root_tree_sha256="f" * 64,
        sidecar_package_inventory_relative_path="inventories/direct-paddle-packages.json",
        sidecar_required_packages=[
            "numpy",
            "opencv-contrib-python",
            "paddlepaddle",
            "pyclipper",
        ],
    )
    bundle = cast(
        VerifiedRasterTextDirectPaddleBundleV1,
        SimpleNamespace(
            document_preview_root=root,
            bundle_root_relative_path=relative_root,
            manifest_sha256=manifest_sha256,
            manifest=manifest,
            bundle_relative_path=lambda value: f"{relative_root}/{value}",
            file_sha256=lambda value: hashes[value],
        ),
    )
    monkeypatch.setattr(
        trained_bundle,
        "load_verified_raster_text_direct_paddle_bundle_v1",
        lambda **_kwargs: bundle,
    )
    return bundle


def test_materialize_load_and_idempotent_reuse_bind_exact_source_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, detector, recognizer = _materials(tmp_path, monkeypatch)

    first = _materialize(root, detector, recognizer)
    second = _materialize(root, detector, recognizer)
    loaded = trained_bundle.load_verified_raster_text_trained_paddle_bundle_v1(
        document_preview_root=root, manifest_sha256=first.manifest_sha256
    )

    assert first.manifest_sha256 == second.manifest_sha256 == loaded.manifest_sha256
    assert first.bundle_root == second.bundle_root == loaded.bundle_root
    assert first.bundle_root_relative_path.endswith(first.manifest_sha256)
    assert first.manifest.source_detector_bundle_manifest_sha256 == detector.manifest_sha256
    assert first.manifest.source_recognizer_release_sha256 == recognizer.release_sha256
    assert first.manifest.source_recognizer_receipt_sha256 == recognizer.receipt.receipt_sha256
    expected_model_files = {
        f"{directory}/{filename}"
        for directory in ("text-det", "text-rec")
        for filename in ("config.json", "inference.json", "inference.pdiparams", "inference.yml")
    }
    actual_model_files = {
        path.relative_to(first.bundle_root / "models").as_posix()
        for path in (first.bundle_root / "models").rglob("*")
        if path.is_file()
    }
    assert actual_model_files == expected_model_files
    inventory = json.loads(
        (first.bundle_root / "inventories" / "direct-paddle-trained-models.json").read_bytes()
    )
    assert [item["relative_path"] for item in inventory["artifacts"]] == sorted(
        expected_model_files
    )
    for filename in ("config.json", "inference.json", "inference.pdiparams", "inference.yml"):
        assert (first.bundle_root / "models" / "text-det" / filename).read_bytes() == (
            detector.bundle_root / "models" / "text-det" / filename
        ).read_bytes()
        assert (first.bundle_root / "models" / "text-rec" / filename).read_bytes() == (
            recognizer.file_paths[f"text-rec/{filename}"]
        ).read_bytes()
    assert (
        first.bundle_root / "provenance" / "mobile-detector-bundle-manifest.json"
    ).read_bytes() == (detector.bundle_root / "manifest.json").read_bytes()
    assert (
        first.bundle_root / "provenance" / "mobile-detector-model-inventory.json"
    ).read_bytes() == (
        detector.bundle_root / detector.manifest.model_inventory_relative_path
    ).read_bytes()
    assert (
        first.bundle_root / "provenance" / "trained-recognizer-release-manifest.json"
    ).read_bytes() == (recognizer.release_path / "manifest.json").read_bytes()
    assert (
        first.bundle_root / "provenance" / "trained-recognizer-training-receipt.json"
    ).read_bytes() == recognizer.file_paths["training-receipt.json"].read_bytes()
    assert hashlib.sha256((first.bundle_root / "manifest.json").read_bytes()).hexdigest() == (
        first.manifest_sha256
    )
    assert (
        first.file_sha256("models/text-det/config.json")
        == hashlib.sha256(
            (first.bundle_root / "models" / "text-det" / "config.json").read_bytes()
        ).hexdigest()
    )
    releases = first.bundle_root.parent
    release_directories = [path for path in releases.iterdir() if path.is_dir()]
    assert len(release_directories) == 1
    assert os.path.samefile(release_directories[0], first.bundle_root)
    assert not list(root.glob(".raster-text-trained-paddle-bundles-v1.staging-*"))


@pytest.mark.parametrize("mutation", ["extra", "missing", "drift"])
def test_loader_fails_closed_for_tree_or_byte_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    root, detector, recognizer = _materials(tmp_path, monkeypatch)
    bundle = _materialize(root, detector, recognizer)
    target = bundle.bundle_root / "models" / "text-det" / "inference.json"
    expected_code = "trained_paddle_bundle_tree_invalid"
    if mutation == "extra":
        _write(bundle.bundle_root / "models" / "unexpected.bin", b"unexpected")
    elif mutation == "missing":
        target.unlink()
    else:
        target.write_bytes(b"mutated-model")
        expected_code = "trained_paddle_bundle_drift"

    with pytest.raises(trained_bundle.RasterTextTrainedPaddleBundleError) as raised:
        trained_bundle.load_verified_raster_text_trained_paddle_bundle_v1(
            document_preview_root=root, manifest_sha256=bundle.manifest_sha256
        )

    assert raised.value.code == expected_code


def test_loader_rejects_a_symlink_in_the_combined_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, detector, recognizer = _materials(tmp_path, monkeypatch)
    bundle = _materialize(root, detector, recognizer)
    link = bundle.bundle_root / "models" / "text-det" / "inference.yml"
    link.unlink()
    try:
        os.symlink(detector.bundle_root / "models" / "text-det" / "inference.yml", link)
    except OSError:
        link.write_bytes(
            (detector.bundle_root / "models" / "text-det" / "inference.yml").read_bytes()
        )
        original_is_symlink = Path.is_symlink
        link_identity = trained_bundle._local_path_identity(link)

        def simulated_is_symlink(path: Path) -> bool:
            return trained_bundle._local_path_identity(
                path
            ) == link_identity or original_is_symlink(path)

        monkeypatch.setattr(Path, "is_symlink", simulated_is_symlink)

    with pytest.raises(trained_bundle.RasterTextTrainedPaddleBundleError) as raised:
        trained_bundle.load_verified_raster_text_trained_paddle_bundle_v1(
            document_preview_root=root, manifest_sha256=bundle.manifest_sha256
        )

    assert raised.value.code == "trained_paddle_bundle_tree_invalid"


@pytest.mark.parametrize("source_kind", ["detector", "recognizer"])
def test_loader_revalidates_each_upstream_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, source_kind: str
) -> None:
    root, detector, recognizer = _materials(tmp_path, monkeypatch)
    bundle = _materialize(root, detector, recognizer)
    if source_kind == "detector":
        (detector.bundle_root / "models" / "text-det" / "inference.pdiparams").write_bytes(
            b"upstream-drift"
        )
    else:
        recognizer.file_paths["text-rec/inference.pdiparams"].write_bytes(b"upstream-drift")

    with pytest.raises(trained_bundle.RasterTextTrainedPaddleBundleError) as raised:
        trained_bundle.load_verified_raster_text_trained_paddle_bundle_v1(
            document_preview_root=root, manifest_sha256=bundle.manifest_sha256
        )

    assert raised.value.code == "trained_paddle_bundle_source_invalid"


def test_materializer_rejects_a_recognizer_with_the_wrong_model_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    detector = _make_mobile_detector_bundle(tmp_path, monkeypatch)
    root = detector.document_preview_root
    recognizer = _make_trained_recognizer_release(root, model_name="different-recognizer")

    with pytest.raises(trained_bundle.RasterTextTrainedPaddleBundleError) as raised:
        _materialize(root, detector, recognizer)

    assert raised.value.code == "trained_paddle_bundle_model_invalid"
    assert not list(root.glob(".raster-text-trained-paddle-bundles-v1.staging-*"))


def test_materializer_preserves_an_existing_differing_release(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, detector, recognizer = _materials(tmp_path, monkeypatch)
    bundle = _materialize(root, detector, recognizer)
    target = bundle.bundle_root / "models" / "text-det" / "inference.json"
    target.write_bytes(b"existing-differing-output")

    with pytest.raises(trained_bundle.RasterTextTrainedPaddleBundleError) as raised:
        _materialize(root, detector, recognizer)

    assert raised.value.code == "trained_paddle_bundle_drift"
    assert bundle.bundle_root.is_dir()
    assert target.read_bytes() == b"existing-differing-output"
    assert not list(root.glob(".raster-text-trained-paddle-bundles-v1.staging-*"))


def test_materializer_removes_only_its_stage_after_a_copy_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, detector, recognizer = _materials(tmp_path, monkeypatch)

    def fail_copy(source: Path, target: Path, *, code: str) -> tuple[str, int]:
        del source, target
        raise trained_bundle.RasterTextTrainedPaddleBundleError(code)

    monkeypatch.setattr(trained_bundle, "_copy_fsynced", fail_copy)

    with pytest.raises(trained_bundle.RasterTextTrainedPaddleBundleError) as raised:
        _materialize(root, detector, recognizer)

    assert raised.value.code == "trained_paddle_bundle_stage_copy_failed"
    assert not list(root.glob(".raster-text-trained-paddle-bundles-v1.staging-*"))
    releases = root / "raster-text-trained-paddle-bundles-v1" / "releases"
    assert releases.is_dir()
    assert not list(releases.iterdir())


def test_materializer_rejects_a_root_outside_document_preview(tmp_path: Path) -> None:
    root = tmp_path / "preview"
    root.mkdir()

    with pytest.raises(trained_bundle.RasterTextTrainedPaddleBundleError) as raised:
        trained_bundle.materialize_verified_raster_text_trained_paddle_bundle_v1(
            document_preview_root=root,
            detector_bundle_manifest_sha256="a" * 64,
            recognizer_release_sha256="b" * 64,
        )

    assert raised.value.code == "trained_paddle_bundle_root_invalid"


def test_prepare_v12_binds_every_combined_source_and_is_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root, detector, recognizer = _materials(tmp_path, monkeypatch)
    combined = _materialize(root, detector, recognizer)
    sidecar = _fake_sidecar_bundle(root, monkeypatch)

    first = trained_bundle.prepare_raster_text_trained_runtime_profile_v12(
        sidecar,
        combined,
        pdfium_version="4.30.0",
        minimum_page_confidence=0.42,
    )
    second = trained_bundle.prepare_raster_text_trained_runtime_profile_v12(
        sidecar,
        combined,
        pdfium_version="4.30.0",
        minimum_page_confidence=0.42,
    )
    raw = root.joinpath(*first.runtime_configuration_relative_path.split("/")).read_bytes()
    profile = raster_text_runtime.parse_raster_text_offline_profile(raw)

    assert isinstance(profile, raster_text_runtime.RasterTextOfflineProfileV12)
    assert first == second
    assert hashlib.sha256(raw).hexdigest() == first.runtime_configuration_sha256
    assert profile.schema_version == "12.0"
    assert profile.sidecar_timeout_seconds == 30
    assert profile.cpu_threads == 6
    assert profile.text_detection_limit_side_len == 960
    assert profile.text_detection_threshold == 0.3
    assert profile.text_detection_box_threshold == 0.3
    assert profile.text_detection_unclip_ratio == 1.5
    assert profile.combined_model_bundle_manifest_sha256 == combined.manifest_sha256
    assert (
        profile.source_detector_bundle_manifest_sha256
        == combined.manifest.source_detector_bundle_manifest_sha256
    )
    assert (
        profile.source_detector_model_inventory_sha256
        == combined.manifest.source_detector_model_inventory_sha256
    )
    assert (
        profile.source_detector_model_snapshot_sha256
        == combined.manifest.source_detector_model_snapshot_sha256
    )
    assert (
        profile.source_recognizer_release_sha256
        == combined.manifest.source_recognizer_release_sha256
    )
    assert (
        profile.source_recognizer_training_intent_sha256
        == combined.manifest.source_recognizer_training_intent_sha256
    )
    assert (
        profile.source_recognizer_character_dictionary_sha256
        == combined.manifest.source_recognizer_character_dictionary_sha256
    )
    assert (
        profile.source_recognizer_package_inventory_sha256
        == combined.manifest.source_recognizer_package_inventory_sha256
    )
    assert profile.text_recognition_model_name == "source_attested_tiny_crnn_bigru_ctc_v1"
    assert "mobile_model_bundle_root_relative_path" not in profile.model_fields_set
    raster_text_runtime._verify_v12_trained_model_bundle(  # noqa: SLF001
        root=root, profile=profile
    )


@pytest.mark.parametrize(
    "binding_field",
    [
        "combined_model_bundle_manifest_sha256",
        "source_detector_bundle_manifest_sha256",
        "source_detector_model_inventory_sha256",
        "source_detector_model_snapshot_sha256",
        "source_recognizer_release_sha256",
        "source_recognizer_receipt_sha256",
        "source_recognizer_training_intent_sha256",
        "source_recognizer_character_dictionary_sha256",
        "source_recognizer_package_inventory_sha256",
    ],
)
def test_v12_fingerprint_covers_each_combined_source_binding(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    binding_field: str,
) -> None:
    root, detector, recognizer = _materials(tmp_path, monkeypatch)
    combined = _materialize(root, detector, recognizer)
    sidecar = _fake_sidecar_bundle(root, monkeypatch)
    prepared = trained_bundle.prepare_raster_text_trained_runtime_profile_v12(
        sidecar,
        combined,
        pdfium_version="4.30.0",
        minimum_page_confidence=0.42,
    )
    raw = root.joinpath(*prepared.runtime_configuration_relative_path.split("/")).read_bytes()
    values = json.loads(raw)
    values[binding_field] = "8" * 64

    with pytest.raises(ValueError):
        raster_text_runtime.RasterTextOfflineProfileV12.model_validate(values)
