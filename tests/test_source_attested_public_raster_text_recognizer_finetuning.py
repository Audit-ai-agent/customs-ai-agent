from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Iterator

import pytest

from customs_ai_agent.evaluation import source_attested_public_raster_text_corpus as corpus
from customs_ai_agent.evaluation import (
    source_attested_public_raster_text_recognizer_finetuning as fine_tune,
)
from customs_ai_agent.evaluation import (
    source_attested_public_raster_text_recognizer_finetuning_runtime as runtime,
)
from customs_ai_agent.evaluation.source_attested_raster_text_recognizer_training import (
    TrainingCropV1,
)
from tools import finetune_source_attested_public_raster_text_recognizer as cli


def _preview_root(tmp_path: Path) -> Path:
    root = tmp_path / "_local" / "data" / "document-preview"
    root.mkdir(parents=True)
    return root


def _crop(sample: str, *, split: str = "train", label: str = "A") -> TrainingCropV1:
    return TrainingCropV1(
        sample_id=sample * 64,
        asset_id=f"asset-{sample}",
        family_id=f"family-{sample}",
        split=split,
        asset_relative_path=f"assets/{sample}.png",
        gold_relative_path=f"gold/{sample}.json",
        source_pdf_sha256="1" * 64,
        raster_sha256="2" * 64,
        gold_sha256="3" * 64,
        span_ids=(f"span-{sample}",),
        bbox=(0.1, 0.1, 0.2, 0.1),
        label_sha256=fine_tune.sha256_bytes(label.encode("utf-8")),
        label_length=len(label),
        page_width=100,
        page_height=100,
    )


def _dictionary(*characters: str) -> fine_tune.BaseRecognizerDictionaryV1:
    values = tuple(characters)
    mapping = {value: index + 1 for index, value in enumerate(values)}
    mapping[" "] = len(values) + 1
    return fine_tune.BaseRecognizerDictionaryV1(
        characters=values,
        character_to_index=mapping,
        dictionary_sha256="4" * 64,
        config_sha256="5" * 64,
        output_class_count=len(values) + 2,
        implicit_space_class_index=len(values) + 1,
        config_value={},
    )


def test_semantic_label_revision_is_minimal_and_explicit() -> None:
    assert fine_tune.semantic_ocr_crop_label_v1(" A  B ") == " A  B "
    assert fine_tune.semantic_ocr_crop_label_v1("A\tB") == "A B"
    assert fine_tune.semantic_ocr_crop_label_v1("A■☐__B") == "AB"
    assert fine_tune.semantic_ocr_crop_label_v1("■☐") is None
    assert fine_tune.semantic_ocr_crop_label_v1("  ____  ") is None
    assert fine_tune.semantic_ocr_crop_label_v1("_") == "_"

    for invalid in (" ",):
        with pytest.raises(fine_tune.SourceAttestedPublicRecognizerFineTuneError) as raised:
            fine_tune.semantic_ocr_crop_label_v1(invalid)
        assert raised.value.code == "public_finetune_label_invalid"


def test_semantic_split_skips_only_decorative_and_keeps_profile_counts() -> None:
    crops = (
        _crop("a", label="A"),
        _crop("b", label="B"),
        _crop("c", label="A"),
        _crop("d", label="■"),
    )
    labels = {"a" * 64: "A", "b" * 64: "B", "c" * 64: "A", "d" * 64: "■"}
    profiles = {
        "a" * 64: "png-216",
        "b" * 64: "jpeg-216-q75",
        "c" * 64: "jpeg-150-q75",
        "d" * 64: "jpeg-150-q75",
    }

    manifest, usable, semantic = fine_tune._semantic_split(
        crops=crops,
        labels_by_sample_id=labels,
        split="train",
        source_manifest_sha256="6" * 64,
        source_receipt_sha256="7" * 64,
        raster_receipt_sha256="8" * 64,
        profile_by_sample_id=profiles,
        scan_duplicate_pair_count=1,
        scan_duplicate_pixel_list_sha256="9" * 64,
        dictionary=_dictionary("A", "B"),
    )

    assert len(usable) == 3
    assert len(semantic) == 3
    assert manifest.source_profile_crop_counts == {
        "png-216": 1,
        "jpeg-216-q75": 1,
        "jpeg-150-q75": 2,
    }
    assert manifest.usable_profile_crop_counts["jpeg-150-q75"] == 1
    serialized = manifest.model_dump_json()
    assert "BLACK" not in serialized
    assert "■" not in serialized


def test_semantic_split_fails_closed_on_character_outside_base_dictionary() -> None:
    crop_value = _crop("e", label="€")
    with pytest.raises(fine_tune.SourceAttestedPublicRecognizerFineTuneError) as raised:
        fine_tune._semantic_split(
            crops=(crop_value,),
            labels_by_sample_id={"e" * 64: "€"},
            split="train",
            source_manifest_sha256="6" * 64,
            source_receipt_sha256="7" * 64,
            raster_receipt_sha256="8" * 64,
            profile_by_sample_id={"e" * 64: "png-216"},
            scan_duplicate_pair_count=1,
            scan_duplicate_pixel_list_sha256="9" * 64,
            dictionary=_dictionary("A"),
        )
    assert raised.value.code == "public_finetune_label_dictionary_unsupported"


def test_checkpoint_requires_strict_improvement_over_epoch_zero() -> None:
    assert not fine_tune._strict_metric_improvement(
        candidate_cer=0.03,
        candidate_exact=0.80,
        baseline_cer=0.03,
        baseline_exact=0.80,
    )
    assert not fine_tune._strict_metric_improvement(
        candidate_cer=0.031,
        candidate_exact=0.90,
        baseline_cer=0.03,
        baseline_exact=0.80,
    )
    assert fine_tune._strict_metric_improvement(
        candidate_cer=0.029,
        candidate_exact=0.79,
        baseline_cer=0.03,
        baseline_exact=0.80,
    )
    assert fine_tune._strict_metric_improvement(
        candidate_cer=0.03,
        candidate_exact=0.81,
        baseline_cer=0.03,
        baseline_exact=0.80,
    )


def test_balanced_split_has_1k_profile_quota_and_excludes_verified_scan_duplicate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    material = tmp_path / "material.bin"
    material.write_bytes(b"material")
    gold_sha = hashlib.sha256(material.read_bytes()).hexdigest()
    candidates = tuple(
        corpus._Candidate("span", (f"span-{index}",), (0.1, 0.1, 0.2, 0.1), "A")
        for index in range(334)
    )
    by_profile: dict[str, dict[str, object]] = {}
    for profile in corpus._ALL_PROFILES:
        by_profile[profile] = {
            "asset_id": "asset-a",
            "asset_relative_path": f"assets/{profile}",
            "byte_size": 8,
            "family_id": "family-a",
            "gold_relative_path": f"gold/{profile}",
            "gold_sha256": gold_sha,
            "height": 100,
            "image_sha256": "2" * 64,
            "media_type": corpus._EXPECTED_MEDIA[profile],
            "profile_id": profile,
            "source_pdf_sha256": "1" * 64,
            "split": "train",
            "width": 100,
        }
    source = {
        "asset_id": "asset-a",
        "byte_size": 8,
        "family_id": "family-a",
        "source_pdf_sha256": "1" * 64,
        "source_relative_path": "source.pdf",
    }
    monkeypatch.setattr(corpus, "_safe", lambda *_args, **_kwargs: material)
    monkeypatch.setattr(corpus, "_verify_regular_material", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(corpus, "_read_json", lambda *_args, **_kwargs: ({}, b"{}"))
    monkeypatch.setattr(corpus, "_gold_candidates", lambda *_args, **_kwargs: candidates)
    monkeypatch.setattr(corpus, "_decode_rgb", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(corpus, "_verify_scan_duplicate", lambda **_kwargs: "a" * 64)

    crops, labels, profiles, pair_count, _pair_sha = corpus._compile_split(
        root=tmp_path,
        source_root=tmp_path,
        sources={"1" * 64: source},
        groups={"asset-a": by_profile},
        split="train",
        seed=1,
    )

    assert len(crops) == len(labels) == len(profiles) == 1_000
    assert pair_count == 1
    assert list(profiles.values()).count("png-216") == 333
    assert list(profiles.values()).count("jpeg-216-q75") == 333
    assert list(profiles.values()).count("jpeg-150-q75") == 334
    assert "scan-pdf-216-q75" not in profiles.values()


def test_v2_family_first_compiler_uses_two_gold_fields_per_family(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    families: dict[str, dict[str, dict[str, dict[str, object]]]] = {}
    for family_index in range(500):
        family_id = f"family-{family_index:04d}"
        family_assets: dict[str, dict[str, dict[str, object]]] = {}
        for document_type in corpus._DOCUMENT_TYPES:
            for variant in ("clean", "error"):
                asset_id = f"{variant}-{family_index:04d}-{document_type}"
                family_assets[asset_id] = {
                    profile: {
                        "asset_id": asset_id,
                        "asset_relative_path": f"assets/{asset_id}-{profile}.png",
                        "document_type": document_type,
                        "family_id": family_id,
                        "gold_relative_path": f"gold/{asset_id}-{profile}.json",
                        "gold_sha256": "3" * 64,
                        "height": 100,
                        "image_sha256": "2" * 64,
                        "profile_id": profile,
                        "source_pdf_sha256": "1" * 64,
                        "split": "train",
                        "width": 100,
                    }
                    for profile in corpus._ALL_PROFILES
                }
        families[family_id] = family_assets

    def verified(**kwargs: object) -> tuple[object, dict[str, object]]:
        by_profile = kwargs["by_profile"]
        selected_profiles = kwargs["selected_profiles"]
        assert isinstance(by_profile, dict)
        assert isinstance(selected_profiles, tuple)
        candidates = tuple(
            corpus._FieldCandidate(
                field_id=f"field_{index}",
                candidate=corpus._Candidate(
                    "span",
                    (f"span-{index}",),
                    (0.1, 0.1, 0.2, 0.1),
                    f"VALUE{index}",
                ),
            )
            for index in range(5)
        )
        return candidates, {profile: by_profile[profile] for profile in selected_profiles}

    monkeypatch.setattr(corpus, "_verified_candidates_for_asset_v2", verified)
    (
        crops,
        labels,
        profiles,
        fields,
        documents,
        profile_counts,
        field_counts,
        document_counts,
    ) = corpus._compile_family_first_split_v2(
        root=Path("."),
        source_root=Path("."),
        sources={},
        families=families,
        selected_family_ids=sorted(families),
        split="train",
        seed=1_937_429,
    )

    assert len(crops) == len(labels) == len(profiles) == 1_000
    assert len(fields) == len(documents) == 1_000
    assert {sum(crop.family_id == family for crop in crops) for family in families} == {2}
    assert document_counts == {
        "bill_of_lading": 334,
        "commercial_invoice": 334,
        "packing_list": 332,
    }
    assert sum(profile_counts.values()) == sum(field_counts.values()) == 1_000
    assert all(key.startswith(f"{documents[sample]}:") for sample, key in fields.items())


def test_v2_manifest_binds_selection_fields_and_balanced_critical_counts() -> None:
    selection_sha = "a" * 64
    field_counts = {
        "bill_of_lading:bl_no": 334,
        "commercial_invoice:inv_no": 334,
        "packing_list:inv_no": 332,
    }
    manifest = fine_tune.PublicFineTuneCropManifestV2(
        split="train",
        source_crop_manifest_sha256="b" * 64,
        source_receipt_sha256="c" * 64,
        raster_receipt_sha256="d" * 64,
        selection_receipt_relative_path=(
            f"synthetic-raster-text-v1/fresh-scope-selections/{selection_sha}.json"
        ),
        selection_receipt_sha256=selection_sha,
        parent_selection_receipt_sha256="e" * 64,
        selected_family_ids_sha256="f" * 64,
        excluded_family_ids_sha256="1" * 64,
        family_count=500,
        source_crop_count=1_000,
        usable_crop_count=1_000,
        semantic_crop_list_sha256="2" * 64,
        skipped_sample_list_sha256="3" * 64,
        base_dictionary_sha256="4" * 64,
        source_profile_crop_counts={
            "png-216": 334,
            "jpeg-216-q75": 337,
            "jpeg-150-q75": 329,
        },
        usable_profile_crop_counts={
            "png-216": 334,
            "jpeg-216-q75": 337,
            "jpeg-150-q75": 329,
        },
        critical_field_crop_count=1_000,
        critical_field_ids_sha256=fine_tune.sha256_bytes(
            fine_tune.canonical_json_bytes(sorted(field_counts))
        ),
        critical_field_crop_counts=field_counts,
        document_type_crop_counts={
            "bill_of_lading": 334,
            "commercial_invoice": 334,
            "packing_list": 332,
        },
        source_label_length_p50=13,
        source_label_length_p95=25,
    )

    assert manifest.holdout_overlap_count == 0
    assert manifest.crops_per_family == 2
    assert manifest.critical_field_crop_count == 1_000


def test_public_page_loader_accepts_png_and_jpeg_but_rejects_rotated_exif(
    tmp_path: Path,
) -> None:
    from PIL import Image

    root = _preview_root(tmp_path)
    assets = root / "assets"
    assets.mkdir()
    specifications = (
        ("a", "png-216", "PNG", "png"),
        ("b", "jpeg-216-q75", "JPEG", "jpg"),
        ("c", "jpeg-150-q75", "JPEG", "jpg"),
    )
    for token, profile, image_format, extension in specifications:
        path = assets / f"{token}.{extension}"
        Image.new("RGB", (100, 40), "white").save(path, format=image_format)
        crop_value = _crop(token).model_copy(
            update={
                "asset_relative_path": f"assets/{token}.{extension}",
                "page_width": 100,
                "page_height": 40,
                "raster_sha256": fine_tune.sha256_file(path),
            }
        )
        loaded = fine_tune._load_verified_training_page(root, crop=crop_value, profile_id=profile)
        assert loaded.size == (100, 40)

    rotated_path = assets / "rotated.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (100, 40), "white").save(rotated_path, format="JPEG", exif=exif)
    rotated = _crop("d").model_copy(
        update={
            "asset_relative_path": "assets/rotated.jpg",
            "page_width": 100,
            "page_height": 40,
            "raster_sha256": fine_tune.sha256_file(rotated_path),
        }
    )
    with pytest.raises(fine_tune.SourceAttestedPublicRecognizerFineTuneError) as raised:
        fine_tune._load_verified_training_page(root, crop=rotated, profile_id="jpeg-216-q75")
    assert raised.value.code == "public_finetune_raster_decode_invalid"


def _metric(crop_count: int, cer: float, exact: float) -> dict[str, object]:
    return {
        "crop_count": crop_count,
        "reference_character_count": crop_count * 5,
        "semantic_cer": cer,
        "semantic_crop_exact_match": exact,
    }


def _receipt_payload(stage: Path) -> dict[str, object]:
    artifacts = {
        relative: fine_tune.sha256_file(stage / relative) for relative in fine_tune._MODEL_FILES
    }
    profile_metrics = {profile: _metric(10, 0.09, 0.81) for profile in fine_tune._TRAINING_PROFILES}
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
            base_dictionary_sha256="7" * 64,
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
        "public_mobile_model_bundle_manifest_sha256": "f" * 64,
        "public_mobile_source_inventory_sha256": "1" * 64,
        "public_mobile_source_snapshot_sha256": "2" * 64,
        "public_recognizer_artifact_sha256s": {
            relative: str(index) * 64
            for index, relative in enumerate(fine_tune._BASE_RECOGNIZER_FILES, start=3)
        },
        "base_dictionary_sha256": "7" * 64,
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
        "best_development_by_profile": profile_metrics,
        "improvement_contract": "overall-cer-then-crop-exact-strict-v1",
        "evaluation_contract": "semantic_crop_cer_and_exact_without_critical_fields_v1",
        "initialization": "pinned_public_inference_graph_warm_start",
        "framework_license": "Apache-2.0",
        "static_roundtrip_verified": True,
        "model_artifact_sha256s": artifacts,
    }


def _v2_crop_manifest(split: str) -> fine_tune.PublicFineTuneCropManifestV2:
    selection_sha = "a" * 64
    train = split == "train"
    crop_count = 1_000 if train else 200
    family_count = 500 if train else 100
    profile_counts = (
        {"png-216": 334, "jpeg-216-q75": 337, "jpeg-150-q75": 329}
        if train
        else {"png-216": 67, "jpeg-216-q75": 67, "jpeg-150-q75": 66}
    )
    document_counts = (
        {
            "bill_of_lading": 334,
            "commercial_invoice": 334,
            "packing_list": 332,
        }
        if train
        else {
            "bill_of_lading": 68,
            "commercial_invoice": 66,
            "packing_list": 66,
        }
    )
    field_counts = {
        "bill_of_lading:bl_no": document_counts["bill_of_lading"],
        "commercial_invoice:inv_no": document_counts["commercial_invoice"],
        "packing_list:inv_no": document_counts["packing_list"],
    }
    return fine_tune.PublicFineTuneCropManifestV2(
        split=split,
        source_crop_manifest_sha256=("b" if train else "c") * 64,
        source_receipt_sha256="d" * 64,
        raster_receipt_sha256="e" * 64,
        selection_receipt_relative_path=(
            f"synthetic-raster-text-v1/fresh-scope-selections/{selection_sha}.json"
        ),
        selection_receipt_sha256=selection_sha,
        parent_selection_receipt_sha256="f" * 64,
        selected_family_ids_sha256=("1" if train else "2") * 64,
        excluded_family_ids_sha256="3" * 64,
        family_count=family_count,
        source_crop_count=crop_count,
        usable_crop_count=crop_count,
        semantic_crop_list_sha256="4" * 64,
        skipped_sample_list_sha256="5" * 64,
        base_dictionary_sha256="7" * 64,
        source_profile_crop_counts=profile_counts,
        usable_profile_crop_counts=profile_counts,
        critical_field_crop_count=crop_count,
        critical_field_ids_sha256=fine_tune.sha256_bytes(
            fine_tune.canonical_json_bytes(sorted(field_counts))
        ),
        critical_field_crop_counts=field_counts,
        document_type_crop_counts=document_counts,
        source_label_length_p50=13,
        source_label_length_p95=25,
    )


def _receipt_payload_v2(stage: Path) -> dict[str, object]:
    artifacts = {
        relative: fine_tune.sha256_file(stage / relative) for relative in fine_tune._MODEL_FILES
    }
    train = _v2_crop_manifest("train")
    development = _v2_crop_manifest("development")
    baseline_metrics = {
        profile: _metric(count, 0.10, 0.80)
        for profile, count in development.usable_profile_crop_counts.items()
    }
    best_metrics = {
        profile: _metric(count, 0.09, 0.81)
        for profile, count in development.usable_profile_crop_counts.items()
    }
    return {
        "artifact_kind": "source_attested_public_recognizer_finetune_receipt",
        "trainer_revision": "public-paddle-jit-ctc-leakage-safe-head-v2",
        "model_revision": "en-ppocrv5-mobile-rec-source-attested-head-v2",
        "semantic_label_revision": "source-attested-ocr-crop-semantic-label-v1",
        "stage": "leakage-safe-v2",
        "tuning_scope": "head",
        "training_intent_sha256": "6" * 64,
        "train_manifest_sha256": fine_tune._manifest_sha256(train),
        "development_manifest_sha256": fine_tune._manifest_sha256(development),
        "train_crop_manifest": train.model_dump(mode="json"),
        "development_crop_manifest": development.model_dump(mode="json"),
        "selection_receipt_relative_path": train.selection_receipt_relative_path,
        "selection_receipt_sha256": train.selection_receipt_sha256,
        "parent_selection_receipt_sha256": train.parent_selection_receipt_sha256,
        "overlap_verifier_revision": "selection-v2-all-holdouts-zero-overlap-v1",
        "train_holdout_overlap_count": 0,
        "direct_runtime_bundle_manifest_sha256": "d" * 64,
        "direct_runtime_package_inventory_sha256": "e" * 64,
        "public_mobile_model_bundle_manifest_sha256": "f" * 64,
        "public_mobile_source_inventory_sha256": "1" * 64,
        "public_mobile_source_snapshot_sha256": "2" * 64,
        "public_recognizer_artifact_sha256s": {
            relative: str(index) * 64
            for index, relative in enumerate(fine_tune._BASE_RECOGNIZER_FILES, start=3)
        },
        "base_dictionary_sha256": "7" * 64,
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
        "maximum_epochs": 4,
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
        "best_development_by_profile": best_metrics,
        "improvement_contract": "overall-cer-then-crop-exact-strict-v1",
        "evaluation_contract": "semantic_crop_cer_and_exact_without_critical_fields_v1",
        "initialization": "pinned_public_inference_graph_warm_start",
        "framework_license": "Apache-2.0",
        "static_roundtrip_verified": True,
        "model_artifact_sha256s": artifacts,
    }
    profile_metrics = {profile: _metric(10, 0.09, 0.81) for profile in fine_tune._TRAINING_PROFILES}
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
            base_dictionary_sha256="7" * 64,
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
        "public_mobile_model_bundle_manifest_sha256": "f" * 64,
        "public_mobile_source_inventory_sha256": "1" * 64,
        "public_mobile_source_snapshot_sha256": "2" * 64,
        "public_recognizer_artifact_sha256s": {
            relative: str(index) * 64
            for index, relative in enumerate(fine_tune._BASE_RECOGNIZER_FILES, start=3)
        },
        "base_dictionary_sha256": "7" * 64,
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
        "best_development_by_profile": profile_metrics,
        "improvement_contract": "overall-cer-then-crop-exact-strict-v1",
        "evaluation_contract": "semantic_crop_cer_and_exact_without_critical_fields_v1",
        "initialization": "pinned_public_inference_graph_warm_start",
        "framework_license": "Apache-2.0",
        "static_roundtrip_verified": True,
        "model_artifact_sha256s": artifacts,
    }


def test_release_is_content_addressed_and_exact_tree_is_reverified(tmp_path: Path) -> None:
    root = _preview_root(tmp_path)
    stage = fine_tune.create_public_finetune_stage_v1(root)
    text_rec = stage / "text-rec"
    text_rec.mkdir()
    for index, relative in enumerate(fine_tune._MODEL_FILES, start=1):
        path = stage / relative
        path.write_bytes(f"artifact-{index}".encode())

    release = fine_tune.publish_public_finetuned_recognizer_release_v1(
        root,
        staged_release=stage,
        receipt_payload=_receipt_payload(stage),
    )

    loaded = fine_tune.load_verified_public_finetuned_recognizer_release_v1(
        root, release.release_sha256
    )
    assert loaded.receipt.output_class_count == 438
    receipt_text = (loaded.release_path / "training-receipt.json").read_text("utf-8")
    assert "artifact-" not in receipt_text
    (loaded.release_path / "unexpected.txt").write_bytes(b"drift")
    with pytest.raises(fine_tune.SourceAttestedPublicRecognizerFineTuneError) as raised:
        fine_tune.load_verified_public_finetuned_recognizer_release_v1(root, release.release_sha256)
    assert raised.value.code == "public_finetune_release_invalid"


def test_v2_release_roundtrip_preserves_exact_selection_and_epoch_contract(
    tmp_path: Path,
) -> None:
    root = _preview_root(tmp_path)
    stage = fine_tune.create_public_finetune_stage_v1(root)
    (stage / "text-rec").mkdir()
    for index, relative in enumerate(fine_tune._MODEL_FILES, start=1):
        (stage / relative).write_bytes(f"v2-artifact-{index}".encode())

    release = fine_tune.publish_public_finetuned_recognizer_release_v2(
        root,
        staged_release=stage,
        receipt_payload=_receipt_payload_v2(stage),
    )
    loaded = fine_tune.load_verified_public_finetuned_recognizer_release_v1(
        root, release.release_sha256
    )

    assert isinstance(loaded.receipt, fine_tune.PublicFineTuneReceiptV2)
    assert loaded.receipt.maximum_epochs == 4
    assert loaded.receipt.train_holdout_overlap_count == 0
    assert loaded.receipt.train_crop_manifest.critical_field_crop_count == 1_000
    assert loaded.receipt.development_crop_manifest.usable_crop_count == 200


def test_runtime_is_locked_offline_and_uses_both_verified_bundles(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _preview_root(tmp_path)
    events: list[str] = []
    direct = object()
    mobile = object()
    inventory = root / "inventory.json"
    inventory.write_bytes(b"{}")
    expected = SimpleNamespace(release=SimpleNamespace(), reused=False)

    @contextmanager
    def no_network() -> Iterator[None]:
        events.append("offline-enter")
        yield
        events.append("offline-exit")

    @contextmanager
    def imports(bundle: object) -> Iterator[Path]:
        assert bundle is direct
        events.append("imports")
        yield inventory

    monkeypatch.setattr(runtime, "network_disabled", no_network)
    monkeypatch.setattr(runtime, "verified_direct_training_imports", imports)

    def direct_bundle(**_kwargs: object) -> object:
        events.append("direct")
        return direct

    def mobile_bundle(**_kwargs: object) -> object:
        events.append("mobile")
        return mobile

    monkeypatch.setattr(
        runtime,
        "materialize_verified_raster_text_direct_paddle_bundle_v1",
        direct_bundle,
    )
    monkeypatch.setattr(
        runtime,
        "materialize_verified_raster_text_direct_mobile_paddle_bundle_v1",
        mobile_bundle,
    )

    def train(path: Path, **kwargs: object) -> object:
        assert path == root
        assert kwargs == {
            "runtime_bundle": direct,
            "mobile_bundle": mobile,
            "package_inventory_path": inventory,
            "tuning_scope": "head",
        }
        events.append("train")
        return expected

    monkeypatch.setattr(runtime, "train_source_attested_public_recognizer_overfit_v1", train)

    def provenance(_root: Path, result: object) -> object:
        events.append("provenance")
        return result

    monkeypatch.setattr(runtime, "_verify_release_provenance", provenance)

    result = runtime.ensure_source_attested_public_recognizer_overfit_v1(root)

    assert result is expected
    assert events == [
        "offline-enter",
        "direct",
        "mobile",
        "imports",
        "train",
        "provenance",
        "offline-exit",
    ]
    assert (root / runtime._LOCK_FILE).is_file()


def test_v2_runtime_is_offline_head_only_and_does_not_expose_epoch_override(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _preview_root(tmp_path)
    events: list[str] = []
    direct = object()
    mobile = object()
    inventory = root / "inventory.json"
    inventory.write_bytes(b"{}")
    expected = SimpleNamespace(release=SimpleNamespace(), reused=False)

    @contextmanager
    def no_network() -> Iterator[None]:
        events.append("offline-enter")
        yield
        events.append("offline-exit")

    @contextmanager
    def imports(bundle: object) -> Iterator[Path]:
        assert bundle is direct
        yield inventory

    monkeypatch.setattr(runtime, "network_disabled", no_network)
    monkeypatch.setattr(runtime, "verified_direct_training_imports", imports)
    monkeypatch.setattr(
        runtime,
        "materialize_verified_raster_text_direct_paddle_bundle_v1",
        lambda **_kwargs: direct,
    )
    monkeypatch.setattr(
        runtime,
        "materialize_verified_raster_text_direct_mobile_paddle_bundle_v1",
        lambda **_kwargs: mobile,
    )

    def train(path: Path, **kwargs: object) -> object:
        assert path == root
        assert kwargs == {
            "runtime_bundle": direct,
            "mobile_bundle": mobile,
            "package_inventory_path": inventory,
        }
        events.append("train-v2")
        return expected

    monkeypatch.setattr(
        runtime,
        "train_source_attested_public_recognizer_leakage_safe_v2",
        train,
    )

    def provenance(_root: Path, result: object) -> object:
        events.append("provenance")
        return result

    monkeypatch.setattr(runtime, "_verify_release_provenance", provenance)

    result = runtime.ensure_source_attested_public_recognizer_leakage_safe_v2(root)

    assert result is expected
    assert events == [
        "offline-enter",
        "train-v2",
        "provenance",
        "offline-exit",
    ]
    with pytest.raises(fine_tune.SourceAttestedPublicRecognizerFineTuneError) as raised:
        runtime.ensure_source_attested_public_recognizer_leakage_safe_v2(
            root,
            tuning_scope="connected_tail",  # type: ignore[arg-type]
        )
    assert raised.value.code == "public_finetune_scope_invalid"


def test_v2_trainer_rejects_any_epoch_count_other_than_four(tmp_path: Path) -> None:
    root = _preview_root(tmp_path)
    with pytest.raises(fine_tune.SourceAttestedPublicRecognizerFineTuneError) as raised:
        fine_tune.train_source_attested_public_recognizer_leakage_safe_v2(
            root,
            runtime_bundle=object(),  # type: ignore[arg-type]
            mobile_bundle=object(),  # type: ignore[arg-type]
            package_inventory_path=root / "inventory.json",
            maximum_epochs=3,
        )
    assert raised.value.code == "public_finetune_configuration_invalid"


@pytest.mark.parametrize(
    ("failure", "expected_code"),
    [
        ("runtime", "public_finetune_runtime_bundle_invalid"),
        ("mobile", "public_finetune_mobile_bundle_invalid"),
        ("root", "public_finetune_bundle_root_mismatch"),
    ],
)
def test_training_bundle_preflight_reports_the_exact_failed_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    expected_code: str,
) -> None:
    root = _preview_root(tmp_path)
    other_root = tmp_path / "other"
    other_root.mkdir()
    supplied = SimpleNamespace(
        document_preview_root=root,
        manifest_sha256="a" * 64,
    )

    def load_runtime(**_kwargs: object) -> object:
        if failure == "runtime":
            raise OSError
        return SimpleNamespace(document_preview_root=root)

    def load_mobile(**_kwargs: object) -> object:
        if failure == "mobile":
            raise OSError
        return SimpleNamespace(document_preview_root=other_root if failure == "root" else root)

    monkeypatch.setattr(
        fine_tune,
        "load_verified_raster_text_direct_paddle_bundle_v1",
        load_runtime,
    )
    monkeypatch.setattr(
        fine_tune,
        "load_verified_raster_text_direct_mobile_paddle_bundle_v1",
        load_mobile,
    )
    with pytest.raises(fine_tune.SourceAttestedPublicRecognizerFineTuneError) as raised:
        fine_tune._verified_training_bundles(supplied, supplied)  # type: ignore[arg-type]
    assert raised.value.code == expected_code


def test_cli_exposes_only_fixed_head_leakage_safe_v2_and_content_free_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = _preview_root(tmp_path)
    release_path = root / "releases" / ("a" * 64)
    release_path.mkdir(parents=True)
    result = SimpleNamespace(
        release=SimpleNamespace(release_path=release_path, release_sha256="a" * 64),
        reused=False,
    )
    monkeypatch.setattr(
        cli,
        "ensure_source_attested_public_recognizer_leakage_safe_v2",
        lambda path, **kwargs: (
            result
            if path == root and kwargs == {"tuning_scope": "head"}
            else pytest.fail("CLI changed the fixed training contract")
        ),
    )

    summary = cli._run(root)

    assert summary == {
        "release_relative_path": f"releases/{'a' * 64}",
        "release_sha256": "a" * 64,
        "stage": "leakage-safe-v2",
        "status": "published",
        "tuning_scope": "head",
    }
    assert "label" not in json.dumps(summary)
