"""Source-attested domain fine-tuning for the pinned public English recognizer.

This module deliberately starts from the already attested
``en_PP-OCRv5_mobile_rec`` static graph.  It never reads calibration or sealed
Gold, never downloads a training repository/checkpoint, and never stores label
text in a manifest or receipt.  The only supervised inputs are source-PDFium
train/development crops produced by the existing immutable corpus compiler.

Paddle 3.3.1 can load the public inference export as ``PirTranslatedLayer``
and differentiate the connected recognition tail.  The bounded first stage
updates either only the two CTC-head tensors or the exact connected tail; the
detector is not trained here.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import random
import re
import shutil
import sys
import unicodedata
import uuid
from typing import Any, Literal, Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from customs_ai_agent.evaluation.raster_text_direct_paddle_bundle import (
    DirectPaddleBundleFileV1,
    RasterTextDirectPaddleBundleError,
    VerifiedRasterTextDirectPaddleBundleV1,
    _operational_path,
    load_verified_raster_text_direct_paddle_bundle_v1,
)
from customs_ai_agent.evaluation.raster_text_mobile_paddle_bundle import (
    RasterTextMobilePaddleBundleError,
    VerifiedRasterTextDirectMobilePaddleBundleV1,
    load_verified_raster_text_direct_mobile_paddle_bundle_v1,
)
from customs_ai_agent.evaluation import source_attested_public_raster_text_corpus
from customs_ai_agent.evaluation import (
    source_attested_raster_text_recognizer_training as source_crop_base,
)
from customs_ai_agent.evaluation.source_attested_public_raster_text_corpus import (
    SourceAttestedPublicCropCorpusError,
    build_verified_balanced_public_crop_corpus_v1,
    build_verified_leakage_safe_public_crop_corpus_v2,
)
from customs_ai_agent.evaluation.source_attested_raster_text_recognizer_training import (
    PageLruCache,
    TrainingCropV1,
    _ctc_decode_indices,
    _edit_distance,
    canonical_json_bytes,
    canonical_json_file_bytes,
    deterministic_width_batches,
    sha256_bytes,
    sha256_file,
    verify_paddle_training_package_inventory,
)


_SHA256 = r"^[a-f0-9]{64}$"
_DOCUMENT_PREVIEW_ROOT_SUFFIX = ("_local", "data", "document-preview")
_COLLECTION = "raster-text-public-finetuned-recognizers-v1"
_RELEASES = f"{_COLLECTION}/releases"
_SEMANTIC_LABEL_REVISION = "source-attested-ocr-crop-semantic-label-v1"
_TRAINER_REVISION = "public-paddle-jit-ctc-domain-finetune-v1"
_MODEL_REVISION = "en-ppocrv5-mobile-rec-source-attested-v1"
_TRAINER_REVISION_V2 = "public-paddle-jit-ctc-leakage-safe-head-v2"
_MODEL_REVISION_V2 = "en-ppocrv5-mobile-rec-source-attested-head-v2"
_CORPUS_REVISION_V2 = "source-attested-family-first-two-profile-crops-v2"
_OVERLAP_VERIFIER_REVISION_V2 = "selection-v2-all-holdouts-zero-overlap-v1"
_FORM_MARKER_POLICY = "remove-black-square-checkbox-and-underscore-run-ge-2-v1"
_WHITESPACE_POLICY = "nfc-map-control-whitespace-one-for-one-preserve-ascii-spaces-v1"
_BASE_MODEL_NAME = "en_PP-OCRv5_mobile_rec"
_OUTPUT_MODEL_NAME = "source_attested_en_PP-OCRv5_mobile_rec_v1"
_OUTPUT_MODEL_NAME_V2 = "source_attested_en_PP-OCRv5_mobile_rec_head_v2"
_BASE_RECOGNIZER_FILES = (
    "models/text-rec/config.json",
    "models/text-rec/inference.json",
    "models/text-rec/inference.pdiparams",
    "models/text-rec/inference.yml",
)
_MODEL_FILES = (
    "text-rec/config.json",
    "text-rec/inference.json",
    "text-rec/inference.pdiparams",
    "text-rec/inference.yml",
)
_RELEASE_FILES = (*_MODEL_FILES, "training-receipt.json")
_MAX_FILE_BYTES = 256 * 1024 * 1024
_DEFAULT_SEED = 1_937_429
_LEAKAGE_SAFE_MAXIMUM_EPOCHS = 4
_HEAD_LEARNING_RATE = 1.0e-4
_TAIL_LEARNING_RATE = 2.0e-5
_TRAINING_PROFILES = ("png-216", "jpeg-216-q75", "jpeg-150-q75")
_DOCUMENT_TYPES = ("bill_of_lading", "commercial_invoice", "packing_list")
_EXPECTED_HEAD_SHAPES: Mapping[str, tuple[int, ...]] = {
    "linear_8.b_0_0": (438,),
    "linear_8.w_0_0": (120, 438),
}
_BLACK_SQUARE = "\u25a0"
_CHECKBOX_MARKERS = frozenset(("\u2610", "\u2611", "\u2612", "\u25a1"))
_DECORATIVE_UNDERSCORE_RUN = re.compile(r"_{2,}")


class SourceAttestedPublicRecognizerFineTuneError(RuntimeError):
    """Stable, content-free failure raised by the public recognizer trainer."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class PublicFineTuneCropManifestV1(_WireModel):
    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["source_attested_public_recognizer_crop_manifest"] = (
        "source_attested_public_recognizer_crop_manifest"
    )
    semantic_label_revision: Literal["source-attested-ocr-crop-semantic-label-v1"] = (
        "source-attested-ocr-crop-semantic-label-v1"
    )
    split: Literal["train", "development"]
    stage: Literal["overfit"] = "overfit"
    source_crop_manifest_sha256: str = Field(pattern=_SHA256)
    source_receipt_sha256: str = Field(pattern=_SHA256)
    raster_receipt_sha256: str = Field(pattern=_SHA256)
    family_count: int = Field(ge=1)
    source_crop_count: int = Field(ge=1, le=1_000)
    usable_crop_count: int = Field(ge=1, le=1_000)
    skipped_decorative_crop_count: int = Field(ge=0, le=1_000)
    semantic_crop_list_sha256: str = Field(pattern=_SHA256)
    skipped_sample_list_sha256: str = Field(pattern=_SHA256)
    base_dictionary_sha256: str = Field(pattern=_SHA256)
    source_profile_crop_counts: dict[str, int]
    usable_profile_crop_counts: dict[str, int]
    scan_duplicate_pair_count: int = Field(ge=1, le=1_000)
    scan_duplicate_pixel_list_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _counts_match(self) -> "PublicFineTuneCropManifestV1":
        if self.usable_crop_count + self.skipped_decorative_crop_count != self.source_crop_count:
            raise ValueError("public fine-tune crop counts do not match")
        if (
            set(self.source_profile_crop_counts) != set(_TRAINING_PROFILES)
            or set(self.usable_profile_crop_counts) != set(_TRAINING_PROFILES)
            or sum(self.source_profile_crop_counts.values()) != self.source_crop_count
            or sum(self.usable_profile_crop_counts.values()) != self.usable_crop_count
            or any(value < 0 for value in self.usable_profile_crop_counts.values())
            or any(value < 1 for value in self.source_profile_crop_counts.values())
        ):
            raise ValueError("public fine-tune profile crop counts do not match")
        return self


class PublicFineTuneCropManifestV2(_WireModel):
    """Content-free manifest for a family-first, holdout-isolated crop split."""

    schema_version: Literal["2.0"] = "2.0"
    artifact_kind: Literal["source_attested_public_recognizer_crop_manifest"] = (
        "source_attested_public_recognizer_crop_manifest"
    )
    semantic_label_revision: Literal["source-attested-ocr-crop-semantic-label-v1"] = (
        "source-attested-ocr-crop-semantic-label-v1"
    )
    split: Literal["train", "development"]
    stage: Literal["leakage-safe-v2"] = "leakage-safe-v2"
    corpus_revision: Literal["source-attested-family-first-two-profile-crops-v2"] = (
        "source-attested-family-first-two-profile-crops-v2"
    )
    overlap_verifier_revision: Literal["selection-v2-all-holdouts-zero-overlap-v1"] = (
        "selection-v2-all-holdouts-zero-overlap-v1"
    )
    source_crop_manifest_sha256: str = Field(pattern=_SHA256)
    source_receipt_sha256: str = Field(pattern=_SHA256)
    raster_receipt_sha256: str = Field(pattern=_SHA256)
    selection_receipt_relative_path: str = Field(min_length=1, max_length=240)
    selection_receipt_sha256: str = Field(pattern=_SHA256)
    parent_selection_receipt_sha256: str = Field(pattern=_SHA256)
    selected_family_ids_sha256: str = Field(pattern=_SHA256)
    excluded_family_ids_sha256: str = Field(pattern=_SHA256)
    excluded_family_count: Literal[500] = 500
    holdout_overlap_count: Literal[0] = 0
    crops_per_family: Literal[2] = 2
    family_count: int = Field(ge=100, le=500)
    source_crop_count: int = Field(ge=200, le=1_000)
    usable_crop_count: int = Field(ge=200, le=1_000)
    skipped_decorative_crop_count: Literal[0] = 0
    semantic_crop_list_sha256: str = Field(pattern=_SHA256)
    skipped_sample_list_sha256: str = Field(pattern=_SHA256)
    base_dictionary_sha256: str = Field(pattern=_SHA256)
    source_profile_crop_counts: dict[str, int]
    usable_profile_crop_counts: dict[str, int]
    critical_evidence_policy: Literal["gold-field-value-equals-ordered-evidence-spans-v1"] = (
        "gold-field-value-equals-ordered-evidence-spans-v1"
    )
    critical_field_crop_count: int = Field(ge=200, le=1_000)
    critical_field_ids_sha256: str = Field(pattern=_SHA256)
    critical_field_crop_counts: dict[str, int]
    document_type_crop_counts: dict[str, int]
    source_label_length_p50: int = Field(ge=1, le=64)
    source_label_length_p95: int = Field(ge=1, le=64)

    @model_validator(mode="after")
    def _family_first_counts_match(self) -> "PublicFineTuneCropManifestV2":
        expected_families = 500 if self.split == "train" else 100
        expected_crops = expected_families * self.crops_per_family
        expected_selection_path = (
            f"synthetic-raster-text-v1/fresh-scope-selections/{self.selection_receipt_sha256}.json"
        )
        if (
            self.family_count != expected_families
            or self.source_crop_count != expected_crops
            or self.usable_crop_count != expected_crops
            or self.selection_receipt_relative_path != expected_selection_path
            or set(self.source_profile_crop_counts) != set(_TRAINING_PROFILES)
            or set(self.usable_profile_crop_counts) != set(_TRAINING_PROFILES)
            or self.source_profile_crop_counts != self.usable_profile_crop_counts
            or sum(self.source_profile_crop_counts.values()) != expected_crops
            or any(value < 1 for value in self.source_profile_crop_counts.values())
            or self.critical_field_crop_count != expected_crops
            or sum(self.critical_field_crop_counts.values()) != expected_crops
            or any(value < 1 for value in self.critical_field_crop_counts.values())
            or set(self.document_type_crop_counts) != set(_DOCUMENT_TYPES)
            or sum(self.document_type_crop_counts.values()) != expected_crops
            or max(self.document_type_crop_counts.values())
            - min(self.document_type_crop_counts.values())
            > 2
            or self.critical_field_ids_sha256
            != sha256_bytes(canonical_json_bytes(sorted(self.critical_field_crop_counts)))
            or self.source_label_length_p50 > self.source_label_length_p95
        ):
            raise ValueError("leakage-safe public fine-tune crop counts do not match")
        return self


class SemanticDevelopmentMetricV1(_WireModel):
    crop_count: int = Field(ge=1, le=1_000)
    reference_character_count: int = Field(ge=1)
    semantic_cer: float = Field(ge=0, le=100)
    semantic_crop_exact_match: float = Field(ge=0, le=1)


class PublicFineTuneReceiptV1(_WireModel):
    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["source_attested_public_recognizer_finetune_receipt"] = (
        "source_attested_public_recognizer_finetune_receipt"
    )
    trainer_revision: Literal["public-paddle-jit-ctc-domain-finetune-v1"] = (
        "public-paddle-jit-ctc-domain-finetune-v1"
    )
    model_revision: Literal["en-ppocrv5-mobile-rec-source-attested-v1"] = (
        "en-ppocrv5-mobile-rec-source-attested-v1"
    )
    semantic_label_revision: Literal["source-attested-ocr-crop-semantic-label-v1"] = (
        "source-attested-ocr-crop-semantic-label-v1"
    )
    stage: Literal["overfit"] = "overfit"
    tuning_scope: Literal["head", "connected_tail"]
    training_intent_sha256: str = Field(pattern=_SHA256)
    train_manifest_sha256: str = Field(pattern=_SHA256)
    development_manifest_sha256: str = Field(pattern=_SHA256)
    train_crop_manifest: PublicFineTuneCropManifestV1
    development_crop_manifest: PublicFineTuneCropManifestV1
    direct_runtime_bundle_manifest_sha256: str = Field(pattern=_SHA256)
    direct_runtime_package_inventory_sha256: str = Field(pattern=_SHA256)
    public_mobile_model_bundle_manifest_sha256: str = Field(pattern=_SHA256)
    public_mobile_source_inventory_sha256: str = Field(pattern=_SHA256)
    public_mobile_source_snapshot_sha256: str = Field(pattern=_SHA256)
    public_recognizer_artifact_sha256s: dict[str, str]
    base_dictionary_sha256: str = Field(pattern=_SHA256)
    base_character_count: int = Field(ge=1, le=512)
    output_class_count: int = Field(ge=2, le=514)
    ctc_blank_index: Literal[0] = 0
    dictionary_first_class_index: Literal[1] = 1
    implicit_space_class_index: int = Field(ge=2, le=513)
    form_marker_policy: Literal["remove-black-square-checkbox-and-underscore-run-ge-2-v1"] = (
        "remove-black-square-checkbox-and-underscore-run-ge-2-v1"
    )
    whitespace_policy: Literal[
        "nfc-map-control-whitespace-one-for-one-preserve-ascii-spaces-v1"
    ] = "nfc-map-control-whitespace-one-for-one-preserve-ascii-spaces-v1"
    training_module_sha256: str = Field(pattern=_SHA256)
    balanced_corpus_module_sha256: str = Field(pattern=_SHA256)
    source_crop_base_module_sha256: str = Field(pattern=_SHA256)
    head_tensor_contract_sha256: str = Field(pattern=_SHA256)
    trainable_parameter_name_list_sha256: str = Field(pattern=_SHA256)
    parent_python_revision: str = Field(pattern=r"^3\.[0-9]+$")
    seed_sha256: str = Field(pattern=_SHA256)
    batch_size: int = Field(ge=1, le=64)
    maximum_epochs: int = Field(ge=1, le=8)
    completed_epochs: int = Field(ge=1, le=8)
    best_epoch: int = Field(ge=1, le=8)
    early_stopped: bool
    learning_rate: float = Field(gt=0, le=1.0e-3)
    trainable_tensor_count: int = Field(ge=2, le=256)
    trainable_parameter_count: int = Field(ge=1, le=5_000_000)
    development_semantic_cer: float = Field(ge=0, le=100)
    development_semantic_crop_exact_match: float = Field(ge=0, le=1)
    baseline_development_semantic_cer: float = Field(ge=0, le=100)
    baseline_development_semantic_crop_exact_match: float = Field(ge=0, le=1)
    baseline_development_by_profile: dict[str, SemanticDevelopmentMetricV1]
    best_development_by_profile: dict[str, SemanticDevelopmentMetricV1]
    improvement_contract: Literal["overall-cer-then-crop-exact-strict-v1"] = (
        "overall-cer-then-crop-exact-strict-v1"
    )
    evaluation_contract: Literal["semantic_crop_cer_and_exact_without_critical_fields_v1"] = (
        "semantic_crop_cer_and_exact_without_critical_fields_v1"
    )
    initialization: Literal["pinned_public_inference_graph_warm_start"] = (
        "pinned_public_inference_graph_warm_start"
    )
    framework_license: Literal["Apache-2.0"] = "Apache-2.0"
    static_roundtrip_verified: Literal[True] = True
    model_artifact_sha256s: dict[str, str]
    receipt_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _receipt_is_bound(self) -> "PublicFineTuneReceiptV1":
        if set(self.public_recognizer_artifact_sha256s) != set(_BASE_RECOGNIZER_FILES):
            raise ValueError("public recognizer provenance is incomplete")
        if set(self.model_artifact_sha256s) != set(_MODEL_FILES):
            raise ValueError("fine-tuned recognizer artifacts are incomplete")
        if (
            self.train_crop_manifest.split != "train"
            or self.development_crop_manifest.split != "development"
            or _manifest_sha256(self.train_crop_manifest) != self.train_manifest_sha256
            or _manifest_sha256(self.development_crop_manifest) != self.development_manifest_sha256
            or self.train_crop_manifest.base_dictionary_sha256 != self.base_dictionary_sha256
            or self.development_crop_manifest.base_dictionary_sha256 != self.base_dictionary_sha256
            or self.train_crop_manifest.source_receipt_sha256
            != self.development_crop_manifest.source_receipt_sha256
            or self.train_crop_manifest.raster_receipt_sha256
            != self.development_crop_manifest.raster_receipt_sha256
        ):
            raise ValueError("public recognizer crop manifests are invalid")
        if (
            self.implicit_space_class_index != self.base_character_count + 1
            or self.output_class_count != self.base_character_count + 2
            or self.base_character_count != 436
            or self.implicit_space_class_index != 437
            or self.output_class_count != 438
        ):
            raise ValueError("public recognizer CTC class mapping is invalid")
        expected_head_contract = sha256_bytes(canonical_json_bytes(_EXPECTED_HEAD_SHAPES))
        expected_head_names = sha256_bytes(canonical_json_bytes(sorted(_EXPECTED_HEAD_SHAPES)))
        if (
            self.head_tensor_contract_sha256 != expected_head_contract
            or (
                self.tuning_scope == "head"
                and (
                    self.trainable_tensor_count != 2
                    or self.trainable_parameter_count != 52_998
                    or self.trainable_parameter_name_list_sha256 != expected_head_names
                    or self.learning_rate != _HEAD_LEARNING_RATE
                )
            )
            or (
                self.tuning_scope == "connected_tail"
                and (self.trainable_tensor_count <= 2 or self.learning_rate != _TAIL_LEARNING_RATE)
            )
        ):
            raise ValueError("public recognizer trainable tensor contract is invalid")
        if (
            set(self.baseline_development_by_profile) != set(_TRAINING_PROFILES)
            or set(self.best_development_by_profile) != set(_TRAINING_PROFILES)
            or not _strict_metric_improvement(
                candidate_cer=self.development_semantic_cer,
                candidate_exact=self.development_semantic_crop_exact_match,
                baseline_cer=self.baseline_development_semantic_cer,
                baseline_exact=self.baseline_development_semantic_crop_exact_match,
            )
        ):
            raise ValueError("public recognizer fine-tune did not improve development")
        expected_development_counts = self.development_crop_manifest.usable_profile_crop_counts
        if any(
            self.baseline_development_by_profile[profile].crop_count
            != expected_development_counts[profile]
            or self.best_development_by_profile[profile].crop_count
            != expected_development_counts[profile]
            for profile in _TRAINING_PROFILES
        ):
            raise ValueError("public recognizer development profile counts are invalid")
        for metrics, overall_cer, overall_exact in (
            (
                self.baseline_development_by_profile,
                self.baseline_development_semantic_cer,
                self.baseline_development_semantic_crop_exact_match,
            ),
            (
                self.best_development_by_profile,
                self.development_semantic_cer,
                self.development_semantic_crop_exact_match,
            ),
        ):
            characters = sum(item.reference_character_count for item in metrics.values())
            crops = sum(item.crop_count for item in metrics.values())
            weighted_cer = (
                sum(item.semantic_cer * item.reference_character_count for item in metrics.values())
                / characters
            )
            weighted_exact = (
                sum(item.semantic_crop_exact_match * item.crop_count for item in metrics.values())
                / crops
            )
            if (
                abs(weighted_cer - overall_cer) > 1.0e-8
                or abs(weighted_exact - overall_exact) > 1.0e-8
            ):
                raise ValueError("public recognizer development aggregate is invalid")
        if self.receipt_sha256 != public_finetune_receipt_sha256(self):
            raise ValueError("fine-tune receipt digest mismatch")
        return self


class PublicFineTuneReceiptV2(PublicFineTuneReceiptV1):
    """Head-only four-epoch receipt bound to the result-blind V2 holdouts."""

    schema_version: Literal["2.0"] = "2.0"  # type: ignore[assignment]
    trainer_revision: Literal[
        "public-paddle-jit-ctc-leakage-safe-head-v2"  # type: ignore[assignment]
    ] = (  # type: ignore[assignment]
        "public-paddle-jit-ctc-leakage-safe-head-v2"  # type: ignore[assignment]
    )
    model_revision: Literal[
        "en-ppocrv5-mobile-rec-source-attested-head-v2"  # type: ignore[assignment]
    ] = (  # type: ignore[assignment]
        "en-ppocrv5-mobile-rec-source-attested-head-v2"  # type: ignore[assignment]
    )
    stage: Literal["leakage-safe-v2"] = "leakage-safe-v2"  # type: ignore[assignment]
    tuning_scope: Literal["head"] = "head"
    train_crop_manifest: PublicFineTuneCropManifestV2  # type: ignore[assignment]
    development_crop_manifest: PublicFineTuneCropManifestV2  # type: ignore[assignment]
    selection_receipt_relative_path: str = Field(min_length=1, max_length=240)
    selection_receipt_sha256: str = Field(pattern=_SHA256)
    parent_selection_receipt_sha256: str = Field(pattern=_SHA256)
    overlap_verifier_revision: Literal["selection-v2-all-holdouts-zero-overlap-v1"] = (
        "selection-v2-all-holdouts-zero-overlap-v1"
    )
    train_holdout_overlap_count: Literal[0] = 0
    maximum_epochs: Literal[4] = 4
    completed_epochs: int = Field(ge=1, le=4)
    best_epoch: int = Field(ge=1, le=4)

    @model_validator(mode="after")
    def _selection_and_family_contract_is_bound(self) -> "PublicFineTuneReceiptV2":
        train = self.train_crop_manifest
        development = self.development_crop_manifest
        expected_path = (
            f"synthetic-raster-text-v1/fresh-scope-selections/{self.selection_receipt_sha256}.json"
        )
        if (
            self.selection_receipt_relative_path != expected_path
            or train.selection_receipt_relative_path != expected_path
            or development.selection_receipt_relative_path != expected_path
            or train.selection_receipt_sha256 != self.selection_receipt_sha256
            or development.selection_receipt_sha256 != self.selection_receipt_sha256
            or train.parent_selection_receipt_sha256 != self.parent_selection_receipt_sha256
            or development.parent_selection_receipt_sha256 != self.parent_selection_receipt_sha256
            or train.excluded_family_ids_sha256 != development.excluded_family_ids_sha256
            or train.holdout_overlap_count != self.train_holdout_overlap_count
            or train.family_count != 500
            or train.usable_crop_count != 1_000
            or development.family_count != 100
            or development.usable_crop_count != 200
        ):
            raise ValueError("leakage-safe public fine-tune selection is invalid")
        return self


PublicFineTuneReceipt = PublicFineTuneReceiptV1 | PublicFineTuneReceiptV2


class PublicFineTuneReleaseManifestV1(_WireModel):
    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["source_attested_public_recognizer_finetune_release"] = (
        "source_attested_public_recognizer_finetune_release"
    )
    receipt_sha256: str = Field(pattern=_SHA256)
    files: tuple[DirectPaddleBundleFileV1, ...] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def _exact_tree(self) -> "PublicFineTuneReleaseManifestV1":
        if tuple(item.path for item in self.files) != _RELEASE_FILES:
            raise ValueError("fine-tune release file set is invalid")
        return self


@dataclass(frozen=True, slots=True)
class BaseRecognizerDictionaryV1:
    characters: tuple[str, ...]
    character_to_index: Mapping[str, int]
    dictionary_sha256: str
    config_sha256: str
    output_class_count: int
    implicit_space_class_index: int
    config_value: Mapping[str, object]


@dataclass(slots=True)
class VerifiedPublicFineTuneCorpusV1:
    train_manifest: PublicFineTuneCropManifestV1
    development_manifest: PublicFineTuneCropManifestV1
    train_crops: tuple[TrainingCropV1, ...]
    development_crops: tuple[TrainingCropV1, ...]
    semantic_labels_by_sample_id: dict[str, str]
    profile_by_sample_id: dict[str, str]


@dataclass(slots=True)
class VerifiedPublicFineTuneCorpusV2:
    train_manifest: PublicFineTuneCropManifestV2
    development_manifest: PublicFineTuneCropManifestV2
    train_crops: tuple[TrainingCropV1, ...]
    development_crops: tuple[TrainingCropV1, ...]
    semantic_labels_by_sample_id: dict[str, str]
    profile_by_sample_id: dict[str, str]


@dataclass(frozen=True, slots=True)
class VerifiedPublicFineTunedRecognizerReleaseV1:
    release_sha256: str
    release_path: Path
    manifest: PublicFineTuneReleaseManifestV1
    receipt: PublicFineTuneReceipt
    file_paths: Mapping[str, Path]
    file_sha256s: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class PublicFineTuneRunResultV1:
    release: VerifiedPublicFineTunedRecognizerReleaseV1
    reused: bool


def _root(path: Path) -> Path:
    try:
        if path.is_symlink():
            raise OSError
        root = path.resolve(strict=True)
        if (
            root.is_symlink()
            or not root.is_dir()
            or tuple(part.casefold() for part in root.parts[-3:]) != _DOCUMENT_PREVIEW_ROOT_SUFFIX
        ):
            raise OSError
        return root
    except OSError:
        raise SourceAttestedPublicRecognizerFineTuneError(
            "public_finetune_document_preview_root_invalid"
        ) from None


def _relative_path(value: str) -> str:
    candidate = PurePosixPath(value)
    if (
        not value
        or "\\" in value
        or candidate.is_absolute()
        or any(part in {"", ".", ".."} for part in candidate.parts)
    ):
        raise ValueError("invalid relative path")
    return candidate.as_posix()


def _safe_path(root: Path, relative: str, *, directory: bool = False) -> Path:
    try:
        resolved_root = root.resolve(strict=True)
        operational_root = _operational_path(resolved_root, force_short=True)
        target = operational_root
        for part in PurePosixPath(_relative_path(relative)).parts:
            target = target / part
            if target.is_symlink():
                raise OSError
        target.parent.resolve(strict=True).relative_to(resolved_root)
        if directory and not target.is_dir():
            raise OSError
        if not directory and not target.is_file():
            raise OSError
        return target
    except (OSError, ValueError, RasterTextDirectPaddleBundleError):
        raise SourceAttestedPublicRecognizerFineTuneError(
            "public_finetune_release_invalid"
        ) from None


def _model_sha256(model: BaseModel) -> str:
    return sha256_bytes(canonical_json_bytes(model.model_dump(mode="json", warnings=False)))


def _manifest_sha256(model: BaseModel) -> str:
    return _model_sha256(model)


def public_finetune_receipt_sha256(receipt: PublicFineTuneReceipt) -> str:
    value = receipt.model_dump(mode="json", warnings=False)
    value["receipt_sha256"] = ""
    return sha256_bytes(canonical_json_bytes(value))


def _read_json(path: Path, *, limit: int = 4 * 1024 * 1024) -> tuple[dict[str, Any], bytes]:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > limit:
            raise OSError
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError
        return value, raw
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise SourceAttestedPublicRecognizerFineTuneError(
            "public_finetune_material_invalid"
        ) from None


def _canonical_model_bytes(model: BaseModel) -> bytes:
    return canonical_json_file_bytes(model.model_dump(mode="json", warnings=False))


def semantic_ocr_crop_label_v1(value: str) -> str | None:
    """Project one source span into the public recognizer's semantic alphabet.

    The removed symbols are visual form controls or fill-in rules, not user
    content.  Ordinary spaces, including repeated spaces, remain untouched;
    only non-space whitespace code points are mapped one-for-one to ASCII
    spaces because a one-line OCR crop cannot encode their control semantics.
    """

    if not isinstance(value, str) or not value:
        raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_label_invalid")
    normalized = unicodedata.normalize("NFC", value)
    normalized = "".join(" " if character.isspace() else character for character in normalized)
    normalized = normalized.replace(_BLACK_SQUARE, "")
    for marker in _CHECKBOX_MARKERS:
        normalized = normalized.replace(marker, "")
    normalized = _DECORATIVE_UNDERSCORE_RUN.sub("", normalized)
    if normalized and any(not character.isspace() for character in normalized):
        return normalized
    source_nonspace = "".join(
        character for character in unicodedata.normalize("NFC", value) if not character.isspace()
    )
    decorative_source = "".join(
        character
        for character in source_nonspace
        if character not in {_BLACK_SQUARE, *_CHECKBOX_MARKERS}
    )
    if not source_nonspace or (
        decorative_source and _DECORATIVE_UNDERSCORE_RUN.fullmatch(decorative_source) is None
    ):
        raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_label_invalid")
    return None


def load_public_mobile_recognizer_dictionary_v1(
    bundle: VerifiedRasterTextDirectMobilePaddleBundleV1,
) -> BaseRecognizerDictionaryV1:
    """Load the exact public CTC mapping from one already verified mobile bundle."""

    try:
        verified = load_verified_raster_text_direct_mobile_paddle_bundle_v1(
            document_preview_root=bundle.document_preview_root,
            manifest_sha256=bundle.manifest_sha256,
        )
        path = _safe_path(verified.bundle_root, "models/text-rec/config.json")
        value, raw = _read_json(path)
        if verified.file_sha256("models/text-rec/config.json") != hashlib.sha256(raw).hexdigest():
            raise ValueError
        global_value = value.get("Global")
        postprocess = value.get("PostProcess")
        if (
            not isinstance(global_value, dict)
            or global_value.get("model_name") != _BASE_MODEL_NAME
            or not isinstance(postprocess, dict)
            or postprocess.get("name") != "CTCLabelDecode"
        ):
            raise ValueError
        raw_characters = postprocess.get("character_dict")
        if (
            not isinstance(raw_characters, list)
            or len(raw_characters) != 436
            or any(not isinstance(item, str) or len(item) != 1 for item in raw_characters)
            or len(raw_characters) != len(set(raw_characters))
            or " " in raw_characters
        ):
            raise ValueError
        characters = tuple(raw_characters)
        mapping = {character: index + 1 for index, character in enumerate(characters)}
        mapping[" "] = len(characters) + 1
        dictionary_sha = sha256_bytes(canonical_json_bytes(list(characters)))
        return BaseRecognizerDictionaryV1(
            characters=characters,
            character_to_index=mapping,
            dictionary_sha256=dictionary_sha,
            config_sha256=hashlib.sha256(raw).hexdigest(),
            output_class_count=len(characters) + 2,
            implicit_space_class_index=len(characters) + 1,
            config_value=value,
        )
    except SourceAttestedPublicRecognizerFineTuneError:
        raise
    except (KeyError, OSError, ValueError, RasterTextMobilePaddleBundleError):
        raise SourceAttestedPublicRecognizerFineTuneError(
            "public_finetune_base_dictionary_invalid"
        ) from None


def _strict_metric_improvement(
    *,
    candidate_cer: float,
    candidate_exact: float,
    baseline_cer: float,
    baseline_exact: float,
) -> bool:
    return candidate_cer < baseline_cer - 1.0e-9 or (
        abs(candidate_cer - baseline_cer) <= 1.0e-9 and candidate_exact > baseline_exact + 1.0e-9
    )


def _raw_balanced_manifest_sha256(
    *,
    crops: Sequence[TrainingCropV1],
    profile_by_sample_id: Mapping[str, str],
    source_receipt_sha256: str,
    raster_receipt_sha256: str,
    scan_duplicate_pair_count: int,
    scan_duplicate_pixel_list_sha256: str,
) -> str:
    records = [
        {
            "asset_id_sha256": sha256_bytes(crop.asset_id.encode("utf-8")),
            "bbox": [round(value, 12) for value in crop.bbox],
            "family_id_sha256": sha256_bytes(crop.family_id.encode("utf-8")),
            "gold_sha256": crop.gold_sha256,
            "label_length": crop.label_length,
            "label_sha256": crop.label_sha256,
            "profile_id": profile_by_sample_id[crop.sample_id],
            "raster_sha256": crop.raster_sha256,
            "sample_id": crop.sample_id,
            "source_pdf_sha256": crop.source_pdf_sha256,
        }
        for crop in crops
    ]
    return sha256_bytes(
        canonical_json_bytes(
            {
                "raster_receipt_sha256": raster_receipt_sha256,
                "records": records,
                "scan_duplicate_pair_count": scan_duplicate_pair_count,
                "scan_duplicate_pixel_list_sha256": scan_duplicate_pixel_list_sha256,
                "source_receipt_sha256": source_receipt_sha256,
            }
        )
    )


def _semantic_split(
    *,
    crops: Sequence[TrainingCropV1],
    labels_by_sample_id: Mapping[str, str],
    split: Literal["train", "development"],
    source_manifest_sha256: str,
    source_receipt_sha256: str,
    raster_receipt_sha256: str,
    profile_by_sample_id: Mapping[str, str],
    scan_duplicate_pair_count: int,
    scan_duplicate_pixel_list_sha256: str,
    dictionary: BaseRecognizerDictionaryV1,
) -> tuple[PublicFineTuneCropManifestV1, tuple[TrainingCropV1, ...], dict[str, str]]:
    transformed: list[TrainingCropV1] = []
    semantic_labels: dict[str, str] = {}
    skipped: list[str] = []
    source_profile_counts = {profile: 0 for profile in _TRAINING_PROFILES}
    usable_profile_counts = {profile: 0 for profile in _TRAINING_PROFILES}
    for crop in crops:
        if crop.split != split:
            raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_split_invalid")
        raw_label = labels_by_sample_id.get(crop.sample_id)
        profile = profile_by_sample_id.get(crop.sample_id)
        if profile not in source_profile_counts:
            raise SourceAttestedPublicRecognizerFineTuneError(
                "public_finetune_profile_binding_invalid"
            )
        source_profile_counts[profile] += 1
        if raw_label is None or sha256_bytes(raw_label.encode("utf-8")) != crop.label_sha256:
            raise SourceAttestedPublicRecognizerFineTuneError(
                "public_finetune_label_binding_invalid"
            )
        semantic = semantic_ocr_crop_label_v1(raw_label)
        if semantic is None:
            skipped.append(crop.sample_id)
            continue
        unsupported = {
            character for character in semantic if character not in dictionary.character_to_index
        }
        if unsupported:
            raise SourceAttestedPublicRecognizerFineTuneError(
                "public_finetune_label_dictionary_unsupported"
            )
        if len(semantic) > 512:
            raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_label_invalid")
        transformed_crop = crop.model_copy(
            update={
                "label_sha256": sha256_bytes(semantic.encode("utf-8")),
                "label_length": len(semantic),
            }
        )
        transformed.append(transformed_crop)
        semantic_labels[crop.sample_id] = semantic
        usable_profile_counts[profile] += 1
    if not transformed:
        raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_no_usable_crops")
    semantic_records = [
        {
            "asset_id": crop.asset_id,
            "family_id": crop.family_id,
            "label_length": crop.label_length,
            "label_sha256": crop.label_sha256,
            "sample_id": crop.sample_id,
            "sample_kind": crop.sample_kind,
        }
        for crop in transformed
    ]
    manifest = PublicFineTuneCropManifestV1(
        split=split,
        source_crop_manifest_sha256=source_manifest_sha256,
        source_receipt_sha256=source_receipt_sha256,
        raster_receipt_sha256=raster_receipt_sha256,
        family_count=len({crop.family_id for crop in transformed}),
        source_crop_count=len(crops),
        usable_crop_count=len(transformed),
        skipped_decorative_crop_count=len(skipped),
        semantic_crop_list_sha256=sha256_bytes(canonical_json_bytes(semantic_records)),
        skipped_sample_list_sha256=sha256_bytes(canonical_json_bytes(sorted(skipped))),
        base_dictionary_sha256=dictionary.dictionary_sha256,
        source_profile_crop_counts=source_profile_counts,
        usable_profile_crop_counts=usable_profile_counts,
        scan_duplicate_pair_count=scan_duplicate_pair_count,
        scan_duplicate_pixel_list_sha256=scan_duplicate_pixel_list_sha256,
    )
    return manifest, tuple(transformed), semantic_labels


def build_verified_public_finetune_overfit_corpus_v1(
    document_preview_root: Path,
    *,
    dictionary: BaseRecognizerDictionaryV1,
    seed: int = _DEFAULT_SEED,
) -> VerifiedPublicFineTuneCorpusV1:
    """Reverify and project exactly the bounded train/development crop corpus."""

    root = _root(document_preview_root)
    try:
        source = build_verified_balanced_public_crop_corpus_v1(root, seed=seed)
    except SourceAttestedPublicCropCorpusError as error:
        raise SourceAttestedPublicRecognizerFineTuneError(error.code) from None
    train_source_manifest_sha = _raw_balanced_manifest_sha256(
        crops=source.train_crops,
        profile_by_sample_id=source.profile_by_sample_id,
        source_receipt_sha256=source.source_receipt_sha256,
        raster_receipt_sha256=source.raster_receipt_sha256,
        scan_duplicate_pair_count=source.train_scan_duplicate_pair_count,
        scan_duplicate_pixel_list_sha256=source.train_scan_duplicate_pixel_list_sha256,
    )
    development_source_manifest_sha = _raw_balanced_manifest_sha256(
        crops=source.development_crops,
        profile_by_sample_id=source.profile_by_sample_id,
        source_receipt_sha256=source.source_receipt_sha256,
        raster_receipt_sha256=source.raster_receipt_sha256,
        scan_duplicate_pair_count=source.development_scan_duplicate_pair_count,
        scan_duplicate_pixel_list_sha256=(source.development_scan_duplicate_pixel_list_sha256),
    )
    train_manifest, train, train_labels = _semantic_split(
        crops=source.train_crops,
        labels_by_sample_id=source.labels_by_sample_id,
        split="train",
        source_manifest_sha256=train_source_manifest_sha,
        source_receipt_sha256=source.source_receipt_sha256,
        raster_receipt_sha256=source.raster_receipt_sha256,
        profile_by_sample_id=source.profile_by_sample_id,
        scan_duplicate_pair_count=source.train_scan_duplicate_pair_count,
        scan_duplicate_pixel_list_sha256=source.train_scan_duplicate_pixel_list_sha256,
        dictionary=dictionary,
    )
    development_manifest, development, development_labels = _semantic_split(
        crops=source.development_crops,
        labels_by_sample_id=source.labels_by_sample_id,
        split="development",
        source_manifest_sha256=development_source_manifest_sha,
        source_receipt_sha256=source.source_receipt_sha256,
        raster_receipt_sha256=source.raster_receipt_sha256,
        profile_by_sample_id=source.profile_by_sample_id,
        scan_duplicate_pair_count=source.development_scan_duplicate_pair_count,
        scan_duplicate_pixel_list_sha256=(source.development_scan_duplicate_pixel_list_sha256),
        dictionary=dictionary,
    )
    if {crop.family_id for crop in train} & {crop.family_id for crop in development}:
        raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_family_leakage")
    labels = {**train_labels, **development_labels}
    if len(labels) != len(train) + len(development):
        raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_label_binding_invalid")
    return VerifiedPublicFineTuneCorpusV1(
        train_manifest=train_manifest,
        development_manifest=development_manifest,
        train_crops=train,
        development_crops=development,
        semantic_labels_by_sample_id=labels,
        profile_by_sample_id={
            crop.sample_id: source.profile_by_sample_id[crop.sample_id]
            for crop in (*train, *development)
        },
    )


def _raw_family_first_manifest_sha256_v2(
    *,
    crops: Sequence[TrainingCropV1],
    profile_by_sample_id: Mapping[str, str],
    field_key_by_sample_id: Mapping[str, str],
    document_type_by_sample_id: Mapping[str, str],
    source_receipt_sha256: str,
    raster_receipt_sha256: str,
    selection_receipt_relative_path: str,
    selection_receipt_sha256: str,
    parent_selection_receipt_sha256: str,
    selected_family_ids_sha256: str,
    excluded_family_ids_sha256: str,
) -> str:
    records = [
        {
            "asset_id_sha256": sha256_bytes(crop.asset_id.encode("utf-8")),
            "bbox": [round(value, 12) for value in crop.bbox],
            "family_id_sha256": sha256_bytes(crop.family_id.encode("utf-8")),
            "gold_sha256": crop.gold_sha256,
            "critical_field_key": field_key_by_sample_id[crop.sample_id],
            "document_type": document_type_by_sample_id[crop.sample_id],
            "label_length": crop.label_length,
            "label_sha256": crop.label_sha256,
            "profile_id": profile_by_sample_id[crop.sample_id],
            "raster_sha256": crop.raster_sha256,
            "sample_id": crop.sample_id,
            "source_pdf_sha256": crop.source_pdf_sha256,
        }
        for crop in crops
    ]
    return sha256_bytes(
        canonical_json_bytes(
            {
                "corpus_revision": _CORPUS_REVISION_V2,
                "excluded_family_ids_sha256": excluded_family_ids_sha256,
                "overlap_verifier_revision": _OVERLAP_VERIFIER_REVISION_V2,
                "parent_selection_receipt_sha256": (parent_selection_receipt_sha256),
                "raster_receipt_sha256": raster_receipt_sha256,
                "records": records,
                "selected_family_ids_sha256": selected_family_ids_sha256,
                "selection_receipt_relative_path": selection_receipt_relative_path,
                "selection_receipt_sha256": selection_receipt_sha256,
                "source_receipt_sha256": source_receipt_sha256,
            }
        )
    )


def _semantic_split_v2(
    *,
    crops: Sequence[TrainingCropV1],
    labels_by_sample_id: Mapping[str, str],
    split: Literal["train", "development"],
    source_manifest_sha256: str,
    source_receipt_sha256: str,
    raster_receipt_sha256: str,
    profile_by_sample_id: Mapping[str, str],
    field_key_by_sample_id: Mapping[str, str],
    document_type_by_sample_id: Mapping[str, str],
    selection_receipt_relative_path: str,
    selection_receipt_sha256: str,
    parent_selection_receipt_sha256: str,
    selected_family_ids_sha256: str,
    excluded_family_ids_sha256: str,
    dictionary: BaseRecognizerDictionaryV1,
) -> tuple[PublicFineTuneCropManifestV2, tuple[TrainingCropV1, ...], dict[str, str]]:
    transformed: list[TrainingCropV1] = []
    semantic_labels: dict[str, str] = {}
    profile_counts = {profile: 0 for profile in _TRAINING_PROFILES}
    field_counts: dict[str, int] = {}
    document_counts = {document_type: 0 for document_type in _DOCUMENT_TYPES}
    for crop in crops:
        if crop.split != split:
            raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_split_invalid")
        raw_label = labels_by_sample_id.get(crop.sample_id)
        profile = profile_by_sample_id.get(crop.sample_id)
        field_key = field_key_by_sample_id.get(crop.sample_id)
        document_type = document_type_by_sample_id.get(crop.sample_id)
        if profile not in profile_counts:
            raise SourceAttestedPublicRecognizerFineTuneError(
                "public_finetune_profile_binding_invalid"
            )
        if (
            not isinstance(field_key, str)
            or ":" not in field_key
            or document_type not in document_counts
            or not field_key.startswith(f"{document_type}:")
        ):
            raise SourceAttestedPublicRecognizerFineTuneError(
                "public_finetune_field_binding_invalid"
            )
        if raw_label is None or sha256_bytes(raw_label.encode("utf-8")) != crop.label_sha256:
            raise SourceAttestedPublicRecognizerFineTuneError(
                "public_finetune_label_binding_invalid"
            )
        semantic = semantic_ocr_crop_label_v1(raw_label)
        if semantic is None:
            raise SourceAttestedPublicRecognizerFineTuneError(
                "public_finetune_family_first_candidate_invalid"
            )
        if any(character not in dictionary.character_to_index for character in semantic):
            raise SourceAttestedPublicRecognizerFineTuneError(
                "public_finetune_label_dictionary_unsupported"
            )
        transformed_crop = crop.model_copy(
            update={
                "label_sha256": sha256_bytes(semantic.encode("utf-8")),
                "label_length": len(semantic),
            }
        )
        transformed.append(transformed_crop)
        semantic_labels[crop.sample_id] = semantic
        profile_counts[profile] += 1
        field_counts[field_key] = field_counts.get(field_key, 0) + 1
        document_counts[document_type] += 1
    expected_families = 500 if split == "train" else 100
    expected_crops = expected_families * 2
    families = {crop.family_id for crop in transformed}
    family_counts = {
        family_id: sum(crop.family_id == family_id for crop in transformed)
        for family_id in families
    }
    if (
        len(transformed) != expected_crops
        or len(families) != expected_families
        or set(family_counts.values()) != {2}
    ):
        raise SourceAttestedPublicRecognizerFineTuneError(
            "public_finetune_family_first_coverage_invalid"
        )
    semantic_records = [
        {
            "asset_id": crop.asset_id,
            "family_id": crop.family_id,
            "label_length": crop.label_length,
            "label_sha256": crop.label_sha256,
            "sample_id": crop.sample_id,
            "sample_kind": crop.sample_kind,
        }
        for crop in transformed
    ]
    label_lengths = sorted(crop.label_length for crop in transformed)

    def percentile_nearest_rank(percent: int) -> int:
        index = min(
            len(label_lengths) - 1,
            (len(label_lengths) * percent + 99) // 100 - 1,
        )
        return label_lengths[index]

    manifest = PublicFineTuneCropManifestV2(
        split=split,
        source_crop_manifest_sha256=source_manifest_sha256,
        source_receipt_sha256=source_receipt_sha256,
        raster_receipt_sha256=raster_receipt_sha256,
        selection_receipt_relative_path=selection_receipt_relative_path,
        selection_receipt_sha256=selection_receipt_sha256,
        parent_selection_receipt_sha256=parent_selection_receipt_sha256,
        selected_family_ids_sha256=selected_family_ids_sha256,
        excluded_family_ids_sha256=excluded_family_ids_sha256,
        family_count=expected_families,
        source_crop_count=expected_crops,
        usable_crop_count=expected_crops,
        semantic_crop_list_sha256=sha256_bytes(canonical_json_bytes(semantic_records)),
        skipped_sample_list_sha256=sha256_bytes(canonical_json_bytes([])),
        base_dictionary_sha256=dictionary.dictionary_sha256,
        source_profile_crop_counts=profile_counts,
        usable_profile_crop_counts=profile_counts,
        critical_field_crop_count=expected_crops,
        critical_field_ids_sha256=sha256_bytes(canonical_json_bytes(sorted(field_counts))),
        critical_field_crop_counts=field_counts,
        document_type_crop_counts=document_counts,
        source_label_length_p50=percentile_nearest_rank(50),
        source_label_length_p95=percentile_nearest_rank(95),
    )
    return manifest, tuple(transformed), semantic_labels


def build_verified_public_finetune_leakage_safe_corpus_v2(
    document_preview_root: Path,
    *,
    dictionary: BaseRecognizerDictionaryV1,
    seed: int = _DEFAULT_SEED,
) -> VerifiedPublicFineTuneCorpusV2:
    """Rebuild the selection-V2-isolated 1,000/200 family-first corpus."""

    root = _root(document_preview_root)
    try:
        source = build_verified_leakage_safe_public_crop_corpus_v2(root, seed=seed)
    except SourceAttestedPublicCropCorpusError as error:
        raise SourceAttestedPublicRecognizerFineTuneError(error.code) from None
    train_source_manifest_sha = _raw_family_first_manifest_sha256_v2(
        crops=source.train_crops,
        profile_by_sample_id=source.profile_by_sample_id,
        field_key_by_sample_id=source.field_key_by_sample_id,
        document_type_by_sample_id=source.document_type_by_sample_id,
        source_receipt_sha256=source.source_receipt_sha256,
        raster_receipt_sha256=source.raster_receipt_sha256,
        selection_receipt_relative_path=source.selection_receipt_relative_path,
        selection_receipt_sha256=source.selection_receipt_sha256,
        parent_selection_receipt_sha256=source.parent_selection_receipt_sha256,
        selected_family_ids_sha256=source.train_family_ids_sha256,
        excluded_family_ids_sha256=source.excluded_family_ids_sha256,
    )
    development_source_manifest_sha = _raw_family_first_manifest_sha256_v2(
        crops=source.development_crops,
        profile_by_sample_id=source.profile_by_sample_id,
        field_key_by_sample_id=source.field_key_by_sample_id,
        document_type_by_sample_id=source.document_type_by_sample_id,
        source_receipt_sha256=source.source_receipt_sha256,
        raster_receipt_sha256=source.raster_receipt_sha256,
        selection_receipt_relative_path=source.selection_receipt_relative_path,
        selection_receipt_sha256=source.selection_receipt_sha256,
        parent_selection_receipt_sha256=source.parent_selection_receipt_sha256,
        selected_family_ids_sha256=source.development_family_ids_sha256,
        excluded_family_ids_sha256=source.excluded_family_ids_sha256,
    )
    train_manifest, train, train_labels = _semantic_split_v2(
        crops=source.train_crops,
        labels_by_sample_id=source.labels_by_sample_id,
        split="train",
        source_manifest_sha256=train_source_manifest_sha,
        source_receipt_sha256=source.source_receipt_sha256,
        raster_receipt_sha256=source.raster_receipt_sha256,
        profile_by_sample_id=source.profile_by_sample_id,
        field_key_by_sample_id=source.field_key_by_sample_id,
        document_type_by_sample_id=source.document_type_by_sample_id,
        selection_receipt_relative_path=source.selection_receipt_relative_path,
        selection_receipt_sha256=source.selection_receipt_sha256,
        parent_selection_receipt_sha256=source.parent_selection_receipt_sha256,
        selected_family_ids_sha256=source.train_family_ids_sha256,
        excluded_family_ids_sha256=source.excluded_family_ids_sha256,
        dictionary=dictionary,
    )
    development_manifest, development, development_labels = _semantic_split_v2(
        crops=source.development_crops,
        labels_by_sample_id=source.labels_by_sample_id,
        split="development",
        source_manifest_sha256=development_source_manifest_sha,
        source_receipt_sha256=source.source_receipt_sha256,
        raster_receipt_sha256=source.raster_receipt_sha256,
        profile_by_sample_id=source.profile_by_sample_id,
        field_key_by_sample_id=source.field_key_by_sample_id,
        document_type_by_sample_id=source.document_type_by_sample_id,
        selection_receipt_relative_path=source.selection_receipt_relative_path,
        selection_receipt_sha256=source.selection_receipt_sha256,
        parent_selection_receipt_sha256=source.parent_selection_receipt_sha256,
        selected_family_ids_sha256=source.development_family_ids_sha256,
        excluded_family_ids_sha256=source.excluded_family_ids_sha256,
        dictionary=dictionary,
    )
    if {crop.family_id for crop in train} & {crop.family_id for crop in development}:
        raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_family_leakage")
    labels = {**train_labels, **development_labels}
    if len(labels) != 1_200:
        raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_label_binding_invalid")
    return VerifiedPublicFineTuneCorpusV2(
        train_manifest=train_manifest,
        development_manifest=development_manifest,
        train_crops=train,
        development_crops=development,
        semantic_labels_by_sample_id=labels,
        profile_by_sample_id={
            crop.sample_id: source.profile_by_sample_id[crop.sample_id]
            for crop in (*train, *development)
        },
    )


def _public_recognizer_artifacts(
    bundle: VerifiedRasterTextDirectMobilePaddleBundleV1,
) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in _BASE_RECOGNIZER_FILES:
        path = _safe_path(bundle.bundle_root, relative)
        digest = sha256_file(path)
        if bundle.file_sha256(relative) != digest:
            raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_base_model_drift")
        result[relative] = digest
    return result


def _base_recognizer_prefix(bundle: VerifiedRasterTextDirectMobilePaddleBundleV1) -> Path:
    directory = _safe_path(bundle.bundle_root, "models/text-rec", directory=True)
    for relative in _BASE_RECOGNIZER_FILES:
        _safe_path(bundle.bundle_root, relative)
    return directory / "inference"


def _original_named_parameters(layer: Any) -> dict[str, Any]:
    try:
        values: dict[str, Any] = {}
        for _generated, parameter in layer.named_parameters():
            original = getattr(parameter, "name", None)
            if not isinstance(original, str) or not original or original in values:
                raise ValueError
            values[original] = parameter
        if not values:
            raise ValueError
        return values
    except (AttributeError, TypeError, ValueError):
        raise SourceAttestedPublicRecognizerFineTuneError(
            "public_finetune_translated_layer_invalid"
        ) from None


def _probability_output(
    paddle: Any, output: Any, *, batch_size: int, output_class_count: int
) -> Any:
    try:
        if isinstance(output, (list, tuple)):
            if len(output) != 1:
                raise ValueError
            output = output[0]
        shape = tuple(int(value) for value in output.shape)
        if (
            len(shape) != 3
            or shape[0] != batch_size
            or shape[1] < 1
            or shape[2] != output_class_count
            or not bool(paddle.isfinite(output).all().item())
            or float(output.min().item()) < -1.0e-6
            or float(output.max().item()) > 1.000001
            or float(paddle.abs(output.sum(axis=-1) - 1.0).max().item()) > 1.0e-4
        ):
            raise ValueError
        return output
    except (AttributeError, TypeError, ValueError):
        raise SourceAttestedPublicRecognizerFineTuneError(
            "public_finetune_model_output_invalid"
        ) from None


def _ctc_loss(
    paddle: Any,
    *,
    probabilities: Any,
    labels: Any,
    label_lengths: Any,
) -> Any:
    try:
        time_steps = int(probabilities.shape[1])
        batch_size = int(probabilities.shape[0])
        input_lengths = paddle.full((batch_size,), time_steps, dtype="int64")
        log_probabilities = paddle.log(paddle.clip(probabilities, min=1.0e-7, max=1.0))
        loss = paddle.nn.functional.ctc_loss(
            log_probabilities.transpose([1, 0, 2]),
            labels,
            input_lengths,
            label_lengths,
            blank=0,
            reduction="mean",
        )
        if not bool(paddle.isfinite(loss).all().item()):
            raise ValueError
        return loss
    except SourceAttestedPublicRecognizerFineTuneError:
        raise
    except Exception:
        raise SourceAttestedPublicRecognizerFineTuneError(
            "public_finetune_ctc_loss_invalid"
        ) from None


def select_public_recognizer_trainable_parameters_v1(
    paddle: Any,
    *,
    layer: Any,
    output_class_count: int,
    tuning_scope: Literal["head", "connected_tail"],
) -> tuple[tuple[str, Any], ...]:
    """Select an exact, graph-proven trainable subset from the pinned export."""

    parameters = _original_named_parameters(layer)
    for parameter in parameters.values():
        parameter.stop_gradient = False
        parameter.clear_gradient()
    try:
        probe = _probability_output(
            paddle,
            layer(paddle.zeros((1, 3, 48, 160), dtype="float32")),
            batch_size=1,
            output_class_count=output_class_count,
        )
        labels = paddle.to_tensor([[1, 2, 3]], dtype="int32")
        label_lengths = paddle.to_tensor([3], dtype="int64")
        _ctc_loss(
            paddle,
            probabilities=probe,
            labels=labels,
            label_lengths=label_lengths,
        ).backward()
        connected = tuple(
            sorted(name for name, value in parameters.items() if value.grad is not None)
        )
        head = tuple(sorted(name for name in connected if name in _EXPECTED_HEAD_SHAPES))
        if set(head) != set(_EXPECTED_HEAD_SHAPES):
            raise SourceAttestedPublicRecognizerFineTuneError(
                "public_finetune_head_topology_invalid"
            )
        chosen = head if tuning_scope == "head" else connected
        if tuning_scope not in {"head", "connected_tail"}:
            raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_scope_invalid")
        for name, parameter in parameters.items():
            parameter.clear_gradient()
            parameter.stop_gradient = name not in chosen
        selected = tuple((name, parameters[name]) for name in chosen)
        if tuning_scope == "head":
            shapes = {
                name: tuple(int(value) for value in parameter.shape) for name, parameter in selected
            }
            expected_shapes = {
                "linear_8.b_0_0": (output_class_count,),
                "linear_8.w_0_0": (120, output_class_count),
            }
            if shapes != expected_shapes or expected_shapes != _EXPECTED_HEAD_SHAPES:
                raise SourceAttestedPublicRecognizerFineTuneError(
                    "public_finetune_head_topology_invalid"
                )
        return selected
    except SourceAttestedPublicRecognizerFineTuneError:
        raise
    except Exception:
        raise SourceAttestedPublicRecognizerFineTuneError(
            "public_finetune_gradient_topology_invalid"
        ) from None


def _load_verified_training_page(
    document_preview_root: Path,
    *,
    crop: TrainingCropV1,
    profile_id: str,
) -> Any:
    if profile_id not in _TRAINING_PROFILES:
        raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_profile_binding_invalid")
    path = _safe_path(document_preview_root, crop.asset_relative_path)
    if sha256_file(path) != crop.raster_sha256:
        raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_raster_drift")
    try:
        from PIL import Image

        with Image.open(path) as opened:
            expected_format = "PNG" if profile_id == "png-216" else "JPEG"
            orientation = opened.getexif().get(274)
            if (
                opened.format != expected_format
                or opened.size != (crop.page_width, crop.page_height)
                or getattr(opened, "n_frames", 1) != 1
                or orientation not in {None, 1}
            ):
                raise ValueError
            opened.load()
            return opened.convert("RGB")
    except SourceAttestedPublicRecognizerFineTuneError:
        raise
    except Exception:
        raise SourceAttestedPublicRecognizerFineTuneError(
            "public_finetune_raster_decode_invalid"
        ) from None


def _prepare_public_paddle_batch(
    paddle: Any,
    *,
    document_preview_root: Path,
    crops: Sequence[TrainingCropV1],
    labels_by_sample_id: Mapping[str, str],
    profile_by_sample_id: Mapping[str, str],
    character_to_index: Mapping[str, int],
    page_cache: PageLruCache[Any],
) -> tuple[Any, Any, Any]:
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        raise SourceAttestedPublicRecognizerFineTuneError(
            "public_finetune_runtime_unavailable"
        ) from None
    prepared: list[Any] = []
    widths: list[int] = []
    encoded: list[list[int]] = []
    for crop in crops:
        profile = profile_by_sample_id.get(crop.sample_id)
        if profile not in _TRAINING_PROFILES:
            raise SourceAttestedPublicRecognizerFineTuneError(
                "public_finetune_profile_binding_invalid"
            )

        def load_page(
            current_crop: TrainingCropV1 = crop,
            current_profile: str = profile,
        ) -> Any:
            return _load_verified_training_page(
                document_preview_root,
                crop=current_crop,
                profile_id=current_profile,
            )

        page = page_cache.get((profile, crop.raster_sha256), load_page)
        x, y, width, height = crop.bbox
        pad_x = width * 0.08
        pad_y = height * 0.15
        left = max(0, math.floor((x - pad_x) * crop.page_width))
        top = max(0, math.floor((y - pad_y) * crop.page_height))
        right = min(crop.page_width, math.ceil((x + width + pad_x) * crop.page_width))
        bottom = min(crop.page_height, math.ceil((y + height + pad_y) * crop.page_height))
        if right - left < 2 or bottom - top < 2:
            raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_crop_invalid")
        image = page.crop((left, top, right, bottom))
        resized_width = min(
            3_200,
            max(8, int(math.ceil(48.0 * image.width / max(1, image.height)))),
        )
        image = image.resize((resized_width, 48), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype="float32") / np.float32(255.0)
        array = array[:, :, ::-1]
        array = (array - np.float32(0.5)) / np.float32(0.5)
        prepared.append(np.ascontiguousarray(array.transpose((2, 0, 1))))
        widths.append(resized_width)
        label = labels_by_sample_id.get(crop.sample_id)
        if label is None or sha256_bytes(label.encode("utf-8")) != crop.label_sha256:
            raise SourceAttestedPublicRecognizerFineTuneError(
                "public_finetune_label_binding_invalid"
            )
        try:
            encoded.append([character_to_index[character] for character in label])
        except KeyError:
            raise SourceAttestedPublicRecognizerFineTuneError(
                "public_finetune_label_dictionary_unsupported"
            ) from None
    maximum_width = max(widths)
    images = np.zeros((len(crops), 3, 48, maximum_width), dtype="float32")
    for index, array in enumerate(prepared):
        images[index, :, :, : array.shape[2]] = array
    maximum_label = max(len(value) for value in encoded)
    label_array = np.zeros((len(crops), maximum_label), dtype="int32")
    for index, value in enumerate(encoded):
        label_array[index, : len(value)] = value
    label_lengths = np.asarray([len(value) for value in encoded], dtype="int64")
    return (
        paddle.to_tensor(images),
        paddle.to_tensor(label_array),
        paddle.to_tensor(label_lengths),
    )


def _train_batch(
    paddle: Any,
    *,
    model: Any,
    optimizer: Any,
    images: Any,
    labels: Any,
    label_lengths: Any,
    output_class_count: int,
    selected_parameters: Sequence[tuple[str, Any]],
) -> None:
    output = _probability_output(
        paddle,
        model(images),
        batch_size=int(images.shape[0]),
        output_class_count=output_class_count,
    )
    loss = _ctc_loss(
        paddle,
        probabilities=output,
        labels=labels,
        label_lengths=label_lengths,
    )
    loss.backward()
    for _name, parameter in selected_parameters:
        if parameter.grad is None or not bool(paddle.isfinite(parameter.grad).all().item()):
            raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_gradient_invalid")
    optimizer.step()
    optimizer.clear_grad()


def _evaluate(
    paddle: Any,
    *,
    model: Any,
    document_preview_root: Path,
    crops: Sequence[TrainingCropV1],
    labels_by_sample_id: Mapping[str, str],
    profile_by_sample_id: Mapping[str, str],
    character_to_index: Mapping[str, int],
    output_class_count: int,
    seed: int,
    batch_size: int,
) -> tuple[float, float, dict[str, SemanticDevelopmentMetricV1]]:
    edits = 0
    reference_characters = 0
    exact = 0
    total = 0
    profile_totals = {
        profile: {"edits": 0, "characters": 0, "exact": 0, "total": 0}
        for profile in _TRAINING_PROFILES
    }
    cache: PageLruCache[Any] = PageLruCache(16)
    model.eval()
    with paddle.no_grad():
        for batch in deterministic_width_batches(crops, epoch=0, seed=seed, batch_size=batch_size):
            images, labels, label_lengths = _prepare_public_paddle_batch(
                paddle,
                document_preview_root=document_preview_root,
                crops=batch,
                labels_by_sample_id=labels_by_sample_id,
                profile_by_sample_id=profile_by_sample_id,
                character_to_index=character_to_index,
                page_cache=cache,
            )
            output = _probability_output(
                paddle,
                model(images),
                batch_size=len(batch),
                output_class_count=output_class_count,
            )
            predictions = paddle.argmax(output, axis=-1).numpy().tolist()
            label_values = labels.numpy().tolist()
            length_values = label_lengths.numpy().tolist()
            for crop, prediction, expected, length in zip(
                batch, predictions, label_values, length_values, strict=True
            ):
                profile = profile_by_sample_id.get(crop.sample_id)
                if profile not in profile_totals:
                    raise SourceAttestedPublicRecognizerFineTuneError(
                        "public_finetune_profile_binding_invalid"
                    )
                candidate = _ctc_decode_indices([int(value) for value in prediction])
                reference = [int(value) for value in expected[: int(length)]]
                item_edits = _edit_distance(reference, candidate)
                item_exact = candidate == reference
                edits += item_edits
                reference_characters += len(reference)
                exact += item_exact
                total += 1
                profile_totals[profile]["edits"] += item_edits
                profile_totals[profile]["characters"] += len(reference)
                profile_totals[profile]["exact"] += item_exact
                profile_totals[profile]["total"] += 1
    if total != len(crops) or reference_characters < 1:
        raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_evaluation_invalid")
    model.train()
    by_profile: dict[str, SemanticDevelopmentMetricV1] = {}
    for profile, values in profile_totals.items():
        if values["total"] < 1 or values["characters"] < 1:
            raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_evaluation_invalid")
        by_profile[profile] = SemanticDevelopmentMetricV1(
            crop_count=values["total"],
            reference_character_count=values["characters"],
            semantic_cer=values["edits"] / values["characters"],
            semantic_crop_exact_match=values["exact"] / values["total"],
        )
    return edits / reference_characters, exact / total, by_profile


def _write_new(path: Path, value: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(value)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        raise SourceAttestedPublicRecognizerFineTuneError(
            "public_finetune_publish_failed"
        ) from None


def _model_artifact_hashes(directory: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for relative in _MODEL_FILES:
        path = directory.parent.joinpath(*PurePosixPath(relative).parts)
        if (
            path.is_symlink()
            or not path.is_file()
            or not 1 <= path.stat().st_size <= _MAX_FILE_BYTES
        ):
            raise SourceAttestedPublicRecognizerFineTuneError(
                "public_finetune_static_export_invalid"
            )
        result[relative] = sha256_file(path)
    return result


def _native_export_prefix(text_rec_directory: Path) -> Path:
    try:
        canonical = text_rec_directory.resolve(strict=True)
        if os.name != "nt":
            return canonical / "inference"
        import ctypes

        required = ctypes.windll.kernel32.GetShortPathNameW(str(canonical), None, 0)
        if not isinstance(required, int) or required < 1:
            raise OSError
        buffer = ctypes.create_unicode_buffer(required + 1)
        written = ctypes.windll.kernel32.GetShortPathNameW(str(canonical), buffer, len(buffer))
        if not isinstance(written, int) or not 0 < written < len(buffer):
            raise OSError
        if not buffer.value or not buffer.value.isascii():
            raise OSError
        return Path(buffer.value) / "inference"
    except Exception:
        raise SourceAttestedPublicRecognizerFineTuneError(
            "public_finetune_export_path_unsupported"
        ) from None


def export_public_finetuned_recognizer_v1(
    paddle: Any,
    *,
    model: Any,
    text_rec_directory: Path,
    dictionary: BaseRecognizerDictionaryV1,
    output_model_name: str = _OUTPUT_MODEL_NAME,
) -> dict[str, str]:
    """Export, JIT-reload, and native-predictor-check one trained graph."""

    failure_stage = "preflight"
    try:
        if text_rec_directory.exists() or text_rec_directory.is_symlink():
            raise OSError
        text_rec_directory.mkdir(parents=True)
        prefix = _native_export_prefix(text_rec_directory)
        model.eval()
        widths = (128, 320)
        probes = {width: paddle.zeros((1, 3, 48, width), dtype="float32") for width in widths}
        failure_stage = "eager_probe"
        before = {
            width: _probability_output(
                paddle,
                model(probe),
                batch_size=1,
                output_class_count=dictionary.output_class_count,
            ).numpy()
            for width, probe in probes.items()
        }
        failure_stage = "jit_save"
        paddle.jit.save(
            model,
            str(prefix),
            input_spec=[paddle.static.InputSpec([None, 3, 48, None], "float32")],
        )
        failure_stage = "artifact_set"
        allowed = {"inference.json", "inference.pdiparams"}
        actual = {item.name for item in text_rec_directory.iterdir() if item.is_file()}
        if actual != allowed or any(item.is_symlink() for item in text_rec_directory.iterdir()):
            raise ValueError
        if output_model_name not in {_OUTPUT_MODEL_NAME, _OUTPUT_MODEL_NAME_V2}:
            raise ValueError
        failure_stage = "metadata_write"
        config_value = {
            "Global": {"model_name": output_model_name},
            "PostProcess": {
                "character_dict": list(dictionary.characters),
                "name": "CTCLabelDecode",
            },
            "PreProcess": dictionary.config_value.get("PreProcess", {}),
        }
        _write_new(
            text_rec_directory / "config.json",
            canonical_json_file_bytes(config_value),
        )
        _write_new(
            text_rec_directory / "inference.yml",
            (f"Global:\n  model_name: {output_model_name}\n").encode("utf-8"),
        )
        failure_stage = "jit_roundtrip"
        reloaded = paddle.jit.load(str(prefix))
        reloaded.eval()
        import numpy as np

        for width, probe in probes.items():
            after = _probability_output(
                paddle,
                reloaded(probe),
                batch_size=1,
                output_class_count=dictionary.output_class_count,
            ).numpy()
            if before[width].shape != after.shape or not bool(
                np.allclose(before[width], after, rtol=1.0e-5, atol=1.0e-6)
            ):
                raise ValueError
        failure_stage = "native_configuration"
        configuration = paddle.inference.Config(
            str(prefix.with_suffix(".json")),
            str(prefix.with_suffix(".pdiparams")),
        )
        configuration.disable_gpu()
        disable_mkldnn = getattr(configuration, "disable_mkldnn", None)
        if not callable(disable_mkldnn):
            raise ValueError
        disable_mkldnn()
        configuration.disable_glog_info()
        configuration.set_cpu_math_library_num_threads(1)
        predictor = paddle.inference.create_predictor(configuration)
        input_names = list(predictor.get_input_names())
        output_names = list(predictor.get_output_names())
        if len(input_names) != 1 or len(output_names) != 1:
            raise ValueError
        native_input = predictor.get_input_handle(input_names[0])
        failure_stage = "native_prediction"
        for width in widths:
            native_input.reshape([1, 3, 48, width])
            native_input.copy_from_cpu(np.zeros((1, 3, 48, width), dtype="float32"))
            predictor.run()
            native = predictor.get_output_handle(output_names[0]).copy_to_cpu()
            if (
                native.ndim != 3
                or native.shape[0] != 1
                or native.shape[1] != before[width].shape[1]
                or native.shape[2] != dictionary.output_class_count
                or not bool(np.isfinite(native).all())
            ):
                raise ValueError
        failure_stage = "artifact_hash"
        try:
            return _model_artifact_hashes(text_rec_directory)
        except SourceAttestedPublicRecognizerFineTuneError as exc:
            if exc.code == "public_finetune_static_export_invalid":
                raise SourceAttestedPublicRecognizerFineTuneError(
                    "public_finetune_static_export_artifact_hash_invalid"
                ) from None
            raise
    except SourceAttestedPublicRecognizerFineTuneError:
        raise
    except Exception:
        raise SourceAttestedPublicRecognizerFineTuneError(
            f"public_finetune_static_export_{failure_stage}_invalid"
        ) from None


def create_public_finetune_stage_v1(document_preview_root: Path) -> Path:
    root = _root(document_preview_root)
    collection = root / _COLLECTION
    try:
        collection.mkdir(parents=True, exist_ok=True)
        if collection.is_symlink() or not collection.is_dir():
            raise OSError
        stage = collection / f".staging-{uuid.uuid4().hex}"
        stage.mkdir(mode=0o700)
        return stage
    except OSError:
        raise SourceAttestedPublicRecognizerFineTuneError(
            "public_finetune_publish_failed"
        ) from None


def _tree_entries(root: Path) -> tuple[set[str], set[str]]:
    files: set[str] = set()
    directories: set[str] = set()
    try:
        for item in root.rglob("*"):
            if item.is_symlink():
                raise OSError
            relative = item.relative_to(root).as_posix()
            if item.is_file():
                files.add(relative)
            elif item.is_dir():
                directories.add(relative)
            else:
                raise OSError
        return files, directories
    except (OSError, ValueError):
        raise SourceAttestedPublicRecognizerFineTuneError(
            "public_finetune_release_invalid"
        ) from None


def publish_public_finetuned_recognizer_release_v1(
    document_preview_root: Path,
    *,
    staged_release: Path,
    receipt_payload: Mapping[str, object],
) -> VerifiedPublicFineTunedRecognizerReleaseV1:
    root = _root(document_preview_root)
    collection = root / _COLLECTION
    try:
        stage_identity = staged_release.resolve(strict=True)
        stage_identity.relative_to(collection.resolve(strict=True))
        stage = _operational_path(stage_identity, force_short=True)
        if (
            stage_identity.is_symlink()
            or not stage.is_dir()
            or not stage_identity.name.startswith(".staging-")
        ):
            raise OSError
        files, directories = _tree_entries(stage)
        if files != set(_MODEL_FILES) or directories != {"text-rec"}:
            raise OSError
        artifacts = {relative: sha256_file(stage / relative) for relative in _MODEL_FILES}
        if receipt_payload.get("model_artifact_sha256s") != artifacts:
            raise ValueError
        candidate = dict(receipt_payload)
        candidate["schema_version"] = "1.0"
        candidate["receipt_sha256"] = ""
        receipt = PublicFineTuneReceiptV1.model_validate(
            {
                **candidate,
                "receipt_sha256": sha256_bytes(canonical_json_bytes(candidate)),
            }
        )
        _write_new(stage / "training-receipt.json", _canonical_model_bytes(receipt))
        records = tuple(
            DirectPaddleBundleFileV1(
                path=relative,
                byte_size=(stage / relative).stat().st_size,
                sha256=sha256_file(stage / relative),
            )
            for relative in _RELEASE_FILES
        )
        manifest = PublicFineTuneReleaseManifestV1(
            receipt_sha256=receipt.receipt_sha256,
            files=records,
        )
        raw_manifest = _canonical_model_bytes(manifest)
        release_sha = hashlib.sha256(raw_manifest).hexdigest()
        _write_new(stage / "manifest.json", raw_manifest)
        releases = collection / "releases"
        releases.mkdir(parents=True, exist_ok=True)
        final = _operational_path(releases.resolve(strict=True), force_short=True) / release_sha
        if final.exists() or final.is_symlink():
            existing = load_verified_public_finetuned_recognizer_release_v1(root, release_sha)
            shutil.rmtree(stage)
            return existing
        os.rename(stage, final)
        return load_verified_public_finetuned_recognizer_release_v1(root, release_sha)
    except SourceAttestedPublicRecognizerFineTuneError:
        raise
    except (OSError, ValueError):
        raise SourceAttestedPublicRecognizerFineTuneError(
            "public_finetune_publish_failed"
        ) from None


def publish_public_finetuned_recognizer_release_v2(
    document_preview_root: Path,
    *,
    staged_release: Path,
    receipt_payload: Mapping[str, object],
) -> VerifiedPublicFineTunedRecognizerReleaseV1:
    """Publish a schema-2 receipt without changing the exact release tree."""

    root = _root(document_preview_root)
    collection = root / _COLLECTION
    try:
        stage_identity = staged_release.resolve(strict=True)
        stage_identity.relative_to(collection.resolve(strict=True))
        stage = _operational_path(stage_identity, force_short=True)
        if (
            stage_identity.is_symlink()
            or not stage.is_dir()
            or not stage_identity.name.startswith(".staging-")
        ):
            raise OSError
        files, directories = _tree_entries(stage)
        if files != set(_MODEL_FILES) or directories != {"text-rec"}:
            raise OSError
        artifacts = {relative: sha256_file(stage / relative) for relative in _MODEL_FILES}
        if receipt_payload.get("model_artifact_sha256s") != artifacts:
            raise ValueError
        candidate = dict(receipt_payload)
        candidate["schema_version"] = "2.0"
        candidate["receipt_sha256"] = ""
        receipt = PublicFineTuneReceiptV2.model_validate(
            {
                **candidate,
                "receipt_sha256": sha256_bytes(canonical_json_bytes(candidate)),
            }
        )
        _write_new(stage / "training-receipt.json", _canonical_model_bytes(receipt))
        records = tuple(
            DirectPaddleBundleFileV1(
                path=relative,
                byte_size=(stage / relative).stat().st_size,
                sha256=sha256_file(stage / relative),
            )
            for relative in _RELEASE_FILES
        )
        manifest = PublicFineTuneReleaseManifestV1(
            receipt_sha256=receipt.receipt_sha256,
            files=records,
        )
        raw_manifest = _canonical_model_bytes(manifest)
        release_sha = hashlib.sha256(raw_manifest).hexdigest()
        _write_new(stage / "manifest.json", raw_manifest)
        releases = collection / "releases"
        releases.mkdir(parents=True, exist_ok=True)
        final = _operational_path(releases.resolve(strict=True), force_short=True) / release_sha
        if final.exists() or final.is_symlink():
            existing = load_verified_public_finetuned_recognizer_release_v1(root, release_sha)
            shutil.rmtree(stage)
            if not isinstance(existing.receipt, PublicFineTuneReceiptV2):
                raise ValueError
            return existing
        os.rename(stage, final)
        published = load_verified_public_finetuned_recognizer_release_v1(root, release_sha)
        if not isinstance(published.receipt, PublicFineTuneReceiptV2):
            raise ValueError
        return published
    except SourceAttestedPublicRecognizerFineTuneError:
        raise
    except (OSError, ValueError):
        raise SourceAttestedPublicRecognizerFineTuneError(
            "public_finetune_publish_failed"
        ) from None


def load_verified_public_finetuned_recognizer_release_v1(
    document_preview_root: Path,
    release_sha256: str,
) -> VerifiedPublicFineTunedRecognizerReleaseV1:
    root = _root(document_preview_root)
    if re.fullmatch(_SHA256, release_sha256) is None:
        raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_release_invalid")
    release = _safe_path(root, f"{_RELEASES}/{release_sha256}", directory=True)
    manifest_path = _safe_path(release, "manifest.json")
    value, raw = _read_json(manifest_path)
    try:
        manifest = PublicFineTuneReleaseManifestV1.model_validate(value)
        if (
            raw != _canonical_model_bytes(manifest)
            or hashlib.sha256(raw).hexdigest() != release_sha256
        ):
            raise ValueError
        files, directories = _tree_entries(release)
        if files != {*_RELEASE_FILES, "manifest.json"} or directories != {"text-rec"}:
            raise ValueError
        paths: dict[str, Path] = {}
        digests: dict[str, str] = {}
        for record in manifest.files:
            path = _safe_path(release, record.path)
            if path.stat().st_size != record.byte_size or sha256_file(path) != record.sha256:
                raise ValueError
            paths[record.path] = path
            digests[record.path] = record.sha256
        receipt_value, receipt_raw = _read_json(paths["training-receipt.json"])
        if receipt_value.get("schema_version") == "1.0":
            receipt: PublicFineTuneReceipt = PublicFineTuneReceiptV1.model_validate(receipt_value)
        elif receipt_value.get("schema_version") == "2.0":
            receipt = PublicFineTuneReceiptV2.model_validate(receipt_value)
        else:
            raise ValueError
        if (
            receipt_raw != _canonical_model_bytes(receipt)
            or receipt.receipt_sha256 != manifest.receipt_sha256
        ):
            raise ValueError
        return VerifiedPublicFineTunedRecognizerReleaseV1(
            release_sha256=release_sha256,
            release_path=release,
            manifest=manifest,
            receipt=receipt,
            file_paths=paths,
            file_sha256s=digests,
        )
    except SourceAttestedPublicRecognizerFineTuneError:
        raise
    except (KeyError, OSError, ValueError):
        raise SourceAttestedPublicRecognizerFineTuneError(
            "public_finetune_release_invalid"
        ) from None


def find_verified_public_finetune_release_by_intent_v1(
    document_preview_root: Path,
    training_intent_sha256: str,
) -> VerifiedPublicFineTunedRecognizerReleaseV1 | None:
    root = _root(document_preview_root)
    if re.fullmatch(_SHA256, training_intent_sha256) is None:
        raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_intent_invalid")
    releases = root / _RELEASES
    if not releases.exists():
        return None
    if releases.is_symlink() or not releases.is_dir():
        raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_release_invalid")
    found: list[VerifiedPublicFineTunedRecognizerReleaseV1] = []
    for candidate in sorted(releases.iterdir(), key=lambda item: item.name):
        if (
            candidate.is_symlink()
            or not candidate.is_dir()
            or re.fullmatch(_SHA256, candidate.name) is None
        ):
            raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_release_invalid")
        release = load_verified_public_finetuned_recognizer_release_v1(root, candidate.name)
        if release.receipt.training_intent_sha256 == training_intent_sha256:
            found.append(release)
    if len(found) > 1:
        raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_release_ambiguous")
    return found[0] if found else None


def _verified_training_bundles(
    runtime_bundle: VerifiedRasterTextDirectPaddleBundleV1,
    mobile_bundle: VerifiedRasterTextDirectMobilePaddleBundleV1,
) -> tuple[VerifiedRasterTextDirectPaddleBundleV1, VerifiedRasterTextDirectMobilePaddleBundleV1]:
    try:
        runtime = load_verified_raster_text_direct_paddle_bundle_v1(
            document_preview_root=runtime_bundle.document_preview_root,
            manifest_sha256=runtime_bundle.manifest_sha256,
        )
    except (
        OSError,
        ValueError,
        RasterTextDirectPaddleBundleError,
    ):
        raise SourceAttestedPublicRecognizerFineTuneError(
            "public_finetune_runtime_bundle_invalid"
        ) from None
    try:
        mobile = load_verified_raster_text_direct_mobile_paddle_bundle_v1(
            document_preview_root=mobile_bundle.document_preview_root,
            manifest_sha256=mobile_bundle.manifest_sha256,
        )
    except (
        OSError,
        ValueError,
        RasterTextMobilePaddleBundleError,
    ):
        raise SourceAttestedPublicRecognizerFineTuneError(
            "public_finetune_mobile_bundle_invalid"
        ) from None
    if runtime.document_preview_root != mobile.document_preview_root:
        raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_bundle_root_mismatch")
    return runtime, mobile


def train_source_attested_public_recognizer_overfit_v1(
    document_preview_root: Path,
    *,
    runtime_bundle: VerifiedRasterTextDirectPaddleBundleV1,
    mobile_bundle: VerifiedRasterTextDirectMobilePaddleBundleV1,
    package_inventory_path: Path,
    tuning_scope: Literal["head", "connected_tail"] = "head",
    maximum_epochs: int = 6,
    batch_size: int = 32,
    seed: int = _DEFAULT_SEED,
) -> PublicFineTuneRunResultV1:
    """Run the bounded 1,000-crop public-model warm-start stage."""

    if tuning_scope not in {"head", "connected_tail"}:
        raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_scope_invalid")
    if not 1 <= maximum_epochs <= 8 or not 1 <= batch_size <= 64:
        raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_configuration_invalid")
    root = _root(document_preview_root)
    runtime, mobile = _verified_training_bundles(runtime_bundle, mobile_bundle)
    try:
        package_inventory_sha = verify_paddle_training_package_inventory(package_inventory_path)
    except Exception:
        raise SourceAttestedPublicRecognizerFineTuneError(
            "public_finetune_package_inventory_invalid"
        ) from None
    if package_inventory_sha != runtime.file_sha256(
        runtime.manifest.sidecar_package_inventory_relative_path
    ):
        raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_package_inventory_drift")
    dictionary = load_public_mobile_recognizer_dictionary_v1(mobile)
    base_artifacts = _public_recognizer_artifacts(mobile)
    corpus = build_verified_public_finetune_overfit_corpus_v1(
        root, dictionary=dictionary, seed=seed
    )
    learning_rate = _HEAD_LEARNING_RATE if tuning_scope == "head" else _TAIL_LEARNING_RATE
    module_path = Path(__file__).resolve(strict=True)
    corpus_module_value = source_attested_public_raster_text_corpus.__file__
    source_crop_base_value = source_crop_base.__file__
    if not isinstance(corpus_module_value, str) or not isinstance(source_crop_base_value, str):
        raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_source_compiler_invalid")
    try:
        corpus_module_path = Path(corpus_module_value).resolve(strict=True)
        source_crop_base_path = Path(source_crop_base_value).resolve(strict=True)
    except OSError:
        raise SourceAttestedPublicRecognizerFineTuneError(
            "public_finetune_source_compiler_invalid"
        ) from None
    train_manifest_sha = _manifest_sha256(corpus.train_manifest)
    development_manifest_sha = _manifest_sha256(corpus.development_manifest)
    intent = {
        "batch_size": batch_size,
        "base_character_count": len(dictionary.characters),
        "base_dictionary_sha256": dictionary.dictionary_sha256,
        "balanced_corpus_module_sha256": sha256_file(corpus_module_path),
        "development_manifest_sha256": development_manifest_sha,
        "direct_runtime_bundle_manifest_sha256": runtime.manifest_sha256,
        "direct_runtime_package_inventory_sha256": package_inventory_sha,
        "learning_rate": learning_rate,
        "ctc_blank_index": 0,
        "dictionary_first_class_index": 1,
        "form_marker_policy": _FORM_MARKER_POLICY,
        "head_tensor_contract_sha256": sha256_bytes(canonical_json_bytes(_EXPECTED_HEAD_SHAPES)),
        "implicit_space_class_index": dictionary.implicit_space_class_index,
        "maximum_epochs": maximum_epochs,
        "model_revision": _MODEL_REVISION,
        "parent_python_revision": f"{sys.version_info.major}.{sys.version_info.minor}",
        "public_mobile_model_bundle_manifest_sha256": mobile.manifest_sha256,
        "public_mobile_source_inventory_sha256": mobile.manifest.source_model_inventory_sha256,
        "public_mobile_source_snapshot_sha256": mobile.manifest.source_model_snapshot_sha256,
        "public_recognizer_artifact_sha256s": base_artifacts,
        "seed_sha256": sha256_bytes(str(seed).encode("ascii")),
        "semantic_label_revision": _SEMANTIC_LABEL_REVISION,
        "source_crop_base_module_sha256": sha256_file(source_crop_base_path),
        "stage": "overfit",
        "trainer_revision": _TRAINER_REVISION,
        "training_module_sha256": sha256_file(module_path),
        "train_manifest_sha256": train_manifest_sha,
        "tuning_scope": tuning_scope,
        "whitespace_policy": _WHITESPACE_POLICY,
    }
    intent_sha = sha256_bytes(canonical_json_bytes(intent))
    existing = find_verified_public_finetune_release_by_intent_v1(root, intent_sha)
    if existing is not None:
        return PublicFineTuneRunResultV1(release=existing, reused=True)
    try:
        import paddle

        import numpy as np
    except ImportError:
        raise SourceAttestedPublicRecognizerFineTuneError(
            "public_finetune_runtime_unavailable"
        ) from None
    os.environ.setdefault("OMP_NUM_THREADS", "6")
    os.environ.setdefault("FLAGS_json_format_model", "1")
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    paddle.set_device("cpu")
    stage = create_public_finetune_stage_v1(root)
    try:
        model = paddle.jit.load(str(_base_recognizer_prefix(mobile)))
        model.train()
        selected = select_public_recognizer_trainable_parameters_v1(
            paddle,
            layer=model,
            output_class_count=dictionary.output_class_count,
            tuning_scope=tuning_scope,
        )
        trainable_count = sum(int(parameter.numel()) for _name, parameter in selected)
        trainable_name_list_sha = sha256_bytes(
            canonical_json_bytes([name for name, _parameter in selected])
        )
        baseline_cer, baseline_exact, baseline_by_profile = _evaluate(
            paddle,
            model=model,
            document_preview_root=root,
            crops=corpus.development_crops,
            labels_by_sample_id=corpus.semantic_labels_by_sample_id,
            profile_by_sample_id=corpus.profile_by_sample_id,
            character_to_index=dictionary.character_to_index,
            output_class_count=dictionary.output_class_count,
            seed=seed,
            batch_size=batch_size,
        )
        optimizer = paddle.optimizer.AdamW(
            learning_rate=learning_rate,
            parameters=[parameter for _name, parameter in selected],
            weight_decay=1.0e-5,
        )
        best_cer = math.inf
        best_exact = 0.0
        best_epoch = 0
        best_by_profile: dict[str, SemanticDevelopmentMetricV1] | None = None
        best_state: dict[str, Any] | None = None
        completed_epochs = 0
        stale_epochs = 0
        cache: PageLruCache[Any] = PageLruCache(16)
        for epoch in range(maximum_epochs):
            model.train()
            for batch in deterministic_width_batches(
                corpus.train_crops,
                epoch=epoch,
                seed=seed,
                batch_size=batch_size,
            ):
                images, labels, label_lengths = _prepare_public_paddle_batch(
                    paddle,
                    document_preview_root=root,
                    crops=batch,
                    labels_by_sample_id=corpus.semantic_labels_by_sample_id,
                    profile_by_sample_id=corpus.profile_by_sample_id,
                    character_to_index=dictionary.character_to_index,
                    page_cache=cache,
                )
                _train_batch(
                    paddle,
                    model=model,
                    optimizer=optimizer,
                    images=images,
                    labels=labels,
                    label_lengths=label_lengths,
                    output_class_count=dictionary.output_class_count,
                    selected_parameters=selected,
                )
            completed_epochs = epoch + 1
            cer, exact, by_profile = _evaluate(
                paddle,
                model=model,
                document_preview_root=root,
                crops=corpus.development_crops,
                labels_by_sample_id=corpus.semantic_labels_by_sample_id,
                profile_by_sample_id=corpus.profile_by_sample_id,
                character_to_index=dictionary.character_to_index,
                output_class_count=dictionary.output_class_count,
                seed=seed,
                batch_size=batch_size,
            )
            improves_baseline = _strict_metric_improvement(
                candidate_cer=cer,
                candidate_exact=exact,
                baseline_cer=baseline_cer,
                baseline_exact=baseline_exact,
            )
            improves_best = best_epoch == 0 or _strict_metric_improvement(
                candidate_cer=cer,
                candidate_exact=exact,
                baseline_cer=best_cer,
                baseline_exact=best_exact,
            )
            if improves_baseline and improves_best:
                best_cer = cer
                best_exact = exact
                best_by_profile = by_profile
                best_epoch = completed_epochs
                best_state = {name: value.clone() for name, value in model.state_dict().items()}
                stale_epochs = 0
            elif best_epoch > 0:
                stale_epochs += 1
                if stale_epochs >= 2:
                    break
        if best_epoch < 1 or best_state is None or best_by_profile is None:
            raise SourceAttestedPublicRecognizerFineTuneError("fine_tune_no_improvement")
        model.set_state_dict(best_state)
        artifacts = export_public_finetuned_recognizer_v1(
            paddle,
            model=model,
            text_rec_directory=stage / "text-rec",
            dictionary=dictionary,
        )
        receipt_payload: dict[str, object] = {
            **intent,
            "artifact_kind": "source_attested_public_recognizer_finetune_receipt",
            "base_character_count": len(dictionary.characters),
            "completed_epochs": completed_epochs,
            "development_crop_manifest": corpus.development_manifest.model_dump(
                mode="json", warnings=False
            ),
            "best_epoch": best_epoch,
            "baseline_development_by_profile": {
                profile: metric.model_dump(mode="json", warnings=False)
                for profile, metric in baseline_by_profile.items()
            },
            "baseline_development_semantic_cer": round(float(baseline_cer), 10),
            "baseline_development_semantic_crop_exact_match": round(float(baseline_exact), 10),
            "development_semantic_cer": round(float(best_cer), 10),
            "development_semantic_crop_exact_match": round(float(best_exact), 10),
            "best_development_by_profile": {
                profile: metric.model_dump(mode="json", warnings=False)
                for profile, metric in best_by_profile.items()
            },
            "direct_runtime_package_inventory_sha256": package_inventory_sha,
            "early_stopped": completed_epochs < maximum_epochs,
            "evaluation_contract": "semantic_crop_cer_and_exact_without_critical_fields_v1",
            "framework_license": "Apache-2.0",
            "initialization": "pinned_public_inference_graph_warm_start",
            "improvement_contract": "overall-cer-then-crop-exact-strict-v1",
            "model_artifact_sha256s": artifacts,
            "output_class_count": dictionary.output_class_count,
            "public_mobile_source_inventory_sha256": (
                mobile.manifest.source_model_inventory_sha256
            ),
            "public_mobile_source_snapshot_sha256": (mobile.manifest.source_model_snapshot_sha256),
            "static_roundtrip_verified": True,
            "trainable_parameter_count": trainable_count,
            "trainable_parameter_name_list_sha256": trainable_name_list_sha,
            "trainable_tensor_count": len(selected),
            "training_intent_sha256": intent_sha,
            "train_crop_manifest": corpus.train_manifest.model_dump(mode="json", warnings=False),
        }
        release = publish_public_finetuned_recognizer_release_v1(
            root,
            staged_release=stage,
            receipt_payload=receipt_payload,
        )
        return PublicFineTuneRunResultV1(release=release, reused=False)
    except SourceAttestedPublicRecognizerFineTuneError:
        raise
    except Exception:
        raise SourceAttestedPublicRecognizerFineTuneError(
            "public_finetune_execution_failed"
        ) from None
    finally:
        if stage.exists() and stage.name.startswith(".staging-"):
            try:
                stage.resolve(strict=True).relative_to((root / _COLLECTION).resolve(strict=True))
                shutil.rmtree(stage)
            except (OSError, ValueError):
                pass


def train_source_attested_public_recognizer_leakage_safe_v2(
    document_preview_root: Path,
    *,
    runtime_bundle: VerifiedRasterTextDirectPaddleBundleV1,
    mobile_bundle: VerifiedRasterTextDirectMobilePaddleBundleV1,
    package_inventory_path: Path,
    maximum_epochs: int = _LEAKAGE_SAFE_MAXIMUM_EPOCHS,
    batch_size: int = 32,
    seed: int = _DEFAULT_SEED,
) -> PublicFineTuneRunResultV1:
    """Warm-start only the CTC head on the fixed selection-V2-isolated corpus."""

    if maximum_epochs != _LEAKAGE_SAFE_MAXIMUM_EPOCHS or not 1 <= batch_size <= 64:
        raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_configuration_invalid")
    root = _root(document_preview_root)
    runtime, mobile = _verified_training_bundles(runtime_bundle, mobile_bundle)
    try:
        package_inventory_sha = verify_paddle_training_package_inventory(package_inventory_path)
    except Exception:
        raise SourceAttestedPublicRecognizerFineTuneError(
            "public_finetune_package_inventory_invalid"
        ) from None
    if package_inventory_sha != runtime.file_sha256(
        runtime.manifest.sidecar_package_inventory_relative_path
    ):
        raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_package_inventory_drift")
    dictionary = load_public_mobile_recognizer_dictionary_v1(mobile)
    base_artifacts = _public_recognizer_artifacts(mobile)
    corpus = build_verified_public_finetune_leakage_safe_corpus_v2(
        root, dictionary=dictionary, seed=seed
    )
    module_path = Path(__file__).resolve(strict=True)
    corpus_module_value = source_attested_public_raster_text_corpus.__file__
    source_crop_base_value = source_crop_base.__file__
    if not isinstance(corpus_module_value, str) or not isinstance(source_crop_base_value, str):
        raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_source_compiler_invalid")
    try:
        corpus_module_path = Path(corpus_module_value).resolve(strict=True)
        source_crop_base_path = Path(source_crop_base_value).resolve(strict=True)
    except OSError:
        raise SourceAttestedPublicRecognizerFineTuneError(
            "public_finetune_source_compiler_invalid"
        ) from None
    train_manifest_sha = _manifest_sha256(corpus.train_manifest)
    development_manifest_sha = _manifest_sha256(corpus.development_manifest)
    intent = {
        "batch_size": batch_size,
        "base_character_count": len(dictionary.characters),
        "base_dictionary_sha256": dictionary.dictionary_sha256,
        "balanced_corpus_module_sha256": sha256_file(corpus_module_path),
        "development_manifest_sha256": development_manifest_sha,
        "direct_runtime_bundle_manifest_sha256": runtime.manifest_sha256,
        "direct_runtime_package_inventory_sha256": package_inventory_sha,
        "learning_rate": _HEAD_LEARNING_RATE,
        "ctc_blank_index": 0,
        "dictionary_first_class_index": 1,
        "form_marker_policy": _FORM_MARKER_POLICY,
        "head_tensor_contract_sha256": sha256_bytes(canonical_json_bytes(_EXPECTED_HEAD_SHAPES)),
        "implicit_space_class_index": dictionary.implicit_space_class_index,
        "maximum_epochs": maximum_epochs,
        "model_revision": _MODEL_REVISION_V2,
        "overlap_verifier_revision": _OVERLAP_VERIFIER_REVISION_V2,
        "parent_python_revision": f"{sys.version_info.major}.{sys.version_info.minor}",
        "parent_selection_receipt_sha256": (corpus.train_manifest.parent_selection_receipt_sha256),
        "public_mobile_model_bundle_manifest_sha256": mobile.manifest_sha256,
        "public_mobile_source_inventory_sha256": (mobile.manifest.source_model_inventory_sha256),
        "public_mobile_source_snapshot_sha256": (mobile.manifest.source_model_snapshot_sha256),
        "public_recognizer_artifact_sha256s": base_artifacts,
        "seed_sha256": sha256_bytes(str(seed).encode("ascii")),
        "selection_receipt_relative_path": (corpus.train_manifest.selection_receipt_relative_path),
        "selection_receipt_sha256": corpus.train_manifest.selection_receipt_sha256,
        "semantic_label_revision": _SEMANTIC_LABEL_REVISION,
        "source_crop_base_module_sha256": sha256_file(source_crop_base_path),
        "stage": "leakage-safe-v2",
        "train_holdout_overlap_count": 0,
        "trainer_revision": _TRAINER_REVISION_V2,
        "training_module_sha256": sha256_file(module_path),
        "train_manifest_sha256": train_manifest_sha,
        "tuning_scope": "head",
        "whitespace_policy": _WHITESPACE_POLICY,
    }
    intent_sha = sha256_bytes(canonical_json_bytes(intent))
    existing = find_verified_public_finetune_release_by_intent_v1(root, intent_sha)
    if existing is not None:
        if not isinstance(existing.receipt, PublicFineTuneReceiptV2):
            raise SourceAttestedPublicRecognizerFineTuneError("public_finetune_release_invalid")
        return PublicFineTuneRunResultV1(release=existing, reused=True)
    try:
        import paddle

        import numpy as np
    except ImportError:
        raise SourceAttestedPublicRecognizerFineTuneError(
            "public_finetune_runtime_unavailable"
        ) from None
    os.environ.setdefault("OMP_NUM_THREADS", "6")
    os.environ.setdefault("FLAGS_json_format_model", "1")
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)
    paddle.set_device("cpu")
    stage = create_public_finetune_stage_v1(root)
    try:
        model = paddle.jit.load(str(_base_recognizer_prefix(mobile)))
        model.train()
        selected = select_public_recognizer_trainable_parameters_v1(
            paddle,
            layer=model,
            output_class_count=dictionary.output_class_count,
            tuning_scope="head",
        )
        trainable_count = sum(int(parameter.numel()) for _name, parameter in selected)
        trainable_name_list_sha = sha256_bytes(
            canonical_json_bytes([name for name, _parameter in selected])
        )
        baseline_cer, baseline_exact, baseline_by_profile = _evaluate(
            paddle,
            model=model,
            document_preview_root=root,
            crops=corpus.development_crops,
            labels_by_sample_id=corpus.semantic_labels_by_sample_id,
            profile_by_sample_id=corpus.profile_by_sample_id,
            character_to_index=dictionary.character_to_index,
            output_class_count=dictionary.output_class_count,
            seed=seed,
            batch_size=batch_size,
        )
        optimizer = paddle.optimizer.AdamW(
            learning_rate=_HEAD_LEARNING_RATE,
            parameters=[parameter for _name, parameter in selected],
            weight_decay=1.0e-5,
        )
        best_cer = math.inf
        best_exact = 0.0
        best_epoch = 0
        best_by_profile: dict[str, SemanticDevelopmentMetricV1] | None = None
        best_state: dict[str, Any] | None = None
        completed_epochs = 0
        stale_epochs = 0
        cache: PageLruCache[Any] = PageLruCache(16)
        for epoch in range(maximum_epochs):
            model.train()
            for batch in deterministic_width_batches(
                corpus.train_crops,
                epoch=epoch,
                seed=seed,
                batch_size=batch_size,
            ):
                images, labels, label_lengths = _prepare_public_paddle_batch(
                    paddle,
                    document_preview_root=root,
                    crops=batch,
                    labels_by_sample_id=corpus.semantic_labels_by_sample_id,
                    profile_by_sample_id=corpus.profile_by_sample_id,
                    character_to_index=dictionary.character_to_index,
                    page_cache=cache,
                )
                _train_batch(
                    paddle,
                    model=model,
                    optimizer=optimizer,
                    images=images,
                    labels=labels,
                    label_lengths=label_lengths,
                    output_class_count=dictionary.output_class_count,
                    selected_parameters=selected,
                )
            completed_epochs = epoch + 1
            cer, exact, by_profile = _evaluate(
                paddle,
                model=model,
                document_preview_root=root,
                crops=corpus.development_crops,
                labels_by_sample_id=corpus.semantic_labels_by_sample_id,
                profile_by_sample_id=corpus.profile_by_sample_id,
                character_to_index=dictionary.character_to_index,
                output_class_count=dictionary.output_class_count,
                seed=seed,
                batch_size=batch_size,
            )
            improves_baseline = _strict_metric_improvement(
                candidate_cer=cer,
                candidate_exact=exact,
                baseline_cer=baseline_cer,
                baseline_exact=baseline_exact,
            )
            improves_best = best_epoch == 0 or _strict_metric_improvement(
                candidate_cer=cer,
                candidate_exact=exact,
                baseline_cer=best_cer,
                baseline_exact=best_exact,
            )
            if improves_baseline and improves_best:
                best_cer = cer
                best_exact = exact
                best_by_profile = by_profile
                best_epoch = completed_epochs
                best_state = {name: value.clone() for name, value in model.state_dict().items()}
                stale_epochs = 0
            elif best_epoch > 0:
                stale_epochs += 1
                if stale_epochs >= 2:
                    break
        if best_epoch < 1 or best_state is None or best_by_profile is None:
            raise SourceAttestedPublicRecognizerFineTuneError("fine_tune_no_improvement")
        model.set_state_dict(best_state)
        artifacts = export_public_finetuned_recognizer_v1(
            paddle,
            model=model,
            text_rec_directory=stage / "text-rec",
            dictionary=dictionary,
            output_model_name=_OUTPUT_MODEL_NAME_V2,
        )
        receipt_payload: dict[str, object] = {
            **intent,
            "artifact_kind": "source_attested_public_recognizer_finetune_receipt",
            "base_character_count": len(dictionary.characters),
            "completed_epochs": completed_epochs,
            "development_crop_manifest": corpus.development_manifest.model_dump(
                mode="json", warnings=False
            ),
            "best_epoch": best_epoch,
            "baseline_development_by_profile": {
                profile: metric.model_dump(mode="json", warnings=False)
                for profile, metric in baseline_by_profile.items()
            },
            "baseline_development_semantic_cer": round(float(baseline_cer), 10),
            "baseline_development_semantic_crop_exact_match": round(float(baseline_exact), 10),
            "development_semantic_cer": round(float(best_cer), 10),
            "development_semantic_crop_exact_match": round(float(best_exact), 10),
            "best_development_by_profile": {
                profile: metric.model_dump(mode="json", warnings=False)
                for profile, metric in best_by_profile.items()
            },
            "direct_runtime_package_inventory_sha256": package_inventory_sha,
            "early_stopped": completed_epochs < maximum_epochs,
            "evaluation_contract": ("semantic_crop_cer_and_exact_without_critical_fields_v1"),
            "framework_license": "Apache-2.0",
            "initialization": "pinned_public_inference_graph_warm_start",
            "improvement_contract": "overall-cer-then-crop-exact-strict-v1",
            "model_artifact_sha256s": artifacts,
            "output_class_count": dictionary.output_class_count,
            "static_roundtrip_verified": True,
            "trainable_parameter_count": trainable_count,
            "trainable_parameter_name_list_sha256": trainable_name_list_sha,
            "trainable_tensor_count": len(selected),
            "training_intent_sha256": intent_sha,
            "train_crop_manifest": corpus.train_manifest.model_dump(mode="json", warnings=False),
        }
        release = publish_public_finetuned_recognizer_release_v2(
            root,
            staged_release=stage,
            receipt_payload=receipt_payload,
        )
        return PublicFineTuneRunResultV1(release=release, reused=False)
    except SourceAttestedPublicRecognizerFineTuneError:
        raise
    except Exception:
        raise SourceAttestedPublicRecognizerFineTuneError(
            "public_finetune_execution_failed"
        ) from None
    finally:
        if stage.exists() and stage.name.startswith(".staging-"):
            try:
                stage.resolve(strict=True).relative_to((root / _COLLECTION).resolve(strict=True))
                shutil.rmtree(stage)
            except (OSError, ValueError):
                pass


__all__ = [
    "BaseRecognizerDictionaryV1",
    "PublicFineTuneCropManifestV1",
    "PublicFineTuneCropManifestV2",
    "PublicFineTuneReceiptV1",
    "PublicFineTuneReceiptV2",
    "PublicFineTuneReleaseManifestV1",
    "PublicFineTuneRunResultV1",
    "SourceAttestedPublicRecognizerFineTuneError",
    "VerifiedPublicFineTuneCorpusV1",
    "VerifiedPublicFineTuneCorpusV2",
    "VerifiedPublicFineTunedRecognizerReleaseV1",
    "build_verified_public_finetune_overfit_corpus_v1",
    "build_verified_public_finetune_leakage_safe_corpus_v2",
    "create_public_finetune_stage_v1",
    "export_public_finetuned_recognizer_v1",
    "find_verified_public_finetune_release_by_intent_v1",
    "load_public_mobile_recognizer_dictionary_v1",
    "load_verified_public_finetuned_recognizer_release_v1",
    "public_finetune_receipt_sha256",
    "publish_public_finetuned_recognizer_release_v1",
    "publish_public_finetuned_recognizer_release_v2",
    "select_public_recognizer_trainable_parameters_v1",
    "semantic_ocr_crop_label_v1",
    "train_source_attested_public_recognizer_overfit_v1",
    "train_source_attested_public_recognizer_leakage_safe_v2",
]
