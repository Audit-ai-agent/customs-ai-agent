"""Leakage-safe Voyage replay for the attested public OCR recognizer.

The replay is deliberately additive.  It warm-starts the one immutable V2
recognizer release, updates only its two CTC-head tensors and publishes to a
separate content-addressed collection.  Calibration, sealed and independent
audit Gold are not accepted by the corpus contract.

The corpus compiler lives in a separate module so source/Gold selection can be
audited independently.  This module consumes only its content-free manifests,
hash-bound crops and in-memory labels; label text is never serialized into a
receipt or release manifest.
"""

from __future__ import annotations

from collections import defaultdict, OrderedDict
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import random
import re
import shutil
import sys
from typing import Any, Final, Literal, Mapping, Sequence, cast
import uuid

from pydantic import BaseModel, ConfigDict, Field, model_validator

from customs_ai_agent.evaluation import (
    source_attested_public_raster_text_voyage_replay_corpus as voyage_corpus,
)
from customs_ai_agent.evaluation.raster_text_direct_paddle_bundle import (
    DirectPaddleBundleFileV1,
    RasterTextDirectPaddleBundleError,
    VerifiedRasterTextDirectPaddleBundleV1,
    _operational_path,
    _safe_path as _direct_safe_path,
    load_verified_raster_text_direct_paddle_bundle_v1,
)
from customs_ai_agent.evaluation.raster_text_mobile_paddle_bundle import (
    RasterTextMobilePaddleBundleError,
    VerifiedRasterTextDirectMobilePaddleBundleV1,
    load_verified_raster_text_direct_mobile_paddle_bundle_v1,
)
from customs_ai_agent.evaluation.source_attested_public_raster_text_recognizer_finetuning import (
    BaseRecognizerDictionaryV1,
    PublicFineTuneReceiptV2,
    SourceAttestedPublicRecognizerFineTuneError,
    _ctc_loss,
    _original_named_parameters,
    _prepare_public_paddle_batch,
    _probability_output,
    _train_batch,
    export_public_finetuned_recognizer_v1,
    load_public_mobile_recognizer_dictionary_v1,
    load_verified_public_finetuned_recognizer_release_v1,
    select_public_recognizer_trainable_parameters_v1,
)
from customs_ai_agent.evaluation.source_attested_raster_text_recognizer_training import (
    PageLruCache,
    TrainingCropV1,
    _ctc_decode_indices,
    _edit_distance,
    canonical_json_bytes,
    canonical_json_file_bytes,
    sha256_bytes,
    sha256_file,
    verify_paddle_training_package_inventory,
)
from customs_ai_agent.evaluation.source_attested_public_raster_text_voyage_replay_corpus import (
    VerifiedPublicRasterTextVoyageReplayCorpusV1,
    VoyageReplayCorpusManifestV1,
    VoyageReplayCropV1,
    build_verified_public_raster_text_voyage_replay_corpus_v1,
    prepare_voyage_replay_crop_image_v1,
)


_SHA256 = r"^[a-f0-9]{64}$"
_DOCUMENT_PREVIEW_ROOT_SUFFIX = ("_local", "data", "document-preview")
_COLLECTION = "raster-text-public-voyage-replay-recognizers-v1"
_RELEASES = f"{_COLLECTION}/releases"
_PARENT_RELEASE_SHA256: Final = "51f4324be43c1840a574cdb4b9febb58ec175df80de79cd95a0ef9ff33e0bea5"
_PARENT_RECEIPT_SHA256: Final = "b09e8a208407af78e6e1b3535a25abacce3a765f952d73dbfbc68193cdaa42a6"
_TRAINER_REVISION: Final = "public-paddle-jit-ctc-leakage-safe-voyage-replay-head-v3"
_MODEL_REVISION: Final = "en-ppocrv5-mobile-rec-source-attested-voyage-replay-head-v3"
_OUTPUT_MODEL_NAME: Final = "source_attested_en_PP-OCRv5_mobile_rec_voyage_replay_head_v3"
_OVERLAP_REVISION: Final = "selection-v2-voyage-replay-all-holdouts-audit-zero-v1"
_DEFAULT_SEED = 1_937_429
_MAXIMUM_EPOCHS: Final[Literal[2]] = 2
_BATCH_SIZE: Final[Literal[30]] = 30
_VOYAGE_PER_BATCH: Final[Literal[20]] = 20
_GENERAL_PER_BATCH: Final[Literal[10]] = 10
_TRAIN_VOYAGE_COUNT = 500
_TRAIN_GENERAL_COUNT = 250
_DEVELOPMENT_VOYAGE_COUNT = 200
_DEVELOPMENT_GENERAL_COUNT = 200
_BATCHES_PER_EPOCH: Final[Literal[25]] = 25
_LEARNING_RATE = 5.0e-5
_WEIGHT_DECAY = 1.0e-5
_GRADIENT_CLIP_NORM = 1.0
_EXPECTED_TRAINABLE_TENSORS: Final[Literal[2]] = 2
_EXPECTED_TRAINABLE_PARAMETERS: Final[Literal[52998]] = 52_998
_LOGICAL_HEAD_SHAPES: Final[Mapping[str, tuple[int, ...]]] = {
    "linear_8.b_0_0": (438,),
    "linear_8.w_0_0": (120, 438),
}
_PARENT_V2_JIT_RELOAD_HEAD_NAMES: Final[Mapping[str, str]] = {
    "linear_8.b_0_0": "linear_8.b_0_0_0",
    "linear_8.w_0_0": "linear_8.w_0_0_0",
}
_TRAINING_PROFILES = ("png-216", "jpeg-216-q75", "jpeg-150-q75")
_VOYAGE_LANES = tuple(item.recipe_id for item in voyage_corpus.VOYAGE_REPLAY_RECIPES_V1)
_MODEL_FILES = (
    "text-rec/config.json",
    "text-rec/inference.json",
    "text-rec/inference.pdiparams",
    "text-rec/inference.yml",
)
_RELEASE_FILES = (*_MODEL_FILES, "training-receipt.json")
_BASE_EXPORT_MODEL_NAME = "source_attested_en_PP-OCRv5_mobile_rec_head_v2"
_MAX_FILE_BYTES = 256 * 1024 * 1024
_FLOAT_EPSILON = 1.0e-10
_PREPARED_CACHE_REVISION: Final = "exact-width-normalized-lru-v1"
_PREPARED_CACHE_LIMIT_BYTES: Final[Literal[536870912]] = 536_870_912
_GENERAL_CROP_RECIPE_SHA256: Final = sha256_bytes(b"general-v2-pad08-v1")


class VoyageReplayError(RuntimeError):
    """Content-free stable failure at the Voyage replay boundary."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ReplayMetricV1(_WireModel):
    crop_count: int = Field(ge=1, le=1_000)
    reference_character_count: int = Field(ge=1)
    character_error_count: int = Field(ge=0)
    exact_match_count: int = Field(ge=0)
    character_error_rate: float = Field(ge=0, le=100)
    exact_match: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def _ratios_match_counts(self) -> "ReplayMetricV1":
        if (
            self.exact_match_count > self.crop_count
            or abs(
                self.character_error_rate
                - self.character_error_count / self.reference_character_count
            )
            > _FLOAT_EPSILON
            or abs(self.exact_match - self.exact_match_count / self.crop_count) > _FLOAT_EPSILON
        ):
            raise ValueError("Voyage replay metric counts are invalid")
        return self


class VoyageReplayDevelopmentEvidenceV1(_WireModel):
    voyage_overall: ReplayMetricV1
    voyage_by_lane: dict[str, ReplayMetricV1]
    general_overall: ReplayMetricV1
    general_by_profile: dict[str, ReplayMetricV1]
    general_by_field_key: dict[str, ReplayMetricV1]

    @model_validator(mode="after")
    def _coverage_is_exact(self) -> "VoyageReplayDevelopmentEvidenceV1":
        if (
            self.voyage_overall.crop_count != _DEVELOPMENT_VOYAGE_COUNT
            or set(self.voyage_by_lane) != set(_VOYAGE_LANES)
            or set(metric.crop_count for metric in self.voyage_by_lane.values()) != {25}
            or self.general_overall.crop_count != _DEVELOPMENT_GENERAL_COUNT
            or set(self.general_by_profile) != set(_TRAINING_PROFILES)
            or sum(metric.crop_count for metric in self.general_by_profile.values())
            != _DEVELOPMENT_GENERAL_COUNT
            or not self.general_by_field_key
            or sum(metric.crop_count for metric in self.general_by_field_key.values())
            != _DEVELOPMENT_GENERAL_COUNT
        ):
            raise ValueError("Voyage replay development evidence is incomplete")
        return self


class VoyageReplayReceiptV3(_WireModel):
    schema_version: Literal["3.0"] = "3.0"
    artifact_kind: Literal["source_attested_public_recognizer_voyage_replay_receipt"] = (
        "source_attested_public_recognizer_voyage_replay_receipt"
    )
    trainer_revision: Literal["public-paddle-jit-ctc-leakage-safe-voyage-replay-head-v3"] = (
        _TRAINER_REVISION
    )
    model_revision: Literal["en-ppocrv5-mobile-rec-source-attested-voyage-replay-head-v3"] = (
        _MODEL_REVISION
    )
    output_model_name: Literal["source_attested_en_PP-OCRv5_mobile_rec_voyage_replay_head_v3"] = (
        _OUTPUT_MODEL_NAME
    )
    stage: Literal["voyage-replay-v3"] = "voyage-replay-v3"
    tuning_scope: Literal["ctc_head_only"] = "ctc_head_only"
    training_intent_sha256: str = Field(pattern=_SHA256)
    parent_release_sha256: Literal[
        "51f4324be43c1840a574cdb4b9febb58ec175df80de79cd95a0ef9ff33e0bea5"
    ] = _PARENT_RELEASE_SHA256
    parent_receipt_sha256: Literal[
        "b09e8a208407af78e6e1b3535a25abacce3a765f952d73dbfbc68193cdaa42a6"
    ] = _PARENT_RECEIPT_SHA256
    parent_model_artifact_sha256s: dict[str, str]
    train_manifest_sha256: str = Field(pattern=_SHA256)
    development_manifest_sha256: str = Field(pattern=_SHA256)
    train_corpus_manifest: VoyageReplayCorpusManifestV1
    development_corpus_manifest: VoyageReplayCorpusManifestV1
    direct_runtime_bundle_manifest_sha256: str = Field(pattern=_SHA256)
    direct_runtime_package_inventory_sha256: str = Field(pattern=_SHA256)
    public_mobile_model_bundle_manifest_sha256: str = Field(pattern=_SHA256)
    base_dictionary_sha256: str = Field(pattern=_SHA256)
    output_class_count: Literal[438] = 438
    parent_python_revision: str = Field(pattern=r"^3\.[0-9]+$")
    training_module_sha256: str = Field(pattern=_SHA256)
    corpus_module_sha256: str = Field(pattern=_SHA256)
    source_crop_base_module_sha256: str = Field(pattern=_SHA256)
    seed_sha256: str = Field(pattern=_SHA256)
    batch_size: Literal[30] = _BATCH_SIZE
    voyage_per_batch: Literal[20] = _VOYAGE_PER_BATCH
    general_per_batch: Literal[10] = _GENERAL_PER_BATCH
    batches_per_epoch: Literal[25] = _BATCHES_PER_EPOCH
    maximum_epochs: Literal[2] = _MAXIMUM_EPOCHS
    completed_epochs: int = Field(ge=1, le=2)
    published_epoch: int = Field(ge=1, le=2)
    earliest_passing_checkpoint: Literal[True] = True
    learning_rate: float = Field(default=_LEARNING_RATE, ge=5.0e-5, le=5.0e-5)
    weight_decay: float = Field(default=_WEIGHT_DECAY, ge=1.0e-5, le=1.0e-5)
    gradient_clip_global_norm: float = Field(default=_GRADIENT_CLIP_NORM, ge=1.0, le=1.0)
    optimizer: Literal["AdamW"] = "AdamW"
    prepared_crop_cache_revision: Literal["exact-width-normalized-lru-v1"] = (
        _PREPARED_CACHE_REVISION
    )
    prepared_crop_cache_limit_bytes: Literal[536870912] = _PREPARED_CACHE_LIMIT_BYTES
    trainable_tensor_count: Literal[2] = _EXPECTED_TRAINABLE_TENSORS
    trainable_parameter_count: Literal[52998] = _EXPECTED_TRAINABLE_PARAMETERS
    trainable_parameter_name_list_sha256: str = Field(pattern=_SHA256)
    parent_baseline_development: VoyageReplayDevelopmentEvidenceV1
    published_checkpoint_development: VoyageReplayDevelopmentEvidenceV1
    development_gate: Literal["voyage-exact-all-eight-lanes-cer0-general-no-regression-v1"] = (
        "voyage-exact-all-eight-lanes-cer0-general-no-regression-v1"
    )
    framework_license: Literal["Apache-2.0"] = "Apache-2.0"
    initialization: Literal["verified_v2_release_warm_start"] = "verified_v2_release_warm_start"
    static_roundtrip_verified: Literal[True] = True
    model_artifact_sha256s: dict[str, str]
    receipt_sha256: str = Field(pattern=_SHA256)

    @model_validator(mode="after")
    def _receipt_is_self_bound_and_passes(self) -> "VoyageReplayReceiptV3":
        train = self.train_corpus_manifest
        development = self.development_corpus_manifest
        if (
            set(self.parent_model_artifact_sha256s) != set(_MODEL_FILES)
            or set(self.model_artifact_sha256s) != set(_MODEL_FILES)
            or train.split != "train"
            or development.split != "development"
            or _digest_model(train) != self.train_manifest_sha256
            or _digest_model(development) != self.development_manifest_sha256
            or train.source_receipt_sha256 != development.source_receipt_sha256
            or train.raster_receipt_sha256 != development.raster_receipt_sha256
            or train.selection_receipt_sha256 != development.selection_receipt_sha256
            or train.parent_selection_receipt_sha256 != development.parent_selection_receipt_sha256
            or train.independent_audit_recipe_sha256 != development.independent_audit_recipe_sha256
            or train.independent_audit_family_ids_sha256
            != development.independent_audit_family_ids_sha256
            or train.independent_audit_source_sha256s_sha256
            != development.independent_audit_source_sha256s_sha256
            or self.completed_epochs != self.published_epoch
            or not development_gate_passes(
                baseline=self.parent_baseline_development,
                candidate=self.published_checkpoint_development,
            )
            or self.receipt_sha256 != voyage_replay_receipt_sha256(self)
        ):
            raise ValueError("Voyage replay receipt is invalid")
        return self


class VoyageReplayReleaseManifestV1(_WireModel):
    schema_version: Literal["1.0"] = "1.0"
    artifact_kind: Literal["source_attested_voyage_replay_release"] = (
        "source_attested_voyage_replay_release"
    )
    receipt_sha256: str = Field(pattern=_SHA256)
    files: tuple[DirectPaddleBundleFileV1, ...] = Field(min_length=5, max_length=5)

    @model_validator(mode="after")
    def _tree_is_exact(self) -> "VoyageReplayReleaseManifestV1":
        if tuple(item.path for item in self.files) != _RELEASE_FILES:
            raise ValueError("Voyage replay release file set is invalid")
        return self


@dataclass(frozen=True, slots=True)
class VerifiedVoyageReplayReleaseV1:
    release_sha256: str
    release_path: Path
    manifest: VoyageReplayReleaseManifestV1
    receipt: VoyageReplayReceiptV3
    file_paths: Mapping[str, Path]
    file_sha256s: Mapping[str, str]


@dataclass(frozen=True, slots=True)
class VoyageReplayRunResultV1:
    release: VerifiedVoyageReplayReleaseV1
    reused: bool


@dataclass(slots=True)
class _MetricCounts:
    crops: int = 0
    characters: int = 0
    errors: int = 0
    exact: int = 0

    def metric(self) -> ReplayMetricV1:
        return ReplayMetricV1(
            crop_count=self.crops,
            reference_character_count=self.characters,
            character_error_count=self.errors,
            exact_match_count=self.exact,
            character_error_rate=self.errors / self.characters,
            exact_match=self.exact / self.crops,
        )


class PreparedCropLruCache:
    """Bounded cache of exact-width normalized arrays; no Gold or tensors."""

    def __init__(self, limit_bytes: int = _PREPARED_CACHE_LIMIT_BYTES) -> None:
        if limit_bytes != _PREPARED_CACHE_LIMIT_BYTES:
            raise VoyageReplayError("voyage_replay_cache_policy_invalid")
        self.limit_bytes = limit_bytes
        self.byte_size = 0
        self._values: OrderedDict[tuple[str, str, str], Any] = OrderedDict()

    def get(self, key: tuple[str, str, str]) -> Any | None:
        value = self._values.pop(key, None)
        if value is not None:
            self._values[key] = value
        return value

    def put(self, key: tuple[str, str, str], value: Any) -> None:
        try:
            byte_size = int(value.nbytes)
        except Exception:
            raise VoyageReplayError("voyage_replay_cache_value_invalid") from None
        if byte_size <= 0 or byte_size > self.limit_bytes:
            raise VoyageReplayError("voyage_replay_cache_value_invalid")
        previous = self._values.pop(key, None)
        if previous is not None:
            self.byte_size -= int(previous.nbytes)
        while self._values and self.byte_size + byte_size > self.limit_bytes:
            _removed_key, removed = self._values.popitem(last=False)
            self.byte_size -= int(removed.nbytes)
        self._values[key] = value
        self.byte_size += byte_size


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
        raise VoyageReplayError("voyage_replay_document_preview_root_invalid") from None


def _safe_path(root: Path, relative: str, *, directory: bool = False) -> Path:
    try:
        try:
            return _direct_safe_path(root, relative, directory=directory)
        except RasterTextDirectPaddleBundleError:
            operational_root = _operational_path(root, force_short=True)
            return _direct_safe_path(operational_root, relative, directory=directory)
    except (OSError, ValueError, RasterTextDirectPaddleBundleError):
        raise VoyageReplayError("voyage_replay_path_invalid") from None


def _canonical_model_bytes(model: BaseModel) -> bytes:
    return canonical_json_bytes(model.model_dump(mode="json", warnings=False))


def _digest_model(model: BaseModel) -> str:
    return sha256_bytes(_canonical_model_bytes(model))


def voyage_replay_receipt_sha256(receipt: VoyageReplayReceiptV3) -> str:
    payload = receipt.model_dump(mode="json", warnings=False)
    payload["receipt_sha256"] = ""
    return sha256_bytes(canonical_json_bytes(payload))


def _read_json(path: Path) -> tuple[dict[str, object], bytes]:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_FILE_BYTES:
            raise OSError
        raw = path.read_bytes()
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise ValueError
        return value, raw
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError):
        raise VoyageReplayError("voyage_replay_json_invalid") from None


def _write_new(path: Path, raw: bytes) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        raise VoyageReplayError("voyage_replay_publish_failed") from None


def _tree_entries(root: Path) -> tuple[set[str], set[str]]:
    try:
        files: set[str] = set()
        directories: set[str] = set()
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
        raise VoyageReplayError("voyage_replay_release_invalid") from None


def development_gate_passes(
    *,
    baseline: VoyageReplayDevelopmentEvidenceV1,
    candidate: VoyageReplayDevelopmentEvidenceV1,
) -> bool:
    """Return the fixed Voyage-perfect/general-no-regression checkpoint Gate."""

    voyage_metrics = [candidate.voyage_overall, *candidate.voyage_by_lane.values()]
    if any(
        metric.character_error_count != 0
        or metric.character_error_rate != 0.0
        or metric.exact_match_count != metric.crop_count
        or metric.exact_match != 1.0
        for metric in voyage_metrics
    ):
        return False
    if (
        set(candidate.voyage_by_lane) != set(_VOYAGE_LANES)
        or set(baseline.voyage_by_lane) != set(_VOYAGE_LANES)
        or set(candidate.general_by_profile) != set(_TRAINING_PROFILES)
        or set(baseline.general_by_profile) != set(_TRAINING_PROFILES)
        or set(candidate.general_by_field_key) != set(baseline.general_by_field_key)
    ):
        return False
    pairs = [
        (candidate.general_overall, baseline.general_overall),
        *(
            (candidate.general_by_profile[profile], baseline.general_by_profile[profile])
            for profile in _TRAINING_PROFILES
        ),
        *(
            (
                candidate.general_by_field_key[field_key],
                baseline.general_by_field_key[field_key],
            )
            for field_key in sorted(baseline.general_by_field_key)
        ),
    ]
    if any(
        current.character_error_rate > parent.character_error_rate + _FLOAT_EPSILON
        or current.exact_match + _FLOAT_EPSILON < parent.exact_match
        for current, parent in pairs
    ):
        return False
    return True


def _estimated_replay_width(crop: TrainingCropV1) -> int:
    x, y, width, height = crop.bbox
    if isinstance(crop, VoyageReplayCropV1):
        recipes = {item.recipe_id: item for item in voyage_corpus.VOYAGE_REPLAY_RECIPES_V1}
        recipe = recipes[crop.recipe_id]
        height_pixels = height * crop.page_height
        horizontal = min(
            recipe.horizontal_max_pixels,
            max(
                recipe.horizontal_min_pixels,
                int(math.ceil(recipe.horizontal_height_ratio * height_pixels)),
            ),
        )
        vertical = int(math.ceil(recipe.vertical_height_ratio * height_pixels))
        left = max(0, math.floor(x * crop.page_width) - horizontal)
        top = max(0, math.floor(y * crop.page_height) - vertical)
        right = min(
            crop.page_width,
            math.ceil((x + width) * crop.page_width) + horizontal,
        )
        bottom = min(
            crop.page_height,
            math.ceil((y + height) * crop.page_height) + vertical,
        )
    else:
        pad_x = width * 0.08
        pad_y = height * 0.15
        left = max(0, math.floor((x - pad_x) * crop.page_width))
        top = max(0, math.floor((y - pad_y) * crop.page_height))
        right = min(
            crop.page_width,
            math.ceil((x + width + pad_x) * crop.page_width),
        )
        bottom = min(
            crop.page_height,
            math.ceil((y + height + pad_y) * crop.page_height),
        )
    if right - left < 2 or bottom - top < 2:
        raise VoyageReplayError("voyage_replay_crop_geometry_invalid")
    return min(
        3_200,
        max(8, int(math.ceil(48.0 * (right - left) / (bottom - top)))),
    )


def deterministic_voyage_replay_batches(
    voyage_crops: Sequence[TrainingCropV1],
    general_crops: Sequence[TrainingCropV1],
    *,
    epoch: int,
    seed: int,
) -> tuple[tuple[TrainingCropV1, ...], ...]:
    """Build 25 deterministic batches, each with exactly 20 Voyage + 10 general."""

    if (
        len(voyage_crops) != _TRAIN_VOYAGE_COUNT
        or len(general_crops) != _TRAIN_GENERAL_COUNT
        or epoch not in {0, 1}
    ):
        raise VoyageReplayError("voyage_replay_batch_scope_invalid")
    if len({crop.sample_id for crop in (*voyage_crops, *general_crops)}) != (
        _TRAIN_VOYAGE_COUNT + _TRAIN_GENERAL_COUNT
    ):
        raise VoyageReplayError("voyage_replay_batch_sample_collision")

    def balanced(
        crops: Sequence[TrainingCropV1], lane: str
    ) -> tuple[tuple[TrainingCropV1, ...], ...]:
        lanes: list[list[TrainingCropV1]] = [[] for _index in range(_BATCHES_PER_EPOCH)]
        ranked = sorted(
            crops,
            key=lambda crop: (
                -_estimated_replay_width(crop),
                sha256_bytes(
                    f"{_TRAINER_REVISION}:{seed}:{epoch}:{lane}:{crop.sample_id}".encode("ascii")
                ),
                crop.sample_id,
            ),
        )
        for index, crop in enumerate(ranked):
            lanes[index % _BATCHES_PER_EPOCH].append(crop)
        return tuple(tuple(values) for values in lanes)

    def mixed_rank(crops: Sequence[TrainingCropV1], batch: int) -> list[TrainingCropV1]:
        return sorted(
            crops,
            key=lambda crop: (
                sha256_bytes(
                    (f"{_TRAINER_REVISION}:{seed}:{epoch}:mixed:{batch}:{crop.sample_id}").encode(
                        "ascii"
                    )
                ),
                crop.sample_id,
            ),
        )

    voyage = balanced(voyage_crops, "voyage")
    general = balanced(general_crops, "general")
    batches: list[tuple[TrainingCropV1, ...]] = []
    for index in range(_BATCHES_PER_EPOCH):
        if len(voyage[index]) != _VOYAGE_PER_BATCH or len(general[index]) != _GENERAL_PER_BATCH:
            raise VoyageReplayError("voyage_replay_batch_ratio_invalid")
        batches.append(tuple(mixed_rank((*voyage[index], *general[index]), index)))
    if (
        len(batches) != _BATCHES_PER_EPOCH
        or any(len(batch) != _BATCH_SIZE for batch in batches)
        or {crop.sample_id for batch in batches for crop in batch}
        != {crop.sample_id for crop in (*voyage_crops, *general_crops)}
    ):
        raise VoyageReplayError("voyage_replay_batch_coverage_invalid")
    return tuple(batches)


def build_verified_voyage_replay_corpus(
    document_preview_root: Path, *, seed: int = _DEFAULT_SEED
) -> tuple[VerifiedPublicRasterTextVoyageReplayCorpusV1, Path]:
    try:
        module_value = getattr(voyage_corpus, "__file__", None)
        if not isinstance(module_value, str):
            raise ValueError
        module_path = Path(module_value).resolve(strict=True)
        corpus = build_verified_public_raster_text_voyage_replay_corpus_v1(
            document_preview_root, seed=seed
        )
        _verify_corpus_contract(corpus)
        return corpus, module_path
    except VoyageReplayError:
        raise
    except Exception:
        raise VoyageReplayError("voyage_replay_corpus_invalid") from None


def _verify_corpus_contract(
    corpus: VerifiedPublicRasterTextVoyageReplayCorpusV1,
) -> None:
    train_voyage = tuple(corpus.train_voyage_crops)
    train_general = tuple(corpus.train_general_crops)
    development_voyage = tuple(corpus.development_voyage_crops)
    development_general = tuple(corpus.development_general_crops)
    all_crops = (*train_voyage, *train_general, *development_voyage, *development_general)
    sample_ids = {crop.sample_id for crop in all_crops}
    if (
        not isinstance(corpus.train_manifest, VoyageReplayCorpusManifestV1)
        or not isinstance(corpus.development_manifest, VoyageReplayCorpusManifestV1)
        or len(train_voyage) != _TRAIN_VOYAGE_COUNT
        or len(train_general) != _TRAIN_GENERAL_COUNT
        or len(development_voyage) != _DEVELOPMENT_VOYAGE_COUNT
        or len(development_general) != _DEVELOPMENT_GENERAL_COUNT
        or len(sample_ids)
        != sum(
            (
                _TRAIN_VOYAGE_COUNT,
                _TRAIN_GENERAL_COUNT,
                _DEVELOPMENT_VOYAGE_COUNT,
                _DEVELOPMENT_GENERAL_COUNT,
            )
        )
        or set(corpus.labels_by_sample_id) != sample_ids
        or set(corpus.profile_by_sample_id) != sample_ids
        or set(corpus.field_key_by_sample_id) != sample_ids
        or set(corpus.recipe_by_sample_id)
        != {crop.sample_id for crop in (*train_voyage, *development_voyage)}
        or any(crop.split != "train" for crop in (*train_voyage, *train_general))
        or any(crop.split != "development" for crop in (*development_voyage, *development_general))
    ):
        raise VoyageReplayError("voyage_replay_corpus_invalid")
    for crop in all_crops:
        label = corpus.labels_by_sample_id[crop.sample_id]
        profile = corpus.profile_by_sample_id[crop.sample_id]
        field_key = corpus.field_key_by_sample_id[crop.sample_id]
        if (
            not isinstance(label, str)
            or not label
            or sha256_bytes(label.encode("utf-8")) != crop.label_sha256
            or profile not in _TRAINING_PROFILES
            or not isinstance(field_key, str)
            or not field_key
        ):
            raise VoyageReplayError("voyage_replay_corpus_binding_invalid")
    if any(
        corpus.profile_by_sample_id[crop.sample_id] not in _TRAINING_PROFILES
        for crop in (*train_voyage, *train_general, *development_general)
    ):
        raise VoyageReplayError("voyage_replay_training_profile_invalid")
    lanes = _counts_by_value(crop.recipe_id for crop in development_voyage)
    if lanes != corpus.development_manifest.voyage_recipe_lane_counts:
        raise VoyageReplayError("voyage_replay_development_lane_invalid")


def _counts_by_value(values: Sequence[str] | Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return result


def _verified_parent_release(document_preview_root: Path) -> Any:
    try:
        parent = load_verified_public_finetuned_recognizer_release_v1(
            document_preview_root, _PARENT_RELEASE_SHA256
        )
        receipt = parent.receipt
        if (
            parent.release_sha256 != _PARENT_RELEASE_SHA256
            or not isinstance(receipt, PublicFineTuneReceiptV2)
            or receipt.receipt_sha256 != _PARENT_RECEIPT_SHA256
            or receipt.schema_version != "2.0"
            or receipt.trainer_revision != "public-paddle-jit-ctc-leakage-safe-head-v2"
            or receipt.model_revision != "en-ppocrv5-mobile-rec-source-attested-head-v2"
            or receipt.tuning_scope != "head"
            or receipt.train_holdout_overlap_count != 0
            or receipt.trainable_tensor_count != _EXPECTED_TRAINABLE_TENSORS
            or receipt.trainable_parameter_count != _EXPECTED_TRAINABLE_PARAMETERS
        ):
            raise ValueError
        return parent
    except (SourceAttestedPublicRecognizerFineTuneError, ValueError):
        raise VoyageReplayError("voyage_replay_parent_release_invalid") from None


def _parent_recognizer_prefix(parent: Any) -> Path:
    try:
        release_path = Path(parent.release_path).resolve(strict=True)
        operational = _operational_path(release_path, force_short=True)
        return operational / "text-rec" / "inference"
    except (OSError, TypeError, RasterTextDirectPaddleBundleError):
        raise VoyageReplayError("voyage_replay_parent_path_unsupported") from None


def _verified_bundles(
    runtime_bundle: VerifiedRasterTextDirectPaddleBundleV1,
    mobile_bundle: VerifiedRasterTextDirectMobilePaddleBundleV1,
) -> tuple[VerifiedRasterTextDirectPaddleBundleV1, VerifiedRasterTextDirectMobilePaddleBundleV1]:
    try:
        runtime = load_verified_raster_text_direct_paddle_bundle_v1(
            document_preview_root=runtime_bundle.document_preview_root,
            manifest_sha256=runtime_bundle.manifest_sha256,
        )
        mobile = load_verified_raster_text_direct_mobile_paddle_bundle_v1(
            document_preview_root=mobile_bundle.document_preview_root,
            manifest_sha256=mobile_bundle.manifest_sha256,
        )
        if runtime.document_preview_root != mobile.document_preview_root:
            raise ValueError
        return runtime, mobile
    except (
        OSError,
        ValueError,
        RasterTextDirectPaddleBundleError,
        RasterTextMobilePaddleBundleError,
    ):
        raise VoyageReplayError("voyage_replay_bundle_invalid") from None


def _prepare_replay_paddle_batch(
    paddle: Any,
    *,
    document_preview_root: Path,
    crops: Sequence[TrainingCropV1],
    corpus: VerifiedPublicRasterTextVoyageReplayCorpusV1,
    dictionary: BaseRecognizerDictionaryV1,
    page_cache: PageLruCache[Any],
    prepared_cache: PreparedCropLruCache,
) -> tuple[Any, Any, Any]:
    """Prepare recipe-aware Voyage crops and unchanged V2 replay anchors."""

    try:
        import numpy as np

        if not crops:
            raise ValueError
        voyage = [crop for crop in crops if crop.sample_id in corpus.recipe_by_sample_id]
        general = [crop for crop in crops if crop.sample_id not in corpus.recipe_by_sample_id]
        arrays: dict[str, Any] = {}
        for crop in voyage:
            if not isinstance(crop, VoyageReplayCropV1):
                raise ValueError
            recipe = corpus.recipe_by_sample_id.get(crop.sample_id)
            if recipe is None or recipe.recipe_id != crop.recipe_id:
                raise ValueError
            key = (crop.sample_id, crop.raster_sha256, crop.recipe_sha256)
            array = prepared_cache.get(key)
            if array is None:
                image = prepare_voyage_replay_crop_image_v1(document_preview_root, crop)
                array = np.asarray(image, dtype="float32") / np.float32(255.0)
                array = array[:, :, ::-1]
                array = (array - np.float32(0.5)) / np.float32(0.5)
                array = np.ascontiguousarray(array.transpose((2, 0, 1)))
                prepared_cache.put(key, array)
            arrays[crop.sample_id] = array
        general_misses: list[TrainingCropV1] = []
        for crop in general:
            key = (crop.sample_id, crop.raster_sha256, _GENERAL_CROP_RECIPE_SHA256)
            array = prepared_cache.get(key)
            if array is None:
                general_misses.append(crop)
            else:
                arrays[crop.sample_id] = array
        if general_misses:
            general_images, _general_labels, _general_lengths = _prepare_public_paddle_batch(
                paddle,
                document_preview_root=document_preview_root,
                crops=general_misses,
                labels_by_sample_id=corpus.labels_by_sample_id,
                profile_by_sample_id=corpus.profile_by_sample_id,
                character_to_index=dictionary.character_to_index,
                page_cache=page_cache,
            )
            general_arrays = general_images.numpy()
            if int(general_arrays.shape[0]) != len(general_misses):
                raise ValueError
            for index, crop in enumerate(general_misses):
                width = _estimated_replay_width(crop)
                array = np.ascontiguousarray(general_arrays[index, :, :, :width])
                key = (crop.sample_id, crop.raster_sha256, _GENERAL_CROP_RECIPE_SHA256)
                prepared_cache.put(key, array)
                arrays[crop.sample_id] = array
        if set(arrays) != {crop.sample_id for crop in crops}:
            raise ValueError
        maximum_width = max(int(arrays[crop.sample_id].shape[2]) for crop in crops)
        images = np.zeros((len(crops), 3, 48, maximum_width), dtype="float32")
        encoded: list[list[int]] = []
        for index, crop in enumerate(crops):
            array = arrays[crop.sample_id]
            images[index, :, :, : array.shape[2]] = array
            label = corpus.labels_by_sample_id.get(crop.sample_id)
            if label is None or sha256_bytes(label.encode("utf-8")) != crop.label_sha256:
                raise ValueError
            encoded.append([dictionary.character_to_index[character] for character in label])
        maximum_label = max(len(value) for value in encoded)
        labels = np.zeros((len(crops), maximum_label), dtype="int32")
        for index, value in enumerate(encoded):
            labels[index, : len(value)] = value
        lengths = np.asarray([len(value) for value in encoded], dtype="int64")
        return (
            paddle.to_tensor(images),
            paddle.to_tensor(labels),
            paddle.to_tensor(lengths),
        )
    except VoyageReplayError:
        raise
    except Exception:
        raise VoyageReplayError("voyage_replay_batch_material_invalid") from None


def _evaluate_group(
    paddle: Any,
    *,
    model: Any,
    root: Path,
    crops: Sequence[TrainingCropV1],
    dictionary: BaseRecognizerDictionaryV1,
    corpus: VerifiedPublicRasterTextVoyageReplayCorpusV1,
    seed: int,
    bucket_by_sample_id: Mapping[str, str],
    expected_buckets: set[str],
    prepared_cache: PreparedCropLruCache,
    secondary_bucket_by_sample_id: Mapping[str, str] | None = None,
    expected_secondary_buckets: set[str] | None = None,
) -> tuple[ReplayMetricV1, dict[str, ReplayMetricV1], dict[str, ReplayMetricV1]]:
    overall = _MetricCounts()
    buckets: dict[str, _MetricCounts] = defaultdict(_MetricCounts)
    secondary: dict[str, _MetricCounts] = defaultdict(_MetricCounts)
    cache: PageLruCache[Any] = PageLruCache(16)
    ordered = sorted(
        crops,
        key=lambda crop: (
            sha256_bytes(f"{_TRAINER_REVISION}:{seed}:evaluate:{crop.sample_id}".encode("ascii")),
            crop.sample_id,
        ),
    )
    model.eval()
    with paddle.no_grad():
        for start in range(0, len(ordered), _BATCH_SIZE):
            batch = ordered[start : start + _BATCH_SIZE]
            images, encoded, lengths = _prepare_replay_paddle_batch(
                paddle,
                document_preview_root=root,
                crops=batch,
                corpus=corpus,
                dictionary=dictionary,
                page_cache=cache,
                prepared_cache=prepared_cache,
            )
            output = _probability_output(
                paddle,
                model(images),
                batch_size=len(batch),
                output_class_count=dictionary.output_class_count,
            )
            predictions = paddle.argmax(output, axis=-1).numpy().tolist()
            expected_values = encoded.numpy().tolist()
            length_values = lengths.numpy().tolist()
            for crop, prediction, expected, length in zip(
                batch, predictions, expected_values, length_values, strict=True
            ):
                reference = [int(value) for value in expected[: int(length)]]
                candidate = _ctc_decode_indices([int(value) for value in prediction])
                errors = _edit_distance(reference, candidate)
                exact = int(reference == candidate)
                bucket = bucket_by_sample_id.get(crop.sample_id)
                if bucket not in expected_buckets:
                    raise VoyageReplayError("voyage_replay_evaluation_bucket_invalid")
                for accumulator in (overall, buckets[bucket]):
                    accumulator.crops += 1
                    accumulator.characters += len(reference)
                    accumulator.errors += errors
                    accumulator.exact += exact
                if secondary_bucket_by_sample_id is not None:
                    secondary_key = secondary_bucket_by_sample_id.get(crop.sample_id)
                    if (
                        expected_secondary_buckets is None
                        or secondary_key not in expected_secondary_buckets
                    ):
                        raise VoyageReplayError("voyage_replay_evaluation_bucket_invalid")
                    accumulator = secondary[cast(str, secondary_key)]
                    accumulator.crops += 1
                    accumulator.characters += len(reference)
                    accumulator.errors += errors
                    accumulator.exact += exact
    model.train()
    if set(buckets) != expected_buckets or overall.crops != len(crops):
        raise VoyageReplayError("voyage_replay_evaluation_coverage_invalid")
    if secondary_bucket_by_sample_id is not None and set(secondary) != (
        expected_secondary_buckets or set()
    ):
        raise VoyageReplayError("voyage_replay_evaluation_coverage_invalid")
    return (
        overall.metric(),
        {key: value.metric() for key, value in buckets.items()},
        {key: value.metric() for key, value in secondary.items()},
    )


def evaluate_voyage_replay_development(
    paddle: Any,
    *,
    model: Any,
    root: Path,
    corpus: VerifiedPublicRasterTextVoyageReplayCorpusV1,
    dictionary: BaseRecognizerDictionaryV1,
    seed: int,
    prepared_cache: PreparedCropLruCache,
) -> VoyageReplayDevelopmentEvidenceV1:
    voyage_bucket = {crop.sample_id: crop.recipe_id for crop in corpus.development_voyage_crops}
    voyage_overall, voyage_by_lane, _ = _evaluate_group(
        paddle,
        model=model,
        root=root,
        crops=corpus.development_voyage_crops,
        dictionary=dictionary,
        corpus=corpus,
        seed=seed,
        bucket_by_sample_id=voyage_bucket,
        expected_buckets=set(_VOYAGE_LANES),
        prepared_cache=prepared_cache,
    )
    general_bucket = {
        crop.sample_id: corpus.profile_by_sample_id[crop.sample_id]
        for crop in corpus.development_general_crops
    }
    general_field_bucket = {
        crop.sample_id: corpus.field_key_by_sample_id[crop.sample_id]
        for crop in corpus.development_general_crops
    }
    general_overall, general_by_profile, general_by_field_key = _evaluate_group(
        paddle,
        model=model,
        root=root,
        crops=corpus.development_general_crops,
        dictionary=dictionary,
        corpus=corpus,
        seed=seed + 1,
        bucket_by_sample_id=general_bucket,
        expected_buckets=set(_TRAINING_PROFILES),
        prepared_cache=prepared_cache,
        secondary_bucket_by_sample_id=general_field_bucket,
        expected_secondary_buckets=set(general_field_bucket.values()),
    )
    return VoyageReplayDevelopmentEvidenceV1(
        voyage_overall=voyage_overall,
        voyage_by_lane=voyage_by_lane,
        general_overall=general_overall,
        general_by_profile=general_by_profile,
        general_by_field_key=general_by_field_key,
    )


def _create_stage(root: Path) -> Path:
    try:
        collection = root / _COLLECTION
        collection.mkdir(parents=True, exist_ok=True)
        if collection.is_symlink() or not collection.is_dir():
            raise OSError
        stage = collection / f".staging-{uuid.uuid4().hex}"
        stage.mkdir(mode=0o700)
        return stage
    except OSError:
        raise VoyageReplayError("voyage_replay_publish_failed") from None


def _export_model(
    paddle: Any,
    *,
    model: Any,
    stage: Path,
    dictionary: BaseRecognizerDictionaryV1,
) -> dict[str, str]:
    directory = stage / "text-rec"
    try:
        export_public_finetuned_recognizer_v1(
            paddle,
            model=model,
            text_rec_directory=directory,
            dictionary=dictionary,
            output_model_name=_BASE_EXPORT_MODEL_NAME,
        )
        config_path = directory / "config.json"
        value, _raw = _read_json(config_path)
        global_value = value.get("Global")
        if not isinstance(global_value, dict) or global_value.get("model_name") != (
            _BASE_EXPORT_MODEL_NAME
        ):
            raise ValueError
        global_value["model_name"] = _OUTPUT_MODEL_NAME
        config_path.write_bytes(canonical_json_file_bytes(value))
        (directory / "inference.yml").write_bytes(
            f"Global:\n  model_name: {_OUTPUT_MODEL_NAME}\n".encode("utf-8")
        )
        return {relative: sha256_file(stage / relative) for relative in _MODEL_FILES}
    except VoyageReplayError:
        raise
    except (OSError, ValueError, SourceAttestedPublicRecognizerFineTuneError):
        raise VoyageReplayError("voyage_replay_export_invalid") from None


def publish_voyage_replay_release_v1(
    document_preview_root: Path,
    *,
    staged_release: Path,
    receipt_payload: Mapping[str, object],
) -> VerifiedVoyageReplayReleaseV1:
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
        # The self-hash must cover the exact fully materialized wire form.
        # The training caller intentionally supplies measured values only;
        # fixed schema fields are owned here.  If a Pydantic default is added
        # only during validation, hashing the pre-validation mapping would
        # make the receipt reject its own otherwise valid digest.
        candidate = {
            **receipt_payload,
            "schema_version": "3.0",
            "development_gate": (
                "voyage-exact-all-eight-lanes-cer0-general-no-regression-v1"
            ),
            "receipt_sha256": "",
        }
        receipt = VoyageReplayReceiptV3.model_validate(
            {**candidate, "receipt_sha256": sha256_bytes(canonical_json_bytes(candidate))}
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
        manifest = VoyageReplayReleaseManifestV1(
            receipt_sha256=receipt.receipt_sha256, files=records
        )
        manifest_raw = _canonical_model_bytes(manifest)
        release_sha = sha256_bytes(manifest_raw)
        _write_new(stage / "manifest.json", manifest_raw)
        releases = collection / "releases"
        releases.mkdir(parents=True, exist_ok=True)
        final = _operational_path(releases.resolve(strict=True), force_short=True) / release_sha
        if final.exists() or final.is_symlink():
            existing = load_verified_voyage_replay_release_v1(root, release_sha)
            shutil.rmtree(stage)
            return existing
        os.rename(stage, final)
        return load_verified_voyage_replay_release_v1(root, release_sha)
    except VoyageReplayError:
        raise
    except (OSError, ValueError):
        raise VoyageReplayError("voyage_replay_publish_failed") from None


def load_verified_voyage_replay_release_v1(
    document_preview_root: Path, release_sha256: str
) -> VerifiedVoyageReplayReleaseV1:
    root = _root(document_preview_root)
    if re.fullmatch(_SHA256, release_sha256) is None:
        raise VoyageReplayError("voyage_replay_release_invalid")
    release = _safe_path(root, f"{_RELEASES}/{release_sha256}", directory=True)
    try:
        manifest_value, manifest_raw = _read_json(_safe_path(release, "manifest.json"))
        manifest = VoyageReplayReleaseManifestV1.model_validate(manifest_value)
        if (
            manifest_raw != _canonical_model_bytes(manifest)
            or sha256_bytes(manifest_raw) != release_sha256
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
        receipt = VoyageReplayReceiptV3.model_validate(receipt_value)
        if (
            receipt_raw != _canonical_model_bytes(receipt)
            or receipt.receipt_sha256 != manifest.receipt_sha256
        ):
            raise ValueError
        return VerifiedVoyageReplayReleaseV1(
            release_sha256=release_sha256,
            release_path=release,
            manifest=manifest,
            receipt=receipt,
            file_paths=paths,
            file_sha256s=digests,
        )
    except VoyageReplayError:
        raise
    except (OSError, ValueError, KeyError):
        raise VoyageReplayError("voyage_replay_release_invalid") from None


def find_verified_voyage_replay_release_by_intent_v1(
    document_preview_root: Path, training_intent_sha256: str
) -> VerifiedVoyageReplayReleaseV1 | None:
    root = _root(document_preview_root)
    if re.fullmatch(_SHA256, training_intent_sha256) is None:
        raise VoyageReplayError("voyage_replay_intent_invalid")
    releases = root / _RELEASES
    if not releases.exists():
        return None
    if releases.is_symlink() or not releases.is_dir():
        raise VoyageReplayError("voyage_replay_release_invalid")
    found: list[VerifiedVoyageReplayReleaseV1] = []
    for candidate in sorted(releases.iterdir(), key=lambda item: item.name):
        if (
            candidate.is_symlink()
            or not candidate.is_dir()
            or re.fullmatch(_SHA256, candidate.name) is None
        ):
            raise VoyageReplayError("voyage_replay_release_invalid")
        release = load_verified_voyage_replay_release_v1(root, candidate.name)
        if release.receipt.training_intent_sha256 == training_intent_sha256:
            found.append(release)
    if len(found) > 1:
        raise VoyageReplayError("voyage_replay_release_ambiguous")
    return found[0] if found else None


def select_voyage_replay_trainable_parameters_v1(
    paddle: Any,
    *,
    layer: Any,
    output_class_count: int,
) -> tuple[tuple[str, Any], ...]:
    """Select the exact V2 CTC head after one deterministic JIT reload.

    The original V2 training graph exposes ``linear_8.*_0_0``. Paddle 3.3.1
    appends one final ``_0`` when that already fine-tuned static release is
    loaded again for Voyage replay. Only those two exact spellings are
    aliases; shape-only discovery, suffix stripping, and arbitrary tail
    selection are deliberately rejected.
    """

    parameters: dict[str, Any] = {}

    def freeze_all() -> None:
        for parameter in parameters.values():
            try:
                parameter.clear_gradient()
                parameter.stop_gradient = True
            except Exception:
                continue

    try:
        if output_class_count != 438:
            raise SourceAttestedPublicRecognizerFineTuneError(
                "public_finetune_head_topology_invalid"
            )
        parameters = _original_named_parameters(layer)
        expected_shapes = set(_LOGICAL_HEAD_SHAPES.values())
        shape_matches = {
            name
            for name, parameter in parameters.items()
            if tuple(int(value) for value in parameter.shape) in expected_shapes
        }
        logical_names = set(_LOGICAL_HEAD_SHAPES)
        reloaded_names = set(_PARENT_V2_JIT_RELOAD_HEAD_NAMES.values())
        if shape_matches == logical_names:
            selected = select_public_recognizer_trainable_parameters_v1(
                paddle,
                layer=layer,
                output_class_count=output_class_count,
                tuning_scope="head",
            )
            actual_names = {name: name for name in logical_names}
        elif shape_matches == reloaded_names:
            for parameter in parameters.values():
                parameter.stop_gradient = False
                parameter.clear_gradient()
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
            connected = {
                name for name, parameter in parameters.items() if parameter.grad is not None
            }
            if not reloaded_names <= connected:
                raise SourceAttestedPublicRecognizerFineTuneError(
                    "public_finetune_head_topology_invalid"
                )
            actual_names = dict(_PARENT_V2_JIT_RELOAD_HEAD_NAMES)
            chosen_actual = set(actual_names.values())
            for name, parameter in parameters.items():
                parameter.clear_gradient()
                parameter.stop_gradient = name not in chosen_actual
            selected = tuple(
                (logical_name, parameters[actual_names[logical_name]])
                for logical_name in sorted(logical_names)
            )
        else:
            raise SourceAttestedPublicRecognizerFineTuneError(
                "public_finetune_head_topology_invalid"
            )

        selected_by_name = dict(selected)
        chosen_actual = set(actual_names.values())
        if (
            tuple(name for name, _parameter in selected) != tuple(sorted(logical_names))
            or set(selected_by_name) != logical_names
            or len({id(parameter) for parameter in selected_by_name.values()})
            != _EXPECTED_TRAINABLE_TENSORS
            or {
                name: tuple(int(value) for value in parameter.shape) for name, parameter in selected
            }
            != _LOGICAL_HEAD_SHAPES
            or sum(int(parameter.numel()) for parameter in selected_by_name.values())
            != _EXPECTED_TRAINABLE_PARAMETERS
            or any(parameter.stop_gradient for parameter in selected_by_name.values())
            or any(
                bool(parameter.stop_gradient) != (name not in chosen_actual)
                for name, parameter in parameters.items()
            )
        ):
            raise SourceAttestedPublicRecognizerFineTuneError(
                "public_finetune_head_topology_invalid"
            )
        return selected
    except SourceAttestedPublicRecognizerFineTuneError:
        freeze_all()
        raise
    except Exception:
        freeze_all()
        raise SourceAttestedPublicRecognizerFineTuneError(
            "public_finetune_gradient_topology_invalid"
        ) from None


def train_source_attested_public_recognizer_voyage_replay_v1(
    document_preview_root: Path,
    *,
    runtime_bundle: VerifiedRasterTextDirectPaddleBundleV1,
    mobile_bundle: VerifiedRasterTextDirectMobilePaddleBundleV1,
    package_inventory_path: Path,
) -> VoyageReplayRunResultV1:
    """Run the fixed two-epoch replay and publish the earliest passing checkpoint."""

    root = _root(document_preview_root)
    runtime, mobile = _verified_bundles(runtime_bundle, mobile_bundle)
    parent = _verified_parent_release(root)
    try:
        package_inventory_sha = verify_paddle_training_package_inventory(package_inventory_path)
    except Exception:
        raise VoyageReplayError("voyage_replay_package_inventory_invalid") from None
    inventory_relative = runtime.manifest.sidecar_package_inventory_relative_path
    if package_inventory_sha != runtime.file_sha256(inventory_relative):
        raise VoyageReplayError("voyage_replay_package_inventory_drift")
    dictionary = load_public_mobile_recognizer_dictionary_v1(mobile)
    corpus, corpus_module_path = build_verified_voyage_replay_corpus(root)
    module_path = Path(__file__).resolve(strict=True)
    source_crop_module = sys.modules[TrainingCropV1.__module__]
    source_crop_value = getattr(source_crop_module, "__file__", None)
    if not isinstance(source_crop_value, str):
        raise VoyageReplayError("voyage_replay_source_module_invalid")
    source_crop_path = Path(source_crop_value).resolve(strict=True)
    train_manifest_sha = _digest_model(corpus.train_manifest)
    development_manifest_sha = _digest_model(corpus.development_manifest)
    intent: dict[str, object] = {
        "artifact_kind": "source_attested_public_recognizer_voyage_replay_receipt",
        "base_dictionary_sha256": dictionary.dictionary_sha256,
        "batch_size": _BATCH_SIZE,
        "batches_per_epoch": _BATCHES_PER_EPOCH,
        "corpus_module_sha256": sha256_file(corpus_module_path),
        "development_manifest_sha256": development_manifest_sha,
        "direct_runtime_bundle_manifest_sha256": runtime.manifest_sha256,
        "direct_runtime_package_inventory_sha256": package_inventory_sha,
        "general_per_batch": _GENERAL_PER_BATCH,
        "gradient_clip_global_norm": _GRADIENT_CLIP_NORM,
        "learning_rate": _LEARNING_RATE,
        "maximum_epochs": _MAXIMUM_EPOCHS,
        "model_revision": _MODEL_REVISION,
        "optimizer": "AdamW",
        "output_model_name": _OUTPUT_MODEL_NAME,
        "prepared_crop_cache_limit_bytes": _PREPARED_CACHE_LIMIT_BYTES,
        "prepared_crop_cache_revision": _PREPARED_CACHE_REVISION,
        "parent_python_revision": f"{sys.version_info.major}.{sys.version_info.minor}",
        "parent_receipt_sha256": _PARENT_RECEIPT_SHA256,
        "parent_release_sha256": _PARENT_RELEASE_SHA256,
        "public_mobile_model_bundle_manifest_sha256": mobile.manifest_sha256,
        "seed_sha256": sha256_bytes(str(_DEFAULT_SEED).encode("ascii")),
        "source_crop_base_module_sha256": sha256_file(source_crop_path),
        "stage": "voyage-replay-v3",
        "train_manifest_sha256": train_manifest_sha,
        "trainer_revision": _TRAINER_REVISION,
        "training_module_sha256": sha256_file(module_path),
        "tuning_scope": "ctc_head_only",
        "voyage_per_batch": _VOYAGE_PER_BATCH,
        "weight_decay": _WEIGHT_DECAY,
    }
    intent_sha = sha256_bytes(canonical_json_bytes(intent))
    existing = find_verified_voyage_replay_release_by_intent_v1(root, intent_sha)
    if existing is not None:
        return VoyageReplayRunResultV1(release=existing, reused=True)
    try:
        import numpy as np
        import paddle
    except ImportError:
        raise VoyageReplayError("voyage_replay_runtime_unavailable") from None
    os.environ.setdefault("OMP_NUM_THREADS", "6")
    os.environ.setdefault("FLAGS_json_format_model", "1")
    random.seed(_DEFAULT_SEED)
    np.random.seed(_DEFAULT_SEED)
    paddle.seed(_DEFAULT_SEED)
    paddle.set_device("cpu")
    stage_identity = _create_stage(root)
    stage = _operational_path(stage_identity.resolve(strict=True), force_short=True)
    try:
        parent_prefix = _parent_recognizer_prefix(parent)
        model = paddle.jit.load(str(parent_prefix))
        model.train()
        selected = select_voyage_replay_trainable_parameters_v1(
            paddle,
            layer=model,
            output_class_count=dictionary.output_class_count,
        )
        trainable_count = sum(int(parameter.numel()) for _name, parameter in selected)
        if len(selected) != _EXPECTED_TRAINABLE_TENSORS or trainable_count != (
            _EXPECTED_TRAINABLE_PARAMETERS
        ):
            raise VoyageReplayError("voyage_replay_head_contract_invalid")
        trainable_name_sha = sha256_bytes(
            canonical_json_bytes([name for name, _parameter in selected])
        )
        prepared_cache = PreparedCropLruCache()
        baseline = evaluate_voyage_replay_development(
            paddle,
            model=model,
            root=root,
            corpus=corpus,
            dictionary=dictionary,
            seed=_DEFAULT_SEED,
            prepared_cache=prepared_cache,
        )
        optimizer = paddle.optimizer.AdamW(
            learning_rate=_LEARNING_RATE,
            parameters=[parameter for _name, parameter in selected],
            weight_decay=_WEIGHT_DECAY,
            grad_clip=paddle.nn.ClipGradByGlobalNorm(clip_norm=_GRADIENT_CLIP_NORM),
        )
        published_epoch = 0
        published: VoyageReplayDevelopmentEvidenceV1 | None = None
        for epoch in range(_MAXIMUM_EPOCHS):
            model.train()
            cache: PageLruCache[Any] = PageLruCache(16)
            voyage_ids = {crop.sample_id for crop in corpus.train_voyage_crops}
            for batch in deterministic_voyage_replay_batches(
                corpus.train_voyage_crops,
                corpus.train_general_crops,
                epoch=epoch,
                seed=_DEFAULT_SEED,
            ):
                if sum(crop.sample_id in voyage_ids for crop in batch) != _VOYAGE_PER_BATCH:
                    raise VoyageReplayError("voyage_replay_batch_ratio_invalid")
                images, labels, label_lengths = _prepare_replay_paddle_batch(
                    paddle,
                    document_preview_root=root,
                    crops=batch,
                    corpus=corpus,
                    dictionary=dictionary,
                    page_cache=cache,
                    prepared_cache=prepared_cache,
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
            checkpoint = evaluate_voyage_replay_development(
                paddle,
                model=model,
                root=root,
                corpus=corpus,
                dictionary=dictionary,
                seed=_DEFAULT_SEED,
                prepared_cache=prepared_cache,
            )
            if development_gate_passes(baseline=baseline, candidate=checkpoint):
                published_epoch = epoch + 1
                published = checkpoint
                break
        if published_epoch == 0 or published is None:
            raise VoyageReplayError("voyage_replay_development_gate_failed")
        artifacts = _export_model(paddle, model=model, stage=stage, dictionary=dictionary)
        receipt_payload: dict[str, object] = {
            **intent,
            "base_dictionary_sha256": dictionary.dictionary_sha256,
            "completed_epochs": published_epoch,
            "development_corpus_manifest": corpus.development_manifest.model_dump(
                mode="json", warnings=False
            ),
            "earliest_passing_checkpoint": True,
            "framework_license": "Apache-2.0",
            "initialization": "verified_v2_release_warm_start",
            "model_artifact_sha256s": artifacts,
            "output_class_count": dictionary.output_class_count,
            "parent_baseline_development": baseline.model_dump(mode="json", warnings=False),
            "parent_model_artifact_sha256s": dict(parent.receipt.model_artifact_sha256s),
            "published_checkpoint_development": published.model_dump(mode="json", warnings=False),
            "published_epoch": published_epoch,
            "static_roundtrip_verified": True,
            "train_corpus_manifest": corpus.train_manifest.model_dump(mode="json", warnings=False),
            "trainable_parameter_count": trainable_count,
            "trainable_parameter_name_list_sha256": trainable_name_sha,
            "trainable_tensor_count": len(selected),
            "training_intent_sha256": intent_sha,
        }
        release = publish_voyage_replay_release_v1(
            root, staged_release=stage_identity, receipt_payload=receipt_payload
        )
        return VoyageReplayRunResultV1(release=release, reused=False)
    except VoyageReplayError:
        raise
    except SourceAttestedPublicRecognizerFineTuneError as error:
        raise VoyageReplayError(error.code.replace("public_finetune", "voyage_replay")) from None
    except Exception:
        raise VoyageReplayError("voyage_replay_execution_failed") from None
    finally:
        if stage_identity.exists() and stage_identity.name.startswith(".staging-"):
            try:
                stage_identity.resolve(strict=True).relative_to(
                    (root / _COLLECTION).resolve(strict=True)
                )
                shutil.rmtree(stage_identity)
            except (OSError, ValueError):
                pass


__all__ = [
    "ReplayMetricV1",
    "VerifiedVoyageReplayReleaseV1",
    "VoyageReplayCorpusManifestV1",
    "VoyageReplayDevelopmentEvidenceV1",
    "VoyageReplayError",
    "VoyageReplayReceiptV3",
    "VoyageReplayReleaseManifestV1",
    "VoyageReplayRunResultV1",
    "build_verified_voyage_replay_corpus",
    "deterministic_voyage_replay_batches",
    "development_gate_passes",
    "evaluate_voyage_replay_development",
    "find_verified_voyage_replay_release_by_intent_v1",
    "load_verified_voyage_replay_release_v1",
    "publish_voyage_replay_release_v1",
    "select_voyage_replay_trainable_parameters_v1",
    "train_source_attested_public_recognizer_voyage_replay_v1",
    "voyage_replay_receipt_sha256",
]
