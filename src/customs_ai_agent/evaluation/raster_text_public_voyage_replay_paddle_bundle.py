"""Immutable mobile-detector plus Voyage-replay recognizer bundle.

The two source releases are re-opened through their owning loaders before any
byte is copied.  The resulting serving tree is content addressed and embeds
canonical copies of both provenance chains.  This module never imports
Paddle, trains a model, reads evaluation Gold, or activates a runtime pointer.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
from typing import Any, Iterator, Literal, Mapping, Sequence
import uuid

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from customs_ai_agent.evaluation.local_process_lock import (
    LocalProcessLockBusy,
    LocalProcessLockInvalid,
    exclusive_local_process_lock,
)
from customs_ai_agent.evaluation.raster_text_direct_paddle_bundle import (
    RasterTextDirectPaddleBundleError,
    VerifiedRasterTextDirectPaddleBundleV1,
    _canonical_json,
    _local_path_identity,
    _operational_path,
    _relative_path,
    _root,
    _safe_path,
    _sha256_regular_file,
    _write_create_only,
    load_verified_raster_text_direct_paddle_bundle_v1,
)
from customs_ai_agent.evaluation.raster_text_mobile_paddle_bundle import (
    RasterTextMobilePaddleBundleError,
    RasterTextMobilePaddleBundleManifestV1,
    VerifiedRasterTextDirectMobilePaddleBundleV1,
    load_verified_raster_text_direct_mobile_paddle_bundle_v1,
)
from customs_ai_agent.evaluation.source_attested_public_raster_text_recognizer_voyage_replay import (
    VoyageReplayError,
    VoyageReplayReceiptV3,
    VoyageReplayReleaseManifestV1,
    VerifiedVoyageReplayReleaseV1,
    load_verified_voyage_replay_release_v1,
)
from customs_ai_agent.evaluation.source_attested_public_raster_text_voyage_replay_corpus import (
    VoyageReplayCorpusManifestV1,
)
from customs_ai_agent.evaluation.source_attested_raster_text_recognizer_training import (
    canonical_json_bytes,
    canonical_json_file_bytes,
)


_SHA256 = r"^[a-f0-9]{64}$"
_COLLECTION = "raster-text-public-voyage-replay-paddle-bundles-v1"
_RELEASES = f"{_COLLECTION}/releases"
_LOCK_FILE = ".raster-text-public-voyage-replay-paddle-bundle-v1.lock"
_MODEL_INVENTORY = "inventories/direct-paddle-public-voyage-replay-models.json"
_MOBILE_MANIFEST_PROVENANCE = "provenance/mobile-detector-bundle-manifest.json"
_MOBILE_INVENTORY_PROVENANCE = "provenance/mobile-detector-model-inventory.json"
_RECOGNIZER_MANIFEST_PROVENANCE = "provenance/public-voyage-replay-recognizer-release-manifest.json"
_RECOGNIZER_RECEIPT_PROVENANCE = "provenance/public-voyage-replay-recognizer-training-receipt.json"
_PROVENANCE_PATHS = (
    _MOBILE_MANIFEST_PROVENANCE,
    _MOBILE_INVENTORY_PROVENANCE,
    _RECOGNIZER_MANIFEST_PROVENANCE,
    _RECOGNIZER_RECEIPT_PROVENANCE,
)
_DETECTOR_NAME = "PP-OCRv5_mobile_det"
_RECOGNIZER_NAME = "source_attested_en_PP-OCRv5_mobile_rec_voyage_replay_head_v3"
_TRAINER_REVISION = "public-paddle-jit-ctc-leakage-safe-voyage-replay-head-v3"
_MODEL_REVISION = "en-ppocrv5-mobile-rec-source-attested-voyage-replay-head-v3"
_FILENAMES = ("config.json", "inference.json", "inference.pdiparams", "inference.yml")
_MODEL_PATHS = tuple(
    f"{directory}/{filename}" for directory in ("text-det", "text-rec") for filename in _FILENAMES
)
_DECLARED_FILES = {
    _MODEL_INVENTORY,
    *_PROVENANCE_PATHS,
    *(f"models/{relative}" for relative in _MODEL_PATHS),
}
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_FILE_BYTES = 1024 * 1024 * 1024
_COPY_CHUNK_BYTES = 1024 * 1024
_CHARACTER_COUNT = 436
_OUTPUT_CLASS_COUNT = 438


class RasterTextPublicVoyageReplayPaddleBundleError(RuntimeError):
    """Content-free failure at the Voyage-replay bundle boundary."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RasterTextPublicVoyageReplayPaddleBundleFileV1(_WireModel):
    path: str = Field(min_length=1, max_length=480)
    sha256: str = Field(pattern=_SHA256)
    byte_size: int = Field(ge=1, le=_MAX_FILE_BYTES)

    _valid_path = field_validator("path")(_relative_path)


class RasterTextPublicVoyageReplayPaddleBundleManifestV1(_WireModel):
    """Exhaustive provenance binding for the V20 combined model release."""

    schema_version: Literal["1.0"] = "1.0"
    bundle_id: Literal["raster_text_public_voyage_replay_paddle_combined_model_bundle_v1"] = (
        "raster_text_public_voyage_replay_paddle_combined_model_bundle_v1"
    )
    source_mobile_bundle_manifest_sha256: str = Field(pattern=_SHA256)
    source_detector_model_snapshot_sha256: str = Field(pattern=_SHA256)
    source_recognizer_release_sha256: str = Field(pattern=_SHA256)
    source_recognizer_receipt_sha256: str = Field(pattern=_SHA256)
    source_recognizer_training_intent_sha256: str = Field(pattern=_SHA256)
    source_recognizer_parent_release_sha256: Literal[
        "51f4324be43c1840a574cdb4b9febb58ec175df80de79cd95a0ef9ff33e0bea5"
    ]
    source_recognizer_parent_receipt_sha256: Literal[
        "b09e8a208407af78e6e1b3535a25abacce3a765f952d73dbfbc68193cdaa42a6"
    ]
    source_recognizer_parent_model_artifact_sha256s: dict[str, str]
    source_recognizer_train_manifest_sha256: str = Field(pattern=_SHA256)
    source_recognizer_development_manifest_sha256: str = Field(pattern=_SHA256)
    source_recognizer_train_corpus_manifest: VoyageReplayCorpusManifestV1
    source_recognizer_development_corpus_manifest: VoyageReplayCorpusManifestV1
    source_recognizer_selection_receipt_relative_path: str = Field(min_length=1, max_length=240)
    source_recognizer_selection_receipt_sha256: str = Field(pattern=_SHA256)
    source_recognizer_parent_selection_receipt_sha256: str = Field(pattern=_SHA256)
    source_recognizer_voyage_recipe_set_sha256: str = Field(pattern=_SHA256)
    source_recognizer_voyage_recipe_sha256s: dict[str, str]
    source_recognizer_train_voyage_recipe_lane_counts: dict[str, int]
    source_recognizer_development_voyage_recipe_lane_counts: dict[str, int]
    source_recognizer_independent_audit_recipe_sha256: str = Field(pattern=_SHA256)
    source_recognizer_independent_audit_family_ids_sha256: str = Field(pattern=_SHA256)
    source_recognizer_independent_audit_source_sha256s_sha256: str = Field(pattern=_SHA256)
    source_recognizer_trainer_revision: Literal[
        "public-paddle-jit-ctc-leakage-safe-voyage-replay-head-v3"
    ] = "public-paddle-jit-ctc-leakage-safe-voyage-replay-head-v3"
    source_recognizer_model_revision: Literal[
        "en-ppocrv5-mobile-rec-source-attested-voyage-replay-head-v3"
    ] = "en-ppocrv5-mobile-rec-source-attested-voyage-replay-head-v3"
    source_recognizer_training_module_sha256: str = Field(pattern=_SHA256)
    source_recognizer_corpus_module_sha256: str = Field(pattern=_SHA256)
    source_recognizer_source_crop_base_module_sha256: str = Field(pattern=_SHA256)
    source_recognizer_direct_runtime_bundle_manifest_sha256: str = Field(pattern=_SHA256)
    source_recognizer_direct_runtime_package_inventory_sha256: str = Field(pattern=_SHA256)
    source_recognizer_public_mobile_model_bundle_manifest_sha256: str = Field(pattern=_SHA256)
    source_recognizer_base_dictionary_sha256: str = Field(pattern=_SHA256)
    source_recognizer_model_snapshot_sha256: str = Field(pattern=_SHA256)
    source_recognizer_output_class_count: Literal[438] = 438
    text_detection_model_name: Literal["PP-OCRv5_mobile_det"] = "PP-OCRv5_mobile_det"
    text_recognition_model_name: Literal[
        "source_attested_en_PP-OCRv5_mobile_rec_voyage_replay_head_v3"
    ] = "source_attested_en_PP-OCRv5_mobile_rec_voyage_replay_head_v3"
    local_model_root_relative_path: Literal["models"] = "models"
    model_inventory_relative_path: Literal[
        "inventories/direct-paddle-public-voyage-replay-models.json"
    ] = "inventories/direct-paddle-public-voyage-replay-models.json"
    mobile_manifest_provenance_relative_path: Literal[
        "provenance/mobile-detector-bundle-manifest.json"
    ] = "provenance/mobile-detector-bundle-manifest.json"
    mobile_inventory_provenance_relative_path: Literal[
        "provenance/mobile-detector-model-inventory.json"
    ] = "provenance/mobile-detector-model-inventory.json"
    recognizer_manifest_provenance_relative_path: Literal[
        "provenance/public-voyage-replay-recognizer-release-manifest.json"
    ] = "provenance/public-voyage-replay-recognizer-release-manifest.json"
    recognizer_receipt_provenance_relative_path: Literal[
        "provenance/public-voyage-replay-recognizer-training-receipt.json"
    ] = "provenance/public-voyage-replay-recognizer-training-receipt.json"
    model_root_tree_sha256: str = Field(pattern=_SHA256)
    files: tuple[RasterTextPublicVoyageReplayPaddleBundleFileV1, ...] = Field(
        min_length=len(_DECLARED_FILES), max_length=len(_DECLARED_FILES)
    )

    @model_validator(mode="after")
    def _exact_shape(self) -> "RasterTextPublicVoyageReplayPaddleBundleManifestV1":
        train = self.source_recognizer_train_corpus_manifest
        development = self.source_recognizer_development_corpus_manifest
        paths = tuple(item.path for item in self.files)
        if (
            paths != tuple(sorted(paths))
            or len(paths) != len(set(paths))
            or set(paths) != _DECLARED_FILES
            or set(self.source_recognizer_parent_model_artifact_sha256s)
            != {f"text-rec/{name}" for name in _FILENAMES}
            or any(
                re.fullmatch(_SHA256, value) is None
                for value in self.source_recognizer_parent_model_artifact_sha256s.values()
            )
            or train.split != "train"
            or development.split != "development"
            or train.purpose != "optimizer"
            or development.purpose != "validation-only"
            or hashlib.sha256(
                canonical_json_bytes(train.model_dump(mode="json", warnings=False))
            ).hexdigest()
            != self.source_recognizer_train_manifest_sha256
            or hashlib.sha256(
                canonical_json_bytes(development.model_dump(mode="json", warnings=False))
            ).hexdigest()
            != self.source_recognizer_development_manifest_sha256
            or train.source_receipt_sha256 != development.source_receipt_sha256
            or train.raster_receipt_sha256 != development.raster_receipt_sha256
            or train.selection_receipt_relative_path
            != self.source_recognizer_selection_receipt_relative_path
            or development.selection_receipt_relative_path
            != self.source_recognizer_selection_receipt_relative_path
            or train.selection_receipt_sha256 != self.source_recognizer_selection_receipt_sha256
            or development.selection_receipt_sha256
            != self.source_recognizer_selection_receipt_sha256
            or train.parent_selection_receipt_sha256
            != self.source_recognizer_parent_selection_receipt_sha256
            or development.parent_selection_receipt_sha256
            != self.source_recognizer_parent_selection_receipt_sha256
            or self.source_recognizer_selection_receipt_relative_path
            != (
                "synthetic-raster-text-v1/fresh-scope-selections/"
                f"{self.source_recognizer_selection_receipt_sha256}.json"
            )
            or train.voyage_recipe_set_sha256 != self.source_recognizer_voyage_recipe_set_sha256
            or development.voyage_recipe_set_sha256
            != self.source_recognizer_voyage_recipe_set_sha256
            or train.voyage_recipe_sha256s != self.source_recognizer_voyage_recipe_sha256s
            or development.voyage_recipe_sha256s != self.source_recognizer_voyage_recipe_sha256s
            or train.voyage_recipe_lane_counts
            != self.source_recognizer_train_voyage_recipe_lane_counts
            or development.voyage_recipe_lane_counts
            != self.source_recognizer_development_voyage_recipe_lane_counts
            or train.independent_audit_recipe_sha256
            != self.source_recognizer_independent_audit_recipe_sha256
            or development.independent_audit_recipe_sha256
            != self.source_recognizer_independent_audit_recipe_sha256
            or train.independent_audit_family_ids_sha256
            != self.source_recognizer_independent_audit_family_ids_sha256
            or development.independent_audit_family_ids_sha256
            != self.source_recognizer_independent_audit_family_ids_sha256
            or train.independent_audit_source_sha256s_sha256
            != self.source_recognizer_independent_audit_source_sha256s_sha256
            or development.independent_audit_source_sha256s_sha256
            != self.source_recognizer_independent_audit_source_sha256s_sha256
            or train.holdout_overlap_count != 0
            or development.holdout_overlap_count != 0
            or train.independent_audit_family_overlap_count != 0
            or development.independent_audit_family_overlap_count != 0
            or train.independent_audit_source_overlap_count != 0
            or development.independent_audit_source_overlap_count != 0
        ):
            raise ValueError("public Voyage-replay Paddle bundle shape is invalid")
        return self


@dataclass(frozen=True, slots=True)
class VerifiedRasterTextPublicVoyageReplayPaddleBundleV1:
    document_preview_root: Path
    bundle_root: Path
    bundle_root_relative_path: str
    manifest: RasterTextPublicVoyageReplayPaddleBundleManifestV1
    manifest_sha256: str
    file_sha256_by_relative_path: Mapping[str, str]

    def bundle_relative_path(self, path: str) -> str:
        _relative_path(path)
        return f"{self.bundle_root_relative_path}/{path}"

    def file_sha256(self, path: str) -> str:
        try:
            return self.file_sha256_by_relative_path[path]
        except KeyError:
            raise RasterTextPublicVoyageReplayPaddleBundleError(
                "public_voyage_replay_paddle_bundle_manifest_invalid"
            ) from None


@dataclass(frozen=True, slots=True)
class PreparedRasterTextPublicVoyageReplayRuntimeProfileV20:
    raster_text_profile_revision: str
    runtime_configuration_relative_path: str
    runtime_configuration_sha256: str
    minimum_page_confidence: float


@dataclass(frozen=True, slots=True)
class _SourceFile:
    source: Path
    source_relative_path: str
    target_relative_path: str
    sha256: str
    byte_size: int


@dataclass(frozen=True, slots=True)
class _VerifiedSources:
    root: Path
    mobile: VerifiedRasterTextDirectMobilePaddleBundleV1
    recognizer: VerifiedVoyageReplayReleaseV1
    receipt: VoyageReplayReceiptV3
    detector_files: tuple[_SourceFile, ...]
    recognizer_files: tuple[_SourceFile, ...]
    detector_snapshot_sha256: str
    recognizer_snapshot_sha256: str
    characters: tuple[str, ...]


def _canonical(value: object) -> bytes:
    return _canonical_json(value)


def _verified_root(path: Path) -> Path:
    try:
        return _root(path)
    except RasterTextDirectPaddleBundleError:
        raise RasterTextPublicVoyageReplayPaddleBundleError(
            "public_voyage_replay_paddle_bundle_root_invalid"
        ) from None


def _safe(root: Path, relative: str, *, directory: bool = False) -> Path:
    try:
        try:
            return _safe_path(root, relative, directory=directory)
        except RasterTextDirectPaddleBundleError:
            operational_root = _operational_path(root, force_short=True)
            return _safe_path(operational_root, relative, directory=directory)
    except RasterTextDirectPaddleBundleError:
        raise RasterTextPublicVoyageReplayPaddleBundleError(
            "public_voyage_replay_paddle_bundle_manifest_invalid"
        ) from None


def _hash_file(path: Path, *, code: str) -> tuple[str, int]:
    try:
        return _sha256_regular_file(path, code=code)
    except RasterTextDirectPaddleBundleError:
        raise RasterTextPublicVoyageReplayPaddleBundleError(code) from None


def _record_snapshot(files: Sequence[_SourceFile]) -> str:
    return hashlib.sha256(
        _canonical(
            {
                "artifacts": [
                    {
                        "byte_size": item.byte_size,
                        "relative_path": item.source_relative_path,
                        "sha256": item.sha256,
                    }
                    for item in files
                ]
            }
        )
    ).hexdigest()


def _model_tree_sha256(
    records: Sequence[RasterTextPublicVoyageReplayPaddleBundleFileV1],
) -> str:
    return hashlib.sha256(
        _canonical(
            {
                "files": [
                    {
                        "byte_size": item.byte_size,
                        "path": item.path,
                        "sha256": item.sha256,
                    }
                    for item in records
                ]
            }
        )
    ).hexdigest()


def _read_canonical_json(path: Path, *, code: str) -> tuple[dict[str, Any], bytes]:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_JSON_BYTES:
            raise OSError
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict) or raw != _canonical(value):
            raise ValueError
        return value, raw
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise RasterTextPublicVoyageReplayPaddleBundleError(code) from None


def _read_pretty_canonical_json(path: Path, *, code: str) -> tuple[dict[str, Any], bytes]:
    try:
        if path.is_symlink() or not path.is_file() or path.stat().st_size > _MAX_JSON_BYTES:
            raise OSError
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict) or raw != canonical_json_file_bytes(value):
            raise ValueError
        return value, raw
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError):
        raise RasterTextPublicVoyageReplayPaddleBundleError(code) from None


def _config_characters(path: Path) -> tuple[str, ...]:
    value, _raw = _read_pretty_canonical_json(
        path, code="public_voyage_replay_paddle_bundle_source_invalid"
    )
    try:
        global_value = value["Global"]
        postprocess = value["PostProcess"]
        characters = postprocess["character_dict"]
        if (
            set(value) != {"Global", "PostProcess", "PreProcess"}
            or global_value != {"model_name": _RECOGNIZER_NAME}
            or not isinstance(postprocess, dict)
            or set(postprocess) != {"character_dict", "name"}
            or postprocess.get("name") != "CTCLabelDecode"
            or not isinstance(characters, list)
            or len(characters) != _CHARACTER_COUNT
            or any(not isinstance(item, str) or len(item) != 1 for item in characters)
            or len(characters) != len(set(characters))
            or " " in characters
            or not isinstance(value["PreProcess"], dict)
        ):
            raise ValueError
        return tuple(characters)
    except (KeyError, TypeError, ValueError):
        raise RasterTextPublicVoyageReplayPaddleBundleError(
            "public_voyage_replay_paddle_bundle_source_invalid"
        ) from None


def _source_file(source: Path, source_relative: str, target_relative: str) -> _SourceFile:
    digest, size = _hash_file(source, code="public_voyage_replay_paddle_bundle_source_drift")
    return _SourceFile(
        source=source,
        source_relative_path=source_relative,
        target_relative_path=target_relative,
        sha256=digest,
        byte_size=size,
    )


def _receipt_arguments(
    receipt: VoyageReplayReceiptV3,
    *,
    mobile_manifest_sha256: str,
    recognizer_release_sha256: str,
    detector_snapshot_sha256: str,
    recognizer_snapshot_sha256: str,
) -> dict[str, object]:
    train = receipt.train_corpus_manifest
    development = receipt.development_corpus_manifest
    return {
        "source_mobile_bundle_manifest_sha256": mobile_manifest_sha256,
        "source_detector_model_snapshot_sha256": detector_snapshot_sha256,
        "source_recognizer_release_sha256": recognizer_release_sha256,
        "source_recognizer_receipt_sha256": receipt.receipt_sha256,
        "source_recognizer_training_intent_sha256": receipt.training_intent_sha256,
        "source_recognizer_parent_release_sha256": receipt.parent_release_sha256,
        "source_recognizer_parent_receipt_sha256": receipt.parent_receipt_sha256,
        "source_recognizer_parent_model_artifact_sha256s": (receipt.parent_model_artifact_sha256s),
        "source_recognizer_train_manifest_sha256": receipt.train_manifest_sha256,
        "source_recognizer_development_manifest_sha256": (receipt.development_manifest_sha256),
        "source_recognizer_train_corpus_manifest": train,
        "source_recognizer_development_corpus_manifest": development,
        "source_recognizer_selection_receipt_relative_path": (
            train.selection_receipt_relative_path
        ),
        "source_recognizer_selection_receipt_sha256": train.selection_receipt_sha256,
        "source_recognizer_parent_selection_receipt_sha256": (
            train.parent_selection_receipt_sha256
        ),
        "source_recognizer_voyage_recipe_set_sha256": train.voyage_recipe_set_sha256,
        "source_recognizer_voyage_recipe_sha256s": train.voyage_recipe_sha256s,
        "source_recognizer_train_voyage_recipe_lane_counts": (train.voyage_recipe_lane_counts),
        "source_recognizer_development_voyage_recipe_lane_counts": (
            development.voyage_recipe_lane_counts
        ),
        "source_recognizer_independent_audit_recipe_sha256": (
            train.independent_audit_recipe_sha256
        ),
        "source_recognizer_independent_audit_family_ids_sha256": (
            train.independent_audit_family_ids_sha256
        ),
        "source_recognizer_independent_audit_source_sha256s_sha256": (
            train.independent_audit_source_sha256s_sha256
        ),
        "source_recognizer_trainer_revision": receipt.trainer_revision,
        "source_recognizer_model_revision": receipt.model_revision,
        "source_recognizer_training_module_sha256": receipt.training_module_sha256,
        "source_recognizer_corpus_module_sha256": receipt.corpus_module_sha256,
        "source_recognizer_source_crop_base_module_sha256": (
            receipt.source_crop_base_module_sha256
        ),
        "source_recognizer_direct_runtime_bundle_manifest_sha256": (
            receipt.direct_runtime_bundle_manifest_sha256
        ),
        "source_recognizer_direct_runtime_package_inventory_sha256": (
            receipt.direct_runtime_package_inventory_sha256
        ),
        "source_recognizer_public_mobile_model_bundle_manifest_sha256": (
            receipt.public_mobile_model_bundle_manifest_sha256
        ),
        "source_recognizer_base_dictionary_sha256": receipt.base_dictionary_sha256,
        "source_recognizer_model_snapshot_sha256": recognizer_snapshot_sha256,
        "source_recognizer_output_class_count": receipt.output_class_count,
    }


def _receipt_manifest_arguments(sources: _VerifiedSources) -> dict[str, object]:
    return _receipt_arguments(
        sources.receipt,
        mobile_manifest_sha256=sources.mobile.manifest_sha256,
        recognizer_release_sha256=sources.recognizer.release_sha256,
        detector_snapshot_sha256=sources.detector_snapshot_sha256,
        recognizer_snapshot_sha256=sources.recognizer_snapshot_sha256,
    )


def _resolve_sources(
    mobile_bundle: VerifiedRasterTextDirectMobilePaddleBundleV1,
    recognizer_release: VerifiedVoyageReplayReleaseV1,
) -> _VerifiedSources:
    try:
        root = _verified_root(mobile_bundle.document_preview_root)
        recognizer_root = _verified_root(recognizer_release.release_path.parents[2])
        try:
            mobile = load_verified_raster_text_direct_mobile_paddle_bundle_v1(
                document_preview_root=root,
                manifest_sha256=mobile_bundle.manifest_sha256,
            )
            recognizer = load_verified_voyage_replay_release_v1(
                recognizer_root, recognizer_release.release_sha256
            )
        except (RasterTextMobilePaddleBundleError, VoyageReplayError):
            raise RasterTextPublicVoyageReplayPaddleBundleError(
                "public_voyage_replay_paddle_bundle_source_drift"
            ) from None
        if (
            _local_path_identity(root) != _local_path_identity(recognizer_root)
            or mobile.bundle_root_relative_path != mobile_bundle.bundle_root_relative_path
            or recognizer.release_path != recognizer_release.release_path
        ):
            raise RasterTextPublicVoyageReplayPaddleBundleError(
                "public_voyage_replay_paddle_bundle_source_binding_invalid"
            )
        receipt = recognizer.receipt
        if (
            mobile.manifest.text_detection_model_name != _DETECTOR_NAME
            or not isinstance(receipt, VoyageReplayReceiptV3)
            or receipt.trainer_revision != _TRAINER_REVISION
            or receipt.model_revision != _MODEL_REVISION
            or receipt.output_model_name != _RECOGNIZER_NAME
            or receipt.stage != "voyage-replay-v3"
            or receipt.tuning_scope != "ctc_head_only"
            or receipt.output_class_count != _OUTPUT_CLASS_COUNT
            or receipt.initialization != "verified_v2_release_warm_start"
            or receipt.static_roundtrip_verified is not True
            or receipt.public_mobile_model_bundle_manifest_sha256 != mobile.manifest_sha256
            or set(receipt.parent_model_artifact_sha256s)
            != {f"text-rec/{name}" for name in _FILENAMES}
            or set(receipt.model_artifact_sha256s) != {f"text-rec/{name}" for name in _FILENAMES}
        ):
            raise ValueError
        characters = _config_characters(recognizer.file_paths["text-rec/config.json"])
        dictionary_sha = hashlib.sha256(canonical_json_bytes(list(characters))).hexdigest()
        if dictionary_sha != receipt.base_dictionary_sha256:
            raise ValueError
        detector: list[_SourceFile] = []
        for name in _FILENAMES:
            relative = f"models/text-det/{name}"
            source = _safe(mobile.bundle_root, relative)
            item = _source_file(source, relative, f"text-det/{name}")
            if mobile.file_sha256(relative) != item.sha256:
                raise ValueError
            detector.append(item)
        recognition: list[_SourceFile] = []
        for name in _FILENAMES:
            relative = f"text-rec/{name}"
            source = recognizer.file_paths[relative]
            item = _source_file(source, relative, relative)
            if (
                recognizer.file_sha256s[relative] != item.sha256
                or receipt.model_artifact_sha256s[relative] != item.sha256
            ):
                raise ValueError
            recognition.append(item)
        detector_tuple = tuple(detector)
        recognizer_tuple = tuple(recognition)
        return _VerifiedSources(
            root=root,
            mobile=mobile,
            recognizer=recognizer,
            receipt=receipt,
            detector_files=detector_tuple,
            recognizer_files=recognizer_tuple,
            detector_snapshot_sha256=_record_snapshot(detector_tuple),
            recognizer_snapshot_sha256=_record_snapshot(recognizer_tuple),
            characters=characters,
        )
    except RasterTextPublicVoyageReplayPaddleBundleError:
        raise
    except (
        IndexError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
        RasterTextMobilePaddleBundleError,
        VoyageReplayError,
        RasterTextDirectPaddleBundleError,
    ):
        raise RasterTextPublicVoyageReplayPaddleBundleError(
            "public_voyage_replay_paddle_bundle_source_invalid"
        ) from None


def _runtime_record(
    item: RasterTextPublicVoyageReplayPaddleBundleFileV1,
) -> dict[str, object]:
    return {
        "relative_path": item.path,
        "sha256": item.sha256,
        "byte_size": item.byte_size,
    }


def _inventory_provenance(
    arguments: Mapping[str, object],
) -> dict[str, object]:
    return {
        key: (
            value.model_dump(mode="json", warnings=False) if isinstance(value, BaseModel) else value
        )
        for key, value in arguments.items()
    }


def _inventory(
    sources: _VerifiedSources,
    records: Sequence[RasterTextPublicVoyageReplayPaddleBundleFileV1],
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "inventory_kind": "raster_text_direct_paddle_public_voyage_replay_models",
        **_inventory_provenance(_receipt_manifest_arguments(sources)),
        "text_detection_model_name": _DETECTOR_NAME,
        "text_recognition_model_name": _RECOGNIZER_NAME,
        "artifacts": [_runtime_record(item) for item in records],
    }


def _write_bytes(path: Path, raw: bytes, *, code: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.is_symlink():
            raise OSError
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        raise RasterTextPublicVoyageReplayPaddleBundleError(code) from None


def _write_record(
    stage: Path, relative: str, raw: bytes, *, code: str
) -> RasterTextPublicVoyageReplayPaddleBundleFileV1:
    _write_bytes(stage.joinpath(*PurePosixPath(relative).parts), raw, code=code)
    return RasterTextPublicVoyageReplayPaddleBundleFileV1(
        path=relative,
        sha256=hashlib.sha256(raw).hexdigest(),
        byte_size=len(raw),
    )


def _copy_model(stage: Path, item: _SourceFile) -> RasterTextPublicVoyageReplayPaddleBundleFileV1:
    target_relative = f"models/{item.target_relative_path}"
    target = stage.joinpath(*PurePosixPath(target_relative).parts)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        digest = hashlib.sha256()
        size = 0
        with item.source.open("rb") as source, os.fdopen(descriptor, "wb") as destination:
            while chunk := source.read(_COPY_CHUNK_BYTES):
                size += len(chunk)
                if size > _MAX_FILE_BYTES:
                    raise OSError
                digest.update(chunk)
                destination.write(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        if size != item.byte_size or digest.hexdigest() != item.sha256:
            raise OSError
        return RasterTextPublicVoyageReplayPaddleBundleFileV1(
            path=target_relative,
            sha256=item.sha256,
            byte_size=item.byte_size,
        )
    except (OSError, ValidationError):
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass
        raise RasterTextPublicVoyageReplayPaddleBundleError(
            "public_voyage_replay_paddle_bundle_source_drift"
        ) from None


def _source_provenance(sources: _VerifiedSources) -> tuple[tuple[str, bytes], ...]:
    mobile_manifest = sources.mobile.bundle_root / "manifest.json"
    mobile_inventory = sources.mobile.bundle_root.joinpath(
        *PurePosixPath(sources.mobile.manifest.model_inventory_relative_path).parts
    )
    recognizer_manifest = sources.recognizer.release_path / "manifest.json"
    recognizer_receipt = sources.recognizer.file_paths["training-receipt.json"]
    result: list[tuple[str, bytes]] = []
    for relative, path in (
        (_MOBILE_MANIFEST_PROVENANCE, mobile_manifest),
        (_MOBILE_INVENTORY_PROVENANCE, mobile_inventory),
        (_RECOGNIZER_MANIFEST_PROVENANCE, recognizer_manifest),
        (_RECOGNIZER_RECEIPT_PROVENANCE, recognizer_receipt),
    ):
        _value, raw = _read_canonical_json(
            path, code="public_voyage_replay_paddle_bundle_source_drift"
        )
        result.append((relative, raw))
    return tuple(result)


def _build_stage(stage: Path, sources: _VerifiedSources) -> str:
    model_records = [
        *(_copy_model(stage, item) for item in sources.detector_files),
        *(_copy_model(stage, item) for item in sources.recognizer_files),
    ]
    model_records.sort(key=lambda item: item.path)
    inventory_model_records = [
        item.model_copy(update={"path": item.path.removeprefix("models/")})
        for item in model_records
    ]
    inventory_record = _write_record(
        stage,
        _MODEL_INVENTORY,
        _canonical(_inventory(sources, inventory_model_records)),
        code="public_voyage_replay_paddle_bundle_stage_write_failed",
    )
    provenance_records = [
        _write_record(
            stage,
            relative,
            raw,
            code="public_voyage_replay_paddle_bundle_stage_write_failed",
        )
        for relative, raw in _source_provenance(sources)
    ]
    manifest = RasterTextPublicVoyageReplayPaddleBundleManifestV1.model_validate(
        {
            **_receipt_manifest_arguments(sources),
            "model_root_tree_sha256": _model_tree_sha256(inventory_model_records),
            "files": tuple(
                sorted(
                    [*model_records, inventory_record, *provenance_records],
                    key=lambda item: item.path,
                )
            ),
        }
    )
    raw = _canonical(manifest.model_dump(mode="json", warnings=False))
    _write_bytes(
        stage / "manifest.json",
        raw,
        code="public_voyage_replay_paddle_bundle_stage_write_failed",
    )
    return hashlib.sha256(raw).hexdigest()


def _tree_files(root: Path) -> dict[str, Path]:
    try:
        result: dict[str, Path] = {}
        for path in root.rglob("*"):
            if path.is_symlink():
                raise OSError
            if path.is_file():
                result[path.relative_to(root).as_posix()] = path
            elif not path.is_dir():
                raise OSError
        return result
    except (OSError, ValueError):
        raise RasterTextPublicVoyageReplayPaddleBundleError(
            "public_voyage_replay_paddle_bundle_manifest_invalid"
        ) from None


def _manifest_provenance(
    manifest: RasterTextPublicVoyageReplayPaddleBundleManifestV1,
) -> dict[str, object]:
    fields = {
        "source_mobile_bundle_manifest_sha256",
        "source_detector_model_snapshot_sha256",
        "source_recognizer_release_sha256",
        "source_recognizer_receipt_sha256",
        "source_recognizer_training_intent_sha256",
        "source_recognizer_parent_release_sha256",
        "source_recognizer_parent_receipt_sha256",
        "source_recognizer_parent_model_artifact_sha256s",
        "source_recognizer_train_manifest_sha256",
        "source_recognizer_development_manifest_sha256",
        "source_recognizer_train_corpus_manifest",
        "source_recognizer_development_corpus_manifest",
        "source_recognizer_selection_receipt_relative_path",
        "source_recognizer_selection_receipt_sha256",
        "source_recognizer_parent_selection_receipt_sha256",
        "source_recognizer_voyage_recipe_set_sha256",
        "source_recognizer_voyage_recipe_sha256s",
        "source_recognizer_train_voyage_recipe_lane_counts",
        "source_recognizer_development_voyage_recipe_lane_counts",
        "source_recognizer_independent_audit_recipe_sha256",
        "source_recognizer_independent_audit_family_ids_sha256",
        "source_recognizer_independent_audit_source_sha256s_sha256",
        "source_recognizer_trainer_revision",
        "source_recognizer_model_revision",
        "source_recognizer_training_module_sha256",
        "source_recognizer_corpus_module_sha256",
        "source_recognizer_source_crop_base_module_sha256",
        "source_recognizer_direct_runtime_bundle_manifest_sha256",
        "source_recognizer_direct_runtime_package_inventory_sha256",
        "source_recognizer_public_mobile_model_bundle_manifest_sha256",
        "source_recognizer_base_dictionary_sha256",
        "source_recognizer_model_snapshot_sha256",
        "source_recognizer_output_class_count",
    }
    return _inventory_provenance({name: getattr(manifest, name) for name in fields})


def _validate_provenance(
    manifest: RasterTextPublicVoyageReplayPaddleBundleManifestV1,
    tree: Mapping[str, Path],
    model_records: Sequence[RasterTextPublicVoyageReplayPaddleBundleFileV1],
) -> None:
    mobile_value, mobile_raw = _read_canonical_json(
        tree[_MOBILE_MANIFEST_PROVENANCE],
        code="public_voyage_replay_paddle_bundle_manifest_invalid",
    )
    mobile = RasterTextMobilePaddleBundleManifestV1.model_validate(mobile_value)
    recognizer_value, recognizer_raw = _read_canonical_json(
        tree[_RECOGNIZER_MANIFEST_PROVENANCE],
        code="public_voyage_replay_paddle_bundle_manifest_invalid",
    )
    recognizer = VoyageReplayReleaseManifestV1.model_validate(recognizer_value)
    receipt_value, receipt_raw = _read_canonical_json(
        tree[_RECOGNIZER_RECEIPT_PROVENANCE],
        code="public_voyage_replay_paddle_bundle_manifest_invalid",
    )
    receipt = VoyageReplayReceiptV3.model_validate(receipt_value)
    mobile_inventory_value, mobile_inventory_raw = _read_canonical_json(
        tree[_MOBILE_INVENTORY_PROVENANCE],
        code="public_voyage_replay_paddle_bundle_manifest_invalid",
    )
    records = {item.path: item for item in model_records}
    detector_source = tuple(
        _SourceFile(
            source=tree[f"models/text-det/{name}"],
            source_relative_path=f"models/text-det/{name}",
            target_relative_path=f"text-det/{name}",
            sha256=records[f"text-det/{name}"].sha256,
            byte_size=records[f"text-det/{name}"].byte_size,
        )
        for name in _FILENAMES
    )
    recognizer_source = tuple(
        _SourceFile(
            source=tree[f"models/text-rec/{name}"],
            source_relative_path=f"text-rec/{name}",
            target_relative_path=f"text-rec/{name}",
            sha256=records[f"text-rec/{name}"].sha256,
            byte_size=records[f"text-rec/{name}"].byte_size,
        )
        for name in _FILENAMES
    )
    expected_arguments = _receipt_arguments(
        receipt,
        mobile_manifest_sha256=hashlib.sha256(mobile_raw).hexdigest(),
        recognizer_release_sha256=hashlib.sha256(recognizer_raw).hexdigest(),
        detector_snapshot_sha256=_record_snapshot(detector_source),
        recognizer_snapshot_sha256=_record_snapshot(recognizer_source),
    )
    mobile_records = {item.path: item for item in mobile.files}
    recognizer_records = {item.path: item for item in recognizer.files}
    mobile_inventory_record = mobile_records[mobile.model_inventory_relative_path]
    receipt_record = recognizer_records["training-receipt.json"]
    if (
        _inventory_provenance(expected_arguments) != _manifest_provenance(manifest)
        or mobile_raw != _canonical(mobile.model_dump(mode="json", warnings=False))
        or recognizer_raw != _canonical(recognizer.model_dump(mode="json", warnings=False))
        or hashlib.sha256(mobile_raw).hexdigest() != manifest.source_mobile_bundle_manifest_sha256
        or hashlib.sha256(recognizer_raw).hexdigest() != manifest.source_recognizer_release_sha256
        or recognizer.receipt_sha256 != receipt.receipt_sha256
        or receipt.receipt_sha256 != manifest.source_recognizer_receipt_sha256
        or receipt_raw != canonical_json_bytes(receipt.model_dump(mode="json", warnings=False))
        or mobile_inventory_raw != _canonical(mobile_inventory_value)
        or hashlib.sha256(mobile_inventory_raw).hexdigest() != mobile_inventory_record.sha256
        or len(mobile_inventory_raw) != mobile_inventory_record.byte_size
        or hashlib.sha256(receipt_raw).hexdigest() != receipt_record.sha256
        or len(receipt_raw) != receipt_record.byte_size
        or receipt.trainer_revision != _TRAINER_REVISION
        or receipt.model_revision != _MODEL_REVISION
        or receipt.output_model_name != _RECOGNIZER_NAME
        or receipt.stage != "voyage-replay-v3"
        or receipt.tuning_scope != "ctc_head_only"
        or receipt.output_class_count != _OUTPUT_CLASS_COUNT
        or receipt.static_roundtrip_verified is not True
        or mobile.text_detection_model_name != _DETECTOR_NAME
        or receipt.public_mobile_model_bundle_manifest_sha256
        != manifest.source_mobile_bundle_manifest_sha256
        or any(
            mobile_records[f"models/text-det/{name}"].sha256 != records[f"text-det/{name}"].sha256
            or mobile_records[f"models/text-det/{name}"].byte_size
            != records[f"text-det/{name}"].byte_size
            for name in _FILENAMES
        )
        or any(
            recognizer_records[f"text-rec/{name}"].sha256 != records[f"text-rec/{name}"].sha256
            or recognizer_records[f"text-rec/{name}"].byte_size
            != records[f"text-rec/{name}"].byte_size
            or receipt.model_artifact_sha256s[f"text-rec/{name}"]
            != records[f"text-rec/{name}"].sha256
            for name in _FILENAMES
        )
    ):
        raise ValueError


def _load_from_root(
    root: Path, bundle_root: Path, expected_sha256: str
) -> VerifiedRasterTextPublicVoyageReplayPaddleBundleV1:
    try:
        manifest_value, manifest_raw = _read_canonical_json(
            bundle_root / "manifest.json",
            code="public_voyage_replay_paddle_bundle_manifest_invalid",
        )
        if hashlib.sha256(manifest_raw).hexdigest() != expected_sha256:
            raise ValueError
        manifest = RasterTextPublicVoyageReplayPaddleBundleManifestV1.model_validate(manifest_value)
        tree = _tree_files(bundle_root)
        if set(tree) != {"manifest.json", *(item.path for item in manifest.files)}:
            raise ValueError
        records = {item.path: item for item in manifest.files}
        for relative, record in records.items():
            digest, size = _hash_file(
                tree[relative],
                code="public_voyage_replay_paddle_bundle_bundle_drift",
            )
            if digest != record.sha256 or size != record.byte_size:
                raise RasterTextPublicVoyageReplayPaddleBundleError(
                    "public_voyage_replay_paddle_bundle_bundle_drift"
                )
        model_records = tuple(
            records[f"models/{relative}"].model_copy(update={"path": relative})
            for relative in _MODEL_PATHS
        )
        if _model_tree_sha256(model_records) != manifest.model_root_tree_sha256:
            raise ValueError
        characters = _config_characters(tree["models/text-rec/config.json"])
        if (
            hashlib.sha256(canonical_json_bytes(list(characters))).hexdigest()
            != manifest.source_recognizer_base_dictionary_sha256
        ):
            raise ValueError
        _validate_provenance(manifest, tree, model_records)
        inventory_value, inventory_raw = _read_canonical_json(
            tree[_MODEL_INVENTORY],
            code="public_voyage_replay_paddle_bundle_manifest_invalid",
        )
        expected_inventory = {
            "schema_version": "1.0",
            "inventory_kind": "raster_text_direct_paddle_public_voyage_replay_models",
            **_manifest_provenance(manifest),
            "text_detection_model_name": _DETECTOR_NAME,
            "text_recognition_model_name": _RECOGNIZER_NAME,
            "artifacts": [_runtime_record(item) for item in model_records],
        }
        if inventory_value != expected_inventory or inventory_raw != _canonical(expected_inventory):
            raise ValueError
        return VerifiedRasterTextPublicVoyageReplayPaddleBundleV1(
            document_preview_root=root,
            bundle_root=bundle_root,
            bundle_root_relative_path=f"{_RELEASES}/{expected_sha256}",
            manifest=manifest,
            manifest_sha256=expected_sha256,
            file_sha256_by_relative_path={item.path: item.sha256 for item in manifest.files},
        )
    except RasterTextPublicVoyageReplayPaddleBundleError:
        raise
    except (KeyError, OSError, TypeError, ValueError, ValidationError):
        raise RasterTextPublicVoyageReplayPaddleBundleError(
            "public_voyage_replay_paddle_bundle_manifest_invalid"
        ) from None


def load_verified_raster_text_public_voyage_replay_paddle_bundle_v1(
    document_preview_root: Path,
    manifest_sha256: str,
) -> VerifiedRasterTextPublicVoyageReplayPaddleBundleV1:
    root = _verified_root(document_preview_root)
    if re.fullmatch(_SHA256, manifest_sha256) is None:
        raise RasterTextPublicVoyageReplayPaddleBundleError(
            "public_voyage_replay_paddle_bundle_manifest_invalid"
        )
    bundle_root = _safe(root, f"{_RELEASES}/{manifest_sha256}", directory=True)
    return _load_from_root(root, bundle_root, manifest_sha256)


@contextmanager
def _lock(root: Path) -> Iterator[None]:
    try:
        with exclusive_local_process_lock(root / _LOCK_FILE):
            yield
    except LocalProcessLockBusy:
        raise RasterTextPublicVoyageReplayPaddleBundleError(
            "public_voyage_replay_paddle_bundle_locked"
        ) from None
    except LocalProcessLockInvalid:
        raise RasterTextPublicVoyageReplayPaddleBundleError(
            "public_voyage_replay_paddle_bundle_lock_invalid"
        ) from None


def _remove_stage(root: Path, stage: Path) -> None:
    try:
        collection = root / _COLLECTION
        if stage.parent != collection or not stage.name.startswith(".staging-"):
            return
        operational_collection = _operational_path(collection, force_short=True)
        operational_stage = _operational_path(stage, force_short=True)
        if (
            _local_path_identity(operational_stage.parent)
            != _local_path_identity(operational_collection)
            or operational_stage.is_symlink()
            or not operational_stage.is_dir()
        ):
            return
        shutil.rmtree(operational_stage)
    except (OSError, RasterTextDirectPaddleBundleError):
        return


def materialize_verified_raster_text_public_voyage_replay_paddle_bundle_v1(
    mobile_bundle: VerifiedRasterTextDirectMobilePaddleBundleV1,
    recognizer_release: VerifiedVoyageReplayReleaseV1,
) -> VerifiedRasterTextPublicVoyageReplayPaddleBundleV1:
    sources = _resolve_sources(mobile_bundle, recognizer_release)
    root = sources.root
    with _lock(root):
        collection = root / _COLLECTION
        releases = collection / "releases"
        try:
            releases.mkdir(parents=True, exist_ok=True)
            if collection.is_symlink() or releases.is_symlink():
                raise OSError
            stage = collection / f".staging-{uuid.uuid4().hex}"
            stage.mkdir(mode=0o700)
        except OSError:
            raise RasterTextPublicVoyageReplayPaddleBundleError(
                "public_voyage_replay_paddle_bundle_publish_failed"
            ) from None
        try:
            operational_stage = _operational_path(stage.resolve(strict=True), force_short=True)
            digest = _build_stage(operational_stage, sources)
            staged = _load_from_root(root, operational_stage, digest)
            final = _operational_path(releases.resolve(strict=True), force_short=True) / digest
            if final.exists() or final.is_symlink():
                return load_verified_raster_text_public_voyage_replay_paddle_bundle_v1(root, digest)
            os.rename(stage, final)
            return VerifiedRasterTextPublicVoyageReplayPaddleBundleV1(
                document_preview_root=root,
                bundle_root=final,
                bundle_root_relative_path=f"{_RELEASES}/{digest}",
                manifest=staged.manifest,
                manifest_sha256=digest,
                file_sha256_by_relative_path=staged.file_sha256_by_relative_path,
            )
        except RasterTextPublicVoyageReplayPaddleBundleError:
            raise
        except OSError:
            raise RasterTextPublicVoyageReplayPaddleBundleError(
                "public_voyage_replay_paddle_bundle_publish_failed"
            ) from None
        finally:
            _remove_stage(root, stage)


def _ensure_directory(root: Path, relative: str) -> Path:
    try:
        target = root.joinpath(*PurePosixPath(_relative_path(relative)).parts)
        target.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or not target.is_dir():
            raise OSError
        return target
    except (OSError, ValueError, RasterTextDirectPaddleBundleError):
        raise RasterTextPublicVoyageReplayPaddleBundleError(
            "public_voyage_replay_paddle_bundle_runtime_profile_write_failed"
        ) from None


def prepare_raster_text_public_voyage_replay_runtime_profile_v20(
    sidecar_bundle: VerifiedRasterTextDirectPaddleBundleV1,
    model_bundle: VerifiedRasterTextPublicVoyageReplayPaddleBundleV1,
    *,
    pdfium_version: str,
    minimum_page_confidence: float,
    cpu_threads: int = 6,
) -> PreparedRasterTextPublicVoyageReplayRuntimeProfileV20:
    """Create or verify one exact V20 profile without selecting a pointer."""

    if (
        not isinstance(pdfium_version, str)
        or not pdfium_version
        or len(pdfium_version) > 128
        or isinstance(minimum_page_confidence, bool)
        or not isinstance(minimum_page_confidence, (int, float))
        or not 0 <= float(minimum_page_confidence) <= 1
        or isinstance(cpu_threads, bool)
        or cpu_threads not in {4, 6}
    ):
        raise RasterTextPublicVoyageReplayPaddleBundleError(
            "public_voyage_replay_paddle_bundle_runtime_profile_invalid"
        )
    minimum_confidence = float(minimum_page_confidence)
    try:
        from customs_ai_agent.service import raster_text_preview_runtime as runtime
        from customs_ai_agent.service.raster_text_direct_paddle_sidecar import (
            _profile_from_configuration,
        )

        sidecar = load_verified_raster_text_direct_paddle_bundle_v1(
            document_preview_root=sidecar_bundle.document_preview_root,
            manifest_sha256=sidecar_bundle.manifest_sha256,
        )
        combined = load_verified_raster_text_public_voyage_replay_paddle_bundle_v1(
            model_bundle.document_preview_root, model_bundle.manifest_sha256
        )
        if (
            _local_path_identity(sidecar.document_preview_root)
            != _local_path_identity(combined.document_preview_root)
            or sidecar.bundle_root_relative_path != sidecar_bundle.bundle_root_relative_path
            or combined.bundle_root_relative_path != model_bundle.bundle_root_relative_path
        ):
            raise ValueError
        sidecar_manifest = sidecar.manifest
        manifest = combined.manifest
        payload: dict[str, object] = {
            "schema_version": "20.0",
            "engine_kind": (
                "direct_paddle_mobile_det_public_voyage_replay_rec_harmonic_recall_"
                "native960_lowbox_strip2_semantic_repair_v2_onednn"
            ),
            "runtime_api": "paddle-inference-static-v1",
            "extractor_revision": (
                f"{sidecar_manifest.extractor_revision}:mobile-det-v20-public-voyage-replay-"
                "rec-native960-lowbox-strip2-semantic-repair-v2-onednn-crop0p32"
            ),
            "bundle_root_relative_path": sidecar.bundle_root_relative_path,
            "bundle_manifest_sha256": sidecar.manifest_sha256,
            "mobile_model_bundle_root_relative_path": combined.bundle_root_relative_path,
            "mobile_model_bundle_manifest_sha256": combined.manifest_sha256,
            "source_model_inventory_relative_path": combined.bundle_relative_path(
                manifest.recognizer_receipt_provenance_relative_path
            ),
            "source_model_inventory_sha256": combined.file_sha256(
                manifest.recognizer_receipt_provenance_relative_path
            ),
            "source_recognizer_release_sha256": manifest.source_recognizer_release_sha256,
            "source_recognizer_receipt_sha256": manifest.source_recognizer_receipt_sha256,
            "source_recognizer_training_intent_sha256": (
                manifest.source_recognizer_training_intent_sha256
            ),
            "source_recognizer_parent_release_sha256": (
                manifest.source_recognizer_parent_release_sha256
            ),
            "source_recognizer_parent_receipt_sha256": (
                manifest.source_recognizer_parent_receipt_sha256
            ),
            "source_recognizer_train_manifest_sha256": (
                manifest.source_recognizer_train_manifest_sha256
            ),
            "source_recognizer_development_manifest_sha256": (
                manifest.source_recognizer_development_manifest_sha256
            ),
            "source_recognizer_selection_receipt_relative_path": (
                manifest.source_recognizer_selection_receipt_relative_path
            ),
            "source_recognizer_selection_receipt_sha256": (
                manifest.source_recognizer_selection_receipt_sha256
            ),
            "source_recognizer_parent_selection_receipt_sha256": (
                manifest.source_recognizer_parent_selection_receipt_sha256
            ),
            "source_recognizer_voyage_recipe_set_sha256": (
                manifest.source_recognizer_voyage_recipe_set_sha256
            ),
            "source_recognizer_train_corpus_manifest": (
                manifest.source_recognizer_train_corpus_manifest.model_dump(
                    mode="json", warnings=False
                )
            ),
            "source_recognizer_development_corpus_manifest": (
                manifest.source_recognizer_development_corpus_manifest.model_dump(
                    mode="json", warnings=False
                )
            ),
            "source_recognizer_base_dictionary_sha256": (
                manifest.source_recognizer_base_dictionary_sha256
            ),
            "source_recognizer_output_class_count": _OUTPUT_CLASS_COUNT,
            "pdfium_version": pdfium_version,
            "raster_dpi": sidecar_manifest.raster_dpi,
            "sidecar_timeout_seconds": 30,
            "cpu_threads": cpu_threads,
            "text_detection_limit_side_len": 960,
            "text_detection_limit_type": "max",
            "text_detection_threshold": 0.3,
            "text_detection_box_threshold": 0.3,
            "text_detection_unclip_ratio": 1.5,
            "minimum_page_confidence": minimum_confidence,
            "parent_renderer_package_inventory_path": sidecar.bundle_relative_path(
                sidecar_manifest.parent_renderer_package_inventory_relative_path
            ),
            "parent_renderer_package_inventory_sha256": sidecar.file_sha256(
                sidecar_manifest.parent_renderer_package_inventory_relative_path
            ),
            "parent_renderer_required_packages": (
                sidecar_manifest.parent_renderer_required_packages
            ),
            "sidecar_interpreter_path": sidecar.bundle_relative_path(
                sidecar_manifest.sidecar_interpreter_relative_path
            ),
            "sidecar_interpreter_sha256": sidecar.file_sha256(
                sidecar_manifest.sidecar_interpreter_relative_path
            ),
            "sidecar_interpreter_root": sidecar.bundle_relative_path(
                sidecar_manifest.sidecar_interpreter_root_relative_path
            ),
            "sidecar_interpreter_root_tree_sha256": (
                sidecar_manifest.sidecar_interpreter_root_tree_sha256
            ),
            "sidecar_module_path": sidecar.bundle_relative_path(
                sidecar_manifest.sidecar_module_relative_path
            ),
            "sidecar_module_sha256": sidecar.file_sha256(
                sidecar_manifest.sidecar_module_relative_path
            ),
            "sidecar_module_root": sidecar.bundle_relative_path(
                sidecar_manifest.sidecar_module_root_relative_path
            ),
            "sidecar_module_root_tree_sha256": (sidecar_manifest.sidecar_module_root_tree_sha256),
            "sidecar_package_inventory_path": sidecar.bundle_relative_path(
                sidecar_manifest.sidecar_package_inventory_relative_path
            ),
            "sidecar_package_inventory_sha256": sidecar.file_sha256(
                sidecar_manifest.sidecar_package_inventory_relative_path
            ),
            "sidecar_required_packages": sidecar_manifest.sidecar_required_packages,
            "local_model_root": combined.bundle_relative_path(
                manifest.local_model_root_relative_path
            ),
            "model_inventory_path": combined.bundle_relative_path(
                manifest.model_inventory_relative_path
            ),
            "model_inventory_sha256": combined.file_sha256(manifest.model_inventory_relative_path),
            "text_detection_model_name": _DETECTOR_NAME,
            "text_detection_model_directory": "text-det",
            "text_recognition_model_name": _RECOGNIZER_NAME,
            "text_recognition_model_directory": "text-rec",
            "enable_mkldnn": True,
            "mkldnn_cache_capacity": 10,
            "recognition_batch_size": 8,
            "opencv_threads": 1,
            "warmup_shape_policy": "detector-736x960-recognizer-48x320-v1",
            "onednn_model_policy": "recognizer-only-v1",
            "recognition_crop_policy": "horizontal-height-0p32-max8-v1",
            "recognition_resize_interpolation": "linear-v1",
            "ctc_label_policy": "optional-space-v1",
        }
        fingerprint = runtime.raster_text_profile_fingerprint_sha256(payload)
        payload["profile_fingerprint_sha256"] = fingerprint
        payload["raster_text_profile_revision"] = f"synthetic-raster-text-v1:{fingerprint[:24]}"
        profile_type = getattr(runtime, "RasterTextOfflineProfileV20", None)
        if profile_type is None:
            direct_profile = _profile_from_configuration(_canonical(payload))
            if (
                direct_profile.model_spec != "mobile_det_public_voyage_replay_rec"
                or direct_profile.recognition_crop_policy != "horizontal-height-0p32-max8-v1"
                or direct_profile.recognition_resize_interpolation != "linear-v1"
                or direct_profile.recognition_batch_size != 8
                or direct_profile.enable_mkldnn is not True
                or direct_profile.voyage_retry_mode != "none"
            ):
                raise ValueError
            profile_value = payload
        else:
            profile = profile_type.model_validate(payload)
            profile_value = profile.model_dump(mode="json", warnings=False)
        suffix = str(payload["raster_text_profile_revision"]).rsplit(":", 1)[1]
        relative = f"raster-text-runtime-profiles-v1/{suffix}/profile.json"
        directory = _ensure_directory(
            combined.document_preview_root,
            f"raster-text-runtime-profiles-v1/{suffix}",
        )
        raw = _write_create_only(
            directory / "profile.json",
            profile_value,
            code="public_voyage_replay_paddle_bundle_runtime_profile_write_failed",
        )
    except RasterTextPublicVoyageReplayPaddleBundleError:
        raise
    except (
        OSError,
        TypeError,
        ValueError,
        ValidationError,
        RasterTextDirectPaddleBundleError,
    ):
        raise RasterTextPublicVoyageReplayPaddleBundleError(
            "public_voyage_replay_paddle_bundle_runtime_profile_invalid"
        ) from None
    return PreparedRasterTextPublicVoyageReplayRuntimeProfileV20(
        raster_text_profile_revision=str(payload["raster_text_profile_revision"]),
        runtime_configuration_relative_path=relative,
        runtime_configuration_sha256=hashlib.sha256(raw).hexdigest(),
        minimum_page_confidence=minimum_confidence,
    )


__all__ = [
    "PreparedRasterTextPublicVoyageReplayRuntimeProfileV20",
    "RasterTextPublicVoyageReplayPaddleBundleError",
    "RasterTextPublicVoyageReplayPaddleBundleFileV1",
    "RasterTextPublicVoyageReplayPaddleBundleManifestV1",
    "VerifiedRasterTextPublicVoyageReplayPaddleBundleV1",
    "load_verified_raster_text_public_voyage_replay_paddle_bundle_v1",
    "materialize_verified_raster_text_public_voyage_replay_paddle_bundle_v1",
    "prepare_raster_text_public_voyage_replay_runtime_profile_v20",
]
