"""Immutable mobile-detector plus public-fine-tuned recognizer bundle.

The detector and recognizer are verified by their owning loaders before any
byte is copied.  This module then publishes one exact, content-addressed
runtime tree and keeps canonical copies of both provenance chains.  It never
trains a model, reads evaluation Gold, downloads a package, or activates a
runtime pointer.
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
from customs_ai_agent.evaluation.source_attested_public_raster_text_recognizer_finetuning import (
    PublicFineTuneCropManifestV2,
    PublicFineTuneReceiptV2,
    PublicFineTuneReleaseManifestV1,
    SourceAttestedPublicRecognizerFineTuneError,
    VerifiedPublicFineTunedRecognizerReleaseV1,
    load_verified_public_finetuned_recognizer_release_v1,
)
from customs_ai_agent.evaluation.source_attested_raster_text_recognizer_training import (
    canonical_json_bytes,
    canonical_json_file_bytes,
)


_SHA256 = r"^[a-f0-9]{64}$"
_COLLECTION = "raster-text-public-finetuned-paddle-bundles-v1"
_RELEASES = f"{_COLLECTION}/releases"
_BUNDLE_ID = "raster_text_public_finetuned_paddle_combined_model_bundle_v1"
_LOCK_FILE = ".raster-text-public-finetuned-paddle-bundle-v1.lock"
_MODEL_ROOT = "models"
_MODEL_INVENTORY = "inventories/direct-paddle-public-finetuned-models.json"
_MOBILE_MANIFEST_PROVENANCE = "provenance/mobile-detector-bundle-manifest.json"
_MOBILE_INVENTORY_PROVENANCE = "provenance/mobile-detector-model-inventory.json"
_RECOGNIZER_MANIFEST_PROVENANCE = "provenance/public-finetuned-recognizer-release-manifest.json"
_RECOGNIZER_RECEIPT_PROVENANCE = "provenance/public-finetuned-recognizer-training-receipt.json"
_PROVENANCE_PATHS = (
    _MOBILE_MANIFEST_PROVENANCE,
    _MOBILE_INVENTORY_PROVENANCE,
    _RECOGNIZER_MANIFEST_PROVENANCE,
    _RECOGNIZER_RECEIPT_PROVENANCE,
)
_DETECTOR_NAME = "PP-OCRv5_mobile_det"
_RECOGNIZER_NAME = "source_attested_en_PP-OCRv5_mobile_rec_head_v2"
_V19EngineKind = Literal[
    "direct_paddle_mobile_det_public_finetuned_rec_harmonic_recall_native960_"
    "lowbox_strip2_semantic_repair_v2_onednn",
    "direct_paddle_mobile_det_public_finetuned_rec_harmonic_recall_native960_"
    "lowbox_strip2_semantic_repair_v2_voyage_retry_onednn",
]
_V19_ONE_PASS_ENGINE_KIND: _V19EngineKind = (
    "direct_paddle_mobile_det_public_finetuned_rec_harmonic_recall_native960_"
    "lowbox_strip2_semantic_repair_v2_onednn"
)
_V19_VOYAGE_RETRY_ENGINE_KIND: _V19EngineKind = (
    "direct_paddle_mobile_det_public_finetuned_rec_harmonic_recall_native960_"
    "lowbox_strip2_semantic_repair_v2_voyage_retry_onednn"
)
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


class RasterTextPublicFineTunedPaddleBundleError(RuntimeError):
    """Content-free failure at the public-fine-tuned bundle boundary."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RasterTextPublicFineTunedPaddleBundleFileV1(_WireModel):
    path: str = Field(min_length=1, max_length=480)
    sha256: str = Field(pattern=_SHA256)
    byte_size: int = Field(ge=1, le=_MAX_FILE_BYTES)

    _valid_path = field_validator("path")(_relative_path)


class RasterTextPublicFineTuneCorpusProvenanceV1(_WireModel):
    """Content-free critical-field coverage bound into the serving release."""

    split: Literal["train", "development"]
    critical_field_crop_count: int = Field(ge=200, le=1_000)
    critical_field_ids_sha256: str = Field(pattern=_SHA256)
    critical_field_crop_counts: dict[str, int]
    document_type_crop_counts: dict[str, int]
    source_label_length_p50: int = Field(ge=1, le=64)
    source_label_length_p95: int = Field(ge=1, le=64)

    @model_validator(mode="after")
    def _exact_content_free_statistics(self) -> "RasterTextPublicFineTuneCorpusProvenanceV1":
        expected = 1_000 if self.split == "train" else 200
        if (
            self.critical_field_crop_count != expected
            or sum(self.critical_field_crop_counts.values()) != expected
            or any(value < 1 for value in self.critical_field_crop_counts.values())
            or set(self.document_type_crop_counts)
            != {"bill_of_lading", "commercial_invoice", "packing_list"}
            or sum(self.document_type_crop_counts.values()) != expected
            or max(self.document_type_crop_counts.values())
            - min(self.document_type_crop_counts.values())
            > 2
            or self.critical_field_ids_sha256
            != hashlib.sha256(
                canonical_json_bytes(sorted(self.critical_field_crop_counts))
            ).hexdigest()
            or self.source_label_length_p50 > self.source_label_length_p95
        ):
            raise ValueError("public fine-tune corpus provenance is invalid")
        return self


class RasterTextPublicFineTunedPaddleBundleManifestV1(_WireModel):
    """Exhaustive binding for the combined detector/recognizer release."""

    schema_version: Literal["1.0"] = "1.0"
    bundle_id: Literal["raster_text_public_finetuned_paddle_combined_model_bundle_v1"] = (
        "raster_text_public_finetuned_paddle_combined_model_bundle_v1"
    )
    source_mobile_bundle_manifest_sha256: str = Field(pattern=_SHA256)
    source_detector_model_snapshot_sha256: str = Field(pattern=_SHA256)
    source_recognizer_release_sha256: str = Field(pattern=_SHA256)
    source_recognizer_receipt_sha256: str = Field(pattern=_SHA256)
    source_recognizer_training_intent_sha256: str = Field(pattern=_SHA256)
    source_recognizer_selection_receipt_relative_path: str = Field(min_length=1, max_length=240)
    source_recognizer_selection_receipt_sha256: str = Field(pattern=_SHA256)
    source_recognizer_parent_selection_receipt_sha256: str = Field(pattern=_SHA256)
    source_recognizer_train_corpus_provenance: RasterTextPublicFineTuneCorpusProvenanceV1
    source_recognizer_development_corpus_provenance: RasterTextPublicFineTuneCorpusProvenanceV1
    source_recognizer_base_dictionary_sha256: str = Field(pattern=_SHA256)
    source_recognizer_model_snapshot_sha256: str = Field(pattern=_SHA256)
    source_recognizer_output_class_count: Literal[438] = 438
    text_detection_model_name: Literal["PP-OCRv5_mobile_det"] = "PP-OCRv5_mobile_det"
    text_recognition_model_name: Literal["source_attested_en_PP-OCRv5_mobile_rec_head_v2"] = (
        "source_attested_en_PP-OCRv5_mobile_rec_head_v2"
    )
    local_model_root_relative_path: Literal["models"] = "models"
    model_inventory_relative_path: Literal[
        "inventories/direct-paddle-public-finetuned-models.json"
    ] = "inventories/direct-paddle-public-finetuned-models.json"
    mobile_manifest_provenance_relative_path: Literal[
        "provenance/mobile-detector-bundle-manifest.json"
    ] = "provenance/mobile-detector-bundle-manifest.json"
    mobile_inventory_provenance_relative_path: Literal[
        "provenance/mobile-detector-model-inventory.json"
    ] = "provenance/mobile-detector-model-inventory.json"
    recognizer_manifest_provenance_relative_path: Literal[
        "provenance/public-finetuned-recognizer-release-manifest.json"
    ] = "provenance/public-finetuned-recognizer-release-manifest.json"
    recognizer_receipt_provenance_relative_path: Literal[
        "provenance/public-finetuned-recognizer-training-receipt.json"
    ] = "provenance/public-finetuned-recognizer-training-receipt.json"
    model_root_tree_sha256: str = Field(pattern=_SHA256)
    files: tuple[RasterTextPublicFineTunedPaddleBundleFileV1, ...] = Field(
        min_length=len(_DECLARED_FILES), max_length=len(_DECLARED_FILES)
    )

    @model_validator(mode="after")
    def _exact_shape(self) -> "RasterTextPublicFineTunedPaddleBundleManifestV1":
        paths = tuple(item.path for item in self.files)
        if (
            paths != tuple(sorted(paths))
            or len(paths) != len(set(paths))
            or set(paths) != _DECLARED_FILES
            or self.source_recognizer_selection_receipt_relative_path
            != (
                "synthetic-raster-text-v1/fresh-scope-selections/"
                f"{self.source_recognizer_selection_receipt_sha256}.json"
            )
            or self.source_recognizer_train_corpus_provenance.split != "train"
            or self.source_recognizer_development_corpus_provenance.split != "development"
        ):
            raise ValueError("public fine-tuned Paddle bundle shape is invalid")
        return self


@dataclass(frozen=True, slots=True)
class VerifiedRasterTextPublicFineTunedPaddleBundleV1:
    document_preview_root: Path
    bundle_root: Path
    bundle_root_relative_path: str
    manifest: RasterTextPublicFineTunedPaddleBundleManifestV1
    manifest_sha256: str
    file_sha256_by_relative_path: Mapping[str, str]

    def bundle_relative_path(self, path: str) -> str:
        _relative_path(path)
        return f"{self.bundle_root_relative_path}/{path}"

    def file_sha256(self, path: str) -> str:
        try:
            return self.file_sha256_by_relative_path[path]
        except KeyError:
            raise RasterTextPublicFineTunedPaddleBundleError(
                "public_finetuned_paddle_bundle_manifest_invalid"
            ) from None


@dataclass(frozen=True, slots=True)
class PreparedRasterTextPublicFineTunedRuntimeProfileV19:
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
    recognizer: VerifiedPublicFineTunedRecognizerReleaseV1
    receipt: PublicFineTuneReceiptV2
    detector_files: tuple[_SourceFile, ...]
    recognizer_files: tuple[_SourceFile, ...]
    detector_snapshot_sha256: str
    recognizer_snapshot_sha256: str
    characters: tuple[str, ...]


def _corpus_provenance(
    manifest: PublicFineTuneCropManifestV2,
) -> RasterTextPublicFineTuneCorpusProvenanceV1:
    return RasterTextPublicFineTuneCorpusProvenanceV1(
        split=manifest.split,
        critical_field_crop_count=manifest.critical_field_crop_count,
        critical_field_ids_sha256=manifest.critical_field_ids_sha256,
        critical_field_crop_counts=manifest.critical_field_crop_counts,
        document_type_crop_counts=manifest.document_type_crop_counts,
        source_label_length_p50=manifest.source_label_length_p50,
        source_label_length_p95=manifest.source_label_length_p95,
    )


def _canonical(value: object) -> bytes:
    return _canonical_json(value)


def _verified_root(path: Path) -> Path:
    try:
        return _root(path)
    except RasterTextDirectPaddleBundleError:
        raise RasterTextPublicFineTunedPaddleBundleError(
            "public_finetuned_paddle_bundle_root_invalid"
        ) from None


def _safe(root: Path, relative: str, *, directory: bool = False) -> Path:
    try:
        return _safe_path(root, relative, directory=directory)
    except RasterTextDirectPaddleBundleError:
        raise RasterTextPublicFineTunedPaddleBundleError(
            "public_finetuned_paddle_bundle_manifest_invalid"
        ) from None


def _hash_file(path: Path, *, code: str) -> tuple[str, int]:
    try:
        return _sha256_regular_file(path, code=code)
    except RasterTextDirectPaddleBundleError:
        raise RasterTextPublicFineTunedPaddleBundleError(code) from None


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
    records: Sequence[RasterTextPublicFineTunedPaddleBundleFileV1],
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
        raise RasterTextPublicFineTunedPaddleBundleError(code) from None


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
        raise RasterTextPublicFineTunedPaddleBundleError(code) from None


def _config_characters(path: Path) -> tuple[str, ...]:
    value, _raw = _read_pretty_canonical_json(
        path, code="public_finetuned_paddle_bundle_source_invalid"
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
        raise RasterTextPublicFineTunedPaddleBundleError(
            "public_finetuned_paddle_bundle_source_invalid"
        ) from None


def _source_file(source: Path, source_relative: str, target_relative: str) -> _SourceFile:
    digest, size = _hash_file(source, code="public_finetuned_paddle_bundle_source_drift")
    return _SourceFile(
        source=source,
        source_relative_path=source_relative,
        target_relative_path=target_relative,
        sha256=digest,
        byte_size=size,
    )


def _resolve_sources(
    mobile_bundle: VerifiedRasterTextDirectMobilePaddleBundleV1,
    recognizer_release: VerifiedPublicFineTunedRecognizerReleaseV1,
) -> _VerifiedSources:
    try:
        root = _verified_root(mobile_bundle.document_preview_root)
        recognizer_root = _verified_root(recognizer_release.release_path.parents[2])
        try:
            mobile = load_verified_raster_text_direct_mobile_paddle_bundle_v1(
                document_preview_root=root, manifest_sha256=mobile_bundle.manifest_sha256
            )
            recognizer = load_verified_public_finetuned_recognizer_release_v1(
                recognizer_root, recognizer_release.release_sha256
            )
        except (
            RasterTextMobilePaddleBundleError,
            SourceAttestedPublicRecognizerFineTuneError,
        ):
            raise RasterTextPublicFineTunedPaddleBundleError(
                "public_finetuned_paddle_bundle_source_drift"
            ) from None
        if (
            _local_path_identity(root) != _local_path_identity(recognizer_root)
            or mobile.bundle_root_relative_path != mobile_bundle.bundle_root_relative_path
            or recognizer.release_path != recognizer_release.release_path
        ):
            raise RasterTextPublicFineTunedPaddleBundleError(
                "public_finetuned_paddle_bundle_source_binding_invalid"
            )
        if (
            mobile.manifest.text_detection_model_name != _DETECTOR_NAME
            or not isinstance(recognizer.receipt, PublicFineTuneReceiptV2)
            or recognizer.receipt.trainer_revision != "public-paddle-jit-ctc-leakage-safe-head-v2"
            or recognizer.receipt.stage != "leakage-safe-v2"
            or recognizer.receipt.tuning_scope != "head"
            or recognizer.receipt.maximum_epochs != 4
            or recognizer.receipt.train_holdout_overlap_count != 0
            or recognizer.receipt.output_class_count != _OUTPUT_CLASS_COUNT
            or recognizer.receipt.base_character_count != _CHARACTER_COUNT
            or recognizer.receipt.implicit_space_class_index != 437
            or recognizer.receipt.ctc_blank_index != 0
            or recognizer.receipt.dictionary_first_class_index != 1
            or recognizer.receipt.initialization != "pinned_public_inference_graph_warm_start"
            or recognizer.receipt.static_roundtrip_verified is not True
        ):
            raise ValueError
        characters = _config_characters(recognizer.file_paths["text-rec/config.json"])
        dictionary_sha = hashlib.sha256(canonical_json_bytes(list(characters))).hexdigest()
        if dictionary_sha != recognizer.receipt.base_dictionary_sha256:
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
                or recognizer.receipt.model_artifact_sha256s[relative] != item.sha256
            ):
                raise ValueError
            recognition.append(item)
        detector_tuple = tuple(detector)
        recognizer_tuple = tuple(recognition)
        return _VerifiedSources(
            root=root,
            mobile=mobile,
            recognizer=recognizer,
            receipt=recognizer.receipt,
            detector_files=detector_tuple,
            recognizer_files=recognizer_tuple,
            detector_snapshot_sha256=_record_snapshot(detector_tuple),
            recognizer_snapshot_sha256=_record_snapshot(recognizer_tuple),
            characters=characters,
        )
    except RasterTextPublicFineTunedPaddleBundleError:
        raise
    except (
        KeyError,
        OSError,
        TypeError,
        ValueError,
        RasterTextMobilePaddleBundleError,
        SourceAttestedPublicRecognizerFineTuneError,
        RasterTextDirectPaddleBundleError,
    ):
        raise RasterTextPublicFineTunedPaddleBundleError(
            "public_finetuned_paddle_bundle_source_invalid"
        ) from None


def _runtime_record(
    item: RasterTextPublicFineTunedPaddleBundleFileV1,
) -> dict[str, object]:
    return {
        "relative_path": item.path,
        "sha256": item.sha256,
        "byte_size": item.byte_size,
    }


def _inventory(
    sources: _VerifiedSources,
    records: Sequence[RasterTextPublicFineTunedPaddleBundleFileV1],
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "inventory_kind": "raster_text_direct_paddle_public_finetuned_models",
        "source_mobile_bundle_manifest_sha256": sources.mobile.manifest_sha256,
        "source_recognizer_release_sha256": sources.recognizer.release_sha256,
        "source_recognizer_receipt_sha256": sources.receipt.receipt_sha256,
        "source_recognizer_training_intent_sha256": (sources.receipt.training_intent_sha256),
        "source_recognizer_selection_receipt_relative_path": (
            sources.receipt.selection_receipt_relative_path
        ),
        "source_recognizer_selection_receipt_sha256": (sources.receipt.selection_receipt_sha256),
        "source_recognizer_parent_selection_receipt_sha256": (
            sources.receipt.parent_selection_receipt_sha256
        ),
        "source_recognizer_train_corpus_provenance": _corpus_provenance(
            sources.receipt.train_crop_manifest
        ).model_dump(mode="json", warnings=False),
        "source_recognizer_development_corpus_provenance": _corpus_provenance(
            sources.receipt.development_crop_manifest
        ).model_dump(mode="json", warnings=False),
        "source_recognizer_base_dictionary_sha256": (sources.receipt.base_dictionary_sha256),
        "source_recognizer_output_class_count": _OUTPUT_CLASS_COUNT,
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
        raise RasterTextPublicFineTunedPaddleBundleError(code) from None


def _write_record(
    stage: Path, relative: str, raw: bytes, *, code: str
) -> RasterTextPublicFineTunedPaddleBundleFileV1:
    _write_bytes(stage.joinpath(*PurePosixPath(relative).parts), raw, code=code)
    return RasterTextPublicFineTunedPaddleBundleFileV1(
        path=relative, sha256=hashlib.sha256(raw).hexdigest(), byte_size=len(raw)
    )


def _copy_model(stage: Path, item: _SourceFile) -> RasterTextPublicFineTunedPaddleBundleFileV1:
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
        return RasterTextPublicFineTunedPaddleBundleFileV1(
            path=target_relative, sha256=item.sha256, byte_size=item.byte_size
        )
    except (OSError, ValidationError):
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass
        raise RasterTextPublicFineTunedPaddleBundleError(
            "public_finetuned_paddle_bundle_source_drift"
        ) from None


def _source_provenance(sources: _VerifiedSources) -> tuple[tuple[str, bytes], ...]:
    mobile_manifest = sources.mobile.bundle_root / "manifest.json"
    mobile_inventory = sources.mobile.bundle_root.joinpath(
        *PurePosixPath(sources.mobile.manifest.model_inventory_relative_path).parts
    )
    recognizer_manifest = sources.recognizer.release_path / "manifest.json"
    recognizer_receipt = sources.recognizer.file_paths["training-receipt.json"]
    result: list[tuple[str, bytes]] = []
    for relative, path, pretty in (
        (_MOBILE_MANIFEST_PROVENANCE, mobile_manifest, False),
        (_MOBILE_INVENTORY_PROVENANCE, mobile_inventory, False),
        (_RECOGNIZER_MANIFEST_PROVENANCE, recognizer_manifest, True),
        (_RECOGNIZER_RECEIPT_PROVENANCE, recognizer_receipt, True),
    ):
        reader = _read_pretty_canonical_json if pretty else _read_canonical_json
        _value, raw = reader(path, code="public_finetuned_paddle_bundle_source_drift")
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
        code="public_finetuned_paddle_bundle_stage_write_failed",
    )
    provenance_records = [
        _write_record(
            stage,
            relative,
            raw,
            code="public_finetuned_paddle_bundle_stage_write_failed",
        )
        for relative, raw in _source_provenance(sources)
    ]
    manifest = RasterTextPublicFineTunedPaddleBundleManifestV1(
        source_mobile_bundle_manifest_sha256=sources.mobile.manifest_sha256,
        source_detector_model_snapshot_sha256=sources.detector_snapshot_sha256,
        source_recognizer_release_sha256=sources.recognizer.release_sha256,
        source_recognizer_receipt_sha256=sources.receipt.receipt_sha256,
        source_recognizer_training_intent_sha256=(sources.receipt.training_intent_sha256),
        source_recognizer_selection_receipt_relative_path=(
            sources.receipt.selection_receipt_relative_path
        ),
        source_recognizer_selection_receipt_sha256=(sources.receipt.selection_receipt_sha256),
        source_recognizer_parent_selection_receipt_sha256=(
            sources.receipt.parent_selection_receipt_sha256
        ),
        source_recognizer_train_corpus_provenance=_corpus_provenance(
            sources.receipt.train_crop_manifest
        ),
        source_recognizer_development_corpus_provenance=_corpus_provenance(
            sources.receipt.development_crop_manifest
        ),
        source_recognizer_base_dictionary_sha256=(sources.receipt.base_dictionary_sha256),
        source_recognizer_model_snapshot_sha256=sources.recognizer_snapshot_sha256,
        model_root_tree_sha256=_model_tree_sha256(inventory_model_records),
        files=tuple(
            sorted(
                [*model_records, inventory_record, *provenance_records],
                key=lambda item: item.path,
            )
        ),
    )
    raw = _canonical(manifest.model_dump(mode="json", warnings=False))
    _write_bytes(
        stage / "manifest.json",
        raw,
        code="public_finetuned_paddle_bundle_stage_write_failed",
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
        raise RasterTextPublicFineTunedPaddleBundleError(
            "public_finetuned_paddle_bundle_manifest_invalid"
        ) from None


def _validate_provenance(
    manifest: RasterTextPublicFineTunedPaddleBundleManifestV1,
    tree: Mapping[str, Path],
    model_records: Sequence[RasterTextPublicFineTunedPaddleBundleFileV1],
) -> None:
    mobile_value, mobile_raw = _read_canonical_json(
        tree[_MOBILE_MANIFEST_PROVENANCE],
        code="public_finetuned_paddle_bundle_manifest_invalid",
    )
    mobile = RasterTextMobilePaddleBundleManifestV1.model_validate(mobile_value)
    recognizer_value, recognizer_raw = _read_pretty_canonical_json(
        tree[_RECOGNIZER_MANIFEST_PROVENANCE],
        code="public_finetuned_paddle_bundle_manifest_invalid",
    )
    recognizer = PublicFineTuneReleaseManifestV1.model_validate(recognizer_value)
    receipt_value, receipt_raw = _read_pretty_canonical_json(
        tree[_RECOGNIZER_RECEIPT_PROVENANCE],
        code="public_finetuned_paddle_bundle_manifest_invalid",
    )
    receipt = PublicFineTuneReceiptV2.model_validate(receipt_value)
    _mobile_inventory, mobile_inventory_raw = _read_canonical_json(
        tree[_MOBILE_INVENTORY_PROVENANCE],
        code="public_finetuned_paddle_bundle_manifest_invalid",
    )
    if (
        hashlib.sha256(mobile_raw).hexdigest() != manifest.source_mobile_bundle_manifest_sha256
        or hashlib.sha256(recognizer_raw).hexdigest() != manifest.source_recognizer_release_sha256
        or recognizer.receipt_sha256 != receipt.receipt_sha256
        or receipt_raw != canonical_json_file_bytes(receipt.model_dump(mode="json", warnings=False))
        or receipt.receipt_sha256 != manifest.source_recognizer_receipt_sha256
        or receipt.training_intent_sha256 != manifest.source_recognizer_training_intent_sha256
        or receipt.selection_receipt_relative_path
        != manifest.source_recognizer_selection_receipt_relative_path
        or receipt.selection_receipt_sha256 != manifest.source_recognizer_selection_receipt_sha256
        or receipt.parent_selection_receipt_sha256
        != manifest.source_recognizer_parent_selection_receipt_sha256
        or _corpus_provenance(receipt.train_crop_manifest)
        != manifest.source_recognizer_train_corpus_provenance
        or _corpus_provenance(receipt.development_crop_manifest)
        != manifest.source_recognizer_development_corpus_provenance
        or receipt.trainer_revision != "public-paddle-jit-ctc-leakage-safe-head-v2"
        or receipt.stage != "leakage-safe-v2"
        or receipt.tuning_scope != "head"
        or receipt.maximum_epochs != 4
        or receipt.train_holdout_overlap_count != 0
        or receipt.base_dictionary_sha256 != manifest.source_recognizer_base_dictionary_sha256
        or receipt.output_class_count != _OUTPUT_CLASS_COUNT
        or receipt.base_character_count != _CHARACTER_COUNT
        or receipt.implicit_space_class_index != 437
        or receipt.initialization != "pinned_public_inference_graph_warm_start"
        or receipt.static_roundtrip_verified is not True
        or not mobile_inventory_raw
    ):
        raise ValueError
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
    mobile_records = {item.path: item for item in mobile.files}
    recognizer_records = {item.path: item for item in recognizer.files}
    if (
        any(
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
        or _record_snapshot(detector_source) != manifest.source_detector_model_snapshot_sha256
        or _record_snapshot(recognizer_source) != manifest.source_recognizer_model_snapshot_sha256
    ):
        raise ValueError


def _load_from_root(
    root: Path, bundle_root: Path, expected_sha256: str
) -> VerifiedRasterTextPublicFineTunedPaddleBundleV1:
    try:
        manifest_value, manifest_raw = _read_canonical_json(
            bundle_root / "manifest.json",
            code="public_finetuned_paddle_bundle_manifest_invalid",
        )
        if hashlib.sha256(manifest_raw).hexdigest() != expected_sha256:
            raise ValueError
        manifest = RasterTextPublicFineTunedPaddleBundleManifestV1.model_validate(manifest_value)
        tree = _tree_files(bundle_root)
        if set(tree) != {"manifest.json", *(item.path for item in manifest.files)}:
            raise ValueError
        records = {item.path: item for item in manifest.files}
        for relative, record in records.items():
            digest, size = _hash_file(
                tree[relative], code="public_finetuned_paddle_bundle_bundle_drift"
            )
            if digest != record.sha256 or size != record.byte_size:
                raise RasterTextPublicFineTunedPaddleBundleError(
                    "public_finetuned_paddle_bundle_bundle_drift"
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
            code="public_finetuned_paddle_bundle_manifest_invalid",
        )
        expected_inventory = {
            "schema_version": "1.0",
            "inventory_kind": "raster_text_direct_paddle_public_finetuned_models",
            "source_mobile_bundle_manifest_sha256": (manifest.source_mobile_bundle_manifest_sha256),
            "source_recognizer_release_sha256": manifest.source_recognizer_release_sha256,
            "source_recognizer_receipt_sha256": manifest.source_recognizer_receipt_sha256,
            "source_recognizer_training_intent_sha256": (
                manifest.source_recognizer_training_intent_sha256
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
            "source_recognizer_train_corpus_provenance": (
                manifest.source_recognizer_train_corpus_provenance.model_dump(
                    mode="json", warnings=False
                )
            ),
            "source_recognizer_development_corpus_provenance": (
                manifest.source_recognizer_development_corpus_provenance.model_dump(
                    mode="json", warnings=False
                )
            ),
            "source_recognizer_base_dictionary_sha256": (
                manifest.source_recognizer_base_dictionary_sha256
            ),
            "source_recognizer_output_class_count": _OUTPUT_CLASS_COUNT,
            "text_detection_model_name": _DETECTOR_NAME,
            "text_recognition_model_name": _RECOGNIZER_NAME,
            "artifacts": [_runtime_record(item) for item in model_records],
        }
        if inventory_value != expected_inventory or inventory_raw != _canonical(expected_inventory):
            raise ValueError
        return VerifiedRasterTextPublicFineTunedPaddleBundleV1(
            document_preview_root=root,
            bundle_root=bundle_root,
            bundle_root_relative_path=f"{_RELEASES}/{expected_sha256}",
            manifest=manifest,
            manifest_sha256=expected_sha256,
            file_sha256_by_relative_path={item.path: item.sha256 for item in manifest.files},
        )
    except RasterTextPublicFineTunedPaddleBundleError:
        raise
    except (KeyError, OSError, TypeError, ValueError, ValidationError):
        raise RasterTextPublicFineTunedPaddleBundleError(
            "public_finetuned_paddle_bundle_manifest_invalid"
        ) from None


def load_verified_raster_text_public_finetuned_paddle_bundle_v1(
    document_preview_root: Path,
    manifest_sha256: str,
) -> VerifiedRasterTextPublicFineTunedPaddleBundleV1:
    root = _verified_root(document_preview_root)
    if re.fullmatch(_SHA256, manifest_sha256) is None:
        raise RasterTextPublicFineTunedPaddleBundleError(
            "public_finetuned_paddle_bundle_manifest_invalid"
        )
    bundle_root = _safe(root, f"{_RELEASES}/{manifest_sha256}", directory=True)
    return _load_from_root(root, bundle_root, manifest_sha256)


@contextmanager
def _lock(root: Path) -> Iterator[None]:
    try:
        with exclusive_local_process_lock(root / _LOCK_FILE):
            yield
    except LocalProcessLockBusy:
        raise RasterTextPublicFineTunedPaddleBundleError(
            "public_finetuned_paddle_bundle_locked"
        ) from None
    except LocalProcessLockInvalid:
        raise RasterTextPublicFineTunedPaddleBundleError(
            "public_finetuned_paddle_bundle_lock_invalid"
        ) from None


def _remove_stage(root: Path, stage: Path) -> None:
    try:
        canonical = stage.resolve(strict=True)
        collection = (root / _COLLECTION).resolve(strict=True)
        canonical.relative_to(collection)
        if canonical.parent != collection or not canonical.name.startswith(".staging-"):
            return
        shutil.rmtree(canonical)
    except OSError:
        return


def materialize_verified_raster_text_public_finetuned_paddle_bundle_v1(
    mobile_bundle: VerifiedRasterTextDirectMobilePaddleBundleV1,
    recognizer_release: VerifiedPublicFineTunedRecognizerReleaseV1,
) -> VerifiedRasterTextPublicFineTunedPaddleBundleV1:
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
            raise RasterTextPublicFineTunedPaddleBundleError(
                "public_finetuned_paddle_bundle_publish_failed"
            ) from None
        try:
            operational_stage = _operational_path(stage.resolve(strict=True), force_short=True)
            digest = _build_stage(operational_stage, sources)
            staged = _load_from_root(root, operational_stage, digest)
            final = _operational_path(releases.resolve(strict=True), force_short=True) / digest
            if final.exists() or final.is_symlink():
                return load_verified_raster_text_public_finetuned_paddle_bundle_v1(root, digest)
            os.rename(stage, final)
            return VerifiedRasterTextPublicFineTunedPaddleBundleV1(
                document_preview_root=root,
                bundle_root=final,
                bundle_root_relative_path=f"{_RELEASES}/{digest}",
                manifest=staged.manifest,
                manifest_sha256=digest,
                file_sha256_by_relative_path=staged.file_sha256_by_relative_path,
            )
        except RasterTextPublicFineTunedPaddleBundleError:
            raise
        except OSError:
            raise RasterTextPublicFineTunedPaddleBundleError(
                "public_finetuned_paddle_bundle_publish_failed"
            ) from None
        finally:
            if stage.exists():
                _remove_stage(root, stage)


def _ensure_directory(root: Path, relative: str) -> Path:
    try:
        target = root.joinpath(*PurePosixPath(_relative_path(relative)).parts)
        target.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or not target.is_dir():
            raise OSError
        return target
    except (OSError, ValueError, RasterTextDirectPaddleBundleError):
        raise RasterTextPublicFineTunedPaddleBundleError(
            "public_finetuned_paddle_bundle_runtime_profile_write_failed"
        ) from None


def _prepare_raster_text_public_finetuned_runtime_profile_v19(
    sidecar_bundle: VerifiedRasterTextDirectPaddleBundleV1,
    model_bundle: VerifiedRasterTextPublicFineTunedPaddleBundleV1,
    *,
    pdfium_version: str,
    minimum_page_confidence: float,
    cpu_threads: int = 6,
    engine_kind: _V19EngineKind,
) -> PreparedRasterTextPublicFineTunedRuntimeProfileV19:
    """Create one of the two closed V19 runtime variants."""

    if (
        not isinstance(pdfium_version, str)
        or not pdfium_version
        or len(pdfium_version) > 128
        or isinstance(minimum_page_confidence, bool)
        or not isinstance(minimum_page_confidence, (int, float))
        or not 0 <= float(minimum_page_confidence) <= 1
        or isinstance(cpu_threads, bool)
        or cpu_threads not in {4, 6}
        or engine_kind not in {_V19_ONE_PASS_ENGINE_KIND, _V19_VOYAGE_RETRY_ENGINE_KIND}
    ):
        raise RasterTextPublicFineTunedPaddleBundleError(
            "public_finetuned_paddle_bundle_runtime_profile_invalid"
        )
    try:
        from customs_ai_agent.service.raster_text_preview_runtime import (
            RasterTextOfflineProfileV19,
            raster_text_profile_fingerprint_sha256,
        )

        sidecar = load_verified_raster_text_direct_paddle_bundle_v1(
            document_preview_root=sidecar_bundle.document_preview_root,
            manifest_sha256=sidecar_bundle.manifest_sha256,
        )
        combined = load_verified_raster_text_public_finetuned_paddle_bundle_v1(
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
        voyage_retry = engine_kind == _V19_VOYAGE_RETRY_ENGINE_KIND
        extractor_variant = (
            "mobile-det-v19b-public-finetuned-rec-native960-lowbox-strip2-semantic-repair-"
            "v2-voyage-retry-onednn"
            if voyage_retry
            else "mobile-det-v19-public-finetuned-rec-native960-lowbox-strip2-semantic-repair-"
            "v2-onednn"
        )
        payload: dict[str, object] = {
            "schema_version": "19.0",
            "engine_kind": engine_kind,
            "runtime_api": "paddle-inference-static-v1",
            "extractor_revision": f"{sidecar_manifest.extractor_revision}:{extractor_variant}",
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
            "source_recognizer_selection_receipt_relative_path": (
                manifest.source_recognizer_selection_receipt_relative_path
            ),
            "source_recognizer_selection_receipt_sha256": (
                manifest.source_recognizer_selection_receipt_sha256
            ),
            "source_recognizer_parent_selection_receipt_sha256": (
                manifest.source_recognizer_parent_selection_receipt_sha256
            ),
            "source_recognizer_train_corpus_provenance": (
                manifest.source_recognizer_train_corpus_provenance.model_dump(
                    mode="json", warnings=False
                )
            ),
            "source_recognizer_development_corpus_provenance": (
                manifest.source_recognizer_development_corpus_provenance.model_dump(
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
            "minimum_page_confidence": float(minimum_page_confidence),
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
            "ctc_label_policy": "optional-space-v1",
        }
        fingerprint = raster_text_profile_fingerprint_sha256(payload)
        payload["profile_fingerprint_sha256"] = fingerprint
        payload["raster_text_profile_revision"] = f"synthetic-raster-text-v1:{fingerprint[:24]}"
        profile = RasterTextOfflineProfileV19.model_validate(payload)
        suffix = profile.raster_text_profile_revision.rsplit(":", 1)[1]
        relative = f"raster-text-runtime-profiles-v1/{suffix}/profile.json"
        directory = _ensure_directory(
            combined.document_preview_root, f"raster-text-runtime-profiles-v1/{suffix}"
        )
        raw = _write_create_only(
            directory / "profile.json",
            profile.model_dump(mode="json", warnings=False),
            code="public_finetuned_paddle_bundle_runtime_profile_write_failed",
        )
    except RasterTextPublicFineTunedPaddleBundleError:
        raise
    except (
        OSError,
        TypeError,
        ValueError,
        ValidationError,
        RasterTextDirectPaddleBundleError,
    ):
        raise RasterTextPublicFineTunedPaddleBundleError(
            "public_finetuned_paddle_bundle_runtime_profile_invalid"
        ) from None
    return PreparedRasterTextPublicFineTunedRuntimeProfileV19(
        raster_text_profile_revision=profile.raster_text_profile_revision,
        runtime_configuration_relative_path=relative,
        runtime_configuration_sha256=hashlib.sha256(raw).hexdigest(),
        minimum_page_confidence=profile.minimum_page_confidence,
    )


def prepare_raster_text_public_finetuned_runtime_profile_v19(
    sidecar_bundle: VerifiedRasterTextDirectPaddleBundleV1,
    model_bundle: VerifiedRasterTextPublicFineTunedPaddleBundleV1,
    *,
    pdfium_version: str,
    minimum_page_confidence: float,
    cpu_threads: int = 6,
) -> PreparedRasterTextPublicFineTunedRuntimeProfileV19:
    """Preserve the independently audited V19b voyage-retry profile."""

    return _prepare_raster_text_public_finetuned_runtime_profile_v19(
        sidecar_bundle,
        model_bundle,
        pdfium_version=pdfium_version,
        minimum_page_confidence=minimum_page_confidence,
        cpu_threads=cpu_threads,
        engine_kind=_V19_VOYAGE_RETRY_ENGINE_KIND,
    )


def prepare_raster_text_public_finetuned_one_pass_runtime_profile_v19(
    sidecar_bundle: VerifiedRasterTextDirectPaddleBundleV1,
    model_bundle: VerifiedRasterTextPublicFineTunedPaddleBundleV1,
    *,
    pdfium_version: str,
    minimum_page_confidence: float,
    cpu_threads: int = 6,
) -> PreparedRasterTextPublicFineTunedRuntimeProfileV19:
    """Create the retained best one-pass V19 profile on this native sidecar."""

    return _prepare_raster_text_public_finetuned_runtime_profile_v19(
        sidecar_bundle,
        model_bundle,
        pdfium_version=pdfium_version,
        minimum_page_confidence=minimum_page_confidence,
        cpu_threads=cpu_threads,
        engine_kind=_V19_ONE_PASS_ENGINE_KIND,
    )


__all__ = [
    "PreparedRasterTextPublicFineTunedRuntimeProfileV19",
    "RasterTextPublicFineTuneCorpusProvenanceV1",
    "RasterTextPublicFineTunedPaddleBundleError",
    "RasterTextPublicFineTunedPaddleBundleFileV1",
    "RasterTextPublicFineTunedPaddleBundleManifestV1",
    "VerifiedRasterTextPublicFineTunedPaddleBundleV1",
    "load_verified_raster_text_public_finetuned_paddle_bundle_v1",
    "materialize_verified_raster_text_public_finetuned_paddle_bundle_v1",
    "prepare_raster_text_public_finetuned_one_pass_runtime_profile_v19",
    "prepare_raster_text_public_finetuned_runtime_profile_v19",
]
