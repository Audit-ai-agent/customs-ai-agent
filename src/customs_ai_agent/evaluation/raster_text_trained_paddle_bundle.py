"""Content-addressed mobile-detector plus trained-recognizer model bundle.

The detector and recognizer remain independently attested upstream.  This
module copies only their already-verified model bytes into one exact runtime
tree and preserves canonical copies of both provenance chains.  It never
creates a runtime profile, selects an approval pointer, downloads a model, or
reads Gold text.
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
    load_verified_raster_text_direct_paddle_bundle_v1,
)
from customs_ai_agent.evaluation.raster_text_mobile_paddle_bundle import (
    RasterTextMobilePaddleBundleError,
    VerifiedRasterTextDirectMobilePaddleBundleV1,
    load_verified_raster_text_direct_mobile_paddle_bundle_v1,
)
from customs_ai_agent.evaluation.source_attested_raster_text_recognizer_training import (
    SourceAttestedRecognizerTrainingError,
    VerifiedRecognizerRelease,
    canonical_json_bytes,
    load_verified_source_attested_raster_text_recognizer_release,
)


_SHA256_PATTERN = r"^[a-f0-9]{64}$"
_MAX_JSON_BYTES = 128 * 1024 * 1024
_MAX_FILE_BYTES = 1024 * 1024 * 1024
_DOCUMENT_PREVIEW_ROOT_SUFFIX = ("_local", "data", "document-preview")
_COLLECTION = "raster-text-trained-paddle-bundles-v1"
_RELEASES = f"{_COLLECTION}/releases"
_RUNTIME_PROFILE_COLLECTION = "raster-text-runtime-profiles-v1"
_LOCK_FILE = ".raster-text-trained-paddle-bundle-v1.lock"
_BUNDLE_ID = "raster_text_trained_paddle_combined_model_bundle_v1"
_DETECTOR_NAME = "PP-OCRv5_mobile_det"
_RECOGNIZER_NAME = "source_attested_tiny_crnn_bigru_ctc_v1"
_MODEL_FILENAMES = (
    "config.json",
    "inference.json",
    "inference.pdiparams",
    "inference.yml",
)
_MODEL_PATHS = tuple(
    f"{directory}/{filename}"
    for directory in ("text-det", "text-rec")
    for filename in _MODEL_FILENAMES
)
_MODEL_INVENTORY_PATH = "inventories/direct-paddle-trained-models.json"
_DETECTOR_MANIFEST_PROVENANCE_PATH = "provenance/mobile-detector-bundle-manifest.json"
_DETECTOR_INVENTORY_PROVENANCE_PATH = "provenance/mobile-detector-model-inventory.json"
_RECOGNIZER_MANIFEST_PROVENANCE_PATH = "provenance/trained-recognizer-release-manifest.json"
_RECOGNIZER_RECEIPT_PROVENANCE_PATH = "provenance/trained-recognizer-training-receipt.json"
_PROVENANCE_PATHS = (
    _DETECTOR_MANIFEST_PROVENANCE_PATH,
    _DETECTOR_INVENTORY_PROVENANCE_PATH,
    _RECOGNIZER_MANIFEST_PROVENANCE_PATH,
    _RECOGNIZER_RECEIPT_PROVENANCE_PATH,
)
_DECLARED_FILES = {
    _MODEL_INVENTORY_PATH,
    *(f"models/{path}" for path in _MODEL_PATHS),
    *_PROVENANCE_PATHS,
}
_DECLARED_DIRECTORIES = {
    "inventories",
    "models",
    "models/text-det",
    "models/text-rec",
    "provenance",
}


class RasterTextTrainedPaddleBundleError(RuntimeError):
    """Stable, content-free error for the combined model bundle."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or path.as_posix() != value
        or "\\" in value
        or ":" in value
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("trained Paddle relative path is invalid")
    return value


class TrainedPaddleBundleFileV1(_WireModel):
    path: str = Field(min_length=1, max_length=480)
    sha256: str = Field(pattern=_SHA256_PATTERN)
    byte_size: int = Field(ge=1, le=_MAX_FILE_BYTES)

    _valid_path = field_validator("path")(_relative_path)


class RasterTextTrainedPaddleBundleManifestV1(_WireModel):
    """Exhaustive binding for the combined detector/recognizer release."""

    schema_version: Literal["1.0"] = "1.0"
    bundle_id: Literal["raster_text_trained_paddle_combined_model_bundle_v1"] = (
        "raster_text_trained_paddle_combined_model_bundle_v1"
    )
    source_detector_bundle_manifest_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_detector_model_inventory_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_detector_model_snapshot_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_recognizer_release_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_recognizer_receipt_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_recognizer_training_intent_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_recognizer_character_dictionary_sha256: str = Field(pattern=_SHA256_PATTERN)
    source_recognizer_package_inventory_sha256: str = Field(pattern=_SHA256_PATTERN)
    text_detection_model_name: Literal["PP-OCRv5_mobile_det"] = "PP-OCRv5_mobile_det"
    text_recognition_model_name: Literal["source_attested_tiny_crnn_bigru_ctc_v1"] = (
        "source_attested_tiny_crnn_bigru_ctc_v1"
    )
    local_model_root_relative_path: Literal["models"] = "models"
    model_inventory_relative_path: Literal["inventories/direct-paddle-trained-models.json"] = (
        "inventories/direct-paddle-trained-models.json"
    )
    detector_manifest_provenance_relative_path: Literal[
        "provenance/mobile-detector-bundle-manifest.json"
    ] = "provenance/mobile-detector-bundle-manifest.json"
    detector_inventory_provenance_relative_path: Literal[
        "provenance/mobile-detector-model-inventory.json"
    ] = "provenance/mobile-detector-model-inventory.json"
    recognizer_manifest_provenance_relative_path: Literal[
        "provenance/trained-recognizer-release-manifest.json"
    ] = "provenance/trained-recognizer-release-manifest.json"
    recognizer_receipt_provenance_relative_path: Literal[
        "provenance/trained-recognizer-training-receipt.json"
    ] = "provenance/trained-recognizer-training-receipt.json"
    model_root_tree_sha256: str = Field(pattern=_SHA256_PATTERN)
    files: tuple[TrainedPaddleBundleFileV1, ...] = Field(
        min_length=len(_DECLARED_FILES), max_length=len(_DECLARED_FILES)
    )

    @model_validator(mode="after")
    def _exact_shape(self) -> "RasterTextTrainedPaddleBundleManifestV1":
        paths = tuple(item.path for item in self.files)
        if (
            paths != tuple(sorted(paths))
            or len(paths) != len(set(paths))
            or set(paths) != _DECLARED_FILES
            or self.source_recognizer_release_sha256 != self.source_recognizer_receipt_sha256
        ):
            raise ValueError("trained Paddle bundle manifest shape is invalid")
        return self


@dataclass(frozen=True, slots=True)
class VerifiedRasterTextTrainedPaddleBundleV1:
    document_preview_root: Path
    bundle_root: Path
    bundle_root_relative_path: str
    manifest: RasterTextTrainedPaddleBundleManifestV1
    manifest_sha256: str
    file_sha256_by_relative_path: Mapping[str, str]

    def bundle_relative_path(self, path: str) -> str:
        _relative_path(path)
        return f"{self.bundle_root_relative_path}/{path}"

    def file_sha256(self, path: str) -> str:
        try:
            return self.file_sha256_by_relative_path[path]
        except KeyError:
            raise RasterTextTrainedPaddleBundleError(
                "trained_paddle_bundle_manifest_invalid"
            ) from None


@dataclass(frozen=True, slots=True)
class PreparedRasterTextTrainedRuntimeProfileV12:
    """One immutable V12 profile binding the sidecar and combined models."""

    raster_text_profile_revision: str
    runtime_configuration_relative_path: str
    runtime_configuration_sha256: str
    minimum_page_confidence: float


@dataclass(frozen=True, slots=True)
class _VerifiedSources:
    detector: VerifiedRasterTextDirectMobilePaddleBundleV1
    recognizer: VerifiedRecognizerRelease


def _canonical_bytes(value: object) -> bytes:
    return canonical_json_bytes(value)


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _verified_root(path: Path) -> Path:
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
        raise RasterTextTrainedPaddleBundleError("trained_paddle_bundle_root_invalid") from None


def _same_path(first: Path, second: Path) -> bool:
    try:
        return os.path.samefile(first, second)
    except OSError:
        return os.path.normcase(str(first.resolve())) == os.path.normcase(str(second.resolve()))


def _local_path_identity(path: Path) -> str:
    value = os.path.realpath(str(path))
    if os.name == "nt" and value.startswith("\\\\?\\"):
        value = value[4:]
    return os.path.normcase(os.path.normpath(value))


def _same_or_child(root: Path, path: Path) -> bool:
    try:
        root_identity = _local_path_identity(root)
        path_identity = _local_path_identity(path)
        return os.path.commonpath((root_identity, path_identity)) == root_identity
    except ValueError:
        return False


def _operational_path(path: Path, *, force_short: bool = False) -> Path:
    """Use an equivalent existing 8.3 spelling for deep Windows operations."""

    if os.name != "nt":
        return path
    value = str(path)
    if not force_short and value.isascii() and len(value) <= 160:
        return path
    try:
        import ctypes

        kernel32 = getattr(ctypes, "windll").kernel32
        required = kernel32.GetShortPathNameW(value, None, 0)
        if not isinstance(required, int) or required < 1:
            raise OSError
        buffer = ctypes.create_unicode_buffer(required + 1)
        written = kernel32.GetShortPathNameW(value, buffer, len(buffer))
        if not isinstance(written, int) or not 0 < written < len(buffer):
            raise OSError
        short = buffer.value
        if not short or not short.isascii():
            raise OSError
        return Path(short)
    except Exception:
        raise RasterTextTrainedPaddleBundleError("trained_paddle_bundle_path_invalid") from None


def _safe_path(root: Path, relative: str, *, directory: bool = False) -> Path:
    try:
        parts = PurePosixPath(_relative_path(relative)).parts
        candidate = root
        for part in parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise OSError
        if candidate.is_symlink() or not _same_or_child(root, candidate):
            raise OSError
        if directory:
            if not candidate.is_dir():
                raise OSError
        elif not candidate.is_file():
            raise OSError
        return _operational_path(candidate)
    except (OSError, ValueError):
        raise RasterTextTrainedPaddleBundleError("trained_paddle_bundle_path_invalid") from None


def _hash_file(path: Path, *, code: str) -> tuple[str, int]:
    try:
        if path.is_symlink() or not path.is_file():
            raise OSError
        size = path.stat().st_size
        if not 1 <= size <= _MAX_FILE_BYTES:
            raise OSError
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        return digest.hexdigest(), size
    except OSError:
        raise RasterTextTrainedPaddleBundleError(code) from None


def _read_canonical_object(path: Path, *, code: str) -> tuple[dict[str, Any], bytes]:
    try:
        if (
            path.is_symlink()
            or not path.is_file()
            or not 1 <= path.stat().st_size <= _MAX_JSON_BYTES
        ):
            raise OSError
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict) or raw != _canonical_bytes(value):
            raise ValueError
        return value, raw
    except (OSError, UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError):
        raise RasterTextTrainedPaddleBundleError(code) from None


def _write_bytes_fsynced(path: Path, raw: bytes, *, code: str) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.is_symlink():
            raise OSError
        with path.open("xb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError:
        raise RasterTextTrainedPaddleBundleError(code) from None


def _write_json_fsynced(path: Path, value: object, *, code: str) -> bytes:
    raw = _canonical_bytes(value)
    _write_bytes_fsynced(path, raw, code=code)
    return raw


def _write_json_create_or_verify(path: Path, value: object, *, code: str) -> bytes:
    raw = _canonical_bytes(value)
    try:
        _write_bytes_fsynced(path, raw, code=code)
    except RasterTextTrainedPaddleBundleError:
        try:
            if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
                raise OSError
        except OSError:
            raise RasterTextTrainedPaddleBundleError(code) from None
    return raw


def _copy_fsynced(source: Path, target: Path, *, code: str) -> tuple[str, int]:
    expected_sha256, expected_size = _hash_file(source, code=code)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            raise OSError
        digest = hashlib.sha256()
        size = 0
        with source.open("rb") as source_stream, target.open("xb") as target_stream:
            while chunk := source_stream.read(1024 * 1024):
                target_stream.write(chunk)
                digest.update(chunk)
                size += len(chunk)
            target_stream.flush()
            os.fsync(target_stream.fileno())
    except OSError:
        raise RasterTextTrainedPaddleBundleError(code) from None
    if digest.hexdigest() != expected_sha256 or size != expected_size:
        raise RasterTextTrainedPaddleBundleError(code)
    return expected_sha256, expected_size


def _release_relative_path(manifest_sha256: str) -> str:
    if re.fullmatch(_SHA256_PATTERN, manifest_sha256) is None:
        raise RasterTextTrainedPaddleBundleError("trained_paddle_bundle_manifest_invalid")
    return f"{_RELEASES}/{manifest_sha256}"


def _load_sources(
    *, root: Path, detector_bundle_manifest_sha256: str, recognizer_release_sha256: str
) -> _VerifiedSources:
    try:
        detector = load_verified_raster_text_direct_mobile_paddle_bundle_v1(
            document_preview_root=root, manifest_sha256=detector_bundle_manifest_sha256
        )
        recognizer = load_verified_source_attested_raster_text_recognizer_release(
            root, recognizer_release_sha256
        )
    except (RasterTextMobilePaddleBundleError, SourceAttestedRecognizerTrainingError):
        raise RasterTextTrainedPaddleBundleError("trained_paddle_bundle_source_invalid") from None
    try:
        if not _same_or_child(root, recognizer.release_path) or not _same_path(
            detector.document_preview_root, root
        ):
            raise ValueError
    except (OSError, ValueError):
        raise RasterTextTrainedPaddleBundleError("trained_paddle_bundle_source_invalid") from None
    return _VerifiedSources(detector=detector, recognizer=recognizer)


def _source_model_records(sources: _VerifiedSources) -> tuple[TrainedPaddleBundleFileV1, ...]:
    records: list[TrainedPaddleBundleFileV1] = []
    for filename in _MODEL_FILENAMES:
        detector_relative = f"models/text-det/{filename}"
        detector_path = sources.detector.bundle_root.joinpath(
            *PurePosixPath(detector_relative).parts
        )
        detector_sha256, detector_size = _hash_file(
            detector_path, code="trained_paddle_bundle_source_invalid"
        )
        if detector_sha256 != sources.detector.file_sha256(detector_relative):
            raise RasterTextTrainedPaddleBundleError("trained_paddle_bundle_source_invalid")
        records.append(
            TrainedPaddleBundleFileV1(
                path=f"text-det/{filename}",
                sha256=detector_sha256,
                byte_size=detector_size,
            )
        )
    for filename in _MODEL_FILENAMES:
        recognizer_relative = f"text-rec/{filename}"
        try:
            recognizer_path = sources.recognizer.file_paths[recognizer_relative]
            expected_sha256 = sources.recognizer.file_sha256s[recognizer_relative]
        except KeyError:
            raise RasterTextTrainedPaddleBundleError(
                "trained_paddle_bundle_source_invalid"
            ) from None
        recognizer_sha256, recognizer_size = _hash_file(
            recognizer_path, code="trained_paddle_bundle_source_invalid"
        )
        if recognizer_sha256 != expected_sha256:
            raise RasterTextTrainedPaddleBundleError("trained_paddle_bundle_source_invalid")
        records.append(
            TrainedPaddleBundleFileV1(
                path=f"text-rec/{filename}",
                sha256=recognizer_sha256,
                byte_size=recognizer_size,
            )
        )
    return tuple(records)


def _model_tree_sha256(records: Sequence[TrainedPaddleBundleFileV1]) -> str:
    return _sha256_bytes(
        _canonical_bytes(
            {
                "files": [
                    {
                        "path": item.path,
                        "sha256": item.sha256,
                        "byte_size": item.byte_size,
                    }
                    for item in sorted(records, key=lambda record: record.path)
                ]
            }
        )
    )


def _model_identifier(raw: bytes, *, expected: str) -> dict[str, Any]:
    try:
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError
        names: set[str] = set()
        top_level = value.get("model_name")
        if isinstance(top_level, str):
            names.add(top_level)
        global_configuration = value.get("Global")
        if isinstance(global_configuration, dict):
            nested = global_configuration.get("model_name")
            if isinstance(nested, str):
                names.add(nested)
        if names != {expected}:
            raise ValueError
        return value
    except (UnicodeDecodeError, ValueError, TypeError, json.JSONDecodeError):
        raise RasterTextTrainedPaddleBundleError("trained_paddle_bundle_model_invalid") from None


def _verify_model_identities(*, model_root: Path, sources: _VerifiedSources) -> None:
    detector_config = _safe_path(model_root, "text-det/config.json").read_bytes()
    _model_identifier(detector_config, expected=_DETECTOR_NAME)
    recognizer_config = _model_identifier(
        _safe_path(model_root, "text-rec/config.json").read_bytes(), expected=_RECOGNIZER_NAME
    )
    try:
        post_process = recognizer_config["PostProcess"]
        characters = post_process["character_dict"]
        if (
            not isinstance(post_process, dict)
            or not isinstance(characters, list)
            or not characters
            or any(not isinstance(character, str) or not character for character in characters)
            or len(characters) != len(set(characters))
            or _sha256_bytes(_canonical_bytes(characters))
            != sources.recognizer.receipt.character_dictionary_sha256
        ):
            raise ValueError
    except (KeyError, TypeError, ValueError):
        raise RasterTextTrainedPaddleBundleError("trained_paddle_bundle_model_invalid") from None


def _combined_inventory(
    *, sources: _VerifiedSources, model_records: Sequence[TrainedPaddleBundleFileV1]
) -> dict[str, object]:
    detector_records = [item for item in model_records if item.path.startswith("text-det/")]
    recognizer_records = [item for item in model_records if item.path.startswith("text-rec/")]
    detector_manifest = sources.detector.manifest
    receipt = sources.recognizer.receipt
    return {
        "schema_version": "1.0",
        "inventory_kind": "raster_text_trained_paddle_combined_models",
        "artifacts": [
            {
                "relative_path": item.path,
                "sha256": item.sha256,
                "byte_size": item.byte_size,
            }
            for item in model_records
        ],
        "detector": {
            "bundle_manifest_sha256": sources.detector.manifest_sha256,
            "source_model_inventory_relative_path": (
                detector_manifest.source_model_inventory_relative_path
            ),
            "source_model_inventory_sha256": detector_manifest.source_model_inventory_sha256,
            "source_model_snapshot_sha256": detector_manifest.source_model_snapshot_sha256,
            "model_name": _DETECTOR_NAME,
            "artifacts": [
                item.model_dump(mode="json", warnings=False) for item in detector_records
            ],
        },
        "recognizer": {
            "release_sha256": sources.recognizer.release_sha256,
            "receipt_sha256": receipt.receipt_sha256,
            "training_intent_sha256": receipt.training_intent_sha256,
            "trainer_revision": receipt.trainer_revision,
            "model_revision": receipt.model_revision,
            "character_dictionary_sha256": receipt.character_dictionary_sha256,
            "package_inventory_sha256": receipt.package_inventory_sha256,
            "model_name": _RECOGNIZER_NAME,
            "artifacts": [
                item.model_dump(mode="json", warnings=False) for item in recognizer_records
            ],
        },
    }


def _source_provenance_paths(sources: _VerifiedSources) -> Mapping[str, Path]:
    detector_inventory = sources.detector.bundle_root.joinpath(
        *PurePosixPath(sources.detector.manifest.model_inventory_relative_path).parts
    )
    try:
        recognizer_receipt = sources.recognizer.file_paths["training-receipt.json"]
    except KeyError:
        raise RasterTextTrainedPaddleBundleError("trained_paddle_bundle_source_invalid") from None
    return {
        _DETECTOR_MANIFEST_PROVENANCE_PATH: sources.detector.bundle_root / "manifest.json",
        _DETECTOR_INVENTORY_PROVENANCE_PATH: detector_inventory,
        _RECOGNIZER_MANIFEST_PROVENANCE_PATH: sources.recognizer.release_path / "manifest.json",
        _RECOGNIZER_RECEIPT_PROVENANCE_PATH: recognizer_receipt,
    }


def _manifest_for(
    *,
    sources: _VerifiedSources,
    model_records: Sequence[TrainedPaddleBundleFileV1],
    files: Sequence[TrainedPaddleBundleFileV1],
) -> RasterTextTrainedPaddleBundleManifestV1:
    detector_inventory_sha256 = sources.detector.file_sha256(
        sources.detector.manifest.model_inventory_relative_path
    )
    receipt = sources.recognizer.receipt
    return RasterTextTrainedPaddleBundleManifestV1(
        source_detector_bundle_manifest_sha256=sources.detector.manifest_sha256,
        source_detector_model_inventory_sha256=detector_inventory_sha256,
        source_detector_model_snapshot_sha256=(
            sources.detector.manifest.source_model_snapshot_sha256
        ),
        source_recognizer_release_sha256=sources.recognizer.release_sha256,
        source_recognizer_receipt_sha256=receipt.receipt_sha256,
        source_recognizer_training_intent_sha256=receipt.training_intent_sha256,
        source_recognizer_character_dictionary_sha256=receipt.character_dictionary_sha256,
        source_recognizer_package_inventory_sha256=receipt.package_inventory_sha256,
        model_root_tree_sha256=_model_tree_sha256(model_records),
        files=tuple(sorted(files, key=lambda item: item.path)),
    )


def _tree(root: Path) -> tuple[dict[str, Path], set[str]]:
    files: dict[str, Path] = {}
    directories: set[str] = set()
    try:
        for path in root.rglob("*"):
            if path.is_symlink():
                raise OSError
            relative = path.relative_to(root).as_posix()
            if path.is_file():
                if relative in files:
                    raise OSError
                files[relative] = path
            elif path.is_dir():
                directories.add(relative)
            else:
                raise OSError
        return files, directories
    except (OSError, ValueError):
        raise RasterTextTrainedPaddleBundleError("trained_paddle_bundle_tree_invalid") from None


def _load_verified_from_root(
    *,
    root: Path,
    bundle_root: Path,
    expected_manifest_sha256: str,
    expected_staging_root: Path | None = None,
) -> VerifiedRasterTextTrainedPaddleBundleV1:
    try:
        if bundle_root.is_symlink() or not bundle_root.is_dir():
            raise RasterTextTrainedPaddleBundleError("trained_paddle_bundle_tree_invalid")
        manifest_value, manifest_raw = _read_canonical_object(
            bundle_root / "manifest.json", code="trained_paddle_bundle_manifest_invalid"
        )
        if _sha256_bytes(manifest_raw) != expected_manifest_sha256:
            raise RasterTextTrainedPaddleBundleError("trained_paddle_bundle_manifest_invalid")
        manifest = RasterTextTrainedPaddleBundleManifestV1.model_validate(manifest_value)
        if manifest_raw != _canonical_bytes(manifest.model_dump(mode="json", warnings=False)):
            raise RasterTextTrainedPaddleBundleError("trained_paddle_bundle_manifest_invalid")
        sources = _load_sources(
            root=root,
            detector_bundle_manifest_sha256=manifest.source_detector_bundle_manifest_sha256,
            recognizer_release_sha256=manifest.source_recognizer_release_sha256,
        )
        if (
            manifest.source_detector_model_inventory_sha256
            != sources.detector.file_sha256(sources.detector.manifest.model_inventory_relative_path)
            or manifest.source_detector_model_snapshot_sha256
            != sources.detector.manifest.source_model_snapshot_sha256
            or manifest.source_recognizer_receipt_sha256
            != sources.recognizer.receipt.receipt_sha256
            or manifest.source_recognizer_training_intent_sha256
            != sources.recognizer.receipt.training_intent_sha256
            or manifest.source_recognizer_character_dictionary_sha256
            != sources.recognizer.receipt.character_dictionary_sha256
            or manifest.source_recognizer_package_inventory_sha256
            != sources.recognizer.receipt.package_inventory_sha256
        ):
            raise RasterTextTrainedPaddleBundleError("trained_paddle_bundle_binding_invalid")
        files, directories = _tree(bundle_root)
        if (
            set(files) != {"manifest.json", *_DECLARED_FILES}
            or directories != _DECLARED_DIRECTORIES
        ):
            raise RasterTextTrainedPaddleBundleError("trained_paddle_bundle_tree_invalid")
        records = {item.path: item for item in manifest.files}
        for relative, record in records.items():
            path = _safe_path(bundle_root, relative)
            digest, byte_size = _hash_file(path, code="trained_paddle_bundle_drift")
            if digest != record.sha256 or byte_size != record.byte_size:
                raise RasterTextTrainedPaddleBundleError("trained_paddle_bundle_drift")
        expected_model_records = _source_model_records(sources)
        observed_model_records = tuple(
            TrainedPaddleBundleFileV1(
                path=relative,
                sha256=records[f"models/{relative}"].sha256,
                byte_size=records[f"models/{relative}"].byte_size,
            )
            for relative in _MODEL_PATHS
        )
        if observed_model_records != expected_model_records:
            raise RasterTextTrainedPaddleBundleError("trained_paddle_bundle_binding_invalid")
        if _model_tree_sha256(observed_model_records) != manifest.model_root_tree_sha256:
            raise RasterTextTrainedPaddleBundleError("trained_paddle_bundle_binding_invalid")
        model_root = _safe_path(
            bundle_root, manifest.local_model_root_relative_path, directory=True
        )
        _verify_model_identities(model_root=model_root, sources=sources)
        inventory, inventory_raw = _read_canonical_object(
            _safe_path(bundle_root, manifest.model_inventory_relative_path),
            code="trained_paddle_bundle_inventory_invalid",
        )
        if (
            inventory != _combined_inventory(sources=sources, model_records=observed_model_records)
            or _sha256_bytes(inventory_raw) != records[_MODEL_INVENTORY_PATH].sha256
            or len(inventory_raw) != records[_MODEL_INVENTORY_PATH].byte_size
        ):
            raise RasterTextTrainedPaddleBundleError("trained_paddle_bundle_inventory_invalid")
        for relative, source_path in _source_provenance_paths(sources).items():
            source_sha256, source_size = _hash_file(
                source_path, code="trained_paddle_bundle_source_invalid"
            )
            record = records[relative]
            if record.sha256 != source_sha256 or record.byte_size != source_size:
                raise RasterTextTrainedPaddleBundleError("trained_paddle_bundle_binding_invalid")
        if expected_staging_root is not None:
            if (
                expected_staging_root.is_symlink()
                or not _same_path(expected_staging_root.parent, root)
                or not expected_staging_root.name.startswith(f".{_COLLECTION}.staging-")
                or not _same_path(bundle_root, expected_staging_root)
            ):
                raise RasterTextTrainedPaddleBundleError("trained_paddle_bundle_manifest_invalid")
            relative_root = expected_staging_root.name
        else:
            expected_relative = _release_relative_path(expected_manifest_sha256)
            expected_root = _safe_path(root, expected_relative, directory=True)
            if not _same_path(bundle_root, expected_root):
                raise RasterTextTrainedPaddleBundleError("trained_paddle_bundle_manifest_invalid")
            relative_root = expected_relative
        return VerifiedRasterTextTrainedPaddleBundleV1(
            document_preview_root=root,
            bundle_root=bundle_root,
            bundle_root_relative_path=relative_root,
            manifest=manifest,
            manifest_sha256=expected_manifest_sha256,
            file_sha256_by_relative_path={path: item.sha256 for path, item in records.items()},
        )
    except RasterTextTrainedPaddleBundleError:
        raise
    except (OSError, ValueError, TypeError, ValidationError):
        raise RasterTextTrainedPaddleBundleError("trained_paddle_bundle_manifest_invalid") from None


def _ensure_directory(root: Path, relative: str) -> Path:
    try:
        candidate = root
        for part in PurePosixPath(_relative_path(relative)).parts:
            candidate = candidate / part
            if candidate.exists():
                if candidate.is_symlink() or not candidate.is_dir():
                    raise OSError
            else:
                candidate.mkdir()
        if not _same_or_child(root, candidate):
            raise ValueError
        return _operational_path(candidate)
    except (OSError, ValueError):
        raise RasterTextTrainedPaddleBundleError("trained_paddle_bundle_output_invalid") from None


@contextmanager
def _lock(root: Path) -> Iterator[None]:
    try:
        with exclusive_local_process_lock(root / _LOCK_FILE):
            yield
    except LocalProcessLockBusy:
        raise RasterTextTrainedPaddleBundleError("trained_paddle_bundle_locked") from None
    except LocalProcessLockInvalid:
        raise RasterTextTrainedPaddleBundleError("trained_paddle_bundle_lock_invalid") from None


def _remove_owned_stage(root: Path, stage: Path) -> None:
    try:
        if stage.is_symlink():
            return
        resolved = stage.resolve(strict=True)
        if (
            not _same_path(resolved.parent, root)
            or resolved.is_symlink()
            or not resolved.name.startswith(f".{_COLLECTION}.staging-")
        ):
            return
        shutil.rmtree(resolved)
    except OSError:
        return


def _build_stage(*, stage: Path, sources: _VerifiedSources) -> str:
    source_records = _source_model_records(sources)
    written: list[TrainedPaddleBundleFileV1] = []
    for record in source_records:
        if record.path.startswith("text-det/"):
            source_relative = f"models/{record.path}"
            source = sources.detector.bundle_root.joinpath(*PurePosixPath(source_relative).parts)
        else:
            try:
                source = sources.recognizer.file_paths[record.path]
            except KeyError:
                raise RasterTextTrainedPaddleBundleError(
                    "trained_paddle_bundle_source_invalid"
                ) from None
        target_relative = f"models/{record.path}"
        copied_sha256, copied_size = _copy_fsynced(
            source,
            stage.joinpath(*PurePosixPath(target_relative).parts),
            code="trained_paddle_bundle_stage_copy_failed",
        )
        written.append(
            TrainedPaddleBundleFileV1(
                path=target_relative, sha256=copied_sha256, byte_size=copied_size
            )
        )
    inventory_raw = _write_json_fsynced(
        stage / _MODEL_INVENTORY_PATH,
        _combined_inventory(sources=sources, model_records=source_records),
        code="trained_paddle_bundle_stage_write_failed",
    )
    written.append(
        TrainedPaddleBundleFileV1(
            path=_MODEL_INVENTORY_PATH,
            sha256=_sha256_bytes(inventory_raw),
            byte_size=len(inventory_raw),
        )
    )
    for relative, source in _source_provenance_paths(sources).items():
        copied_sha256, copied_size = _copy_fsynced(
            source,
            stage.joinpath(*PurePosixPath(relative).parts),
            code="trained_paddle_bundle_stage_copy_failed",
        )
        written.append(
            TrainedPaddleBundleFileV1(path=relative, sha256=copied_sha256, byte_size=copied_size)
        )
    manifest = _manifest_for(
        sources=sources,
        model_records=source_records,
        files=written,
    )
    raw = _write_json_fsynced(
        stage / "manifest.json",
        manifest.model_dump(mode="json", warnings=False),
        code="trained_paddle_bundle_stage_write_failed",
    )
    return _sha256_bytes(raw)


def load_verified_raster_text_trained_paddle_bundle_v1(
    *, document_preview_root: Path, manifest_sha256: str
) -> VerifiedRasterTextTrainedPaddleBundleV1:
    """Load one exact combined release and reverify both upstream releases."""

    root = _verified_root(document_preview_root)
    relative = _release_relative_path(manifest_sha256)
    bundle_root = _safe_path(root, relative, directory=True)
    return _load_verified_from_root(
        root=root,
        bundle_root=bundle_root,
        expected_manifest_sha256=manifest_sha256,
    )


def materialize_verified_raster_text_trained_paddle_bundle_v1(
    *,
    document_preview_root: Path,
    detector_bundle_manifest_sha256: str,
    recognizer_release_sha256: str,
) -> VerifiedRasterTextTrainedPaddleBundleV1:
    """Create or reuse the exact combined model tree without fallback."""

    root = _verified_root(document_preview_root)
    with _lock(root):
        sources = _load_sources(
            root=root,
            detector_bundle_manifest_sha256=detector_bundle_manifest_sha256,
            recognizer_release_sha256=recognizer_release_sha256,
        )
        _ensure_directory(root, _COLLECTION)
        releases = _ensure_directory(root, _RELEASES)
        stage_canonical = root / f".{_COLLECTION}.staging-{uuid.uuid4().hex}"
        try:
            if stage_canonical.exists() or stage_canonical.is_symlink():
                raise RasterTextTrainedPaddleBundleError("trained_paddle_bundle_output_invalid")
            stage_canonical.mkdir(mode=0o700)
            stage = _operational_path(stage_canonical, force_short=True)
            manifest_sha256 = _build_stage(stage=stage, sources=sources)
            _load_verified_from_root(
                root=root,
                bundle_root=stage,
                expected_manifest_sha256=manifest_sha256,
                expected_staging_root=stage_canonical,
            )
            final = releases / manifest_sha256
            if final.exists() or final.is_symlink():
                return load_verified_raster_text_trained_paddle_bundle_v1(
                    document_preview_root=root, manifest_sha256=manifest_sha256
                )
            try:
                os.rename(stage, final)
            except OSError:
                if final.exists() and not final.is_symlink():
                    return load_verified_raster_text_trained_paddle_bundle_v1(
                        document_preview_root=root, manifest_sha256=manifest_sha256
                    )
                raise RasterTextTrainedPaddleBundleError(
                    "trained_paddle_bundle_publish_failed"
                ) from None
            return load_verified_raster_text_trained_paddle_bundle_v1(
                document_preview_root=root, manifest_sha256=manifest_sha256
            )
        except RasterTextTrainedPaddleBundleError:
            raise
        except (OSError, ValueError):
            raise RasterTextTrainedPaddleBundleError(
                "trained_paddle_bundle_materialize_failed"
            ) from None
        finally:
            if stage_canonical.exists() or stage_canonical.is_symlink():
                _remove_owned_stage(root, stage_canonical)


def prepare_raster_text_trained_runtime_profile_v12(
    sidecar_bundle: VerifiedRasterTextDirectPaddleBundleV1,
    combined_bundle: VerifiedRasterTextTrainedPaddleBundleV1,
    *,
    pdfium_version: str,
    minimum_page_confidence: float,
) -> PreparedRasterTextTrainedRuntimeProfileV12:
    """Create or verify the source-attested V12 strip2 runtime profile."""

    from customs_ai_agent.service.raster_text_preview_runtime import (
        RasterTextOfflineProfileV12,
        raster_text_profile_fingerprint_sha256,
    )

    if (
        not isinstance(pdfium_version, str)
        or not pdfium_version
        or len(pdfium_version) > 128
        or isinstance(minimum_page_confidence, bool)
        or not isinstance(minimum_page_confidence, (int, float))
        or not 0 <= float(minimum_page_confidence) <= 1
    ):
        raise RasterTextTrainedPaddleBundleError("trained_paddle_bundle_runtime_profile_invalid")
    try:
        sidecar = load_verified_raster_text_direct_paddle_bundle_v1(
            document_preview_root=sidecar_bundle.document_preview_root,
            manifest_sha256=sidecar_bundle.manifest_sha256,
        )
        combined = load_verified_raster_text_trained_paddle_bundle_v1(
            document_preview_root=combined_bundle.document_preview_root,
            manifest_sha256=combined_bundle.manifest_sha256,
        )
        if (
            not _same_path(sidecar.document_preview_root, combined.document_preview_root)
            or sidecar.bundle_root_relative_path != sidecar_bundle.bundle_root_relative_path
            or combined.bundle_root_relative_path != combined_bundle.bundle_root_relative_path
        ):
            raise RasterTextTrainedPaddleBundleError(
                "trained_paddle_bundle_runtime_profile_invalid"
            )
        sidecar_manifest = sidecar.manifest
        combined_manifest = combined.manifest
        profile_payload: dict[str, object] = {
            "schema_version": "12.0",
            "engine_kind": (
                "direct_paddle_source_attested_text_harmonic_recall_native960_lowbox_strip2"
            ),
            "runtime_api": "paddle-inference-static-v1",
            "extractor_revision": (
                f"{sidecar_manifest.extractor_revision}:"
                "source-attested-rec-v12-native960-lowbox-strip2"
            ),
            "bundle_root_relative_path": sidecar.bundle_root_relative_path,
            "bundle_manifest_sha256": sidecar.manifest_sha256,
            "combined_model_bundle_root_relative_path": combined.bundle_root_relative_path,
            "combined_model_bundle_manifest_sha256": combined.manifest_sha256,
            "source_detector_bundle_manifest_sha256": (
                combined_manifest.source_detector_bundle_manifest_sha256
            ),
            "source_detector_model_inventory_sha256": (
                combined_manifest.source_detector_model_inventory_sha256
            ),
            "source_detector_model_snapshot_sha256": (
                combined_manifest.source_detector_model_snapshot_sha256
            ),
            "source_recognizer_release_sha256": (
                combined_manifest.source_recognizer_release_sha256
            ),
            "source_recognizer_receipt_sha256": (
                combined_manifest.source_recognizer_receipt_sha256
            ),
            "source_recognizer_training_intent_sha256": (
                combined_manifest.source_recognizer_training_intent_sha256
            ),
            "source_recognizer_character_dictionary_sha256": (
                combined_manifest.source_recognizer_character_dictionary_sha256
            ),
            "source_recognizer_package_inventory_sha256": (
                combined_manifest.source_recognizer_package_inventory_sha256
            ),
            "pdfium_version": pdfium_version,
            "raster_dpi": sidecar_manifest.raster_dpi,
            "sidecar_timeout_seconds": 30,
            "cpu_threads": 6,
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
                combined_manifest.local_model_root_relative_path
            ),
            "model_inventory_path": combined.bundle_relative_path(
                combined_manifest.model_inventory_relative_path
            ),
            "model_inventory_sha256": combined.file_sha256(
                combined_manifest.model_inventory_relative_path
            ),
            "text_detection_model_name": _DETECTOR_NAME,
            "text_detection_model_directory": "text-det",
            "text_recognition_model_name": _RECOGNIZER_NAME,
            "text_recognition_model_directory": "text-rec",
        }
        fingerprint = raster_text_profile_fingerprint_sha256(profile_payload)
        profile = RasterTextOfflineProfileV12.model_validate(
            {
                **profile_payload,
                "profile_fingerprint_sha256": fingerprint,
                "raster_text_profile_revision": (f"synthetic-raster-text-v1:{fingerprint[:24]}"),
            }
        )
    except (
        OSError,
        TypeError,
        ValidationError,
        ValueError,
        RasterTextDirectPaddleBundleError,
        RasterTextTrainedPaddleBundleError,
    ):
        raise RasterTextTrainedPaddleBundleError(
            "trained_paddle_bundle_runtime_profile_invalid"
        ) from None
    suffix = profile.raster_text_profile_revision.rsplit(":", 1)[1]
    relative_path = f"{_RUNTIME_PROFILE_COLLECTION}/{suffix}/profile.json"
    try:
        directory = _ensure_directory(
            combined.document_preview_root,
            f"{_RUNTIME_PROFILE_COLLECTION}/{suffix}",
        )
        payload = _write_json_create_or_verify(
            directory / "profile.json",
            profile.model_dump(mode="json", warnings=False),
            code="trained_paddle_bundle_runtime_profile_write_failed",
        )
    except RasterTextTrainedPaddleBundleError:
        raise RasterTextTrainedPaddleBundleError(
            "trained_paddle_bundle_runtime_profile_write_failed"
        ) from None
    return PreparedRasterTextTrainedRuntimeProfileV12(
        raster_text_profile_revision=profile.raster_text_profile_revision,
        runtime_configuration_relative_path=relative_path,
        runtime_configuration_sha256=_sha256_bytes(payload),
        minimum_page_confidence=profile.minimum_page_confidence,
    )


__all__ = [
    "PreparedRasterTextTrainedRuntimeProfileV12",
    "RasterTextTrainedPaddleBundleError",
    "RasterTextTrainedPaddleBundleManifestV1",
    "TrainedPaddleBundleFileV1",
    "VerifiedRasterTextTrainedPaddleBundleV1",
    "load_verified_raster_text_trained_paddle_bundle_v1",
    "materialize_verified_raster_text_trained_paddle_bundle_v1",
    "prepare_raster_text_trained_runtime_profile_v12",
]
