"""Build and attest the local PP-OCRv5 mobile-detector model bundle.

This module deliberately owns only the mobile model bytes.  The separately
verified direct-Paddle microbundle continues to own the interpreter, package
closure and first-party sidecar.  Each V3--V11/V13--V15 profile binds both
content-addressed release hashes, so a mobile candidate cannot silently
consume V2 server models or an arbitrary Paddle cache.

No URL is opened here.  The only source is one fixed, hash-bound local
PaddleOCR3 inventory and the eight files it declares.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
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
    DirectPaddleBundleFileV1,
    RasterTextDirectPaddleBundleError,
    VerifiedRasterTextDirectPaddleBundleV1,
    _canonical_json,
    _local_path_identity,
    _operational_path,
    _relative_path,
    _root,
    _safe_path,
    _sha256_bytes,
    _sha256_regular_file,
    _tree_manifest_sha256,
    _write_create_only,
    load_verified_raster_text_direct_paddle_bundle_v1,
)


_SHA256 = r"^[a-f0-9]{64}$"
_MAX_JSON_BYTES = 1024 * 1024
_MAX_FILE_BYTES = 1024 * 1024 * 1024
_MAX_FILES = 64
_MOBILE_COLLECTION = "raster-text-direct-mobile-bundles-v1"
_MOBILE_RELEASES = f"{_MOBILE_COLLECTION}/releases"
_MOBILE_BUNDLE_ID = "raster_text_direct_paddle_mobile_model_bundle_v1"
_MOBILE_LOCK_FILE = ".raster-text-direct-mobile-bundle-v1.lock"
_MOBILE_SOURCE_INVENTORY_RELATIVE_PATH = (
    "model-inventories/57d566b507b7fff67ef0cb3eb605671d862facb2f8aa4765f74bab546baf130d.json"
)
_MOBILE_SOURCE_INVENTORY_SHA256 = "57d566b507b7fff67ef0cb3eb605671d862facb2f8aa4765f74bab546baf130d"
_MOBILE_SOURCE_MODEL_ROOT = "models/paddleocr3"
_MOBILE_MODEL_INVENTORY_RELATIVE_PATH = "inventories/direct-paddle-mobile-models.json"
_MOBILE_MODEL_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("models/paddleocr3/det/config.json", "text-det/config.json"),
    ("models/paddleocr3/det/inference.json", "text-det/inference.json"),
    ("models/paddleocr3/det/inference.pdiparams", "text-det/inference.pdiparams"),
    ("models/paddleocr3/det/inference.yml", "text-det/inference.yml"),
    ("models/paddleocr3/rec/config.json", "text-rec/config.json"),
    ("models/paddleocr3/rec/inference.json", "text-rec/inference.json"),
    ("models/paddleocr3/rec/inference.pdiparams", "text-rec/inference.pdiparams"),
    ("models/paddleocr3/rec/inference.yml", "text-rec/inference.yml"),
)
_MOBILE_TARGET_PATHS = tuple(target for _source, target in _MOBILE_MODEL_ARTIFACTS)
_MOBILE_DETECTOR = "PP-OCRv5_mobile_det"
_MOBILE_RECOGNIZER = "en_PP-OCRv5_mobile_rec"


class RasterTextMobilePaddleBundleError(RuntimeError):
    """Content-free error for the isolated mobile-model candidate path."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RasterTextMobilePaddleBundleManifestV1(_WireModel):
    """Exhaustive, independently verified mobile-model bundle manifest."""

    schema_version: str
    bundle_id: str
    source_model_inventory_relative_path: str
    source_model_inventory_sha256: str = Field(pattern=_SHA256)
    source_model_snapshot_sha256: str = Field(pattern=_SHA256)
    text_detection_model_name: str
    text_recognition_model_name: str
    local_model_root_relative_path: str
    model_inventory_relative_path: str
    model_root_tree_sha256: str = Field(pattern=_SHA256)
    files: list[DirectPaddleBundleFileV1] = Field(min_length=1, max_length=_MAX_FILES)

    @field_validator(
        "source_model_inventory_relative_path",
        "local_model_root_relative_path",
        "model_inventory_relative_path",
    )
    @classmethod
    def _valid_paths(cls, value: str) -> str:
        return _relative_path(value)

    @model_validator(mode="after")
    def _shape(self) -> "RasterTextMobilePaddleBundleManifestV1":
        if (
            self.schema_version != "1.0"
            or self.bundle_id != _MOBILE_BUNDLE_ID
            or self.source_model_inventory_relative_path
            != f"model-inventories/{self.source_model_inventory_sha256}.json"
            or self.text_detection_model_name != _MOBILE_DETECTOR
            or self.text_recognition_model_name != _MOBILE_RECOGNIZER
            or self.local_model_root_relative_path != "models"
            or self.model_inventory_relative_path != _MOBILE_MODEL_INVENTORY_RELATIVE_PATH
        ):
            raise ValueError("mobile Paddle bundle manifest shape is invalid")
        paths = [item.path for item in self.files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("mobile Paddle bundle file manifest is invalid")
        expected = {
            _MOBILE_MODEL_INVENTORY_RELATIVE_PATH,
            *(f"models/{target}" for target in _MOBILE_TARGET_PATHS),
        }
        if set(paths) != expected:
            raise ValueError("mobile Paddle bundle file manifest is invalid")
        return self


@dataclass(frozen=True, slots=True)
class MobilePaddleModelSourceV1:
    document_preview_root: Path
    source_inventory_relative_path: str
    source_inventory_sha256: str
    source_model_snapshot_sha256: str
    artifacts: tuple[DirectPaddleBundleFileV1, ...]


@dataclass(frozen=True, slots=True)
class VerifiedRasterTextDirectMobilePaddleBundleV1:
    document_preview_root: Path
    bundle_root: Path
    bundle_root_relative_path: str
    manifest: RasterTextMobilePaddleBundleManifestV1
    manifest_sha256: str
    file_sha256_by_relative_path: Mapping[str, str]

    def bundle_relative_path(self, path: str) -> str:
        _relative_path(path)
        return f"{self.bundle_root_relative_path}/{path}"

    def file_sha256(self, path: str) -> str:
        try:
            return self.file_sha256_by_relative_path[path]
        except KeyError:
            raise RasterTextMobilePaddleBundleError(
                "mobile_paddle_bundle_manifest_invalid"
            ) from None


@dataclass(frozen=True, slots=True)
class PreparedRasterTextDirectMobileRuntimeProfileV3:
    raster_text_profile_revision: str
    runtime_configuration_relative_path: str
    runtime_configuration_sha256: str
    minimum_page_confidence: float


@dataclass(frozen=True, slots=True)
class PreparedRasterTextDirectMobileRuntimeProfileV4:
    """Content-addressed V4 harmonic-order candidate profile receipt."""

    raster_text_profile_revision: str
    runtime_configuration_relative_path: str
    runtime_configuration_sha256: str
    minimum_page_confidence: float


@dataclass(frozen=True, slots=True)
class PreparedRasterTextDirectMobileRuntimeProfileV5:
    """Content-addressed V5 harmonic-recall candidate profile receipt."""

    raster_text_profile_revision: str
    runtime_configuration_relative_path: str
    runtime_configuration_sha256: str
    minimum_page_confidence: float


@dataclass(frozen=True, slots=True)
class PreparedRasterTextDirectMobileRuntimeProfileV6:
    """Content-addressed V6 1024-side candidate profile receipt."""

    raster_text_profile_revision: str
    runtime_configuration_relative_path: str
    runtime_configuration_sha256: str
    minimum_page_confidence: float


@dataclass(frozen=True, slots=True)
class PreparedRasterTextDirectMobileRuntimeProfileV7:
    """Content-addressed V7 1024-side low-box candidate profile receipt."""

    raster_text_profile_revision: str
    runtime_configuration_relative_path: str
    runtime_configuration_sha256: str
    minimum_page_confidence: float


@dataclass(frozen=True, slots=True)
class PreparedRasterTextDirectMobileRuntimeProfileV8:
    """Content-addressed V8 native-960 low-box candidate profile receipt."""

    raster_text_profile_revision: str
    runtime_configuration_relative_path: str
    runtime_configuration_sha256: str
    minimum_page_confidence: float


@dataclass(frozen=True, slots=True)
class PreparedRasterTextDirectMobileRuntimeProfileV9:
    """Content-addressed V9 native-960 two-strip candidate profile receipt."""

    raster_text_profile_revision: str
    runtime_configuration_relative_path: str
    runtime_configuration_sha256: str
    minimum_page_confidence: float


@dataclass(frozen=True, slots=True)
class PreparedRasterTextDirectMobileRuntimeProfileV10:
    """Content-addressed V10 native-960 grid2 candidate profile receipt."""

    raster_text_profile_revision: str
    runtime_configuration_relative_path: str
    runtime_configuration_sha256: str
    minimum_page_confidence: float


@dataclass(frozen=True, slots=True)
class PreparedRasterTextDirectMobileRuntimeProfileV11:
    """Content-addressed V11 native-960 guarded-grid candidate receipt."""

    raster_text_profile_revision: str
    runtime_configuration_relative_path: str
    runtime_configuration_sha256: str
    minimum_page_confidence: float


@dataclass(frozen=True, slots=True)
class PreparedRasterTextDirectMobileRuntimeProfileV13:
    """Content-addressed V13 public-mobile semantic-repair receipt."""

    raster_text_profile_revision: str
    runtime_configuration_relative_path: str
    runtime_configuration_sha256: str
    minimum_page_confidence: float


@dataclass(frozen=True, slots=True)
class PreparedRasterTextDirectMobileRuntimeProfileV14:
    """Content-addressed V14 public-mobile semantic-repair receipt."""

    raster_text_profile_revision: str
    runtime_configuration_relative_path: str
    runtime_configuration_sha256: str
    minimum_page_confidence: float


@dataclass(frozen=True, slots=True)
class PreparedRasterTextDirectMobileRuntimeProfileV15:
    """Content-addressed V15 bounded-oneDNN candidate receipt."""

    raster_text_profile_revision: str
    runtime_configuration_relative_path: str
    runtime_configuration_sha256: str
    minimum_page_confidence: float


@dataclass(frozen=True, slots=True)
class PreparedRasterTextDirectMobileRuntimeProfileV16:
    """Content-addressed V16 cubic-recognition candidate receipt."""

    raster_text_profile_revision: str
    runtime_configuration_relative_path: str
    runtime_configuration_sha256: str
    minimum_page_confidence: float


@dataclass(frozen=True, slots=True)
class PreparedRasterTextDirectMobileRuntimeProfileV17:
    """Content-addressed V17 padded-recognition candidate receipt."""

    raster_text_profile_revision: str
    runtime_configuration_relative_path: str
    runtime_configuration_sha256: str
    minimum_page_confidence: float


def _canonical_bytes(value: object) -> bytes:
    return _canonical_json(value)


def _mobile_root(path: Path) -> Path:
    try:
        return _root(path)
    except RasterTextDirectPaddleBundleError:
        raise RasterTextMobilePaddleBundleError("mobile_paddle_bundle_root_invalid") from None


def _sha256_file(path: Path, *, code: str) -> tuple[str, int]:
    try:
        return _sha256_regular_file(path, code=code)
    except RasterTextDirectPaddleBundleError:
        raise RasterTextMobilePaddleBundleError(code) from None


def _safe_mobile_path(root: Path, relative: str, *, directory: bool = False) -> Path:
    try:
        return _safe_path(root, relative, directory=directory)
    except RasterTextDirectPaddleBundleError:
        raise RasterTextMobilePaddleBundleError("mobile_paddle_bundle_path_invalid") from None


def _read_canonical_object(path: Path, *, code: str) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        if not raw or len(raw) > _MAX_JSON_BYTES:
            raise OSError
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict) or raw != _canonical_bytes(value):
            raise ValueError
        return value, raw
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        raise RasterTextMobilePaddleBundleError(code) from None


def _read_fixed_source_inventory(path: Path) -> tuple[dict[str, Any], bytes]:
    """Read the pre-existing newline-terminated PaddleOCR3 inventory exactly.

    The fixed source inventory predates the no-newline bundle receipts.  Its
    published SHA includes the single final LF, so accepting the newer writer
    form would make the source binding subtly different from the inventory
    that was actually audited.
    """

    try:
        raw = path.read_bytes()
        if not raw or len(raw) > _MAX_JSON_BYTES:
            raise OSError
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict) or raw != _canonical_bytes(value) + b"\n":
            raise ValueError
        return value, raw
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        raise RasterTextMobilePaddleBundleError(
            "mobile_paddle_bundle_source_inventory_invalid"
        ) from None


def _release_relative_path(manifest_sha256: str) -> str:
    if len(manifest_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in manifest_sha256
    ):
        raise RasterTextMobilePaddleBundleError("mobile_paddle_bundle_manifest_invalid")
    return f"{_MOBILE_RELEASES}/{manifest_sha256}"


def _source_snapshot(artifacts: Sequence[DirectPaddleBundleFileV1]) -> str:
    return hashlib.sha256(
        _canonical_bytes(
            {
                "artifacts": [
                    {
                        "relative_path": item.path,
                        "sha256": item.sha256,
                        "byte_size": item.byte_size,
                    }
                    for item in artifacts
                ]
            }
        )
    ).hexdigest()


def _verify_model_identifier(path: Path, *, expected: str) -> None:
    """Accept the two known PP-OCR config encodings, and nothing else.

    The fixed PP-OCRv5 mobile artifacts record ``model_name`` below
    ``Global``.  Some older local inventories used a top-level spelling, so
    preserve that compatibility only when every declared spelling agrees.
    The inventory hash and per-file hash still bind the actual model bytes;
    this check prevents an accidentally swapped detector/recognizer pair.
    """

    try:
        raw = path.read_bytes()
        if not raw or len(raw) > _MAX_JSON_BYTES:
            raise OSError
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict):
            raise ValueError
        identifiers: set[str] = set()
        top_level = value.get("model_name")
        if isinstance(top_level, str):
            identifiers.add(top_level)
        global_configuration = value.get("Global")
        if isinstance(global_configuration, dict):
            nested = global_configuration.get("model_name")
            if isinstance(nested, str):
                identifiers.add(nested)
        if identifiers != {expected}:
            raise ValueError
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        raise RasterTextMobilePaddleBundleError(
            "mobile_paddle_bundle_source_model_invalid"
        ) from None


def verify_local_paddleocr3_mobile_model_source_v1(
    *, document_preview_root: Path
) -> MobilePaddleModelSourceV1:
    """Attest the fixed local PaddleOCR3 inventory and all eight model bytes."""

    root = _mobile_root(document_preview_root)
    inventory_path = _safe_mobile_path(root, _MOBILE_SOURCE_INVENTORY_RELATIVE_PATH)
    inventory, raw = _read_fixed_source_inventory(inventory_path)
    if hashlib.sha256(raw).hexdigest() != _MOBILE_SOURCE_INVENTORY_SHA256:
        raise RasterTextMobilePaddleBundleError("mobile_paddle_bundle_source_inventory_invalid")
    required = {"schema_version", "engine_kind", "runtime_api", "packages", "artifacts"}
    if (
        set(inventory) != required
        or inventory.get("schema_version") != "1.0"
        or inventory.get("engine_kind") != "paddleocr"
        or inventory.get("runtime_api") != "paddleocr3-paddlex-3.2"
        or not isinstance(inventory.get("packages"), list)
        or not isinstance(inventory.get("artifacts"), list)
        or len(inventory["artifacts"]) != len(_MOBILE_MODEL_ARTIFACTS)
    ):
        raise RasterTextMobilePaddleBundleError("mobile_paddle_bundle_source_inventory_invalid")
    normalized: list[DirectPaddleBundleFileV1] = []
    for (source_relative, target_relative), item in zip(
        _MOBILE_MODEL_ARTIFACTS, inventory["artifacts"], strict=True
    ):
        if (
            not isinstance(item, dict)
            or set(item)
            != {
                "artifact_id",
                "byte_size",
                "license_id",
                "relative_path",
                "sha256",
                "source_uri",
            }
            or item.get("relative_path") != source_relative
            or not isinstance(item.get("sha256"), str)
            or not isinstance(item.get("byte_size"), int)
            or isinstance(item["byte_size"], bool)
            or item["byte_size"] < 0
        ):
            raise RasterTextMobilePaddleBundleError("mobile_paddle_bundle_source_inventory_invalid")
        source_path = _safe_mobile_path(root, source_relative)
        digest, byte_size = _sha256_file(
            source_path, code="mobile_paddle_bundle_source_model_invalid"
        )
        if digest != item["sha256"] or byte_size != item["byte_size"]:
            raise RasterTextMobilePaddleBundleError("mobile_paddle_bundle_source_model_invalid")
        normalized.append(
            DirectPaddleBundleFileV1(
                path=target_relative,
                sha256=digest,
                byte_size=byte_size,
            )
        )
    if tuple(item.path for item in normalized) != _MOBILE_TARGET_PATHS:
        raise RasterTextMobilePaddleBundleError("mobile_paddle_bundle_source_inventory_invalid")
    _verify_model_identifier(
        _safe_mobile_path(root, "models/paddleocr3/det/config.json"), expected=_MOBILE_DETECTOR
    )
    _verify_model_identifier(
        _safe_mobile_path(root, "models/paddleocr3/rec/config.json"), expected=_MOBILE_RECOGNIZER
    )
    source_root = _safe_mobile_path(root, _MOBILE_SOURCE_MODEL_ROOT, directory=True)
    if source_root.is_symlink():
        raise RasterTextMobilePaddleBundleError("mobile_paddle_bundle_source_model_invalid")
    artifacts = tuple(normalized)
    return MobilePaddleModelSourceV1(
        document_preview_root=root,
        source_inventory_relative_path=_MOBILE_SOURCE_INVENTORY_RELATIVE_PATH,
        source_inventory_sha256=hashlib.sha256(raw).hexdigest(),
        source_model_snapshot_sha256=_source_snapshot(artifacts),
        artifacts=artifacts,
    )


def _write_stage_file(
    *, stage: Path, relative: str, source: Path, expected: DirectPaddleBundleFileV1
) -> None:
    try:
        target = stage.joinpath(*PurePosixPath(relative).parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            raise OSError
        shutil.copyfile(source, target)
        digest, byte_size = _sha256_file(target, code="mobile_paddle_bundle_stage_copy_failed")
        if digest != expected.sha256 or byte_size != expected.byte_size:
            raise OSError
    except RasterTextMobilePaddleBundleError:
        raise
    except OSError:
        raise RasterTextMobilePaddleBundleError("mobile_paddle_bundle_stage_copy_failed") from None


def _write_mobile_json(*, path: Path, value: object, code: str) -> bytes:
    """Use the direct builder's fsync writer without leaking its error type."""

    try:
        return _write_create_only(path, value, code=code)
    except RasterTextDirectPaddleBundleError:
        raise RasterTextMobilePaddleBundleError(code) from None


def _stage_files(stage: Path) -> list[DirectPaddleBundleFileV1]:
    try:
        files: list[DirectPaddleBundleFileV1] = []
        for path in sorted(stage.rglob("*"), key=lambda item: item.as_posix()):
            if path.is_symlink():
                raise OSError
            if path.is_file():
                relative = path.relative_to(stage).as_posix()
                digest, byte_size = _sha256_file(
                    path, code="mobile_paddle_bundle_stage_write_failed"
                )
                files.append(
                    DirectPaddleBundleFileV1(path=relative, sha256=digest, byte_size=byte_size)
                )
            elif not path.is_dir():
                raise OSError
        if not files:
            raise OSError
        return files
    except RasterTextMobilePaddleBundleError:
        raise
    except (OSError, ValueError):
        raise RasterTextMobilePaddleBundleError("mobile_paddle_bundle_stage_write_failed") from None


def _mobile_model_inventory(source: MobilePaddleModelSourceV1) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "inventory_kind": "raster_text_direct_paddle_mobile_models",
        "source_model_inventory_relative_path": source.source_inventory_relative_path,
        "source_model_inventory_sha256": source.source_inventory_sha256,
        "source_model_snapshot_sha256": source.source_model_snapshot_sha256,
        "text_detection_model_name": _MOBILE_DETECTOR,
        "text_recognition_model_name": _MOBILE_RECOGNIZER,
        "artifacts": [
            {
                "relative_path": item.path,
                "sha256": item.sha256,
                "byte_size": item.byte_size,
            }
            for item in source.artifacts
        ],
    }


def _build_stage(*, stage: Path, source: MobilePaddleModelSourceV1) -> str:
    for source_relative, target_relative in _MOBILE_MODEL_ARTIFACTS:
        artifact = next(item for item in source.artifacts if item.path == target_relative)
        source_path = _safe_mobile_path(source.document_preview_root, source_relative)
        _write_stage_file(
            stage=stage,
            relative=f"models/{target_relative}",
            source=source_path,
            expected=artifact,
        )
    _write_mobile_json(
        path=stage / _MOBILE_MODEL_INVENTORY_RELATIVE_PATH,
        value=_mobile_model_inventory(source),
        code="mobile_paddle_bundle_stage_write_failed",
    )
    files = _stage_files(stage)
    manifest = RasterTextMobilePaddleBundleManifestV1(
        schema_version="1.0",
        bundle_id=_MOBILE_BUNDLE_ID,
        source_model_inventory_relative_path=source.source_inventory_relative_path,
        source_model_inventory_sha256=source.source_inventory_sha256,
        source_model_snapshot_sha256=source.source_model_snapshot_sha256,
        text_detection_model_name=_MOBILE_DETECTOR,
        text_recognition_model_name=_MOBILE_RECOGNIZER,
        local_model_root_relative_path="models",
        model_inventory_relative_path=_MOBILE_MODEL_INVENTORY_RELATIVE_PATH,
        model_root_tree_sha256=_tree_manifest_sha256(list(source.artifacts)),
        files=files,
    )
    payload = _write_mobile_json(
        path=stage / "manifest.json",
        value=manifest.model_dump(mode="json", warnings=False),
        code="mobile_paddle_bundle_stage_write_failed",
    )
    return _sha256_bytes(payload)


def _load_verified_bundle_from_root(
    *,
    root: Path,
    bundle_root: Path,
    expected_manifest_sha256: str,
    allow_staging: bool = False,
) -> VerifiedRasterTextDirectMobilePaddleBundleV1:
    try:
        if bundle_root.is_symlink() or not bundle_root.is_dir():
            raise OSError
        manifest_path = bundle_root / "manifest.json"
        raw_value, raw = _read_canonical_object(
            manifest_path, code="mobile_paddle_bundle_manifest_invalid"
        )
        if hashlib.sha256(raw).hexdigest() != expected_manifest_sha256:
            raise RasterTextMobilePaddleBundleError("mobile_paddle_bundle_manifest_invalid")
        manifest = RasterTextMobilePaddleBundleManifestV1.model_validate(raw_value)
        if raw != _canonical_bytes(manifest.model_dump(mode="json", warnings=False)):
            raise RasterTextMobilePaddleBundleError("mobile_paddle_bundle_manifest_invalid")
        records = {item.path: item for item in manifest.files}
        actual = _stage_files_without_manifest(bundle_root)
        if set(actual) != set(records):
            raise RasterTextMobilePaddleBundleError("mobile_paddle_bundle_manifest_invalid")
        for relative, record in records.items():
            digest, byte_size = _sha256_file(
                bundle_root.joinpath(*PurePosixPath(relative).parts),
                code="mobile_paddle_bundle_manifest_invalid",
            )
            if digest != record.sha256 or byte_size != record.byte_size:
                raise RasterTextMobilePaddleBundleError("mobile_paddle_bundle_manifest_invalid")
        model_root = _safe_mobile_path(
            bundle_root, manifest.local_model_root_relative_path, directory=True
        )
        expected_model_paths = set(_MOBILE_TARGET_PATHS)
        actual_model_paths = _regular_tree_files(model_root)
        if set(actual_model_paths) != expected_model_paths:
            raise RasterTextMobilePaddleBundleError("mobile_paddle_bundle_manifest_invalid")
        model_records = [
            DirectPaddleBundleFileV1(
                path=relative,
                sha256=records[f"models/{relative}"].sha256,
                byte_size=records[f"models/{relative}"].byte_size,
            )
            for relative in _MOBILE_TARGET_PATHS
        ]
        if _tree_manifest_sha256(model_records) != manifest.model_root_tree_sha256:
            raise RasterTextMobilePaddleBundleError("mobile_paddle_bundle_manifest_invalid")
        inventory_path = _safe_mobile_path(bundle_root, manifest.model_inventory_relative_path)
        inventory, inventory_raw = _read_canonical_object(
            inventory_path, code="mobile_paddle_bundle_manifest_invalid"
        )
        inventory_record = records[manifest.model_inventory_relative_path]
        if (
            hashlib.sha256(inventory_raw).hexdigest() != inventory_record.sha256
            or len(inventory_raw) != inventory_record.byte_size
            or set(inventory)
            != {
                "schema_version",
                "inventory_kind",
                "source_model_inventory_relative_path",
                "source_model_inventory_sha256",
                "source_model_snapshot_sha256",
                "text_detection_model_name",
                "text_recognition_model_name",
                "artifacts",
            }
            or inventory.get("schema_version") != "1.0"
            or inventory.get("inventory_kind") != "raster_text_direct_paddle_mobile_models"
            or inventory.get("source_model_inventory_relative_path")
            != manifest.source_model_inventory_relative_path
            or inventory.get("source_model_inventory_sha256")
            != manifest.source_model_inventory_sha256
            or inventory.get("source_model_snapshot_sha256")
            != manifest.source_model_snapshot_sha256
            or inventory.get("text_detection_model_name") != _MOBILE_DETECTOR
            or inventory.get("text_recognition_model_name") != _MOBILE_RECOGNIZER
            or not isinstance(inventory.get("artifacts"), list)
        ):
            raise RasterTextMobilePaddleBundleError("mobile_paddle_bundle_manifest_invalid")
        inventory_records: list[DirectPaddleBundleFileV1] = []
        for item in inventory["artifacts"]:
            if (
                not isinstance(item, dict)
                or set(item) != {"relative_path", "sha256", "byte_size"}
                or not isinstance(item.get("relative_path"), str)
                or not isinstance(item.get("sha256"), str)
                or isinstance(item.get("byte_size"), bool)
                or not isinstance(item.get("byte_size"), int)
            ):
                raise RasterTextMobilePaddleBundleError("mobile_paddle_bundle_manifest_invalid")
            inventory_records.append(
                DirectPaddleBundleFileV1(
                    path=item["relative_path"],
                    sha256=item["sha256"],
                    byte_size=item["byte_size"],
                )
            )
        if tuple(item.path for item in inventory_records) != _MOBILE_TARGET_PATHS:
            raise RasterTextMobilePaddleBundleError("mobile_paddle_bundle_manifest_invalid")
        if tuple(inventory_records) != tuple(model_records):
            raise RasterTextMobilePaddleBundleError("mobile_paddle_bundle_manifest_invalid")
        if allow_staging:
            # On Windows the actual file operations can use an 8.3 spelling
            # while ``root`` keeps its canonical path.  Compare filesystem
            # identities instead of Path.relative_to(), then permit only our
            # direct child staging directory.  This is validation before the
            # single atomic rename; the returned relative name is never
            # published or used as a runtime path.
            staging_name = os.path.basename(_local_path_identity(bundle_root))
            if _local_path_identity(bundle_root.parent) != _local_path_identity(
                root
            ) or not staging_name.startswith(f".{_MOBILE_COLLECTION}.staging-"):
                raise RasterTextMobilePaddleBundleError("mobile_paddle_bundle_manifest_invalid")
            relative = staging_name
        else:
            expected_relative = _release_relative_path(expected_manifest_sha256)
            expected_root = _safe_mobile_path(root, expected_relative, directory=True)
            if _local_path_identity(bundle_root) != _local_path_identity(expected_root):
                raise RasterTextMobilePaddleBundleError("mobile_paddle_bundle_manifest_invalid")
            relative = expected_relative
        return VerifiedRasterTextDirectMobilePaddleBundleV1(
            document_preview_root=root,
            bundle_root=bundle_root,
            bundle_root_relative_path=relative,
            manifest=manifest,
            manifest_sha256=expected_manifest_sha256,
            file_sha256_by_relative_path={
                relative_path: record.sha256 for relative_path, record in records.items()
            },
        )
    except RasterTextMobilePaddleBundleError:
        raise
    except (OSError, ValidationError, ValueError, TypeError):
        raise RasterTextMobilePaddleBundleError("mobile_paddle_bundle_manifest_invalid") from None


def _stage_files_without_manifest(root: Path) -> dict[str, Path]:
    try:
        files: dict[str, Path] = {}
        for path in root.rglob("*"):
            if path.is_symlink():
                raise OSError
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                if relative == "manifest.json":
                    continue
                if relative in files:
                    raise OSError
                files[relative] = path
            elif not path.is_dir():
                raise OSError
        return files
    except (OSError, ValueError):
        raise RasterTextMobilePaddleBundleError("mobile_paddle_bundle_manifest_invalid") from None


def _regular_tree_files(root: Path) -> dict[str, Path]:
    try:
        files: dict[str, Path] = {}
        for path in root.rglob("*"):
            if path.is_symlink():
                raise OSError
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                if relative in files:
                    raise OSError
                files[relative] = path
            elif not path.is_dir():
                raise OSError
        return files
    except (OSError, ValueError):
        raise RasterTextMobilePaddleBundleError("mobile_paddle_bundle_manifest_invalid") from None


def load_verified_raster_text_direct_mobile_paddle_bundle_v1(
    *, document_preview_root: Path, manifest_sha256: str
) -> VerifiedRasterTextDirectMobilePaddleBundleV1:
    root = _mobile_root(document_preview_root)
    relative = _release_relative_path(manifest_sha256)
    bundle_root = _safe_mobile_path(root, relative, directory=True)
    return _load_verified_bundle_from_root(
        root=root, bundle_root=bundle_root, expected_manifest_sha256=manifest_sha256
    )


def _ensure_directory(root: Path, relative: str) -> Path:
    try:
        target = root.joinpath(*PurePosixPath(_relative_path(relative)).parts)
        target.mkdir(parents=True, exist_ok=True)
        if target.is_symlink() or not target.is_dir():
            raise OSError
        return target
    except (OSError, ValueError):
        raise RasterTextMobilePaddleBundleError("mobile_paddle_bundle_output_invalid") from None


@contextmanager
def _lock(root: Path) -> Iterator[None]:
    try:
        with exclusive_local_process_lock(root / _MOBILE_LOCK_FILE):
            yield
    except LocalProcessLockBusy:
        raise RasterTextMobilePaddleBundleError("mobile_paddle_bundle_locked") from None
    except LocalProcessLockInvalid:
        raise RasterTextMobilePaddleBundleError("mobile_paddle_bundle_lock_invalid") from None


def _remove_owned_stage(root: Path, stage: Path) -> None:
    try:
        resolved = stage.resolve(strict=True)
        if (
            _local_path_identity(resolved.parent) != _local_path_identity(root)
            or resolved.is_symlink()
            or not resolved.name.startswith(f".{_MOBILE_COLLECTION}.staging-")
        ):
            return
        shutil.rmtree(_operational_path(resolved, force_short=True))
    except (OSError, RasterTextDirectPaddleBundleError):
        return


def materialize_verified_raster_text_direct_mobile_paddle_bundle_v1(
    *, document_preview_root: Path
) -> VerifiedRasterTextDirectMobilePaddleBundleV1:
    """Copy/reuse the fixed mobile model set without package install or network."""

    root = _mobile_root(document_preview_root)
    with _lock(root):
        source = verify_local_paddleocr3_mobile_model_source_v1(document_preview_root=root)
        _ensure_directory(root, _MOBILE_COLLECTION)
        releases = _ensure_directory(root, _MOBILE_RELEASES)
        stage_canonical = root / f".{_MOBILE_COLLECTION}.staging-{uuid.uuid4().hex}"
        stage: Path | None = None
        try:
            if stage_canonical.exists() or stage_canonical.is_symlink():
                raise RasterTextMobilePaddleBundleError("mobile_paddle_bundle_output_invalid")
            stage_canonical.mkdir(mode=0o700)
            stage = _operational_path(stage_canonical, force_short=True)
            manifest_sha256 = _build_stage(stage=stage, source=source)
            staged = _load_verified_bundle_from_root(
                root=root,
                bundle_root=stage,
                expected_manifest_sha256=manifest_sha256,
                allow_staging=True,
            )
            final_relative = _release_relative_path(manifest_sha256)
            final_parent = _operational_path(releases, force_short=True)
            final = final_parent / manifest_sha256
            if final.exists() or final.is_symlink():
                return load_verified_raster_text_direct_mobile_paddle_bundle_v1(
                    document_preview_root=root, manifest_sha256=manifest_sha256
                )
            try:
                os.rename(stage, final)
            except OSError:
                if final.exists() and not final.is_symlink():
                    return load_verified_raster_text_direct_mobile_paddle_bundle_v1(
                        document_preview_root=root, manifest_sha256=manifest_sha256
                    )
                raise RasterTextMobilePaddleBundleError(
                    "mobile_paddle_bundle_publish_failed"
                ) from None
            return VerifiedRasterTextDirectMobilePaddleBundleV1(
                document_preview_root=root,
                bundle_root=final,
                bundle_root_relative_path=final_relative,
                manifest=staged.manifest,
                manifest_sha256=manifest_sha256,
                file_sha256_by_relative_path=staged.file_sha256_by_relative_path,
            )
        except RasterTextMobilePaddleBundleError:
            raise
        except (OSError, ValueError, RasterTextDirectPaddleBundleError):
            raise RasterTextMobilePaddleBundleError(
                "mobile_paddle_bundle_materialize_failed"
            ) from None
        finally:
            if stage_canonical.exists():
                _remove_owned_stage(root, stage_canonical)


def _prepare_raster_text_direct_mobile_runtime_profile(
    sidecar_bundle: VerifiedRasterTextDirectPaddleBundleV1,
    model_bundle: VerifiedRasterTextDirectMobilePaddleBundleV1,
    *,
    pdfium_version: str,
    minimum_page_confidence: float,
    schema_version: Literal[
        "3.0",
        "4.0",
        "5.0",
        "6.0",
        "7.0",
        "8.0",
        "9.0",
        "10.0",
        "11.0",
        "13.0",
        "14.0",
        "15.0",
        "16.0",
        "17.0",
    ],
    engine_kind: Literal[
        "direct_paddle_mobile_text",
        "direct_paddle_mobile_text_harmonic",
        "direct_paddle_mobile_text_harmonic_recall",
        "direct_paddle_mobile_text_1024",
        "direct_paddle_mobile_text_harmonic_recall_1024_lowbox",
        "direct_paddle_mobile_text_harmonic_recall_native960_lowbox",
        "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2",
        "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_grid2",
        "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_guarded_grid2",
        "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_semantic_repair",
        "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_semantic_repair_v2",
        "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_semantic_repair_v2_onednn",
        "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_semantic_repair_v2_onednn_cubic_recognition",
        "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_semantic_repair_v2_onednn_padded_recognition",
    ],
    extractor_revision_suffix: Literal[
        "mobile-det-v3",
        "mobile-det-v4-harmonic",
        "mobile-det-v5-harmonic-recall",
        "mobile-det-v6-1024",
        "mobile-det-v7-1024-lowbox",
        "mobile-det-v8-native960-lowbox",
        "mobile-det-v9-native960-lowbox-strip2",
        "mobile-det-v10-native960-lowbox-grid2",
        "mobile-det-v11-native960-lowbox-guarded-grid2",
        "mobile-det-v13-native960-lowbox-strip2-semantic-repair",
        "mobile-det-v14-native960-lowbox-strip2-semantic-repair-v2",
        "mobile-det-v15-native960-lowbox-strip2-semantic-repair-v2-onednn",
        "mobile-det-v16-native960-lowbox-strip2-semantic-repair-v2-onednn-cubic-recognition",
        "mobile-det-v17-native960-lowbox-strip2-semantic-repair-v2-onednn-padded-recognition",
    ],
    cpu_threads: Literal[4, 6] = 6,
    recognition_batch_size: Literal[8, 16] = 8,
    opencv_threads: Literal[1] = 1,
) -> tuple[str, str, str, float]:
    """Create one immutable V3--V11/V13--V15 profile bound to both local releases."""

    from customs_ai_agent.service.raster_text_preview_runtime import (
        RasterTextOfflineProfileV3,
        RasterTextOfflineProfileV4,
        RasterTextOfflineProfileV5,
        RasterTextOfflineProfileV6,
        RasterTextOfflineProfileV7,
        RasterTextOfflineProfileV8,
        RasterTextOfflineProfileV9,
        RasterTextOfflineProfileV10,
        RasterTextOfflineProfileV11,
        RasterTextOfflineProfileV13,
        RasterTextOfflineProfileV14,
        RasterTextOfflineProfileV15,
        RasterTextOfflineProfileV16,
        RasterTextOfflineProfileV17,
        raster_text_profile_fingerprint_sha256,
    )

    if (
        (schema_version, engine_kind, extractor_revision_suffix)
        not in {
            ("3.0", "direct_paddle_mobile_text", "mobile-det-v3"),
            ("4.0", "direct_paddle_mobile_text_harmonic", "mobile-det-v4-harmonic"),
            (
                "5.0",
                "direct_paddle_mobile_text_harmonic_recall",
                "mobile-det-v5-harmonic-recall",
            ),
            ("6.0", "direct_paddle_mobile_text_1024", "mobile-det-v6-1024"),
            (
                "7.0",
                "direct_paddle_mobile_text_harmonic_recall_1024_lowbox",
                "mobile-det-v7-1024-lowbox",
            ),
            (
                "8.0",
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox",
                "mobile-det-v8-native960-lowbox",
            ),
            (
                "9.0",
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2",
                "mobile-det-v9-native960-lowbox-strip2",
            ),
            (
                "10.0",
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_grid2",
                "mobile-det-v10-native960-lowbox-grid2",
            ),
            (
                "11.0",
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_guarded_grid2",
                "mobile-det-v11-native960-lowbox-guarded-grid2",
            ),
            (
                "13.0",
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_semantic_repair",
                "mobile-det-v13-native960-lowbox-strip2-semantic-repair",
            ),
            (
                "14.0",
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_semantic_repair_v2",
                "mobile-det-v14-native960-lowbox-strip2-semantic-repair-v2",
            ),
            (
                "15.0",
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_semantic_repair_v2_onednn",
                "mobile-det-v15-native960-lowbox-strip2-semantic-repair-v2-onednn",
            ),
            (
                "16.0",
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_semantic_repair_v2_onednn_cubic_recognition",
                "mobile-det-v16-native960-lowbox-strip2-semantic-repair-v2-onednn-cubic-recognition",
            ),
            (
                "17.0",
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_semantic_repair_v2_onednn_padded_recognition",
                "mobile-det-v17-native960-lowbox-strip2-semantic-repair-v2-onednn-padded-recognition",
            ),
        }
        or not isinstance(pdfium_version, str)
        or not pdfium_version
        or len(pdfium_version) > 128
        or isinstance(minimum_page_confidence, bool)
        or not isinstance(minimum_page_confidence, (int, float))
        or not 0 <= float(minimum_page_confidence) <= 1
        or (schema_version not in {"15.0", "16.0", "17.0"} and cpu_threads != 6)
        or (schema_version not in {"15.0", "16.0", "17.0"} and recognition_batch_size != 8)
        or cpu_threads not in {4, 6}
        or recognition_batch_size not in {8, 16}
        or opencv_threads != 1
    ):
        raise RasterTextMobilePaddleBundleError("mobile_paddle_bundle_runtime_profile_invalid")
    try:
        sidecar = load_verified_raster_text_direct_paddle_bundle_v1(
            document_preview_root=sidecar_bundle.document_preview_root,
            manifest_sha256=sidecar_bundle.manifest_sha256,
        )
        mobile = load_verified_raster_text_direct_mobile_paddle_bundle_v1(
            document_preview_root=model_bundle.document_preview_root,
            manifest_sha256=model_bundle.manifest_sha256,
        )
        if (
            _local_path_identity(sidecar.document_preview_root)
            != _local_path_identity(mobile.document_preview_root)
            or sidecar.bundle_root_relative_path != sidecar_bundle.bundle_root_relative_path
            or mobile.bundle_root_relative_path != model_bundle.bundle_root_relative_path
        ):
            raise RasterTextMobilePaddleBundleError("mobile_paddle_bundle_runtime_profile_invalid")
        sidecar_manifest = sidecar.manifest
        mobile_manifest = mobile.manifest
        profile_payload: dict[str, object] = {
            "schema_version": schema_version,
            "engine_kind": engine_kind,
            "runtime_api": "paddle-inference-static-v1",
            "extractor_revision": (
                f"{sidecar_manifest.extractor_revision}:{extractor_revision_suffix}"
            ),
            "bundle_root_relative_path": sidecar.bundle_root_relative_path,
            "bundle_manifest_sha256": sidecar.manifest_sha256,
            "mobile_model_bundle_root_relative_path": mobile.bundle_root_relative_path,
            "mobile_model_bundle_manifest_sha256": mobile.manifest_sha256,
            "source_model_inventory_relative_path": (
                mobile_manifest.source_model_inventory_relative_path
            ),
            "source_model_inventory_sha256": mobile_manifest.source_model_inventory_sha256,
            "pdfium_version": pdfium_version,
            "raster_dpi": sidecar_manifest.raster_dpi,
            "sidecar_timeout_seconds": 30,
            "cpu_threads": cpu_threads if schema_version in {"15.0", "16.0", "17.0"} else 6,
            "text_detection_limit_side_len": (1024 if schema_version in {"6.0", "7.0"} else 960),
            "text_detection_limit_type": "max",
            "text_detection_threshold": 0.3,
            "text_detection_box_threshold": (
                0.3
                if schema_version
                in {
                    "7.0",
                    "8.0",
                    "9.0",
                    "10.0",
                    "11.0",
                    "13.0",
                    "14.0",
                    "15.0",
                    "16.0",
                    "17.0",
                }
                else 0.4
                if schema_version in {"5.0", "6.0"}
                else 0.6
            ),
            "text_detection_unclip_ratio": 1.5,
            "minimum_page_confidence": float(minimum_page_confidence),
            "parent_renderer_package_inventory_path": sidecar.bundle_relative_path(
                sidecar_manifest.parent_renderer_package_inventory_relative_path
            ),
            "parent_renderer_package_inventory_sha256": sidecar.file_sha256(
                sidecar_manifest.parent_renderer_package_inventory_relative_path
            ),
            "parent_renderer_required_packages": sidecar_manifest.parent_renderer_required_packages,
            "sidecar_interpreter_path": sidecar.bundle_relative_path(
                sidecar_manifest.sidecar_interpreter_relative_path
            ),
            "sidecar_interpreter_sha256": sidecar.file_sha256(
                sidecar_manifest.sidecar_interpreter_relative_path
            ),
            "sidecar_interpreter_root": sidecar.bundle_relative_path(
                sidecar_manifest.sidecar_interpreter_root_relative_path
            ),
            "sidecar_interpreter_root_tree_sha256": sidecar_manifest.sidecar_interpreter_root_tree_sha256,
            "sidecar_module_path": sidecar.bundle_relative_path(
                sidecar_manifest.sidecar_module_relative_path
            ),
            "sidecar_module_sha256": sidecar.file_sha256(
                sidecar_manifest.sidecar_module_relative_path
            ),
            "sidecar_module_root": sidecar.bundle_relative_path(
                sidecar_manifest.sidecar_module_root_relative_path
            ),
            "sidecar_module_root_tree_sha256": sidecar_manifest.sidecar_module_root_tree_sha256,
            "sidecar_package_inventory_path": sidecar.bundle_relative_path(
                sidecar_manifest.sidecar_package_inventory_relative_path
            ),
            "sidecar_package_inventory_sha256": sidecar.file_sha256(
                sidecar_manifest.sidecar_package_inventory_relative_path
            ),
            "sidecar_required_packages": sidecar_manifest.sidecar_required_packages,
            "local_model_root": mobile.bundle_relative_path(
                mobile_manifest.local_model_root_relative_path
            ),
            "model_inventory_path": mobile.bundle_relative_path(
                mobile_manifest.model_inventory_relative_path
            ),
            "model_inventory_sha256": mobile.file_sha256(
                mobile_manifest.model_inventory_relative_path
            ),
            "text_detection_model_name": _MOBILE_DETECTOR,
            "text_detection_model_directory": "text-det",
            "text_recognition_model_name": _MOBILE_RECOGNIZER,
            "text_recognition_model_directory": "text-rec",
        }
        if schema_version in {"15.0", "16.0", "17.0"}:
            profile_payload.update(
                {
                    "enable_mkldnn": True,
                    "mkldnn_cache_capacity": 10,
                    "recognition_batch_size": recognition_batch_size,
                    "opencv_threads": opencv_threads,
                    "warmup_shape_policy": "detector-736x960-recognizer-48x320-v1",
                    "onednn_model_policy": "recognizer-only-v1",
                }
            )
        if schema_version == "16.0":
            profile_payload.update(
                {
                    "recognition_crop_policy": "detector-quad-v1",
                    "recognition_resize_interpolation": "cubic-v1",
                }
            )
        elif schema_version == "17.0":
            profile_payload.update(
                {
                    "recognition_crop_policy": "horizontal-height-0p08-max4-v1",
                    "recognition_resize_interpolation": "linear-v1",
                }
            )
        fingerprint = raster_text_profile_fingerprint_sha256(profile_payload)
        profile_payload["profile_fingerprint_sha256"] = fingerprint
        profile_payload["raster_text_profile_revision"] = (
            f"synthetic-raster-text-v1:{fingerprint[:24]}"
        )
        profile: (
            RasterTextOfflineProfileV3
            | RasterTextOfflineProfileV4
            | RasterTextOfflineProfileV5
            | RasterTextOfflineProfileV6
            | RasterTextOfflineProfileV7
            | RasterTextOfflineProfileV8
            | RasterTextOfflineProfileV9
            | RasterTextOfflineProfileV10
            | RasterTextOfflineProfileV11
            | RasterTextOfflineProfileV13
            | RasterTextOfflineProfileV14
            | RasterTextOfflineProfileV15
            | RasterTextOfflineProfileV16
            | RasterTextOfflineProfileV17
        )
        if schema_version == "3.0":
            profile = RasterTextOfflineProfileV3.model_validate(profile_payload)
        elif schema_version == "4.0":
            profile = RasterTextOfflineProfileV4.model_validate(profile_payload)
        elif schema_version == "5.0":
            profile = RasterTextOfflineProfileV5.model_validate(profile_payload)
        elif schema_version == "6.0":
            profile = RasterTextOfflineProfileV6.model_validate(profile_payload)
        elif schema_version == "7.0":
            profile = RasterTextOfflineProfileV7.model_validate(profile_payload)
        elif schema_version == "8.0":
            profile = RasterTextOfflineProfileV8.model_validate(profile_payload)
        elif schema_version == "9.0":
            profile = RasterTextOfflineProfileV9.model_validate(profile_payload)
        elif schema_version == "10.0":
            profile = RasterTextOfflineProfileV10.model_validate(profile_payload)
        elif schema_version == "11.0":
            profile = RasterTextOfflineProfileV11.model_validate(profile_payload)
        elif schema_version == "13.0":
            profile = RasterTextOfflineProfileV13.model_validate(profile_payload)
        elif schema_version == "14.0":
            profile = RasterTextOfflineProfileV14.model_validate(profile_payload)
        elif schema_version == "15.0":
            profile = RasterTextOfflineProfileV15.model_validate(profile_payload)
        elif schema_version == "16.0":
            profile = RasterTextOfflineProfileV16.model_validate(profile_payload)
        else:
            profile = RasterTextOfflineProfileV17.model_validate(profile_payload)
    except (RasterTextDirectPaddleBundleError, RasterTextMobilePaddleBundleError):
        raise RasterTextMobilePaddleBundleError(
            "mobile_paddle_bundle_runtime_profile_invalid"
        ) from None
    except (OSError, TypeError, ValidationError, ValueError):
        raise RasterTextMobilePaddleBundleError(
            "mobile_paddle_bundle_runtime_profile_invalid"
        ) from None
    suffix = profile.raster_text_profile_revision.rsplit(":", 1)[1]
    relative_path = f"raster-text-runtime-profiles-v1/{suffix}/profile.json"
    directory = _ensure_directory(
        mobile.document_preview_root, f"raster-text-runtime-profiles-v1/{suffix}"
    )
    payload = _write_mobile_json(
        path=directory / "profile.json",
        value=profile.model_dump(mode="json", warnings=False),
        code="mobile_paddle_bundle_runtime_profile_write_failed",
    )
    return (
        profile.raster_text_profile_revision,
        relative_path,
        _sha256_bytes(payload),
        profile.minimum_page_confidence,
    )


def prepare_raster_text_direct_mobile_runtime_profile_v3(
    sidecar_bundle: VerifiedRasterTextDirectPaddleBundleV1,
    model_bundle: VerifiedRasterTextDirectMobilePaddleBundleV1,
    *,
    pdfium_version: str,
    minimum_page_confidence: float,
) -> PreparedRasterTextDirectMobileRuntimeProfileV3:
    """Create-or-verify the legacy-compatible V3 mobile candidate."""

    revision, relative_path, configuration_sha256, confidence = (
        _prepare_raster_text_direct_mobile_runtime_profile(
            sidecar_bundle,
            model_bundle,
            pdfium_version=pdfium_version,
            minimum_page_confidence=minimum_page_confidence,
            schema_version="3.0",
            engine_kind="direct_paddle_mobile_text",
            extractor_revision_suffix="mobile-det-v3",
        )
    )
    return PreparedRasterTextDirectMobileRuntimeProfileV3(
        raster_text_profile_revision=revision,
        runtime_configuration_relative_path=relative_path,
        runtime_configuration_sha256=configuration_sha256,
        minimum_page_confidence=confidence,
    )


def prepare_raster_text_direct_mobile_runtime_profile_v4(
    sidecar_bundle: VerifiedRasterTextDirectPaddleBundleV1,
    model_bundle: VerifiedRasterTextDirectMobilePaddleBundleV1,
    *,
    pdfium_version: str,
    minimum_page_confidence: float,
) -> PreparedRasterTextDirectMobileRuntimeProfileV4:
    """Create-or-verify the separately fingerprinted V4 harmonic candidate."""

    revision, relative_path, configuration_sha256, confidence = (
        _prepare_raster_text_direct_mobile_runtime_profile(
            sidecar_bundle,
            model_bundle,
            pdfium_version=pdfium_version,
            minimum_page_confidence=minimum_page_confidence,
            schema_version="4.0",
            engine_kind="direct_paddle_mobile_text_harmonic",
            extractor_revision_suffix="mobile-det-v4-harmonic",
        )
    )
    return PreparedRasterTextDirectMobileRuntimeProfileV4(
        raster_text_profile_revision=revision,
        runtime_configuration_relative_path=relative_path,
        runtime_configuration_sha256=configuration_sha256,
        minimum_page_confidence=confidence,
    )


def prepare_raster_text_direct_mobile_runtime_profile_v5(
    sidecar_bundle: VerifiedRasterTextDirectPaddleBundleV1,
    model_bundle: VerifiedRasterTextDirectMobilePaddleBundleV1,
    *,
    pdfium_version: str,
    minimum_page_confidence: float,
) -> PreparedRasterTextDirectMobileRuntimeProfileV5:
    """Create-or-verify the separately fingerprinted V5 recall candidate."""

    revision, relative_path, configuration_sha256, confidence = (
        _prepare_raster_text_direct_mobile_runtime_profile(
            sidecar_bundle,
            model_bundle,
            pdfium_version=pdfium_version,
            minimum_page_confidence=minimum_page_confidence,
            schema_version="5.0",
            engine_kind="direct_paddle_mobile_text_harmonic_recall",
            extractor_revision_suffix="mobile-det-v5-harmonic-recall",
        )
    )
    return PreparedRasterTextDirectMobileRuntimeProfileV5(
        raster_text_profile_revision=revision,
        runtime_configuration_relative_path=relative_path,
        runtime_configuration_sha256=configuration_sha256,
        minimum_page_confidence=confidence,
    )


def prepare_raster_text_direct_mobile_runtime_profile_v6(
    sidecar_bundle: VerifiedRasterTextDirectPaddleBundleV1,
    model_bundle: VerifiedRasterTextDirectMobilePaddleBundleV1,
    *,
    pdfium_version: str,
    minimum_page_confidence: float,
) -> PreparedRasterTextDirectMobileRuntimeProfileV6:
    """Create-or-verify the separately fingerprinted V6 1024-side candidate."""

    revision, relative_path, configuration_sha256, confidence = (
        _prepare_raster_text_direct_mobile_runtime_profile(
            sidecar_bundle,
            model_bundle,
            pdfium_version=pdfium_version,
            minimum_page_confidence=minimum_page_confidence,
            schema_version="6.0",
            engine_kind="direct_paddle_mobile_text_1024",
            extractor_revision_suffix="mobile-det-v6-1024",
        )
    )
    return PreparedRasterTextDirectMobileRuntimeProfileV6(
        raster_text_profile_revision=revision,
        runtime_configuration_relative_path=relative_path,
        runtime_configuration_sha256=configuration_sha256,
        minimum_page_confidence=confidence,
    )


def prepare_raster_text_direct_mobile_runtime_profile_v7(
    sidecar_bundle: VerifiedRasterTextDirectPaddleBundleV1,
    model_bundle: VerifiedRasterTextDirectMobilePaddleBundleV1,
    *,
    pdfium_version: str,
    minimum_page_confidence: float,
) -> PreparedRasterTextDirectMobileRuntimeProfileV7:
    """Create-or-verify the separately fingerprinted V7 low-box candidate."""

    revision, relative_path, configuration_sha256, confidence = (
        _prepare_raster_text_direct_mobile_runtime_profile(
            sidecar_bundle,
            model_bundle,
            pdfium_version=pdfium_version,
            minimum_page_confidence=minimum_page_confidence,
            schema_version="7.0",
            engine_kind="direct_paddle_mobile_text_harmonic_recall_1024_lowbox",
            extractor_revision_suffix="mobile-det-v7-1024-lowbox",
        )
    )
    return PreparedRasterTextDirectMobileRuntimeProfileV7(
        raster_text_profile_revision=revision,
        runtime_configuration_relative_path=relative_path,
        runtime_configuration_sha256=configuration_sha256,
        minimum_page_confidence=confidence,
    )


def prepare_raster_text_direct_mobile_runtime_profile_v8(
    sidecar_bundle: VerifiedRasterTextDirectPaddleBundleV1,
    model_bundle: VerifiedRasterTextDirectMobilePaddleBundleV1,
    *,
    pdfium_version: str,
    minimum_page_confidence: float,
) -> PreparedRasterTextDirectMobileRuntimeProfileV8:
    """Create-or-verify the separately fingerprinted V8 low-box candidate."""

    revision, relative_path, configuration_sha256, confidence = (
        _prepare_raster_text_direct_mobile_runtime_profile(
            sidecar_bundle,
            model_bundle,
            pdfium_version=pdfium_version,
            minimum_page_confidence=minimum_page_confidence,
            schema_version="8.0",
            engine_kind="direct_paddle_mobile_text_harmonic_recall_native960_lowbox",
            extractor_revision_suffix="mobile-det-v8-native960-lowbox",
        )
    )
    return PreparedRasterTextDirectMobileRuntimeProfileV8(
        raster_text_profile_revision=revision,
        runtime_configuration_relative_path=relative_path,
        runtime_configuration_sha256=configuration_sha256,
        minimum_page_confidence=confidence,
    )


def prepare_raster_text_direct_mobile_runtime_profile_v9(
    sidecar_bundle: VerifiedRasterTextDirectPaddleBundleV1,
    model_bundle: VerifiedRasterTextDirectMobilePaddleBundleV1,
    *,
    pdfium_version: str,
    minimum_page_confidence: float,
) -> PreparedRasterTextDirectMobileRuntimeProfileV9:
    """Create-or-verify the separately fingerprinted V9 strip2 candidate."""

    revision, relative_path, configuration_sha256, confidence = (
        _prepare_raster_text_direct_mobile_runtime_profile(
            sidecar_bundle,
            model_bundle,
            pdfium_version=pdfium_version,
            minimum_page_confidence=minimum_page_confidence,
            schema_version="9.0",
            engine_kind="direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2",
            extractor_revision_suffix="mobile-det-v9-native960-lowbox-strip2",
        )
    )
    return PreparedRasterTextDirectMobileRuntimeProfileV9(
        raster_text_profile_revision=revision,
        runtime_configuration_relative_path=relative_path,
        runtime_configuration_sha256=configuration_sha256,
        minimum_page_confidence=confidence,
    )


def prepare_raster_text_direct_mobile_runtime_profile_v10(
    sidecar_bundle: VerifiedRasterTextDirectPaddleBundleV1,
    model_bundle: VerifiedRasterTextDirectMobilePaddleBundleV1,
    *,
    pdfium_version: str,
    minimum_page_confidence: float,
) -> PreparedRasterTextDirectMobileRuntimeProfileV10:
    """Create-or-verify the separately fingerprinted V10 grid2 candidate."""

    revision, relative_path, configuration_sha256, confidence = (
        _prepare_raster_text_direct_mobile_runtime_profile(
            sidecar_bundle,
            model_bundle,
            pdfium_version=pdfium_version,
            minimum_page_confidence=minimum_page_confidence,
            schema_version="10.0",
            engine_kind="direct_paddle_mobile_text_harmonic_recall_native960_lowbox_grid2",
            extractor_revision_suffix="mobile-det-v10-native960-lowbox-grid2",
        )
    )
    return PreparedRasterTextDirectMobileRuntimeProfileV10(
        raster_text_profile_revision=revision,
        runtime_configuration_relative_path=relative_path,
        runtime_configuration_sha256=configuration_sha256,
        minimum_page_confidence=confidence,
    )


def prepare_raster_text_direct_mobile_runtime_profile_v11(
    sidecar_bundle: VerifiedRasterTextDirectPaddleBundleV1,
    model_bundle: VerifiedRasterTextDirectMobilePaddleBundleV1,
    *,
    pdfium_version: str,
    minimum_page_confidence: float,
) -> PreparedRasterTextDirectMobileRuntimeProfileV11:
    """Create-or-verify the separately fingerprinted V11 guarded-grid candidate."""

    revision, relative_path, configuration_sha256, confidence = (
        _prepare_raster_text_direct_mobile_runtime_profile(
            sidecar_bundle,
            model_bundle,
            pdfium_version=pdfium_version,
            minimum_page_confidence=minimum_page_confidence,
            schema_version="11.0",
            engine_kind="direct_paddle_mobile_text_harmonic_recall_native960_lowbox_guarded_grid2",
            extractor_revision_suffix="mobile-det-v11-native960-lowbox-guarded-grid2",
        )
    )
    return PreparedRasterTextDirectMobileRuntimeProfileV11(
        raster_text_profile_revision=revision,
        runtime_configuration_relative_path=relative_path,
        runtime_configuration_sha256=configuration_sha256,
        minimum_page_confidence=confidence,
    )


def prepare_raster_text_direct_mobile_runtime_profile_v13(
    sidecar_bundle: VerifiedRasterTextDirectPaddleBundleV1,
    model_bundle: VerifiedRasterTextDirectMobilePaddleBundleV1,
    *,
    pdfium_version: str,
    minimum_page_confidence: float,
) -> PreparedRasterTextDirectMobileRuntimeProfileV13:
    """Create-or-verify the public V9 tuple plus fixed semantic repair."""

    revision, relative_path, configuration_sha256, confidence = (
        _prepare_raster_text_direct_mobile_runtime_profile(
            sidecar_bundle,
            model_bundle,
            pdfium_version=pdfium_version,
            minimum_page_confidence=minimum_page_confidence,
            schema_version="13.0",
            engine_kind=(
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_semantic_repair"
            ),
            extractor_revision_suffix="mobile-det-v13-native960-lowbox-strip2-semantic-repair",
        )
    )
    return PreparedRasterTextDirectMobileRuntimeProfileV13(
        raster_text_profile_revision=revision,
        runtime_configuration_relative_path=relative_path,
        runtime_configuration_sha256=configuration_sha256,
        minimum_page_confidence=confidence,
    )


def prepare_raster_text_direct_mobile_runtime_profile_v14(
    sidecar_bundle: VerifiedRasterTextDirectPaddleBundleV1,
    model_bundle: VerifiedRasterTextDirectMobilePaddleBundleV1,
    *,
    pdfium_version: str,
    minimum_page_confidence: float,
) -> PreparedRasterTextDirectMobileRuntimeProfileV14:
    """Create-or-verify V13's public OCR tuple plus V14's text delta."""

    revision, relative_path, configuration_sha256, confidence = (
        _prepare_raster_text_direct_mobile_runtime_profile(
            sidecar_bundle,
            model_bundle,
            pdfium_version=pdfium_version,
            minimum_page_confidence=minimum_page_confidence,
            schema_version="14.0",
            engine_kind=(
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_semantic_repair_v2"
            ),
            extractor_revision_suffix=("mobile-det-v14-native960-lowbox-strip2-semantic-repair-v2"),
        )
    )
    return PreparedRasterTextDirectMobileRuntimeProfileV14(
        raster_text_profile_revision=revision,
        runtime_configuration_relative_path=relative_path,
        runtime_configuration_sha256=configuration_sha256,
        minimum_page_confidence=confidence,
    )


def prepare_raster_text_direct_mobile_runtime_profile_v15(
    sidecar_bundle: VerifiedRasterTextDirectPaddleBundleV1,
    model_bundle: VerifiedRasterTextDirectMobilePaddleBundleV1,
    *,
    pdfium_version: str,
    minimum_page_confidence: float,
    cpu_threads: Literal[4, 6] = 6,
    recognition_batch_size: Literal[8, 16] = 8,
    opencv_threads: Literal[1] = 1,
) -> PreparedRasterTextDirectMobileRuntimeProfileV15:
    """Create V14-equivalent OCR with bounded, hash-bound oneDNN controls."""

    revision, relative_path, configuration_sha256, confidence = (
        _prepare_raster_text_direct_mobile_runtime_profile(
            sidecar_bundle,
            model_bundle,
            pdfium_version=pdfium_version,
            minimum_page_confidence=minimum_page_confidence,
            schema_version="15.0",
            engine_kind=(
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_"
                "semantic_repair_v2_onednn"
            ),
            extractor_revision_suffix=(
                "mobile-det-v15-native960-lowbox-strip2-semantic-repair-v2-onednn"
            ),
            cpu_threads=cpu_threads,
            recognition_batch_size=recognition_batch_size,
            opencv_threads=opencv_threads,
        )
    )
    return PreparedRasterTextDirectMobileRuntimeProfileV15(
        raster_text_profile_revision=revision,
        runtime_configuration_relative_path=relative_path,
        runtime_configuration_sha256=configuration_sha256,
        minimum_page_confidence=confidence,
    )


def prepare_raster_text_direct_mobile_runtime_profile_v16(
    sidecar_bundle: VerifiedRasterTextDirectPaddleBundleV1,
    model_bundle: VerifiedRasterTextDirectMobilePaddleBundleV1,
    *,
    pdfium_version: str,
    minimum_page_confidence: float,
    cpu_threads: Literal[4, 6] = 6,
    recognition_batch_size: Literal[8, 16] = 8,
    opencv_threads: Literal[1] = 1,
) -> PreparedRasterTextDirectMobileRuntimeProfileV16:
    """Create V15 native controls with one fixed cubic recognizer resize."""

    revision, relative_path, configuration_sha256, confidence = (
        _prepare_raster_text_direct_mobile_runtime_profile(
            sidecar_bundle,
            model_bundle,
            pdfium_version=pdfium_version,
            minimum_page_confidence=minimum_page_confidence,
            schema_version="16.0",
            engine_kind=(
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_"
                "semantic_repair_v2_onednn_cubic_recognition"
            ),
            extractor_revision_suffix=(
                "mobile-det-v16-native960-lowbox-strip2-semantic-repair-v2-onednn-cubic-recognition"
            ),
            cpu_threads=cpu_threads,
            recognition_batch_size=recognition_batch_size,
            opencv_threads=opencv_threads,
        )
    )
    return PreparedRasterTextDirectMobileRuntimeProfileV16(
        raster_text_profile_revision=revision,
        runtime_configuration_relative_path=relative_path,
        runtime_configuration_sha256=configuration_sha256,
        minimum_page_confidence=confidence,
    )


def prepare_raster_text_direct_mobile_runtime_profile_v17(
    sidecar_bundle: VerifiedRasterTextDirectPaddleBundleV1,
    model_bundle: VerifiedRasterTextDirectMobilePaddleBundleV1,
    *,
    pdfium_version: str,
    minimum_page_confidence: float,
    cpu_threads: Literal[4, 6] = 6,
    recognition_batch_size: Literal[8, 16] = 8,
    opencv_threads: Literal[1] = 1,
) -> PreparedRasterTextDirectMobileRuntimeProfileV17:
    """Create V15 native controls with one fixed padded recognizer crop."""

    revision, relative_path, configuration_sha256, confidence = (
        _prepare_raster_text_direct_mobile_runtime_profile(
            sidecar_bundle,
            model_bundle,
            pdfium_version=pdfium_version,
            minimum_page_confidence=minimum_page_confidence,
            schema_version="17.0",
            engine_kind=(
                "direct_paddle_mobile_text_harmonic_recall_native960_lowbox_strip2_"
                "semantic_repair_v2_onednn_padded_recognition"
            ),
            extractor_revision_suffix=(
                "mobile-det-v17-native960-lowbox-strip2-semantic-repair-v2-"
                "onednn-padded-recognition"
            ),
            cpu_threads=cpu_threads,
            recognition_batch_size=recognition_batch_size,
            opencv_threads=opencv_threads,
        )
    )
    return PreparedRasterTextDirectMobileRuntimeProfileV17(
        raster_text_profile_revision=revision,
        runtime_configuration_relative_path=relative_path,
        runtime_configuration_sha256=configuration_sha256,
        minimum_page_confidence=confidence,
    )


__all__ = [
    "MobilePaddleModelSourceV1",
    "PreparedRasterTextDirectMobileRuntimeProfileV3",
    "PreparedRasterTextDirectMobileRuntimeProfileV4",
    "PreparedRasterTextDirectMobileRuntimeProfileV5",
    "PreparedRasterTextDirectMobileRuntimeProfileV6",
    "PreparedRasterTextDirectMobileRuntimeProfileV7",
    "PreparedRasterTextDirectMobileRuntimeProfileV8",
    "PreparedRasterTextDirectMobileRuntimeProfileV9",
    "PreparedRasterTextDirectMobileRuntimeProfileV10",
    "PreparedRasterTextDirectMobileRuntimeProfileV11",
    "PreparedRasterTextDirectMobileRuntimeProfileV13",
    "PreparedRasterTextDirectMobileRuntimeProfileV14",
    "PreparedRasterTextDirectMobileRuntimeProfileV15",
    "PreparedRasterTextDirectMobileRuntimeProfileV16",
    "PreparedRasterTextDirectMobileRuntimeProfileV17",
    "RasterTextMobilePaddleBundleError",
    "RasterTextMobilePaddleBundleManifestV1",
    "VerifiedRasterTextDirectMobilePaddleBundleV1",
    "load_verified_raster_text_direct_mobile_paddle_bundle_v1",
    "materialize_verified_raster_text_direct_mobile_paddle_bundle_v1",
    "prepare_raster_text_direct_mobile_runtime_profile_v3",
    "prepare_raster_text_direct_mobile_runtime_profile_v4",
    "prepare_raster_text_direct_mobile_runtime_profile_v5",
    "prepare_raster_text_direct_mobile_runtime_profile_v6",
    "prepare_raster_text_direct_mobile_runtime_profile_v7",
    "prepare_raster_text_direct_mobile_runtime_profile_v8",
    "prepare_raster_text_direct_mobile_runtime_profile_v9",
    "prepare_raster_text_direct_mobile_runtime_profile_v10",
    "prepare_raster_text_direct_mobile_runtime_profile_v11",
    "prepare_raster_text_direct_mobile_runtime_profile_v13",
    "prepare_raster_text_direct_mobile_runtime_profile_v14",
    "prepare_raster_text_direct_mobile_runtime_profile_v15",
    "prepare_raster_text_direct_mobile_runtime_profile_v16",
    "prepare_raster_text_direct_mobile_runtime_profile_v17",
    "verify_local_paddleocr3_mobile_model_source_v1",
]
