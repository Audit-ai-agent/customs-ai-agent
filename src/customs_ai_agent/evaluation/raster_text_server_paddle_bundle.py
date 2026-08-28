"""Build one immutable mobile-detector/server-recognizer model bundle.

The detector is copied from an already verified public PP-OCRv5 mobile
bundle.  The recognizer static graph is copied from an already verified
PP-StructureV3 offline bundle.  No cache, URL, package installer, or model
fallback is consulted.

The server recognizer does not ship the small JSON configuration consumed by
the direct-Paddle sidecar.  This module compiles that file deterministically
from ``PostProcess.character_dict`` in the hash-bound ``inference.yml``.  The
source list currently contains 18,383 entries, including quoted punctuation,
an apostrophe and multi-code-point flag glyphs; it must not be forced through
the mobile recognizer's 512-entry assumption.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
from types import MappingProxyType
from typing import Any, Iterator, Mapping, Sequence
import uuid

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from customs_ai_agent.evaluation.local_process_lock import (
    LocalProcessLockBusy,
    LocalProcessLockInvalid,
    exclusive_local_process_lock,
)
from customs_ai_agent.evaluation.raster_offline_bundle import (
    RasterOfflineBundleError,
    VerifiedRasterOfflineBundleV1,
    load_verified_raster_offline_bundle,
    load_verified_raster_offline_bundle_release,
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
    _sha256_bytes,
    _sha256_regular_file,
    _write_create_only,
    load_verified_raster_text_direct_paddle_bundle_v1,
)
from customs_ai_agent.evaluation.raster_text_mobile_paddle_bundle import (
    RasterTextMobilePaddleBundleError,
    VerifiedRasterTextDirectMobilePaddleBundleV1,
    load_verified_raster_text_direct_mobile_paddle_bundle_v1,
)


_SHA256 = r"^[a-f0-9]{64}$"
_COLLECTION = "raster-text-direct-server-rec-bundles-v1"
_RELEASES = f"{_COLLECTION}/releases"
_BUNDLE_ID = "raster_text_direct_paddle_server_rec_model_bundle_v1"
_LOCK_FILE = ".raster-text-direct-server-rec-bundle-v1.lock"
_MODEL_ROOT = "models"
_MODEL_INVENTORY = "inventories/direct-paddle-server-rec-models.json"
_PROVENANCE_RECEIPT = "provenance/source-bundles.json"
_SERVER_RECOGNIZER_NAME = "PP-OCRv5_server_rec"
_MOBILE_DETECTOR_NAME = "PP-OCRv5_mobile_det"
_SERVER_CHARACTER_COUNT = 18_383
_MAX_YAML_BYTES = 4 * 1024 * 1024
_MAX_JSON_BYTES = 16 * 1024 * 1024
_MAX_MODEL_FILE_BYTES = 1024 * 1024 * 1024
_MAX_BUNDLE_FILES = 32
_COPY_CHUNK_BYTES = 1024 * 1024

_DETECTOR_FILENAMES = (
    "config.json",
    "inference.json",
    "inference.pdiparams",
    "inference.yml",
)
_RECOGNIZER_SOURCE_FILENAMES = (
    "inference.json",
    "inference.pdiparams",
    "inference.yml",
)
_MODEL_PATHS = tuple(
    [*(f"text-det/{name}" for name in _DETECTOR_FILENAMES)]
    + [
        "text-rec/config.json",
        *(f"text-rec/{name}" for name in _RECOGNIZER_SOURCE_FILENAMES),
    ]
)


class RasterTextServerPaddleBundleError(RuntimeError):
    """A content-free failure at the combined server-recognizer boundary."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RasterTextServerPaddleBundleFileV1(_WireModel):
    path: str = Field(min_length=1, max_length=480)
    sha256: str = Field(pattern=_SHA256)
    byte_size: int = Field(ge=0, le=_MAX_MODEL_FILE_BYTES)

    _relative = field_validator("path")(_relative_path)


class RasterTextServerPaddleBundleManifestV1(_WireModel):
    """Exhaustive declaration of one combined model release."""

    schema_version: str
    bundle_id: str
    source_mobile_bundle_manifest_sha256: str = Field(pattern=_SHA256)
    source_offline_bundle_manifest_sha256: str = Field(pattern=_SHA256)
    source_detector_model_snapshot_sha256: str = Field(pattern=_SHA256)
    source_recognizer_model_snapshot_sha256: str = Field(pattern=_SHA256)
    source_recognizer_inference_yml_sha256: str = Field(pattern=_SHA256)
    source_recognizer_character_dictionary_sha256: str = Field(pattern=_SHA256)
    source_recognizer_character_count: int = Field(ge=1, le=_SERVER_CHARACTER_COUNT)
    generated_recognizer_configuration_sha256: str = Field(pattern=_SHA256)
    text_detection_model_name: str
    text_recognition_model_name: str
    local_model_root_relative_path: str
    model_inventory_relative_path: str
    provenance_receipt_relative_path: str
    model_root_tree_sha256: str = Field(pattern=_SHA256)
    files: list[RasterTextServerPaddleBundleFileV1] = Field(
        min_length=1, max_length=_MAX_BUNDLE_FILES
    )

    @field_validator(
        "local_model_root_relative_path",
        "model_inventory_relative_path",
        "provenance_receipt_relative_path",
    )
    @classmethod
    def _paths(cls, value: str) -> str:
        return _relative_path(value)

    @model_validator(mode="after")
    def _fixed_shape(self) -> "RasterTextServerPaddleBundleManifestV1":
        if (
            self.schema_version != "1.0"
            or self.bundle_id != _BUNDLE_ID
            or self.source_recognizer_character_count != _SERVER_CHARACTER_COUNT
            or self.text_detection_model_name != _MOBILE_DETECTOR_NAME
            or self.text_recognition_model_name != _SERVER_RECOGNIZER_NAME
            or self.local_model_root_relative_path != _MODEL_ROOT
            or self.model_inventory_relative_path != _MODEL_INVENTORY
            or self.provenance_receipt_relative_path != _PROVENANCE_RECEIPT
        ):
            raise ValueError("server Paddle bundle manifest shape is invalid")
        paths = [item.path for item in self.files]
        expected = {
            _MODEL_INVENTORY,
            _PROVENANCE_RECEIPT,
            *(f"models/{path}" for path in _MODEL_PATHS),
        }
        if paths != sorted(paths) or len(paths) != len(set(paths)) or set(paths) != expected:
            raise ValueError("server Paddle bundle file list is invalid")
        return self


@dataclass(frozen=True, slots=True)
class VerifiedRasterTextDirectServerPaddleBundleV1:
    document_preview_root: Path
    bundle_root: Path
    bundle_root_relative_path: str
    manifest: RasterTextServerPaddleBundleManifestV1
    manifest_sha256: str
    file_sha256_by_relative_path: Mapping[str, str]

    def bundle_relative_path(self, path: str) -> str:
        _relative_path(path)
        return f"{self.bundle_root_relative_path}/{path}"

    def file_sha256(self, path: str) -> str:
        try:
            return self.file_sha256_by_relative_path[path]
        except KeyError:
            raise RasterTextServerPaddleBundleError(
                "server_paddle_bundle_manifest_invalid"
            ) from None


@dataclass(frozen=True, slots=True)
class PreparedRasterTextDirectServerRuntimeProfileV18:
    """Content-addressed V18 mobile-detector/server-recognizer profile."""

    raster_text_profile_revision: str
    runtime_configuration_relative_path: str
    runtime_configuration_sha256: str
    minimum_page_confidence: float


@dataclass(frozen=True, slots=True)
class _SourceFile:
    source: Path
    target_relative_path: str
    source_relative_path: str
    sha256: str
    byte_size: int


@dataclass(frozen=True, slots=True)
class _VerifiedSources:
    root: Path
    mobile: VerifiedRasterTextDirectMobilePaddleBundleV1
    offline: VerifiedRasterOfflineBundleV1
    detector_files: tuple[_SourceFile, ...]
    recognizer_files: tuple[_SourceFile, ...]
    characters: tuple[str, ...]
    character_dictionary_sha256: str
    recognizer_configuration: Mapping[str, object]
    recognizer_configuration_sha256: str
    detector_snapshot_sha256: str
    recognizer_snapshot_sha256: str


def _canonical_bytes(value: object) -> bytes:
    return _canonical_json(value)


def _server_root(path: Path) -> Path:
    try:
        return _root(path)
    except RasterTextDirectPaddleBundleError:
        raise RasterTextServerPaddleBundleError("server_paddle_bundle_root_invalid") from None


def _safe_server_path(root: Path, relative: str, *, directory: bool = False) -> Path:
    try:
        return _safe_path(root, relative, directory=directory)
    except RasterTextDirectPaddleBundleError:
        raise RasterTextServerPaddleBundleError("server_paddle_bundle_path_invalid") from None


def _sha256_file(path: Path, *, code: str) -> tuple[str, int]:
    try:
        return _sha256_regular_file(path, code=code)
    except RasterTextDirectPaddleBundleError:
        raise RasterTextServerPaddleBundleError(code) from None


def _source_path(root: Path, relative: str) -> Path:
    try:
        return _safe_path(root, relative)
    except RasterTextDirectPaddleBundleError:
        raise RasterTextServerPaddleBundleError("server_paddle_bundle_source_drift") from None


def _parse_single_quoted_yaml_scalar(value: str) -> str:
    if len(value) < 2 or value[0] != "'" or value[-1] != "'":
        raise ValueError
    inner = value[1:-1]
    output: list[str] = []
    index = 0
    while index < len(inner):
        if inner[index] != "'":
            output.append(inner[index])
            index += 1
            continue
        if index + 1 >= len(inner) or inner[index + 1] != "'":
            raise ValueError
        output.append("'")
        index += 2
    return "".join(output)


def _parse_yaml_character_scalar(value: str) -> str:
    if not value or value != value.strip(" \t"):
        raise ValueError
    if value.startswith("'"):
        parsed = _parse_single_quoted_yaml_scalar(value)
    elif value.startswith('"'):
        parsed_value = json.loads(value)
        if not isinstance(parsed_value, str):
            raise ValueError
        parsed = parsed_value
    else:
        # This is deliberately a string-list parser, not a general YAML
        # loader.  Plain values are preserved byte-for-text and YAML comments,
        # flow collections, aliases, tags and block scalars are not accepted.
        if (
            " #" in value
            or value[0] in "&*!>[{"
            or value in {"~", "null", "Null", "NULL", "true", "false", "True", "False"}
        ):
            raise ValueError
        parsed = value
    if not parsed or any(ord(character) < 32 for character in parsed):
        raise ValueError
    return parsed


def parse_ppocrv5_server_rec_character_dict_v1(raw: bytes) -> tuple[str, ...]:
    """Read only the fixed ``PostProcess.character_dict`` YAML subset.

    The parser is intentionally independent of PyYAML and returns the source
    order exactly.  It accepts at most the known 18,383 entries; callers that
    build the production bundle additionally require that exact count.
    """

    try:
        if not raw or len(raw) > _MAX_YAML_BYTES or b"\x00" in raw or b"\t" in raw:
            raise ValueError
        text = raw.decode("utf-8")
        if text.startswith("\ufeff") or "\r" in text:
            raise ValueError
        lines = text.splitlines()
        if lines.count("Global:") != 1:
            raise ValueError
        global_start = lines.index("Global:") + 1
        global_end = len(lines)
        for index in range(global_start, len(lines)):
            line = lines[index]
            if line and line[0] not in " ":
                global_end = index
                break
        if lines[global_start:global_end].count(f"  model_name: {_SERVER_RECOGNIZER_NAME}") != 1:
            raise ValueError
        if lines.count("PostProcess:") != 1:
            raise ValueError
        start = lines.index("PostProcess:") + 1
        end = len(lines)
        for index in range(start, len(lines)):
            line = lines[index]
            if line and line[0] not in " ":
                end = index
                break
        section = lines[start:end]
        if section.count("  name: CTCLabelDecode") != 1 or section.count("  character_dict:") != 1:
            raise ValueError
        dictionary_index = section.index("  character_dict:")
        if section.index("  name: CTCLabelDecode") > dictionary_index:
            raise ValueError
        entries: list[str] = []
        for line in section[dictionary_index + 1 :]:
            if not line.startswith("  - "):
                raise ValueError
            entries.append(_parse_yaml_character_scalar(line[4:]))
            if len(entries) > _SERVER_CHARACTER_COUNT:
                raise ValueError
        if not entries or len(entries) != len(set(entries)):
            raise ValueError
        return tuple(entries)
    except (UnicodeDecodeError, ValueError, json.JSONDecodeError):
        raise RasterTextServerPaddleBundleError(
            "server_paddle_bundle_character_dictionary_invalid"
        ) from None


def _character_dictionary_sha256(characters: Sequence[str]) -> str:
    return hashlib.sha256(_canonical_bytes(list(characters))).hexdigest()


def _server_recognizer_configuration(characters: Sequence[str]) -> dict[str, object]:
    return {
        "Global": {"model_name": _SERVER_RECOGNIZER_NAME},
        "PostProcess": {
            "character_dict": list(characters),
            "name": "CTCLabelDecode",
        },
    }


def _records_snapshot(files: Sequence[_SourceFile]) -> str:
    return hashlib.sha256(
        _canonical_bytes(
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


def _model_tree_sha256(records: Sequence[RasterTextServerPaddleBundleFileV1]) -> str:
    return hashlib.sha256(
        _canonical_bytes(
            {
                "files": [
                    {"byte_size": item.byte_size, "path": item.path, "sha256": item.sha256}
                    for item in records
                ]
            }
        )
    ).hexdigest()


def _resolve_sources(
    *,
    document_preview_root: Path | None,
    mobile_bundle: VerifiedRasterTextDirectMobilePaddleBundleV1 | None,
    offline_bundle: VerifiedRasterOfflineBundleV1 | None,
    mobile_bundle_manifest_sha256: str | None,
    offline_bundle_manifest_sha256: str | None,
) -> _VerifiedSources:
    try:
        if mobile_bundle is None or offline_bundle is None:
            if (
                mobile_bundle is not None
                or offline_bundle is not None
                or document_preview_root is None
                or mobile_bundle_manifest_sha256 is None
                or offline_bundle_manifest_sha256 is None
            ):
                raise RasterTextServerPaddleBundleError(
                    "server_paddle_bundle_source_binding_invalid"
                )
            root = _server_root(document_preview_root)
            mobile_bundle = load_verified_raster_text_direct_mobile_paddle_bundle_v1(
                document_preview_root=root, manifest_sha256=mobile_bundle_manifest_sha256
            )
            offline_bundle = load_verified_raster_offline_bundle_release(
                document_preview_root=root, manifest_sha256=offline_bundle_manifest_sha256
            )
        else:
            if (
                mobile_bundle_manifest_sha256 is not None
                or offline_bundle_manifest_sha256 is not None
            ):
                raise RasterTextServerPaddleBundleError(
                    "server_paddle_bundle_source_binding_invalid"
                )
            root = _server_root(
                mobile_bundle.document_preview_root
                if document_preview_root is None
                else document_preview_root
            )
            if _local_path_identity(root) != _local_path_identity(
                mobile_bundle.document_preview_root
            ) or _local_path_identity(root) != _local_path_identity(
                offline_bundle.document_preview_root
            ):
                raise RasterTextServerPaddleBundleError(
                    "server_paddle_bundle_source_binding_invalid"
                )
            # A dataclass is not a durable trust token.  Rewalk both releases
            # by their exact manifests immediately before copying.
            mobile_bundle = load_verified_raster_text_direct_mobile_paddle_bundle_v1(
                document_preview_root=root, manifest_sha256=mobile_bundle.manifest_sha256
            )
            expected_offline_release = (
                f"raster-offline-bundles-v1/releases/{offline_bundle.manifest_sha256}"
            )
            if offline_bundle.bundle_root_relative_path == expected_offline_release:
                offline_bundle = load_verified_raster_offline_bundle_release(
                    document_preview_root=root, manifest_sha256=offline_bundle.manifest_sha256
                )
            elif offline_bundle.bundle_root_relative_path == "raster-offline-bundle-v1":
                supplied_sha256 = offline_bundle.manifest_sha256
                offline_bundle = load_verified_raster_offline_bundle(root)
                if (
                    offline_bundle.bundle_root_relative_path != "raster-offline-bundle-v1"
                    or offline_bundle.manifest_sha256 != supplied_sha256
                ):
                    raise RasterTextServerPaddleBundleError(
                        "server_paddle_bundle_source_binding_invalid"
                    )
            else:
                raise RasterTextServerPaddleBundleError(
                    "server_paddle_bundle_source_binding_invalid"
                )
        if (
            _local_path_identity(root) != _local_path_identity(mobile_bundle.document_preview_root)
            or _local_path_identity(root)
            != _local_path_identity(offline_bundle.document_preview_root)
            or mobile_bundle.manifest.text_detection_model_name != _MOBILE_DETECTOR_NAME
        ):
            raise RasterTextServerPaddleBundleError("server_paddle_bundle_source_binding_invalid")

        detector: list[_SourceFile] = []
        for filename in _DETECTOR_FILENAMES:
            source_relative = f"models/text-det/{filename}"
            source = _source_path(mobile_bundle.bundle_root, source_relative)
            digest, byte_size = _sha256_file(source, code="server_paddle_bundle_source_drift")
            if digest != mobile_bundle.file_sha256(source_relative):
                raise RasterTextServerPaddleBundleError("server_paddle_bundle_source_drift")
            detector.append(
                _SourceFile(source, f"text-det/{filename}", source_relative, digest, byte_size)
            )

        recognizer: list[_SourceFile] = []
        offline_model_root = offline_bundle.manifest.local_model_root_relative_path
        for filename in _RECOGNIZER_SOURCE_FILENAMES:
            source_relative = f"{offline_model_root}/text-rec/{filename}"
            source = _source_path(offline_bundle.bundle_root, source_relative)
            digest, byte_size = _sha256_file(source, code="server_paddle_bundle_source_drift")
            if digest != offline_bundle.file_sha256(source_relative):
                raise RasterTextServerPaddleBundleError("server_paddle_bundle_source_drift")
            recognizer.append(
                _SourceFile(source, f"text-rec/{filename}", source_relative, digest, byte_size)
            )
        inference_yml = next(
            item for item in recognizer if item.target_relative_path == "text-rec/inference.yml"
        )
        raw_yml = inference_yml.source.read_bytes()
        if hashlib.sha256(raw_yml).hexdigest() != inference_yml.sha256:
            raise RasterTextServerPaddleBundleError("server_paddle_bundle_source_drift")
        characters = parse_ppocrv5_server_rec_character_dict_v1(raw_yml)
        if len(characters) != _SERVER_CHARACTER_COUNT:
            raise RasterTextServerPaddleBundleError(
                "server_paddle_bundle_character_dictionary_invalid"
            )
        configuration = _server_recognizer_configuration(characters)
        configuration_bytes = _canonical_bytes(configuration)
        detector_tuple = tuple(detector)
        recognizer_tuple = tuple(recognizer)
        return _VerifiedSources(
            root=root,
            mobile=mobile_bundle,
            offline=offline_bundle,
            detector_files=detector_tuple,
            recognizer_files=recognizer_tuple,
            characters=characters,
            character_dictionary_sha256=_character_dictionary_sha256(characters),
            recognizer_configuration=configuration,
            recognizer_configuration_sha256=hashlib.sha256(configuration_bytes).hexdigest(),
            detector_snapshot_sha256=_records_snapshot(detector_tuple),
            recognizer_snapshot_sha256=_records_snapshot(recognizer_tuple),
        )
    except RasterTextServerPaddleBundleError:
        raise
    except (RasterTextMobilePaddleBundleError, RasterOfflineBundleError):
        raise RasterTextServerPaddleBundleError("server_paddle_bundle_source_unavailable") from None
    except (OSError, StopIteration, ValueError, TypeError, RasterTextDirectPaddleBundleError):
        raise RasterTextServerPaddleBundleError("server_paddle_bundle_source_drift") from None


def _runtime_inventory_record(item: RasterTextServerPaddleBundleFileV1) -> dict[str, object]:
    """Use the direct sidecar's immutable ``relative_path`` wire contract."""

    return {
        "relative_path": item.path,
        "sha256": item.sha256,
        "byte_size": item.byte_size,
    }


def _inventory(
    sources: _VerifiedSources, model_records: Sequence[RasterTextServerPaddleBundleFileV1]
) -> dict[str, object]:
    detector_records = [item for item in model_records if item.path.startswith("text-det/")]
    recognizer_records = [item for item in model_records if item.path.startswith("text-rec/")]
    return {
        "artifacts": [_runtime_inventory_record(item) for item in model_records],
        "detector": {
            "artifacts": [_runtime_inventory_record(item) for item in detector_records],
            "model_name": _MOBILE_DETECTOR_NAME,
            "source_bundle_manifest_sha256": sources.mobile.manifest_sha256,
            "source_model_snapshot_sha256": sources.detector_snapshot_sha256,
        },
        "inventory_kind": "raster_text_direct_paddle_server_rec_models",
        "recognizer": {
            "artifacts": [_runtime_inventory_record(item) for item in recognizer_records],
            "character_count": len(sources.characters),
            "character_dictionary_sha256": sources.character_dictionary_sha256,
            "generated_configuration_sha256": sources.recognizer_configuration_sha256,
            "model_name": _SERVER_RECOGNIZER_NAME,
            "source_bundle_manifest_sha256": sources.offline.manifest_sha256,
            "source_inference_yml_sha256": next(
                item.sha256
                for item in sources.recognizer_files
                if item.target_relative_path == "text-rec/inference.yml"
            ),
            "source_model_snapshot_sha256": sources.recognizer_snapshot_sha256,
        },
        "schema_version": "1.0",
    }


def _provenance(sources: _VerifiedSources) -> dict[str, object]:
    return {
        "detector": {
            "artifacts": [
                {
                    "byte_size": item.byte_size,
                    "source_relative_path": item.source_relative_path,
                    "target_relative_path": item.target_relative_path,
                    "sha256": item.sha256,
                }
                for item in sources.detector_files
            ],
            "bundle_manifest_sha256": sources.mobile.manifest_sha256,
            "bundle_relative_path": sources.mobile.bundle_root_relative_path,
            "model_name": _MOBILE_DETECTOR_NAME,
        },
        "provenance_kind": "verified_mobile_detector_and_offline_server_recognizer",
        "recognizer": {
            "artifacts": [
                {
                    "byte_size": item.byte_size,
                    "source_relative_path": item.source_relative_path,
                    "target_relative_path": item.target_relative_path,
                    "sha256": item.sha256,
                }
                for item in sources.recognizer_files
            ],
            "bundle_manifest_sha256": sources.offline.manifest_sha256,
            "bundle_relative_path": sources.offline.bundle_root_relative_path,
            "character_count": len(sources.characters),
            "character_dictionary_sha256": sources.character_dictionary_sha256,
            "generated_configuration_sha256": sources.recognizer_configuration_sha256,
            "model_name": _SERVER_RECOGNIZER_NAME,
        },
        "schema_version": "1.0",
    }


def _write_bytes(path: Path, raw: bytes, *, code: str) -> tuple[str, int]:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or path.is_symlink():
            raise OSError
        descriptor = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        return hashlib.sha256(raw).hexdigest(), len(raw)
    except OSError:
        raise RasterTextServerPaddleBundleError(code) from None


def _write_stage_bytes(
    *, stage: Path, relative: str, raw: bytes, code: str
) -> RasterTextServerPaddleBundleFileV1:
    digest, byte_size = _write_bytes(stage.joinpath(*PurePosixPath(relative).parts), raw, code=code)
    try:
        return RasterTextServerPaddleBundleFileV1(path=relative, sha256=digest, byte_size=byte_size)
    except ValidationError:
        raise RasterTextServerPaddleBundleError(code) from None


def _copy_stage_file(*, stage: Path, item: _SourceFile) -> RasterTextServerPaddleBundleFileV1:
    target = stage.joinpath(*PurePosixPath(f"models/{item.target_relative_path}").parts)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.exists() or target.is_symlink():
            raise OSError
        descriptor = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        digest = hashlib.sha256()
        byte_size = 0
        with item.source.open("rb") as source, os.fdopen(descriptor, "wb") as destination:
            while chunk := source.read(_COPY_CHUNK_BYTES):
                byte_size += len(chunk)
                if byte_size > _MAX_MODEL_FILE_BYTES:
                    raise OSError
                digest.update(chunk)
                destination.write(chunk)
            destination.flush()
            os.fsync(destination.fileno())
        if digest.hexdigest() != item.sha256 or byte_size != item.byte_size:
            raise OSError
        return RasterTextServerPaddleBundleFileV1(
            path=f"models/{item.target_relative_path}", sha256=item.sha256, byte_size=item.byte_size
        )
    except (OSError, ValidationError):
        try:
            target.unlink(missing_ok=True)
        except OSError:
            pass
        raise RasterTextServerPaddleBundleError("server_paddle_bundle_source_drift") from None


def _build_stage(*, stage: Path, sources: _VerifiedSources) -> str:
    model_records = [
        *(_copy_stage_file(stage=stage, item=item) for item in sources.detector_files),
        _write_stage_bytes(
            stage=stage,
            relative="models/text-rec/config.json",
            raw=_canonical_bytes(sources.recognizer_configuration),
            code="server_paddle_bundle_stage_write_failed",
        ),
        *(_copy_stage_file(stage=stage, item=item) for item in sources.recognizer_files),
    ]
    model_records.sort(key=lambda item: item.path)
    if tuple(item.path.removeprefix("models/") for item in model_records) != tuple(
        sorted(_MODEL_PATHS)
    ):
        raise RasterTextServerPaddleBundleError("server_paddle_bundle_stage_write_failed")
    inventory_record = _write_stage_bytes(
        stage=stage,
        relative=_MODEL_INVENTORY,
        raw=_canonical_bytes(
            _inventory(
                sources, [item.model_copy(update={"path": item.path[7:]}) for item in model_records]
            )
        ),
        code="server_paddle_bundle_stage_write_failed",
    )
    provenance_record = _write_stage_bytes(
        stage=stage,
        relative=_PROVENANCE_RECEIPT,
        raw=_canonical_bytes(_provenance(sources)),
        code="server_paddle_bundle_stage_write_failed",
    )
    inference_yml_sha = next(
        item.sha256
        for item in sources.recognizer_files
        if item.target_relative_path == "text-rec/inference.yml"
    )
    files = sorted(
        [*model_records, inventory_record, provenance_record], key=lambda item: item.path
    )
    manifest = RasterTextServerPaddleBundleManifestV1(
        schema_version="1.0",
        bundle_id=_BUNDLE_ID,
        source_mobile_bundle_manifest_sha256=sources.mobile.manifest_sha256,
        source_offline_bundle_manifest_sha256=sources.offline.manifest_sha256,
        source_detector_model_snapshot_sha256=sources.detector_snapshot_sha256,
        source_recognizer_model_snapshot_sha256=sources.recognizer_snapshot_sha256,
        source_recognizer_inference_yml_sha256=inference_yml_sha,
        source_recognizer_character_dictionary_sha256=sources.character_dictionary_sha256,
        source_recognizer_character_count=len(sources.characters),
        generated_recognizer_configuration_sha256=sources.recognizer_configuration_sha256,
        text_detection_model_name=_MOBILE_DETECTOR_NAME,
        text_recognition_model_name=_SERVER_RECOGNIZER_NAME,
        local_model_root_relative_path=_MODEL_ROOT,
        model_inventory_relative_path=_MODEL_INVENTORY,
        provenance_receipt_relative_path=_PROVENANCE_RECEIPT,
        model_root_tree_sha256=_model_tree_sha256(
            [item.model_copy(update={"path": item.path[7:]}) for item in model_records]
        ),
        files=files,
    )
    raw = _canonical_bytes(manifest.model_dump(mode="json", warnings=False))
    _write_stage_bytes(
        stage=stage,
        relative="manifest.json",
        raw=raw,
        code="server_paddle_bundle_stage_write_failed",
    )
    return hashlib.sha256(raw).hexdigest()


def _release_relative_path(manifest_sha256: str) -> str:
    if len(manifest_sha256) != 64 or any(
        value not in "0123456789abcdef" for value in manifest_sha256
    ):
        raise RasterTextServerPaddleBundleError("server_paddle_bundle_manifest_invalid")
    return f"{_RELEASES}/{manifest_sha256}"


def _tree_files(root: Path) -> dict[str, Path]:
    try:
        result: dict[str, Path] = {}
        for path in root.rglob("*"):
            if path.is_symlink():
                raise OSError
            if path.is_file():
                relative = path.relative_to(root).as_posix()
                if relative in result:
                    raise OSError
                result[relative] = path
            elif not path.is_dir():
                raise OSError
        return result
    except (OSError, ValueError):
        raise RasterTextServerPaddleBundleError("server_paddle_bundle_manifest_invalid") from None


def _read_canonical_json(path: Path) -> tuple[dict[str, Any], bytes]:
    try:
        raw = path.read_bytes()
        if not raw or len(raw) > _MAX_JSON_BYTES:
            raise ValueError
        value = json.loads(raw.decode("utf-8"))
        if not isinstance(value, dict) or raw != _canonical_bytes(value):
            raise ValueError
        return value, raw
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        raise RasterTextServerPaddleBundleError("server_paddle_bundle_manifest_invalid") from None


def _load_verified_from_root(
    *, root: Path, bundle_root: Path, expected_manifest_sha256: str, allow_staging: bool = False
) -> VerifiedRasterTextDirectServerPaddleBundleV1:
    try:
        if bundle_root.is_symlink() or not bundle_root.is_dir():
            raise ValueError
        manifest_value, manifest_raw = _read_canonical_json(bundle_root / "manifest.json")
        if hashlib.sha256(manifest_raw).hexdigest() != expected_manifest_sha256:
            raise ValueError
        manifest = RasterTextServerPaddleBundleManifestV1.model_validate(manifest_value)
        tree = _tree_files(bundle_root)
        expected_paths = {"manifest.json", *(item.path for item in manifest.files)}
        if set(tree) != expected_paths:
            raise ValueError
        records = {item.path: item for item in manifest.files}
        for relative, record in records.items():
            digest, byte_size = _sha256_file(
                tree[relative], code="server_paddle_bundle_manifest_invalid"
            )
            if digest != record.sha256 or byte_size != record.byte_size:
                raise ValueError
        model_records = [
            records[f"models/{relative}"].model_copy(update={"path": relative})
            for relative in _MODEL_PATHS
        ]
        if _model_tree_sha256(model_records) != manifest.model_root_tree_sha256:
            raise ValueError
        config_value, config_raw = _read_canonical_json(tree["models/text-rec/config.json"])
        characters = (
            config_value.get("PostProcess", {}).get("character_dict")
            if isinstance(config_value.get("PostProcess"), dict)
            else None
        )
        if (
            set(config_value) != {"Global", "PostProcess"}
            or config_value.get("Global") != {"model_name": _SERVER_RECOGNIZER_NAME}
            or not isinstance(config_value.get("PostProcess"), dict)
            or set(config_value["PostProcess"]) != {"character_dict", "name"}
            or config_value["PostProcess"].get("name") != "CTCLabelDecode"
            or not isinstance(characters, list)
            or len(characters) != _SERVER_CHARACTER_COUNT
            or any(not isinstance(item, str) or not item for item in characters)
            or len(characters) != len(set(characters))
            or hashlib.sha256(config_raw).hexdigest()
            != manifest.generated_recognizer_configuration_sha256
            or _character_dictionary_sha256(characters)
            != manifest.source_recognizer_character_dictionary_sha256
        ):
            raise ValueError
        inference_yml = tree["models/text-rec/inference.yml"].read_bytes()
        if (
            hashlib.sha256(inference_yml).hexdigest()
            != manifest.source_recognizer_inference_yml_sha256
            or parse_ppocrv5_server_rec_character_dict_v1(inference_yml) != tuple(characters)
        ):
            raise ValueError
        inventory, _inventory_raw = _read_canonical_json(tree[_MODEL_INVENTORY])
        provenance, _provenance_raw = _read_canonical_json(tree[_PROVENANCE_RECEIPT])
        detector_records = [item for item in model_records if item.path.startswith("text-det/")]
        recognizer_records = [item for item in model_records if item.path.startswith("text-rec/")]
        expected_inventory = {
            "artifacts": [_runtime_inventory_record(item) for item in model_records],
            "detector": {
                "artifacts": [_runtime_inventory_record(item) for item in detector_records],
                "model_name": _MOBILE_DETECTOR_NAME,
                "source_bundle_manifest_sha256": manifest.source_mobile_bundle_manifest_sha256,
                "source_model_snapshot_sha256": manifest.source_detector_model_snapshot_sha256,
            },
            "inventory_kind": "raster_text_direct_paddle_server_rec_models",
            "recognizer": {
                "artifacts": [_runtime_inventory_record(item) for item in recognizer_records],
                "character_count": manifest.source_recognizer_character_count,
                "character_dictionary_sha256": (
                    manifest.source_recognizer_character_dictionary_sha256
                ),
                "generated_configuration_sha256": (
                    manifest.generated_recognizer_configuration_sha256
                ),
                "model_name": _SERVER_RECOGNIZER_NAME,
                "source_bundle_manifest_sha256": (manifest.source_offline_bundle_manifest_sha256),
                "source_inference_yml_sha256": (manifest.source_recognizer_inference_yml_sha256),
                "source_model_snapshot_sha256": (manifest.source_recognizer_model_snapshot_sha256),
            },
            "schema_version": "1.0",
        }
        if (
            inventory != expected_inventory
            or set(provenance)
            != {
                "detector",
                "provenance_kind",
                "recognizer",
                "schema_version",
            }
            or provenance.get("schema_version") != "1.0"
            or provenance.get("provenance_kind")
            != "verified_mobile_detector_and_offline_server_recognizer"
        ):
            raise ValueError
        detector_provenance = provenance.get("detector")
        recognizer_provenance = provenance.get("recognizer")
        if not isinstance(detector_provenance, dict) or not isinstance(recognizer_provenance, dict):
            raise ValueError
        if (
            set(detector_provenance)
            != {"artifacts", "bundle_manifest_sha256", "bundle_relative_path", "model_name"}
            or set(recognizer_provenance)
            != {
                "artifacts",
                "bundle_manifest_sha256",
                "bundle_relative_path",
                "character_count",
                "character_dictionary_sha256",
                "generated_configuration_sha256",
                "model_name",
            }
            or detector_provenance.get("bundle_manifest_sha256")
            != manifest.source_mobile_bundle_manifest_sha256
            or detector_provenance.get("bundle_relative_path")
            != (
                "raster-text-direct-mobile-bundles-v1/releases/"
                f"{manifest.source_mobile_bundle_manifest_sha256}"
            )
            or detector_provenance.get("model_name") != _MOBILE_DETECTOR_NAME
            or recognizer_provenance.get("bundle_manifest_sha256")
            != manifest.source_offline_bundle_manifest_sha256
            or recognizer_provenance.get("bundle_relative_path")
            not in {
                "raster-offline-bundle-v1",
                (
                    "raster-offline-bundles-v1/releases/"
                    f"{manifest.source_offline_bundle_manifest_sha256}"
                ),
            }
            or recognizer_provenance.get("character_count")
            != manifest.source_recognizer_character_count
            or recognizer_provenance.get("character_dictionary_sha256")
            != manifest.source_recognizer_character_dictionary_sha256
            or recognizer_provenance.get("generated_configuration_sha256")
            != manifest.generated_recognizer_configuration_sha256
            or recognizer_provenance.get("model_name") != _SERVER_RECOGNIZER_NAME
        ):
            raise ValueError
        provenance_groups = (
            (
                detector_provenance.get("artifacts"),
                detector_records,
                manifest.source_detector_model_snapshot_sha256,
            ),
            (
                recognizer_provenance.get("artifacts"),
                [item for item in recognizer_records if item.path != "text-rec/config.json"],
                manifest.source_recognizer_model_snapshot_sha256,
            ),
        )
        for raw_artifacts, expected_records, expected_snapshot in provenance_groups:
            if not isinstance(raw_artifacts, list) or len(raw_artifacts) != len(expected_records):
                raise ValueError
            source_records: list[_SourceFile] = []
            for raw_artifact, expected_record in zip(raw_artifacts, expected_records, strict=True):
                if (
                    not isinstance(raw_artifact, dict)
                    or set(raw_artifact)
                    != {
                        "byte_size",
                        "source_relative_path",
                        "target_relative_path",
                        "sha256",
                    }
                    or raw_artifact.get("target_relative_path") != expected_record.path
                    or raw_artifact.get("sha256") != expected_record.sha256
                    or raw_artifact.get("byte_size") != expected_record.byte_size
                    or not isinstance(raw_artifact.get("source_relative_path"), str)
                ):
                    raise ValueError
                source_relative = _relative_path(raw_artifact["source_relative_path"])
                source_records.append(
                    _SourceFile(
                        source=Path(),
                        target_relative_path=expected_record.path,
                        source_relative_path=source_relative,
                        sha256=expected_record.sha256,
                        byte_size=expected_record.byte_size,
                    )
                )
            if _records_snapshot(source_records) != expected_snapshot:
                raise ValueError
        if allow_staging:
            name = os.path.basename(_local_path_identity(bundle_root))
            if _local_path_identity(bundle_root.parent) != _local_path_identity(
                root
            ) or not name.startswith(f".{_COLLECTION}.staging-"):
                raise ValueError
            relative = name
        else:
            relative = _release_relative_path(expected_manifest_sha256)
            expected_root = _safe_server_path(root, relative, directory=True)
            if _local_path_identity(expected_root) != _local_path_identity(bundle_root):
                raise ValueError
        return VerifiedRasterTextDirectServerPaddleBundleV1(
            document_preview_root=root,
            bundle_root=bundle_root,
            bundle_root_relative_path=relative,
            manifest=manifest,
            manifest_sha256=expected_manifest_sha256,
            file_sha256_by_relative_path=MappingProxyType(
                {path: record.sha256 for path, record in records.items()}
            ),
        )
    except RasterTextServerPaddleBundleError:
        raise
    except (OSError, ValidationError, ValueError, TypeError):
        raise RasterTextServerPaddleBundleError("server_paddle_bundle_manifest_invalid") from None


def load_verified_raster_text_direct_server_paddle_bundle_v1(
    *, document_preview_root: Path, manifest_sha256: str
) -> VerifiedRasterTextDirectServerPaddleBundleV1:
    root = _server_root(document_preview_root)
    relative = _release_relative_path(manifest_sha256)
    bundle_root = _safe_server_path(root, relative, directory=True)
    return _load_verified_from_root(
        root=root, bundle_root=bundle_root, expected_manifest_sha256=manifest_sha256
    )


def _ensure_directory(root: Path, relative: str) -> Path:
    try:
        target = root
        for part in PurePosixPath(_relative_path(relative)).parts:
            target = target / part
            if target.is_symlink():
                raise OSError
            target.mkdir(exist_ok=True)
        if target.is_symlink() or not target.is_dir():
            raise OSError
        if os.path.commonpath((_local_path_identity(root), _local_path_identity(target))) != (
            _local_path_identity(root)
        ):
            raise OSError
        return target
    except (OSError, ValueError):
        raise RasterTextServerPaddleBundleError("server_paddle_bundle_output_invalid") from None


@contextmanager
def _lock(root: Path) -> Iterator[None]:
    try:
        with exclusive_local_process_lock(root / _LOCK_FILE):
            yield
    except LocalProcessLockBusy:
        raise RasterTextServerPaddleBundleError("server_paddle_bundle_locked") from None
    except LocalProcessLockInvalid:
        raise RasterTextServerPaddleBundleError("server_paddle_bundle_lock_invalid") from None


def _remove_owned_stage(root: Path, stage: Path) -> None:
    try:
        resolved = stage.resolve(strict=True)
        if (
            _local_path_identity(resolved.parent) != _local_path_identity(root)
            or resolved.is_symlink()
            or not resolved.name.startswith(f".{_COLLECTION}.staging-")
        ):
            return
        shutil.rmtree(_operational_path(resolved, force_short=True))
    except (OSError, RasterTextDirectPaddleBundleError):
        return


def materialize_verified_raster_text_direct_server_paddle_bundle_v1(
    *,
    document_preview_root: Path | None = None,
    mobile_bundle: VerifiedRasterTextDirectMobilePaddleBundleV1 | None = None,
    offline_bundle: VerifiedRasterOfflineBundleV1 | None = None,
    mobile_bundle_manifest_sha256: str | None = None,
    offline_bundle_manifest_sha256: str | None = None,
) -> VerifiedRasterTextDirectServerPaddleBundleV1:
    """Create/reuse the exact mobile-detector/server-recognizer release.

    Supply either both verified objects, or ``document_preview_root`` plus the
    two manifest SHA-256 values.  Mixing the two input forms fails closed.
    """

    sources = _resolve_sources(
        document_preview_root=document_preview_root,
        mobile_bundle=mobile_bundle,
        offline_bundle=offline_bundle,
        mobile_bundle_manifest_sha256=mobile_bundle_manifest_sha256,
        offline_bundle_manifest_sha256=offline_bundle_manifest_sha256,
    )
    root = sources.root
    with _lock(root):
        _ensure_directory(root, _COLLECTION)
        releases = _ensure_directory(root, _RELEASES)
        stage_canonical = root / f".{_COLLECTION}.staging-{uuid.uuid4().hex}"
        try:
            if stage_canonical.exists() or stage_canonical.is_symlink():
                raise RasterTextServerPaddleBundleError("server_paddle_bundle_output_invalid")
            stage_canonical.mkdir(mode=0o700)
            stage = _operational_path(stage_canonical, force_short=True)
            manifest_sha256 = _build_stage(stage=stage, sources=sources)
            staged = _load_verified_from_root(
                root=root,
                bundle_root=stage,
                expected_manifest_sha256=manifest_sha256,
                allow_staging=True,
            )
            final_relative = _release_relative_path(manifest_sha256)
            final = _operational_path(releases, force_short=True) / manifest_sha256
            if final.exists() or final.is_symlink():
                return load_verified_raster_text_direct_server_paddle_bundle_v1(
                    document_preview_root=root, manifest_sha256=manifest_sha256
                )
            try:
                os.rename(stage, final)
            except OSError:
                if final.exists() and not final.is_symlink():
                    return load_verified_raster_text_direct_server_paddle_bundle_v1(
                        document_preview_root=root, manifest_sha256=manifest_sha256
                    )
                raise RasterTextServerPaddleBundleError(
                    "server_paddle_bundle_publish_failed"
                ) from None
            return VerifiedRasterTextDirectServerPaddleBundleV1(
                document_preview_root=root,
                bundle_root=final,
                bundle_root_relative_path=final_relative,
                manifest=staged.manifest,
                manifest_sha256=manifest_sha256,
                file_sha256_by_relative_path=staged.file_sha256_by_relative_path,
            )
        except RasterTextServerPaddleBundleError:
            raise
        except (OSError, ValueError, RasterTextDirectPaddleBundleError):
            raise RasterTextServerPaddleBundleError(
                "server_paddle_bundle_materialize_failed"
            ) from None
        finally:
            if stage_canonical.exists():
                _remove_owned_stage(root, stage_canonical)


def prepare_raster_text_direct_server_runtime_profile_v18(
    sidecar_bundle: VerifiedRasterTextDirectPaddleBundleV1,
    model_bundle: VerifiedRasterTextDirectServerPaddleBundleV1,
    *,
    pdfium_version: str,
    minimum_page_confidence: float,
    cpu_threads: int = 6,
) -> PreparedRasterTextDirectServerRuntimeProfileV18:
    """Create/reuse one V18 profile bound to both verified releases."""

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
        raise RasterTextServerPaddleBundleError("server_paddle_runtime_profile_invalid")
    try:
        # Local imports avoid a module-import cycle: runtime verifies this
        # bundle, while this explicit build function validates its wire model.
        from customs_ai_agent.service.raster_text_preview_runtime import (
            RasterTextOfflineProfileV18,
            raster_text_profile_fingerprint_sha256,
        )

        sidecar = load_verified_raster_text_direct_paddle_bundle_v1(
            document_preview_root=sidecar_bundle.document_preview_root,
            manifest_sha256=sidecar_bundle.manifest_sha256,
        )
        server = load_verified_raster_text_direct_server_paddle_bundle_v1(
            document_preview_root=model_bundle.document_preview_root,
            manifest_sha256=model_bundle.manifest_sha256,
        )
        if (
            _local_path_identity(sidecar.document_preview_root)
            != _local_path_identity(server.document_preview_root)
            or sidecar.bundle_root_relative_path != sidecar_bundle.bundle_root_relative_path
            or server.bundle_root_relative_path != model_bundle.bundle_root_relative_path
        ):
            raise RasterTextServerPaddleBundleError("server_paddle_runtime_profile_invalid")
        sidecar_manifest = sidecar.manifest
        server_manifest = server.manifest
        payload: dict[str, object] = {
            "schema_version": "18.0",
            "engine_kind": (
                "direct_paddle_mobile_det_server_rec_harmonic_recall_native960_lowbox_"
                "strip2_semantic_repair_v2_onednn"
            ),
            "runtime_api": "paddle-inference-static-v1",
            "extractor_revision": (
                f"{sidecar_manifest.extractor_revision}:"
                "mobile-det-v18-server-rec-native960-lowbox-strip2-"
                "semantic-repair-v2-onednn"
            ),
            "bundle_root_relative_path": sidecar.bundle_root_relative_path,
            "bundle_manifest_sha256": sidecar.manifest_sha256,
            "mobile_model_bundle_root_relative_path": server.bundle_root_relative_path,
            "mobile_model_bundle_manifest_sha256": server.manifest_sha256,
            "source_model_inventory_relative_path": server.bundle_relative_path(
                server_manifest.provenance_receipt_relative_path
            ),
            "source_model_inventory_sha256": server.file_sha256(
                server_manifest.provenance_receipt_relative_path
            ),
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
            "local_model_root": server.bundle_relative_path(
                server_manifest.local_model_root_relative_path
            ),
            "model_inventory_path": server.bundle_relative_path(
                server_manifest.model_inventory_relative_path
            ),
            "model_inventory_sha256": server.file_sha256(
                server_manifest.model_inventory_relative_path
            ),
            "text_detection_model_name": _MOBILE_DETECTOR_NAME,
            "text_detection_model_directory": "text-det",
            "text_recognition_model_name": _SERVER_RECOGNIZER_NAME,
            "text_recognition_model_directory": "text-rec",
            "enable_mkldnn": True,
            "mkldnn_cache_capacity": 10,
            "recognition_batch_size": 8,
            "opencv_threads": 1,
            "warmup_shape_policy": "detector-736x960-recognizer-48x320-v1",
            "onednn_model_policy": "recognizer-only-v1",
            "recognition_crop_policy": "detector-quad-v1",
            "recognition_resize_interpolation": "linear-v1",
            "ctc_label_policy": "append-space-v1",
        }
        fingerprint = raster_text_profile_fingerprint_sha256(payload)
        payload["profile_fingerprint_sha256"] = fingerprint
        payload["raster_text_profile_revision"] = f"synthetic-raster-text-v1:{fingerprint[:24]}"
        profile = RasterTextOfflineProfileV18.model_validate(payload)
        suffix = profile.raster_text_profile_revision.rsplit(":", 1)[1]
        relative_path = f"raster-text-runtime-profiles-v1/{suffix}/profile.json"
        directory = _ensure_directory(
            server.document_preview_root, f"raster-text-runtime-profiles-v1/{suffix}"
        )
        raw = _write_create_only(
            directory / "profile.json",
            profile.model_dump(mode="json", warnings=False),
            code="server_paddle_runtime_profile_write_failed",
        )
    except RasterTextServerPaddleBundleError:
        raise
    except (
        RasterTextDirectPaddleBundleError,
        OSError,
        TypeError,
        ValidationError,
        ValueError,
    ):
        raise RasterTextServerPaddleBundleError("server_paddle_runtime_profile_invalid") from None
    return PreparedRasterTextDirectServerRuntimeProfileV18(
        raster_text_profile_revision=profile.raster_text_profile_revision,
        runtime_configuration_relative_path=relative_path,
        runtime_configuration_sha256=_sha256_bytes(raw),
        minimum_page_confidence=profile.minimum_page_confidence,
    )


__all__ = [
    "PreparedRasterTextDirectServerRuntimeProfileV18",
    "RasterTextServerPaddleBundleError",
    "RasterTextServerPaddleBundleFileV1",
    "RasterTextServerPaddleBundleManifestV1",
    "VerifiedRasterTextDirectServerPaddleBundleV1",
    "load_verified_raster_text_direct_server_paddle_bundle_v1",
    "materialize_verified_raster_text_direct_server_paddle_bundle_v1",
    "parse_ppocrv5_server_rec_character_dict_v1",
    "prepare_raster_text_direct_server_runtime_profile_v18",
]
