"""Create and verify the minimal offline Paddle static-text bundle.

The older PP-Structure release contains layout/table packages and models that
the editable text-only OCR lane never imports.  This module intentionally does
not call the generic raster bundle resolver: it has one fixed source release,
copies only a reviewable direct-Paddle closure into a private staging tree, and
publishes a separately content-addressed microbundle after a real local smoke
test.  The source release is never selected for runtime use by this module.

There is no wheel download, package install, model fallback, or network
selector here.  A missing/incomplete closure simply prevents publication.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import stat
import subprocess
import sysconfig
import tempfile
import time
from typing import Any, Iterator, Mapping, Sequence
import uuid

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from customs_ai_agent.evaluation.local_process_lock import (
    LocalProcessLockBusy,
    LocalProcessLockInvalid,
    exclusive_local_process_lock,
)
from customs_ai_agent.evaluation.raster_offline_bundle import (
    current_raster_runtime_platform_binding,
)


_SHA256 = "^[a-f0-9]{64}$"
_DOCUMENT_PREVIEW_ROOT_SUFFIX = ("_local", "data", "document-preview")
_BASE_COLLECTION = "raster-offline-bundles-v1"
_BASE_RELEASES = f"{_BASE_COLLECTION}/releases"
# This is deliberately a fixed, unselected source.  The structural selector
# remains owned by the PP-Structure lane and is never read here.
_E3_DIRECT_BASE_MANIFEST_SHA256 = "e3ffc30c907cc6308db8f06e9a94269719e415dded68fa70fd017b2fec253b63"
_DIRECT_COLLECTION = "raster-text-direct-bundles-v1"
_DIRECT_RELEASES = f"{_DIRECT_COLLECTION}/releases"
_DIRECT_INTENTS = f"{_DIRECT_COLLECTION}/intents"
# The profile is deliberately outside a release: calibration changes only its
# confidence floor and must therefore create a separate immutable material,
# not rewrite a content-addressed runtime bundle.
_DIRECT_RUNTIME_PROFILES = "raster-text-runtime-profiles-v1"
_DIRECT_BUNDLE_ID = "raster-text-direct-paddle-bundle-v1"
_MANIFEST_FILE = "manifest.json"
_LOCK_FILE = ".raster-text-direct-paddle-bundle-v1.lock"
_MAX_MANIFEST_BYTES = 16 * 1024 * 1024
_MAX_JSON_BYTES = 1024 * 1024
_MAX_FILE_BYTES = 1024 * 1024 * 1024
_MAX_FILES = 100_000
_DIRECT_SIDECAR_TIMEOUT_SECONDS = 30
_DIRECT_SMOKE_TIMEOUT_SECONDS = 60
# Windows can retain a just-exited child process's directory handle for a
# moment.  Retrying only those two documented sharing/access violations keeps
# publication atomic while avoiding a generic retry policy for filesystem
# failures.  Two fixed retry waits total one second, so publication has at
# most three rename attempts and remains within the project's max-three
# recovery policy.
_PUBLISH_TRANSIENT_WINDOWS_ERROR_CODES = frozenset({5, 32})
# Native smoke exit can leave a short-lived Windows directory/AV handle.
# Retry only the two known sharing/access codes and keep every retry guarded
# by a complete staged-tree revalidation plus an absent destination check.
_PUBLISH_TRANSIENT_RETRY_DELAYS_SECONDS = (0.25, 0.75, 1.5, 3.0, 6.0)
_PUBLISH_MAX_TRANSIENT_RETRIES = len(_PUBLISH_TRANSIENT_RETRY_DELAYS_SECONDS)

_SOURCE_PARENT_INVENTORY = "inventories/parent-renderer.json"
_SOURCE_DIRECT_MODEL_INVENTORY = "inventories/raster-text-models.json"
_SOURCE_DIRECT_MODEL_ROOT = "models/raster-text"
_SOURCE_INTERPRETER_ROOT = "sidecar/interpreter"
# Historical schema-1/e3 alias retained for existing Windows evidence/tests.
_SOURCE_INTERPRETER = f"{_SOURCE_INTERPRETER_ROOT}/python.exe"
_SOURCE_MODULE_ROOT = "sidecar/module-root"

_DIRECT_MODULE_DIRECTORIES = (
    "paddle",
    "numpy",
    "numpy.libs",
    "cv2",
    "PIL",
    "google",
    "setuptools",
    "safetensors",
    "httpx",
    "packaging",
    "click",
    "idna",
    "more_itertools",
    "pyclipper",
    "opt_einsum",
    "tqdm",
    "colorama",
)
_DIRECT_MODULE_FILES = ("typing_extensions.py", "six.py")
# The package closure is intentionally explicit.  In particular, it excludes
# PaddleOCR/PaddleX and every PP-Structure-only layout/table dependency.
_DIRECT_DISTRIBUTIONS: tuple[tuple[str, str, str], ...] = (
    # ``METADATA`` declares this distribution in lower case.  Keep that
    # exact signed source spelling here instead of applying generic
    # case-folding to all distribution metadata.
    ("pillow", "12.3.0", "pillow-12.3.0.dist-info"),
    ("click", "8.4.2", "click-8.4.2.dist-info"),
    ("colorama", "0.4.6", "colorama-0.4.6.dist-info"),
    ("httpx", "0.28.1", "httpx-0.28.1.dist-info"),
    ("idna", "3.19", "idna-3.19.dist-info"),
    ("more-itertools", "11.1.0", "more_itertools-11.1.0.dist-info"),
    ("numpy", "2.3.5", "numpy-2.3.5.dist-info"),
    ("opencv-contrib-python", "4.10.0.84", "opencv_contrib_python-4.10.0.84.dist-info"),
    ("opt-einsum", "3.3.0", "opt_einsum-3.3.0.dist-info"),
    ("paddlepaddle", "3.3.1", "paddlepaddle-3.3.1.dist-info"),
    ("packaging", "26.3", "packaging-26.3.dist-info"),
    ("protobuf", "7.36.0", "protobuf-7.36.0.dist-info"),
    ("pyclipper", "1.4.0", "pyclipper-1.4.0.dist-info"),
    ("safetensors", "0.8.0", "safetensors-0.8.0.dist-info"),
    ("setuptools", "84.0.0", "setuptools-84.0.0.dist-info"),
    ("six", "1.17.0", "six-1.17.0.dist-info"),
    ("tqdm", "4.70.0", "tqdm-4.70.0.dist-info"),
    ("typing_extensions", "4.16.0", "typing_extensions-4.16.0.dist-info"),
)
# The Linux lock resolves Click one patch newer and correctly omits Colorama,
# whose wheel is selected only by Click's Windows environment marker.  Keep
# both closures explicit: deriving a microbundle from whatever happens to be
# installed would silently approve dependency upgrades.  Every selected file
# is still bound to the signed source manifest and rehashed during staging.
_DIRECT_LINUX_MODULE_DIRECTORIES = (
    *(name for name in _DIRECT_MODULE_DIRECTORIES if name != "colorama"),
    # Linux native wheels keep their vendored shared objects beside the
    # import package.  Omitting either directory creates an incomplete wheel
    # closure even though the corresponding Python package is present.
    "opencv_contrib_python.libs",
    "pillow.libs",
)
_DIRECT_LINUX_DISTRIBUTIONS: tuple[tuple[str, str, str], ...] = tuple(
    ("click", "8.5.0", "click-8.5.0.dist-info") if package == "click" else item
    for item in _DIRECT_DISTRIBUTIONS
    if (package := item[0]) != "colorama"
)
_DIRECT_REQUIRED_PACKAGES = (
    "numpy",
    "opencv-contrib-python",
    "paddlepaddle",
    "pyclipper",
)


def _direct_runtime_closure(
    operating_system: str,
) -> tuple[tuple[str, ...], tuple[tuple[str, str, str], ...]]:
    if operating_system == "windows":
        return _DIRECT_MODULE_DIRECTORIES, _DIRECT_DISTRIBUTIONS
    if operating_system == "linux":
        return _DIRECT_LINUX_MODULE_DIRECTORIES, _DIRECT_LINUX_DISTRIBUTIONS
    raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_platform_mismatch")


_DIRECT_MODEL_PATHS = (
    "text-det/inference.json",
    "text-det/inference.pdiparams",
    "text-det/inference.yml",
    "text-rec/config.json",
    "text-rec/inference.json",
    "text-rec/inference.pdiparams",
    "text-rec/inference.yml",
)
_FIRST_PARTY_FILES = (
    "__init__.py",
    "service/__init__.py",
    "service/raster_text_direct_paddle_sidecar.py",
)


class RasterTextDirectPaddleBundleError(RuntimeError):
    """Content-free failure for automatic local microbundle construction."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


class _WireModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


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
        raise ValueError("direct Paddle relative path is invalid")
    return value


class DirectPaddleBundleFileV1(_WireModel):
    path: str = Field(min_length=1, max_length=480)
    sha256: str = Field(pattern=_SHA256)
    byte_size: int = Field(ge=0, le=_MAX_FILE_BYTES)

    _valid_path = field_validator("path")(_relative_path)


class DirectPaddleBundleManifestV1(_WireModel):
    """Exhaustive manifest for one independently runnable direct OCR bundle."""

    schema_version: str
    bundle_id: str
    runtime_operating_system: str | None = None
    runtime_architecture: str | None = None
    runtime_python_abi: str | None = None
    source_bundle_relative_path: str
    source_bundle_manifest_sha256: str = Field(pattern=_SHA256)
    derivation_intent_sha256: str = Field(pattern=_SHA256)
    extractor_revision: str = Field(min_length=1, max_length=256)
    raster_dpi: int = Field(ge=72, le=600)
    sidecar_timeout_seconds: int = Field(ge=1, le=30)
    parent_renderer_package_inventory_relative_path: str
    parent_renderer_required_packages: list[str]
    sidecar_interpreter_root_relative_path: str
    sidecar_interpreter_relative_path: str
    sidecar_interpreter_root_tree_sha256: str = Field(pattern=_SHA256)
    sidecar_module_root_relative_path: str
    sidecar_module_relative_path: str
    sidecar_module_root_tree_sha256: str = Field(pattern=_SHA256)
    sidecar_configuration_template_relative_path: str
    sidecar_package_inventory_relative_path: str
    sidecar_required_packages: list[str]
    local_model_root_relative_path: str
    model_inventory_relative_path: str
    model_root_tree_sha256: str = Field(pattern=_SHA256)
    files: list[DirectPaddleBundleFileV1] = Field(min_length=1, max_length=_MAX_FILES)

    @field_validator(
        "source_bundle_relative_path",
        "parent_renderer_package_inventory_relative_path",
        "sidecar_interpreter_root_relative_path",
        "sidecar_interpreter_relative_path",
        "sidecar_module_root_relative_path",
        "sidecar_module_relative_path",
        "sidecar_configuration_template_relative_path",
        "sidecar_package_inventory_relative_path",
        "local_model_root_relative_path",
        "model_inventory_relative_path",
    )
    @classmethod
    def _valid_paths(cls, value: str) -> str:
        return _relative_path(value)

    @model_validator(mode="after")
    def _fixed_shape(self) -> "DirectPaddleBundleManifestV1":
        if self.schema_version == "1.0":
            platform_valid = (
                self.source_bundle_manifest_sha256 == _E3_DIRECT_BASE_MANIFEST_SHA256
                and self.runtime_operating_system is None
                and self.runtime_architecture is None
                and self.runtime_python_abi is None
                and self.sidecar_interpreter_relative_path == "sidecar/interpreter/python.exe"
            )
        elif self.schema_version == "2.0":
            expected_interpreter = (
                "sidecar/interpreter/python.exe"
                if self.runtime_operating_system == "windows"
                else "sidecar/interpreter/bin/python3.11"
            )
            platform_valid = (
                self.runtime_operating_system in {"windows", "linux"}
                and self.runtime_architecture == "amd64"
                and self.runtime_python_abi == "cp311"
                and self.sidecar_interpreter_relative_path == expected_interpreter
            )
        else:
            platform_valid = False
        if (
            not platform_valid
            or self.bundle_id != _DIRECT_BUNDLE_ID
            or self.source_bundle_relative_path
            != _source_bundle_relative_path(self.source_bundle_manifest_sha256)
            or self.sidecar_timeout_seconds != _DIRECT_SIDECAR_TIMEOUT_SECONDS
            or self.parent_renderer_package_inventory_relative_path
            != "inventories/parent-renderer.json"
            or self.parent_renderer_required_packages != ["pypdfium2", "Pillow"]
            or self.sidecar_interpreter_root_relative_path != "sidecar/interpreter"
            or self.sidecar_module_root_relative_path != "sidecar/module-root"
            or self.sidecar_module_relative_path
            != "sidecar/module-root/customs_ai_agent/service/raster_text_direct_paddle_sidecar.py"
            or self.sidecar_configuration_template_relative_path
            != "sidecar/direct-configuration-template.json"
            or self.sidecar_package_inventory_relative_path
            != "inventories/direct-paddle-packages.json"
            or self.sidecar_required_packages != list(_DIRECT_REQUIRED_PACKAGES)
            or self.local_model_root_relative_path != "models"
            or self.model_inventory_relative_path != "inventories/direct-paddle-models.json"
        ):
            raise ValueError("direct Paddle bundle manifest shape is invalid")
        paths = [item.path for item in self.files]
        if paths != sorted(paths) or len(paths) != len(set(paths)):
            raise ValueError("direct Paddle bundle file manifest is invalid")
        return self


class DirectPaddleBundleIntentV1(_WireModel):
    """Immutable source selection used to safely reuse one microbundle."""

    schema_version: str
    artifact_kind: str
    source_bundle_relative_path: str
    source_bundle_manifest_sha256: str = Field(pattern=_SHA256)
    source_parent_renderer_inventory_sha256: str = Field(pattern=_SHA256)
    source_direct_model_inventory_sha256: str = Field(pattern=_SHA256)
    source_direct_model_snapshot_sha256: str = Field(pattern=_SHA256)
    source_interpreter_manifest_tree_sha256: str = Field(pattern=_SHA256)
    source_module_closure_manifest_tree_sha256: str = Field(pattern=_SHA256)
    first_party_files: list[DirectPaddleBundleFileV1]

    @field_validator("source_bundle_relative_path")
    @classmethod
    def _valid_source_path(cls, value: str) -> str:
        return _relative_path(value)

    @model_validator(mode="after")
    def _fixed_shape(self) -> "DirectPaddleBundleIntentV1":
        if (
            self.schema_version != "1.0"
            or self.artifact_kind != "raster_text_direct_paddle_bundle_intent"
            or self.source_bundle_relative_path
            != _source_bundle_relative_path(self.source_bundle_manifest_sha256)
        ):
            raise ValueError("direct Paddle bundle intent is invalid")
        paths = [item.path for item in self.first_party_files]
        if paths != list(_FIRST_PARTY_FILES):
            raise ValueError("direct Paddle sidecar source intent is invalid")
        return self


class DirectPaddleBundleIntentIndexV1(_WireModel):
    schema_version: str
    artifact_kind: str
    derivation_intent: DirectPaddleBundleIntentV1
    derivation_intent_sha256: str = Field(pattern=_SHA256)
    release_manifest_sha256: str = Field(pattern=_SHA256)
    release_relative_path: str

    @field_validator("release_relative_path")
    @classmethod
    def _valid_release_path(cls, value: str) -> str:
        return _relative_path(value)

    @model_validator(mode="after")
    def _shape(self) -> "DirectPaddleBundleIntentIndexV1":
        if (
            self.schema_version != "1.0"
            or self.artifact_kind != "raster_text_direct_paddle_bundle_intent_index"
            or self.derivation_intent_sha256 != _model_sha256(self.derivation_intent)
            or self.release_relative_path != _release_relative_path(self.release_manifest_sha256)
        ):
            raise ValueError("direct Paddle bundle intent index is invalid")
        return self


@dataclass(frozen=True, slots=True)
class DirectPaddleBundleSourceV1:
    document_preview_root: Path
    source_bundle_root: Path
    source_bundle_relative_path: str
    source_manifest_sha256: str
    extractor_revision: str
    raster_dpi: int
    source_sidecar_timeout_seconds: int
    runtime_operating_system: str
    runtime_architecture: str
    runtime_python_abi: str
    source_interpreter_relative_path: str
    file_records: Mapping[str, DirectPaddleBundleFileV1]
    parent_renderer_inventory_sha256: str
    direct_model_inventory_sha256: str
    direct_model_snapshot_sha256: str


@dataclass(frozen=True, slots=True)
class VerifiedRasterTextDirectPaddleBundleV1:
    """One direct OCR binding with every runtime path inside the microbundle."""

    document_preview_root: Path
    bundle_root: Path
    bundle_root_relative_path: str
    manifest: DirectPaddleBundleManifestV1
    manifest_sha256: str
    file_sha256_by_relative_path: Mapping[str, str]

    def bundle_relative_path(self, path: str) -> str:
        _relative_path(path)
        return f"{self.bundle_root_relative_path}/{path}"

    def file_sha256(self, path: str) -> str:
        try:
            return self.file_sha256_by_relative_path[path]
        except KeyError:
            raise RasterTextDirectPaddleBundleError(
                "direct_paddle_bundle_manifest_invalid"
            ) from None

    @property
    def raster_dpi(self) -> int:
        return self.manifest.raster_dpi

    @property
    def extractor_revision(self) -> str:
        return self.manifest.extractor_revision

    @property
    def sidecar_timeout_seconds(self) -> int:
        return self.manifest.sidecar_timeout_seconds


@dataclass(frozen=True, slots=True)
class PreparedRasterTextDirectPaddleRuntimeProfileV2:
    """One immutable V2 runtime profile derived from a verified microbundle.

    The calibrated page-confidence floor participates in the profile
    fingerprint.  Thus a calibration result can never mutate or silently
    reuse the configuration that was evaluated with a different floor.
    """

    raster_text_profile_revision: str
    runtime_configuration_relative_path: str
    runtime_configuration_sha256: str
    minimum_page_confidence: float


def _model_sha256(model: BaseModel) -> str:
    return _sha256_bytes(_canonical_json(model.model_dump(mode="json", warnings=False)))


def _source_bundle_relative_path(manifest_sha256: str | None = None) -> str:
    if manifest_sha256 is None:
        manifest_sha256 = _E3_DIRECT_BASE_MANIFEST_SHA256
    if len(manifest_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in manifest_sha256
    ):
        raise ValueError("source bundle manifest hash is invalid")
    return f"{_BASE_RELEASES}/{manifest_sha256}"


def _release_relative_path(manifest_sha256: str) -> str:
    if len(manifest_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in manifest_sha256
    ):
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_manifest_invalid")
    return f"{_DIRECT_RELEASES}/{manifest_sha256}"


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
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_root_invalid") from None


def _operational_path(path: Path, *, force_short: bool = False) -> Path:
    """Return a Windows short spelling for deep native Paddle paths.

    The canonical long path remains the identity checked by the verifier.  A
    copied CPython/Paddle closure can exceed legacy Win32 path limits below
    the Korean-named workspace, so only actual filesystem operations use the
    equivalent existing 8.3 spelling.
    """

    if os.name != "nt":
        return path
    value = str(path)
    # A release root can be just below MAX_PATH while a legitimate child
    # (for example ``models/raster-text/text-det/inference.pdiparams``) is
    # well above it.  Keep a generous child-path margin instead of deciding
    # only from the directory currently being opened.
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
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_path_unsupported") from None


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


def _safe_path(root: Path, relative: str, *, directory: bool = False) -> Path:
    _relative_path(relative)
    candidate = root
    try:
        for part in PurePosixPath(relative).parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise OSError
        if (
            candidate.is_symlink()
            or not _same_or_child(root, candidate)
            or (directory and not candidate.is_dir())
            or (not directory and not candidate.is_file())
        ):
            raise OSError
        return _operational_path(candidate)
    except OSError:
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_path_invalid") from None


def _read_canonical_object(path: Path, *, limit: int, code: str) -> tuple[dict[str, Any], bytes]:
    try:
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= limit:
            raise OSError
        raw = path.read_bytes()
        value = json.loads(raw)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise RasterTextDirectPaddleBundleError(code) from None
    if not isinstance(value, dict) or raw != _canonical_json(value):
        raise RasterTextDirectPaddleBundleError(code)
    return value, raw


def _sha256_regular_file(path: Path, *, code: str) -> tuple[str, int]:
    try:
        before = path.lstat()
        if stat.S_ISLNK(before.st_mode) or not stat.S_ISREG(before.st_mode):
            raise OSError
        size = before.st_size
        if size < 0 or size > _MAX_FILE_BYTES:
            raise OSError
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        after = path.lstat()
        if (
            stat.S_ISLNK(after.st_mode)
            or not stat.S_ISREG(after.st_mode)
            or after.st_size != size
            or (before.st_dev, before.st_ino) != (after.st_dev, after.st_ino)
        ):
            raise OSError
        return digest.hexdigest(), size
    except OSError:
        raise RasterTextDirectPaddleBundleError(code) from None


def _regular_tree_files(root: Path, *, code: str) -> dict[str, Path]:
    """Walk one tree without allowing a link or special entry into a bundle."""

    try:
        root_metadata = root.lstat()
        if stat.S_ISLNK(root_metadata.st_mode) or not stat.S_ISDIR(root_metadata.st_mode):
            raise OSError
        files: dict[str, Path] = {}
        for path in sorted(root.rglob("*"), key=lambda value: value.as_posix()):
            metadata = path.lstat()
            if stat.S_ISLNK(metadata.st_mode) or not (
                stat.S_ISDIR(metadata.st_mode) or stat.S_ISREG(metadata.st_mode)
            ):
                raise OSError
            if stat.S_ISREG(metadata.st_mode):
                relative = path.relative_to(root).as_posix()
                if not relative or relative in files or len(relative) > 480:
                    raise OSError
                files[relative] = path
        if len(files) > _MAX_FILES:
            raise OSError
        return files
    except (OSError, ValueError):
        raise RasterTextDirectPaddleBundleError(code) from None


def _parse_file_records(raw: object, *, code: str) -> dict[str, DirectPaddleBundleFileV1]:
    if not isinstance(raw, list) or not raw:
        raise RasterTextDirectPaddleBundleError(code)
    try:
        records = [DirectPaddleBundleFileV1.model_validate(item) for item in raw]
    except ValidationError:
        raise RasterTextDirectPaddleBundleError(code) from None
    paths = [item.path for item in records]
    if paths != sorted(paths) or len(paths) != len(set(paths)):
        raise RasterTextDirectPaddleBundleError(code)
    return {item.path: item for item in records}


def _record_for(
    records: Mapping[str, DirectPaddleBundleFileV1], *, path: str, code: str
) -> DirectPaddleBundleFileV1:
    try:
        return records[path]
    except KeyError:
        raise RasterTextDirectPaddleBundleError(code) from None


def _verify_expected_file(
    *, root: Path, relative: str, record: DirectPaddleBundleFileV1, code: str
) -> Path:
    path = _safe_path(root, relative)
    digest, size = _sha256_regular_file(path, code=code)
    if digest != record.sha256 or size != record.byte_size:
        raise RasterTextDirectPaddleBundleError(code)
    return path


def _tree_manifest_sha256(records: Sequence[DirectPaddleBundleFileV1]) -> str:
    return _sha256_bytes(
        _canonical_json(
            {
                "files": [
                    {"path": item.path, "sha256": item.sha256, "byte_size": item.byte_size}
                    for item in sorted(records, key=lambda item: item.path)
                ]
            }
        )
    )


def _records_under(
    records: Mapping[str, DirectPaddleBundleFileV1], *, prefix: str
) -> list[DirectPaddleBundleFileV1]:
    root = PurePosixPath(prefix)
    selected: list[DirectPaddleBundleFileV1] = []
    for path, record in records.items():
        try:
            relative = PurePosixPath(path).relative_to(root)
        except ValueError:
            continue
        if relative != PurePosixPath("."):
            selected.append(record)
    return sorted(selected, key=lambda item: item.path)


def _verify_parent_inventory(path: Path, *, expected_record: DirectPaddleBundleFileV1) -> str:
    digest, size = _sha256_regular_file(path, code="direct_paddle_bundle_parent_inventory_invalid")
    if digest != expected_record.sha256 or size != expected_record.byte_size:
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_parent_inventory_invalid")
    value, raw = _read_canonical_object(
        path, limit=_MAX_JSON_BYTES, code="direct_paddle_bundle_parent_inventory_invalid"
    )
    packages = value.get("packages")
    if not isinstance(packages, list):
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_parent_inventory_invalid")
    versions: dict[str, str] = {}
    for package in packages:
        if (
            not isinstance(package, dict)
            or set(package) != {"name", "version"}
            or not isinstance(package.get("name"), str)
            or not isinstance(package.get("version"), str)
            or not package["name"]
            or not package["version"]
            or package["name"] in versions
        ):
            raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_parent_inventory_invalid")
        versions[package["name"]] = package["version"]
    if (
        versions
        != {
            "pypdfium2": versions.get("pypdfium2"),
            "Pillow": versions.get("Pillow"),
        }
        or None in versions.values()
    ):
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_parent_inventory_invalid")
    try:
        purelib_value = sysconfig.get_path("purelib")
        if not isinstance(purelib_value, str) or not purelib_value:
            raise OSError
        purelib = Path(purelib_value)
        if purelib.is_symlink():
            raise OSError
        purelib = purelib.resolve(strict=True)
        if not purelib.is_dir():
            raise OSError
        installed: dict[str, str] = {}
        for package_name in ("pypdfium2", "Pillow"):
            distributions = list(
                importlib.metadata.distributions(
                    name=package_name,
                    path=[str(purelib)],
                )
            )
            if len(distributions) != 1:
                raise OSError
            distribution = distributions[0]
            declared_name = distribution.metadata["Name"]
            if (
                not isinstance(declared_name, str)
                or declared_name.casefold() != package_name.casefold()
                or not distribution.version
            ):
                raise OSError
            installed[package_name] = distribution.version
        if installed != versions:
            raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_parent_renderer_drift")
    except (OSError, importlib.metadata.PackageNotFoundError):
        raise RasterTextDirectPaddleBundleError(
            "direct_paddle_bundle_parent_renderer_drift"
        ) from None
    return _sha256_bytes(raw)


def _verify_source_direct_models(
    *, source_root: Path, records: Mapping[str, DirectPaddleBundleFileV1]
) -> tuple[str, str]:
    inventory_relative = _SOURCE_DIRECT_MODEL_INVENTORY
    inventory_record = _record_for(
        records, path=inventory_relative, code="direct_paddle_bundle_source_model_invalid"
    )
    inventory_path = _verify_expected_file(
        root=source_root,
        relative=inventory_relative,
        record=inventory_record,
        code="direct_paddle_bundle_source_model_invalid",
    )
    inventory, raw = _read_canonical_object(
        inventory_path, limit=_MAX_JSON_BYTES, code="direct_paddle_bundle_source_model_invalid"
    )
    required = {
        "schema_version",
        "inventory_kind",
        "source_bundle_manifest_sha256",
        "source_model_root_relative_path",
        "source_model_inventory_sha256",
        "source_snapshot_sha256",
        "artifacts",
    }
    if (
        set(inventory) != required
        or inventory.get("schema_version") != "1.0"
        or inventory.get("inventory_kind") != "raster-text-direct-models-v1"
        or inventory.get("source_model_root_relative_path") != "models"
        or not isinstance(inventory.get("artifacts"), list)
    ):
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_source_model_invalid")
    artifacts = inventory["artifacts"]
    if len(artifacts) != len(_DIRECT_MODEL_PATHS):
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_source_model_invalid")
    normalized: list[dict[str, object]] = []
    source_model_root = _safe_path(source_root, _SOURCE_DIRECT_MODEL_ROOT, directory=True)
    actual = _regular_tree_files(
        source_model_root, code="direct_paddle_bundle_source_model_invalid"
    )
    if tuple(sorted(actual)) != _DIRECT_MODEL_PATHS:
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_source_model_invalid")
    for expected_path, row in zip(_DIRECT_MODEL_PATHS, artifacts, strict=True):
        if (
            not isinstance(row, dict)
            or set(row) != {"relative_path", "sha256", "byte_size"}
            or row.get("relative_path") != expected_path
            or not isinstance(row.get("sha256"), str)
            or len(row["sha256"]) != 64
            or not isinstance(row.get("byte_size"), int)
            or isinstance(row["byte_size"], bool)
        ):
            raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_source_model_invalid")
        source_relative = f"{_SOURCE_DIRECT_MODEL_ROOT}/{expected_path}"
        source_record = _record_for(
            records, path=source_relative, code="direct_paddle_bundle_source_model_invalid"
        )
        source_path = _verify_expected_file(
            root=source_root,
            relative=source_relative,
            record=source_record,
            code="direct_paddle_bundle_source_model_invalid",
        )
        if (
            source_record.sha256 != row["sha256"]
            or source_record.byte_size != row["byte_size"]
            or _local_path_identity(source_path) != _local_path_identity(actual[expected_path])
        ):
            raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_source_model_invalid")
        normalized.append(
            {
                "relative_path": expected_path,
                "sha256": source_record.sha256,
                "byte_size": source_record.byte_size,
            }
        )
    snapshot = _sha256_bytes(_canonical_json({"files": normalized}))
    if inventory.get("source_snapshot_sha256") != snapshot:
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_source_model_invalid")
    return _sha256_bytes(raw), snapshot


def verify_e3_raster_text_direct_paddle_base_v1(
    *,
    document_preview_root: Path,
    source_manifest_sha256: str | None = None,
) -> DirectPaddleBundleSourceV1:
    """Attest one explicit parent inventory, interpreter, and seven models.

    The historical e3 Windows release retains its narrow fixed-hash path.
    A caller may explicitly select a schema-2 content-addressed release for
    Linux; that path verifies the manifest hash, exact host identity, every
    selected source record, and every copied byte. No selector discovery or
    fallback is permitted.
    """

    root = _root(document_preview_root)
    if source_manifest_sha256 is None:
        source_manifest_sha256 = _E3_DIRECT_BASE_MANIFEST_SHA256
    source_relative = _source_bundle_relative_path(source_manifest_sha256)
    source_root = _safe_path(root, source_relative, directory=True)
    manifest_path = _safe_path(source_root, _MANIFEST_FILE)
    manifest_raw, manifest_bytes = _read_canonical_object(
        manifest_path,
        limit=_MAX_MANIFEST_BYTES,
        code="direct_paddle_bundle_source_manifest_invalid",
    )
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    if manifest_sha256 != source_manifest_sha256:
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_source_manifest_invalid")
    legacy_source = source_manifest_sha256 == _E3_DIRECT_BASE_MANIFEST_SHA256
    source_schema = manifest_raw.get("schema_version")
    operating_system = manifest_raw.get("runtime_operating_system")
    architecture = manifest_raw.get("runtime_architecture")
    python_abi = manifest_raw.get("runtime_python_abi")
    source_interpreter = manifest_raw.get("sidecar_interpreter_relative_path")
    platform_valid = (
        legacy_source
        and source_schema == "1.0"
        and operating_system is None
        and architecture is None
        and python_abi is None
        and source_interpreter == "sidecar/interpreter/python.exe"
    ) or (
        not legacy_source
        and source_schema == "2.0"
        and operating_system in {"windows", "linux"}
        and architecture == "amd64"
        and python_abi == "cp311"
        and source_interpreter
        == (
            "sidecar/interpreter/python.exe"
            if operating_system == "windows"
            else "sidecar/interpreter/bin/python3.11"
        )
        and (operating_system, architecture, python_abi)
        == current_raster_runtime_platform_binding()
    )
    if (
        not platform_valid
        or manifest_raw.get("bundle_id") != "raster-offline-bundle-v1"
        or manifest_raw.get("parent_renderer_package_inventory_relative_path")
        != _SOURCE_PARENT_INVENTORY
        or manifest_raw.get("raster_text_model_inventory_relative_path")
        != _SOURCE_DIRECT_MODEL_INVENTORY
        or manifest_raw.get("raster_text_local_model_root_relative_path")
        != _SOURCE_DIRECT_MODEL_ROOT
        or manifest_raw.get("sidecar_interpreter_root_relative_path") != _SOURCE_INTERPRETER_ROOT
        or manifest_raw.get("sidecar_module_root_relative_path") != _SOURCE_MODULE_ROOT
        or not isinstance(manifest_raw.get("extractor_revision"), str)
        or not isinstance(manifest_raw.get("raster_dpi"), int)
        or isinstance(manifest_raw.get("raster_dpi"), bool)
        or not 72 <= manifest_raw["raster_dpi"] <= 600
        or not isinstance(manifest_raw.get("sidecar_timeout_seconds"), int)
        or isinstance(manifest_raw.get("sidecar_timeout_seconds"), bool)
    ):
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_source_manifest_invalid")
    records = _parse_file_records(
        manifest_raw.get("files"), code="direct_paddle_bundle_source_manifest_invalid"
    )
    # The paths are declared but not fully rehashed here.  They are copied
    # only through the fixed allowlist below and rehashed in staging.
    if not isinstance(source_interpreter, str):
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_source_manifest_invalid")
    for required in (
        source_interpreter,
        _SOURCE_PARENT_INVENTORY,
        _SOURCE_DIRECT_MODEL_INVENTORY,
    ):
        _record_for(records, path=required, code="direct_paddle_bundle_source_manifest_invalid")
    parent_record = _record_for(
        records, path=_SOURCE_PARENT_INVENTORY, code="direct_paddle_bundle_source_manifest_invalid"
    )
    parent_sha = _verify_parent_inventory(
        _safe_path(source_root, _SOURCE_PARENT_INVENTORY), expected_record=parent_record
    )
    model_inventory_sha, model_snapshot_sha = _verify_source_direct_models(
        source_root=source_root, records=records
    )
    # These directories are not recursively verified during the narrow base
    # check.  Each selected file is verified against the manifest while copied.
    _safe_path(source_root, _SOURCE_INTERPRETER_ROOT, directory=True)
    _safe_path(source_root, _SOURCE_MODULE_ROOT, directory=True)
    return DirectPaddleBundleSourceV1(
        document_preview_root=root,
        source_bundle_root=source_root,
        source_bundle_relative_path=source_relative,
        source_manifest_sha256=manifest_sha256,
        extractor_revision=manifest_raw["extractor_revision"],
        raster_dpi=manifest_raw["raster_dpi"],
        source_sidecar_timeout_seconds=manifest_raw["sidecar_timeout_seconds"],
        runtime_operating_system=("windows" if legacy_source else str(operating_system)),
        runtime_architecture=("amd64" if legacy_source else str(architecture)),
        runtime_python_abi=("cp311" if legacy_source else str(python_abi)),
        source_interpreter_relative_path=source_interpreter,
        file_records=records,
        parent_renderer_inventory_sha256=parent_sha,
        direct_model_inventory_sha256=model_inventory_sha,
        direct_model_snapshot_sha256=model_snapshot_sha,
    )


def _project_root() -> Path:
    # ``.../src/customs_ai_agent/evaluation/<this file>`` → project root.
    return Path(__file__).resolve().parents[3]


def _snapshot_first_party_sidecar() -> tuple[DirectPaddleBundleFileV1, ...]:
    source_root = _project_root() / "src" / "customs_ai_agent"
    records: list[DirectPaddleBundleFileV1] = []
    for relative in _FIRST_PARTY_FILES:
        try:
            candidate = source_root.joinpath(*PurePosixPath(relative).parts)
            if candidate.is_symlink() or not candidate.is_file():
                raise OSError
            digest, size = _sha256_regular_file(
                candidate, code="direct_paddle_bundle_sidecar_source_invalid"
            )
        except (OSError, RasterTextDirectPaddleBundleError):
            raise RasterTextDirectPaddleBundleError(
                "direct_paddle_bundle_sidecar_source_invalid"
            ) from None
        records.append(DirectPaddleBundleFileV1(path=relative, sha256=digest, byte_size=size))
    return tuple(records)


def _selected_source_records(
    source: DirectPaddleBundleSourceV1,
) -> tuple[list[DirectPaddleBundleFileV1], list[DirectPaddleBundleFileV1]]:
    """Return fixed manifest records for interpreter and direct package closure."""

    interpreter = _records_under(source.file_records, prefix=_SOURCE_INTERPRETER_ROOT)
    if not interpreter or {item.path for item in interpreter} != {
        f"{_SOURCE_INTERPRETER_ROOT}/{relative}"
        for relative in _regular_tree_files(
            _safe_path(source.source_bundle_root, _SOURCE_INTERPRETER_ROOT, directory=True),
            code="direct_paddle_bundle_source_interpreter_invalid",
        )
    }:
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_source_interpreter_invalid")
    selected: list[DirectPaddleBundleFileV1] = []
    module_root = _safe_path(source.source_bundle_root, _SOURCE_MODULE_ROOT, directory=True)
    module_directories, distributions = _direct_runtime_closure(
        source.runtime_operating_system
    )
    roots: list[tuple[str, bool]] = [
        *((name, True) for name in module_directories),
        *((name, False) for name in _DIRECT_MODULE_FILES),
        *((name, True) for _package, _version, name in distributions),
    ]
    seen: set[str] = set()
    for name, directory in roots:
        relative_root = f"{_SOURCE_MODULE_ROOT}/{name}"
        if directory:
            tree = _safe_path(module_root, name, directory=True)
            actual = _regular_tree_files(tree, code="direct_paddle_bundle_source_closure_invalid")
            expected_paths = {f"{relative_root}/{relative}" for relative in actual}
            declared = _records_under(source.file_records, prefix=relative_root)
            if not actual or {item.path for item in declared} != expected_paths:
                raise RasterTextDirectPaddleBundleError(
                    "direct_paddle_bundle_source_closure_invalid"
                )
            selected.extend(declared)
        else:
            _safe_path(module_root, name)
            record = _record_for(
                source.file_records,
                path=relative_root,
                code="direct_paddle_bundle_source_closure_invalid",
            )
            selected.append(record)
    for record in selected:
        if record.path in seen:
            raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_source_closure_invalid")
        seen.add(record.path)
    # The allowlist must never accidentally capture the high-level wrappers.
    forbidden_prefixes = (
        f"{_SOURCE_MODULE_ROOT}/paddleocr",
        f"{_SOURCE_MODULE_ROOT}/paddlex",
        f"{_SOURCE_MODULE_ROOT}/modelscope",
    )
    if any(record.path.startswith(forbidden_prefixes) for record in selected):
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_source_closure_invalid")
    return interpreter, sorted(selected, key=lambda item: item.path)


def _verify_distribution_metadata(module_root: Path, *, operating_system: str) -> None:
    """Ensure every copied ``*.dist-info`` names the fixed allowed version."""

    _module_directories, distributions = _direct_runtime_closure(operating_system)
    for package, version, directory in distributions:
        metadata = _safe_path(module_root, f"{directory}/METADATA")
        try:
            raw = metadata.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            raise RasterTextDirectPaddleBundleError(
                "direct_paddle_bundle_source_closure_invalid"
            ) from None
        values: dict[str, str] = {}
        for line in raw.splitlines():
            if line.startswith("Name: ") and "Name" not in values:
                values["Name"] = line[6:]
            elif line.startswith("Version: ") and "Version" not in values:
                values["Version"] = line[9:]
        if values != {"Name": package, "Version": version}:
            raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_source_closure_invalid")


def _intent_from_source(
    source: DirectPaddleBundleSourceV1,
    *,
    interpreter_records: Sequence[DirectPaddleBundleFileV1],
    module_records: Sequence[DirectPaddleBundleFileV1],
    first_party_records: Sequence[DirectPaddleBundleFileV1],
) -> DirectPaddleBundleIntentV1:
    return DirectPaddleBundleIntentV1(
        schema_version="1.0",
        artifact_kind="raster_text_direct_paddle_bundle_intent",
        source_bundle_relative_path=source.source_bundle_relative_path,
        source_bundle_manifest_sha256=source.source_manifest_sha256,
        source_parent_renderer_inventory_sha256=source.parent_renderer_inventory_sha256,
        source_direct_model_inventory_sha256=source.direct_model_inventory_sha256,
        source_direct_model_snapshot_sha256=source.direct_model_snapshot_sha256,
        source_interpreter_manifest_tree_sha256=_tree_manifest_sha256(interpreter_records),
        source_module_closure_manifest_tree_sha256=_tree_manifest_sha256(module_records),
        first_party_files=list(first_party_records),
    )


def _ensure_directory(root: Path, relative: str, *, code: str) -> Path:
    _relative_path(relative)
    destination = root
    try:
        for part in PurePosixPath(relative).parts:
            destination = destination / part
            if destination.is_symlink():
                raise OSError
            destination.mkdir(mode=0o700, exist_ok=True)
            if (
                destination.is_symlink()
                or not destination.is_dir()
                or not _same_or_child(root, destination)
            ):
                raise OSError
        return _operational_path(destination)
    except OSError:
        raise RasterTextDirectPaddleBundleError(code) from None


def _copy_expected_file(
    *,
    source_root: Path,
    source_relative: str,
    record: DirectPaddleBundleFileV1,
    stage: Path,
    destination_relative: str,
    code: str,
) -> None:
    source = _verify_expected_file(
        root=source_root, relative=source_relative, record=record, code=code
    )
    destination = stage.joinpath(*PurePosixPath(destination_relative).parts)
    try:
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.parent.is_symlink() or destination.exists():
            raise OSError
        shutil.copy2(source, destination, follow_symlinks=False)
    except OSError:
        raise RasterTextDirectPaddleBundleError(code) from None
    digest, size = _sha256_regular_file(destination, code=code)
    if digest != record.sha256 or size != record.byte_size:
        raise RasterTextDirectPaddleBundleError(code)


def _copy_current_first_party_file(
    *, stage: Path, record: DirectPaddleBundleFileV1, code: str
) -> None:
    source = _project_root() / "src" / "customs_ai_agent" / Path(record.path)
    destination = stage / "sidecar" / "module-root" / "customs_ai_agent" / Path(record.path)
    try:
        if source.is_symlink() or not source.is_file() or destination.exists():
            raise OSError
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination, follow_symlinks=False)
    except OSError:
        raise RasterTextDirectPaddleBundleError(code) from None
    digest, size = _sha256_regular_file(destination, code=code)
    if digest != record.sha256 or size != record.byte_size:
        raise RasterTextDirectPaddleBundleError(code)


def _write_create_only(path: Path, value: object, *, code: str) -> bytes:
    payload = _canonical_json(value)
    try:
        if path.is_symlink():
            raise OSError
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        try:
            if path.is_symlink() or path.read_bytes() != payload:
                raise OSError
        except OSError:
            raise RasterTextDirectPaddleBundleError(code) from None
    except OSError:
        raise RasterTextDirectPaddleBundleError(code) from None
    return payload


def _direct_package_inventory(*, operating_system: str) -> dict[str, object]:
    _module_directories, distributions = _direct_runtime_closure(operating_system)
    return {
        "inventory_kind": "raster_text_direct_paddle_packages",
        "schema_version": "1.0",
        "packages": [
            {"name": package, "version": version}
            for package, version, _directory in distributions
        ],
    }


def _direct_model_inventory(source: DirectPaddleBundleSourceV1) -> dict[str, object]:
    artifacts: list[dict[str, object]] = []
    for relative in _DIRECT_MODEL_PATHS:
        record = _record_for(
            source.file_records,
            path=f"{_SOURCE_DIRECT_MODEL_ROOT}/{relative}",
            code="direct_paddle_bundle_source_model_invalid",
        )
        artifacts.append(
            {"relative_path": relative, "sha256": record.sha256, "byte_size": record.byte_size}
        )
    return {
        "schema_version": "1.0",
        "inventory_kind": "raster_text_direct_paddle_models",
        "source_bundle_manifest_sha256": source.source_manifest_sha256,
        "source_direct_model_snapshot_sha256": source.direct_model_snapshot_sha256,
        "artifacts": artifacts,
    }


def _direct_configuration_template(
    *, source: DirectPaddleBundleSourceV1, intent_sha256: str
) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "engine_kind": "paddle_static_text_poc",
        "runtime_api": "paddle-inference-static-v1",
        "extractor_revision": f"{source.extractor_revision}:raster-text-direct-paddle-v1",
        "raster_text_profile_revision": f"synthetic-raster-text-v1:{intent_sha256[:24]}",
        "text_detection_model_name": "PP-OCRv5_server_det",
        "text_detection_model_directory": "text-det",
        "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
        "text_recognition_model_directory": "text-rec",
        "cpu_threads": 6,
        "text_detection_limit_side_len": 2048,
        "text_detection_limit_type": "max",
        "text_detection_threshold": 0.3,
        "text_detection_box_threshold": 0.6,
        "text_detection_unclip_ratio": 1.5,
    }


def _stage_files(stage: Path) -> list[DirectPaddleBundleFileV1]:
    files = _regular_tree_files(stage, code="direct_paddle_bundle_stage_invalid")
    if _MANIFEST_FILE in files:
        files.pop(_MANIFEST_FILE)
    result: list[DirectPaddleBundleFileV1] = []
    for relative, path in sorted(files.items()):
        sha256, byte_size = _sha256_regular_file(path, code="direct_paddle_bundle_stage_invalid")
        result.append(DirectPaddleBundleFileV1(path=relative, sha256=sha256, byte_size=byte_size))
    return result


def _tree_digest_from_manifest(manifest: DirectPaddleBundleManifestV1, *, directory: str) -> str:
    prefix = PurePosixPath(directory)
    files: list[DirectPaddleBundleFileV1] = []
    for item in manifest.files:
        try:
            relative = PurePosixPath(item.path).relative_to(prefix)
        except ValueError:
            continue
        if relative == PurePosixPath("."):
            raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_manifest_invalid")
        files.append(
            DirectPaddleBundleFileV1(
                path=relative.as_posix(), sha256=item.sha256, byte_size=item.byte_size
            )
        )
    if not files:
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_manifest_invalid")
    return _tree_manifest_sha256(files)


def _verify_direct_package_inventory(
    path: Path, *, expected_sha256: str, operating_system: str
) -> None:
    value, raw = _read_canonical_object(
        path, limit=_MAX_JSON_BYTES, code="direct_paddle_bundle_manifest_invalid"
    )
    if _sha256_bytes(raw) != expected_sha256 or value != _direct_package_inventory(
        operating_system=operating_system
    ):
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_manifest_invalid")


def _verify_direct_model_inventory(
    *, bundle_root: Path, manifest: DirectPaddleBundleManifestV1, expected_sha256: str
) -> None:
    inventory_path = _safe_path(bundle_root, manifest.model_inventory_relative_path)
    value, raw = _read_canonical_object(
        inventory_path, limit=_MAX_JSON_BYTES, code="direct_paddle_bundle_manifest_invalid"
    )
    if _sha256_bytes(raw) != expected_sha256:
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_manifest_invalid")
    artifacts = value.get("artifacts") if isinstance(value, dict) else None
    if (
        not isinstance(artifacts, list)
        or len(artifacts) != len(_DIRECT_MODEL_PATHS)
        or value.get("schema_version") != "1.0"
        or value.get("inventory_kind") != "raster_text_direct_paddle_models"
        or value.get("source_bundle_manifest_sha256") != manifest.source_bundle_manifest_sha256
        or not isinstance(value.get("source_direct_model_snapshot_sha256"), str)
    ):
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_manifest_invalid")
    model_root = _safe_path(bundle_root, manifest.local_model_root_relative_path, directory=True)
    actual = _regular_tree_files(model_root, code="direct_paddle_bundle_manifest_invalid")
    if tuple(sorted(actual)) != _DIRECT_MODEL_PATHS:
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_manifest_invalid")
    for expected_relative, artifact in zip(_DIRECT_MODEL_PATHS, artifacts, strict=True):
        if (
            not isinstance(artifact, dict)
            or set(artifact) != {"relative_path", "sha256", "byte_size"}
            or artifact.get("relative_path") != expected_relative
            or not isinstance(artifact.get("sha256"), str)
            or not isinstance(artifact.get("byte_size"), int)
            or isinstance(artifact["byte_size"], bool)
        ):
            raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_manifest_invalid")
        record = next(
            (
                item
                for item in manifest.files
                if item.path == f"{manifest.local_model_root_relative_path}/{expected_relative}"
            ),
            None,
        )
        digest, size = _sha256_regular_file(
            actual[expected_relative], code="direct_paddle_bundle_manifest_invalid"
        )
        if (
            record is None
            or digest != artifact["sha256"]
            or size != artifact["byte_size"]
            or record.sha256 != digest
            or record.byte_size != size
        ):
            raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_manifest_invalid")


def _verify_direct_configuration(path: Path) -> None:
    value, _raw = _read_canonical_object(
        path, limit=_MAX_JSON_BYTES, code="direct_paddle_bundle_manifest_invalid"
    )
    required = {
        "schema_version",
        "engine_kind",
        "runtime_api",
        "extractor_revision",
        "raster_text_profile_revision",
        "text_detection_model_name",
        "text_detection_model_directory",
        "text_recognition_model_name",
        "text_recognition_model_directory",
        "cpu_threads",
        "text_detection_limit_side_len",
        "text_detection_limit_type",
        "text_detection_threshold",
        "text_detection_box_threshold",
        "text_detection_unclip_ratio",
    }
    revision = value.get("raster_text_profile_revision")
    if (
        set(value) != required
        or value.get("schema_version") != "1.0"
        or value.get("engine_kind") != "paddle_static_text_poc"
        or value.get("runtime_api") != "paddle-inference-static-v1"
        or not isinstance(revision, str)
        or not revision.startswith("synthetic-raster-text-v1:")
        or len(revision) != 49
        or value.get("text_detection_model_name") != "PP-OCRv5_server_det"
        or value.get("text_detection_model_directory") != "text-det"
        or value.get("text_recognition_model_name") != "en_PP-OCRv5_mobile_rec"
        or value.get("text_recognition_model_directory") != "text-rec"
        or value.get("cpu_threads") != 6
        or value.get("text_detection_limit_side_len") != 2048
        or value.get("text_detection_limit_type") != "max"
        or value.get("text_detection_threshold") != 0.3
        or value.get("text_detection_box_threshold") != 0.6
        or value.get("text_detection_unclip_ratio") != 1.5
    ):
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_manifest_invalid")


def _load_verified_bundle_from_root(
    *, root: Path, bundle_root: Path, expected_manifest_sha256: str | None, allow_staging: bool
) -> VerifiedRasterTextDirectPaddleBundleV1:
    try:
        if bundle_root.is_symlink() or not bundle_root.is_dir():
            raise OSError
        canonical_root = bundle_root.resolve(strict=True)
        relative = canonical_root.relative_to(root).as_posix()
        _relative_path(relative)
        valid = relative.startswith(f"{_DIRECT_RELEASES}/")
        if allow_staging:
            valid = (
                valid
                or canonical_root.parent == root
                and canonical_root.name.startswith(f".{_DIRECT_COLLECTION}.staging-")
            )
        if not valid:
            raise OSError
    except (OSError, ValueError):
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_path_invalid") from None
    operational_root = _operational_path(canonical_root, force_short=True)
    manifest_path = _safe_path(operational_root, _MANIFEST_FILE)
    raw, manifest_bytes = _read_canonical_object(
        manifest_path, limit=_MAX_MANIFEST_BYTES, code="direct_paddle_bundle_manifest_invalid"
    )
    try:
        manifest = DirectPaddleBundleManifestV1.model_validate(raw)
    except ValidationError:
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_manifest_invalid") from None
    manifest_sha256 = _sha256_bytes(manifest_bytes)
    if manifest.schema_version == "2.0":
        try:
            current_platform = current_raster_runtime_platform_binding()
        except Exception:
            raise RasterTextDirectPaddleBundleError(
                "direct_paddle_bundle_platform_mismatch"
            ) from None
        if (
            manifest.runtime_operating_system,
            manifest.runtime_architecture,
            manifest.runtime_python_abi,
        ) != current_platform:
            raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_platform_mismatch")
    if expected_manifest_sha256 is not None and manifest_sha256 != expected_manifest_sha256:
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_manifest_invalid")
    if relative.startswith(f"{_DIRECT_RELEASES}/") and relative != _release_relative_path(
        manifest_sha256
    ):
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_manifest_invalid")
    declared = {item.path: item for item in manifest.files}
    actual = _regular_tree_files(operational_root, code="direct_paddle_bundle_manifest_invalid")
    actual.pop(_MANIFEST_FILE, None)
    if set(actual) != set(declared):
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_manifest_invalid")
    for path, item in declared.items():
        digest, size = _sha256_regular_file(
            actual[path], code="direct_paddle_bundle_manifest_invalid"
        )
        if digest != item.sha256 or size != item.byte_size:
            raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_manifest_invalid")
    required = (
        manifest.parent_renderer_package_inventory_relative_path,
        manifest.sidecar_interpreter_relative_path,
        manifest.sidecar_module_relative_path,
        manifest.sidecar_configuration_template_relative_path,
        manifest.sidecar_package_inventory_relative_path,
        manifest.model_inventory_relative_path,
    )
    if any(path not in declared for path in required):
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_manifest_invalid")
    if (
        _tree_digest_from_manifest(
            manifest, directory=manifest.sidecar_interpreter_root_relative_path
        )
        != manifest.sidecar_interpreter_root_tree_sha256
        or _tree_digest_from_manifest(
            manifest, directory=manifest.sidecar_module_root_relative_path
        )
        != manifest.sidecar_module_root_tree_sha256
        or _tree_digest_from_manifest(manifest, directory=manifest.local_model_root_relative_path)
        != manifest.model_root_tree_sha256
    ):
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_manifest_invalid")
    _safe_path(operational_root, manifest.sidecar_interpreter_root_relative_path, directory=True)
    _safe_path(operational_root, manifest.sidecar_interpreter_relative_path)
    _safe_path(operational_root, manifest.sidecar_module_root_relative_path, directory=True)
    _safe_path(operational_root, manifest.sidecar_module_relative_path)
    _verify_direct_configuration(
        _safe_path(operational_root, manifest.sidecar_configuration_template_relative_path)
    )
    _verify_direct_package_inventory(
        _safe_path(operational_root, manifest.sidecar_package_inventory_relative_path),
        expected_sha256=declared[manifest.sidecar_package_inventory_relative_path].sha256,
        operating_system=manifest.runtime_operating_system or "windows",
    )
    _verify_direct_model_inventory(
        bundle_root=operational_root,
        manifest=manifest,
        expected_sha256=declared[manifest.model_inventory_relative_path].sha256,
    )
    parent_path = _safe_path(
        operational_root, manifest.parent_renderer_package_inventory_relative_path
    )
    _verify_parent_inventory(
        parent_path,
        expected_record=declared[manifest.parent_renderer_package_inventory_relative_path],
    )
    return VerifiedRasterTextDirectPaddleBundleV1(
        document_preview_root=root,
        bundle_root=operational_root,
        bundle_root_relative_path=relative,
        manifest=manifest,
        manifest_sha256=manifest_sha256,
        file_sha256_by_relative_path={path: item.sha256 for path, item in declared.items()},
    )


def load_verified_raster_text_direct_paddle_bundle_v1(
    *, document_preview_root: Path, manifest_sha256: str
) -> VerifiedRasterTextDirectPaddleBundleV1:
    """Load only the explicit content-addressed microbundle release."""

    root = _root(document_preview_root)
    relative = _release_relative_path(manifest_sha256)
    return _load_verified_bundle_from_root(
        root=root,
        bundle_root=_safe_path(root, relative, directory=True),
        expected_manifest_sha256=manifest_sha256,
        allow_staging=False,
    )


def _direct_sidecar_environment(
    *,
    interpreter: Path,
    configuration: Path,
    configuration_sha256: str,
    models: Path,
    package_inventory: Path,
    package_inventory_sha256: str,
    model_inventory: Path,
    model_inventory_sha256: str,
    sidecar_sha256: str,
    temporary_home: Path,
) -> dict[str, str]:
    """Construct a non-inheriting, local-only direct sidecar environment."""

    try:
        metadata = temporary_home.lstat()
        if not stat.S_ISDIR(metadata.st_mode) or stat.S_ISLNK(metadata.st_mode):
            raise OSError
        resolved = temporary_home.resolve(strict=True)
        drive = resolved.drive
        home = str(resolved)
        home_path = home[len(drive) :] if drive and home.startswith(drive) else ""
        if os.name == "nt" and (not drive or not home_path.startswith("\\")):
            raise OSError
    except OSError:
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_smoke_failed") from None
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in {"COMSPEC", "SYSTEMROOT", "WINDIR"}
    }
    environment.update(
        {
            "PATH": str(interpreter.parent),
            "HOME": home,
            "USERPROFILE": home,
            "APPDATA": home,
            "LOCALAPPDATA": home,
            "TMP": home,
            "TEMP": home,
            "CUSTOMS_AGENT_LOCAL_FILES_ONLY": "1",
            "HF_HUB_OFFLINE": "1",
            # Keep bytecode created by native-package helper interpreters out
            # of the immutable staged/released module root.  The main smoke
            # interpreter also carries ``-B``; these variables cover helpers
            # spawned by imported packages.
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONPYCACHEPREFIX": str(resolved / "pycache"),
            "TRANSFORMERS_OFFLINE": "1",
            "CUSTOMS_AGENT_RASTER_TEXT_DIRECT_PADDLE_CONFIG_PATH": str(configuration),
            "CUSTOMS_AGENT_RASTER_TEXT_DIRECT_PADDLE_CONFIG_SHA256": configuration_sha256,
            "CUSTOMS_AGENT_RASTER_TEXT_DIRECT_PADDLE_MODEL_ROOT": str(models),
            "CUSTOMS_AGENT_RASTER_TEXT_DIRECT_PADDLE_PACKAGE_INVENTORY_PATH": str(
                package_inventory
            ),
            "CUSTOMS_AGENT_RASTER_TEXT_DIRECT_PADDLE_PACKAGE_INVENTORY_SHA256": package_inventory_sha256,
            "CUSTOMS_AGENT_RASTER_TEXT_DIRECT_PADDLE_MODEL_INVENTORY_PATH": str(model_inventory),
            "CUSTOMS_AGENT_RASTER_TEXT_DIRECT_PADDLE_MODEL_INVENTORY_SHA256": model_inventory_sha256,
            "CUSTOMS_AGENT_RASTER_TEXT_DIRECT_PADDLE_REQUIRED_PACKAGES": json.dumps(
                list(_DIRECT_REQUIRED_PACKAGES), ensure_ascii=True, separators=(",", ":")
            ),
            "CUSTOMS_AGENT_RASTER_TEXT_DIRECT_PADDLE_MODULE_SHA256": sidecar_sha256,
        }
    )
    if os.name == "nt":
        environment.update({"HOMEDRIVE": drive, "HOMEPATH": home_path})
    return environment


_DIRECT_ISOLATED_BOOTSTRAP = (
    "import os,runpy,sys,sysconfig\n"
    "interpreter_root=os.path.realpath(sys.argv[1])\n"
    "module_root=os.path.realpath(sys.argv[2])\n"
    "command_path=os.path.realpath(sys.argv[3])\n"
    "def inside(root,path): return os.path.commonpath((os.path.normcase(root),os.path.normcase(path))) == os.path.normcase(root)\n"
    "if (not os.path.isdir(interpreter_root) or os.path.islink(interpreter_root) or not inside(interpreter_root,os.path.realpath(sys.executable))): raise SystemExit(70)\n"
    "if (not os.path.isdir(module_root) or os.path.islink(module_root) or os.path.islink(command_path) or not inside(module_root,command_path)): raise SystemExit(70)\n"
    "stdlib=[]\n"
    "for key in ('stdlib','platstdlib'):\n"
    " path=sysconfig.get_path(key)\n"
    " if path and path not in stdlib: stdlib.append(path)\n"
    "for path in (os.path.join(sys.base_prefix,'DLLs'),os.path.join(stdlib[0],'lib-dynload') if stdlib else ''):\n"
    " if path and os.path.isdir(path) and path not in stdlib: stdlib.append(path)\n"
    "if any(not inside(interpreter_root,os.path.realpath(path)) for path in stdlib): raise SystemExit(70)\n"
    "sys.path[:]=[sys.argv[2],*stdlib]\n"
    "sys.argv=[sys.argv[3],*sys.argv[4:]]\n"
    "runpy.run_path(sys.argv[0],run_name='__main__')\n"
)


@contextmanager
def _temporary_smoke_home() -> Iterator[Path]:
    try:
        temporary = tempfile.TemporaryDirectory(prefix="customs-direct-paddle-")
        path = Path(temporary.name)
        if path.is_symlink() or not path.is_dir():
            temporary.cleanup()
            raise OSError
    except OSError:
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_smoke_failed") from None
    try:
        yield path
    finally:
        temporary.cleanup()


def _smoke_direct_bundle(bundle: VerifiedRasterTextDirectPaddleBundleV1) -> None:
    """Start the copied static engine and run one blank local RGB request."""

    manifest = bundle.manifest
    interpreter = _safe_path(bundle.bundle_root, manifest.sidecar_interpreter_relative_path)
    module_root = _safe_path(
        bundle.bundle_root, manifest.sidecar_module_root_relative_path, directory=True
    )
    sidecar = _safe_path(bundle.bundle_root, manifest.sidecar_module_relative_path)
    configuration = _safe_path(
        bundle.bundle_root, manifest.sidecar_configuration_template_relative_path
    )
    package_inventory = _safe_path(
        bundle.bundle_root, manifest.sidecar_package_inventory_relative_path
    )
    model_inventory = _safe_path(bundle.bundle_root, manifest.model_inventory_relative_path)
    models = _safe_path(bundle.bundle_root, manifest.local_model_root_relative_path, directory=True)
    config_raw = configuration.read_bytes()
    try:
        configuration_value = json.loads(config_raw)
        revision = configuration_value["raster_text_profile_revision"]
        if not isinstance(revision, str):
            raise ValueError
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_smoke_failed") from None
    pixels = b"\xff" * (64 * 64 * 3)
    source_sha256 = _sha256_bytes(pixels)
    payload = (
        _canonical_json(
            {
                "protocol": "customs-raster-text-sidecar-v1",
                "source_sha256": source_sha256,
                "raster_text_profile_revision": revision,
                "pages": [{"page_number": 1, "width": 64, "height": 64}],
            }
        )
        + b"\n"
        + pixels
    )
    try:
        with _temporary_smoke_home() as home:
            completed = subprocess.run(
                [
                    str(interpreter),
                    "-I",
                    "-B",
                    "-c",
                    _DIRECT_ISOLATED_BOOTSTRAP,
                    str(
                        _safe_path(
                            bundle.bundle_root,
                            manifest.sidecar_interpreter_root_relative_path,
                            directory=True,
                        )
                    ),
                    str(module_root),
                    str(sidecar),
                    "--customs-raster-text-sidecar-v1",
                ],
                input=payload,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                cwd=str(module_root),
                env=_direct_sidecar_environment(
                    interpreter=interpreter,
                    configuration=configuration,
                    configuration_sha256=_sha256_bytes(config_raw),
                    models=models,
                    package_inventory=package_inventory,
                    package_inventory_sha256=bundle.file_sha256(
                        manifest.sidecar_package_inventory_relative_path
                    ),
                    model_inventory=model_inventory,
                    model_inventory_sha256=bundle.file_sha256(
                        manifest.model_inventory_relative_path
                    ),
                    sidecar_sha256=bundle.file_sha256(manifest.sidecar_module_relative_path),
                    temporary_home=home,
                ),
                check=False,
                shell=False,
                timeout=_DIRECT_SMOKE_TIMEOUT_SECONDS,
            )
    except (OSError, subprocess.TimeoutExpired, RasterTextDirectPaddleBundleError):
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_smoke_failed") from None
    if completed.returncode != 0 or not completed.stdout or len(completed.stdout) > _MAX_JSON_BYTES:
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_smoke_failed")
    try:
        response = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_smoke_failed") from None
    if (
        not isinstance(response, dict)
        or response.get("protocol") != "customs-raster-text-sidecar-v1"
        or response.get("source_sha256") != source_sha256
        or response.get("raster_text_profile_revision") != revision
    ):
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_smoke_failed")


def _remove_owned_stage(root: Path, stage: Path) -> None:
    try:
        resolved = stage.resolve(strict=True)
        if (
            _local_path_identity(resolved.parent) != _local_path_identity(root)
            or resolved.is_symlink()
            or not resolved.name.startswith(f".{_DIRECT_COLLECTION}.staging-")
        ):
            return
        shutil.rmtree(_operational_path(resolved, force_short=True))
    except OSError:
        # A stale private staging directory is recoverable and safer than a
        # broad cleanup attempt.  It is never a final/selected bundle.
        return


def _is_transient_windows_publish_lock(error: OSError) -> bool:
    """Return whether ``error`` is one of the two retryable Windows locks."""

    return getattr(error, "winerror", None) in _PUBLISH_TRANSIENT_WINDOWS_ERROR_CODES


def _require_retry_destination_absent(final_canonical: Path) -> None:
    """Fail closed when a retry would race an immutable release destination."""

    try:
        collision = final_canonical.exists() or final_canonical.is_symlink()
    except OSError:
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_publish_failed") from None
    if collision:
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_publish_conflict")


@contextmanager
def _lock(root: Path) -> Iterator[None]:
    try:
        with exclusive_local_process_lock(root / _LOCK_FILE):
            yield
    except LocalProcessLockBusy:
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_locked") from None
    except LocalProcessLockInvalid:
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_lock_invalid") from None


def _load_intent_reuse(
    *, root: Path, intent: DirectPaddleBundleIntentV1, intent_sha256: str
) -> VerifiedRasterTextDirectPaddleBundleV1 | None:
    relative = f"{_DIRECT_INTENTS}/{intent_sha256}.json"
    candidate = root.joinpath(*PurePosixPath(relative).parts)
    try:
        if not candidate.exists():
            return None
        index_path = _safe_path(root, relative)
    except RasterTextDirectPaddleBundleError:
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_intent_invalid") from None
    raw, _bytes = _read_canonical_object(
        index_path, limit=_MAX_MANIFEST_BYTES, code="direct_paddle_bundle_intent_invalid"
    )
    try:
        index = DirectPaddleBundleIntentIndexV1.model_validate(raw)
    except ValidationError:
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_intent_invalid") from None
    if index.derivation_intent != intent or index.derivation_intent_sha256 != intent_sha256:
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_intent_invalid")
    bundle = load_verified_raster_text_direct_paddle_bundle_v1(
        document_preview_root=root, manifest_sha256=index.release_manifest_sha256
    )
    if (
        bundle.bundle_root_relative_path != index.release_relative_path
        or bundle.manifest.derivation_intent_sha256 != intent_sha256
    ):
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_intent_invalid")
    return bundle


def _write_intent_index(
    *,
    root: Path,
    intent: DirectPaddleBundleIntentV1,
    intent_sha256: str,
    bundle: VerifiedRasterTextDirectPaddleBundleV1,
) -> None:
    index = DirectPaddleBundleIntentIndexV1(
        schema_version="1.0",
        artifact_kind="raster_text_direct_paddle_bundle_intent_index",
        derivation_intent=intent,
        derivation_intent_sha256=intent_sha256,
        release_manifest_sha256=bundle.manifest_sha256,
        release_relative_path=bundle.bundle_root_relative_path,
    )
    _write_create_only(
        root / _DIRECT_INTENTS / f"{intent_sha256}.json",
        index.model_dump(mode="json", warnings=False),
        code="direct_paddle_bundle_intent_write_failed",
    )


def _build_stage(
    *,
    stage: Path,
    source: DirectPaddleBundleSourceV1,
    interpreter_records: Sequence[DirectPaddleBundleFileV1],
    module_records: Sequence[DirectPaddleBundleFileV1],
    first_party_records: Sequence[DirectPaddleBundleFileV1],
    intent_sha256: str,
) -> str:
    """Copy every selected byte, hash it, then write the strict output manifest."""

    for record in interpreter_records:
        source_relative = record.path
        destination_relative = source_relative
        _copy_expected_file(
            source_root=source.source_bundle_root,
            source_relative=source_relative,
            record=record,
            stage=stage,
            destination_relative=destination_relative,
            code="direct_paddle_bundle_stage_copy_failed",
        )
    for record in module_records:
        source_relative = record.path
        relative_to_source = PurePosixPath(source_relative).relative_to(
            PurePosixPath(_SOURCE_MODULE_ROOT)
        )
        _copy_expected_file(
            source_root=source.source_bundle_root,
            source_relative=source_relative,
            record=record,
            stage=stage,
            destination_relative=f"sidecar/module-root/{relative_to_source.as_posix()}",
            code="direct_paddle_bundle_stage_copy_failed",
        )
    for record in first_party_records:
        _copy_current_first_party_file(
            stage=stage, record=record, code="direct_paddle_bundle_stage_copy_failed"
        )
    parent_record = _record_for(
        source.file_records,
        path=_SOURCE_PARENT_INVENTORY,
        code="direct_paddle_bundle_source_manifest_invalid",
    )
    _copy_expected_file(
        source_root=source.source_bundle_root,
        source_relative=_SOURCE_PARENT_INVENTORY,
        record=parent_record,
        stage=stage,
        destination_relative="inventories/parent-renderer.json",
        code="direct_paddle_bundle_stage_copy_failed",
    )
    for relative in _DIRECT_MODEL_PATHS:
        record = _record_for(
            source.file_records,
            path=f"{_SOURCE_DIRECT_MODEL_ROOT}/{relative}",
            code="direct_paddle_bundle_source_model_invalid",
        )
        _copy_expected_file(
            source_root=source.source_bundle_root,
            source_relative=record.path,
            record=record,
            stage=stage,
            destination_relative=f"models/{relative}",
            code="direct_paddle_bundle_stage_copy_failed",
        )
    module_root = stage / "sidecar" / "module-root"
    _verify_distribution_metadata(
        module_root, operating_system=source.runtime_operating_system
    )
    _write_create_only(
        stage / "inventories" / "direct-paddle-packages.json",
        _direct_package_inventory(operating_system=source.runtime_operating_system),
        code="direct_paddle_bundle_stage_write_failed",
    )
    _write_create_only(
        stage / "inventories" / "direct-paddle-models.json",
        _direct_model_inventory(source),
        code="direct_paddle_bundle_stage_write_failed",
    )
    _write_create_only(
        stage / "sidecar" / "direct-configuration-template.json",
        _direct_configuration_template(source=source, intent_sha256=intent_sha256),
        code="direct_paddle_bundle_stage_write_failed",
    )
    files = _stage_files(stage)
    legacy_source = source.source_manifest_sha256 == _E3_DIRECT_BASE_MANIFEST_SHA256
    manifest = DirectPaddleBundleManifestV1(
        schema_version="1.0" if legacy_source else "2.0",
        bundle_id=_DIRECT_BUNDLE_ID,
        runtime_operating_system=None if legacy_source else source.runtime_operating_system,
        runtime_architecture=None if legacy_source else source.runtime_architecture,
        runtime_python_abi=None if legacy_source else source.runtime_python_abi,
        source_bundle_relative_path=source.source_bundle_relative_path,
        source_bundle_manifest_sha256=source.source_manifest_sha256,
        derivation_intent_sha256=intent_sha256,
        extractor_revision=f"{source.extractor_revision}:raster-text-direct-paddle-v1",
        raster_dpi=source.raster_dpi,
        sidecar_timeout_seconds=_DIRECT_SIDECAR_TIMEOUT_SECONDS,
        parent_renderer_package_inventory_relative_path="inventories/parent-renderer.json",
        parent_renderer_required_packages=["pypdfium2", "Pillow"],
        sidecar_interpreter_root_relative_path="sidecar/interpreter",
        sidecar_interpreter_relative_path=source.source_interpreter_relative_path,
        sidecar_interpreter_root_tree_sha256=_tree_manifest_sha256(
            [
                DirectPaddleBundleFileV1(
                    path=PurePosixPath(item.path)
                    .relative_to(PurePosixPath(_SOURCE_INTERPRETER_ROOT))
                    .as_posix(),
                    sha256=item.sha256,
                    byte_size=item.byte_size,
                )
                for item in interpreter_records
            ]
        ),
        sidecar_module_root_relative_path="sidecar/module-root",
        sidecar_module_relative_path="sidecar/module-root/customs_ai_agent/service/raster_text_direct_paddle_sidecar.py",
        sidecar_module_root_tree_sha256=_tree_manifest_sha256(
            [
                DirectPaddleBundleFileV1(
                    path=item.path.removeprefix(f"{_SOURCE_MODULE_ROOT}/"),
                    sha256=item.sha256,
                    byte_size=item.byte_size,
                )
                for item in module_records
            ]
            + [
                DirectPaddleBundleFileV1(
                    path=f"customs_ai_agent/{item.path}",
                    sha256=item.sha256,
                    byte_size=item.byte_size,
                )
                for item in first_party_records
            ]
        ),
        sidecar_configuration_template_relative_path="sidecar/direct-configuration-template.json",
        sidecar_package_inventory_relative_path="inventories/direct-paddle-packages.json",
        sidecar_required_packages=list(_DIRECT_REQUIRED_PACKAGES),
        local_model_root_relative_path="models",
        model_inventory_relative_path="inventories/direct-paddle-models.json",
        model_root_tree_sha256=_tree_manifest_sha256(
            [
                DirectPaddleBundleFileV1(
                    path=PurePosixPath(item.path)
                    .relative_to(PurePosixPath(_SOURCE_DIRECT_MODEL_ROOT))
                    .as_posix(),
                    sha256=item.sha256,
                    byte_size=item.byte_size,
                )
                for item in (
                    _record_for(
                        source.file_records,
                        path=f"{_SOURCE_DIRECT_MODEL_ROOT}/{relative}",
                        code="direct_paddle_bundle_source_model_invalid",
                    )
                    for relative in _DIRECT_MODEL_PATHS
                )
            ]
        ),
        files=files,
    )
    payload = _write_create_only(
        stage / _MANIFEST_FILE,
        manifest.model_dump(mode="json", warnings=False),
        code="direct_paddle_bundle_stage_write_failed",
    )
    return _sha256_bytes(payload)


def materialize_verified_raster_text_direct_paddle_bundle_v1(
    *,
    document_preview_root: Path,
    source_manifest_sha256: str | None = None,
) -> VerifiedRasterTextDirectPaddleBundleV1:
    """Create/reuse a local direct Paddle microbundle without network access.

    An incomplete closure, source drift, smoke failure, lock conflict, or a
    different pre-existing output is fail-closed.  Existing valid releases are
    never overwritten; the index merely points to the immutable release whose
    complete derivation intent matches the current local source inputs.
    """

    root = _root(document_preview_root)
    with _lock(root):
        source = verify_e3_raster_text_direct_paddle_base_v1(
            document_preview_root=root,
            source_manifest_sha256=source_manifest_sha256,
        )
        interpreter_records, module_records = _selected_source_records(source)
        first_party_records = _snapshot_first_party_sidecar()
        intent = _intent_from_source(
            source,
            interpreter_records=interpreter_records,
            module_records=module_records,
            first_party_records=first_party_records,
        )
        intent_sha256 = _model_sha256(intent)
        reused = _load_intent_reuse(root=root, intent=intent, intent_sha256=intent_sha256)
        if reused is not None:
            return reused
        _ensure_directory(root, _DIRECT_COLLECTION, code="direct_paddle_bundle_output_invalid")
        _ensure_directory(root, _DIRECT_RELEASES, code="direct_paddle_bundle_output_invalid")
        _ensure_directory(root, _DIRECT_INTENTS, code="direct_paddle_bundle_output_invalid")
        stage_canonical = root / f".{_DIRECT_COLLECTION}.staging-{uuid.uuid4().hex}"
        stage: Path | None = None
        try:
            if stage_canonical.exists() or stage_canonical.is_symlink():
                raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_output_invalid")
            stage_canonical.mkdir(mode=0o700)
            stage = _operational_path(stage_canonical, force_short=True)
            manifest_sha256 = _build_stage(
                stage=stage,
                source=source,
                interpreter_records=interpreter_records,
                module_records=module_records,
                first_party_records=first_party_records,
                intent_sha256=intent_sha256,
            )
            staged = _load_verified_bundle_from_root(
                root=root,
                bundle_root=stage,
                expected_manifest_sha256=manifest_sha256,
                allow_staging=True,
            )
            _smoke_direct_bundle(staged)
            # The child runs under ``-B``.  Rehash after its real native
            # initialization so an unexpected write can never be published.
            staged = _load_verified_bundle_from_root(
                root=root,
                bundle_root=stage,
                expected_manifest_sha256=manifest_sha256,
                allow_staging=True,
            )
            final_relative = _release_relative_path(manifest_sha256)
            # ``stage`` is an 8.3 operational spelling used only while
            # creating/deep-verifying the native closure.  Publishing a
            # directory must use the two canonical same-volume identities:
            # mixing an operational source with an operational destination
            # can make Windows reject an otherwise valid atomic rename after
            # the smoke process has completed.  Keep a short final spelling
            # only for later native-path consumers of the published release.
            final_canonical = root.joinpath(*PurePosixPath(_DIRECT_RELEASES).parts, manifest_sha256)
            final_parent = _operational_path(final_canonical.parent, force_short=True)
            final = final_parent / manifest_sha256
            if final_canonical.exists() or final_canonical.is_symlink():
                existing = load_verified_raster_text_direct_paddle_bundle_v1(
                    document_preview_root=root, manifest_sha256=manifest_sha256
                )
                if existing.manifest.derivation_intent_sha256 != intent_sha256:
                    raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_publish_conflict")
                _write_intent_index(
                    root=root, intent=intent, intent_sha256=intent_sha256, bundle=existing
                )
                return existing
            # Same-volume private staging → immutable final path.  Windows
            # does not replace a destination here, so no valid release can be
            # silently overwritten.
            retries = 0
            while True:
                try:
                    os.rename(stage_canonical, final_canonical)
                    break
                except OSError as error:
                    # Keep the operating-system detail out of transcripts
                    # while still distinguishing a failed atomic publication
                    # from an invalid source/provisioned byte boundary.  The
                    # sole exception is a short retry for the two Windows
                    # directory-handle lock codes observed after sidecar exit.
                    if (
                        retries >= _PUBLISH_MAX_TRANSIENT_RETRIES
                        or not _is_transient_windows_publish_lock(error)
                    ):
                        raise RasterTextDirectPaddleBundleError(
                            "direct_paddle_bundle_publish_failed"
                        ) from None
                    time.sleep(_PUBLISH_TRANSIENT_RETRY_DELAYS_SECONDS[retries])
                    retries += 1
                    # Do not use a replace/copy fallback: after every
                    # transient failure, the original private stage must
                    # still validate exactly and the immutable destination
                    # must remain absent before another atomic rename.
                    _require_retry_destination_absent(final_canonical)
                    staged = _load_verified_bundle_from_root(
                        root=root,
                        bundle_root=stage,
                        expected_manifest_sha256=manifest_sha256,
                        allow_staging=True,
                    )
            published = VerifiedRasterTextDirectPaddleBundleV1(
                document_preview_root=root,
                bundle_root=final,
                bundle_root_relative_path=final_relative,
                manifest=staged.manifest,
                manifest_sha256=manifest_sha256,
                file_sha256_by_relative_path=staged.file_sha256_by_relative_path,
            )
            _write_intent_index(
                root=root, intent=intent, intent_sha256=intent_sha256, bundle=published
            )
            return published
        except RasterTextDirectPaddleBundleError:
            raise
        except OSError:
            raise RasterTextDirectPaddleBundleError(
                "direct_paddle_bundle_materialize_failed"
            ) from None
        finally:
            if stage_canonical.exists():
                _remove_owned_stage(root, stage_canonical)


def prepare_raster_text_direct_paddle_runtime_profile_v2(
    bundle: VerifiedRasterTextDirectPaddleBundleV1,
    *,
    pdfium_version: str,
    minimum_page_confidence: float,
) -> PreparedRasterTextDirectPaddleRuntimeProfileV2:
    """Create-or-verify the V2 direct-Paddle text runtime profile.

    This is intentionally independent from the earlier PP-Structure and
    PaddleOCR/PaddleX profile builders.  Before writing anything, it reloads
    this module's own exhaustive microbundle verifier, so a stale binding or
    any post-publication package/model drift is fail-closed.  No generic
    bundle loader, package installer, network lookup, or model fallback is
    reachable from this path.
    """

    from customs_ai_agent.service.raster_text_preview_runtime import (
        RasterTextOfflineProfileV2,
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
        raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_runtime_profile_invalid")

    try:
        verified = load_verified_raster_text_direct_paddle_bundle_v1(
            document_preview_root=bundle.document_preview_root,
            manifest_sha256=bundle.manifest_sha256,
        )
        if (
            _local_path_identity(verified.bundle_root) != _local_path_identity(bundle.bundle_root)
            or verified.bundle_root_relative_path != bundle.bundle_root_relative_path
            or verified.manifest != bundle.manifest
        ):
            raise RasterTextDirectPaddleBundleError("direct_paddle_bundle_runtime_profile_invalid")
        manifest = verified.manifest
        profile_payload: dict[str, object] = {
            "schema_version": "2.0",
            "engine_kind": "direct_paddle_text",
            "runtime_api": "paddle-inference-static-v1",
            "extractor_revision": manifest.extractor_revision,
            "bundle_root_relative_path": verified.bundle_root_relative_path,
            "bundle_manifest_sha256": verified.manifest_sha256,
            "pdfium_version": pdfium_version,
            "raster_dpi": manifest.raster_dpi,
            # The full service request reserves remaining time for strict
            # decode, renderer cleanup and response serialization.  Native
            # OCR itself must never exceed the owned 30-second lane.
            "sidecar_timeout_seconds": min(30, manifest.sidecar_timeout_seconds),
            "cpu_threads": 6,
            # This is the measured cap for the 30-second customer lane.
            # It is a V2 literal, not a tunable caller option.
            "text_detection_limit_side_len": 960,
            "text_detection_limit_type": "max",
            "text_detection_threshold": 0.3,
            "text_detection_box_threshold": 0.6,
            "text_detection_unclip_ratio": 1.5,
            "minimum_page_confidence": float(minimum_page_confidence),
            "parent_renderer_package_inventory_path": verified.bundle_relative_path(
                manifest.parent_renderer_package_inventory_relative_path
            ),
            "parent_renderer_package_inventory_sha256": verified.file_sha256(
                manifest.parent_renderer_package_inventory_relative_path
            ),
            "parent_renderer_required_packages": manifest.parent_renderer_required_packages,
            "sidecar_interpreter_path": verified.bundle_relative_path(
                manifest.sidecar_interpreter_relative_path
            ),
            "sidecar_interpreter_sha256": verified.file_sha256(
                manifest.sidecar_interpreter_relative_path
            ),
            "sidecar_interpreter_root": verified.bundle_relative_path(
                manifest.sidecar_interpreter_root_relative_path
            ),
            "sidecar_interpreter_root_tree_sha256": manifest.sidecar_interpreter_root_tree_sha256,
            "sidecar_module_path": verified.bundle_relative_path(
                manifest.sidecar_module_relative_path
            ),
            "sidecar_module_sha256": verified.file_sha256(manifest.sidecar_module_relative_path),
            "sidecar_module_root": verified.bundle_relative_path(
                manifest.sidecar_module_root_relative_path
            ),
            "sidecar_module_root_tree_sha256": manifest.sidecar_module_root_tree_sha256,
            "sidecar_package_inventory_path": verified.bundle_relative_path(
                manifest.sidecar_package_inventory_relative_path
            ),
            "sidecar_package_inventory_sha256": verified.file_sha256(
                manifest.sidecar_package_inventory_relative_path
            ),
            "sidecar_required_packages": manifest.sidecar_required_packages,
            "local_model_root": verified.bundle_relative_path(
                manifest.local_model_root_relative_path
            ),
            "model_inventory_path": verified.bundle_relative_path(
                manifest.model_inventory_relative_path
            ),
            "model_inventory_sha256": verified.file_sha256(manifest.model_inventory_relative_path),
            "text_detection_model_name": "PP-OCRv5_server_det",
            "text_detection_model_directory": "text-det",
            "text_recognition_model_name": "en_PP-OCRv5_mobile_rec",
            "text_recognition_model_directory": "text-rec",
        }
        fingerprint = raster_text_profile_fingerprint_sha256(profile_payload)
        profile_payload["profile_fingerprint_sha256"] = fingerprint
        profile_payload["raster_text_profile_revision"] = (
            f"synthetic-raster-text-v1:{fingerprint[:24]}"
        )
        profile = RasterTextOfflineProfileV2.model_validate(profile_payload)
    except RasterTextDirectPaddleBundleError:
        raise
    except (OSError, TypeError, ValidationError, ValueError):
        raise RasterTextDirectPaddleBundleError(
            "direct_paddle_bundle_runtime_profile_invalid"
        ) from None

    suffix = profile.raster_text_profile_revision.rsplit(":", 1)[1]
    relative_path = f"{_DIRECT_RUNTIME_PROFILES}/{suffix}/profile.json"
    directory = _ensure_directory(
        verified.document_preview_root,
        f"{_DIRECT_RUNTIME_PROFILES}/{suffix}",
        code="direct_paddle_bundle_runtime_profile_write_failed",
    )
    payload = _write_create_only(
        directory / "profile.json",
        profile.model_dump(mode="json", warnings=False),
        code="direct_paddle_bundle_runtime_profile_write_failed",
    )
    return PreparedRasterTextDirectPaddleRuntimeProfileV2(
        raster_text_profile_revision=profile.raster_text_profile_revision,
        runtime_configuration_relative_path=relative_path,
        runtime_configuration_sha256=_sha256_bytes(payload),
        minimum_page_confidence=profile.minimum_page_confidence,
    )


__all__ = [
    "DirectPaddleBundleFileV1",
    "DirectPaddleBundleIntentIndexV1",
    "DirectPaddleBundleIntentV1",
    "DirectPaddleBundleManifestV1",
    "DirectPaddleBundleSourceV1",
    "PreparedRasterTextDirectPaddleRuntimeProfileV2",
    "RasterTextDirectPaddleBundleError",
    "VerifiedRasterTextDirectPaddleBundleV1",
    "load_verified_raster_text_direct_paddle_bundle_v1",
    "materialize_verified_raster_text_direct_paddle_bundle_v1",
    "prepare_raster_text_direct_paddle_runtime_profile_v2",
    "verify_e3_raster_text_direct_paddle_base_v1",
]
