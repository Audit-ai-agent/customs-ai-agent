"""Strict local inventory for the automatic synthetic-raster OCR profile.

The automatic path deliberately does *not* install, download, discover, or
select OCR components.  A separately provisioned local legacy bundle, or one
content-addressed release selected from it, is the sole input.  This module
validates that release before the expensive source/Gold build starts and
derives the stable runtime fingerprint used by the active pointer.  It
contains no OCR output, document text, or secrets.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
import re
from typing import Any, Iterable, Mapping, Protocol

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator


_SHA256 = "^[a-f0-9]{64}$"
_DOCUMENT_PREVIEW_ROOT_SUFFIX = ("_local", "data", "document-preview")
_BUNDLE_DIRECTORY = "raster-offline-bundle-v1"
# ``raster-offline-bundle-v1`` is the original, fixed-location release.  It
# remains readable forever so an already-approved pointer can keep resolving
# exactly the bytes it was approved against.  Later local releases live in a
# separate, content-addressed collection; the collection's one selector is
# intentionally the only mutable bundle-selection state.
_BUNDLE_RELEASE_COLLECTION_DIRECTORY = "raster-offline-bundles-v1"
_BUNDLE_RELEASES_DIRECTORY = f"{_BUNDLE_RELEASE_COLLECTION_DIRECTORY}/releases"
_BUNDLE_RELEASE_SELECTOR_FILE = "current.json"
_MANIFEST_FILE = "manifest.json"
# This must stay outside ``synthetic-raster-v1``.  That directory is a
# transactional dataset publish target: its mere presence means a reader must
# be able to verify a complete immutable dataset.  Runtime configuration is
# prepared before candidate execution and therefore cannot live below that
# publish target without turning a harmless preflight/smoke run into a false
# "partial dataset" failure.
_RUNTIME_PROFILES_DIRECTORY = "raster-runtime-profiles-v1"
_RASTER_TEXT_RUNTIME_PROFILES_DIRECTORY = "raster-text-runtime-profiles-v1"
# A real CPython + Paddle sidecar has roughly 25,000 regular files.  Its
# exhaustive file-level manifest is still compact enough for local loading,
# but it is larger than the earlier mock-bundle-only 4 MiB cap.
_MAX_MANIFEST_BYTES = 32 * 1024 * 1024
_MAX_CONFIGURATION_BYTES = 1024 * 1024
_MAX_BUNDLE_FILE_BYTES = 1024 * 1024 * 1024
_SIDECAR_REQUIRED_PACKAGES = frozenset({"paddleocr", "paddlex", "paddlepaddle"})
# The direct text lane deliberately has a separate, minimal material boundary.
# It is not allowed to reuse the structural PP-Structure inventory/root just
# because both pipelines happen to use the same detector/recognizer names.
_RASTER_TEXT_MODEL_INVENTORY_KIND = "raster-text-direct-models-v1"
_RASTER_TEXT_MODEL_INVENTORY_SCHEMA_VERSION = "1.0"
_RASTER_TEXT_DIRECT_MODEL_PATHS = (
    "text-det/inference.json",
    "text-det/inference.pdiparams",
    "text-det/inference.yml",
    "text-rec/config.json",
    "text-rec/inference.json",
    "text-rec/inference.pdiparams",
    "text-rec/inference.yml",
)
# The source-attested text lane uses the same fixed six-core CPU budget as the
# manifest-bound sidecar template.  Keeping this value here prevents the
# runtime profile from silently falling back to two threads after the bundle
# correctly declared six.
_RASTER_TEXT_CPU_THREADS = 6
# Only this exact, complete field set may be absent from a historical legacy
# template while the release builder migrates it.  Normal loading never fills
# these values: doing so would change an already-evaluated sidecar JSON's
# behavior without changing its SHA/profile/sealed evidence.
_HISTORICAL_SAFE_SIDECAR_CONFIGURATION_VALUES: dict[str, object] = {
    "inference_engine": "paddle_static",
    "paddle_static_run_mode": "paddle",
    "paddle_static_enable_new_ir": False,
    "paddle_static_enable_cinn": False,
    "text_detection_limit_side_len": 960,
    "text_detection_limit_type": "max",
    "text_detection_threshold": 0.3,
    "text_detection_box_threshold": 0.6,
    "text_detection_unclip_ratio": 1.5,
}
# A prior local native probe can make CPython create these cache files below
# the bundled import root.  They are not listed in the immutable manifest and
# must never be part of a selected release.  The migration-only exception is
# deliberately narrower than "ignore *.pyc": it accepts only CPython's cache
# name for a *declared* sibling source file.
_HISTORICAL_GENERATED_BYTECODE_CACHE_NAME = re.compile(
    r"(?P<stem>[^.]+)\.cpython-[0-9]+(?:\.opt-[12])?\.pyc\Z"
)


class RasterOfflineBundleError(RuntimeError):
    """A content-free failure that must keep the old active pointer intact."""

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
        raise ValueError("local relative path is invalid")
    return value


class RasterOfflineBundleFileV1(_WireModel):
    """One regular bundle file, bound by its exact byte hash and size."""

    path: str = Field(min_length=1, max_length=480)
    sha256: str = Field(pattern=_SHA256)
    # Empty ``__init__.py``, ``py.typed`` and dist-info marker files are valid
    # parts of a Python package closure.  Their SHA-256 is still exact, so
    # zero bytes do not weaken the exhaustive bundle binding.
    byte_size: int = Field(ge=0, le=_MAX_BUNDLE_FILE_BYTES)

    _path_is_relative = field_validator("path")(_relative_path)


class RasterOfflineBundleManifestV1(_WireModel):
    """Complete, intentionally small declaration of one local OCR bundle.

    ``files`` is exhaustive for every file below the bundle except this
    manifest itself.  The listed paths are not operator flags: the wrapper
    always loads this one manifest from the fixed bundle directory.
    """

    schema_version: str = "1.0"
    bundle_id: str = "raster-offline-bundle-v1"
    extractor_revision: str = Field(min_length=1, max_length=256)
    raster_dpi: int = Field(default=216, ge=72, le=600)
    sidecar_timeout_seconds: int = Field(default=420, ge=1, le=420)

    parent_renderer_package_inventory_relative_path: str = Field(min_length=1, max_length=480)
    parent_renderer_required_packages: list[str] = Field(min_length=2, max_length=16)

    sidecar_package_inventory_relative_path: str = Field(min_length=1, max_length=480)
    sidecar_required_packages: list[str] = Field(min_length=3, max_length=32)
    sidecar_command_relative_path: str = Field(min_length=1, max_length=480)
    # v2 never resolves a generic ``python``/wrapper from PATH.  The bundle
    # declares the exact child interpreter that executes the sidecar instead.
    sidecar_interpreter_root_relative_path: str = Field(min_length=1, max_length=480)
    sidecar_interpreter_relative_path: str = Field(min_length=1, max_length=480)
    # The child receives this one hash-bound import root explicitly.  It holds
    # the first-party sidecar package and every third-party module it needs;
    # inherited PYTHONPATH and interpreter site-packages are not part of the
    # v2 import surface.
    sidecar_module_root_relative_path: str = Field(min_length=1, max_length=480)
    sidecar_module_relative_path: str = Field(min_length=1, max_length=480)
    sidecar_configuration_template_relative_path: str = Field(min_length=1, max_length=480)
    model_inventory_relative_path: str = Field(min_length=1, max_length=480)
    local_model_root_relative_path: str = Field(min_length=1, max_length=480)
    # Present only in an immutable release prepared for the independent
    # direct-text lane.  Keeping these optional preserves the structural
    # bundle contract and makes the text runtime fail closed rather than
    # silently pointing at its much larger eight-model inventory.
    raster_text_model_inventory_relative_path: str | None = Field(
        default=None, min_length=1, max_length=480
    )
    raster_text_local_model_root_relative_path: str | None = Field(
        default=None, min_length=1, max_length=480
    )
    files: list[RasterOfflineBundleFileV1] = Field(min_length=1, max_length=200_000)

    @field_validator(
        "parent_renderer_package_inventory_relative_path",
        "sidecar_package_inventory_relative_path",
        "sidecar_command_relative_path",
        "sidecar_interpreter_root_relative_path",
        "sidecar_interpreter_relative_path",
        "sidecar_module_root_relative_path",
        "sidecar_module_relative_path",
        "sidecar_configuration_template_relative_path",
        "model_inventory_relative_path",
        "local_model_root_relative_path",
    )
    @classmethod
    def _paths_are_relative(cls, value: str) -> str:
        return _relative_path(value)

    @field_validator(
        "raster_text_model_inventory_relative_path",
        "raster_text_local_model_root_relative_path",
    )
    @classmethod
    def _optional_paths_are_relative(cls, value: str | None) -> str | None:
        return None if value is None else _relative_path(value)

    @model_validator(mode="after")
    def _raster_text_paths_are_paired(self) -> "RasterOfflineBundleManifestV1":
        if (self.raster_text_model_inventory_relative_path is None) != (
            self.raster_text_local_model_root_relative_path is None
        ):
            raise ValueError("raster text material paths must be paired")
        return self

    @field_validator("parent_renderer_required_packages", "sidecar_required_packages")
    @classmethod
    def _package_names_are_unique(cls, value: list[str]) -> list[str]:
        if len(value) != len(set(value)) or any(
            not isinstance(name, str) or not name or len(name) > 128 for name in value
        ):
            raise ValueError("required package list is invalid")
        return value

    def required_file_paths(self) -> tuple[str, ...]:
        return (
            self.parent_renderer_package_inventory_relative_path,
            self.sidecar_package_inventory_relative_path,
            self.sidecar_command_relative_path,
            self.sidecar_interpreter_relative_path,
            self.sidecar_module_relative_path,
            self.sidecar_configuration_template_relative_path,
            self.model_inventory_relative_path,
        )


class RasterOfflineBundleReleaseSelectorV1(_WireModel):
    """One canonical current-release selection for post-legacy bundles.

    Its release path is constrained again while loading because the allowed
    content-addressed directory also depends on the declared manifest hash.
    Keeping that check at the resolver boundary prevents a selector from
    becoming a generic directory chooser.
    """

    schema_version: str = "1.0"
    bundle_id: str = _BUNDLE_DIRECTORY
    release_manifest_sha256: str = Field(pattern=_SHA256)
    release_relative_path: str = Field(min_length=1, max_length=240)

    _path_is_relative = field_validator("release_relative_path")(_relative_path)

    @model_validator(mode="after")
    def _is_the_fixed_release_kind(self) -> "RasterOfflineBundleReleaseSelectorV1":
        if (
            not {
                "schema_version",
                "bundle_id",
                "release_manifest_sha256",
                "release_relative_path",
            }
            <= self.model_fields_set
            or self.schema_version != "1.0"
            or self.bundle_id != _BUNDLE_DIRECTORY
        ):
            raise ValueError("raster release selector is invalid")
        return self


@dataclass(frozen=True, slots=True)
class VerifiedRasterOfflineBundleV1:
    """Verified local material used to prepare one runtime configuration."""

    document_preview_root: Path
    bundle_root: Path
    bundle_root_relative_path: str
    manifest: RasterOfflineBundleManifestV1
    manifest_sha256: str
    file_sha256_by_relative_path: dict[str, str]
    # This is non-empty only while a fixed legacy root is being migrated.  It
    # records paths that passed the narrow generated-bytecode exception so
    # the release builder can exclude them rather than carry unmanifested
    # bytes into a new immutable release.
    historical_generated_bytecode_cache_relative_paths: frozenset[str]

    def bundle_relative_path(self, path: str) -> str:
        """Return a document-preview-relative path for one declared bundle file."""

        _relative_path(path)
        return f"{self.bundle_root_relative_path}/{path}"

    def file_sha256(self, path: str) -> str:
        try:
            return self.file_sha256_by_relative_path[path]
        except KeyError:
            raise RasterOfflineBundleError("offline_bundle_manifest_invalid") from None

    @property
    def raster_profile_revision(self) -> str:
        """Stable non-secret fingerprint for the whole selected OCR runtime."""

        # The canonical manifest is exhaustive for all bundle files.  Its
        # digest is therefore the single release fingerprint also understood
        # by the runtime's public ``make_*`` helper.
        return f"synthetic-raster-v1:{self.manifest_sha256[:24]}"


class RasterTextRuntimeProfileMaterialV1(Protocol):
    """The narrow material surface needed to write a direct-text profile.

    ``VerifiedRasterOfflineBundleV1`` satisfies this protocol after an
    exhaustive bundle walk.  The automatic text activation may also use the
    separately attested ``VerifiedRasterTextReleaseReferenceV1`` before that
    expensive generic walk.  Generic bundle consumers must continue to
    require ``VerifiedRasterOfflineBundleV1`` explicitly.
    """

    @property
    def document_preview_root(self) -> Path: ...

    @property
    def bundle_root(self) -> Path: ...

    @property
    def bundle_root_relative_path(self) -> str: ...

    @property
    def manifest(self) -> RasterOfflineBundleManifestV1: ...

    @property
    def manifest_sha256(self) -> str: ...

    @property
    def file_sha256_by_relative_path(self) -> Mapping[str, str]: ...

    def bundle_relative_path(self, path: str) -> str: ...

    def file_sha256(self, path: str) -> str: ...


@dataclass(frozen=True, slots=True)
class VerifiedRasterTextReleaseReferenceV1:
    """Direct-text release evidence, deliberately not a generic bundle.

    The bootstrapper creates this only after checking the source-bound text
    intent, derivation receipt, direct sidecar files and seven-file model
    root.  It does *not* assert that every unrelated structural bundle file
    has been walked.  Candidate/runtime validation and the final activation
    recheck retain those broader boundaries.
    """

    document_preview_root: Path
    bundle_root: Path
    bundle_root_relative_path: str
    manifest: RasterOfflineBundleManifestV1
    manifest_sha256: str
    file_sha256_by_relative_path: Mapping[str, str]

    def bundle_relative_path(self, path: str) -> str:
        _relative_path(path)
        return f"{self.bundle_root_relative_path}/{path}"

    def file_sha256(self, path: str) -> str:
        try:
            return self.file_sha256_by_relative_path[path]
        except KeyError:
            raise RasterOfflineBundleError(
                "raster_text_offline_bundle_material_unavailable"
            ) from None


@dataclass(frozen=True, slots=True)
class PreparedRasterRuntimeProfileV2:
    """Generated immutable runtime config derived only from a verified bundle."""

    raster_profile_revision: str
    runtime_configuration_relative_path: str
    runtime_configuration_sha256: str
    sidecar_configuration_relative_path: str
    sidecar_configuration_sha256: str


@dataclass(frozen=True, slots=True)
class PreparedRasterTextRuntimeProfileV1:
    """Hash-bound direct PP-OCRv5 text profile derived from one bundle."""

    raster_text_profile_revision: str
    runtime_configuration_relative_path: str
    runtime_configuration_sha256: str
    minimum_page_confidence: float


@dataclass(frozen=True, slots=True)
class VerifiedRasterTextDirectModelMaterialV1:
    """The exact seven-file detector/recognizer boundary for text OCR."""

    local_model_root_relative_path: str
    model_inventory_relative_path: str
    model_inventory_sha256: str
    source_snapshot_sha256: str


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _document_preview_root(path: Path) -> Path:
    try:
        if path.is_symlink():
            raise RasterOfflineBundleError("offline_bundle_root_invalid")
        root = path.resolve(strict=True)
        if (
            root.is_symlink()
            or not root.is_dir()
            or tuple(part.casefold() for part in root.parts[-3:]) != _DOCUMENT_PREVIEW_ROOT_SUFFIX
        ):
            raise RasterOfflineBundleError("offline_bundle_root_invalid")
        return root
    except RasterOfflineBundleError:
        raise
    except OSError:
        raise RasterOfflineBundleError("offline_bundle_root_invalid") from None


def _windows_operational_path(path: Path, *, force_short: bool = False) -> Path:
    """Use an ASCII DOS path for deep Windows bundle trees after validation.

    The canonical long path remains the security identity.  Windows CPython
    package files below a Korean-named project root can exceed the legacy path
    limit during copy/import, however, so filesystem operations use the
    equivalent existing 8.3 path.  A volume without short names cannot safely
    host this fixed native bundle and fails closed.
    """

    if os.name != "nt":
        return path
    try:
        import ctypes

        value = str(path)
        if not force_short and value.isascii() and len(value) <= 220:
            return path
        kernel32 = ctypes.windll.kernel32
        required = kernel32.GetShortPathNameW(value, None, 0)
        if required <= 0:
            raise OSError("short path unavailable")
        buffer = ctypes.create_unicode_buffer(required + 1)
        written = kernel32.GetShortPathNameW(value, buffer, len(buffer))
        short_value = buffer.value if 0 < written < len(buffer) else ""
    except Exception:
        raise RasterOfflineBundleError("offline_bundle_path_unsupported") from None
    if not short_value or not short_value.isascii():
        raise RasterOfflineBundleError("offline_bundle_path_unsupported")
    return Path(short_value)


def _local_path_identity(path: Path) -> str:
    """Normalize Windows extended/short spellings for containment comparison."""

    value = os.path.realpath(str(path))
    if os.name == "nt" and value.startswith("\\\\?\\"):
        value = value[4:]
    return os.path.normcase(os.path.normpath(value))


def _bundle_path(bundle_root: Path, relative_path: str, *, directory: bool = False) -> Path:
    _relative_path(relative_path)
    candidate = bundle_root
    try:
        canonical_root = bundle_root.resolve(strict=True)
        for part in PurePosixPath(relative_path).parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise RasterOfflineBundleError("offline_bundle_manifest_invalid")
        resolved = candidate.resolve(strict=True)
        root_identity = _local_path_identity(canonical_root)
        resolved_identity = _local_path_identity(resolved)
        if os.path.commonpath((root_identity, resolved_identity)) != root_identity:
            raise RasterOfflineBundleError("offline_bundle_manifest_invalid")
        if (
            resolved.is_symlink()
            or (directory and not resolved.is_dir())
            or (not directory and not resolved.is_file())
        ):
            raise RasterOfflineBundleError("offline_bundle_manifest_invalid")
        return _windows_operational_path(resolved)
    except RasterOfflineBundleError:
        raise
    except OSError:
        raise RasterOfflineBundleError("offline_bundle_manifest_invalid") from None


def _regular_files(root: Path) -> Iterable[tuple[str, Path]]:
    """Enumerate without following any linked component or hidden extra file."""

    try:
        for child in sorted(root.iterdir(), key=lambda value: value.name):
            if child.is_symlink():
                raise RasterOfflineBundleError("offline_bundle_manifest_invalid")
            if child.is_dir():
                for nested in _regular_files(child):
                    yield (f"{child.name}/{nested[0]}", nested[1])
            elif child.is_file():
                yield (child.name, child)
            else:
                raise RasterOfflineBundleError("offline_bundle_manifest_invalid")
    except RasterOfflineBundleError:
        raise
    except OSError:
        raise RasterOfflineBundleError("offline_bundle_manifest_invalid") from None


def _sha256_regular_file(path: Path, *, expected_size: int | None = None) -> tuple[str, int]:
    try:
        if path.is_symlink() or not path.is_file():
            raise RasterOfflineBundleError("offline_bundle_manifest_invalid")
        size = path.stat().st_size
        if size > _MAX_BUNDLE_FILE_BYTES or (expected_size is not None and size != expected_size):
            raise RasterOfflineBundleError("offline_bundle_manifest_invalid")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest(), size
    except RasterOfflineBundleError:
        raise
    except OSError:
        raise RasterOfflineBundleError("offline_bundle_manifest_invalid") from None


def _read_canonical_json(path: Path, *, limit: int) -> tuple[dict[str, Any], bytes]:
    try:
        if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= limit:
            raise RasterOfflineBundleError("offline_bundle_manifest_invalid")
        raw = path.read_bytes()
        value = json.loads(raw)
    except RasterOfflineBundleError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise RasterOfflineBundleError("offline_bundle_manifest_invalid") from None
    if not isinstance(value, dict) or raw != _canonical_json(value):
        raise RasterOfflineBundleError("offline_bundle_manifest_invalid")
    return value, raw


def _validate_package_inventory(
    *, bundle_root: Path, relative_path: str, required_packages: list[str]
) -> None:
    """Reject a malformed inventory before either interpreter imports OCR code.

    Package versions are verified by their owning interpreter later.  This
    first boundary only ensures that a bundle did not omit a required package
    from the inventory it claims to pin.
    """

    raw, _bytes = _read_canonical_json(
        _bundle_path(bundle_root, relative_path), limit=_MAX_CONFIGURATION_BYTES
    )
    packages = raw.get("packages")
    if not isinstance(packages, list):
        raise RasterOfflineBundleError("offline_bundle_manifest_invalid")
    names = {
        item.get("name")
        for item in packages
        if isinstance(item, dict)
        and isinstance(item.get("name"), str)
        and isinstance(item.get("version"), str)
        and item["name"]
        and item["version"]
    }
    if len(names) != len(packages) or not set(required_packages) <= names:
        raise RasterOfflineBundleError("offline_bundle_manifest_invalid")


def verify_parent_renderer_inventory(bundle: RasterTextRuntimeProfileMaterialV1) -> None:
    """Verify only the lightweight parent PDF renderer environment.

    PP-Structure packages are deliberately absent here.  They belong to the
    separate sidecar interpreter and are verified by that child before it
    imports PaddleOCR/PaddleX/PaddlePaddle during the candidate run.
    """

    manifest = bundle.manifest
    if manifest.parent_renderer_required_packages != ["pypdfium2", "Pillow"]:
        raise RasterOfflineBundleError("offline_bundle_manifest_invalid")
    payload, _raw = _read_canonical_json(
        _bundle_path(bundle.bundle_root, manifest.parent_renderer_package_inventory_relative_path),
        limit=_MAX_CONFIGURATION_BYTES,
    )
    package_rows = payload.get("packages")
    if not isinstance(package_rows, list):
        raise RasterOfflineBundleError("offline_bundle_manifest_invalid")
    versions: dict[str, str] = {}
    for row in package_rows:
        if (
            not isinstance(row, dict)
            or not isinstance(row.get("name"), str)
            or not isinstance(row.get("version"), str)
            or not row["name"]
            or not row["version"]
            or row["name"] in versions
        ):
            raise RasterOfflineBundleError("offline_bundle_manifest_invalid")
        versions[row["name"]] = row["version"]
    try:
        if any(
            importlib.metadata.version(package_name) != versions[package_name]
            for package_name in manifest.parent_renderer_required_packages
        ):
            raise RasterOfflineBundleError("offline_bundle_parent_renderer_drift")
    except RasterOfflineBundleError:
        raise
    except (KeyError, importlib.metadata.PackageNotFoundError):
        raise RasterOfflineBundleError("offline_bundle_parent_renderer_drift") from None


def _validate_sidecar_template(
    *,
    bundle_root: Path,
    manifest: RasterOfflineBundleManifestV1,
    allow_historical_safe_configuration_absence: bool = False,
) -> None:
    """Require all eight explicit PP-Structure model directories up front."""

    from customs_ai_agent.service.ppstructurev3_sidecar import (
        PpStructureV3SidecarConfigurationV1,
        PpStructureV3SidecarError,
    )

    raw, _bytes = _read_canonical_json(
        _bundle_path(bundle_root, manifest.sidecar_configuration_template_relative_path),
        limit=_MAX_CONFIGURATION_BYTES,
    )
    try:
        configuration = PpStructureV3SidecarConfigurationV1.model_validate(raw)
    except ValidationError:
        if not allow_historical_safe_configuration_absence:
            raise RasterOfflineBundleError("offline_bundle_manifest_invalid") from None
        # This narrow path is used only to copy a previously provisioned
        # legacy bundle into a new release.  Require *all* new fields to be
        # absent: a partial/invalid hand edit must not be repaired by code.
        safe_fields = set(_HISTORICAL_SAFE_SIDECAR_CONFIGURATION_VALUES)
        if safe_fields.intersection(raw) or not safe_fields:
            raise RasterOfflineBundleError("offline_bundle_manifest_invalid") from None
        try:
            configuration = PpStructureV3SidecarConfigurationV1.model_validate(
                {**raw, **_HISTORICAL_SAFE_SIDECAR_CONFIGURATION_VALUES}
            )
        except ValidationError:
            raise RasterOfflineBundleError("offline_bundle_manifest_invalid") from None
    try:
        if configuration.extractor_revision != manifest.extractor_revision:
            raise RasterOfflineBundleError("offline_bundle_manifest_invalid")
        model_root = _bundle_path(
            bundle_root, manifest.local_model_root_relative_path, directory=True
        )
        configuration.validate_model_directories(model_root)
        if (
            len(configuration.model_directories()) != 8
            or len(set(configuration.model_directories())) != 8
        ):
            raise RasterOfflineBundleError("offline_bundle_manifest_invalid")
    except RasterOfflineBundleError:
        raise
    except (PpStructureV3SidecarError, ValueError):
        raise RasterOfflineBundleError("offline_bundle_manifest_invalid") from None


def _is_allowed_historical_generated_bytecode_cache(
    *,
    path: str,
    manifest: RasterOfflineBundleManifestV1,
    declared_paths: set[str],
) -> bool:
    """Accept one non-executable legacy cache artifact only during migration.

    CPython stores compiled bytecode as ``<parent>/__pycache__/<stem>.
    cpython-<tag>[.opt-N].pyc``.  The cache is safe to discard only when the
    matching ``<parent>/<stem>.py`` already has a manifest entry whose bytes
    are verified separately.  The function intentionally does not inspect or
    trust the cache bytes themselves.
    """

    try:
        module_root = PurePosixPath(manifest.sidecar_module_root_relative_path)
        candidate = PurePosixPath(path)
        relative = candidate.relative_to(module_root)
    except ValueError:
        return False
    if relative.parent.name != "__pycache__":
        return False
    cache_name = _HISTORICAL_GENERATED_BYTECODE_CACHE_NAME.fullmatch(relative.name)
    if cache_name is None:
        return False
    source = module_root / relative.parent.parent / f"{cache_name.group('stem')}.py"
    return source.as_posix() in declared_paths


def _relative_file_is_within(*, directory: str, file_path: str) -> bool:
    """Return whether a declared file is a strict child of a declared directory."""

    root = PurePosixPath(directory)
    candidate = PurePosixPath(file_path)
    try:
        return candidate != root and candidate.relative_to(root) != PurePosixPath(".")
    except ValueError:
        return False


def _declared_directory_tree_sha256(
    *, manifest: RasterOfflineBundleManifestV1, directory: str
) -> str:
    """Derive one complete controlled-directory digest from declared files.

    A module root is a directory, so it cannot have one file SHA itself.  This
    digest binds the complete relative file set, content hashes, and sizes
    below it.  The runtime recomputes the same value before every v2 launch,
    which detects both a replacement and an injected shadow package.
    """

    root = PurePosixPath(directory)
    files: list[dict[str, object]] = []
    for item in manifest.files:
        path = PurePosixPath(item.path)
        try:
            relative = path.relative_to(root)
        except ValueError:
            continue
        if relative == PurePosixPath("."):
            raise RasterOfflineBundleError("offline_bundle_manifest_invalid")
        files.append(
            {
                "path": relative.as_posix(),
                "sha256": item.sha256,
                "byte_size": item.byte_size,
            }
        )
    if not files:
        raise RasterOfflineBundleError("offline_bundle_manifest_invalid")
    return hashlib.sha256(_canonical_json({"files": files})).hexdigest()


def _module_root_tree_sha256_from_manifest(manifest: RasterOfflineBundleManifestV1) -> str:
    return _declared_directory_tree_sha256(
        manifest=manifest,
        directory=manifest.sidecar_module_root_relative_path,
    )


def _interpreter_root_tree_sha256_from_manifest(manifest: RasterOfflineBundleManifestV1) -> str:
    return _declared_directory_tree_sha256(
        manifest=manifest,
        directory=manifest.sidecar_interpreter_root_relative_path,
    )


def verify_raster_text_direct_model_material(
    bundle: RasterTextRuntimeProfileMaterialV1,
) -> VerifiedRasterTextDirectModelMaterialV1:
    """Verify the dedicated seven-file direct-text model boundary.

    The structural manifest is exhaustive too, but that is not sufficient for
    the text-only lane: accepting its full ``models`` root would make a
    profile look lightweight while silently binding all eight PP-Structure
    models.  This helper verifies a separate canonical inventory and its
    exact detector/recognizer subtree before a profile can name it.
    """

    manifest = bundle.manifest
    inventory_relative_path = manifest.raster_text_model_inventory_relative_path
    model_root_relative_path = manifest.raster_text_local_model_root_relative_path
    try:
        if (
            inventory_relative_path is None
            or model_root_relative_path is None
            or inventory_relative_path == manifest.model_inventory_relative_path
            or model_root_relative_path == manifest.local_model_root_relative_path
        ):
            raise RasterOfflineBundleError("raster_text_offline_bundle_material_unavailable")
        inventory_path = _bundle_path(bundle.bundle_root, inventory_relative_path)
        model_root = _bundle_path(bundle.bundle_root, model_root_relative_path, directory=True)
        inventory, _raw = _read_canonical_json(inventory_path, limit=_MAX_CONFIGURATION_BYTES)
        if set(inventory) != {
            "schema_version",
            "inventory_kind",
            "source_bundle_manifest_sha256",
            "source_model_root_relative_path",
            "source_model_inventory_sha256",
            "source_snapshot_sha256",
            "artifacts",
        }:
            raise RasterOfflineBundleError("raster_text_offline_bundle_material_unavailable")
        source_bundle_manifest_sha256 = inventory.get("source_bundle_manifest_sha256")
        source_model_inventory_sha256 = inventory.get("source_model_inventory_sha256")
        source_snapshot_sha256_value = inventory.get("source_snapshot_sha256")
        if (
            inventory.get("schema_version") != _RASTER_TEXT_MODEL_INVENTORY_SCHEMA_VERSION
            or inventory.get("inventory_kind") != _RASTER_TEXT_MODEL_INVENTORY_KIND
            or not isinstance(source_bundle_manifest_sha256, str)
            or re.fullmatch(_SHA256, source_bundle_manifest_sha256) is None
            or inventory.get("source_model_root_relative_path")
            != manifest.local_model_root_relative_path
            or not isinstance(source_model_inventory_sha256, str)
            or re.fullmatch(_SHA256, source_model_inventory_sha256) is None
            or not isinstance(source_snapshot_sha256_value, str)
            or re.fullmatch(_SHA256, source_snapshot_sha256_value) is None
        ):
            raise RasterOfflineBundleError("raster_text_offline_bundle_material_unavailable")
        artifacts = inventory.get("artifacts")
        if not isinstance(artifacts, list) or len(artifacts) != len(
            _RASTER_TEXT_DIRECT_MODEL_PATHS
        ):
            raise RasterOfflineBundleError("raster_text_offline_bundle_material_unavailable")
        normalized_artifacts: list[dict[str, object]] = []
        for expected_path, artifact in zip(_RASTER_TEXT_DIRECT_MODEL_PATHS, artifacts, strict=True):
            if not isinstance(artifact, dict) or set(artifact) != {
                "relative_path",
                "sha256",
                "byte_size",
            }:
                raise RasterOfflineBundleError("raster_text_offline_bundle_material_unavailable")
            relative_path = artifact.get("relative_path")
            digest = artifact.get("sha256")
            byte_size = artifact.get("byte_size")
            if (
                relative_path != expected_path
                or not isinstance(digest, str)
                or re.fullmatch(_SHA256, digest) is None
                or isinstance(byte_size, bool)
                or not isinstance(byte_size, int)
                or not 0 < byte_size <= _MAX_BUNDLE_FILE_BYTES
            ):
                raise RasterOfflineBundleError("raster_text_offline_bundle_material_unavailable")
            normalized_artifacts.append(
                {"relative_path": relative_path, "sha256": digest, "byte_size": byte_size}
            )
        source_snapshot_sha256 = hashlib.sha256(
            _canonical_json({"files": normalized_artifacts})
        ).hexdigest()
        if source_snapshot_sha256_value != source_snapshot_sha256:
            raise RasterOfflineBundleError("raster_text_offline_bundle_material_unavailable")

        actual_paths: set[str] = set()
        for candidate in model_root.rglob("*"):
            if candidate.is_symlink():
                raise RasterOfflineBundleError("raster_text_offline_bundle_material_unavailable")
            if candidate.is_file():
                actual_paths.add(candidate.relative_to(model_root).as_posix())
            elif not candidate.is_dir():
                raise RasterOfflineBundleError("raster_text_offline_bundle_material_unavailable")
        if actual_paths != set(_RASTER_TEXT_DIRECT_MODEL_PATHS):
            raise RasterOfflineBundleError("raster_text_offline_bundle_material_unavailable")
        for artifact in normalized_artifacts:
            relative_path = str(artifact["relative_path"])
            expected_digest = str(artifact["sha256"])
            expected_size_value = artifact["byte_size"]
            if not isinstance(expected_size_value, int):
                raise RasterOfflineBundleError("raster_text_offline_bundle_material_unavailable")
            expected_size = expected_size_value
            candidate = _bundle_path(model_root, relative_path)
            digest, size = _sha256_regular_file(candidate, expected_size=expected_size)
            bundle_path = f"{model_root_relative_path}/{relative_path}"
            if (
                size != expected_size
                or digest != expected_digest
                or bundle.file_sha256(bundle_path) != expected_digest
            ):
                raise RasterOfflineBundleError("raster_text_offline_bundle_material_unavailable")
        if (
            bundle.file_sha256(inventory_relative_path)
            != hashlib.sha256(_canonical_json(inventory)).hexdigest()
        ):
            raise RasterOfflineBundleError("raster_text_offline_bundle_material_unavailable")
        return VerifiedRasterTextDirectModelMaterialV1(
            local_model_root_relative_path=model_root_relative_path,
            model_inventory_relative_path=inventory_relative_path,
            model_inventory_sha256=bundle.file_sha256(inventory_relative_path),
            source_snapshot_sha256=source_snapshot_sha256,
        )
    except RasterOfflineBundleError:
        raise
    except (OSError, TypeError, ValueError):
        raise RasterOfflineBundleError("raster_text_offline_bundle_material_unavailable") from None


def _load_verified_raster_offline_bundle_from_paths(
    *,
    document_preview_root: Path,
    bundle_root: Path,
    unavailable_code: str,
    allow_historical_safe_configuration_absence: bool = False,
    allow_historical_generated_bytecode_cache_extras: bool = False,
) -> VerifiedRasterOfflineBundleV1:
    """Verify one bundle tree without permitting any fallback material.

    The normal loader supplies the fixed final location.  The automatic
    bootstrapper uses this same verifier for a sibling staging directory
    before the directory is atomically published.  Keeping the byte-level
    checks shared prevents a staging-only validation gap.
    """

    root = _document_preview_root(document_preview_root)
    try:
        if bundle_root.is_symlink() or not bundle_root.is_dir():
            raise RasterOfflineBundleError(unavailable_code)
        canonical_bundle_root = bundle_root.resolve(strict=True)
        bundle_relative_path = canonical_bundle_root.relative_to(root).as_posix()
        _relative_path(bundle_relative_path)
        # A content-addressed release adds a 64-character directory below an
        # already deep document-preview root.  Convert it to its existing
        # DOS spelling even when the root itself happens to be below the
        # ordinary threshold; otherwise descendants can cross Win32's legacy
        # path limit during manifest traversal.
        bundle_root = _windows_operational_path(
            canonical_bundle_root,
            force_short=(
                bundle_relative_path.startswith(f"{_BUNDLE_RELEASES_DIRECTORY}/")
                or canonical_bundle_root.name.startswith(
                    f".{_BUNDLE_RELEASE_COLLECTION_DIRECTORY}.release-staging-"
                )
            ),
        )
    except RasterOfflineBundleError:
        raise
    except OSError:
        raise RasterOfflineBundleError(unavailable_code) from None
    manifest_path = bundle_root / _MANIFEST_FILE
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise RasterOfflineBundleError(unavailable_code)
    raw_manifest, manifest_bytes = _read_canonical_json(manifest_path, limit=_MAX_MANIFEST_BYTES)
    try:
        manifest = RasterOfflineBundleManifestV1.model_validate(raw_manifest)
    except ValidationError:
        raise RasterOfflineBundleError("offline_bundle_manifest_invalid") from None
    if (
        manifest.schema_version != "1.0"
        or manifest.bundle_id != _BUNDLE_DIRECTORY
        or manifest.parent_renderer_required_packages != ["pypdfium2", "Pillow"]
        # The isolated child imports PP-Structure through all three of these
        # packages.  Allowing a bundle to name an arbitrary three-package
        # subset would make its import-before-validation boundary incomplete.
        # The parent never imports them; only the child verifies their pinned
        # versions immediately before its Paddle import.
        or set(manifest.sidecar_required_packages) != _SIDECAR_REQUIRED_PACKAGES
        or len(manifest.sidecar_required_packages) != len(_SIDECAR_REQUIRED_PACKAGES)
        or tuple(item.path for item in manifest.files)
        != tuple(sorted(item.path for item in manifest.files))
        or len({item.path for item in manifest.files}) != len(manifest.files)
    ):
        raise RasterOfflineBundleError("offline_bundle_manifest_invalid")
    declared = {item.path: item for item in manifest.files}
    actual = {path: file for path, file in _regular_files(bundle_root) if path != _MANIFEST_FILE}
    extra_paths = set(actual).difference(declared)
    missing_paths = set(declared).difference(actual)
    if (
        missing_paths
        or any(path == _MANIFEST_FILE for path in declared)
        or (
            extra_paths
            and (
                not allow_historical_generated_bytecode_cache_extras
                or not all(
                    _is_allowed_historical_generated_bytecode_cache(
                        path=path,
                        manifest=manifest,
                        declared_paths=set(declared),
                    )
                    for path in extra_paths
                )
            )
        )
    ):
        raise RasterOfflineBundleError("offline_bundle_manifest_invalid")
    hashes: dict[str, str] = {}
    for relative_path, declaration in declared.items():
        digest, _size = _sha256_regular_file(
            actual[relative_path], expected_size=declaration.byte_size
        )
        if digest != declaration.sha256:
            raise RasterOfflineBundleError("offline_bundle_drift")
        hashes[relative_path] = digest
    if any(path not in declared for path in manifest.required_file_paths()):
        raise RasterOfflineBundleError("offline_bundle_manifest_invalid")
    # v2 has no independently executable wrapper contract. The declared
    # command is the audited sidecar source itself, so the bytes checked as
    # ``sidecar_module`` are exactly the bytes passed to the isolated child
    # interpreter. Keeping two paths would create a second launch surface
    # that the module binding does not describe.
    if manifest.sidecar_command_relative_path != manifest.sidecar_module_relative_path:
        raise RasterOfflineBundleError("offline_bundle_manifest_invalid")
    if not _relative_file_is_within(
        directory=manifest.sidecar_module_root_relative_path,
        file_path=manifest.sidecar_module_relative_path,
    ):
        raise RasterOfflineBundleError("offline_bundle_manifest_invalid")
    if not _relative_file_is_within(
        directory=manifest.sidecar_module_root_relative_path,
        file_path=manifest.sidecar_command_relative_path,
    ) or not _relative_file_is_within(
        directory=manifest.sidecar_interpreter_root_relative_path,
        file_path=manifest.sidecar_interpreter_relative_path,
    ):
        raise RasterOfflineBundleError("offline_bundle_manifest_invalid")
    _bundle_path(bundle_root, manifest.sidecar_interpreter_relative_path)
    _bundle_path(bundle_root, manifest.sidecar_interpreter_root_relative_path, directory=True)
    _bundle_path(bundle_root, manifest.sidecar_module_root_relative_path, directory=True)
    _bundle_path(bundle_root, manifest.sidecar_command_relative_path)
    _bundle_path(bundle_root, manifest.sidecar_module_relative_path)
    _module_root_tree_sha256_from_manifest(manifest)
    _interpreter_root_tree_sha256_from_manifest(manifest)
    _bundle_path(bundle_root, manifest.local_model_root_relative_path, directory=True)
    _validate_package_inventory(
        bundle_root=bundle_root,
        relative_path=manifest.parent_renderer_package_inventory_relative_path,
        required_packages=manifest.parent_renderer_required_packages,
    )
    _validate_package_inventory(
        bundle_root=bundle_root,
        relative_path=manifest.sidecar_package_inventory_relative_path,
        required_packages=manifest.sidecar_required_packages,
    )
    _validate_sidecar_template(
        bundle_root=bundle_root,
        manifest=manifest,
        allow_historical_safe_configuration_absence=allow_historical_safe_configuration_absence,
    )
    return VerifiedRasterOfflineBundleV1(
        document_preview_root=root,
        bundle_root=bundle_root,
        bundle_root_relative_path=bundle_relative_path,
        manifest=manifest,
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
        file_sha256_by_relative_path=hashes,
        historical_generated_bytecode_cache_relative_paths=frozenset(extra_paths),
    )


def _release_selector_relative_path(manifest_sha256: str) -> str:
    if not isinstance(manifest_sha256, str) or len(manifest_sha256) != 64:
        raise RasterOfflineBundleError("offline_bundle_release_selector_invalid")
    if any(character not in "0123456789abcdef" for character in manifest_sha256):
        raise RasterOfflineBundleError("offline_bundle_release_selector_invalid")
    return f"{_BUNDLE_RELEASES_DIRECTORY}/{manifest_sha256}"


def _load_verified_raster_offline_bundle_from_release_selector(
    *, root: Path, operational_root: Path
) -> VerifiedRasterOfflineBundleV1:
    """Resolve exactly one content-addressed release, never a fallback tree."""

    collection = operational_root / _BUNDLE_RELEASE_COLLECTION_DIRECTORY
    selector_path = collection / _BUNDLE_RELEASE_SELECTOR_FILE
    try:
        if collection.is_symlink() or not collection.is_dir():
            raise RasterOfflineBundleError("offline_bundle_release_selector_invalid")
        selector_raw, _selector_bytes = _read_canonical_json(
            selector_path, limit=_MAX_CONFIGURATION_BYTES
        )
        selector = RasterOfflineBundleReleaseSelectorV1.model_validate(selector_raw)
        expected_relative = _release_selector_relative_path(selector.release_manifest_sha256)
        if selector.release_relative_path != expected_relative:
            raise RasterOfflineBundleError("offline_bundle_release_selector_invalid")
        release_root = operational_root
        for part in PurePosixPath(expected_relative).parts:
            release_root = release_root / part
            if release_root.is_symlink():
                raise RasterOfflineBundleError("offline_bundle_release_selector_invalid")
        bundle = _load_verified_raster_offline_bundle_from_paths(
            document_preview_root=root,
            bundle_root=release_root,
            unavailable_code="offline_bundle_release_selector_invalid",
        )
        if (
            bundle.bundle_root_relative_path != expected_relative
            or bundle.manifest_sha256 != selector.release_manifest_sha256
        ):
            raise RasterOfflineBundleError("offline_bundle_release_selector_invalid")
        return bundle
    except RasterOfflineBundleError as error:
        # A selector must never expose a second error mode that callers could
        # interpret as permission to try the legacy root.  Its release and
        # every byte beneath it are one selected unit.
        if error.code == "offline_bundle_release_selector_invalid":
            raise
        raise RasterOfflineBundleError("offline_bundle_release_selector_invalid") from None
    except (OSError, ValidationError, ValueError):
        raise RasterOfflineBundleError("offline_bundle_release_selector_invalid") from None


def load_verified_raster_offline_bundle_release(
    *, document_preview_root: Path, manifest_sha256: str
) -> VerifiedRasterOfflineBundleV1:
    """Verify one exact content-addressed release without consulting selector."""

    root = _document_preview_root(document_preview_root)
    operational_root = _windows_operational_path(root)
    expected_relative = _release_selector_relative_path(manifest_sha256)
    bundle = _load_verified_raster_offline_bundle_from_paths(
        document_preview_root=root,
        bundle_root=operational_root.joinpath(*PurePosixPath(expected_relative).parts),
        unavailable_code="offline_bundle_release_selector_invalid",
    )
    if (
        bundle.bundle_root_relative_path != expected_relative
        or bundle.manifest_sha256 != manifest_sha256
    ):
        raise RasterOfflineBundleError("offline_bundle_release_selector_invalid")
    return bundle


def load_verified_raster_offline_bundle(
    document_preview_root: Path,
) -> VerifiedRasterOfflineBundleV1:
    """Load the selected immutable release, or legacy only before selection.

    The presence of the release collection changes the resolver contract: a
    malformed/missing selector or release is an availability failure, *not*
    permission to silently run the old fixed directory.  This protects an
    approved pointer from accidentally mixing an old engine with newer
    runtime evidence.
    """

    root = _document_preview_root(document_preview_root)
    operational_root = _windows_operational_path(root)
    release_collection = operational_root / _BUNDLE_RELEASE_COLLECTION_DIRECTORY
    try:
        if release_collection.exists() or release_collection.is_symlink():
            return _load_verified_raster_offline_bundle_from_release_selector(
                root=root, operational_root=operational_root
            )
    except OSError:
        raise RasterOfflineBundleError("offline_bundle_release_selector_invalid") from None
    return _load_verified_raster_offline_bundle_from_paths(
        document_preview_root=root,
        bundle_root=operational_root / _BUNDLE_DIRECTORY,
        unavailable_code="offline_bundle_unavailable",
    )


def load_verified_legacy_raster_offline_bundle_for_safe_configuration_migration(
    document_preview_root: Path,
) -> VerifiedRasterOfflineBundleV1:
    """Verify only a legacy base that lacks the complete new safe config set.

    This is not a runtime resolver and never consults/falls back from a
    release selector.  It exists solely so automatic release derivation can
    copy a fully hash-verified old legacy bundle, replace it with the current
    explicit configuration, then run the normal current-schema verifier and
    sidecar smoke before publication.
    """

    root = _document_preview_root(document_preview_root)
    operational_root = _windows_operational_path(root)
    collection = operational_root / _BUNDLE_RELEASE_COLLECTION_DIRECTORY
    try:
        if collection.exists() or collection.is_symlink():
            raise RasterOfflineBundleError("offline_bundle_release_selector_invalid")
    except RasterOfflineBundleError:
        raise
    except OSError:
        raise RasterOfflineBundleError("offline_bundle_release_selector_invalid") from None
    return _load_verified_raster_offline_bundle_from_paths(
        document_preview_root=root,
        bundle_root=operational_root / _BUNDLE_DIRECTORY,
        unavailable_code="offline_bundle_unavailable",
        allow_historical_safe_configuration_absence=True,
        allow_historical_generated_bytecode_cache_extras=True,
    )


def verify_staged_raster_offline_bundle(
    *, document_preview_root: Path, bundle_root: Path
) -> VerifiedRasterOfflineBundleV1:
    """Validate a private sibling staging tree before atomic publication.

    It intentionally accepts neither an arbitrary directory nor the final
    bundle path.  This helper is only an atomic-publish preflight for the
    fixed bootstrapper and cannot broaden the runtime's loader surface.
    """

    root = _document_preview_root(document_preview_root)
    try:
        canonical_bundle_root = bundle_root.resolve(strict=True)
        parent = canonical_bundle_root.parent
        if (
            bundle_root.is_symlink()
            or parent != root
            or not (
                canonical_bundle_root.name.startswith(f".{_BUNDLE_DIRECTORY}.staging-")
                or canonical_bundle_root.name.startswith(
                    f".{_BUNDLE_RELEASE_COLLECTION_DIRECTORY}.release-staging-"
                )
            )
        ):
            raise RasterOfflineBundleError("offline_bundle_staging_invalid")
    except RasterOfflineBundleError:
        raise
    except OSError:
        raise RasterOfflineBundleError("offline_bundle_staging_invalid") from None
    return _load_verified_raster_offline_bundle_from_paths(
        document_preview_root=root,
        bundle_root=bundle_root,
        unavailable_code="offline_bundle_staging_invalid",
    )


def write_or_verify_canonical_local_json(
    *, document_preview_root: Path, relative_path: str, value: object
) -> tuple[str, str]:
    """Create a deterministic local config, or safely reuse exact bytes.

    The function never replaces an existing generated evidence file.  The
    active pointer is the only deliberately replaceable file and has its own
    fsync+atomic writer in the runtime module.
    """

    root = _document_preview_root(document_preview_root)
    _relative_path(relative_path)
    destination = root
    try:
        for part in PurePosixPath(relative_path).parts:
            destination = destination / part
            if destination.is_symlink():
                raise RasterOfflineBundleError("offline_bundle_output_invalid")
        destination.parent.mkdir(parents=True, exist_ok=True)
        if any(parent.is_symlink() for parent in destination.parents if parent != root.parent):
            raise RasterOfflineBundleError("offline_bundle_output_invalid")
    except RasterOfflineBundleError:
        raise
    except OSError:
        raise RasterOfflineBundleError("offline_bundle_output_invalid") from None
    payload = _canonical_json(value)
    digest = hashlib.sha256(payload).hexdigest()
    try:
        descriptor = os.open(str(destination), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        try:
            if destination.is_symlink() or destination.read_bytes() != payload:
                raise RasterOfflineBundleError("offline_bundle_output_conflict")
        except RasterOfflineBundleError:
            raise
        except OSError:
            raise RasterOfflineBundleError("offline_bundle_output_invalid") from None
        return relative_path, digest
    except OSError:
        raise RasterOfflineBundleError("offline_bundle_output_invalid") from None
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except Exception:
        try:
            destination.unlink(missing_ok=True)
        except OSError:
            pass
        raise
    return relative_path, digest


def prepare_raster_runtime_profile_v2(
    bundle: VerifiedRasterOfflineBundleV1,
    *,
    pdfium_version: str,
) -> PreparedRasterRuntimeProfileV2:
    """Materialize hash-bound v2 config without accepting operator values.

    The bundle carries a *template* sidecar configuration because its
    ``raster_profile_revision`` must be filled from the bundle fingerprint.
    Both generated JSON files are immutable per fingerprint; rerunning the
    wrapper only reuses byte-identical files.
    """

    from customs_ai_agent.service.ppstructurev3_sidecar import PpStructureV3SidecarConfigurationV1
    from customs_ai_agent.service.raster_document_preview_runtime import (
        PpStructureV3OfflineProfileV2,
        make_synthetic_raster_profile_revision,
    )

    manifest = bundle.manifest
    if not pdfium_version or len(pdfium_version) > 128:
        raise RasterOfflineBundleError("offline_bundle_parent_renderer_drift")
    expected_bundle_root_relative_path = f"{_BUNDLE_RELEASES_DIRECTORY}/{bundle.manifest_sha256}"
    # Newly materialized profiles must never point back at the mutable
    # fixed-location legacy bundle.  The automatic release selector promotes
    # a verified legacy bundle to this content-addressed location first.  The
    # no-binding form remains only for old v2 profiles already written before
    # this invariant existed.
    if bundle.bundle_root_relative_path != expected_bundle_root_relative_path:
        raise RasterOfflineBundleError("offline_bundle_release_selector_invalid")
    revision = make_synthetic_raster_profile_revision(bundle.manifest_sha256)
    # Keep the fingerprint rule in this module and the runtime helper aligned.
    # A mismatch would create two identities for the same immutable material.
    if revision != bundle.raster_profile_revision:
        raise RasterOfflineBundleError("offline_bundle_profile_fingerprint_invalid")
    revision_suffix = revision.rsplit(":", 1)[1]
    sidecar_relative = f"{_RUNTIME_PROFILES_DIRECTORY}/{revision_suffix}/sidecar.json"
    runtime_relative = f"{_RUNTIME_PROFILES_DIRECTORY}/{revision_suffix}/profile.json"
    for path in (
        sidecar_relative,
        runtime_relative,
        *(
            bundle.bundle_relative_path(value)
            for value in (
                manifest.parent_renderer_package_inventory_relative_path,
                manifest.sidecar_package_inventory_relative_path,
                manifest.model_inventory_relative_path,
                manifest.local_model_root_relative_path,
                manifest.sidecar_command_relative_path,
                manifest.sidecar_interpreter_root_relative_path,
                manifest.sidecar_interpreter_relative_path,
                manifest.sidecar_module_root_relative_path,
                manifest.sidecar_module_relative_path,
            )
        ),
    ):
        if len(path) > 240:
            raise RasterOfflineBundleError("offline_bundle_manifest_invalid")
    try:
        sidecar_template_raw, _raw = _read_canonical_json(
            _bundle_path(bundle.bundle_root, manifest.sidecar_configuration_template_relative_path),
            limit=_MAX_CONFIGURATION_BYTES,
        )
        sidecar_template = PpStructureV3SidecarConfigurationV1.model_validate(sidecar_template_raw)
        if sidecar_template.extractor_revision != manifest.extractor_revision:
            raise RasterOfflineBundleError("offline_bundle_manifest_invalid")
        sidecar_payload = sidecar_template.model_copy(
            update={"raster_profile_revision": revision}
        ).model_dump(mode="json", warnings=False)
    except RasterOfflineBundleError:
        raise
    except (ValidationError, ValueError):
        raise RasterOfflineBundleError("offline_bundle_manifest_invalid") from None
    sidecar_path, sidecar_sha256 = write_or_verify_canonical_local_json(
        document_preview_root=bundle.document_preview_root,
        relative_path=sidecar_relative,
        value=sidecar_payload,
    )
    try:
        from customs_ai_agent.service.ppstructurev3_sidecar import __file__ as sidecar_module_file

        local_module_sha256, _size = _sha256_regular_file(Path(sidecar_module_file))
        if local_module_sha256 != bundle.file_sha256(manifest.sidecar_module_relative_path):
            raise RasterOfflineBundleError("offline_bundle_sidecar_module_drift")
        profile_payload = {
            "schema_version": "2.0",
            "engine_kind": "ppstructurev3",
            "runtime_api": "paddleocr3-ppstructurev3",
            "extractor_revision": manifest.extractor_revision,
            "raster_profile_revision": revision,
            # Make the selected release explicit in both profile and pointer
            # evidence.  The full manifest hash prevents a short revision
            # prefix from acting as a directory-selection capability.
            "bundle_root_relative_path": bundle.bundle_root_relative_path,
            "bundle_manifest_sha256": bundle.manifest_sha256,
            "pdfium_version": pdfium_version,
            "raster_dpi": manifest.raster_dpi,
            "parent_renderer_package_inventory_sha256": bundle.file_sha256(
                manifest.parent_renderer_package_inventory_relative_path
            ),
            "parent_renderer_package_inventory_path": bundle.bundle_relative_path(
                manifest.parent_renderer_package_inventory_relative_path
            ),
            "parent_renderer_required_packages": manifest.parent_renderer_required_packages,
            "sidecar_package_inventory_sha256": bundle.file_sha256(
                manifest.sidecar_package_inventory_relative_path
            ),
            "sidecar_package_inventory_path": bundle.bundle_relative_path(
                manifest.sidecar_package_inventory_relative_path
            ),
            "sidecar_required_packages": manifest.sidecar_required_packages,
            "model_inventory_sha256": bundle.file_sha256(manifest.model_inventory_relative_path),
            "model_inventory_path": bundle.bundle_relative_path(
                manifest.model_inventory_relative_path
            ),
            "local_model_root": bundle.bundle_relative_path(
                manifest.local_model_root_relative_path
            ),
            "sidecar_interpreter_root": bundle.bundle_relative_path(
                manifest.sidecar_interpreter_root_relative_path
            ),
            "sidecar_interpreter_root_tree_sha256": _interpreter_root_tree_sha256_from_manifest(
                manifest
            ),
            "sidecar_interpreter_path": bundle.bundle_relative_path(
                manifest.sidecar_interpreter_relative_path
            ),
            "sidecar_interpreter_sha256": bundle.file_sha256(
                manifest.sidecar_interpreter_relative_path
            ),
            "sidecar_module_root": bundle.bundle_relative_path(
                manifest.sidecar_module_root_relative_path
            ),
            "sidecar_module_root_tree_sha256": _module_root_tree_sha256_from_manifest(manifest),
            "sidecar_command_path": bundle.bundle_relative_path(
                manifest.sidecar_command_relative_path
            ),
            "sidecar_command_sha256": bundle.file_sha256(manifest.sidecar_command_relative_path),
            "sidecar_module_path": bundle.bundle_relative_path(
                manifest.sidecar_module_relative_path
            ),
            "sidecar_module_sha256": local_module_sha256,
            "sidecar_configuration_path": sidecar_path,
            "sidecar_configuration_sha256": sidecar_sha256,
            "sidecar_timeout_seconds": manifest.sidecar_timeout_seconds,
            "evaluation_batch_size": sidecar_template.evaluation_batch_size,
        }
        profile = PpStructureV3OfflineProfileV2.model_validate(profile_payload)
    except RasterOfflineBundleError:
        raise
    except (OSError, ValidationError, ValueError):
        raise RasterOfflineBundleError("offline_bundle_runtime_configuration_invalid") from None
    runtime_path, runtime_sha256 = write_or_verify_canonical_local_json(
        document_preview_root=bundle.document_preview_root,
        relative_path=runtime_relative,
        value=profile.model_dump(mode="json", warnings=False),
    )
    return PreparedRasterRuntimeProfileV2(
        raster_profile_revision=revision,
        runtime_configuration_relative_path=runtime_path,
        runtime_configuration_sha256=runtime_sha256,
        sidecar_configuration_relative_path=sidecar_path,
        sidecar_configuration_sha256=sidecar_sha256,
    )


def prepare_raster_text_runtime_profile_v1(
    bundle: RasterTextRuntimeProfileMaterialV1,
    *,
    pdfium_version: str,
    minimum_page_confidence: float,
) -> PreparedRasterTextRuntimeProfileV1:
    """Create-or-verify the independent direct-text runtime configuration.

    This does not reuse the PP-Structure profile or its pointer/Gate.  The
    model directories and package/interpreter provenance come solely from a
    verified local release manifest; the only runtime setting derived from
    evaluation is the explicit calibrated confidence floor.
    """

    from customs_ai_agent.service.raster_text_preview_runtime import (
        RasterTextOfflineProfileV1,
        raster_text_profile_fingerprint_sha256,
    )

    manifest = bundle.manifest
    if (
        not pdfium_version
        or len(pdfium_version) > 128
        or isinstance(minimum_page_confidence, bool)
        or not isinstance(minimum_page_confidence, (int, float))
        or not 0 <= float(minimum_page_confidence) <= 1
    ):
        raise RasterOfflineBundleError("raster_text_offline_bundle_material_unavailable")
    expected_release = f"{_BUNDLE_RELEASES_DIRECTORY}/{bundle.manifest_sha256}"
    if bundle.bundle_root_relative_path != expected_release:
        raise RasterOfflineBundleError("offline_bundle_release_selector_invalid")
    text_sidecar_relative = (
        f"{manifest.sidecar_module_root_relative_path}/"
        "customs_ai_agent/service/raster_text_sidecar.py"
    )
    try:
        if text_sidecar_relative not in bundle.file_sha256_by_relative_path:
            raise RasterOfflineBundleError("raster_text_offline_bundle_material_unavailable")
        direct_models = verify_raster_text_direct_model_material(bundle)
        template_raw, _template_bytes = _read_canonical_json(
            _bundle_path(bundle.bundle_root, manifest.sidecar_configuration_template_relative_path),
            limit=_MAX_CONFIGURATION_BYTES,
        )
        # Do not infer model selectors from paths.  A release has to declare
        # exactly the direct detector and English recognizer in its immutable
        # model template, and the inventory must have files below both roots.
        if (
            template_raw.get("text_detection_model_name") != "PP-OCRv5_server_det"
            or template_raw.get("text_detection_model_directory") != "text-det"
            or template_raw.get("text_recognition_model_name") != "en_PP-OCRv5_mobile_rec"
            or template_raw.get("text_recognition_model_directory") != "text-rec"
        ):
            raise RasterOfflineBundleError("raster_text_offline_bundle_material_unavailable")
        # Historical provisioning metadata can retain the archive label that
        # originally populated ``text-rec``.  It is not a runtime selector:
        # the authoritative material boundary is the current exhaustive
        # manifest/model inventory plus the exact sidecar configuration above
        # (all rehashed before each child import).  Read the provenance only
        # to require canonical local evidence; never reinterpret an old label
        # as a model alias or let it override the configured en-mobile model.
        _bootstrap_raw, _bootstrap_bytes = _read_canonical_json(
            _bundle_path(bundle.bundle_root, "provenance/bootstrap-receipt.json"),
            limit=_MAX_CONFIGURATION_BYTES,
        )
        # Calculate the profile fingerprint without its self-referential
        # fingerprint/revision fields, then validate the complete Pydantic
        # representation before writing any material.
        profile_payload: dict[str, object] = {
            "schema_version": "1.0",
            "engine_kind": "paddleocr3_direct_text",
            "runtime_api": "paddleocr3-paddlex-3.2",
            "extractor_revision": f"{manifest.extractor_revision}:raster-text-v1",
            "bundle_root_relative_path": bundle.bundle_root_relative_path,
            "bundle_manifest_sha256": bundle.manifest_sha256,
            "pdfium_version": pdfium_version,
            "raster_dpi": manifest.raster_dpi,
            # Reserve the outer 55-second request budget for strict decode,
            # renderer cleanup and response serialization.  The owned native
            # child is killed/reaped after at most 30 seconds, so a timed-out
            # HTTP request never releases the single CPU lane while OCR is
            # still running.
            "sidecar_timeout_seconds": min(30, manifest.sidecar_timeout_seconds),
            "cpu_threads": _RASTER_TEXT_CPU_THREADS,
            "text_detection_limit_side_len": 2048,
            "text_detection_limit_type": "max",
            "text_detection_threshold": 0.3,
            "text_detection_box_threshold": 0.6,
            "text_detection_unclip_ratio": 1.5,
            "minimum_page_confidence": float(minimum_page_confidence),
            "parent_renderer_package_inventory_path": bundle.bundle_relative_path(
                manifest.parent_renderer_package_inventory_relative_path
            ),
            "parent_renderer_package_inventory_sha256": bundle.file_sha256(
                manifest.parent_renderer_package_inventory_relative_path
            ),
            "parent_renderer_required_packages": manifest.parent_renderer_required_packages,
            "sidecar_interpreter_path": bundle.bundle_relative_path(
                manifest.sidecar_interpreter_relative_path
            ),
            "sidecar_interpreter_sha256": bundle.file_sha256(
                manifest.sidecar_interpreter_relative_path
            ),
            "sidecar_interpreter_root": bundle.bundle_relative_path(
                manifest.sidecar_interpreter_root_relative_path
            ),
            "sidecar_interpreter_root_tree_sha256": _interpreter_root_tree_sha256_from_manifest(
                manifest
            ),
            "sidecar_module_path": bundle.bundle_relative_path(text_sidecar_relative),
            "sidecar_module_sha256": bundle.file_sha256(text_sidecar_relative),
            "sidecar_module_root": bundle.bundle_relative_path(
                manifest.sidecar_module_root_relative_path
            ),
            "sidecar_module_root_tree_sha256": _module_root_tree_sha256_from_manifest(manifest),
            "sidecar_package_inventory_path": bundle.bundle_relative_path(
                manifest.sidecar_package_inventory_relative_path
            ),
            "sidecar_package_inventory_sha256": bundle.file_sha256(
                manifest.sidecar_package_inventory_relative_path
            ),
            "sidecar_required_packages": manifest.sidecar_required_packages,
            "local_model_root": bundle.bundle_relative_path(
                direct_models.local_model_root_relative_path
            ),
            "model_inventory_path": bundle.bundle_relative_path(
                direct_models.model_inventory_relative_path
            ),
            "model_inventory_sha256": direct_models.model_inventory_sha256,
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
        profile = RasterTextOfflineProfileV1.model_validate(profile_payload)
    except RasterOfflineBundleError:
        raise
    except (OSError, ValidationError, ValueError, TypeError):
        raise RasterOfflineBundleError("raster_text_offline_bundle_material_unavailable") from None
    suffix = profile.raster_text_profile_revision.rsplit(":", 1)[1]
    relative_path = f"{_RASTER_TEXT_RUNTIME_PROFILES_DIRECTORY}/{suffix}/profile.json"
    path, digest = write_or_verify_canonical_local_json(
        document_preview_root=bundle.document_preview_root,
        relative_path=relative_path,
        value=profile.model_dump(mode="json", warnings=False),
    )
    return PreparedRasterTextRuntimeProfileV1(
        raster_text_profile_revision=profile.raster_text_profile_revision,
        runtime_configuration_relative_path=path,
        runtime_configuration_sha256=digest,
        minimum_page_confidence=profile.minimum_page_confidence,
    )


__all__ = [
    "RasterOfflineBundleError",
    "RasterOfflineBundleFileV1",
    "RasterOfflineBundleManifestV1",
    "RasterOfflineBundleReleaseSelectorV1",
    "PreparedRasterRuntimeProfileV2",
    "PreparedRasterTextRuntimeProfileV1",
    "RasterTextRuntimeProfileMaterialV1",
    "VerifiedRasterTextDirectModelMaterialV1",
    "VerifiedRasterTextReleaseReferenceV1",
    "VerifiedRasterOfflineBundleV1",
    "load_verified_legacy_raster_offline_bundle_for_safe_configuration_migration",
    "load_verified_raster_offline_bundle_release",
    "load_verified_raster_offline_bundle",
    "prepare_raster_runtime_profile_v2",
    "prepare_raster_text_runtime_profile_v1",
    "verify_staged_raster_offline_bundle",
    "verify_parent_renderer_inventory",
    "verify_raster_text_direct_model_material",
    "write_or_verify_canonical_local_json",
]
