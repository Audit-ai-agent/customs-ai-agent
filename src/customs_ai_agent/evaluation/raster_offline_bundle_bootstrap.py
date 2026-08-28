"""Build the one pinned Windows raster-OCR bundle without manual staging.

The normal service and activation path remain completely offline: they only
accept the exhaustive, hash-bound bundle below ``_local/data/document-preview``.
This module is the deliberately separate *provisioning* boundary used when
that bundle does not exist.  It downloads only a checked-in package lock from
PyPI and eight fixed official Paddle model archives, then immediately turns
them into an immutable local bundle.  No downloaded byte is used by OCR until
the isolated sidecar smoke test and exhaustive manifest verification pass.

It is intentionally Windows/CPython-3.11 specific.  The selected Paddle CPU
wheel is platform-native, so pretending a cross-platform bundle exists would
make activation less reliable rather than more portable.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
from typing import Iterator, Mapping
from urllib.request import Request, urlopen
import uuid

from pydantic import ValidationError

from customs_ai_agent.evaluation.raster_offline_bundle import (
    RasterOfflineBundleError,
    RasterOfflineBundleManifestV1,
    VerifiedRasterOfflineBundleV1,
    VerifiedRasterTextReleaseReferenceV1,
    load_verified_legacy_raster_offline_bundle_for_safe_configuration_migration,
    load_verified_raster_offline_bundle,
    load_verified_raster_offline_bundle_release,
    verify_parent_renderer_inventory,
    verify_raster_text_direct_model_material,
    verify_staged_raster_offline_bundle,
)


_BUNDLE_DIRECTORY = "raster-offline-bundle-v1"
_BUNDLE_RELEASE_COLLECTION_DIRECTORY = "raster-offline-bundles-v1"
_BUNDLE_RELEASES_DIRECTORY = "releases"
_BUNDLE_RELEASE_SELECTOR_FILE = "current.json"
_RASTER_TEXT_RELEASE_INTENT_DIRECTORY = "raster-text-release-intents-v1"
_RASTER_TEXT_RELEASE_INTENT_SCHEMA_VERSION = "1.0"
_RASTER_TEXT_RELEASE_INTENT_FIELD = "raster_text_derivation_intent_sha256"
# The source-bound intent index is deliberately tiny.  A fast text reuse
# never searches generic releases; it can inspect only this bounded,
# create-only namespace before falling back to the exhaustive materializer.
_RASTER_TEXT_RELEASE_INTENT_MAX_ENTRIES = 64
_RASTER_TEXT_RELEASE_INTENT_MAX_BYTES = 64 * 1024
_BOOTSTRAP_LOCK_NAME = ".raster-offline-bundle-v1-bootstrap.lock"
_MAX_FILE_BYTES = 1024 * 1024 * 1024
_MAX_ARCHIVE_BYTES = 1024 * 1024 * 1024
_MODEL_BASE_URL = (
    "https://paddle-model-ecology.bj.bcebos.com/paddlex/official_inference_model/paddle3.0.0"
)
_SIDECAR_REQUIREMENTS_RELATIVE_PATH = "requirements/raster-sidecar-win-amd64-cp311.lock"
_EXTRACTOR_REVISION = "ppstructurev3-local-v1-paddleocr-3.7.0"
_PARENT_PACKAGES = ("pypdfium2", "Pillow")
_SIDECAR_REQUIRED_PACKAGES = ("paddleocr", "paddlex", "paddlepaddle")
_SERVER_RECOGNITION_MODEL_NAME = "PP-OCRv5_server_rec"
_LOCAL_ENGLISH_RECOGNITION_MODEL_NAME = "en_PP-OCRv5_mobile_rec"
_LOCAL_ENGLISH_RECOGNITION_SOURCE_RELATIVE_PATH = PurePosixPath("models/paddleocr3/rec")
_RECOGNITION_MODEL_TARGET_DIRECTORY = "text-rec"
_RASTER_TEXT_MODEL_ROOT_RELATIVE_PATH = PurePosixPath("models/raster-text")
_RASTER_TEXT_MODEL_INVENTORY_RELATIVE_PATH = PurePosixPath("inventories/raster-text-models.json")
_RASTER_TEXT_DIRECT_MODEL_PATHS = (
    "text-det/inference.json",
    "text-det/inference.pdiparams",
    "text-det/inference.yml",
    "text-rec/config.json",
    "text-rec/inference.json",
    "text-rec/inference.pdiparams",
    "text-rec/inference.yml",
)
_RELEASE_MANIFEST_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_MODEL_CONFIGURATION_KEYS = (
    "layout_detection_model_name",
    "layout_detection_model_directory",
    "text_detection_model_name",
    "text_detection_model_directory",
    "text_recognition_model_name",
    "text_recognition_model_directory",
    "table_classification_model_name",
    "table_classification_model_directory",
    "wired_table_structure_recognition_model_name",
    "wired_table_structure_recognition_model_directory",
    "wireless_table_structure_recognition_model_name",
    "wireless_table_structure_recognition_model_directory",
    "wired_table_cells_detection_model_name",
    "wired_table_cells_detection_model_directory",
    "wireless_table_cells_detection_model_name",
    "wireless_table_cells_detection_model_directory",
)


class RasterOfflineBundleBootstrapError(RuntimeError):
    """Content-free bootstrap failure; no final bundle is altered."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass(frozen=True, slots=True)
class RasterOfflineBundleBootstrapResultV1:
    """Small, non-sensitive provisioning result for the command wrapper."""

    created: bool
    manifest_sha256: str
    raster_profile_revision: str


@dataclass(frozen=True, slots=True)
class _ModelSpec:
    target_directory: str
    model_name: str

    @property
    def archive_name(self) -> str:
        return f"{self.model_name}_infer.tar"

    @property
    def source_url(self) -> str:
        return f"{_MODEL_BASE_URL}/{self.archive_name}"


@dataclass(frozen=True, slots=True)
class _PinnedLocalRecognitionArtifact:
    """One byte-pinned file in the pre-provisioned English recognition model."""

    filename: str
    byte_size: int
    sha256: str | None = None
    git_blob_sha1: str | None = None


@dataclass(frozen=True, slots=True)
class _RecognitionModelSnapshotV1:
    """Content-only provenance for a copied local recognition model."""

    source_relative_path: str
    tree_sha256: str


@dataclass(frozen=True, slots=True)
class _RasterTextDirectModelArtifactV1:
    """One exact source file copied into the direct-text-only model root."""

    relative_path: str
    sha256: str
    byte_size: int


@dataclass(frozen=True, slots=True)
class _RasterTextDirectModelSnapshotV1:
    """The seven local detector/recognizer bytes used by raster-text-v1."""

    source_bundle_manifest_sha256: str
    source_model_root_relative_path: str
    source_model_inventory_sha256: str
    source_snapshot_sha256: str
    artifacts: tuple[_RasterTextDirectModelArtifactV1, ...]


# This is intentionally the same fixed, pre-provisioned English recognition
# model used by the direct PaddleOCR 3 baseline.  A PP-Structure release never
# reads that baseline directory at runtime: it may only copy this exact tree
# once into its own manifest-bound ``models/text-rec`` directory.
_PINNED_LOCAL_ENGLISH_RECOGNITION_ARTIFACTS = (
    _PinnedLocalRecognitionArtifact(
        filename="config.json",
        byte_size=10_455,
        git_blob_sha1="ee30b810753068b251de393eee66dff31f6d279c",
    ),
    _PinnedLocalRecognitionArtifact(
        filename="inference.json",
        byte_size=217_712,
        git_blob_sha1="fc20fb935854373220fbeff985cbd310f0450608",
    ),
    _PinnedLocalRecognitionArtifact(
        filename="inference.pdiparams",
        byte_size=7_772_315,
        sha256="3ec8a97ed6cefe8568d3e2ee90bb193299b566a7661aa4fd52d224b96b59f66b",
    ),
    _PinnedLocalRecognitionArtifact(
        filename="inference.yml",
        byte_size=3_964,
        git_blob_sha1="91a401a7220881921c249b92852a96f9dbf2132a",
    ),
)


# This is PP-StructureV3's 3.7.2 default eight-model CPU profile.  The
# wireless structure model is SLANet_plus, not SLANeXt_wireless: the latter is
# an end-to-end option deliberately disabled by the v1 configuration.
_MODEL_SPECS = (
    _ModelSpec("layout", "PP-DocLayout_plus-L"),
    _ModelSpec("text-det", "PP-OCRv5_server_det"),
    _ModelSpec("text-rec", "PP-OCRv5_server_rec"),
    _ModelSpec("table-classifier", "PP-LCNet_x1_0_table_cls"),
    _ModelSpec("wired-structure", "SLANeXt_wired"),
    _ModelSpec("wireless-structure", "SLANet_plus"),
    _ModelSpec("wired-cells", "RT-DETR-L_wired_table_cell_det"),
    _ModelSpec("wireless-cells", "RT-DETR-L_wireless_table_cell_det"),
)


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, separators=(",", ":"), sort_keys=True
    ).encode("utf-8")


def _sha256_file(path: Path) -> tuple[str, int]:
    try:
        if path.is_symlink() or not path.is_file():
            raise RasterOfflineBundleBootstrapError("raster_bundle_bootstrap_file_invalid")
        size = path.stat().st_size
        if size > _MAX_FILE_BYTES:
            raise RasterOfflineBundleBootstrapError("raster_bundle_bootstrap_file_invalid")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest(), size
    except RasterOfflineBundleBootstrapError:
        raise
    except OSError:
        raise RasterOfflineBundleBootstrapError("raster_bundle_bootstrap_file_invalid") from None


def _is_link_or_reparse_point(metadata: os.stat_result) -> bool:
    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(metadata, "st_file_attributes", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(reparse_point and attributes & reparse_point)


def _require_regular_directory(path: Path, *, code: str) -> None:
    """Reject links, junctions, and special files before copying model bytes."""

    try:
        metadata = path.lstat()
        if _is_link_or_reparse_point(metadata) or not stat.S_ISDIR(metadata.st_mode):
            raise RasterOfflineBundleBootstrapError(code)
    except RasterOfflineBundleBootstrapError:
        raise
    except OSError:
        raise RasterOfflineBundleBootstrapError(code) from None


def _regular_file_sha256(path: Path, *, code: str) -> tuple[str, int]:
    """Hash one ordinary model file without following a local link/junction."""

    try:
        before = path.lstat()
        if (
            _is_link_or_reparse_point(before)
            or not stat.S_ISREG(before.st_mode)
            or not 0 < before.st_size <= _MAX_FILE_BYTES
        ):
            raise RasterOfflineBundleBootstrapError(code)
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        after = path.lstat()
        if (
            _is_link_or_reparse_point(after)
            or not stat.S_ISREG(after.st_mode)
            or after.st_size != before.st_size
        ):
            raise RasterOfflineBundleBootstrapError(code)
        return digest.hexdigest(), before.st_size
    except RasterOfflineBundleBootstrapError:
        raise
    except OSError:
        raise RasterOfflineBundleBootstrapError(code) from None


def _git_blob_sha1(path: Path, *, byte_size: int, code: str) -> str:
    """Calculate a Git blob digest without storing or exposing file content."""

    try:
        digest = hashlib.sha1()
        digest.update(f"blob {byte_size}\0".encode("ascii"))
        with path.open("rb") as stream:
            while block := stream.read(1024 * 1024):
                digest.update(block)
        return digest.hexdigest()
    except OSError:
        raise RasterOfflineBundleBootstrapError(code) from None


def _snapshot_pinned_local_english_recognition_directory(directory: Path, *, code: str) -> str:
    """Return a stable tree digest only for the exact fixed English model."""

    _require_regular_directory(directory, code=code)
    expected = {
        artifact.filename: artifact for artifact in _PINNED_LOCAL_ENGLISH_RECOGNITION_ARTIFACTS
    }
    artifacts: list[dict[str, object]] = []
    try:
        entries = list(directory.iterdir())
    except OSError:
        raise RasterOfflineBundleBootstrapError(code) from None
    if {entry.name for entry in entries} != set(expected):
        raise RasterOfflineBundleBootstrapError(code)
    for filename in sorted(expected):
        artifact = expected[filename]
        path = directory / filename
        sha256, byte_size = _regular_file_sha256(path, code=code)
        if byte_size != artifact.byte_size:
            raise RasterOfflineBundleBootstrapError(code)
        if artifact.sha256 is not None and sha256 != artifact.sha256:
            raise RasterOfflineBundleBootstrapError(code)
        if (
            artifact.git_blob_sha1 is not None
            and _git_blob_sha1(path, byte_size=byte_size, code=code) != artifact.git_blob_sha1
        ):
            raise RasterOfflineBundleBootstrapError(code)
        artifacts.append({"relative_path": filename, "sha256": sha256, "byte_size": byte_size})
    return hashlib.sha256(_canonical_json({"files": artifacts})).hexdigest()


def _snapshot_local_english_recognition_model(
    document_preview_root: Path,
) -> _RecognitionModelSnapshotV1:
    """Verify the only local recognition source accepted for this migration."""

    root = _document_preview_root(document_preview_root)
    _require_regular_directory(root, code="raster_bundle_local_english_recognition_unavailable")
    current = root
    try:
        for part in _LOCAL_ENGLISH_RECOGNITION_SOURCE_RELATIVE_PATH.parts:
            current = current / part
            _require_regular_directory(
                current, code="raster_bundle_local_english_recognition_unavailable"
            )
        current.resolve(strict=True).relative_to(root)
    except RasterOfflineBundleBootstrapError:
        raise
    except (OSError, ValueError):
        raise RasterOfflineBundleBootstrapError(
            "raster_bundle_local_english_recognition_unavailable"
        ) from None
    return _RecognitionModelSnapshotV1(
        source_relative_path=_LOCAL_ENGLISH_RECOGNITION_SOURCE_RELATIVE_PATH.as_posix(),
        tree_sha256=_snapshot_pinned_local_english_recognition_directory(
            current, code="raster_bundle_local_english_recognition_invalid"
        ),
    )


def _document_preview_root(path: Path) -> Path:
    try:
        if path.is_symlink():
            raise RasterOfflineBundleBootstrapError("raster_bundle_bootstrap_root_invalid")
        root = path.resolve(strict=True)
        if (
            root.is_symlink()
            or not root.is_dir()
            or tuple(part.casefold() for part in root.parts[-3:])
            != ("_local", "data", "document-preview")
        ):
            raise RasterOfflineBundleBootstrapError("raster_bundle_bootstrap_root_invalid")
        return root
    except RasterOfflineBundleBootstrapError:
        raise
    except OSError:
        raise RasterOfflineBundleBootstrapError("raster_bundle_bootstrap_root_invalid") from None


def _windows_operational_path(path: Path, *, force_short: bool = False) -> Path:
    """Return the existing short path used for deep Windows package trees."""

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
        raise RasterOfflineBundleBootstrapError("raster_bundle_windows_path_unsupported") from None
    if not short_value or not short_value.isascii():
        raise RasterOfflineBundleBootstrapError("raster_bundle_windows_path_unsupported")
    return Path(short_value)


def _windows_extended_write_path(path: Path) -> Path:
    """Return an absolute Win32 long-path spelling for a new deep target.

    A content-addressed release stage does not exist yet, so it has no DOS
    short name to resolve.  Its deepest copied package files can therefore
    exceed Win32's legacy path limit even though the source and the eventual
    published release both have short aliases.  The extended prefix is used
    only for the initial filesystem write; all later security checks and
    native execution continue to use canonical/short paths.
    """

    if os.name != "nt":
        return path
    value = str(path)
    if value.startswith("\\\\?\\"):
        return path
    if not path.is_absolute():
        raise RasterOfflineBundleBootstrapError("raster_bundle_windows_path_unsupported")
    if value.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + value.lstrip("\\"))
    return Path("\\\\?\\" + value)


def _resolved_comparison_path(path: Path) -> Path:
    """Resolve one existing path to a presentation-independent identity.

    ``shutil.copytree`` may pass its ignore callback an extended Windows path
    (``\\\\?\\...``) even when the verified source was opened through its DOS
    short spelling. ``Path.resolve`` deliberately preserves that extended
    prefix, so the two otherwise identical paths cannot be compared with
    ``relative_to`` until both use the canonical extended spelling. This
    helper runs only after strict resolution; it changes no containment or
    symlink decision.
    """

    return _windows_extended_write_path(path.resolve(strict=True))


def _project_root() -> Path:
    # src/customs_ai_agent/evaluation/<this file> -> repository root
    root = Path(__file__).resolve().parents[3]
    if not (root / "pyproject.toml").is_file():
        raise RasterOfflineBundleBootstrapError("raster_bundle_bootstrap_source_invalid")
    return root


def _ensure_supported_platform() -> None:
    if os.name != "nt" or sys.version_info[:2] != (3, 11) or sys.maxsize <= 2**32:
        raise RasterOfflineBundleBootstrapError("raster_bundle_bootstrap_platform_unsupported")


def _required_parent_inventory() -> dict[str, object]:
    packages: list[dict[str, str]] = []
    try:
        for name in _PARENT_PACKAGES:
            packages.append({"name": name, "version": importlib.metadata.version(name)})
    except importlib.metadata.PackageNotFoundError:
        raise RasterOfflineBundleBootstrapError(
            "raster_bundle_parent_renderer_unavailable"
        ) from None
    return {"packages": packages}


@contextmanager
def _bootstrap_lock(root: Path) -> Iterator[None]:
    """Serialize publishing with an OS-owned Windows lock, not a stale marker.

    The prior create-only file lock remained forever if a host interrupted the
    bootstrap process.  That made an otherwise non-human one-command flow
    require somebody to delete a random lock file.  A one-byte advisory lock
    is held by the process handle instead: a living concurrent bootstrap is
    still rejected, while Windows releases the lock automatically if its owner
    dies.  The file is intentionally retained as an inert lock target.
    """

    lock = root / _BOOTSTRAP_LOCK_NAME
    stream = None
    try:
        if lock.is_symlink():
            raise RasterOfflineBundleBootstrapError("raster_bundle_bootstrap_lock_invalid")
        import msvcrt

        stream = lock.open("a+b")
        if lock.is_symlink():
            raise RasterOfflineBundleBootstrapError("raster_bundle_bootstrap_lock_invalid")
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"\0")
            stream.flush()
            os.fsync(stream.fileno())
        stream.seek(0)
        msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
    except (ImportError, OSError):
        if stream is not None:
            stream.close()
        raise RasterOfflineBundleBootstrapError("raster_bundle_bootstrap_locked") from None
    try:
        yield
    finally:
        try:
            if stream is not None:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        except (OSError, UnboundLocalError):
            pass
        finally:
            if stream is not None:
                stream.close()


def _remove_owned_tree(root: Path, path: Path, *, prefix: str) -> None:
    """Remove only an explicitly-created private sibling after validation."""

    try:
        canonical_path = path.resolve(strict=True)
        if (
            canonical_path.parent != root
            or not canonical_path.name.startswith(prefix)
            or canonical_path.is_symlink()
        ):
            return
        if path.exists():
            shutil.rmtree(_windows_operational_path(canonical_path, force_short=True))
    except OSError:
        # A leftover private staging path is safer than broad cleanup.
        pass


def _copy_interpreter(*, destination: Path) -> None:
    source = Path(sys.base_prefix)
    try:
        if source.is_symlink() or not (source / "python.exe").is_file():
            raise RasterOfflineBundleBootstrapError("raster_bundle_interpreter_unavailable")
        shutil.copytree(source, destination, symlinks=False, copy_function=shutil.copy2)
    except RasterOfflineBundleBootstrapError:
        raise
    except OSError:
        raise RasterOfflineBundleBootstrapError("raster_bundle_interpreter_unavailable") from None
    _assert_regular_tree(destination)


def _assert_regular_tree(root: Path) -> None:
    """Reject every link/special file before it can enter a hashed bundle."""

    try:
        if root.is_symlink() or not root.is_dir():
            raise RasterOfflineBundleBootstrapError("raster_bundle_bootstrap_tree_invalid")
        for path in root.rglob("*"):
            if path.is_symlink() or not (path.is_dir() or path.is_file()):
                raise RasterOfflineBundleBootstrapError("raster_bundle_bootstrap_tree_invalid")
    except RasterOfflineBundleBootstrapError:
        raise
    except OSError:
        raise RasterOfflineBundleBootstrapError("raster_bundle_bootstrap_tree_invalid") from None


def _install_sidecar_packages(*, interpreter: Path, module_root: Path) -> None:
    requirements = _project_root() / _SIDECAR_REQUIREMENTS_RELATIVE_PATH
    uv = shutil.which("uv")
    if not requirements.is_file() or uv is None:
        raise RasterOfflineBundleBootstrapError("raster_bundle_sidecar_provisioner_unavailable")
    command = [
        uv,
        "pip",
        "install",
        "--python",
        str(interpreter),
        "--target",
        str(module_root),
        "--require-hashes",
        "--only-binary",
        ":all:",
        "--index-url",
        "https://pypi.org/simple",
        "--no-progress",
        "-r",
        str(requirements),
    ]
    environment = os.environ.copy()
    environment.update({"UV_NO_CONFIG": "1", "UV_NO_PROGRESS": "1"})
    try:
        completed = subprocess.run(
            command,
            cwd=str(_project_root()),
            env=environment,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            shell=False,
            timeout=1_800,
        )
    except (OSError, subprocess.TimeoutExpired):
        raise RasterOfflineBundleBootstrapError(
            "raster_bundle_sidecar_packages_unavailable"
        ) from None
    if completed.returncode != 0:
        raise RasterOfflineBundleBootstrapError("raster_bundle_sidecar_packages_unavailable")
    _assert_regular_tree(module_root)


def _copy_first_party_sidecar(*, module_root: Path) -> Path:
    source_root = _project_root() / "src" / "customs_ai_agent"
    target_root = module_root / "customs_ai_agent"
    try:
        (target_root / "service").mkdir(parents=True, exist_ok=False)
        (target_root / "schemas").mkdir(parents=True, exist_ok=False)
        shutil.copy2(source_root / "__init__.py", target_root / "__init__.py")
        shutil.copy2(
            source_root / "service" / "__init__.py", target_root / "service" / "__init__.py"
        )
        sidecar = target_root / "service" / "ppstructurev3_sidecar.py"
        shutil.copy2(source_root / "service" / "ppstructurev3_sidecar.py", sidecar)
        # The direct text profile has a separate protocol/module but shares
        # the same isolated interpreter and exhaustive release manifest.
        shutil.copy2(
            source_root / "service" / "raster_text_sidecar.py",
            target_root / "service" / "raster_text_sidecar.py",
        )
        for source in (source_root / "schemas").glob("*.py"):
            shutil.copy2(source, target_root / "schemas" / source.name)
    except OSError:
        raise RasterOfflineBundleBootstrapError("raster_bundle_sidecar_source_invalid") from None
    _assert_regular_tree(target_root)
    return sidecar


def _sidecar_package_inventory(module_root: Path) -> dict[str, object]:
    observed: dict[str, str] = {}
    try:
        for distribution in importlib.metadata.distributions(path=[str(module_root)]):
            name = distribution.metadata["Name"] if "Name" in distribution.metadata else None
            version = distribution.version
            if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
                raise RasterOfflineBundleBootstrapError("raster_bundle_sidecar_inventory_invalid")
            if name in observed:
                raise RasterOfflineBundleBootstrapError("raster_bundle_sidecar_inventory_invalid")
            observed[name] = version
    except RasterOfflineBundleBootstrapError:
        raise
    except Exception:
        raise RasterOfflineBundleBootstrapError("raster_bundle_sidecar_inventory_invalid") from None
    if not set(_SIDECAR_REQUIRED_PACKAGES) <= set(observed):
        raise RasterOfflineBundleBootstrapError("raster_bundle_sidecar_inventory_invalid")
    return {
        "packages": [
            {"name": name, "version": observed[name]}
            for name in sorted(observed, key=lambda item: item.casefold())
        ]
    }


def _download_model_archive(*, specification: _ModelSpec, destination: Path) -> tuple[str, int]:
    request = Request(
        specification.source_url,
        headers={"User-Agent": "customs-raster-offline-bundle-bootstrap/1"},
    )
    digest = hashlib.sha256()
    total = 0
    try:
        with urlopen(request, timeout=120) as response, destination.open("xb") as stream:  # noqa: S310
            if not response.geturl().startswith(f"{_MODEL_BASE_URL}/"):
                raise RasterOfflineBundleBootstrapError("raster_bundle_model_source_invalid")
            length = response.headers.get("Content-Length")
            if length is not None and (not length.isdecimal() or int(length) > _MAX_ARCHIVE_BYTES):
                raise RasterOfflineBundleBootstrapError("raster_bundle_model_source_invalid")
            while block := response.read(1024 * 1024):
                total += len(block)
                if total > _MAX_ARCHIVE_BYTES:
                    raise RasterOfflineBundleBootstrapError("raster_bundle_model_source_invalid")
                digest.update(block)
                stream.write(block)
            stream.flush()
            os.fsync(stream.fileno())
    except RasterOfflineBundleBootstrapError:
        raise
    except OSError:
        raise RasterOfflineBundleBootstrapError(
            "raster_bundle_model_download_unavailable"
        ) from None
    if total < 1:
        raise RasterOfflineBundleBootstrapError("raster_bundle_model_source_invalid")
    return digest.hexdigest(), total


def _safe_extract_model_archive(*, archive_path: Path, destination: Path) -> None:
    """Extract only regular tar members below exactly one archive root."""

    try:
        with tarfile.open(archive_path, mode="r:*") as archive:
            members = archive.getmembers()
            roots: set[str] = set()
            for member in members:
                path = PurePosixPath(member.name)
                if (
                    not member.name
                    or path.is_absolute()
                    or ".." in path.parts
                    or "." in path.parts
                    or not (member.isdir() or member.isfile())
                    or not path.parts
                ):
                    raise RasterOfflineBundleBootstrapError("raster_bundle_model_archive_invalid")
                roots.add(path.parts[0])
            if len(roots) != 1 or not any(member.isfile() for member in members):
                raise RasterOfflineBundleBootstrapError("raster_bundle_model_archive_invalid")
            destination.mkdir(parents=True, exist_ok=False)
            for member in members:
                if member.isdir():
                    continue
                parts = PurePosixPath(member.name).parts[1:]
                if not parts:
                    raise RasterOfflineBundleBootstrapError("raster_bundle_model_archive_invalid")
                output = destination.joinpath(*parts)
                output.parent.mkdir(parents=True, exist_ok=True)
                source = archive.extractfile(member)
                if source is None:
                    raise RasterOfflineBundleBootstrapError("raster_bundle_model_archive_invalid")
                with source, output.open("xb") as stream:
                    shutil.copyfileobj(source, stream, length=1024 * 1024)
    except RasterOfflineBundleBootstrapError:
        raise
    except (OSError, tarfile.TarError):
        raise RasterOfflineBundleBootstrapError("raster_bundle_model_archive_invalid") from None
    _assert_regular_tree(destination)


def _download_models(*, models_root: Path, workspace: Path) -> list[dict[str, object]]:
    receipts: list[dict[str, object]] = []
    models_root.mkdir(parents=True, exist_ok=False)
    for specification in _MODEL_SPECS:
        archive = workspace / specification.archive_name
        archive_sha256, archive_size = _download_model_archive(
            specification=specification, destination=archive
        )
        _safe_extract_model_archive(
            archive_path=archive, destination=models_root / specification.target_directory
        )
        receipts.append(
            {
                "archive_sha256": archive_sha256,
                "archive_byte_size": archive_size,
                "model_name": specification.model_name,
                "source_url": specification.source_url,
                "target_directory": specification.target_directory,
            }
        )
    return receipts


def _model_inventory(models_root: Path) -> dict[str, object]:
    artifacts: list[dict[str, str]] = []
    try:
        for path in sorted(models_root.rglob("*"), key=lambda value: value.as_posix()):
            if path.is_symlink() or not path.is_file():
                continue
            relative = path.relative_to(models_root).as_posix()
            digest, _size = _sha256_file(path)
            artifacts.append({"relative_path": relative, "sha256": digest})
    except RasterOfflineBundleBootstrapError:
        raise
    except OSError:
        raise RasterOfflineBundleBootstrapError("raster_bundle_model_inventory_invalid") from None
    if not artifacts:
        raise RasterOfflineBundleBootstrapError("raster_bundle_model_inventory_invalid")
    return {"artifacts": artifacts}


def _raster_text_direct_tree_artifacts(
    root: Path, *, code: str, dedicated_root: bool = False
) -> tuple[_RasterTextDirectModelArtifactV1, ...]:
    """Read only the fixed detector/recognizer files, rejecting every extra.

    The parent structural root can legitimately contain six other models.  A
    direct-text source/target is intentionally narrower: it is exactly the
    seven files below ``text-det`` and ``text-rec`` and no links, caches, or
    opportunistic model additions.
    """

    try:
        _require_regular_directory(root, code=code)
        actual: dict[str, _RasterTextDirectModelArtifactV1] = {}
        for directory_name in ("text-det", "text-rec"):
            _require_regular_directory(root / directory_name, code=code)
        trees = (root,) if dedicated_root else (root / "text-det", root / "text-rec")
        for tree in trees:
            for path in tree.rglob("*"):
                if path.is_symlink():
                    raise RasterOfflineBundleBootstrapError(code)
                if path.is_dir():
                    continue
                if not path.is_file():
                    raise RasterOfflineBundleBootstrapError(code)
                relative_path = path.relative_to(root).as_posix()
                if relative_path in actual:
                    raise RasterOfflineBundleBootstrapError(code)
                sha256, byte_size = _regular_file_sha256(path, code=code)
                actual[relative_path] = _RasterTextDirectModelArtifactV1(
                    relative_path=relative_path,
                    sha256=sha256,
                    byte_size=byte_size,
                )
        if tuple(sorted(actual)) != _RASTER_TEXT_DIRECT_MODEL_PATHS:
            raise RasterOfflineBundleBootstrapError(code)
        return tuple(actual[path] for path in _RASTER_TEXT_DIRECT_MODEL_PATHS)
    except RasterOfflineBundleBootstrapError:
        raise
    except (OSError, ValueError):
        raise RasterOfflineBundleBootstrapError(code) from None


def _raster_text_direct_snapshot_payload(
    snapshot: _RasterTextDirectModelSnapshotV1,
) -> dict[str, object]:
    """Canonical dedicated inventory; its SHA is bound into the text profile."""

    return {
        "schema_version": "1.0",
        "inventory_kind": "raster-text-direct-models-v1",
        "source_bundle_manifest_sha256": snapshot.source_bundle_manifest_sha256,
        "source_model_root_relative_path": snapshot.source_model_root_relative_path,
        "source_model_inventory_sha256": snapshot.source_model_inventory_sha256,
        "source_snapshot_sha256": snapshot.source_snapshot_sha256,
        "artifacts": [
            {
                "relative_path": artifact.relative_path,
                "sha256": artifact.sha256,
                "byte_size": artifact.byte_size,
            }
            for artifact in snapshot.artifacts
        ],
    }


def _snapshot_raster_text_direct_model_source(
    bundle: VerifiedRasterOfflineBundleV1,
) -> _RasterTextDirectModelSnapshotV1:
    """Attest the seven source files before they are copied to a new release."""

    try:
        manifest = bundle.manifest
        source_root_relative_path = manifest.local_model_root_relative_path
        source_inventory_relative_path = manifest.model_inventory_relative_path
        source_root = bundle.bundle_root.joinpath(*PurePosixPath(source_root_relative_path).parts)
        artifacts = _raster_text_direct_tree_artifacts(
            source_root, code="raster_bundle_raster_text_model_source_invalid"
        )
        inventory = _read_canonical_object(
            bundle.bundle_root.joinpath(*PurePosixPath(source_inventory_relative_path).parts),
            code="raster_bundle_raster_text_model_source_invalid",
        )
        rows = inventory.get("artifacts")
        if not isinstance(rows, list):
            raise RasterOfflineBundleBootstrapError(
                "raster_bundle_raster_text_model_source_invalid"
            )
        inventory_sha_by_path: dict[str, str] = {}
        for row in rows:
            relative_path = row.get("relative_path") if isinstance(row, dict) else None
            sha256 = row.get("sha256") if isinstance(row, dict) else None
            if (
                not isinstance(relative_path, str)
                or not isinstance(sha256, str)
                or _RELEASE_MANIFEST_SHA256.fullmatch(sha256) is None
                or relative_path in inventory_sha_by_path
            ):
                raise RasterOfflineBundleBootstrapError(
                    "raster_bundle_raster_text_model_source_invalid"
                )
            inventory_sha_by_path[relative_path] = sha256
        manifest_file_sizes = {item.path: item.byte_size for item in manifest.files}
        for artifact in artifacts:
            bundle_relative_path = f"{source_root_relative_path}/{artifact.relative_path}"
            if (
                inventory_sha_by_path.get(artifact.relative_path) != artifact.sha256
                or bundle.file_sha256(bundle_relative_path) != artifact.sha256
                or manifest_file_sizes.get(bundle_relative_path) != artifact.byte_size
            ):
                raise RasterOfflineBundleBootstrapError(
                    "raster_bundle_raster_text_model_source_invalid"
                )
        source_snapshot_sha256 = hashlib.sha256(
            _canonical_json(
                {
                    "files": [
                        {
                            "relative_path": artifact.relative_path,
                            "sha256": artifact.sha256,
                            "byte_size": artifact.byte_size,
                        }
                        for artifact in artifacts
                    ]
                }
            )
        ).hexdigest()
        return _RasterTextDirectModelSnapshotV1(
            source_bundle_manifest_sha256=bundle.manifest_sha256,
            source_model_root_relative_path=source_root_relative_path,
            source_model_inventory_sha256=bundle.file_sha256(source_inventory_relative_path),
            source_snapshot_sha256=source_snapshot_sha256,
            artifacts=artifacts,
        )
    except RasterOfflineBundleBootstrapError:
        raise
    except (OSError, TypeError, ValueError):
        raise RasterOfflineBundleBootstrapError(
            "raster_bundle_raster_text_model_source_invalid"
        ) from None


def _assert_raster_text_direct_model_source_stable(
    bundle: VerifiedRasterOfflineBundleV1,
    *,
    expected: _RasterTextDirectModelSnapshotV1,
) -> None:
    if _snapshot_raster_text_direct_model_source(bundle) != expected:
        raise RasterOfflineBundleBootstrapError("raster_bundle_raster_text_model_source_drift")


def _copy_raster_text_direct_models_to_stage(
    *,
    base: VerifiedRasterOfflineBundleV1,
    stage: Path,
    expected_snapshot: _RasterTextDirectModelSnapshotV1,
) -> None:
    """Copy the attested seven files into a separate direct-text-only root."""

    _assert_raster_text_direct_model_source_stable(base, expected=expected_snapshot)
    try:
        source_root = base.bundle_root.joinpath(
            *PurePosixPath(expected_snapshot.source_model_root_relative_path).parts
        )
        target_root = stage.joinpath(*_RASTER_TEXT_MODEL_ROOT_RELATIVE_PATH.parts)
        if target_root.exists() or target_root.is_symlink() or target_root.parent.is_symlink():
            raise RasterOfflineBundleBootstrapError("raster_bundle_raster_text_model_copy_invalid")
        target_root.mkdir(parents=False, exist_ok=False)
        for directory_name in ("text-det", "text-rec"):
            shutil.copytree(
                _windows_operational_path(source_root / directory_name, force_short=True),
                _windows_extended_write_path(target_root / directory_name),
                symlinks=False,
                copy_function=shutil.copy2,
            )
        copied = _raster_text_direct_tree_artifacts(
            target_root,
            code="raster_bundle_raster_text_model_copy_invalid",
            dedicated_root=True,
        )
        if copied != expected_snapshot.artifacts:
            raise RasterOfflineBundleBootstrapError("raster_bundle_raster_text_model_copy_invalid")
        # The structural smoke still receives ``models``.  Its exhaustive
        # structural inventory must therefore attest these copied bytes too,
        # even though the direct runtime never reads that larger inventory.
        structural_inventory_path = stage / "inventories" / "models.json"
        _replace_owned_release_file(stage=stage, target=structural_inventory_path)
        _write_canonical(structural_inventory_path, _model_inventory(stage / "models"))
        inventory_path = stage.joinpath(*_RASTER_TEXT_MODEL_INVENTORY_RELATIVE_PATH.parts)
        if inventory_path.exists() or inventory_path.is_symlink():
            raise RasterOfflineBundleBootstrapError("raster_bundle_raster_text_model_copy_invalid")
        _write_canonical(inventory_path, _raster_text_direct_snapshot_payload(expected_snapshot))
        _assert_raster_text_direct_model_source_stable(base, expected=expected_snapshot)
    except RasterOfflineBundleBootstrapError:
        raise
    except OSError:
        raise RasterOfflineBundleBootstrapError(
            "raster_bundle_raster_text_model_copy_failed"
        ) from None


def _raster_text_direct_model_material_matches_source(
    bundle: VerifiedRasterOfflineBundleV1,
    *,
    expected_snapshot: _RasterTextDirectModelSnapshotV1,
) -> bool:
    """Return whether a formally valid direct root matches this exact source."""

    try:
        manifest = bundle.manifest
        if (
            manifest.raster_text_model_inventory_relative_path
            != _RASTER_TEXT_MODEL_INVENTORY_RELATIVE_PATH.as_posix()
            or manifest.raster_text_local_model_root_relative_path
            != _RASTER_TEXT_MODEL_ROOT_RELATIVE_PATH.as_posix()
        ):
            raise RasterOfflineBundleBootstrapError(
                "raster_bundle_raster_text_model_material_invalid"
            )
        verify_raster_text_direct_model_material(bundle)
        inventory = _read_canonical_object(
            bundle.bundle_root.joinpath(
                *PurePosixPath(manifest.raster_text_model_inventory_relative_path).parts
            ),
            code="raster_bundle_raster_text_model_material_invalid",
        )
        return inventory == _raster_text_direct_snapshot_payload(expected_snapshot)
    except RasterOfflineBundleBootstrapError:
        raise
    except RasterOfflineBundleError as error:
        raise RasterOfflineBundleBootstrapError(
            "raster_bundle_raster_text_model_material_invalid"
        ) from error
    except (OSError, ValueError):
        raise RasterOfflineBundleBootstrapError(
            "raster_bundle_raster_text_model_material_invalid"
        ) from None


def _verify_raster_text_direct_model_material_matches_source(
    bundle: VerifiedRasterOfflineBundleV1,
    *,
    expected_snapshot: _RasterTextDirectModelSnapshotV1,
) -> None:
    """Require a reused release's dedicated root to match this exact source."""

    if not _raster_text_direct_model_material_matches_source(
        bundle, expected_snapshot=expected_snapshot
    ):
        raise RasterOfflineBundleBootstrapError("raster_bundle_raster_text_model_material_invalid")


def _configuration_template() -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "runtime_api": "paddleocr3-ppstructurev3",
        "extractor_revision": _EXTRACTOR_REVISION,
        "raster_profile_revision": "synthetic-raster-v1",
        "layout_detection_model_name": "PP-DocLayout_plus-L",
        "layout_detection_model_directory": "layout",
        "text_detection_model_name": "PP-OCRv5_server_det",
        "text_detection_model_directory": "text-det",
        # The release builder admits this model only by copying the fixed
        # local English snapshot into the release itself.  The runtime never
        # points at the separate PaddleOCR 3 baseline directory.
        "text_recognition_model_name": _LOCAL_ENGLISH_RECOGNITION_MODEL_NAME,
        "text_recognition_model_directory": "text-rec",
        "table_classification_model_name": "PP-LCNet_x1_0_table_cls",
        "table_classification_model_directory": "table-classifier",
        "wired_table_structure_recognition_model_name": "SLANeXt_wired",
        "wired_table_structure_recognition_model_directory": "wired-structure",
        "wireless_table_structure_recognition_model_name": "SLANet_plus",
        "wireless_table_structure_recognition_model_directory": "wireless-structure",
        "wired_table_cells_detection_model_name": "RT-DETR-L_wired_table_cell_det",
        "wired_table_cells_detection_model_directory": "wired-cells",
        "wireless_table_cells_detection_model_name": "RT-DETR-L_wireless_table_cell_det",
        "wireless_table_cells_detection_model_directory": "wireless-cells",
        "device": "cpu",
        # The target local machine has six physical CPU cores. This is a
        # fixed, hash-bound profile value rather than a host/environment
        # default, so a changed thread count necessarily creates a new
        # release and must pass calibration and sealed Gates again.
        "cpu_threads": 6,
        # A 2048 px multi-document native list batch does not complete within
        # the same 420-second sidecar bound used by the service. Keep one
        # model pipeline alive for evaluation, but submit one document at a
        # time just like serving. This fixed choice is release material and
        # therefore requires fresh calibration and sealed evidence.
        "evaluation_batch_size": 1,
        "inference_engine": "paddle_static",
        "paddle_static_run_mode": "paddle",
        "paddle_static_enable_new_ir": False,
        "paddle_static_enable_cinn": False,
        # The first production candidate capped a 216-DPI page's detector
        # input at 960 px.  That profile remains reproducible, but it loses
        # too much small-label detail for the source-attested raster corpus.
        # The 1536 px release loses small-label fidelity on the
        # source-attested corpus. The full-216-DPI 2526 candidate is retained
        # as failed immutable local evidence because it could not produce a
        # valid observation within this runtime boundary. New derived releases
        # return to the still-bounded 2048 px profile; it still must satisfy
        # every sealed Gate before any activation is possible.
        "text_detection_limit_side_len": 2048,
        "text_detection_limit_type": "max",
        "text_detection_threshold": 0.3,
        "text_detection_box_threshold": 0.6,
        "text_detection_unclip_ratio": 1.5,
        "enable_hpi": False,
        "enable_mkldnn": False,
        "use_tensorrt": False,
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
        "use_seal_recognition": False,
        "use_formula_recognition": False,
        "use_chart_recognition": False,
        "use_region_detection": False,
        "use_table_recognition": True,
        "use_table_orientation_classify": False,
        "use_ocr_results_with_table_cells": True,
        "use_e2e_wired_table_rec_model": False,
        "use_e2e_wireless_table_rec_model": False,
    }


def _write_canonical(path: Path, value: object) -> bytes:
    payload = _canonical_json(value)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError:
        raise RasterOfflineBundleBootstrapError("raster_bundle_bootstrap_output_conflict") from None
    except OSError:
        raise RasterOfflineBundleBootstrapError("raster_bundle_bootstrap_output_invalid") from None
    return payload


def _sidecar_smoke(
    *,
    stage: Path,
    workspace: Path,
    interpreter: Path,
    module_root: Path,
    sidecar_path: Path,
    package_inventory_path: Path,
    model_inventory_path: Path,
    models_root: Path,
) -> None:
    """Prove the copied, isolated native runtime before publish.

    The probe is deliberately a blank RGB page.  It establishes package
    import, local eight-model initialization, stdin protocol, and no-network
    execution without treating OCR contents as an evaluation result.
    """

    from customs_ai_agent.service.raster_document_preview_runtime import (
        _ISOLATED_SIDECAR_BOOTSTRAP,
        _isolated_v2_sidecar_home,
        _isolated_v2_sidecar_environment,
        RasterRuntimeError,
    )

    configuration_path = workspace / "sidecar-smoke.json"
    _write_canonical(configuration_path, _configuration_template())
    pixels = b"\xff" * (256 * 256 * 3)
    source_sha256 = hashlib.sha256(pixels).hexdigest()
    header = {
        "protocol": "customs-raster-ppstructurev3-v1",
        "source_sha256": source_sha256,
        "raster_profile_revision": "synthetic-raster-v1",
        "pages": [
            {
                "page_number": 1,
                "width": 256,
                "height": 256,
                "pixel_length": len(pixels),
                "pixel_sha256": source_sha256,
            }
        ],
    }
    payload = _canonical_json(header) + b"\n" + pixels
    sidecar_sha256, _size = _sha256_file(sidecar_path)
    package_sha256, _size = _sha256_file(package_inventory_path)
    model_sha256, _size = _sha256_file(model_inventory_path)
    try:
        with _isolated_v2_sidecar_home() as runtime_home:
            environment = _isolated_v2_sidecar_environment(
                interpreter_path=interpreter,
                module_root=module_root,
                runtime_home=runtime_home,
                configuration_path=configuration_path,
                configuration_sha256=_sha256_file(configuration_path)[0],
                model_root=models_root,
                package_inventory_path=package_inventory_path,
                package_inventory_sha256=package_sha256,
                model_inventory_path=model_inventory_path,
                model_inventory_sha256=model_sha256,
                required_packages=_SIDECAR_REQUIRED_PACKAGES,
                sidecar_module_sha256=sidecar_sha256,
            )
            completed = subprocess.run(
                [
                    str(interpreter),
                    "-I",
                    "-B",
                    "-c",
                    _ISOLATED_SIDECAR_BOOTSTRAP,
                    str(stage / "sidecar" / "interpreter"),
                    str(module_root),
                    str(sidecar_path),
                    "--customs-raster-ppstructurev3-v1",
                ],
                input=payload,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                cwd=str(module_root),
                env=environment,
                shell=False,
                timeout=420,
            )
    except (OSError, subprocess.TimeoutExpired, RasterRuntimeError):
        raise RasterOfflineBundleBootstrapError("raster_bundle_sidecar_smoke_failed") from None
    if completed.returncode != 0 or not completed.stdout or len(completed.stdout) > 1024 * 1024:
        raise RasterOfflineBundleBootstrapError("raster_bundle_sidecar_smoke_failed")
    try:
        response = json.loads(completed.stdout)
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RasterOfflineBundleBootstrapError("raster_bundle_sidecar_smoke_failed") from None
    if (
        not isinstance(response, dict)
        or response.get("source_sha256") != source_sha256
        or response.get("raster_profile_revision") != "synthetic-raster-v1"
    ):
        raise RasterOfflineBundleBootstrapError("raster_bundle_sidecar_smoke_failed")


def _manifest_files(stage: Path) -> list[dict[str, object]]:
    files: list[dict[str, object]] = []
    try:
        for path in sorted(stage.rglob("*"), key=lambda value: value.as_posix()):
            if path.is_symlink():
                raise RasterOfflineBundleBootstrapError("raster_bundle_bootstrap_tree_invalid")
            if not path.is_file():
                continue
            relative = path.relative_to(stage).as_posix()
            if relative == "manifest.json" or len(relative) > 480:
                raise RasterOfflineBundleBootstrapError("raster_bundle_bootstrap_tree_invalid")
            digest, size = _sha256_file(path)
            files.append({"path": relative, "sha256": digest, "byte_size": size})
    except RasterOfflineBundleBootstrapError:
        raise
    except OSError:
        raise RasterOfflineBundleBootstrapError("raster_bundle_bootstrap_tree_invalid") from None
    if not files:
        raise RasterOfflineBundleBootstrapError("raster_bundle_bootstrap_tree_invalid")
    return files


def _write_manifest(
    *,
    stage: Path,
    raster_text_models: bool = False,
) -> None:
    manifest = {
        "schema_version": "1.0",
        "bundle_id": _BUNDLE_DIRECTORY,
        "extractor_revision": _EXTRACTOR_REVISION,
        "raster_dpi": 216,
        "sidecar_timeout_seconds": 420,
        "parent_renderer_package_inventory_relative_path": "inventories/parent-renderer.json",
        "parent_renderer_required_packages": list(_PARENT_PACKAGES),
        "sidecar_package_inventory_relative_path": "inventories/sidecar-packages.json",
        "sidecar_required_packages": list(_SIDECAR_REQUIRED_PACKAGES),
        "sidecar_command_relative_path": (
            "sidecar/module-root/customs_ai_agent/service/ppstructurev3_sidecar.py"
        ),
        "sidecar_interpreter_root_relative_path": "sidecar/interpreter",
        "sidecar_interpreter_relative_path": "sidecar/interpreter/python.exe",
        "sidecar_module_root_relative_path": "sidecar/module-root",
        "sidecar_module_relative_path": (
            "sidecar/module-root/customs_ai_agent/service/ppstructurev3_sidecar.py"
        ),
        "sidecar_configuration_template_relative_path": "sidecar/configuration-template.json",
        "model_inventory_relative_path": "inventories/models.json",
        "local_model_root_relative_path": "models",
        "files": _manifest_files(stage),
    }
    if raster_text_models:
        manifest["raster_text_model_inventory_relative_path"] = (
            _RASTER_TEXT_MODEL_INVENTORY_RELATIVE_PATH.as_posix()
        )
        manifest["raster_text_local_model_root_relative_path"] = (
            _RASTER_TEXT_MODEL_ROOT_RELATIVE_PATH.as_posix()
        )
    _write_canonical(stage / "manifest.json", manifest)


def _write_bootstrap_material(
    *,
    stage: Path,
    model_receipts: list[dict[str, object]],
    package_inventory: Mapping[str, object],
) -> None:
    _write_canonical(stage / "inventories" / "parent-renderer.json", _required_parent_inventory())
    _write_canonical(stage / "inventories" / "sidecar-packages.json", package_inventory)
    _write_canonical(stage / "inventories" / "models.json", _model_inventory(stage / "models"))
    _write_canonical(stage / "sidecar" / "configuration-template.json", _configuration_template())
    _write_canonical(
        stage / "provenance" / "bootstrap-receipt.json",
        {
            "schema_version": "1.0",
            "bootstrap_kind": "official-online-raster-offline-bundle-v1",
            "python_version": ".".join(str(item) for item in sys.version_info[:3]),
            "sidecar_lock_sha256": _sha256_file(
                _project_root() / _SIDECAR_REQUIREMENTS_RELATIVE_PATH
            )[0],
            "sidecar_required_packages": list(_SIDECAR_REQUIRED_PACKAGES),
            "models": model_receipts,
        },
    )


def _read_canonical_object(path: Path, *, code: str) -> dict[str, object]:
    """Read one small canonical local JSON object without leaking its bytes."""

    try:
        if (
            path.is_symlink()
            or not path.is_file()
            or not 0 < path.stat().st_size <= _MAX_FILE_BYTES
        ):
            raise RasterOfflineBundleBootstrapError(code)
        raw = path.read_bytes()
        value = json.loads(raw)
        if not isinstance(value, dict) or raw != _canonical_json(value):
            raise RasterOfflineBundleBootstrapError(code)
        return value
    except RasterOfflineBundleBootstrapError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError):
        raise RasterOfflineBundleBootstrapError(code) from None


def _desired_first_party_sidecar_material() -> dict[str, bytes]:
    """Return the exact first-party subtree copied into every sidecar release."""

    source_root = _project_root() / "src" / "customs_ai_agent"
    candidates = [
        source_root / "__init__.py",
        source_root / "service" / "__init__.py",
        source_root / "service" / "ppstructurev3_sidecar.py",
        source_root / "service" / "raster_text_sidecar.py",
        *sorted((source_root / "schemas").glob("*.py"), key=lambda path: path.name),
    ]
    material: dict[str, bytes] = {}
    try:
        for source in candidates:
            if source.is_symlink() or not source.is_file():
                raise RasterOfflineBundleBootstrapError("raster_bundle_sidecar_source_invalid")
            relative = source.relative_to(source_root).as_posix()
            if relative in material:
                raise RasterOfflineBundleBootstrapError("raster_bundle_sidecar_source_invalid")
            material[relative] = source.read_bytes()
    except RasterOfflineBundleBootstrapError:
        raise
    except (OSError, ValueError):
        raise RasterOfflineBundleBootstrapError("raster_bundle_sidecar_source_invalid") from None
    if not material:
        raise RasterOfflineBundleBootstrapError("raster_bundle_sidecar_source_invalid")
    return material


def _bundle_first_party_sidecar_material(bundle: VerifiedRasterOfflineBundleV1) -> dict[str, bytes]:
    """Read the whole bundled first-party import subtree after manifest verification."""

    prefix = f"{bundle.manifest.sidecar_module_root_relative_path}/customs_ai_agent/"
    material: dict[str, bytes] = {}
    try:
        for declared_path in bundle.file_sha256_by_relative_path:
            if not declared_path.startswith(prefix):
                continue
            relative = declared_path[len(prefix) :]
            path = bundle.bundle_root.joinpath(*PurePosixPath(declared_path).parts)
            if path.is_symlink() or not path.is_file():
                raise RasterOfflineBundleBootstrapError("raster_bundle_release_base_invalid")
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != bundle.file_sha256(declared_path):
                raise RasterOfflineBundleBootstrapError("raster_bundle_release_base_invalid")
            material[relative] = raw
    except RasterOfflineBundleBootstrapError:
        raise
    except OSError:
        raise RasterOfflineBundleBootstrapError("raster_bundle_release_base_invalid") from None
    if not material:
        raise RasterOfflineBundleBootstrapError("raster_bundle_release_base_invalid")
    return material


def _release_requires_derivation(bundle: VerifiedRasterOfflineBundleV1) -> bool:
    """Compare the trusted release to current first-party source/config bytes."""

    try:
        current_configuration = _canonical_json(_configuration_template())
        bundled_configuration_path = bundle.bundle_root.joinpath(
            *PurePosixPath(bundle.manifest.sidecar_configuration_template_relative_path).parts
        )
        if (
            bundled_configuration_path.is_symlink()
            or bundled_configuration_path.read_bytes() != current_configuration
        ):
            return True
        return (
            _bundle_first_party_sidecar_material(bundle) != _desired_first_party_sidecar_material()
        )
    except RasterOfflineBundleBootstrapError:
        raise
    except OSError:
        raise RasterOfflineBundleBootstrapError("raster_bundle_release_base_invalid") from None


def _is_explicit_local_english_recognition_upgrade(
    *, base_configuration: Mapping[str, object], desired_configuration: Mapping[str, object]
) -> bool:
    """Allow exactly one pre-attested server-to-English recognition replacement."""

    changed_keys = tuple(
        key
        for key in _MODEL_CONFIGURATION_KEYS
        if base_configuration.get(key) != desired_configuration.get(key)
    )
    return (
        changed_keys == ("text_recognition_model_name",)
        and base_configuration.get("text_recognition_model_name") == _SERVER_RECOGNITION_MODEL_NAME
        and base_configuration.get("text_recognition_model_directory")
        == _RECOGNITION_MODEL_TARGET_DIRECTORY
        and desired_configuration.get("text_recognition_model_name")
        == _LOCAL_ENGLISH_RECOGNITION_MODEL_NAME
        and desired_configuration.get("text_recognition_model_directory")
        == _RECOGNITION_MODEL_TARGET_DIRECTORY
    )


def _verify_release_reuse_compatibility(
    bundle: VerifiedRasterOfflineBundleV1,
) -> _RecognitionModelSnapshotV1 | None:
    """Allow source/config changes only when pinned native material still fits.

    Reusing an already-verified interpreter/packages/models avoids another
    download, but it must not silently cross a package-lock or model contract
    change.  Those changes require a separately provisioned offline bundle.
    """

    try:
        receipt_path = bundle.bundle_root / "provenance" / "bootstrap-receipt.json"
        receipt = _read_canonical_object(
            receipt_path, code="raster_bundle_release_base_provenance_invalid"
        )
        expected_lock = _sha256_file(_project_root() / _SIDECAR_REQUIREMENTS_RELATIVE_PATH)[0]
        if receipt.get("sidecar_lock_sha256") != expected_lock:
            raise RasterOfflineBundleBootstrapError(
                "raster_bundle_release_sidecar_packages_changed"
            )
        configuration_path = bundle.bundle_root.joinpath(
            *PurePosixPath(bundle.manifest.sidecar_configuration_template_relative_path).parts
        )
        base_configuration = _read_canonical_object(
            configuration_path, code="raster_bundle_release_base_invalid"
        )
        desired_configuration = _configuration_template()
        model_configuration_changed = any(
            base_configuration.get(key) != desired_configuration.get(key)
            for key in _MODEL_CONFIGURATION_KEYS
        )
        if not model_configuration_changed:
            return None
        if not _is_explicit_local_english_recognition_upgrade(
            base_configuration=base_configuration, desired_configuration=desired_configuration
        ):
            raise RasterOfflineBundleBootstrapError("raster_bundle_release_models_changed")
        return _snapshot_local_english_recognition_model(bundle.document_preview_root)
    except RasterOfflineBundleBootstrapError:
        raise
    except OSError:
        raise RasterOfflineBundleBootstrapError("raster_bundle_release_base_invalid") from None


def _raster_text_release_derivation_intent(
    *,
    base: VerifiedRasterOfflineBundleV1,
    recognition_model_snapshot: _RecognitionModelSnapshotV1 | None,
    raster_text_model_snapshot: _RasterTextDirectModelSnapshotV1,
) -> tuple[dict[str, object], str]:
    """Bind one text-sidecar derivation to every byte that can change it.

    The result is intentionally a compact hash-attested intent rather than a
    directory discovery mechanism. A later invocation can address only the
    one index file derived from this value; it never searches historical
    releases for something that merely looks compatible.
    """

    try:
        first_party_files = [
            {
                "relative_path": relative_path,
                "sha256": hashlib.sha256(material).hexdigest(),
            }
            for relative_path, material in sorted(_desired_first_party_sidecar_material().items())
        ]
        if not first_party_files:
            raise RasterOfflineBundleBootstrapError("raster_bundle_release_base_invalid")
        configuration_template_sha256 = hashlib.sha256(
            _canonical_json(_configuration_template())
        ).hexdigest()
        sidecar_lock_sha256 = _sha256_file(_project_root() / _SIDECAR_REQUIREMENTS_RELATIVE_PATH)[0]
        recognition_input: dict[str, str] | None = None
        if recognition_model_snapshot is not None:
            recognition_input = {
                "source_relative_path": recognition_model_snapshot.source_relative_path,
                "tree_sha256": recognition_model_snapshot.tree_sha256,
            }
        intent: dict[str, object] = {
            "schema_version": _RASTER_TEXT_RELEASE_INTENT_SCHEMA_VERSION,
            "intent_kind": "raster-text-unselected-release-v1",
            "base_bundle_manifest_sha256": base.manifest_sha256,
            "base_bundle_root_relative_path": base.bundle_root_relative_path,
            "sidecar_lock_sha256": sidecar_lock_sha256,
            "configuration_template_sha256": configuration_template_sha256,
            "first_party_sidecar_files": first_party_files,
            "recognition_model_snapshot": recognition_input,
            "raster_text_model_snapshot": _raster_text_direct_snapshot_payload(
                raster_text_model_snapshot
            ),
        }
        return intent, hashlib.sha256(_canonical_json(intent)).hexdigest()
    except RasterOfflineBundleBootstrapError:
        raise
    except (OSError, TypeError, ValueError):
        raise RasterOfflineBundleBootstrapError("raster_bundle_release_base_invalid") from None


def _remove_owned_release_stage_subtree(*, stage: Path, target: Path) -> None:
    """Delete only a regular subtree below a private, verified stage tree."""

    try:
        canonical_stage = stage.resolve(strict=True)
        canonical_target = target.resolve(strict=True)
        if (
            stage.is_symlink()
            or target.is_symlink()
            or canonical_target == canonical_stage
            or canonical_target.parent != canonical_stage / "sidecar" / "module-root"
            or canonical_target.name != "customs_ai_agent"
        ):
            raise RasterOfflineBundleBootstrapError(
                "raster_bundle_release_stage_first_party_remove_invalid"
            )
        # ``stage`` was copied only from a fully verified regular-file base.
        # Re-walking a long Windows path here can itself expose legacy-path
        # reparse false positives; the exact target/root containment checks
        # above are the deletion boundary, and ``copytree`` never introduces
        # links when ``symlinks=False``.
        shutil.rmtree(_windows_operational_path(canonical_target, force_short=True))
    except RasterOfflineBundleBootstrapError:
        raise
    except OSError:
        raise RasterOfflineBundleBootstrapError(
            "raster_bundle_release_stage_first_party_remove_invalid"
        ) from None


def _remove_owned_release_recognition_model_directory(*, stage: Path, target: Path) -> None:
    """Delete only the inherited server-recognition directory in a private stage."""

    try:
        canonical_stage = stage.resolve(strict=True)
        canonical_target = target.resolve(strict=True)
        expected_target = canonical_stage / "models" / _RECOGNITION_MODEL_TARGET_DIRECTORY
        if (
            stage.is_symlink()
            or target.is_symlink()
            or canonical_target != expected_target
            or canonical_target == canonical_stage
        ):
            raise RasterOfflineBundleBootstrapError(
                "raster_bundle_release_stage_recognition_remove_invalid"
            )
        _require_regular_directory(
            target, code="raster_bundle_release_stage_recognition_remove_invalid"
        )
        _assert_regular_tree(target)
        shutil.rmtree(_windows_operational_path(canonical_target, force_short=True))
    except RasterOfflineBundleBootstrapError:
        raise
    except OSError:
        raise RasterOfflineBundleBootstrapError(
            "raster_bundle_release_stage_recognition_remove_invalid"
        ) from None


def _replacement_source_snapshot_or_drift(
    *, document_preview_root: Path, expected: _RecognitionModelSnapshotV1
) -> _RecognitionModelSnapshotV1:
    try:
        actual = _snapshot_local_english_recognition_model(document_preview_root)
    except RasterOfflineBundleBootstrapError:
        raise RasterOfflineBundleBootstrapError(
            "raster_bundle_release_recognition_source_drift"
        ) from None
    if actual != expected:
        raise RasterOfflineBundleBootstrapError("raster_bundle_release_recognition_source_drift")
    return actual


def _replace_release_recognition_model_from_local_snapshot(
    *,
    document_preview_root: Path,
    stage: Path,
    expected_snapshot: _RecognitionModelSnapshotV1,
) -> None:
    """Copy the fixed English model into a release and attest both endpoints."""

    if (
        expected_snapshot.source_relative_path
        != _LOCAL_ENGLISH_RECOGNITION_SOURCE_RELATIVE_PATH.as_posix()
    ):
        raise RasterOfflineBundleBootstrapError(
            "raster_bundle_release_recognition_snapshot_invalid"
        )
    _replacement_source_snapshot_or_drift(
        document_preview_root=document_preview_root, expected=expected_snapshot
    )
    root = _document_preview_root(document_preview_root)
    source = root.joinpath(*_LOCAL_ENGLISH_RECOGNITION_SOURCE_RELATIVE_PATH.parts)
    target = stage / "models" / _RECOGNITION_MODEL_TARGET_DIRECTORY
    _remove_owned_release_recognition_model_directory(stage=stage, target=target)
    try:
        shutil.copytree(
            _windows_operational_path(source, force_short=True),
            _windows_extended_write_path(target),
            symlinks=False,
            copy_function=shutil.copy2,
        )
    except OSError:
        raise RasterOfflineBundleBootstrapError(
            "raster_bundle_release_recognition_copy_failed"
        ) from None
    try:
        copied_tree_sha256 = _snapshot_pinned_local_english_recognition_directory(
            target, code="raster_bundle_release_recognition_snapshot_invalid"
        )
    except RasterOfflineBundleBootstrapError:
        raise
    if copied_tree_sha256 != expected_snapshot.tree_sha256:
        raise RasterOfflineBundleBootstrapError(
            "raster_bundle_release_recognition_snapshot_invalid"
        )
    _replacement_source_snapshot_or_drift(
        document_preview_root=document_preview_root, expected=expected_snapshot
    )


def _replace_owned_release_file(*, stage: Path, target: Path) -> None:
    """Unlink one known regular stage file before the create-only writer runs."""

    try:
        canonical_stage = stage.resolve(strict=True)
        canonical_target = target.resolve(strict=True)
        canonical_target.relative_to(canonical_stage)
        if target.is_symlink() or not target.is_file():
            raise RasterOfflineBundleBootstrapError(
                "raster_bundle_release_stage_file_replace_invalid"
            )
        target.unlink()
    except RasterOfflineBundleBootstrapError:
        raise
    except (OSError, ValueError):
        raise RasterOfflineBundleBootstrapError(
            "raster_bundle_release_stage_file_replace_invalid"
        ) from None


def _prepare_release_stage(
    *,
    base: VerifiedRasterOfflineBundleV1,
    stage: Path,
    recognition_model_snapshot: _RecognitionModelSnapshotV1 | None,
    raster_text_model_snapshot: _RasterTextDirectModelSnapshotV1 | None = None,
    raster_text_derivation_intent_sha256: str | None = None,
) -> None:
    """Copy verified native bytes, replace first-party source/config, re-manifest."""

    step = "initialization"
    try:
        if (raster_text_model_snapshot is None) != (raster_text_derivation_intent_sha256 is None):
            raise RasterOfflineBundleBootstrapError("raster_bundle_release_stage_invalid")
        # A direct snapshot is made from the base's already English-configured
        # ``text-rec`` tree.  If a structural server-rec upgrade is pending,
        # this text-only materializer refuses to infer a mixed source: a
        # separately attested current local bundle is required instead.
        if raster_text_model_snapshot is not None and recognition_model_snapshot is not None:
            raise RasterOfflineBundleBootstrapError(
                "raster_bundle_raster_text_model_source_invalid"
            )
        if stage.exists() or stage.is_symlink():
            raise RasterOfflineBundleBootstrapError("raster_bundle_release_stage_invalid")
        source = _windows_operational_path(base.bundle_root, force_short=True)
        canonical_source = _resolved_comparison_path(source)
        declared_files = frozenset(base.file_sha256_by_relative_path)
        declared_directories = {""}
        for relative_path in declared_files:
            parent = PurePosixPath(relative_path).parent
            while parent != PurePosixPath("."):
                declared_directories.add(parent.as_posix())
                parent = parent.parent

        def _ignore_non_manifest_material(directory: str, names: list[str]) -> set[str]:
            """Copy only manifest-bound files; discard proven legacy caches."""

            try:
                # ``copytree`` can surface a mixture of long and DOS-short
                # spellings on Windows.  Compare resolved identities, not
                # their presentation, before deriving a manifest path.
                relative_directory = (
                    _resolved_comparison_path(Path(directory))
                    .relative_to(canonical_source)
                    .as_posix()
                )
            except (OSError, ValueError) as error:
                raise RasterOfflineBundleBootstrapError(
                    "raster_bundle_release_stage_copy_filter_invalid"
                ) from error
            prefix = "" if relative_directory == "." else f"{relative_directory}/"
            ignored: set[str] = set()
            for name in names:
                relative_path = f"{prefix}{name}"
                # ``manifest.json`` is copied so the existing controlled
                # replacement helper can remove it before re-manifesting.
                if (
                    relative_path != "manifest.json"
                    and relative_path not in declared_files
                    and relative_path not in declared_directories
                ):
                    ignored.add(name)
            return ignored

        step = "copy"
        shutil.copytree(
            source,
            _windows_extended_write_path(
                _windows_operational_path(stage.parent, force_short=True) / stage.name
            ),
            symlinks=False,
            copy_function=shutil.copy2,
            ignore=_ignore_non_manifest_material,
        )
        operational_stage = _windows_operational_path(stage, force_short=True)
        first_party_target = operational_stage / "sidecar" / "module-root" / "customs_ai_agent"
        step = "first_party_replace"
        _remove_owned_release_stage_subtree(stage=operational_stage, target=first_party_target)
        _copy_first_party_sidecar(module_root=operational_stage / "sidecar" / "module-root")
        if recognition_model_snapshot is not None:
            step = "recognition_model_replace"
            _replace_release_recognition_model_from_local_snapshot(
                document_preview_root=base.document_preview_root,
                stage=operational_stage,
                expected_snapshot=recognition_model_snapshot,
            )
            model_inventory = operational_stage / "inventories" / "models.json"
            _replace_owned_release_file(stage=operational_stage, target=model_inventory)
            _write_canonical(model_inventory, _model_inventory(operational_stage / "models"))
        if raster_text_model_snapshot is not None:
            step = "raster_text_model_copy"
            _copy_raster_text_direct_models_to_stage(
                base=base,
                stage=operational_stage,
                expected_snapshot=raster_text_model_snapshot,
            )
        inherited_template = operational_stage.joinpath(
            *PurePosixPath(base.manifest.sidecar_configuration_template_relative_path).parts
        )
        template = operational_stage / "sidecar" / "configuration-template.json"
        if inherited_template != template:
            _replace_owned_release_file(stage=operational_stage, target=inherited_template)
        if template.exists() or template.is_symlink():
            _replace_owned_release_file(stage=operational_stage, target=template)
        step = "configuration"
        _write_canonical(template, _configuration_template())
        derivation_receipt = operational_stage / "provenance" / "release-derivation.json"
        if derivation_receipt.exists() or derivation_receipt.is_symlink():
            # A selected content-addressed release can itself be the verified
            # base for a new source/config derivation. Its old receipt is
            # owned by that copied stage, not reusable provenance for the
            # new release, so replace it before the create-only writer runs.
            _replace_owned_release_file(stage=operational_stage, target=derivation_receipt)
        derivation_payload: dict[str, object] = {
            "schema_version": "1.0",
            "release_kind": "local-source-config-derivation-v1",
            "base_bundle_manifest_sha256": base.manifest_sha256,
            "base_bundle_root_relative_path": base.bundle_root_relative_path,
            "sidecar_lock_sha256": _sha256_file(
                _project_root() / _SIDECAR_REQUIREMENTS_RELATIVE_PATH
            )[0],
            "first_party_sidecar_sha256": hashlib.sha256(
                _desired_first_party_sidecar_material()["service/ppstructurev3_sidecar.py"]
            ).hexdigest(),
            "configuration_template_sha256": hashlib.sha256(
                _canonical_json(_configuration_template())
            ).hexdigest(),
        }
        if raster_text_derivation_intent_sha256 is not None:
            if _RELEASE_MANIFEST_SHA256.fullmatch(raster_text_derivation_intent_sha256) is None:
                raise RasterOfflineBundleBootstrapError("raster_bundle_release_stage_invalid")
            derivation_payload[_RASTER_TEXT_RELEASE_INTENT_FIELD] = (
                raster_text_derivation_intent_sha256
            )
            if raster_text_model_snapshot is None:
                raise RasterOfflineBundleBootstrapError("raster_bundle_release_stage_invalid")
            derivation_payload["raster_text_model_snapshot"] = _raster_text_direct_snapshot_payload(
                raster_text_model_snapshot
            )
        if recognition_model_snapshot is not None:
            derivation_payload["recognition_model_snapshot"] = {
                "model_name": _LOCAL_ENGLISH_RECOGNITION_MODEL_NAME,
                "source_relative_path": recognition_model_snapshot.source_relative_path,
                "target_directory": _RECOGNITION_MODEL_TARGET_DIRECTORY,
                "tree_sha256": recognition_model_snapshot.tree_sha256,
            }
        _write_canonical(
            derivation_receipt,
            derivation_payload,
        )
        step = "manifest"
        _replace_owned_release_file(
            stage=operational_stage, target=operational_stage / "manifest.json"
        )
        _write_manifest(
            stage=operational_stage,
            raster_text_models=raster_text_model_snapshot is not None,
        )
    except RasterOfflineBundleBootstrapError:
        raise
    except OSError as error:
        if step == "copy":
            # Keep platform error details out of the public/local activation
            # transcript, but distinguish the only actionable Windows copy
            # conditions from an ordinary fail-closed copy failure.
            if getattr(error, "winerror", None) == 206:
                raise RasterOfflineBundleBootstrapError(
                    "raster_bundle_release_stage_copy_path_unsupported"
                ) from None
            if getattr(error, "winerror", None) in {32, 33}:
                raise RasterOfflineBundleBootstrapError(
                    "raster_bundle_release_stage_copy_locked"
                ) from None
            if getattr(error, "winerror", None) == 3:
                raise RasterOfflineBundleBootstrapError(
                    "raster_bundle_release_stage_copy_source_missing"
                ) from None
        raise RasterOfflineBundleBootstrapError(
            f"raster_bundle_release_stage_{step}_failed"
        ) from None


def _release_relative_path(manifest_sha256: str) -> str:
    if len(manifest_sha256) != 64 or any(
        character not in "0123456789abcdef" for character in manifest_sha256
    ):
        raise RasterOfflineBundleBootstrapError("raster_bundle_release_stage_invalid")
    return f"{_BUNDLE_RELEASE_COLLECTION_DIRECTORY}/{_BUNDLE_RELEASES_DIRECTORY}/{manifest_sha256}"


def _selector_payload(*, manifest_sha256: str) -> dict[str, object]:
    return {
        "schema_version": "1.0",
        "bundle_id": _BUNDLE_DIRECTORY,
        "release_manifest_sha256": manifest_sha256,
        "release_relative_path": _release_relative_path(manifest_sha256),
    }


def _atomic_replace_release_selector(*, collection: Path, manifest_sha256: str) -> None:
    """Replace the sole mutable selection only after its release exists."""

    selector = collection / _BUNDLE_RELEASE_SELECTOR_FILE
    payload = _canonical_json(_selector_payload(manifest_sha256=manifest_sha256))
    temporary = (
        collection / f".{_BUNDLE_RELEASE_SELECTOR_FILE}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        if collection.is_symlink() or not collection.is_dir() or selector.is_symlink():
            raise RasterOfflineBundleBootstrapError("raster_bundle_release_selector_invalid")
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, selector)
    except RasterOfflineBundleBootstrapError:
        raise
    except OSError:
        raise RasterOfflineBundleBootstrapError("raster_bundle_release_selector_invalid") from None
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


def _publish_release_from_stage(
    *, root: Path, operational_root: Path, stage: Path, manifest_sha256: str
) -> None:
    """Publish a verified stage without modifying the legacy root."""

    collection = operational_root / _BUNDLE_RELEASE_COLLECTION_DIRECTORY
    releases = collection / _BUNDLE_RELEASES_DIRECTORY
    final = releases / manifest_sha256
    try:
        if collection.exists() or collection.is_symlink():
            if collection.is_symlink() or not collection.is_dir():
                raise RasterOfflineBundleBootstrapError("raster_bundle_release_selector_invalid")
            # A collection exists only after its selector was atomically
            # published.  Recheck it rather than treating an incomplete tree
            # as permission to fall back to the legacy root.
            load_verified_raster_offline_bundle(root)
            if releases.is_symlink() or not releases.is_dir():
                raise RasterOfflineBundleBootstrapError("raster_bundle_release_selector_invalid")
            if final.exists() or final.is_symlink():
                try:
                    existing = load_verified_raster_offline_bundle_release(
                        document_preview_root=root, manifest_sha256=manifest_sha256
                    )
                except RasterOfflineBundleError:
                    raise RasterOfflineBundleBootstrapError(
                        "raster_bundle_release_publish_conflict"
                    ) from None
                if existing.manifest_sha256 != manifest_sha256:
                    raise RasterOfflineBundleBootstrapError(
                        "raster_bundle_release_publish_conflict"
                    )
            else:
                os.rename(_windows_operational_path(stage, force_short=True), final)
            _atomic_replace_release_selector(collection=collection, manifest_sha256=manifest_sha256)
            return

        # The first selected release is published as one complete collection
        # (release tree plus selector) by a same-volume directory rename.
        token = uuid.uuid4().hex
        collection_stage = (
            operational_root / f".{_BUNDLE_RELEASE_COLLECTION_DIRECTORY}.staging-{token}"
        )
        try:
            collection_stage.mkdir(mode=0o700)
            operational_collection_stage = _windows_operational_path(
                collection_stage, force_short=True
            )
            staged_releases = operational_collection_stage / _BUNDLE_RELEASES_DIRECTORY
            staged_releases.mkdir(parents=True)
            os.rename(
                _windows_operational_path(stage, force_short=True),
                staged_releases / manifest_sha256,
            )
            _atomic_replace_release_selector(
                collection=operational_collection_stage, manifest_sha256=manifest_sha256
            )
            os.rename(operational_collection_stage, collection)
        except RasterOfflineBundleBootstrapError:
            raise
        except OSError:
            raise RasterOfflineBundleBootstrapError(
                "raster_bundle_release_publish_conflict"
            ) from None
        finally:
            if collection_stage.exists():
                _remove_owned_tree(
                    root,
                    collection_stage,
                    prefix=f".{_BUNDLE_RELEASE_COLLECTION_DIRECTORY}.staging-",
                )
    except RasterOfflineBundleBootstrapError:
        raise
    except OSError:
        raise RasterOfflineBundleBootstrapError("raster_bundle_release_publish_conflict") from None


def _selected_release_manifest_sha_for_unselected_materialization(
    *, root: Path, operational_root: Path
) -> str:
    """Validate selector *shape* without trusting its selected release bytes.

    A text-only materialization may repair neither ``current.json`` nor its
    target.  It needs the collection only as an immutable-release namespace,
    so this deliberately stops after checking the canonical selector record.
    The normal loader remains the sole service resolver for the selected
    structural release.
    """

    collection = operational_root / _BUNDLE_RELEASE_COLLECTION_DIRECTORY
    releases = collection / _BUNDLE_RELEASES_DIRECTORY
    selector = collection / _BUNDLE_RELEASE_SELECTOR_FILE
    try:
        if (
            collection.is_symlink()
            or not collection.is_dir()
            or releases.is_symlink()
            or not releases.is_dir()
            or selector.is_symlink()
        ):
            raise RasterOfflineBundleBootstrapError("raster_bundle_unselected_release_unavailable")
        value = _read_canonical_object(
            selector, code="raster_bundle_unselected_release_unavailable"
        )
        expected_keys = {
            "schema_version",
            "bundle_id",
            "release_manifest_sha256",
            "release_relative_path",
        }
        manifest_sha256 = value.get("release_manifest_sha256")
        if (
            set(value) != expected_keys
            or value.get("schema_version") != "1.0"
            or value.get("bundle_id") != _BUNDLE_DIRECTORY
            or not isinstance(manifest_sha256, str)
            or _RELEASE_MANIFEST_SHA256.fullmatch(manifest_sha256) is None
            or value.get("release_relative_path") != _release_relative_path(manifest_sha256)
        ):
            raise RasterOfflineBundleBootstrapError("raster_bundle_unselected_release_unavailable")
        return manifest_sha256
    except RasterOfflineBundleBootstrapError:
        raise
    except OSError:
        raise RasterOfflineBundleBootstrapError(
            "raster_bundle_unselected_release_unavailable"
        ) from None


def _verified_predecessor_for_unselected_materialization(
    *, root: Path, operational_root: Path
) -> VerifiedRasterOfflineBundleV1:
    """Verify only the exact predecessor attested by the broken release.

    A selected release with generated cache files is not a normal verifier
    input, but its canonical manifest can still attest one small derivation
    receipt.  We validate the manifest hash, that receipt's declared hash and
    size, and its exact base path *before* using it as a lookup hint.  The
    hinted predecessor is then fully verified by the ordinary exhaustive
    release verifier.  This avoids an expensive and unsafe scan of arbitrary
    historical releases while never accepting bytes from the broken release.
    """

    selected_manifest_sha256 = _selected_release_manifest_sha_for_unselected_materialization(
        root=root, operational_root=operational_root
    )
    selected_root = operational_root.joinpath(
        *_release_relative_path(selected_manifest_sha256).split("/")
    )
    receipt_relative_path = "provenance/release-derivation.json"
    try:
        if selected_root.is_symlink() or not selected_root.is_dir():
            raise RasterOfflineBundleBootstrapError("raster_bundle_unselected_release_unavailable")
        manifest_path = selected_root / "manifest.json"
        manifest_raw = _read_canonical_object(
            manifest_path, code="raster_bundle_unselected_release_unavailable"
        )
        manifest_bytes = manifest_path.read_bytes()
        if hashlib.sha256(manifest_bytes).hexdigest() != selected_manifest_sha256:
            raise RasterOfflineBundleBootstrapError("raster_bundle_unselected_release_unavailable")
        try:
            manifest = RasterOfflineBundleManifestV1.model_validate(manifest_raw)
        except ValidationError:
            raise RasterOfflineBundleBootstrapError(
                "raster_bundle_unselected_release_unavailable"
            ) from None
        if manifest.bundle_id != _BUNDLE_DIRECTORY:
            raise RasterOfflineBundleBootstrapError("raster_bundle_unselected_release_unavailable")
        matching_receipts = [
            entry for entry in manifest.files if entry.path == receipt_relative_path
        ]
        if len(matching_receipts) != 1:
            raise RasterOfflineBundleBootstrapError("raster_bundle_unselected_release_unavailable")
        receipt_path = selected_root
        for part in PurePosixPath(receipt_relative_path).parts:
            receipt_path = receipt_path / part
            if receipt_path.is_symlink():
                raise RasterOfflineBundleBootstrapError(
                    "raster_bundle_unselected_release_unavailable"
                )
        receipt_sha256, receipt_size = _regular_file_sha256(
            receipt_path, code="raster_bundle_unselected_release_unavailable"
        )
        receipt_entry = matching_receipts[0]
        if receipt_sha256 != receipt_entry.sha256 or receipt_size != receipt_entry.byte_size:
            raise RasterOfflineBundleBootstrapError("raster_bundle_unselected_release_unavailable")
        receipt = _read_canonical_object(
            receipt_path, code="raster_bundle_unselected_release_unavailable"
        )
        base_manifest_sha256 = receipt.get("base_bundle_manifest_sha256")
        if (
            receipt.get("release_kind") != "local-source-config-derivation-v1"
            or not isinstance(base_manifest_sha256, str)
            or _RELEASE_MANIFEST_SHA256.fullmatch(base_manifest_sha256) is None
            or base_manifest_sha256 == selected_manifest_sha256
            or receipt.get("base_bundle_root_relative_path")
            != _release_relative_path(base_manifest_sha256)
        ):
            raise RasterOfflineBundleBootstrapError("raster_bundle_unselected_release_unavailable")
        return load_verified_raster_offline_bundle_release(
            document_preview_root=root, manifest_sha256=base_manifest_sha256
        )
    except RasterOfflineBundleBootstrapError:
        raise
    except RasterOfflineBundleError as error:
        raise RasterOfflineBundleBootstrapError(
            "raster_bundle_unselected_release_unavailable"
        ) from error
    except (OSError, ValueError):
        raise RasterOfflineBundleBootstrapError(
            "raster_bundle_unselected_release_unavailable"
        ) from None


def _publish_unselected_release_from_stage(
    *, root: Path, operational_root: Path, stage: Path, manifest_sha256: str
) -> VerifiedRasterOfflineBundleV1:
    """Create/reuse one verified release without replacing ``current.json``.

    The structural selector is deliberately outside the raster-text approval
    transaction.  A newly materialized text sidecar release is addressable by
    its full manifest hash, while the sole user-visible activation remains
    ``approvals/raster-text-active.json`` after the text Gate succeeds.
    """

    collection = operational_root / _BUNDLE_RELEASE_COLLECTION_DIRECTORY
    releases = collection / _BUNDLE_RELEASES_DIRECTORY
    final = releases / manifest_sha256
    try:
        # The selector bytes are only shape-checked; they are neither read as
        # a base nor rewritten.  This permits recovery from an invalid old
        # selected release while preserving it byte-for-byte.
        _selected_release_manifest_sha_for_unselected_materialization(
            root=root, operational_root=operational_root
        )
        if stage.is_symlink() or not stage.is_dir() or final.is_symlink():
            raise RasterOfflineBundleBootstrapError(
                "raster_bundle_unselected_release_publish_conflict"
            )
        if final.exists():
            existing = load_verified_raster_offline_bundle_release(
                document_preview_root=root, manifest_sha256=manifest_sha256
            )
            if existing.manifest_sha256 != manifest_sha256:
                raise RasterOfflineBundleBootstrapError(
                    "raster_bundle_unselected_release_publish_conflict"
                )
            return existing
        # ``stage`` is a same-volume, private sibling and has passed the
        # exhaustive verifier and isolated smoke before this call.  Rename is
        # create-only on Windows; it cannot overwrite an old release.
        os.rename(_windows_operational_path(stage, force_short=True), final)
        published = load_verified_raster_offline_bundle_release(
            document_preview_root=root, manifest_sha256=manifest_sha256
        )
        if published.manifest_sha256 != manifest_sha256:
            raise RasterOfflineBundleBootstrapError(
                "raster_bundle_unselected_release_publish_conflict"
            )
        return published
    except RasterOfflineBundleBootstrapError:
        raise
    except RasterOfflineBundleError as error:
        raise RasterOfflineBundleBootstrapError(
            "raster_bundle_unselected_release_publish_conflict"
        ) from error
    except OSError:
        raise RasterOfflineBundleBootstrapError(
            "raster_bundle_unselected_release_publish_conflict"
        ) from None


def _raster_text_release_intent_index_payload(
    *,
    intent: Mapping[str, object],
    intent_sha256: str,
    release_manifest_sha256: str,
) -> dict[str, object]:
    """Canonical, create-only binding from one exact intent to one release."""

    if (
        _RELEASE_MANIFEST_SHA256.fullmatch(intent_sha256) is None
        or _RELEASE_MANIFEST_SHA256.fullmatch(release_manifest_sha256) is None
    ):
        raise RasterOfflineBundleBootstrapError("raster_bundle_unselected_release_intent_invalid")
    return {
        "schema_version": _RASTER_TEXT_RELEASE_INTENT_SCHEMA_VERSION,
        "index_kind": "raster-text-unselected-release-intent-v1",
        "bundle_id": _BUNDLE_DIRECTORY,
        "intent_sha256": intent_sha256,
        "intent": dict(intent),
        "release_manifest_sha256": release_manifest_sha256,
        "release_relative_path": _release_relative_path(release_manifest_sha256),
    }


def _raster_text_release_intent_index_path(*, operational_root: Path, intent_sha256: str) -> Path:
    if _RELEASE_MANIFEST_SHA256.fullmatch(intent_sha256) is None:
        raise RasterOfflineBundleBootstrapError("raster_bundle_unselected_release_intent_invalid")
    return (
        operational_root
        / _BUNDLE_RELEASE_COLLECTION_DIRECTORY
        / _RASTER_TEXT_RELEASE_INTENT_DIRECTORY
        / f"{intent_sha256}.json"
    )


def _expected_raster_text_release_derivation_receipt(
    *, intent: Mapping[str, object], intent_sha256: str
) -> dict[str, object]:
    """Recover the exact stage receipt expected for the immutable intent."""

    try:
        first_party_files = intent["first_party_sidecar_files"]
        if not isinstance(first_party_files, list):
            raise TypeError("first_party_sidecar_files")
        first_party_sidecar_sha256: str | None = None
        for entry in first_party_files:
            if not isinstance(entry, dict):
                raise TypeError("first_party_sidecar_files")
            if entry.get("relative_path") == "service/ppstructurev3_sidecar.py":
                value = entry.get("sha256")
                if not isinstance(value, str) or _RELEASE_MANIFEST_SHA256.fullmatch(value) is None:
                    raise TypeError("first_party_sidecar_files")
                first_party_sidecar_sha256 = value
        if first_party_sidecar_sha256 is None:
            raise TypeError("first_party_sidecar_files")
        base_manifest_sha256 = intent["base_bundle_manifest_sha256"]
        base_bundle_root_relative_path = intent["base_bundle_root_relative_path"]
        sidecar_lock_sha256 = intent["sidecar_lock_sha256"]
        configuration_template_sha256 = intent["configuration_template_sha256"]
        if (
            not isinstance(base_manifest_sha256, str)
            or _RELEASE_MANIFEST_SHA256.fullmatch(base_manifest_sha256) is None
            or not isinstance(base_bundle_root_relative_path, str)
            or base_bundle_root_relative_path != _release_relative_path(base_manifest_sha256)
            or not isinstance(sidecar_lock_sha256, str)
            or _RELEASE_MANIFEST_SHA256.fullmatch(sidecar_lock_sha256) is None
            or not isinstance(configuration_template_sha256, str)
            or _RELEASE_MANIFEST_SHA256.fullmatch(configuration_template_sha256) is None
            or _RELEASE_MANIFEST_SHA256.fullmatch(intent_sha256) is None
        ):
            raise TypeError("intent")
        receipt: dict[str, object] = {
            "schema_version": "1.0",
            "release_kind": "local-source-config-derivation-v1",
            "base_bundle_manifest_sha256": base_manifest_sha256,
            "base_bundle_root_relative_path": base_bundle_root_relative_path,
            "sidecar_lock_sha256": sidecar_lock_sha256,
            "first_party_sidecar_sha256": first_party_sidecar_sha256,
            "configuration_template_sha256": configuration_template_sha256,
            _RASTER_TEXT_RELEASE_INTENT_FIELD: intent_sha256,
        }
        raster_text_model_snapshot = intent["raster_text_model_snapshot"]
        if (
            not isinstance(raster_text_model_snapshot, dict)
            or set(raster_text_model_snapshot)
            != {
                "schema_version",
                "inventory_kind",
                "source_bundle_manifest_sha256",
                "source_model_root_relative_path",
                "source_model_inventory_sha256",
                "source_snapshot_sha256",
                "artifacts",
            }
            or raster_text_model_snapshot.get("schema_version") != "1.0"
            or raster_text_model_snapshot.get("inventory_kind") != "raster-text-direct-models-v1"
            or raster_text_model_snapshot.get("source_bundle_manifest_sha256")
            != base_manifest_sha256
            or raster_text_model_snapshot.get("source_model_root_relative_path") != "models"
            or not isinstance(raster_text_model_snapshot.get("source_model_inventory_sha256"), str)
            or _RELEASE_MANIFEST_SHA256.fullmatch(
                str(raster_text_model_snapshot.get("source_model_inventory_sha256"))
            )
            is None
            or not isinstance(raster_text_model_snapshot.get("source_snapshot_sha256"), str)
            or _RELEASE_MANIFEST_SHA256.fullmatch(
                str(raster_text_model_snapshot.get("source_snapshot_sha256"))
            )
            is None
            or not isinstance(raster_text_model_snapshot.get("artifacts"), list)
            or len(raster_text_model_snapshot["artifacts"]) != len(_RASTER_TEXT_DIRECT_MODEL_PATHS)
        ):
            raise TypeError("raster_text_model_snapshot")
        normalized_artifacts: list[dict[str, object]] = []
        for expected_path, artifact in zip(
            _RASTER_TEXT_DIRECT_MODEL_PATHS,
            raster_text_model_snapshot["artifacts"],
            strict=True,
        ):
            if (
                not isinstance(artifact, dict)
                or set(artifact) != {"relative_path", "sha256", "byte_size"}
                or artifact.get("relative_path") != expected_path
                or not isinstance(artifact.get("sha256"), str)
                or _RELEASE_MANIFEST_SHA256.fullmatch(str(artifact.get("sha256"))) is None
                or isinstance(artifact.get("byte_size"), bool)
                or not isinstance(artifact.get("byte_size"), int)
                or not 0 < int(artifact["byte_size"]) <= _MAX_FILE_BYTES
            ):
                raise TypeError("raster_text_model_snapshot")
            normalized_artifacts.append(
                {
                    "relative_path": artifact["relative_path"],
                    "sha256": artifact["sha256"],
                    "byte_size": artifact["byte_size"],
                }
            )
        if (
            raster_text_model_snapshot["source_snapshot_sha256"]
            != hashlib.sha256(_canonical_json({"files": normalized_artifacts})).hexdigest()
        ):
            raise TypeError("raster_text_model_snapshot")
        receipt["raster_text_model_snapshot"] = {
            **raster_text_model_snapshot,
            "artifacts": normalized_artifacts,
        }
        recognition_model_snapshot = intent["recognition_model_snapshot"]
        if recognition_model_snapshot is not None:
            if not isinstance(recognition_model_snapshot, dict):
                raise TypeError("recognition_model_snapshot")
            source_relative_path = recognition_model_snapshot.get("source_relative_path")
            tree_sha256 = recognition_model_snapshot.get("tree_sha256")
            if (
                not isinstance(source_relative_path, str)
                or source_relative_path
                != _LOCAL_ENGLISH_RECOGNITION_SOURCE_RELATIVE_PATH.as_posix()
                or not isinstance(tree_sha256, str)
                or _RELEASE_MANIFEST_SHA256.fullmatch(tree_sha256) is None
            ):
                raise TypeError("recognition_model_snapshot")
            receipt["recognition_model_snapshot"] = {
                "model_name": _LOCAL_ENGLISH_RECOGNITION_MODEL_NAME,
                "source_relative_path": source_relative_path,
                "target_directory": _RECOGNITION_MODEL_TARGET_DIRECTORY,
                "tree_sha256": tree_sha256,
            }
        return receipt
    except (KeyError, TypeError, ValueError):
        raise RasterOfflineBundleBootstrapError(
            "raster_bundle_unselected_release_intent_invalid"
        ) from None


def _verify_raster_text_release_derivation_receipt(
    *,
    candidate: VerifiedRasterOfflineBundleV1,
    intent: Mapping[str, object],
    intent_sha256: str,
) -> None:
    """Ensure an indexed candidate was built from this exact base and inputs."""

    receipt_relative_path = "provenance/release-derivation.json"
    try:
        expected = _expected_raster_text_release_derivation_receipt(
            intent=intent, intent_sha256=intent_sha256
        )
        receipt_path = candidate.bundle_root
        for part in PurePosixPath(receipt_relative_path).parts:
            receipt_path = receipt_path / part
            if receipt_path.is_symlink():
                raise RasterOfflineBundleBootstrapError(
                    "raster_bundle_unselected_release_intent_candidate_invalid"
                )
        receipt_sha256, _receipt_size = _regular_file_sha256(
            receipt_path,
            code="raster_bundle_unselected_release_intent_candidate_invalid",
        )
        if receipt_sha256 != candidate.file_sha256(receipt_relative_path):
            raise RasterOfflineBundleBootstrapError(
                "raster_bundle_unselected_release_intent_candidate_invalid"
            )
        receipt = _read_canonical_object(
            receipt_path,
            code="raster_bundle_unselected_release_intent_candidate_invalid",
        )
        if receipt != expected:
            raise RasterOfflineBundleBootstrapError(
                "raster_bundle_unselected_release_intent_candidate_invalid"
            )
        if (
            candidate.manifest.raster_text_model_inventory_relative_path
            != _RASTER_TEXT_MODEL_INVENTORY_RELATIVE_PATH.as_posix()
            or candidate.manifest.raster_text_local_model_root_relative_path
            != _RASTER_TEXT_MODEL_ROOT_RELATIVE_PATH.as_posix()
        ):
            raise RasterOfflineBundleBootstrapError(
                "raster_bundle_unselected_release_intent_candidate_invalid"
            )
        verify_raster_text_direct_model_material(candidate)
        direct_inventory = _read_canonical_object(
            candidate.bundle_root.joinpath(
                *PurePosixPath(candidate.manifest.raster_text_model_inventory_relative_path).parts
            ),
            code="raster_bundle_unselected_release_intent_candidate_invalid",
        )
        if direct_inventory != expected["raster_text_model_snapshot"]:
            raise RasterOfflineBundleBootstrapError(
                "raster_bundle_unselected_release_intent_candidate_invalid"
            )
    except RasterOfflineBundleBootstrapError:
        raise
    except RasterOfflineBundleError as error:
        raise RasterOfflineBundleBootstrapError(
            "raster_bundle_unselected_release_intent_candidate_invalid"
        ) from error
    except (OSError, ValueError):
        raise RasterOfflineBundleBootstrapError(
            "raster_bundle_unselected_release_intent_candidate_invalid"
        ) from None


def _current_raster_text_source_binding() -> dict[str, object]:
    """Return the source-only part of a direct-text release intent.

    This is deliberately computable without resolving a selected bundle.  It
    is not a release selector: the result merely identifies which immutable
    text-intent records may be considered for a narrow, fail-closed reuse.
    """

    try:
        first_party_files = [
            {
                "relative_path": relative_path,
                "sha256": hashlib.sha256(material).hexdigest(),
            }
            for relative_path, material in sorted(_desired_first_party_sidecar_material().items())
        ]
        if not first_party_files:
            raise RasterOfflineBundleBootstrapError("raster_bundle_release_base_invalid")
        return {
            "sidecar_lock_sha256": _sha256_file(
                _project_root() / _SIDECAR_REQUIREMENTS_RELATIVE_PATH
            )[0],
            "configuration_template_sha256": hashlib.sha256(
                _canonical_json(_configuration_template())
            ).hexdigest(),
            "first_party_sidecar_files": first_party_files,
        }
    except RasterOfflineBundleBootstrapError:
        raise
    except (OSError, TypeError, ValueError):
        raise RasterOfflineBundleBootstrapError("raster_bundle_release_base_invalid") from None


def _intent_matches_current_raster_text_source(
    *, intent: Mapping[str, object], source_binding: Mapping[str, object]
) -> bool:
    """Return whether an indexed intent names exactly current text sources."""

    return (
        intent.get("sidecar_lock_sha256") == source_binding["sidecar_lock_sha256"]
        and intent.get("configuration_template_sha256")
        == source_binding["configuration_template_sha256"]
        and intent.get("first_party_sidecar_files") == source_binding["first_party_sidecar_files"]
    )


def _validated_current_raster_text_intent_index(
    *, root: Path, operational_root: Path
) -> tuple[dict[str, object], str, str] | None:
    """Find at most one current-source text intent without loading a bundle.

    The namespace is fixed and bounded; it is never a generic historical
    release search.  Unparseable *unrelated* historical records are ignored
    so a later full materialization can retain its exact-index fail-closed
    behavior.  A record that claims the current source binding is instead
    fully validated or rejected immediately.
    """

    source_binding = _current_raster_text_source_binding()
    intent_directory = (
        operational_root
        / _BUNDLE_RELEASE_COLLECTION_DIRECTORY
        / _RASTER_TEXT_RELEASE_INTENT_DIRECTORY
    )
    try:
        # A fresh local installation has neither a release collection nor a
        # text intent yet. Let the established full materializer report the
        # normal offline-bundle availability code in that case.
        if not intent_directory.exists():
            return None
        # Keep the existing namespace/selector boundary: a text-only path
        # may not repair or reinterpret the structural selector.
        _selected_release_manifest_sha_for_unselected_materialization(
            root=root, operational_root=operational_root
        )
        if intent_directory.is_symlink() or not intent_directory.is_dir():
            raise RasterOfflineBundleBootstrapError(
                "raster_bundle_unselected_release_intent_index_invalid"
            )
        entries = sorted(intent_directory.iterdir(), key=lambda value: value.name)
        if len(entries) > _RASTER_TEXT_RELEASE_INTENT_MAX_ENTRIES:
            raise RasterOfflineBundleBootstrapError(
                "raster_bundle_unselected_release_intent_index_invalid"
            )
    except RasterOfflineBundleBootstrapError:
        raise
    except OSError:
        raise RasterOfflineBundleBootstrapError(
            "raster_bundle_unselected_release_intent_index_invalid"
        ) from None

    matches: list[tuple[dict[str, object], str, str]] = []
    for entry in entries:
        # The normal writer creates only hash-named canonical JSON files. A
        # stray historical/broken record cannot grant reuse; it is ignored
        # unless it can actually claim the current source binding below.
        if (
            entry.is_symlink()
            or not entry.is_file()
            or _RELEASE_MANIFEST_SHA256.fullmatch(entry.stem) is None
            or entry.suffix != ".json"
        ):
            continue
        try:
            if not 0 < entry.stat().st_size <= _RASTER_TEXT_RELEASE_INTENT_MAX_BYTES:
                continue
            index = _read_canonical_object(
                entry, code="raster_bundle_unselected_release_intent_index_invalid"
            )
            intent_raw = index.get("intent")
            if not isinstance(intent_raw, dict) or not _intent_matches_current_raster_text_source(
                intent=intent_raw, source_binding=source_binding
            ):
                continue
            intent_sha256 = entry.stem
            release_manifest_sha256 = index.get("release_manifest_sha256")
            if (
                index.get("intent_sha256") != intent_sha256
                or hashlib.sha256(_canonical_json(intent_raw)).hexdigest() != intent_sha256
                or not isinstance(release_manifest_sha256, str)
            ):
                raise RasterOfflineBundleBootstrapError(
                    "raster_bundle_unselected_release_intent_index_invalid"
                )
            expected_index = _raster_text_release_intent_index_payload(
                intent=intent_raw,
                intent_sha256=intent_sha256,
                release_manifest_sha256=release_manifest_sha256,
            )
            # The receipt constructor also rejects missing, extra, or
            # structurally invalid direct-model provenance before an index
            # can become a release-addressing capability.
            _expected_raster_text_release_derivation_receipt(
                intent=intent_raw, intent_sha256=intent_sha256
            )
            if index != expected_index:
                raise RasterOfflineBundleBootstrapError(
                    "raster_bundle_unselected_release_intent_index_invalid"
                )
            matches.append((intent_raw, intent_sha256, release_manifest_sha256))
        except RasterOfflineBundleBootstrapError:
            # A malformed record is relevant only when it reached the
            # current-source comparison. It must never be bypassed by a
            # generic materialization in that case.
            raise
        except (OSError, TypeError, ValueError):
            # No source binding was safely established for this old entry;
            # leave the exact full-materializer path to decide it later.
            continue
    if len(matches) > 1:
        raise RasterOfflineBundleBootstrapError("raster_bundle_unselected_release_intent_ambiguous")
    return matches[0] if matches else None


def _relative_file_is_within(*, directory: str, file_path: str) -> bool:
    try:
        root = PurePosixPath(directory)
        candidate = PurePosixPath(file_path)
        return candidate != root and candidate.relative_to(root) != PurePosixPath(".")
    except ValueError:
        return False


def _fast_text_release_path(
    *, release_root: Path, relative_path: str, directory: bool, code: str
) -> Path:
    """Resolve one declared direct-text path without a generic tree walk."""

    try:
        _require_regular_directory(release_root, code=code)
        root = _resolved_comparison_path(release_root)
        candidate = release_root
        for part in PurePosixPath(relative_path).parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise RasterOfflineBundleBootstrapError(code)
        metadata = candidate.lstat()
        if (
            _is_link_or_reparse_point(metadata)
            or (directory and not stat.S_ISDIR(metadata.st_mode))
            or (not directory and not stat.S_ISREG(metadata.st_mode))
        ):
            raise RasterOfflineBundleBootstrapError(code)
        resolved = _resolved_comparison_path(candidate)
        resolved.relative_to(root)
        return _windows_operational_path(resolved, force_short=True)
    except RasterOfflineBundleBootstrapError:
        raise
    except (OSError, ValueError):
        raise RasterOfflineBundleBootstrapError(code) from None


def _validate_fast_text_release_manifest(
    *, manifest: RasterOfflineBundleManifestV1
) -> dict[str, str]:
    """Validate declaration shape needed by the direct-text profile only."""

    try:
        paths = tuple(item.path for item in manifest.files)
        if (
            manifest.schema_version != "1.0"
            or manifest.bundle_id != _BUNDLE_DIRECTORY
            or manifest.parent_renderer_required_packages != list(_PARENT_PACKAGES)
            or set(manifest.sidecar_required_packages) != set(_SIDECAR_REQUIRED_PACKAGES)
            or len(manifest.sidecar_required_packages) != len(_SIDECAR_REQUIRED_PACKAGES)
            or paths != tuple(sorted(paths))
            or len(paths) != len(set(paths))
            or manifest.sidecar_command_relative_path != manifest.sidecar_module_relative_path
            or not _relative_file_is_within(
                directory=manifest.sidecar_module_root_relative_path,
                file_path=manifest.sidecar_module_relative_path,
            )
            or not _relative_file_is_within(
                directory=manifest.sidecar_interpreter_root_relative_path,
                file_path=manifest.sidecar_interpreter_relative_path,
            )
        ):
            raise RasterOfflineBundleBootstrapError(
                "raster_bundle_unselected_release_intent_candidate_invalid"
            )
        hashes = {item.path: item.sha256 for item in manifest.files}
        if any(path not in hashes for path in manifest.required_file_paths()):
            raise RasterOfflineBundleBootstrapError(
                "raster_bundle_unselected_release_intent_candidate_invalid"
            )
        return hashes
    except RasterOfflineBundleBootstrapError:
        raise
    except (TypeError, ValueError):
        raise RasterOfflineBundleBootstrapError(
            "raster_bundle_unselected_release_intent_candidate_invalid"
        ) from None


def _verify_fast_text_release_file(
    *, reference: VerifiedRasterTextReleaseReferenceV1, relative_path: str, code: str
) -> Path:
    """Hash one profile-visible file against the immutable declaration."""

    try:
        expected_sha256 = reference.file_sha256(relative_path)
        path = _fast_text_release_path(
            release_root=reference.bundle_root,
            relative_path=relative_path,
            directory=False,
            code=code,
        )
        actual_sha256, _size = _regular_file_sha256(path, code=code)
        if actual_sha256 != expected_sha256:
            raise RasterOfflineBundleBootstrapError(code)
        return path
    except RasterOfflineBundleBootstrapError:
        raise
    except RasterOfflineBundleError as error:
        raise RasterOfflineBundleBootstrapError(code) from error


def _load_fast_verified_raster_text_release_reference(
    *,
    root: Path,
    operational_root: Path,
    intent: Mapping[str, object],
    intent_sha256: str,
    release_manifest_sha256: str,
) -> VerifiedRasterTextReleaseReferenceV1:
    """Narrowly attest one current-source text release before generic load."""

    code = "raster_bundle_unselected_release_intent_candidate_invalid"
    try:
        expected_relative_path = _release_relative_path(release_manifest_sha256)
        release_root = operational_root
        for part in PurePosixPath(expected_relative_path).parts:
            release_root = release_root / part
            _require_regular_directory(release_root, code=code)
        manifest_path = _fast_text_release_path(
            release_root=release_root,
            relative_path="manifest.json",
            directory=False,
            code=code,
        )
        manifest_sha256, _size = _regular_file_sha256(manifest_path, code=code)
        if manifest_sha256 != release_manifest_sha256:
            raise RasterOfflineBundleBootstrapError(code)
        manifest_raw = _read_canonical_object(manifest_path, code=code)
        try:
            manifest = RasterOfflineBundleManifestV1.model_validate(manifest_raw)
        except ValidationError:
            raise RasterOfflineBundleBootstrapError(code) from None
        file_hashes = _validate_fast_text_release_manifest(manifest=manifest)
        reference = VerifiedRasterTextReleaseReferenceV1(
            document_preview_root=root,
            bundle_root=_windows_operational_path(release_root, force_short=True),
            bundle_root_relative_path=expected_relative_path,
            manifest=manifest,
            manifest_sha256=release_manifest_sha256,
            file_sha256_by_relative_path=file_hashes,
        )
        expected_receipt = _expected_raster_text_release_derivation_receipt(
            intent=intent, intent_sha256=intent_sha256
        )
        receipt_path = _verify_fast_text_release_file(
            reference=reference,
            relative_path="provenance/release-derivation.json",
            code=code,
        )
        if _read_canonical_object(receipt_path, code=code) != expected_receipt:
            raise RasterOfflineBundleBootstrapError(code)

        # Every current first-party file is small and must exactly match the
        # source binding that selected this index. This includes the direct
        # text sidecar rather than merely trusting a module-root tree digest.
        first_party = intent.get("first_party_sidecar_files")
        if not isinstance(first_party, list):
            raise RasterOfflineBundleBootstrapError(code)
        expected_first_party_paths: set[str] = set()
        for entry in first_party:
            if (
                not isinstance(entry, dict)
                or set(entry) != {"relative_path", "sha256"}
                or not isinstance(entry.get("relative_path"), str)
                or not isinstance(entry.get("sha256"), str)
                or _RELEASE_MANIFEST_SHA256.fullmatch(str(entry["sha256"])) is None
            ):
                raise RasterOfflineBundleBootstrapError(code)
            relative_path = (
                f"{manifest.sidecar_module_root_relative_path}/customs_ai_agent/"
                f"{entry['relative_path']}"
            )
            if relative_path in expected_first_party_paths or not _relative_file_is_within(
                directory=manifest.sidecar_module_root_relative_path,
                file_path=relative_path,
            ):
                raise RasterOfflineBundleBootstrapError(code)
            expected_first_party_paths.add(relative_path)
            path = _verify_fast_text_release_file(
                reference=reference, relative_path=relative_path, code=code
            )
            actual_sha256, _size = _regular_file_sha256(path, code=code)
            if actual_sha256 != entry["sha256"]:
                raise RasterOfflineBundleBootstrapError(code)

        text_sidecar_relative_path = (
            f"{manifest.sidecar_module_root_relative_path}/"
            "customs_ai_agent/service/raster_text_sidecar.py"
        )
        if text_sidecar_relative_path not in expected_first_party_paths:
            raise RasterOfflineBundleBootstrapError(code)
        for relative_path in (
            manifest.parent_renderer_package_inventory_relative_path,
            manifest.sidecar_package_inventory_relative_path,
            manifest.sidecar_interpreter_relative_path,
            manifest.sidecar_configuration_template_relative_path,
            text_sidecar_relative_path,
        ):
            _verify_fast_text_release_file(
                reference=reference, relative_path=relative_path, code=code
            )
        configuration_path = _fast_text_release_path(
            release_root=reference.bundle_root,
            relative_path=manifest.sidecar_configuration_template_relative_path,
            directory=False,
            code=code,
        )
        if _regular_file_sha256(configuration_path, code=code)[0] != intent.get(
            "configuration_template_sha256"
        ):
            raise RasterOfflineBundleBootstrapError(code)

        direct_models = verify_raster_text_direct_model_material(reference)
        direct_inventory_path = _verify_fast_text_release_file(
            reference=reference,
            relative_path=direct_models.model_inventory_relative_path,
            code=code,
        )
        if (
            _read_canonical_object(direct_inventory_path, code=code)
            != expected_receipt["raster_text_model_snapshot"]
        ):
            raise RasterOfflineBundleBootstrapError(code)
        return reference
    except RasterOfflineBundleBootstrapError:
        raise
    except RasterOfflineBundleError as error:
        raise RasterOfflineBundleBootstrapError(code) from error
    except (OSError, TypeError, ValueError):
        raise RasterOfflineBundleBootstrapError(code) from None


def resolve_or_materialize_verified_raster_text_release_reference_v1(
    *, document_preview_root: Path
) -> VerifiedRasterOfflineBundleV1 | VerifiedRasterTextReleaseReferenceV1:
    """Fast-reuse a source-bound direct-text release, or materialize safely.

    A successful fast branch uses only the bounded direct-text intent index
    plus the profile-visible release material. It never calls the generic
    bundle loader. If no current-source intent exists, the established full
    materializer remains the sole creation path.
    """

    root = _document_preview_root(document_preview_root)
    operational_root = _windows_operational_path(root)
    with _bootstrap_lock(root):
        indexed = _validated_current_raster_text_intent_index(
            root=root, operational_root=operational_root
        )
        if indexed is not None:
            intent, intent_sha256, release_manifest_sha256 = indexed
            return _load_fast_verified_raster_text_release_reference(
                root=root,
                operational_root=operational_root,
                intent=intent,
                intent_sha256=intent_sha256,
                release_manifest_sha256=release_manifest_sha256,
            )
    # Do not hold the bootstrap lock across the existing materializer: it
    # owns the same non-reentrant OS lock and performs its own full checks.
    return materialize_verified_unselected_raster_text_release_v1(document_preview_root=root)


def _load_verified_raster_text_release_intent_reuse(
    *,
    root: Path,
    operational_root: Path,
    intent: Mapping[str, object],
    intent_sha256: str,
) -> VerifiedRasterOfflineBundleV1 | None:
    """Read one exact index path and fully verify only its referenced release."""

    index_path = _raster_text_release_intent_index_path(
        operational_root=operational_root, intent_sha256=intent_sha256
    )
    index_directory = index_path.parent
    try:
        # This only validates the immutable release namespace and selector
        # shape. It deliberately does not resolve or modify ``current.json``.
        _selected_release_manifest_sha_for_unselected_materialization(
            root=root, operational_root=operational_root
        )
        if index_directory.is_symlink():
            raise RasterOfflineBundleBootstrapError(
                "raster_bundle_unselected_release_intent_index_invalid"
            )
        if not index_directory.exists():
            return None
        if not index_directory.is_dir() or index_path.is_symlink():
            raise RasterOfflineBundleBootstrapError(
                "raster_bundle_unselected_release_intent_index_invalid"
            )
        if not index_path.exists():
            return None
        index = _read_canonical_object(
            index_path, code="raster_bundle_unselected_release_intent_index_invalid"
        )
        release_manifest_sha256 = index.get("release_manifest_sha256")
        if not isinstance(release_manifest_sha256, str):
            raise RasterOfflineBundleBootstrapError(
                "raster_bundle_unselected_release_intent_index_invalid"
            )
        expected_index = _raster_text_release_intent_index_payload(
            intent=intent,
            intent_sha256=intent_sha256,
            release_manifest_sha256=release_manifest_sha256,
        )
        if index != expected_index:
            raise RasterOfflineBundleBootstrapError(
                "raster_bundle_unselected_release_intent_index_invalid"
            )
    except RasterOfflineBundleBootstrapError:
        raise
    except (OSError, ValueError):
        raise RasterOfflineBundleBootstrapError(
            "raster_bundle_unselected_release_intent_index_invalid"
        ) from None

    try:
        candidate = load_verified_raster_offline_bundle_release(
            document_preview_root=root, manifest_sha256=release_manifest_sha256
        )
        if candidate.manifest_sha256 != release_manifest_sha256:
            raise RasterOfflineBundleBootstrapError(
                "raster_bundle_unselected_release_intent_candidate_invalid"
            )
        verify_parent_renderer_inventory(candidate)
        _verify_release_reuse_compatibility(candidate)
        if _release_requires_derivation(candidate):
            raise RasterOfflineBundleBootstrapError(
                "raster_bundle_unselected_release_intent_candidate_invalid"
            )
        _verify_raster_text_release_derivation_receipt(
            candidate=candidate, intent=intent, intent_sha256=intent_sha256
        )
        return candidate
    except RasterOfflineBundleBootstrapError:
        raise
    except RasterOfflineBundleError as error:
        raise RasterOfflineBundleBootstrapError(
            "raster_bundle_unselected_release_intent_candidate_invalid"
        ) from error


def _write_or_verify_raster_text_release_intent_index(
    *,
    root: Path,
    operational_root: Path,
    intent: Mapping[str, object],
    intent_sha256: str,
    published: VerifiedRasterOfflineBundleV1,
) -> None:
    """Create one canonical intent index or verify an equal concurrent entry."""

    index_path = _raster_text_release_intent_index_path(
        operational_root=operational_root, intent_sha256=intent_sha256
    )
    index_directory = index_path.parent
    payload = _canonical_json(
        _raster_text_release_intent_index_payload(
            intent=intent,
            intent_sha256=intent_sha256,
            release_manifest_sha256=published.manifest_sha256,
        )
    )
    try:
        _selected_release_manifest_sha_for_unselected_materialization(
            root=root, operational_root=operational_root
        )
        if index_directory.exists() or index_directory.is_symlink():
            if index_directory.is_symlink() or not index_directory.is_dir():
                raise RasterOfflineBundleBootstrapError(
                    "raster_bundle_unselected_release_intent_index_invalid"
                )
        else:
            index_directory.mkdir(mode=0o700)
        if index_path.is_symlink():
            raise RasterOfflineBundleBootstrapError(
                "raster_bundle_unselected_release_intent_index_invalid"
            )
        try:
            with index_path.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            return
        except FileExistsError:
            reused = _load_verified_raster_text_release_intent_reuse(
                root=root,
                operational_root=operational_root,
                intent=intent,
                intent_sha256=intent_sha256,
            )
            if reused is None or reused.manifest_sha256 != published.manifest_sha256:
                raise RasterOfflineBundleBootstrapError(
                    "raster_bundle_unselected_release_intent_index_invalid"
                )
    except RasterOfflineBundleBootstrapError:
        raise
    except OSError:
        raise RasterOfflineBundleBootstrapError(
            "raster_bundle_unselected_release_intent_index_invalid"
        ) from None


def materialize_verified_unselected_raster_text_release_v1(
    *, document_preview_root: Path
) -> VerifiedRasterOfflineBundleV1:
    """Return current text-sidecar bytes without changing the shared selector.

    This is the text-only counterpart to the structural release derivation.
    It accepts a normal selected release when it already contains the exact
    current first-party material.  If the selector's target is invalid, it
    can use only a separately fully verified predecessor release.  In both
    cases any new release is fully rebuilt in private staging, structurally
    smoked, reverified, and create-only published by its manifest hash.  It
    never calls ``_atomic_replace_release_selector``.
    """

    root = _document_preview_root(document_preview_root)
    operational_root = _windows_operational_path(root)
    with _bootstrap_lock(root):
        try:
            base = load_verified_raster_offline_bundle(root)
        except RasterOfflineBundleError as error:
            if error.code != "offline_bundle_release_selector_invalid":
                raise RasterOfflineBundleBootstrapError(error.code) from None
            base = _verified_predecessor_for_unselected_materialization(
                root=root, operational_root=operational_root
            )
        try:
            verify_parent_renderer_inventory(base)
            recognition_model_snapshot = _verify_release_reuse_compatibility(base)
            if recognition_model_snapshot is not None:
                # The direct profile accepts only the already-attested local
                # text-det/text-rec source pair.  It never fabricates a mixed
                # snapshot from a structural server-rec migration.
                raise RasterOfflineBundleBootstrapError(
                    "raster_bundle_raster_text_model_source_unavailable"
                )
            raster_text_model_snapshot = _snapshot_raster_text_direct_model_source(base)
            structural_requires_derivation = _release_requires_derivation(base)
            if (
                base.manifest.raster_text_model_inventory_relative_path is None
                and base.manifest.raster_text_local_model_root_relative_path is None
            ):
                raster_text_requires_derivation = True
            elif (
                base.manifest.raster_text_model_inventory_relative_path is None
                or base.manifest.raster_text_local_model_root_relative_path is None
            ):
                raise RasterOfflineBundleBootstrapError(
                    "raster_bundle_raster_text_model_material_invalid"
                )
            else:
                raster_text_requires_derivation = (
                    not _raster_text_direct_model_material_matches_source(
                        base, expected_snapshot=raster_text_model_snapshot
                    )
                )
            requires_derivation = structural_requires_derivation or raster_text_requires_derivation
            if not requires_derivation:
                return base
            intent, intent_sha256 = _raster_text_release_derivation_intent(
                base=base,
                recognition_model_snapshot=recognition_model_snapshot,
                raster_text_model_snapshot=raster_text_model_snapshot,
            )
            reused = _load_verified_raster_text_release_intent_reuse(
                root=root,
                operational_root=operational_root,
                intent=intent,
                intent_sha256=intent_sha256,
            )
            if reused is not None:
                current_intent, current_intent_sha256 = _raster_text_release_derivation_intent(
                    base=base,
                    recognition_model_snapshot=_verify_release_reuse_compatibility(base),
                    raster_text_model_snapshot=_snapshot_raster_text_direct_model_source(base),
                )
                if current_intent != intent or current_intent_sha256 != intent_sha256:
                    raise RasterOfflineBundleBootstrapError(
                        "raster_bundle_unselected_release_source_drift"
                    )
                return reused
        except RasterOfflineBundleBootstrapError:
            raise
        except RasterOfflineBundleError as error:
            raise RasterOfflineBundleBootstrapError(error.code) from None

        token = uuid.uuid4().hex
        stage_canonical = operational_root / (
            f".{_BUNDLE_RELEASE_COLLECTION_DIRECTORY}.release-staging-{token}"
        )
        workspace_canonical = operational_root / (
            f".{_BUNDLE_RELEASE_COLLECTION_DIRECTORY}.release-work-{token}"
        )
        try:
            _prepare_release_stage(
                base=base,
                stage=stage_canonical,
                recognition_model_snapshot=recognition_model_snapshot,
                raster_text_model_snapshot=raster_text_model_snapshot,
                raster_text_derivation_intent_sha256=intent_sha256,
            )
            verified = verify_staged_raster_offline_bundle(
                document_preview_root=root, bundle_root=stage_canonical
            )
            if (
                verified.manifest_sha256 == base.manifest_sha256
                or _release_requires_derivation(verified)
                or not _raster_text_direct_model_material_matches_source(
                    verified, expected_snapshot=raster_text_model_snapshot
                )
            ):
                # A source/configuration change must be visible in the
                # exhaustive manifest; a second comparison catches a source
                # tree that changed while its private stage was built.
                raise RasterOfflineBundleBootstrapError(
                    "raster_bundle_unselected_release_source_drift"
                )
            workspace_canonical.mkdir(mode=0o700)
            stage = _windows_operational_path(stage_canonical, force_short=True)
            workspace = _windows_operational_path(workspace_canonical, force_short=True)
            manifest = verified.manifest
            _sidecar_smoke(
                stage=stage,
                workspace=workspace,
                interpreter=stage.joinpath(
                    *PurePosixPath(manifest.sidecar_interpreter_relative_path).parts
                ),
                module_root=stage.joinpath(
                    *PurePosixPath(manifest.sidecar_module_root_relative_path).parts
                ),
                sidecar_path=stage.joinpath(
                    *PurePosixPath(manifest.sidecar_module_relative_path).parts
                ),
                package_inventory_path=stage.joinpath(
                    *PurePosixPath(manifest.sidecar_package_inventory_relative_path).parts
                ),
                model_inventory_path=stage.joinpath(
                    *PurePosixPath(manifest.model_inventory_relative_path).parts
                ),
                models_root=stage.joinpath(
                    *PurePosixPath(manifest.local_model_root_relative_path).parts
                ),
            )
            verified = verify_staged_raster_offline_bundle(
                document_preview_root=root, bundle_root=stage_canonical
            )
            if _release_requires_derivation(
                verified
            ) or not _raster_text_direct_model_material_matches_source(
                verified, expected_snapshot=raster_text_model_snapshot
            ):
                raise RasterOfflineBundleBootstrapError(
                    "raster_bundle_unselected_release_source_drift"
                )
            current_intent, current_intent_sha256 = _raster_text_release_derivation_intent(
                base=base,
                recognition_model_snapshot=_verify_release_reuse_compatibility(base),
                raster_text_model_snapshot=_snapshot_raster_text_direct_model_source(base),
            )
            if current_intent != intent or current_intent_sha256 != intent_sha256:
                raise RasterOfflineBundleBootstrapError(
                    "raster_bundle_unselected_release_source_drift"
                )
            published = _publish_unselected_release_from_stage(
                root=root,
                operational_root=operational_root,
                stage=stage_canonical,
                manifest_sha256=verified.manifest_sha256,
            )
            verify_parent_renderer_inventory(published)
            _verify_release_reuse_compatibility(published)
            if _release_requires_derivation(
                published
            ) or not _raster_text_direct_model_material_matches_source(
                published, expected_snapshot=raster_text_model_snapshot
            ):
                raise RasterOfflineBundleBootstrapError(
                    "raster_bundle_unselected_release_source_drift"
                )
            _verify_raster_text_release_derivation_receipt(
                candidate=published,
                intent=intent,
                intent_sha256=intent_sha256,
            )
            _write_or_verify_raster_text_release_intent_index(
                root=root,
                operational_root=operational_root,
                intent=intent,
                intent_sha256=intent_sha256,
                published=published,
            )
            return published
        except RasterOfflineBundleBootstrapError:
            raise
        except RasterOfflineBundleError as error:
            raise RasterOfflineBundleBootstrapError(error.code) from None
        except (OSError, ValueError):
            raise RasterOfflineBundleBootstrapError(
                "raster_bundle_unselected_release_failed"
            ) from None
        finally:
            if stage_canonical.exists():
                _remove_owned_tree(
                    root,
                    stage_canonical,
                    prefix=f".{_BUNDLE_RELEASE_COLLECTION_DIRECTORY}.release-staging-",
                )
            if workspace_canonical.exists():
                _remove_owned_tree(
                    root,
                    workspace_canonical,
                    prefix=f".{_BUNDLE_RELEASE_COLLECTION_DIRECTORY}.release-work-",
                )


def ensure_raster_offline_bundle_release_v1(
    *, document_preview_root: Path
) -> VerifiedRasterOfflineBundleV1:
    """Select or derive a local immutable release without downloading bytes.

    It is deliberately separate from the online provisioning helper.  When
    the current first-party sidecar/configuration changes, it copies only
    from a fully verified local base bundle, keeps that base untouched, and
    atomically selects the resulting content-addressed release.  A malformed
    existing selector is never repaired by silently choosing the legacy tree.
    """

    root = _document_preview_root(document_preview_root)
    operational_root = _windows_operational_path(root)
    with _bootstrap_lock(root):
        try:
            base = load_verified_raster_offline_bundle(root)
        except RasterOfflineBundleError as error:
            # New safe engine/detector fields are deliberately required at
            # normal runtime resolution.  The one migration exception is a
            # fixed legacy root that lacks the *complete* field set; its
            # bytes are reverified, copied into a new release, replaced with
            # the current template, smoked, and reverified before publish.
            # A selector/release is never permitted to fall back this way.
            if error.code != "offline_bundle_manifest_invalid":
                raise RasterOfflineBundleBootstrapError(error.code) from None
            try:
                base = load_verified_legacy_raster_offline_bundle_for_safe_configuration_migration(
                    root
                )
            except RasterOfflineBundleError as migration_error:
                raise RasterOfflineBundleBootstrapError(migration_error.code) from None
        # A new automatic profile/pointer must bind a content-addressed
        # release, never the legacy fixed directory.  Therefore the first
        # invocation promotes even byte-identical legacy material into a
        # release.  Once selected, a release is reused until first-party
        # sidecar/configuration bytes actually change.
        if base.bundle_root_relative_path != _BUNDLE_DIRECTORY and not _release_requires_derivation(
            base
        ):
            return base
        recognition_model_snapshot = _verify_release_reuse_compatibility(base)
        token = uuid.uuid4().hex
        stage_canonical = operational_root / (
            f".{_BUNDLE_RELEASE_COLLECTION_DIRECTORY}.release-staging-{token}"
        )
        workspace_canonical = operational_root / (
            f".{_BUNDLE_RELEASE_COLLECTION_DIRECTORY}.release-work-{token}"
        )
        try:
            _prepare_release_stage(
                base=base,
                stage=stage_canonical,
                recognition_model_snapshot=recognition_model_snapshot,
            )
            verified = verify_staged_raster_offline_bundle(
                document_preview_root=root, bundle_root=stage_canonical
            )
            if verified.manifest_sha256 == base.manifest_sha256:
                # A source/config difference must change the exhaustive
                # manifest.  Otherwise a staging implementation is broken.
                raise RasterOfflineBundleBootstrapError("raster_bundle_release_stage_invalid")
            workspace_canonical.mkdir(mode=0o700)
            stage = _windows_operational_path(stage_canonical, force_short=True)
            workspace = _windows_operational_path(workspace_canonical, force_short=True)
            manifest = verified.manifest
            _sidecar_smoke(
                stage=stage,
                workspace=workspace,
                interpreter=stage.joinpath(
                    *PurePosixPath(manifest.sidecar_interpreter_relative_path).parts
                ),
                module_root=stage.joinpath(
                    *PurePosixPath(manifest.sidecar_module_root_relative_path).parts
                ),
                sidecar_path=stage.joinpath(
                    *PurePosixPath(manifest.sidecar_module_relative_path).parts
                ),
                package_inventory_path=stage.joinpath(
                    *PurePosixPath(manifest.sidecar_package_inventory_relative_path).parts
                ),
                model_inventory_path=stage.joinpath(
                    *PurePosixPath(manifest.model_inventory_relative_path).parts
                ),
                models_root=stage.joinpath(
                    *PurePosixPath(manifest.local_model_root_relative_path).parts
                ),
            )
            # The child is launched with ``-B`` and an isolated home, so this
            # second exhaustive check should be byte-identical.  It makes any
            # accidental stage write during smoke a fail-closed publish stop.
            verified = verify_staged_raster_offline_bundle(
                document_preview_root=root, bundle_root=stage_canonical
            )
            _publish_release_from_stage(
                root=root,
                operational_root=operational_root,
                stage=stage_canonical,
                manifest_sha256=verified.manifest_sha256,
            )
            selected = load_verified_raster_offline_bundle(root)
            if (
                selected.manifest_sha256 != verified.manifest_sha256
                or selected.bundle_root_relative_path
                != _release_relative_path(verified.manifest_sha256)
            ):
                raise RasterOfflineBundleBootstrapError("raster_bundle_release_selector_invalid")
            return selected
        except RasterOfflineBundleBootstrapError:
            raise
        except RasterOfflineBundleError as error:
            raise RasterOfflineBundleBootstrapError(error.code) from None
        except (OSError, ValueError):
            raise RasterOfflineBundleBootstrapError("raster_bundle_release_failed") from None
        finally:
            if stage_canonical.exists():
                _remove_owned_tree(
                    root,
                    stage_canonical,
                    prefix=f".{_BUNDLE_RELEASE_COLLECTION_DIRECTORY}.release-staging-",
                )
            if workspace_canonical.exists():
                _remove_owned_tree(
                    root,
                    workspace_canonical,
                    prefix=f".{_BUNDLE_RELEASE_COLLECTION_DIRECTORY}.release-work-",
                )


def bootstrap_raster_offline_bundle_v1(
    *, document_preview_root: Path
) -> RasterOfflineBundleBootstrapResultV1:
    """Create or verify the fixed raster sidecar bundle without human input.

    A valid existing bundle is reused byte-for-byte.  An invalid/different
    existing final directory is never overwritten.  Download and extraction
    occur only in a private sibling staging directory; the final directory is
    created by one same-volume rename after the isolated sidecar has run.
    """

    _ensure_supported_platform()
    root = _document_preview_root(document_preview_root)
    operational_root = _windows_operational_path(root)
    try:
        existing = load_verified_raster_offline_bundle(root)
    except RasterOfflineBundleError as error:
        if error.code != "offline_bundle_unavailable":
            raise RasterOfflineBundleBootstrapError("raster_bundle_existing_invalid") from None
    else:
        try:
            verify_parent_renderer_inventory(existing)
        except RasterOfflineBundleError as error:
            raise RasterOfflineBundleBootstrapError(error.code) from None
        return RasterOfflineBundleBootstrapResultV1(
            created=False,
            manifest_sha256=existing.manifest_sha256,
            raster_profile_revision=existing.raster_profile_revision,
        )
    _required_parent_inventory()
    final = operational_root / _BUNDLE_DIRECTORY
    # Keep canonical long paths for ownership/publish checks and short aliases
    # only for the deep interpreter/package/model filesystem operations.
    stage_canonical: Path | None = None
    workspace_canonical: Path | None = None
    stage: Path | None = None
    workspace: Path | None = None
    with _bootstrap_lock(root):
        # A concurrent successful bootstrap may have published while this
        # process waited for the lock.  Reuse it only after its full verifier.
        try:
            existing = load_verified_raster_offline_bundle(root)
        except RasterOfflineBundleError as error:
            if error.code != "offline_bundle_unavailable":
                raise RasterOfflineBundleBootstrapError("raster_bundle_existing_invalid") from None
        else:
            verify_parent_renderer_inventory(existing)
            return RasterOfflineBundleBootstrapResultV1(
                created=False,
                manifest_sha256=existing.manifest_sha256,
                raster_profile_revision=existing.raster_profile_revision,
            )
        token = uuid.uuid4().hex
        stage_canonical = operational_root / f".{_BUNDLE_DIRECTORY}.staging-{token}"
        workspace_canonical = operational_root / f".{_BUNDLE_DIRECTORY}.work-{token}"
        try:
            stage_canonical.mkdir(mode=0o700)
            workspace_canonical.mkdir(mode=0o700)
            # The staging name itself is long even though it is ASCII.  Force
            # an 8.3 spelling after mkdir so copied CPython/Paddle dependency
            # trees remain below legacy Win32 path limits.
            stage = _windows_operational_path(stage_canonical, force_short=True)
            workspace = _windows_operational_path(workspace_canonical, force_short=True)
            interpreter_root = stage / "sidecar" / "interpreter"
            module_root = stage / "sidecar" / "module-root"
            _copy_interpreter(destination=interpreter_root)
            interpreter = interpreter_root / "python.exe"
            module_root.mkdir(parents=True, exist_ok=False)
            _install_sidecar_packages(interpreter=interpreter, module_root=module_root)
            sidecar = _copy_first_party_sidecar(module_root=module_root)
            package_inventory = _sidecar_package_inventory(module_root)
            model_receipts = _download_models(models_root=stage / "models", workspace=workspace)
            _write_bootstrap_material(
                stage=stage,
                model_receipts=model_receipts,
                package_inventory=package_inventory,
            )
            _sidecar_smoke(
                stage=stage,
                workspace=workspace,
                interpreter=interpreter,
                module_root=module_root,
                sidecar_path=sidecar,
                package_inventory_path=stage / "inventories" / "sidecar-packages.json",
                model_inventory_path=stage / "inventories" / "models.json",
                models_root=stage / "models",
            )
            _write_manifest(stage=stage)
            verified = verify_staged_raster_offline_bundle(
                document_preview_root=root, bundle_root=stage
            )
            verify_parent_renderer_inventory(verified)
            if final.exists() or final.is_symlink():
                raise RasterOfflineBundleBootstrapError("raster_bundle_publish_conflict")
            # ``rename`` never replaces a destination on Windows.  The
            # staging directory is a same-volume sibling, so this is the only
            # irreversible state transition and it is atomic to readers.
            os.rename(stage, final)
            stage = None
            published = load_verified_raster_offline_bundle(root)
            verify_parent_renderer_inventory(published)
            return RasterOfflineBundleBootstrapResultV1(
                created=True,
                manifest_sha256=published.manifest_sha256,
                raster_profile_revision=published.raster_profile_revision,
            )
        except RasterOfflineBundleBootstrapError:
            raise
        except RasterOfflineBundleError as error:
            raise RasterOfflineBundleBootstrapError(error.code) from None
        except (OSError, ValueError):
            raise RasterOfflineBundleBootstrapError("raster_bundle_bootstrap_failed") from None
        finally:
            if stage_canonical is not None and stage_canonical.exists():
                _remove_owned_tree(root, stage_canonical, prefix=f".{_BUNDLE_DIRECTORY}.staging-")
            if workspace_canonical is not None and workspace_canonical.exists():
                _remove_owned_tree(root, workspace_canonical, prefix=f".{_BUNDLE_DIRECTORY}.work-")


__all__ = [
    "RasterOfflineBundleBootstrapError",
    "RasterOfflineBundleBootstrapResultV1",
    "bootstrap_raster_offline_bundle_v1",
    "ensure_raster_offline_bundle_release_v1",
    "materialize_verified_unselected_raster_text_release_v1",
    "resolve_or_materialize_verified_raster_text_release_reference_v1",
]
