"""Offline runtime orchestration for source-attested recognizer training.

The model/training implementation is deliberately kept in its separately
attested module.  This module owns only the one-shot process boundary:
release discovery, the OS lock, offline Paddle bundle verification, native
import provenance, network denial, and the bounded overfit invocation.
"""

from __future__ import annotations

from contextlib import contextmanager
import importlib
import os
from pathlib import Path, PurePosixPath
import re
import socket
import sys
from typing import Iterator

from customs_ai_agent.evaluation import (
    source_attested_raster_text_recognizer_training as training_implementation,
)
from customs_ai_agent.evaluation.local_process_lock import (
    LocalProcessLockBusy,
    LocalProcessLockInvalid,
    exclusive_local_process_lock,
)
from customs_ai_agent.evaluation.raster_text_direct_paddle_bundle import (
    RasterTextDirectPaddleBundleError,
    VerifiedRasterTextDirectPaddleBundleV1,
    _operational_path,
    load_verified_raster_text_direct_paddle_bundle_v1,
    materialize_verified_raster_text_direct_paddle_bundle_v1,
)
from customs_ai_agent.evaluation.source_attested_raster_text_recognizer_training import (
    SourceAttestedRecognizerTrainingError,
    TrainingRunResult,
    VerifiedRecognizerRelease,
    load_verified_source_attested_raster_text_recognizer_release,
    sha256_file,
    train_source_attested_raster_text_recognizer_overfit,
)


_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_DOCUMENT_PREVIEW_ROOT_SUFFIX = ("_local", "data", "document-preview")
_RELEASES_RELATIVE_PATH = "raster-text-trained-recognizers-v1/releases"
_NATIVE_TRAINING_MODULES = ("paddle", "numpy", "PIL")
_NATIVE_SUPPORT_MODULES = ("_distutils_hack", "setuptools")
_LOCK_FILE = ".source-attested-raster-text-recognizer-training.lock"


def _verified_document_preview_root(path: Path) -> Path:
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
        raise SourceAttestedRecognizerTrainingError(
            "training_document_preview_root_invalid"
        ) from None


def _under(root: Path, candidate: Path) -> bool:
    try:
        candidate.resolve(strict=True).relative_to(root.resolve(strict=True))
        return True
    except (OSError, ValueError):
        return False


def _verified_operational_module_root(
    bundle: VerifiedRasterTextDirectPaddleBundleV1,
) -> tuple[Path, Path]:
    manifest = bundle.manifest
    canonical_module_root = bundle.bundle_root.joinpath(
        *PurePosixPath(manifest.sidecar_module_root_relative_path).parts
    ).resolve(strict=True)
    if canonical_module_root.is_symlink() or not canonical_module_root.is_dir():
        raise SourceAttestedRecognizerTrainingError("training_runtime_bundle_invalid")
    try:
        operational_module_root = _operational_path(canonical_module_root, force_short=True)
    except RasterTextDirectPaddleBundleError:
        raise SourceAttestedRecognizerTrainingError("training_runtime_path_unsupported") from None
    if os.name == "nt" and (
        not str(operational_module_root).isascii()
        or operational_module_root == canonical_module_root
    ):
        raise SourceAttestedRecognizerTrainingError("training_runtime_path_unsupported")
    return canonical_module_root, operational_module_root


def _verified_operational_interpreter_packages(
    bundle: VerifiedRasterTextDirectPaddleBundleV1,
) -> tuple[Path, Path]:
    manifest = bundle.manifest
    canonical = bundle.bundle_root.joinpath(
        *PurePosixPath(manifest.sidecar_interpreter_root_relative_path).parts,
        "Lib",
        "site-packages",
    ).resolve(strict=True)
    if canonical.is_symlink() or not canonical.is_dir():
        raise SourceAttestedRecognizerTrainingError("training_runtime_bundle_invalid")
    try:
        operational = _operational_path(canonical, force_short=True)
    except RasterTextDirectPaddleBundleError:
        raise SourceAttestedRecognizerTrainingError("training_runtime_path_unsupported") from None
    if os.name == "nt" and (not str(operational).isascii() or operational == canonical):
        raise SourceAttestedRecognizerTrainingError("training_runtime_path_unsupported")
    return canonical, operational


@contextmanager
def verified_direct_training_imports(
    bundle: VerifiedRasterTextDirectPaddleBundleV1,
) -> Iterator[Path]:
    """Import native packages only from the exact attested offline bundle."""

    manifest = bundle.manifest
    canonical_module_root, operational_module_root = _verified_operational_module_root(bundle)
    canonical_support_root, operational_support_root = _verified_operational_interpreter_packages(
        bundle
    )
    if any(name in sys.modules for name in (*_NATIVE_TRAINING_MODULES, *_NATIVE_SUPPORT_MODULES)):
        raise SourceAttestedRecognizerTrainingError("training_native_module_preloaded")
    previous_dont_write_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(operational_support_root))
    sys.path.insert(0, str(operational_module_root))
    imported: list[str] = []
    try:
        try:
            for name in _NATIVE_SUPPORT_MODULES:
                module = importlib.import_module(name)
                imported.append(name)
                origin_value = getattr(module, "__file__", None)
                expected_root = (
                    canonical_support_root if name == "_distutils_hack" else canonical_module_root
                )
                if not isinstance(origin_value, str) or not _under(
                    expected_root, Path(origin_value)
                ):
                    raise SourceAttestedRecognizerTrainingError("training_runtime_import_drift")
            for name in _NATIVE_TRAINING_MODULES:
                module = importlib.import_module(name)
                imported.append(name)
                origin_value = getattr(module, "__file__", None)
                if not isinstance(origin_value, str) or not _under(
                    canonical_module_root, Path(origin_value)
                ):
                    raise SourceAttestedRecognizerTrainingError("training_runtime_import_drift")
        except SourceAttestedRecognizerTrainingError:
            raise
        except Exception:
            raise SourceAttestedRecognizerTrainingError("training_runtime_unavailable") from None
        yield bundle.bundle_root.joinpath(
            *PurePosixPath(manifest.sidecar_package_inventory_relative_path).parts
        )
    finally:
        if sys.path and sys.path[0] == str(operational_module_root):
            sys.path.pop(0)
        if sys.path and sys.path[0] == str(operational_support_root):
            sys.path.pop(0)
        sys.dont_write_bytecode = previous_dont_write_bytecode
        if sys.exc_info()[0] is not None:
            for name in imported:
                sys.modules.pop(name, None)


@contextmanager
def network_disabled() -> Iterator[None]:
    """Block DNS plus stream/datagram routes during local materialization/training."""

    original_connect = socket.socket.connect
    original_connect_ex = socket.socket.connect_ex
    original_sendto = socket.socket.sendto
    original_create_connection = socket.create_connection
    original_getaddrinfo = socket.getaddrinfo
    original_gethostbyname = socket.gethostbyname
    original_gethostbyname_ex = socket.gethostbyname_ex

    def blocked(*_args: object, **_kwargs: object) -> None:
        raise OSError("network disabled")

    socket.socket.connect = blocked  # type: ignore[method-assign]
    socket.socket.connect_ex = blocked  # type: ignore[assignment]
    socket.socket.sendto = blocked  # type: ignore[assignment]
    socket.create_connection = blocked  # type: ignore[assignment]
    socket.getaddrinfo = blocked  # type: ignore[assignment]
    socket.gethostbyname = blocked  # type: ignore[assignment]
    socket.gethostbyname_ex = blocked  # type: ignore[assignment]
    try:
        yield
    finally:
        socket.socket.connect = original_connect  # type: ignore[method-assign]
        socket.socket.connect_ex = original_connect_ex  # type: ignore[assignment]
        socket.socket.sendto = original_sendto  # type: ignore[method-assign]
        socket.create_connection = original_create_connection
        socket.getaddrinfo = original_getaddrinfo
        socket.gethostbyname = original_gethostbyname
        socket.gethostbyname_ex = original_gethostbyname_ex


def _verify_release_training_provenance(
    root: Path,
    release: VerifiedRecognizerRelease,
    bundles: dict[str, VerifiedRasterTextDirectPaddleBundleV1],
) -> None:
    """Recheck the local implementation, Python ABI, bundle, and inventory."""

    receipt = release.receipt
    if receipt.parent_python_revision != f"{sys.version_info.major}.{sys.version_info.minor}":
        raise SourceAttestedRecognizerTrainingError("training_release_provenance_drift")
    module_path_value = training_implementation.__file__
    if not isinstance(module_path_value, str):
        raise SourceAttestedRecognizerTrainingError("training_release_provenance_drift")
    try:
        module_path = Path(module_path_value).resolve(strict=True)
    except OSError:
        raise SourceAttestedRecognizerTrainingError("training_release_provenance_drift") from None
    if sha256_file(module_path) != receipt.training_module_sha256:
        raise SourceAttestedRecognizerTrainingError("training_release_provenance_drift")
    manifest_sha256 = receipt.training_runtime_bundle_manifest_sha256
    try:
        bundle = bundles.get(manifest_sha256)
        if bundle is None:
            bundle = load_verified_raster_text_direct_paddle_bundle_v1(
                document_preview_root=root,
                manifest_sha256=manifest_sha256,
            )
            bundles[manifest_sha256] = bundle
        inventory_sha256 = bundle.file_sha256(
            bundle.manifest.sidecar_package_inventory_relative_path
        )
    except RasterTextDirectPaddleBundleError:
        raise SourceAttestedRecognizerTrainingError("training_release_provenance_drift") from None
    if inventory_sha256 != receipt.package_inventory_sha256:
        raise SourceAttestedRecognizerTrainingError("training_release_provenance_drift")


def discover_verified_source_attested_recognizer_release(
    document_preview_root: Path,
) -> VerifiedRecognizerRelease | None:
    """Verify every published release and select one without operator input."""

    root = _verified_document_preview_root(document_preview_root)
    releases = root.joinpath(*PurePosixPath(_RELEASES_RELATIVE_PATH).parts)
    try:
        if releases.is_symlink():
            raise OSError
        if not releases.exists():
            return None
        if not releases.is_dir():
            raise OSError
        candidates = sorted(releases.iterdir(), key=lambda item: item.name)
    except OSError:
        raise SourceAttestedRecognizerTrainingError(
            "recognizer_release_collection_invalid"
        ) from None
    if not candidates:
        return None
    verified: list[VerifiedRecognizerRelease] = []
    bundles: dict[str, VerifiedRasterTextDirectPaddleBundleV1] = {}
    for candidate in candidates:
        if (
            candidate.is_symlink()
            or not candidate.is_dir()
            or _SHA256.fullmatch(candidate.name) is None
        ):
            raise SourceAttestedRecognizerTrainingError("recognizer_release_collection_invalid")
        release = load_verified_source_attested_raster_text_recognizer_release(root, candidate.name)
        _verify_release_training_provenance(root, release, bundles)
        verified.append(release)
    stage_rank = {"full": 0, "pilot": 1, "overfit": 2}
    return min(
        verified,
        key=lambda item: (
            item.receipt.development_cer,
            -item.receipt.development_exact_match,
            stage_rank[item.receipt.stage],
            item.release_sha256,
        ),
    )


def ensure_source_attested_raster_text_recognizer_overfit(
    document_preview_root: Path,
) -> TrainingRunResult:
    """Verify/reuse an existing release or create one bounded overfit release.

    The OS-owned lock covers both the absence check and publication.  No
    candidate/hash/score/checkpoint is accepted from the caller, and a
    malformed existing collection fails closed instead of triggering a new
    model as a fallback.
    """

    root = _verified_document_preview_root(document_preview_root)
    try:
        with exclusive_local_process_lock(root / _LOCK_FILE):
            with network_disabled():
                # Verify every historical release first, but never use its
                # mere presence as a reuse shortcut.  The attested trainer
                # below rebuilds the current source/raster/Gold manifests and
                # reuses a release only when its complete training intent is
                # still byte-identical.
                discover_verified_source_attested_recognizer_release(root)
                try:
                    bundle = materialize_verified_raster_text_direct_paddle_bundle_v1(
                        document_preview_root=root
                    )
                except RasterTextDirectPaddleBundleError:
                    raise SourceAttestedRecognizerTrainingError(
                        "training_runtime_bundle_unavailable"
                    ) from None
                os.environ.setdefault("OMP_NUM_THREADS", "6")
                os.environ.setdefault("FLAGS_json_format_model", "1")
                os.environ.setdefault("FLAGS_selected_gpus", "")
                with verified_direct_training_imports(bundle) as package_inventory_path:
                    result = train_source_attested_raster_text_recognizer_overfit(
                        root,
                        package_inventory_path=package_inventory_path,
                        runtime_bundle_manifest_sha256=bundle.manifest_sha256,
                    )
                _verify_release_training_provenance(
                    root, result.release, {bundle.manifest_sha256: bundle}
                )
                return result
    except LocalProcessLockBusy:
        raise SourceAttestedRecognizerTrainingError("training_locked") from None
    except LocalProcessLockInvalid:
        raise SourceAttestedRecognizerTrainingError("training_lock_invalid") from None
    except SourceAttestedRecognizerTrainingError:
        raise
    except (OSError, ValueError):
        raise SourceAttestedRecognizerTrainingError("training_failed") from None


__all__ = [
    "discover_verified_source_attested_recognizer_release",
    "ensure_source_attested_raster_text_recognizer_overfit",
    "network_disabled",
    "verified_direct_training_imports",
]
