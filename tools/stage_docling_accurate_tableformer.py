"""Stage the one pinned Docling Accurate TableFormer model pair locally.

This is intentionally narrower than the general runtime provisioner.  It may
contact only the two immutable Hugging Face ``/resolve/<commit>/`` URLs below,
and only after an operator supplies ``--confirm-download``.  It does not
write a runtime configuration, model inventory, candidate, evaluation, or
activation artifact.  The existing local-only inventory builder and runtime
provisioner remain separate required steps after this command succeeds.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import stat
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Final, Sequence
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from customs_ai_agent.service.document_preview_runtime import (
    DocumentPreviewRuntimeError,
    validate_document_preview_root,
)


_DEFAULT_MINIMUM_FREE_GIB: Final = 12.0
_DOWNLOAD_CHUNK_BYTES: Final = 1024 * 1024
_HF_HOST: Final = "huggingface.co"
_HF_REPOSITORY: Final = "ds4sd/docling-models"
_HF_CANONICAL_REPOSITORY: Final = "docling-project/docling-models"
_HF_SNAPSHOT: Final = "36bebf56681740529abd09f5473a93a69373fbf0"
_TABLEFORMER_DIRECTORY: Final = "model_artifacts/tableformer/accurate"
_TARGET_RELATIVE_PARTS: Final = (
    "models",
    "docling",
    "model_artifacts",
    "tableformer",
    "accurate",
)
_STAGING_DIRECTORY: Final = "staging"
_PUBLISH_LOCK_FILENAME: Final = ".docling-accurate-tableformer.lock"


@dataclass(frozen=True, slots=True)
class _PinnedArtifact:
    filename: str
    source_uri: str
    byte_size: int
    sha256: str | None = None
    git_blob_sha1: str | None = None


def _source_uri(filename: str) -> str:
    return (
        f"https://{_HF_HOST}/{_HF_REPOSITORY}/resolve/{_HF_SNAPSHOT}/"
        f"{_TABLEFORMER_DIRECTORY}/{filename}"
    )


_ARTIFACTS: Final[tuple[_PinnedArtifact, ...]] = (
    _PinnedArtifact(
        filename="tm_config.json",
        source_uri=_source_uri("tm_config.json"),
        byte_size=7060,
        # Hugging Face's exact snapshot header is the Git blob object id, not
        # a raw-content SHA-1.  Verify the Git ``blob <size>\0`` prefix too.
        git_blob_sha1="5036aa114b9393f607440cb0f6cc54229b5b30e1",
    ),
    _PinnedArtifact(
        filename="tableformer_accurate.safetensors",
        source_uri=_source_uri("tableformer_accurate.safetensors"),
        byte_size=212_758_388,
        sha256="66d8912f290375d3466f91be2048030a16317e84c8f1f69d3dbd7adc6d6cd2a9",
    ),
)
_ALLOWED_SOURCE_URIS: Final[frozenset[str]] = frozenset(
    artifact.source_uri for artifact in _ARTIFACTS
)


def _check_free_space(path: Path, minimum_free_gib: float) -> None:
    if minimum_free_gib <= 0 or minimum_free_gib > 1024:
        raise DocumentPreviewRuntimeError("document_preview_docling_accurate_space_budget_invalid")
    try:
        free_bytes = shutil.disk_usage(path).free
    except OSError as error:
        raise DocumentPreviewRuntimeError(
            "document_preview_docling_accurate_space_check_failed"
        ) from error
    if free_bytes < int(minimum_free_gib * 1024**3):
        raise DocumentPreviewRuntimeError("document_preview_docling_accurate_space_insufficient")


def _is_link_or_reparse_point(metadata: os.stat_result) -> bool:
    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(metadata, "st_file_attributes", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(reparse_point and attributes & reparse_point)


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _path_under_root(*, root: Path, path: Path, must_exist: bool, code: str) -> Path:
    """Return an in-root path only when no existing component redirects it."""

    candidate = _absolute_path(path)
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise DocumentPreviewRuntimeError(code) from error
    current = root
    for part in relative.parts:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if must_exist:
                raise DocumentPreviewRuntimeError(code) from None
            break
        except OSError as error:
            raise DocumentPreviewRuntimeError(code) from error
        if _is_link_or_reparse_point(metadata):
            raise DocumentPreviewRuntimeError(code)
    try:
        resolved = candidate.resolve(strict=must_exist)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise DocumentPreviewRuntimeError(code) from error
    return candidate


def _ensure_safe_directory(*, root: Path, path: Path, code: str) -> Path:
    candidate = _path_under_root(root=root, path=path, must_exist=False, code=code)
    try:
        candidate.mkdir(parents=True, exist_ok=True)
    except OSError as error:
        raise DocumentPreviewRuntimeError(code) from error
    checked = _path_under_root(root=root, path=candidate, must_exist=True, code=code)
    try:
        metadata = checked.lstat()
    except OSError as error:
        raise DocumentPreviewRuntimeError(code) from error
    if _is_link_or_reparse_point(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise DocumentPreviewRuntimeError(code)
    return checked


def _assert_absent(*, root: Path, path: Path, code: str) -> None:
    candidate = _path_under_root(root=root, path=path, must_exist=False, code=code)
    try:
        candidate.lstat()
    except FileNotFoundError:
        return
    except OSError as error:
        raise DocumentPreviewRuntimeError(code) from error
    raise DocumentPreviewRuntimeError(code)


def _assert_exact_source_uri(uri: str) -> None:
    """Reject URL changes before a network request is even attempted."""

    parsed = urlparse(uri)
    if (
        uri not in _ALLOWED_SOURCE_URIS
        or parsed.scheme != "https"
        or parsed.hostname != _HF_HOST
        or parsed.username is not None
        or parsed.password is not None
        or parsed.query
        or parsed.fragment
    ):
        raise DocumentPreviewRuntimeError("document_preview_docling_accurate_source_invalid")


def _is_allowed_final_download_url(value: str) -> bool:
    """Allow only HTTPS Hugging Face resolver/cache or its signed Xet CDN.

    The initial URL is exact and immutable.  The resolver can then redirect a
    large file to a short-lived, signed Xet URL, whose host is not stable; the
    weight's required SHA-256 still authenticates the final bytes.
    """

    parsed = urlparse(value)
    hostname = parsed.hostname or ""
    if parsed.scheme != "https" or parsed.username is not None or parsed.password is not None:
        return False
    return (
        hostname == _HF_HOST
        or hostname.endswith(".huggingface.co")
        or hostname.endswith(".cdn.hf.co")
        or hostname.endswith(".xethub.hf.co")
    )


def _response_content_length(response: Any, *, expected: int) -> None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return
    raw_value = headers.get("Content-Length")
    if raw_value is None:
        return
    if not isinstance(raw_value, str) or not raw_value.isdecimal() or int(raw_value) != expected:
        raise DocumentPreviewRuntimeError("document_preview_docling_accurate_download_invalid")


def _validate_downloaded_config(path: Path) -> None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DocumentPreviewRuntimeError(
            "document_preview_docling_accurate_download_invalid"
        ) from error
    if not isinstance(value, dict):
        raise DocumentPreviewRuntimeError("document_preview_docling_accurate_download_invalid")


def _download_pinned_artifact(
    *,
    artifact: _PinnedArtifact,
    destination: Path,
    opener: Callable[..., Any],
) -> None:
    _assert_exact_source_uri(artifact.source_uri)
    request = Request(artifact.source_uri, headers={"Accept-Encoding": "identity"})
    try:
        response_context = opener(request, timeout=30)
        response = response_context.__enter__()
    except Exception as error:
        raise DocumentPreviewRuntimeError(
            "document_preview_docling_accurate_download_failed"
        ) from error
    try:
        final_url = response.geturl()
        if not isinstance(final_url, str) or not _is_allowed_final_download_url(final_url):
            raise DocumentPreviewRuntimeError("document_preview_docling_accurate_download_invalid")
        _response_content_length(response, expected=artifact.byte_size)
        sha256 = hashlib.sha256()
        git_blob_sha1 = hashlib.sha1()
        git_blob_sha1.update(f"blob {artifact.byte_size}\0".encode("ascii"))
        try:
            with destination.open("xb") as stream:
                remaining = artifact.byte_size
                while remaining:
                    chunk = response.read(min(_DOWNLOAD_CHUNK_BYTES, remaining))
                    if not isinstance(chunk, bytes) or not chunk:
                        raise DocumentPreviewRuntimeError(
                            "document_preview_docling_accurate_download_invalid"
                        )
                    stream.write(chunk)
                    sha256.update(chunk)
                    git_blob_sha1.update(chunk)
                    remaining -= len(chunk)
                if response.read(1):
                    raise DocumentPreviewRuntimeError(
                        "document_preview_docling_accurate_download_invalid"
                    )
                stream.flush()
                os.fsync(stream.fileno())
        except DocumentPreviewRuntimeError:
            raise
        except OSError as error:
            raise DocumentPreviewRuntimeError(
                "document_preview_docling_accurate_download_failed"
            ) from error
        try:
            metadata = destination.lstat()
        except OSError as error:
            raise DocumentPreviewRuntimeError(
                "document_preview_docling_accurate_download_invalid"
            ) from error
        if (
            _is_link_or_reparse_point(metadata)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size != artifact.byte_size
            or (artifact.sha256 is not None and sha256.hexdigest() != artifact.sha256)
            or (
                artifact.git_blob_sha1 is not None
                and git_blob_sha1.hexdigest() != artifact.git_blob_sha1
            )
        ):
            raise DocumentPreviewRuntimeError("document_preview_docling_accurate_download_invalid")
        if artifact.filename == "tm_config.json":
            _validate_downloaded_config(destination)
    finally:
        try:
            response_context.__exit__(None, None, None)
        except Exception:
            # The asset was already fully verified; a close failure must not
            # turn an immutable successful staging result into an overwrite.
            pass


def _assert_exact_staging_files(*, stage: Path) -> None:
    try:
        children = {child.name: child for child in stage.iterdir()}
    except OSError as error:
        raise DocumentPreviewRuntimeError(
            "document_preview_docling_accurate_staging_invalid"
        ) from error
    expected = {artifact.filename for artifact in _ARTIFACTS}
    if set(children) != expected:
        raise DocumentPreviewRuntimeError("document_preview_docling_accurate_staging_invalid")
    for artifact in _ARTIFACTS:
        path = children[artifact.filename]
        try:
            metadata = path.lstat()
        except OSError as error:
            raise DocumentPreviewRuntimeError(
                "document_preview_docling_accurate_staging_invalid"
            ) from error
        if (
            _is_link_or_reparse_point(metadata)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size != artifact.byte_size
        ):
            raise DocumentPreviewRuntimeError("document_preview_docling_accurate_staging_invalid")


def _acquire_publish_lock(*, root: Path, parent: Path) -> Path:
    lock = parent / _PUBLISH_LOCK_FILENAME
    _path_under_root(
        root=root,
        path=lock,
        must_exist=False,
        code="document_preview_docling_accurate_publish_locked",
    )
    try:
        with lock.open("x", encoding="ascii") as stream:
            stream.write("docling-accurate-tableformer\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise DocumentPreviewRuntimeError(
            "document_preview_docling_accurate_publish_locked"
        ) from error
    except OSError as error:
        raise DocumentPreviewRuntimeError(
            "document_preview_docling_accurate_publish_failed"
        ) from error
    return lock


def _release_publish_lock(lock: Path) -> None:
    try:
        metadata = lock.lstat()
        if _is_link_or_reparse_point(metadata) or not stat.S_ISREG(metadata.st_mode):
            return
        lock.unlink()
    except FileNotFoundError:
        return
    except OSError:
        return


def _remove_stage(stage: Path) -> None:
    """Remove only the helper-created regular-file staging directory."""

    try:
        for child in stage.iterdir():
            metadata = child.lstat()
            if _is_link_or_reparse_point(metadata) or not stat.S_ISREG(metadata.st_mode):
                return
            child.unlink()
        stage.rmdir()
    except FileNotFoundError:
        return
    except OSError:
        return


def _publish_stage(*, root: Path, stage: Path, target: Path) -> None:
    parent = _ensure_safe_directory(
        root=root,
        path=target.parent,
        code="document_preview_docling_accurate_publish_failed",
    )
    lock = _acquire_publish_lock(root=root, parent=parent)
    try:
        _assert_absent(
            root=root,
            path=target,
            code="document_preview_docling_accurate_target_exists",
        )
        _assert_exact_staging_files(stage=stage)
        try:
            # Stage and target are under the same local root, so this is a
            # directory rename rather than a cross-volume copy.  On Windows
            # it fails rather than replacing an existing target directory.
            os.rename(stage, target)
        except FileExistsError as error:
            raise DocumentPreviewRuntimeError(
                "document_preview_docling_accurate_target_exists"
            ) from error
        except OSError as error:
            raise DocumentPreviewRuntimeError(
                "document_preview_docling_accurate_publish_failed"
            ) from error
    finally:
        _release_publish_lock(lock)


def stage_docling_accurate_tableformer(
    *,
    document_preview_root: Path,
    minimum_free_gib: float = _DEFAULT_MINIMUM_FREE_GIB,
    opener: Callable[..., Any] = urlopen,
) -> None:
    """Download, verify, and atomically publish only the exact Accurate pair."""

    root = validate_document_preview_root(document_preview_root, create=True)
    _check_free_space(root, minimum_free_gib)
    staging_root = _ensure_safe_directory(
        root=root,
        path=root / _STAGING_DIRECTORY,
        code="document_preview_docling_accurate_staging_invalid",
    )
    target = root.joinpath(*_TARGET_RELATIVE_PARTS)
    _assert_absent(root=root, path=target, code="document_preview_docling_accurate_target_exists")
    try:
        stage = Path(tempfile.mkdtemp(prefix=".docling-accurate-", dir=staging_root))
    except OSError as error:
        raise DocumentPreviewRuntimeError(
            "document_preview_docling_accurate_staging_invalid"
        ) from error
    try:
        stage = _path_under_root(
            root=root,
            path=stage,
            must_exist=True,
            code="document_preview_docling_accurate_staging_invalid",
        )
        for artifact in _ARTIFACTS:
            destination = stage / artifact.filename
            _path_under_root(
                root=root,
                path=destination,
                must_exist=False,
                code="document_preview_docling_accurate_staging_invalid",
            )
            _download_pinned_artifact(artifact=artifact, destination=destination, opener=opener)
        _publish_stage(root=root, stage=stage, target=target)
    except Exception:
        _remove_stage(stage)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and verify only the pinned Docling Accurate TableFormer pair."
    )
    parser.add_argument("--document-preview-root", type=Path, required=True)
    parser.add_argument(
        "--minimum-free-gib",
        type=float,
        default=_DEFAULT_MINIMUM_FREE_GIB,
        help="Required free space on the document-preview volume (default: 12 GiB).",
    )
    parser.add_argument(
        "--confirm-download",
        action="store_true",
        help="Required acknowledgement before the two pinned HTTPS artifacts are downloaded.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if not arguments.confirm_download:
        print(
            json.dumps({"status": "failed", "code": "confirmation_required"}, separators=(",", ":"))
        )
        return 2
    try:
        stage_docling_accurate_tableformer(
            document_preview_root=arguments.document_preview_root,
            minimum_free_gib=arguments.minimum_free_gib,
        )
    except DocumentPreviewRuntimeError as error:
        print(json.dumps({"status": "failed", "code": error.code}, separators=(",", ":")))
        return 2
    print(
        json.dumps(
            {
                "status": "staged",
                "engine_kind": "docling",
                "table_structure_mode": "accurate",
                "artifact_count": len(_ARTIFACTS),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
