"""Stage only the pinned PaddleOCR 3 mobile detection and recognition assets.

This is deliberately a narrow, explicit download boundary.  It contacts only
the eight immutable Hugging Face ``/resolve/<commit>/`` files declared below,
and only after ``--confirm-download`` is supplied.  It does not install the
PaddleOCR 3/PaddleX runtime and it does not write a runtime configuration,
inventory, candidate, evaluation, approval, or activation artifact.

After a successful stage, an operator must still provision the separate
PaddleOCR 3/PaddleX environment, build an inventory from that environment,
and pass the existing local-only provision and source-attested evaluation
steps.  Staging model files alone never enables a preview profile.
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
from urllib.request import HTTPRedirectHandler, Request, build_opener

from customs_ai_agent.service.document_preview_runtime import (
    DocumentPreviewRuntimeError,
    validate_document_preview_root,
)


_DEFAULT_MINIMUM_FREE_GIB: Final = 12.0
_DOWNLOAD_CHUNK_BYTES: Final = 1024 * 1024
_HF_HOST: Final = "huggingface.co"
_PADDLE_ORGANIZATION: Final = "PaddlePaddle"
_DETECTION_REPOSITORY: Final = "PP-OCRv5_mobile_det"
_RECOGNITION_REPOSITORY: Final = "en_PP-OCRv5_mobile_rec"
_DETECTION_SNAPSHOT: Final = "0d63e78e2b680928f6b1747d76a08db6e645efb7"
_RECOGNITION_SNAPSHOT: Final = "267c36e24c331595590fe7bd72bde2436fd286f2"
_TARGET_RELATIVE_PARTS: Final = ("models", "paddleocr3")
_STAGING_DIRECTORY: Final = "staging"
_PUBLISH_LOCK_FILENAME: Final = ".paddleocr3-mobile-models.lock"
_MODEL_DIRECTORIES: Final = ("det", "rec")


@dataclass(frozen=True, slots=True)
class _PinnedArtifact:
    directory: str
    filename: str
    source_uri: str
    byte_size: int
    sha256: str | None = None
    git_blob_sha1: str | None = None


def _source_uri(*, repository: str, snapshot: str, filename: str) -> str:
    return f"https://{_HF_HOST}/{_PADDLE_ORGANIZATION}/{repository}/resolve/{snapshot}/{filename}"


# The exact byte sizes and digests are from the immutable snapshot resolver's
# X-Linked-ETag metadata.  Hugging Face exposes regular Git files as Git blob
# SHA-1 values and the two Xet-backed parameter files as raw SHA-256 values.
_ARTIFACTS: Final[tuple[_PinnedArtifact, ...]] = (
    _PinnedArtifact(
        directory="det",
        filename="config.json",
        source_uri=_source_uri(
            repository=_DETECTION_REPOSITORY,
            snapshot=_DETECTION_SNAPSHOT,
            filename="config.json",
        ),
        byte_size=2_871,
        git_blob_sha1="4fc190d78e4425094b31a2d3a4744a3623d00b50",
    ),
    _PinnedArtifact(
        directory="det",
        filename="inference.json",
        source_uri=_source_uri(
            repository=_DETECTION_REPOSITORY,
            snapshot=_DETECTION_SNAPSHOT,
            filename="inference.json",
        ),
        byte_size=229_777,
        git_blob_sha1="6cd678f39460a27372f8fe570e4e12e7a383418f",
    ),
    _PinnedArtifact(
        directory="det",
        filename="inference.pdiparams",
        source_uri=_source_uri(
            repository=_DETECTION_REPOSITORY,
            snapshot=_DETECTION_SNAPSHOT,
            filename="inference.pdiparams",
        ),
        byte_size=4_692_937,
        sha256="afa1820cb16c1fd0dad589d0f8b389139061c1ef6d68019685fd07be997dda5b",
    ),
    _PinnedArtifact(
        directory="det",
        filename="inference.yml",
        source_uri=_source_uri(
            repository=_DETECTION_REPOSITORY,
            snapshot=_DETECTION_SNAPSHOT,
            filename="inference.yml",
        ),
        byte_size=903,
        git_blob_sha1="579d10695d2dd6e85c5ecba02d151ae5c077aa49",
    ),
    _PinnedArtifact(
        directory="rec",
        filename="config.json",
        source_uri=_source_uri(
            repository=_RECOGNITION_REPOSITORY,
            snapshot=_RECOGNITION_SNAPSHOT,
            filename="config.json",
        ),
        byte_size=10_455,
        git_blob_sha1="ee30b810753068b251de393eee66dff31f6d279c",
    ),
    _PinnedArtifact(
        directory="rec",
        filename="inference.json",
        source_uri=_source_uri(
            repository=_RECOGNITION_REPOSITORY,
            snapshot=_RECOGNITION_SNAPSHOT,
            filename="inference.json",
        ),
        byte_size=217_712,
        git_blob_sha1="fc20fb935854373220fbeff985cbd310f0450608",
    ),
    _PinnedArtifact(
        directory="rec",
        filename="inference.pdiparams",
        source_uri=_source_uri(
            repository=_RECOGNITION_REPOSITORY,
            snapshot=_RECOGNITION_SNAPSHOT,
            filename="inference.pdiparams",
        ),
        byte_size=7_772_315,
        sha256="3ec8a97ed6cefe8568d3e2ee90bb193299b566a7661aa4fd52d224b96b59f66b",
    ),
    _PinnedArtifact(
        directory="rec",
        filename="inference.yml",
        source_uri=_source_uri(
            repository=_RECOGNITION_REPOSITORY,
            snapshot=_RECOGNITION_SNAPSHOT,
            filename="inference.yml",
        ),
        byte_size=3_964,
        git_blob_sha1="91a401a7220881921c249b92852a96f9dbf2132a",
    ),
)
_ALLOWED_SOURCE_URIS: Final[frozenset[str]] = frozenset(
    artifact.source_uri for artifact in _ARTIFACTS
)
_ALLOWED_DIRECTORIES: Final[frozenset[str]] = frozenset(_MODEL_DIRECTORIES)
_ALLOWED_FILENAMES: Final[frozenset[str]] = frozenset(
    {"config.json", "inference.json", "inference.pdiparams", "inference.yml"}
)


def _check_free_space(path: Path, minimum_free_gib: float) -> None:
    if minimum_free_gib <= 0 or minimum_free_gib > 1_024:
        raise DocumentPreviewRuntimeError("document_preview_paddleocr3_space_budget_invalid")
    try:
        free_bytes = shutil.disk_usage(path).free
    except OSError as error:
        raise DocumentPreviewRuntimeError(
            "document_preview_paddleocr3_space_check_failed"
        ) from error
    if free_bytes < int(minimum_free_gib * 1024**3):
        raise DocumentPreviewRuntimeError("document_preview_paddleocr3_space_insufficient")


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
    """Reject URL drift before any request is opened."""

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
        raise DocumentPreviewRuntimeError("document_preview_paddleocr3_source_invalid")


def _is_allowed_final_download_url(value: str) -> bool:
    """Allow only Hugging Face's HTTPS resolver/cache and signed Xet hosts."""

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


class _PinnedRedirectHandler(HTTPRedirectHandler):
    """Reject an unsafe redirect at every hop instead of only after download."""

    def redirect_request(
        self,
        req: Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> Request | None:
        if not _is_allowed_final_download_url(newurl):
            raise DocumentPreviewRuntimeError("document_preview_paddleocr3_download_invalid")
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def _open_pinned_source(request: Request, *, timeout: int) -> Any:
    return build_opener(_PinnedRedirectHandler()).open(request, timeout=timeout)


def _response_content_length(response: Any, *, expected: int) -> None:
    headers = getattr(response, "headers", None)
    if headers is None:
        return
    raw_value = headers.get("Content-Length")
    if raw_value is None:
        return
    if not isinstance(raw_value, str) or not raw_value.isdecimal() or int(raw_value) != expected:
        raise DocumentPreviewRuntimeError("document_preview_paddleocr3_download_invalid")


def _validate_downloaded_json(path: Path) -> None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DocumentPreviewRuntimeError("document_preview_paddleocr3_download_invalid") from error
    if not isinstance(value, dict):
        raise DocumentPreviewRuntimeError("document_preview_paddleocr3_download_invalid")


def _verify_artifact_file(*, artifact: _PinnedArtifact, path: Path, code: str) -> None:
    """Verify regular-file, byte-size and the immutable snapshot digest again."""

    try:
        metadata = path.lstat()
    except OSError as error:
        raise DocumentPreviewRuntimeError(code) from error
    if (
        _is_link_or_reparse_point(metadata)
        or not stat.S_ISREG(metadata.st_mode)
        or metadata.st_size != artifact.byte_size
    ):
        raise DocumentPreviewRuntimeError(code)
    sha256 = hashlib.sha256()
    git_blob_sha1 = hashlib.sha1()
    git_blob_sha1.update(f"blob {artifact.byte_size}\0".encode("ascii"))
    try:
        with path.open("rb") as stream:
            remaining = artifact.byte_size
            while remaining:
                chunk = stream.read(min(_DOWNLOAD_CHUNK_BYTES, remaining))
                if not chunk:
                    raise DocumentPreviewRuntimeError(code)
                sha256.update(chunk)
                git_blob_sha1.update(chunk)
                remaining -= len(chunk)
            if stream.read(1):
                raise DocumentPreviewRuntimeError(code)
        checked = path.lstat()
    except DocumentPreviewRuntimeError:
        raise
    except OSError as error:
        raise DocumentPreviewRuntimeError(code) from error
    if (
        _is_link_or_reparse_point(checked)
        or not stat.S_ISREG(checked.st_mode)
        or checked.st_size != artifact.byte_size
        or (artifact.sha256 is not None and sha256.hexdigest() != artifact.sha256)
        or (
            artifact.git_blob_sha1 is not None
            and git_blob_sha1.hexdigest() != artifact.git_blob_sha1
        )
    ):
        raise DocumentPreviewRuntimeError(code)
    if artifact.filename in {"config.json", "inference.json"}:
        _validate_downloaded_json(path)


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
    except DocumentPreviewRuntimeError:
        raise
    except Exception as error:
        raise DocumentPreviewRuntimeError("document_preview_paddleocr3_download_failed") from error
    try:
        final_url = response.geturl()
        if not isinstance(final_url, str) or not _is_allowed_final_download_url(final_url):
            raise DocumentPreviewRuntimeError("document_preview_paddleocr3_download_invalid")
        _response_content_length(response, expected=artifact.byte_size)
        try:
            with destination.open("xb") as stream:
                remaining = artifact.byte_size
                while remaining:
                    chunk = response.read(min(_DOWNLOAD_CHUNK_BYTES, remaining))
                    if not isinstance(chunk, bytes) or not chunk:
                        raise DocumentPreviewRuntimeError(
                            "document_preview_paddleocr3_download_invalid"
                        )
                    stream.write(chunk)
                    remaining -= len(chunk)
                if response.read(1):
                    raise DocumentPreviewRuntimeError(
                        "document_preview_paddleocr3_download_invalid"
                    )
                stream.flush()
                os.fsync(stream.fileno())
        except DocumentPreviewRuntimeError:
            raise
        except OSError as error:
            raise DocumentPreviewRuntimeError(
                "document_preview_paddleocr3_download_failed"
            ) from error
        _verify_artifact_file(
            artifact=artifact,
            path=destination,
            code="document_preview_paddleocr3_download_invalid",
        )
    finally:
        try:
            response_context.__exit__(None, None, None)
        except Exception:
            # A fully verified immutable file is not retried or overwritten
            # because its remote connection close happened to fail.
            pass


def _expected_artifacts_by_directory() -> dict[str, tuple[_PinnedArtifact, ...]]:
    grouped: dict[str, list[_PinnedArtifact]] = {directory: [] for directory in _MODEL_DIRECTORIES}
    for artifact in _ARTIFACTS:
        if (
            artifact.directory not in _ALLOWED_DIRECTORIES
            or artifact.filename not in _ALLOWED_FILENAMES
        ):
            raise DocumentPreviewRuntimeError("document_preview_paddleocr3_staging_invalid")
        grouped[artifact.directory].append(artifact)
    if any(
        {artifact.filename for artifact in grouped[directory]} != _ALLOWED_FILENAMES
        for directory in _MODEL_DIRECTORIES
    ):
        raise DocumentPreviewRuntimeError("document_preview_paddleocr3_staging_invalid")
    return {directory: tuple(grouped[directory]) for directory in _MODEL_DIRECTORIES}


def _assert_exact_staging_files(*, stage: Path) -> None:
    grouped = _expected_artifacts_by_directory()
    try:
        directories = {child.name: child for child in stage.iterdir()}
    except OSError as error:
        raise DocumentPreviewRuntimeError("document_preview_paddleocr3_staging_invalid") from error
    if set(directories) != _ALLOWED_DIRECTORIES:
        raise DocumentPreviewRuntimeError("document_preview_paddleocr3_staging_invalid")
    for directory, artifacts in grouped.items():
        path = directories[directory]
        try:
            metadata = path.lstat()
            children = {child.name: child for child in path.iterdir()}
        except OSError as error:
            raise DocumentPreviewRuntimeError(
                "document_preview_paddleocr3_staging_invalid"
            ) from error
        if _is_link_or_reparse_point(metadata) or not stat.S_ISDIR(metadata.st_mode):
            raise DocumentPreviewRuntimeError("document_preview_paddleocr3_staging_invalid")
        expected = {artifact.filename for artifact in artifacts}
        if set(children) != expected:
            raise DocumentPreviewRuntimeError("document_preview_paddleocr3_staging_invalid")
        for artifact in artifacts:
            _verify_artifact_file(
                artifact=artifact,
                path=children[artifact.filename],
                code="document_preview_paddleocr3_staging_invalid",
            )


def _acquire_publish_lock(*, root: Path, parent: Path) -> Path:
    lock = parent / _PUBLISH_LOCK_FILENAME
    _path_under_root(
        root=root,
        path=lock,
        must_exist=False,
        code="document_preview_paddleocr3_publish_locked",
    )
    try:
        with lock.open("x", encoding="ascii") as stream:
            stream.write("paddleocr3-mobile-models\n")
            stream.flush()
            os.fsync(stream.fileno())
    except FileExistsError as error:
        raise DocumentPreviewRuntimeError("document_preview_paddleocr3_publish_locked") from error
    except OSError as error:
        raise DocumentPreviewRuntimeError("document_preview_paddleocr3_publish_failed") from error
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


def _remove_stage(*, root: Path, stage: Path) -> None:
    """Remove only helper-created regular files from the temporary stage.

    A malformed stage is deliberately left in place.  This is a narrow manual
    cleanup rather than a recursive delete, so it cannot follow a link or
    erase a path the helper did not create.
    """

    try:
        checked = _path_under_root(
            root=root,
            path=stage,
            must_exist=True,
            code="document_preview_paddleocr3_staging_invalid",
        )
        metadata = checked.lstat()
        if _is_link_or_reparse_point(metadata) or not stat.S_ISDIR(metadata.st_mode):
            return
        directories = {child.name: child for child in checked.iterdir()}
        if not set(directories).issubset(_ALLOWED_DIRECTORIES):
            return
        for directory in _MODEL_DIRECTORIES:
            child_directory = directories.get(directory)
            if child_directory is None:
                continue
            directory_metadata = child_directory.lstat()
            if _is_link_or_reparse_point(directory_metadata) or not stat.S_ISDIR(
                directory_metadata.st_mode
            ):
                return
            children = list(child_directory.iterdir())
            if any(
                child.name not in _ALLOWED_FILENAMES
                or _is_link_or_reparse_point(child.lstat())
                or not stat.S_ISREG(child.lstat().st_mode)
                for child in children
            ):
                return
            for child in children:
                child.unlink()
            child_directory.rmdir()
        checked.rmdir()
    except (FileNotFoundError, OSError, DocumentPreviewRuntimeError):
        return


def _publish_stage(*, root: Path, stage: Path, target: Path) -> None:
    parent = _ensure_safe_directory(
        root=root,
        path=target.parent,
        code="document_preview_paddleocr3_publish_failed",
    )
    lock = _acquire_publish_lock(root=root, parent=parent)
    try:
        _assert_absent(
            root=root,
            path=target,
            code="document_preview_paddleocr3_target_exists",
        )
        _assert_exact_staging_files(stage=stage)
        try:
            # Both stage and target are under the same document-preview root,
            # so this is an atomic directory rename rather than a copy.  It
            # does not replace an existing target on Windows or POSIX.
            os.rename(stage, target)
        except FileExistsError as error:
            raise DocumentPreviewRuntimeError(
                "document_preview_paddleocr3_target_exists"
            ) from error
        except OSError as error:
            raise DocumentPreviewRuntimeError(
                "document_preview_paddleocr3_publish_failed"
            ) from error
    finally:
        _release_publish_lock(lock)


def stage_paddleocr3_mobile_models(
    *,
    document_preview_root: Path,
    minimum_free_gib: float = _DEFAULT_MINIMUM_FREE_GIB,
    opener: Callable[..., Any] = _open_pinned_source,
) -> None:
    """Download, verify, and atomically publish the one P3 mobile model pair."""

    root = validate_document_preview_root(document_preview_root, create=True)
    _check_free_space(root, minimum_free_gib)
    target = root.joinpath(*_TARGET_RELATIVE_PARTS)
    _assert_absent(root=root, path=target, code="document_preview_paddleocr3_target_exists")
    staging_root = _ensure_safe_directory(
        root=root,
        path=root / _STAGING_DIRECTORY,
        code="document_preview_paddleocr3_staging_invalid",
    )
    try:
        stage = Path(tempfile.mkdtemp(prefix=".paddleocr3-mobile-", dir=staging_root))
    except OSError as error:
        raise DocumentPreviewRuntimeError("document_preview_paddleocr3_staging_invalid") from error
    try:
        stage = _path_under_root(
            root=root,
            path=stage,
            must_exist=True,
            code="document_preview_paddleocr3_staging_invalid",
        )
        grouped = _expected_artifacts_by_directory()
        for directory, artifacts in grouped.items():
            stage_directory = _ensure_safe_directory(
                root=root,
                path=stage / directory,
                code="document_preview_paddleocr3_staging_invalid",
            )
            for artifact in artifacts:
                destination = stage_directory / artifact.filename
                _path_under_root(
                    root=root,
                    path=destination,
                    must_exist=False,
                    code="document_preview_paddleocr3_staging_invalid",
                )
                _download_pinned_artifact(
                    artifact=artifact,
                    destination=destination,
                    opener=opener,
                )
        _publish_stage(root=root, stage=stage, target=target)
    except Exception:
        _remove_stage(root=root, stage=stage)
        raise


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and verify only the pinned PaddleOCR 3 mobile model files."
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
        help="Required acknowledgement before the eight pinned HTTPS files are downloaded.",
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
        stage_paddleocr3_mobile_models(
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
                "engine_kind": "paddleocr",
                "runtime_api": "paddleocr3-paddlex-3.2",
                "artifact_count": len(_ARTIFACTS),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
