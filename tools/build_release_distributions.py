"""Build source-bound Agent distributions from one exact clean Git commit."""

from __future__ import annotations

import argparse
import gzip
import io
import os
import shutil
import subprocess
import tarfile
import tempfile
from collections.abc import Mapping
from pathlib import Path
from pathlib import PurePosixPath

from tools.verify_distribution import (
    BUILD_MANIFEST_SDIST_PATH,
    DistributionVerificationError,
    _git_archive_payloads,
    build_manifest_bytes,
    source_snapshot_sha256,
    verify_sdist,
    verify_wheel,
)


class ReleaseBuildError(RuntimeError):
    pass


def _resolve_executable(command: str, *, error_code: str) -> str:
    try:
        discovered = shutil.which(command)
        if discovered is None:
            raise OSError
        executable = Path(discovered).resolve(strict=True)
    except OSError:
        raise ReleaseBuildError(error_code) from None
    if not executable.is_file():
        raise ReleaseBuildError(error_code)
    return str(executable)


def _git(repository: Path, *arguments: str) -> str:
    try:
        git = _resolve_executable("git", error_code="release_git_state_invalid")
        completed = subprocess.run(  # noqa: S603 - resolved executable; fixed argv; no shell
            [git, "-C", str(repository), *arguments],
            check=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            shell=False,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        raise ReleaseBuildError("release_git_state_invalid") from None
    return completed.stdout.strip()


def _assert_exact_clean_head(repository: Path, expected_git_sha: str) -> str:
    head = _git(repository, "rev-parse", "HEAD").casefold()
    if head != expected_git_sha.casefold() or len(head) != 40:
        raise ReleaseBuildError("release_git_head_mismatch")
    status = _git(
        repository,
        "status",
        "--porcelain=v1",
        "--untracked-files=all",
        "--ignore-submodules=none",
    )
    if status:
        raise ReleaseBuildError("release_git_worktree_not_clean")
    return head


def _source_date_epoch(repository: Path, head: str) -> int:
    raw_epoch = _git(repository, "show", "-s", "--format=%ct", head)
    if not raw_epoch.isascii() or not raw_epoch.isdigit():
        raise ReleaseBuildError("release_git_timestamp_invalid")
    epoch = int(raw_epoch)
    if not 315_532_800 <= epoch <= 4_294_967_295:
        # ZIP archives cannot represent dates before 1980-01-01 UTC and the
        # gzip header stores mtime as an unsigned 32-bit integer.
        raise ReleaseBuildError("release_git_timestamp_invalid")
    return epoch


def _reproducible_build_environment(source_date_epoch: int) -> dict[str, str]:
    environment = dict(os.environ)
    environment.update(
        {
            "PYTHONHASHSEED": "0",
            "SOURCE_DATE_EPOCH": str(source_date_epoch),
            "TZ": "UTC",
        }
    )
    return environment


def _materialize_snapshot(
    destination: Path,
    payloads: dict[str, bytes],
    *,
    source_date_epoch: int,
) -> None:
    for name, payload in payloads.items():
        target = destination.joinpath(*name.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)
        os.utime(target, (source_date_epoch, source_date_epoch))


def _normalize_directory_mtimes(root: Path, *, source_date_epoch: int) -> None:
    directories = [path for path in root.rglob("*") if path.is_dir()]
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        os.utime(directory, (source_date_epoch, source_date_epoch))
    os.utime(root, (source_date_epoch, source_date_epoch))


def _canonicalize_sdist(path: Path, *, source_date_epoch: int) -> None:
    """Rewrite a controlled build output with deterministic tar/gzip metadata."""

    payloads: dict[str, bytes] = {}
    casefolded_names: set[str] = set()
    try:
        with tarfile.open(path, mode="r:gz") as source:
            for member in source.getmembers():
                if member.isdir():
                    continue
                if not member.isfile():
                    raise ReleaseBuildError("release_sdist_canonicalization_failed")
                pure_name = PurePosixPath(member.name)
                if (
                    pure_name.is_absolute()
                    or member.name != pure_name.as_posix()
                    or not pure_name.parts
                    or any(part in {"", ".", ".."} for part in pure_name.parts)
                    or "\\" in member.name
                    or ":" in pure_name.parts[0]
                ):
                    raise ReleaseBuildError("release_sdist_canonicalization_failed")
                folded_name = member.name.casefold()
                if folded_name in casefolded_names:
                    raise ReleaseBuildError("release_sdist_canonicalization_failed")
                extracted = source.extractfile(member)
                if extracted is None:
                    raise ReleaseBuildError("release_sdist_canonicalization_failed")
                payload = extracted.read()
                if len(payload) != member.size:
                    raise ReleaseBuildError("release_sdist_canonicalization_failed")
                casefolded_names.add(folded_name)
                payloads[member.name] = payload

        uncompressed = io.BytesIO()
        with tarfile.open(
            fileobj=uncompressed,
            mode="w",
            format=tarfile.PAX_FORMAT,
        ) as destination:
            for name in sorted(payloads):
                payload = payloads[name]
                member = tarfile.TarInfo(name)
                member.size = len(payload)
                member.mode = 0o644
                member.mtime = source_date_epoch
                member.uid = 0
                member.gid = 0
                member.uname = ""
                member.gname = ""
                member.pax_headers = {}
                destination.addfile(member, io.BytesIO(payload))

        canonical_path = path.with_name(f".{path.name}.canonical.tmp")
        canonical_path.write_bytes(
            gzip.compress(
                uncompressed.getvalue(),
                compresslevel=9,
                mtime=source_date_epoch,
            )
        )
        canonical_path.replace(path)
    except ReleaseBuildError:
        raise
    except (EOFError, OSError, tarfile.TarError):
        raise ReleaseBuildError("release_sdist_canonicalization_failed") from None
    finally:
        path.with_name(f".{path.name}.canonical.tmp").unlink(missing_ok=True)


def _run_uv(
    uv: str,
    *arguments: str,
    environment: Mapping[str, str],
) -> None:
    try:
        executable = _resolve_executable(
            uv,
            error_code="release_distribution_build_failed",
        )
        subprocess.run(  # noqa: S603 - resolved executable; fixed argv; no shell
            [executable, *arguments],
            check=True,
            env=dict(environment),
            stdin=subprocess.DEVNULL,
            shell=False,
        )
    except (OSError, subprocess.CalledProcessError):
        raise ReleaseBuildError("release_distribution_build_failed") from None


def build_release_distributions(
    *,
    repository: Path,
    expected_git_sha: str,
    output_root: Path,
    uv_executable: str,
    replace: bool,
) -> tuple[Path, Path]:
    try:
        repository = repository.resolve(strict=True)
    except OSError:
        raise ReleaseBuildError("release_repository_invalid") from None
    head = _assert_exact_clean_head(repository, expected_git_sha)
    try:
        source_payloads = _git_archive_payloads(repository, head)
    except DistributionVerificationError:
        raise ReleaseBuildError("release_source_snapshot_invalid") from None
    snapshot_sha256 = source_snapshot_sha256(source_payloads)
    source_date_epoch = _source_date_epoch(repository, head)
    build_environment = _reproducible_build_environment(source_date_epoch)
    manifest = build_manifest_bytes(
        git_sha=head,
        source_snapshot_sha256_value=snapshot_sha256,
    )

    output_root = output_root.resolve()
    wheel_output = output_root / "wheel" / "customs_ai_agent-0.3.0-py3-none-any.whl"
    sdist_output = output_root / "source" / "customs_ai_agent-0.3.0.tar.gz"
    for target in (wheel_output, sdist_output):
        if target.exists() and not replace:
            raise ReleaseBuildError("release_output_exists")

    with tempfile.TemporaryDirectory(prefix="customs-agent-release-") as raw_temp:
        temporary_root = Path(raw_temp)
        source_tree = temporary_root / "source-tree"
        source_tree.mkdir()
        _materialize_snapshot(
            source_tree,
            source_payloads,
            source_date_epoch=source_date_epoch,
        )
        manifest_path = source_tree.joinpath(*BUILD_MANIFEST_SDIST_PATH.split("/"))
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_bytes(manifest)
        os.utime(manifest_path, (source_date_epoch, source_date_epoch))
        _normalize_directory_mtimes(
            source_tree,
            source_date_epoch=source_date_epoch,
        )

        built_source = temporary_root / "built-source"
        built_wheel = temporary_root / "built-wheel"
        built_source.mkdir()
        built_wheel.mkdir()
        _run_uv(
            uv_executable,
            "build",
            "--sdist",
            "--no-sources",
            "--out-dir",
            str(built_source),
            str(source_tree),
            environment=build_environment,
        )
        temporary_sdist = built_source / sdist_output.name
        _canonicalize_sdist(
            temporary_sdist,
            source_date_epoch=source_date_epoch,
        )
        _run_uv(
            uv_executable,
            "build",
            "--wheel",
            "--no-sources",
            "--out-dir",
            str(built_wheel),
            str(temporary_sdist),
            environment=build_environment,
        )
        temporary_wheel = built_wheel / wheel_output.name
        try:
            verify_sdist(
                temporary_sdist,
                expected_git_sha=head,
                source_root=repository,
            )
            verify_wheel(
                temporary_wheel,
                expected_git_sha=head,
                source_root=repository,
            )
        except DistributionVerificationError:
            raise ReleaseBuildError("release_distribution_verification_failed") from None

        _assert_exact_clean_head(repository, expected_git_sha)
        wheel_output.parent.mkdir(parents=True, exist_ok=True)
        sdist_output.parent.mkdir(parents=True, exist_ok=True)
        try:
            if replace:
                wheel_output.unlink(missing_ok=True)
                sdist_output.unlink(missing_ok=True)
            shutil.copy2(temporary_wheel, wheel_output)
            shutil.copy2(temporary_sdist, sdist_output)
            _assert_exact_clean_head(repository, expected_git_sha)
        except (OSError, ReleaseBuildError):
            wheel_output.unlink(missing_ok=True)
            sdist_output.unlink(missing_ok=True)
            raise ReleaseBuildError("release_output_not_bound_to_clean_head") from None

    return wheel_output, sdist_output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--expected-git-sha", required=True)
    parser.add_argument("--output-root", type=Path, default=Path("dist"))
    parser.add_argument("--uv", default=shutil.which("uv") or "uv")
    parser.add_argument("--replace", action="store_true")
    args = parser.parse_args()
    try:
        wheel, sdist = build_release_distributions(
            repository=args.repository,
            expected_git_sha=args.expected_git_sha,
            output_root=args.output_root,
            uv_executable=args.uv,
            replace=args.replace,
        )
    except ReleaseBuildError as exc:
        print(f"release_build_failed={exc}")
        return 1
    print(f"wheel={wheel}")
    print(f"sdist={sdist}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
