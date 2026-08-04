"""Create a content-addressed source identity for local UAT.

This helper deliberately identifies tracked source files only.  It is not a
wheel/sdist release verifier and does not create a deployable artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import re
import shutil
import subprocess
import tarfile
from collections.abc import Iterable, Mapping
from pathlib import Path, PurePosixPath


EXPECTED_DISTRIBUTION = "customs-ai-agent"
EXPECTED_VERSION = "0.3.0"
BUILD_MANIFEST_CONTRACT = "customs-ai-agent-build-manifest"
BUILD_MANIFEST_SCHEMA_VERSION = "1.0"
SOURCE_SNAPSHOT_ALGORITHM = "git-archive-files-sha256-v1"
_GIT_SHA = re.compile(r"^[a-f0-9]{40}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")


class SourceSnapshotError(RuntimeError):
    """The requested tracked-source identity cannot be constructed safely."""


def _validate_archive_members(
    members: Iterable[tarfile.TarInfo],
) -> tuple[tuple[tarfile.TarInfo, str], ...]:
    validated: list[tuple[tarfile.TarInfo, str]] = []
    seen: set[str] = set()
    allowed_types = {tarfile.REGTYPE, tarfile.AREGTYPE, tarfile.DIRTYPE}
    for member in members:
        raw_name = member.name
        if (
            member.type not in allowed_types
            or getattr(member, "sparse", None)
            or not raw_name
            or "\\" in raw_name
            or any(ord(character) < 32 or ord(character) == 127 for character in raw_name)
            or member.linkname
            or member.size < 0
        ):
            raise SourceSnapshotError("source snapshot contains an unsafe archive member")
        name = raw_name.rstrip("/") if member.isdir() else raw_name
        path = PurePosixPath(name)
        if (
            not name
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in name.split("/"))
            or re.match(r"^[A-Za-z]:", name)
            or (member.isdir() and member.size != 0)
            or (member.isfile() and raw_name.endswith("/"))
        ):
            raise SourceSnapshotError("source snapshot contains an unsafe archive path")
        canonical = path.as_posix()
        folded = canonical.casefold()
        if folded in seen:
            raise SourceSnapshotError("source snapshot contains colliding archive paths")
        seen.add(folded)
        validated.append((member, canonical))
    if not validated:
        raise SourceSnapshotError("source snapshot archive is empty")
    return tuple(validated)


def git_archive_payloads(repository: Path, git_sha: str) -> dict[str, bytes]:
    """Read exactly one tracked Git tree without reading local state or secrets."""

    if _GIT_SHA.fullmatch(git_sha) is None:
        raise SourceSnapshotError("expected Git SHA is invalid")
    try:
        discovered = shutil.which("git")
        if discovered is None:
            raise OSError
        git = Path(discovered).resolve(strict=True)
        if not git.is_file():
            raise OSError
        repository = repository.resolve(strict=True)
        completed = subprocess.run(  # noqa: S603 - resolved executable; fixed argv; no shell
            [str(git), "-C", str(repository), "archive", "--format=tar", git_sha],
            check=True,
            capture_output=True,
            stdin=subprocess.DEVNULL,
            shell=False,
        )
        with tarfile.open(fileobj=io.BytesIO(completed.stdout), mode="r:") as archive:
            payloads: dict[str, bytes] = {}
            for member, name in _validate_archive_members(archive.getmembers()):
                if member.isfile():
                    stream = archive.extractfile(member)
                    if stream is None:
                        raise OSError
                    payloads[name] = stream.read()
    except (OSError, subprocess.CalledProcessError, tarfile.TarError):
        raise SourceSnapshotError("exact Git source snapshot is unavailable") from None
    if not payloads:
        raise SourceSnapshotError("exact Git source snapshot is empty")
    return payloads


def source_snapshot_sha256(payloads: Mapping[str, bytes]) -> str:
    """Return the stable SHA-256 of canonical tracked source payloads."""

    digest = hashlib.sha256()
    digest.update(b"customs-ai-agent-source-snapshot-v1\0")
    for name in sorted(payloads):
        encoded_name = name.encode("utf-8")
        payload = payloads[name]
        digest.update(len(encoded_name).to_bytes(8, "big"))
        digest.update(encoded_name)
        digest.update(len(payload).to_bytes(8, "big"))
        digest.update(payload)
    return digest.hexdigest()


def build_manifest_bytes(*, git_sha: str, source_snapshot_sha256_value: str) -> bytes:
    """Render the canonical non-secret source identity manifest."""

    if (
        _GIT_SHA.fullmatch(git_sha) is None
        or _SHA256.fullmatch(source_snapshot_sha256_value) is None
    ):
        raise SourceSnapshotError("build manifest identity is invalid")
    return (
        json.dumps(
            {
                "contract": BUILD_MANIFEST_CONTRACT,
                "distribution": EXPECTED_DISTRIBUTION,
                "git_sha": git_sha,
                "package_version": EXPECTED_VERSION,
                "schema_version": BUILD_MANIFEST_SCHEMA_VERSION,
                "source_snapshot_algorithm": SOURCE_SNAPSHOT_ALGORITHM,
                "source_snapshot_sha256": source_snapshot_sha256_value,
            },
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")


def write_build_manifest(
    *,
    repository: Path,
    git_sha: str,
    output: Path,
) -> str:
    """Write a source identity manifest outside the checked-out repository.

    Local UAT uses this small helper for provenance only.  Keeping the output
    outside the checkout makes the source input immutable for the operation
    and prevents a diagnostic run from modifying tracked product files.
    """

    resolved_repository = repository.resolve(strict=True)
    resolved_output = output.expanduser().resolve()
    try:
        resolved_output.relative_to(resolved_repository)
    except ValueError:
        pass
    else:
        raise SourceSnapshotError("source snapshot output must be outside the repository")
    payloads = git_archive_payloads(resolved_repository, git_sha)
    digest = source_snapshot_sha256(payloads)
    encoded = build_manifest_bytes(
        git_sha=git_sha,
        source_snapshot_sha256_value=digest,
    )
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    temporary = resolved_output.with_name(f".{resolved_output.name}.{os.getpid()}.tmp")
    try:
        temporary.write_bytes(encoded)
        os.replace(temporary, resolved_output)
    except OSError:
        temporary.unlink(missing_ok=True)
        raise SourceSnapshotError("source snapshot manifest could not be written") from None
    return digest


def main(argv: list[str] | None = None) -> int:
    """Create one non-secret source manifest for a clean Git input tree."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True, type=Path)
    parser.add_argument("--git-sha", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        digest = write_build_manifest(
            repository=args.repository,
            git_sha=args.git_sha,
            output=args.output,
        )
    except SourceSnapshotError as exc:
        print(json.dumps({"status": "FAILED", "code": str(exc)}, ensure_ascii=True))
        return 1
    print(json.dumps({"status": "PASS", "source_snapshot_sha256": digest}, ensure_ascii=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
