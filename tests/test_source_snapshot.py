from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from tools.source_snapshot import (
    SourceSnapshotError,
    _validate_archive_members,
    build_manifest_bytes,
    git_archive_payloads,
    source_snapshot_sha256,
    write_build_manifest,
)


def test_source_snapshot_is_stable_and_content_bound() -> None:
    payloads = {"b.txt": b"two", "a.txt": b"one"}

    assert source_snapshot_sha256(payloads) == source_snapshot_sha256(
        dict(reversed(payloads.items()))
    )
    assert source_snapshot_sha256(payloads) != source_snapshot_sha256({"a.txt": b"changed"})


def test_build_manifest_is_canonical_and_non_secret() -> None:
    payload = build_manifest_bytes(git_sha="a" * 40, source_snapshot_sha256_value="b" * 64)

    assert payload.endswith(b"\n")
    assert json.loads(payload)["source_snapshot_sha256"] == "b" * 64


@pytest.mark.parametrize("git_sha,digest", [("bad", "b" * 64), ("a" * 40, "bad")])
def test_build_manifest_rejects_invalid_identity(git_sha: str, digest: str) -> None:
    with pytest.raises(SourceSnapshotError, match="identity is invalid"):
        build_manifest_bytes(git_sha=git_sha, source_snapshot_sha256_value=digest)


def test_git_archive_payloads_reads_a_real_tracked_tree(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    subprocess.run(
        ["git", "-C", str(tmp_path), "config", "core.autocrlf", "false"],
        check=True,
    )
    (tmp_path / "tracked.txt").write_bytes(b"tracked\n")
    subprocess.run(["git", "-C", str(tmp_path), "add", "tracked.txt"], check=True)
    subprocess.run(
        [
            "git",
            "-C",
            str(tmp_path),
            "-c",
            "user.email=test@example.invalid",
            "-c",
            "user.name=Test",
            "commit",
            "-qm",
            "snapshot",
        ],
        check=True,
    )
    sha = subprocess.run(
        ["git", "-C", str(tmp_path), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()

    assert git_archive_payloads(tmp_path, sha) == {"tracked.txt": b"tracked\n"}


def test_archive_members_reject_control_character_paths() -> None:
    import tarfile

    member = tarfile.TarInfo("bad\x01name.txt")
    member.type = tarfile.REGTYPE
    with pytest.raises(SourceSnapshotError, match="unsafe archive member"):
        _validate_archive_members((member,))


def test_write_build_manifest_is_outside_source_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    output = tmp_path / "state" / "build-manifest.json"
    monkeypatch.setattr(
        "tools.source_snapshot.git_archive_payloads",
        lambda _repository, _git_sha: {"tracked.txt": b"tracked\n"},
    )

    digest = write_build_manifest(repository=repository, git_sha="a" * 40, output=output)

    assert json.loads(output.read_text(encoding="utf-8"))["source_snapshot_sha256"] == digest
    with pytest.raises(SourceSnapshotError, match="outside the repository"):
        write_build_manifest(
            repository=repository,
            git_sha="a" * 40,
            output=repository / "build-manifest.json",
        )
