from __future__ import annotations

import csv
import hashlib
import io
import json
import subprocess
import sys
import tarfile
import tomllib
import zipfile
from pathlib import Path
from typing import Any

import pytest

import tools.build_release_distributions as release_builder
from tools.build_release_distributions import ReleaseBuildError, _assert_exact_clean_head
from tools.verify_distribution import (
    BUILD_MANIFEST_SDIST_PATH,
    BUILD_MANIFEST_WHEEL_PATH,
    EXPECTED_CONSOLE_ENTRY_POINT,
    EXPECTED_SERVICE_CONSOLE_ENTRY_POINT,
    EXPECTED_IMPORT_PACKAGE,
    EXPECTED_VERSION,
    DistributionVerificationError,
    _SDIST_REQUIRED_FILES,
    _WHEEL_REQUIRED_PACKAGE_FILES,
    _direct_url_wheel_path,
    _git_archive_payloads,
    _record_digest,
    _verify_installed_record_against_wheel,
    _verify_wheel_origin,
    build_manifest_bytes,
    source_snapshot_sha256,
    verify_sdist,
    verify_wheel,
)


ROOT = Path(__file__).parents[1]
DIST_INFO = f"{EXPECTED_IMPORT_PACKAGE}-{EXPECTED_VERSION}.dist-info"
METADATA = (
    "Metadata-Version: 2.4\n"
    "Name: customs-ai-agent\n"
    f"Version: {EXPECTED_VERSION}\n"
    "Requires-Python: <3.12,>=3.11\n"
    "Requires-Dist: anthropic<1,>=0.117.0\n"
    "Requires-Dist: cryptography<49,>=48.0.1\n"
    "Requires-Dist: fastapi<1,>=0.115\n"
    "Requires-Dist: jsonschema[format]<5,>=4.23\n"
    "Requires-Dist: langgraph<2,>=0.2.0\n"
    "Requires-Dist: pillow<13,>=11.0\n"
    "Requires-Dist: pydantic<3,>=2.7\n"
    "Requires-Dist: pypdfium2<6,>=5.12\n"
    "Requires-Dist: python-dotenv<2,>=1.0\n"
    "Requires-Dist: rapidfuzz<4,>=3.9\n"
    "Requires-Dist: uvicorn<1,>=0.32\n"
    "\n"
).encode()
TEST_GIT_SHA = "1" * 40
TEST_SOURCE_SNAPSHOT_SHA256 = "2" * 64
TEST_BUILD_MANIFEST = build_manifest_bytes(
    git_sha=TEST_GIT_SHA,
    source_snapshot_sha256_value=TEST_SOURCE_SNAPSHOT_SHA256,
)


def _write_wheel(
    path: Path,
    *,
    metadata: bytes = METADATA,
    extra_files: dict[str, bytes] | None = None,
    excluded_files: set[str] | None = None,
) -> None:
    payloads: dict[str, bytes] = {
        name: (
            b"{}\n" if name.endswith(".json") else b"prompt\n" if name.endswith(".txt") else b"\n"
        )
        for name in _WHEEL_REQUIRED_PACKAGE_FILES
        if name not in (excluded_files or set())
    }
    runtime_lock_name = "customs_ai_agent/contracts/evidence/runtime-release-v2.lock.json"
    if runtime_lock_name in payloads:
        payloads[runtime_lock_name] = (
            ROOT
            / "src"
            / "customs_ai_agent"
            / "contracts"
            / "evidence"
            / "runtime-release-v2.lock.json"
        ).read_bytes()
    provider_lock_name = "customs_ai_agent/contracts/evidence/provider-runtime-lock.json"
    if provider_lock_name in payloads:
        payloads[provider_lock_name] = (
            ROOT
            / "src"
            / "customs_ai_agent"
            / "contracts"
            / "evidence"
            / "provider-runtime-lock.json"
        ).read_bytes()
    v3_runtime_lock_name = "customs_ai_agent/contracts/evidence/v3/runtime-release-v3.lock.json"
    if v3_runtime_lock_name in payloads:
        payloads[v3_runtime_lock_name] = (
            ROOT
            / "src"
            / "customs_ai_agent"
            / "contracts"
            / "evidence"
            / "v3"
            / "runtime-release-v3.lock.json"
        ).read_bytes()
    generated_payloads = {
        BUILD_MANIFEST_WHEEL_PATH: TEST_BUILD_MANIFEST,
        f"{DIST_INFO}/METADATA": metadata,
        f"{DIST_INFO}/WHEEL": (
            b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
        ),
        f"{DIST_INFO}/entry_points.txt": (
            (
                f"[console_scripts]\n{EXPECTED_CONSOLE_ENTRY_POINT}\n"
                f"{EXPECTED_SERVICE_CONSOLE_ENTRY_POINT}\n"
            ).encode()
        ),
    }
    payloads.update(
        {
            name: payload
            for name, payload in generated_payloads.items()
            if name not in (excluded_files or set())
        }
    )
    payloads.update(extra_files or {})
    record_name = f"{DIST_INFO}/RECORD"
    record_buffer = io.StringIO(newline="")
    writer = csv.writer(record_buffer, lineterminator="\n")
    for name, payload in sorted(payloads.items()):
        writer.writerow(
            [
                name,
                f"sha256={_record_digest(payload)}",
                str(len(payload)),
            ]
        )
    writer.writerow([record_name, "", ""])
    payloads[record_name] = record_buffer.getvalue().encode()
    with zipfile.ZipFile(path, mode="w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in payloads.items():
            archive.writestr(name, payload)


class _InstalledDistribution:
    def __init__(self, site_packages: Path) -> None:
        self.site_packages = site_packages

    def read_text(self, filename: str) -> str | None:
        path = self.site_packages / DIST_INFO / filename
        if not path.is_file():
            return None
        return path.read_text(encoding="utf-8")

    def locate_file(self, path: Any) -> Path:
        return self.site_packages / str(path)


def _write_uv_install_tree(
    wheel: Path,
    site_packages: Path,
    *,
    direct_url_archive_info: dict[str, object] | None = None,
) -> _InstalledDistribution:
    with zipfile.ZipFile(wheel) as archive:
        wheel_record_text = archive.read(f"{DIST_INFO}/RECORD").decode("utf-8")
        for name in archive.namelist():
            if name == f"{DIST_INFO}/RECORD":
                continue
            target = site_packages / Path(*name.split("/"))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(archive.read(name))

    installer_payloads = {
        f"{DIST_INFO}/INSTALLER": b"uv",
        f"{DIST_INFO}/REQUESTED": b"",
        f"{DIST_INFO}/direct_url.json": json.dumps(
            {
                "url": wheel.resolve().as_uri(),
                "archive_info": direct_url_archive_info or {},
            },
            separators=(",", ":"),
        ).encode(),
        f"{DIST_INFO}/uv_cache.json": b"{}",
        "../../../bin/customs-ai-agent": b'#!/bin/sh\nexec python -m customs_ai_agent "$@"\n',
        "../../../bin/customs-ai-agent-serve": (
            b'#!/bin/sh\nexec python -m customs_ai_agent.service "$@"\n'
        ),
    }
    for name, payload in installer_payloads.items():
        target = site_packages / Path(*name.split("/"))
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(payload)

    rows = list(csv.reader(wheel_record_text.splitlines()))
    for name, payload in installer_payloads.items():
        rows.append(
            [
                name,
                f"sha256={_record_digest(payload)}",
                str(len(payload)),
            ]
        )
    record_buffer = io.StringIO(newline="")
    writer = csv.writer(record_buffer, lineterminator="\n")
    writer.writerows(rows)
    record_path = site_packages / DIST_INFO / "RECORD"
    record_path.parent.mkdir(parents=True, exist_ok=True)
    record_path.write_text(record_buffer.getvalue(), encoding="utf-8")
    return _InstalledDistribution(site_packages)


def _write_sdist(
    path: Path,
    *,
    extra_files: dict[str, bytes] | None = None,
    extra_members: list[tuple[tarfile.TarInfo, bytes | None]] | None = None,
) -> None:
    payloads = {
        name: (METADATA if name == "PKG-INFO" else b"{}\n" if name.endswith(".json") else b"\n")
        for name in {*_SDIST_REQUIRED_FILES, "PKG-INFO"}
    }
    payloads["contracts/evidence/runtime-release-v2.lock.json"] = (
        ROOT / "contracts" / "evidence" / "runtime-release-v2.lock.json"
    ).read_bytes()
    payloads["src/customs_ai_agent/contracts/evidence/runtime-release-v2.lock.json"] = (
        ROOT
        / "src"
        / "customs_ai_agent"
        / "contracts"
        / "evidence"
        / "runtime-release-v2.lock.json"
    ).read_bytes()
    payloads["contracts/evidence/provider-runtime-lock.json"] = (
        ROOT / "contracts" / "evidence" / "provider-runtime-lock.json"
    ).read_bytes()
    payloads["src/customs_ai_agent/contracts/evidence/provider-runtime-lock.json"] = (
        ROOT / "src" / "customs_ai_agent" / "contracts" / "evidence" / "provider-runtime-lock.json"
    ).read_bytes()
    payloads["src/customs_ai_agent/contracts/evidence/v3/runtime-release-v3.lock.json"] = (
        ROOT
        / "src"
        / "customs_ai_agent"
        / "contracts"
        / "evidence"
        / "v3"
        / "runtime-release-v3.lock.json"
    ).read_bytes()
    payloads[BUILD_MANIFEST_SDIST_PATH] = TEST_BUILD_MANIFEST
    payloads.update(extra_files or {})
    root = f"{EXPECTED_IMPORT_PACKAGE}-{EXPECTED_VERSION}"
    with tarfile.open(path, mode="w:gz") as archive:
        for relative_name, payload in sorted(payloads.items()):
            member = tarfile.TarInfo(f"{root}/{relative_name}")
            member.size = len(payload)
            member.mode = 0o644
            archive.addfile(member, io.BytesIO(payload))
        for member, payload in extra_members or []:
            file_object = io.BytesIO(payload or b"") if member.isfile() else None
            archive.addfile(member, file_object)


def _tar_member(
    name: str,
    *,
    member_type: bytes = tarfile.REGTYPE,
    linkname: str = "",
    payload: bytes = b"malicious",
) -> tuple[tarfile.TarInfo, bytes | None]:
    member = tarfile.TarInfo(name)
    member.type = member_type
    member.linkname = linkname
    if member_type in {tarfile.REGTYPE, tarfile.AREGTYPE}:
        member.size = len(payload)
        return member, payload
    member.size = 0
    return member, None


def _source_git_repository(path: Path) -> Path:
    package_root = ROOT / "src" / "customs_ai_agent"
    for source in package_root.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(ROOT)
        if "__pycache__" in relative.parts or source.name == "build-manifest.json":
            continue
        target = path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(source.read_bytes())

    subprocess.run(["git", "init", str(path)], check=True, capture_output=True)
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Distribution verifier test"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "core.autocrlf", "false"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    subprocess.run(["git", "add", "src/customs_ai_agent"], cwd=path, check=True)
    subprocess.run(
        ["git", "commit", "-m", "source fixture"],
        cwd=path,
        check=True,
        capture_output=True,
    )
    return path


def _head_source_build_payloads(
    repository: Path,
) -> tuple[str, dict[str, bytes], bytes]:
    head = subprocess.run(
        ["git", "-C", str(repository), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    source_payloads = _git_archive_payloads(repository, head)
    manifest = build_manifest_bytes(
        git_sha=head,
        source_snapshot_sha256_value=source_snapshot_sha256(source_payloads),
    )
    package_payloads = {
        name.removeprefix("src/"): payload
        for name, payload in source_payloads.items()
        if name.startswith("src/customs_ai_agent/")
    }
    return head, package_payloads, manifest


def test_project_metadata_limits_runtime_to_python_311_and_next_major_versions() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))

    assert project["requires-python"] == ">=3.11,<3.12"
    assert set(project["dependencies"]) == {
        "anthropic>=0.117.0,<1",
        "cryptography>=48.0.1,<49",
        "fastapi>=0.115,<1",
        "jsonschema[format]>=4.23,<5",
        "langgraph>=0.2.0,<2",
        "pillow>=11.0,<13",
        "pypdfium2>=5.12,<6",
        "pydantic>=2.7,<3",
        "python-dotenv>=1.0,<2",
        "rapidfuzz>=3.9,<4",
        "uvicorn>=0.32,<1",
    }
    assert lock["requires-python"] == "==3.11.*"
    locked_project = next(
        package for package in lock["package"] if package["name"] == "customs-ai-agent"
    )
    assert {
        (requirement["name"], requirement["specifier"])
        for requirement in locked_project["metadata"]["requires-dist"]
    } == {
        ("anthropic", ">=0.117.0,<1"),
        ("cryptography", ">=48.0.1,<49"),
        ("fastapi", ">=0.115,<1"),
        ("jsonschema", ">=4.23,<5"),
        ("langgraph", ">=0.2.0,<2"),
        ("pillow", ">=11.0,<13"),
        ("pypdfium2", ">=5.12,<6"),
        ("pydantic", ">=2.7,<3"),
        ("python-dotenv", ">=1.0,<2"),
        ("rapidfuzz", ">=3.9,<4"),
        ("uvicorn", ">=0.32,<1"),
    }


def test_manifest_is_explicit_and_excludes_raw_documents_secrets_and_outputs() -> None:
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")

    for required in (
        "include .gitleaks.toml",
        "recursive-include .github *.yml *.yaml",
        "recursive-include contracts *.md *.json",
        "recursive-include evals *.py *.md *.json *.in *.txt",
        "exclude evals/ocr_benchmark_v1.json",
        "recursive-include fixtures *.md *.json",
        "recursive-include security *.toml",
        "recursive-include tests *.py",
        "include tests/fixtures/evidence_runtime_v2/README.md",
        "include tests/fixtures/evidence_runtime_v2/provider-runtime-lock.json",
        "recursive-include tools *.py *.json",
        "global-exclude .env .env.*",
        "global-exclude *.pdf *.png *.jpg *.jpeg *.webp *.bmp *.gif *.tif *.tiff",
        "prune _artifacts",
        "prune outputs",
    ):
        assert required in manifest


def test_public_runtime_facade_is_a_required_wheel_payload() -> None:
    assert "customs_ai_agent/runtime.py" in _WHEEL_REQUIRED_PACKAGE_FILES


def test_evidence_contract_runtime_files_are_required_wheel_payloads() -> None:
    required_evidence_files = {
        "customs_ai_agent/contracts/evidence.py",
        "customs_ai_agent/contracts/evidence/v1.lock.json",
        "customs_ai_agent/contracts/evidence/v1.schema.json",
    }

    assert required_evidence_files.issubset(_WHEEL_REQUIRED_PACKAGE_FILES)
    for package_name in required_evidence_files:
        assert (ROOT / "src" / package_name).is_file()


def test_distribution_verifier_accepts_bounded_wheel_and_sdist(tmp_path: Path) -> None:
    wheel = tmp_path / "customs_ai_agent-0.3.0-py3-none-any.whl"
    sdist = tmp_path / "customs_ai_agent-0.3.0.tar.gz"
    _write_wheel(wheel)
    _write_sdist(sdist)

    wheel_result = verify_wheel(wheel)
    sdist_result = verify_sdist(sdist)

    assert len(wheel_result["sha256"]) == 64
    assert int(wheel_result["member_count"]) > 0
    assert len(sdist_result["sha256"]) == 64
    assert int(sdist_result["member_count"]) > 0


def test_distribution_verifier_rejects_released_or_conformance_enabled_wheel(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "customs_ai_agent-0.3.0-py3-none-any.whl"
    lock_name = "customs_ai_agent/contracts/evidence/runtime-release-v2.lock.json"
    released_lock = json.loads(
        (
            ROOT
            / "src"
            / "customs_ai_agent"
            / "contracts"
            / "evidence"
            / "runtime-release-v2.lock.json"
        ).read_bytes()
    )
    released_lock["release_state"] = "released"
    released_lock["identity"] = {}
    released_lock.pop("unreleased_reason")
    _write_wheel(
        wheel,
        extra_files={
            lock_name: json.dumps(released_lock).encode(),
        },
    )

    with pytest.raises(
        DistributionVerificationError,
        match="must remain unreleased",
    ):
        verify_wheel(wheel)

    _write_wheel(
        wheel,
        extra_files={
            "customs_ai_agent/runtime.py": b"_provider_conformance_runtime_identity = True\n",
        },
    )
    with pytest.raises(
        DistributionVerificationError,
        match="evaluator-only conformance",
    ):
        verify_wheel(wheel)


def test_distribution_verifier_rejects_changed_or_released_canonical_sdist_locks(
    tmp_path: Path,
) -> None:
    sdist = tmp_path / "customs_ai_agent-0.3.0.tar.gz"
    root_lock_name = "contracts/evidence/runtime-release-v2.lock.json"
    packaged_lock_name = "src/customs_ai_agent/contracts/evidence/runtime-release-v2.lock.json"
    _write_sdist(
        sdist,
        extra_files={packaged_lock_name: b"{}\n"},
    )
    with pytest.raises(
        DistributionVerificationError,
        match="canonical production runtime release locks differ",
    ):
        verify_sdist(sdist)

    released_lock = json.loads(
        (ROOT / "contracts" / "evidence" / "runtime-release-v2.lock.json").read_bytes()
    )
    released_lock["release_state"] = "released"
    released_lock["identity"] = {}
    released_lock.pop("unreleased_reason")
    released_bytes = json.dumps(released_lock).encode()
    _write_sdist(
        sdist,
        extra_files={
            root_lock_name: released_bytes,
            packaged_lock_name: released_bytes,
        },
    )
    with pytest.raises(
        DistributionVerificationError,
        match="must remain unreleased",
    ):
        verify_sdist(sdist)


@pytest.mark.parametrize(
    "missing_file",
    [
        "customs_ai_agent/contracts/evidence.py",
        "customs_ai_agent/contracts/evidence/v1.lock.json",
        "customs_ai_agent/contracts/evidence/v1.schema.json",
    ],
)
def test_distribution_verifier_rejects_missing_evidence_contract_runtime_file(
    tmp_path: Path,
    missing_file: str,
) -> None:
    wheel = tmp_path / "customs_ai_agent-0.3.0-py3-none-any.whl"
    _write_wheel(wheel, excluded_files={missing_file})

    with pytest.raises(
        DistributionVerificationError,
        match="missing a required runtime file or package resource",
    ):
        verify_wheel(wheel)


def test_wheel_requires_release_only_build_manifest(tmp_path: Path) -> None:
    wheel = tmp_path / "customs_ai_agent-0.3.0-py3-none-any.whl"
    _write_wheel(wheel, excluded_files={BUILD_MANIFEST_WHEEL_PATH})

    with pytest.raises(
        DistributionVerificationError,
        match="release-only build manifest is missing",
    ):
        verify_wheel(wheel)


def test_wheel_payloads_are_bound_to_exact_git_source(tmp_path: Path) -> None:
    wheel = tmp_path / "customs_ai_agent-0.3.0-py3-none-any.whl"
    repository = _source_git_repository(tmp_path / "source-repository")
    head, package_payloads, manifest = _head_source_build_payloads(repository)
    _write_wheel(
        wheel,
        extra_files={
            **package_payloads,
            BUILD_MANIFEST_WHEEL_PATH: manifest,
        },
    )

    result = verify_wheel(
        wheel,
        expected_git_sha=head,
        source_root=repository,
    )
    assert result["git_sha"] == head

    package_payloads["customs_ai_agent/adapters/retrieval.py"] += b"\n# stale\n"
    _write_wheel(
        wheel,
        extra_files={
            **package_payloads,
            BUILD_MANIFEST_WHEEL_PATH: manifest,
        },
    )
    with pytest.raises(
        DistributionVerificationError,
        match="package payloads do not match",
    ):
        verify_wheel(
            wheel,
            expected_git_sha=head,
            source_root=repository,
        )


def test_release_builder_requires_exact_clean_git_head(tmp_path: Path) -> None:
    repository = _source_git_repository(tmp_path / "source-repository")
    head, _, _ = _head_source_build_payloads(repository)

    assert _assert_exact_clean_head(repository, head) == head

    (repository / "untracked.txt").write_text("dirty\n", encoding="utf-8")
    with pytest.raises(ReleaseBuildError, match="release_git_worktree_not_clean"):
        _assert_exact_clean_head(repository, head)
    with pytest.raises(ReleaseBuildError, match="release_git_head_mismatch"):
        _assert_exact_clean_head(repository, "0" * 40)


def test_release_builder_derives_reproducible_epoch_from_exact_commit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(release_builder, "_git", lambda *_args: "1700000000")

    epoch = release_builder._source_date_epoch(Path("."), "1" * 40)

    assert epoch == 1_700_000_000


def test_release_builder_rejects_invalid_commit_timestamp(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(release_builder, "_git", lambda *_args: "not-a-timestamp")

    with pytest.raises(ReleaseBuildError, match="release_git_timestamp_invalid"):
        release_builder._source_date_epoch(tmp_path, "1" * 40)


def test_release_builder_normalizes_environment_and_materialized_mtimes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    epoch = 1_700_000_000
    monkeypatch.setenv("SOURCE_DATE_EPOCH", "untrusted")
    environment = release_builder._reproducible_build_environment(epoch)
    source_tree = tmp_path / "source-tree"
    source_tree.mkdir()

    release_builder._materialize_snapshot(
        source_tree,
        {"src/customs_ai_agent/__init__.py": b"VERSION = 'test'\n"},
        source_date_epoch=epoch,
    )
    release_builder._normalize_directory_mtimes(
        source_tree,
        source_date_epoch=epoch,
    )

    materialized = source_tree / "src" / "customs_ai_agent" / "__init__.py"
    assert environment["SOURCE_DATE_EPOCH"] == str(epoch)
    assert environment["PYTHONHASHSEED"] == "0"
    assert environment["TZ"] == "UTC"
    assert int(materialized.stat().st_mtime) == epoch
    assert int((source_tree / "src").stat().st_mtime) == epoch
    assert int(source_tree.stat().st_mtime) == epoch


def test_release_builder_canonicalizes_sdist_tar_and_gzip_metadata(
    tmp_path: Path,
) -> None:
    epoch = 1_700_000_000
    payloads = {
        "customs_ai_agent-0.3.0/README.md": b"release\n",
        "customs_ai_agent-0.3.0/src/customs_ai_agent/__init__.py": b"VERSION = 'test'\n",
    }
    outputs: list[bytes] = []
    for build_number in (1, 2):
        sdist = tmp_path / f"build-{build_number}.tar.gz"
        with tarfile.open(sdist, mode="w:gz") as archive:
            directory = tarfile.TarInfo("customs_ai_agent-0.3.0/src/")
            directory.type = tarfile.DIRTYPE
            directory.mode = 0o777
            directory.mtime = epoch + build_number
            archive.addfile(directory)
            for name, payload in reversed(payloads.items()):
                member = tarfile.TarInfo(name)
                member.size = len(payload)
                member.mode = 0o777
                member.mtime = epoch + build_number
                member.uid = build_number
                member.gid = build_number
                archive.addfile(member, io.BytesIO(payload))

        release_builder._canonicalize_sdist(
            sdist,
            source_date_epoch=epoch,
        )
        outputs.append(sdist.read_bytes())

        with tarfile.open(sdist, mode="r:gz") as archive:
            members = archive.getmembers()
            assert [member.name for member in members] == sorted(payloads)
            assert all(member.isfile() for member in members)
            assert all(member.mode == 0o644 for member in members)
            assert all(member.mtime == epoch for member in members)
            assert all(member.uid == member.gid == 0 for member in members)

    assert outputs[0] == outputs[1]


def test_release_builder_rejects_non_regular_sdist_member(tmp_path: Path) -> None:
    sdist = tmp_path / "unsafe.tar.gz"
    with tarfile.open(sdist, mode="w:gz") as archive:
        member = tarfile.TarInfo("customs_ai_agent-0.3.0/link")
        member.type = tarfile.SYMTYPE
        member.linkname = "../../outside"
        archive.addfile(member)

    with pytest.raises(
        ReleaseBuildError,
        match="release_sdist_canonicalization_failed",
    ):
        release_builder._canonicalize_sdist(
            sdist,
            source_date_epoch=1_700_000_000,
        )


def test_release_builder_resolves_executable_to_regular_absolute_file(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(release_builder.shutil, "which", lambda _: sys.executable)

    resolved = release_builder._resolve_executable("uv", error_code="invalid")

    assert Path(resolved) == Path(sys.executable).resolve(strict=True)
    assert Path(resolved).is_absolute()
    assert Path(resolved).is_file()


@pytest.mark.parametrize("discovered", [None, "."])
def test_release_builder_rejects_missing_or_non_file_executable(
    discovered: str | None,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(release_builder.shutil, "which", lambda _: discovered)

    with pytest.raises(ReleaseBuildError, match="release_executable_invalid"):
        release_builder._resolve_executable(
            "uv",
            error_code="release_executable_invalid",
        )


def test_sdist_payloads_are_bound_to_exact_git_source(tmp_path: Path) -> None:
    sdist = tmp_path / "customs_ai_agent-0.3.0.tar.gz"
    repository = _source_git_repository(tmp_path / "source-repository")
    head, package_payloads, manifest = _head_source_build_payloads(repository)
    _write_sdist(
        sdist,
        extra_files={
            **{f"src/{name}": payload for name, payload in package_payloads.items()},
            BUILD_MANIFEST_SDIST_PATH: manifest,
        },
    )

    result = verify_sdist(
        sdist,
        expected_git_sha=head,
        source_root=repository,
    )
    assert result["git_sha"] == head


def test_uv_hashless_direct_url_requires_exact_wheel_and_installed_record(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "customs_ai_agent-0.3.0-py3-none-any.whl"
    _write_wheel(wheel)
    site_packages = tmp_path / "venv" / "lib" / "python3.11" / "site-packages"
    distribution = _write_uv_install_tree(wheel, site_packages)
    expected_wheel_sha256 = hashlib.sha256(wheel.read_bytes()).hexdigest()
    direct_url = json.loads(distribution.read_text("direct_url.json") or "{}")

    assert (
        _direct_url_wheel_path(
            direct_url,
            expected_wheel_sha256=expected_wheel_sha256,
        )
        == wheel.resolve()
    )
    result = _verify_installed_record_against_wheel(
        distribution,
        wheel_path=wheel,
    )

    assert int(result["installed_record_count"]) > 0
    assert result["console_records"] == (
        "../../../bin/customs-ai-agent,../../../bin/customs-ai-agent-serve"
    )


def test_uv_direct_url_rejects_conflicting_archive_hash(tmp_path: Path) -> None:
    wheel = tmp_path / "customs_ai_agent-0.3.0-py3-none-any.whl"
    _write_wheel(wheel)
    direct_url = {
        "url": wheel.resolve().as_uri(),
        "archive_info": {"hashes": {"sha256": "0" * 64}},
    }

    with pytest.raises(
        DistributionVerificationError,
        match="origin hash",
    ):
        _direct_url_wheel_path(
            direct_url,
            expected_wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
        )


def test_uv_hashless_direct_url_rejects_a_different_wheel_path(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "customs_ai_agent-0.3.0-py3-none-any.whl"
    other_wheel = tmp_path / "other" / wheel.name
    other_wheel.parent.mkdir()
    _write_wheel(wheel)
    other_wheel.write_bytes(wheel.read_bytes())
    direct_url = {
        "url": other_wheel.resolve().as_uri(),
        "archive_info": {},
    }

    with pytest.raises(
        DistributionVerificationError,
        match="does not reference the attested wheel",
    ):
        _verify_wheel_origin(
            direct_url,
            wheel_path=wheel,
            expected_wheel_sha256=hashlib.sha256(wheel.read_bytes()).hexdigest(),
        )


def test_installed_record_rejects_tampered_wheel_payload(tmp_path: Path) -> None:
    wheel = tmp_path / "customs_ai_agent-0.3.0-py3-none-any.whl"
    _write_wheel(wheel)
    site_packages = tmp_path / "venv" / "lib" / "python3.11" / "site-packages"
    distribution = _write_uv_install_tree(wheel, site_packages)
    (site_packages / "customs_ai_agent" / "__init__.py").write_text(
        "tampered = True\n",
        encoding="utf-8",
    )

    with pytest.raises(
        DistributionVerificationError,
        match="does not match its RECORD",
    ):
        _verify_installed_record_against_wheel(
            distribution,
            wheel_path=wheel,
        )


def test_installed_record_rejects_unapproved_extra_file(tmp_path: Path) -> None:
    wheel = tmp_path / "customs_ai_agent-0.3.0-py3-none-any.whl"
    _write_wheel(wheel)
    site_packages = tmp_path / "venv" / "lib" / "python3.11" / "site-packages"
    distribution = _write_uv_install_tree(wheel, site_packages)
    rogue_name = "customs_ai_agent/rogue.py"
    rogue_payload = b"raise RuntimeError('unexpected')\n"
    rogue_path = site_packages / "customs_ai_agent" / "rogue.py"
    rogue_path.write_bytes(rogue_payload)
    record_path = site_packages / DIST_INFO / "RECORD"
    record_text = record_path.read_text(encoding="utf-8")
    record_path.write_text(
        record_text
        + (f"{rogue_name},sha256={_record_digest(rogue_payload)},{len(rogue_payload)}\n"),
        encoding="utf-8",
    )

    with pytest.raises(
        DistributionVerificationError,
        match="unapproved file",
    ):
        _verify_installed_record_against_wheel(
            distribution,
            wheel_path=wheel,
        )


def test_distribution_certification_source_is_retained_but_inactive_in_general_ci() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "build_release_distributions" not in workflow
    assert "verify_distribution.py" not in workflow
    assert "agent-wheel-requirement.txt" not in workflow
    assert "docker build" not in workflow
    assert (ROOT / "tools" / "build_release_distributions.py").is_file()
    assert (ROOT / "tools" / "verify_distribution.py").is_file()
    assert (ROOT / "Dockerfile").is_file()


def test_agent_container_build_frontend_and_base_image_are_digest_pinned() -> None:
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    lines = dockerfile.splitlines()

    assert lines[0] == (
        "# syntax=docker/dockerfile:1.7@sha256:"
        "a57df69d0ea827fb7266491f2813635de6f17269be881f696fbfdf2d83dda33e"
    )
    python_image = next(line for line in lines if line.startswith("ARG PYTHON_IMAGE="))
    assert python_image == (
        "ARG PYTHON_IMAGE=python:3.11.15-slim-bookworm@sha256:"
        "b18992999dbe963a45a8a4da40ac2b1975be1a776d939d098c647482bcad5cba"
    )
    assert "FROM ${PYTHON_IMAGE}" in lines


@pytest.mark.parametrize(
    "member_name",
    [
        "customs_ai_agent/private/.env",
        "customs_ai_agent/fixtures/raw-invoice.pdf",
        "C:" + "/Users/" + "private/secret.txt",
        "../outside.txt",
    ],
)
def test_distribution_verifier_rejects_secret_raw_or_unsafe_wheel_member(
    tmp_path: Path,
    member_name: str,
) -> None:
    wheel = tmp_path / "customs_ai_agent-0.3.0-py3-none-any.whl"
    _write_wheel(wheel, extra_files={member_name: b"not releasable"})

    with pytest.raises(DistributionVerificationError):
        verify_wheel(wheel)


def test_distribution_verifier_rejects_local_or_sibling_dependency(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "customs_ai_agent-0.3.0-py3-none-any.whl"
    metadata = METADATA.removesuffix(b"\n\n") + (
        b"\nRequires-Dist: customs-data-platform @ file:///workspace/customs-data-platform\n\n"
    )
    _write_wheel(wheel, metadata=metadata)

    with pytest.raises(
        DistributionVerificationError,
        match="direct, local, or sibling dependency",
    ):
        verify_wheel(wheel)


def test_distribution_verifier_rejects_unbounded_runtime_dependency(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "customs_ai_agent-0.3.0-py3-none-any.whl"
    metadata = METADATA.replace(
        b"Requires-Dist: anthropic<1,>=0.117.0",
        b"Requires-Dist: anthropic>=0.117.0",
    )
    _write_wheel(wheel, metadata=metadata)

    with pytest.raises(
        DistributionVerificationError,
        match="runtime dependency set",
    ):
        verify_wheel(wheel)


def test_distribution_verifier_rejects_missing_jsonschema_runtime_dependency(
    tmp_path: Path,
) -> None:
    wheel = tmp_path / "customs_ai_agent-0.3.0-py3-none-any.whl"
    metadata = METADATA.replace(
        b"Requires-Dist: jsonschema[format]<5,>=4.23\n",
        b"",
    )
    _write_wheel(wheel, metadata=metadata)

    with pytest.raises(
        DistributionVerificationError,
        match="runtime dependency set",
    ):
        verify_wheel(wheel)


def test_distribution_verifier_rejects_raw_or_secret_sdist_member(
    tmp_path: Path,
) -> None:
    sdist = tmp_path / "customs_ai_agent-0.3.0.tar.gz"
    _write_sdist(sdist, extra_files={"fixtures/raw-invoice.png": b"raw"})

    with pytest.raises(
        DistributionVerificationError,
        match="raw document, image, or secret-key file",
    ):
        verify_sdist(sdist)


def test_distribution_verifier_rejects_invalidated_legacy_ocr_descriptor(
    tmp_path: Path,
) -> None:
    sdist = tmp_path / "customs_ai_agent-0.3.0.tar.gz"
    _write_sdist(
        sdist,
        extra_files={"evals/ocr_benchmark_v1.json": b"{}\n"},
    )

    with pytest.raises(
        DistributionVerificationError,
        match="invalidated historical release input",
    ):
        verify_sdist(sdist)


@pytest.mark.parametrize(
    ("member_type", "member_name", "linkname"),
    [
        (tarfile.SYMTYPE, "docs/symlink", "../../outside"),
        (tarfile.LNKTYPE, "docs/hardlink", "../../outside"),
        (tarfile.CHRTYPE, "docs/character-device", ""),
        (tarfile.BLKTYPE, "docs/block-device", ""),
        (tarfile.FIFOTYPE, "docs/fifo", ""),
        (tarfile.REGTYPE, "/absolute.py", ""),
        (tarfile.REGTYPE, "../traversal.py", ""),
        (tarfile.REGTYPE, "docs\\..\\traversal.py", ""),
        (tarfile.REGTYPE, "README.md", ""),
        (tarfile.REGTYPE, "readme.md", ""),
        (tarfile.DIRTYPE, "../other-root", ""),
        (tarfile.REGTYPE, "docs/control-\n-name.py", ""),
    ],
)
def test_distribution_verifier_rejects_non_regular_duplicate_or_unsafe_sdist_member(
    tmp_path: Path,
    member_type: bytes,
    member_name: str,
    linkname: str,
) -> None:
    sdist = tmp_path / "customs_ai_agent-0.3.0.tar.gz"
    root = f"{EXPECTED_IMPORT_PACKAGE}-{EXPECTED_VERSION}"
    full_name = member_name if member_name.startswith(("/", "..")) else f"{root}/{member_name}"
    _write_sdist(
        sdist,
        extra_members=[
            _tar_member(
                full_name,
                member_type=member_type,
                linkname=linkname,
            )
        ],
    )

    with pytest.raises(DistributionVerificationError):
        verify_sdist(sdist)


def test_sdist_member_inventory_is_validated_before_any_payload_is_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sdist = tmp_path / "customs_ai_agent-0.3.0.tar.gz"
    root = f"{EXPECTED_IMPORT_PACKAGE}-{EXPECTED_VERSION}"
    _write_sdist(
        sdist,
        extra_members=[
            _tar_member(
                f"{root}/docs/symlink",
                member_type=tarfile.SYMTYPE,
                linkname="../../outside",
            )
        ],
    )

    def unexpected_extract(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("payload read happened before tar member validation")

    monkeypatch.setattr(tarfile.TarFile, "extractfile", unexpected_extract)

    with pytest.raises(DistributionVerificationError):
        verify_sdist(sdist)


def test_distribution_verifier_rejects_incomplete_sdist(tmp_path: Path) -> None:
    sdist = tmp_path / "customs_ai_agent-0.3.0.tar.gz"
    root = f"{EXPECTED_IMPORT_PACKAGE}-{EXPECTED_VERSION}"
    with tarfile.open(sdist, mode="w:gz") as archive:
        member = tarfile.TarInfo(f"{root}/PKG-INFO")
        member.size = len(METADATA)
        archive.addfile(member, io.BytesIO(METADATA))

    with pytest.raises(DistributionVerificationError, match="missing required"):
        verify_sdist(sdist)
