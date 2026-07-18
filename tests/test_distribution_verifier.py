from __future__ import annotations

import csv
import hashlib
import io
import json
import tarfile
import tomllib
import zipfile
from pathlib import Path
from typing import Any

import pytest

from tools.verify_distribution import (
    EXPECTED_CONSOLE_ENTRY_POINT,
    EXPECTED_IMPORT_PACKAGE,
    EXPECTED_VERSION,
    DistributionVerificationError,
    _SDIST_REQUIRED_FILES,
    _WHEEL_REQUIRED_PACKAGE_FILES,
    _direct_url_wheel_path,
    _record_digest,
    _verify_installed_record_against_wheel,
    _verify_wheel_origin,
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
    "Requires-Dist: langgraph<2,>=0.2.0\n"
    "Requires-Dist: pillow<13,>=11.0\n"
    "Requires-Dist: pydantic<3,>=2.7\n"
    "Requires-Dist: pypdfium2<6,>=5.12\n"
    "Requires-Dist: python-dotenv<2,>=1.0\n"
    "Requires-Dist: rapidfuzz<4,>=3.9\n"
    "\n"
).encode()


def _write_wheel(
    path: Path,
    *,
    metadata: bytes = METADATA,
    extra_files: dict[str, bytes] | None = None,
) -> None:
    payloads: dict[str, bytes] = {
        name: (
            b"{}\n" if name.endswith(".json") else b"prompt\n" if name.endswith(".txt") else b"\n"
        )
        for name in _WHEEL_REQUIRED_PACKAGE_FILES
    }
    payloads.update(
        {
            f"{DIST_INFO}/METADATA": metadata,
            f"{DIST_INFO}/WHEEL": (
                b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n"
            ),
            f"{DIST_INFO}/entry_points.txt": (
                f"[console_scripts]\n{EXPECTED_CONSOLE_ENTRY_POINT}\n".encode()
            ),
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


def test_project_metadata_limits_runtime_to_python_311_and_next_major_versions() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))

    assert project["requires-python"] == ">=3.11,<3.12"
    assert set(project["dependencies"]) == {
        "anthropic>=0.117.0,<1",
        "langgraph>=0.2.0,<2",
        "pillow>=11.0,<13",
        "pypdfium2>=5.12,<6",
        "pydantic>=2.7,<3",
        "python-dotenv>=1.0,<2",
        "rapidfuzz>=3.9,<4",
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
        ("langgraph", ">=0.2.0,<2"),
        ("pillow", ">=11.0,<13"),
        ("pypdfium2", ">=5.12,<6"),
        ("pydantic", ">=2.7,<3"),
        ("python-dotenv", ">=1.0,<2"),
        ("rapidfuzz", ">=3.9,<4"),
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
        "recursive-include tools *.py",
        "global-exclude .env .env.*",
        "global-exclude *.pdf *.png *.jpg *.jpeg *.webp *.bmp *.gif *.tif *.tiff",
        "prune _artifacts",
        "prune outputs",
    ):
        assert required in manifest


def test_public_runtime_facade_is_a_required_wheel_payload() -> None:
    assert "customs_ai_agent/runtime.py" in _WHEEL_REQUIRED_PACKAGE_FILES


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
    assert result["console_record"] == "../../../bin/customs-ai-agent"


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


def test_ci_hash_pins_wheel_install_and_supplies_archive_to_verifier() -> None:
    workflow = (ROOT / ".github" / "workflows" / "ci.yml").read_text(encoding="utf-8")

    assert "agent-wheel-requirement.txt" in workflow
    assert "--require-hashes --requirements dist/agent-wheel-requirement.txt" in workflow
    assert '--wheel "$WHEEL"' in workflow


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
