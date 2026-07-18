"""Fail-closed release checks for the Agent wheel, sdist, and clean install.

This script intentionally uses only the Python standard library until
``verify_installed_distribution`` imports the already-installed Agent wheel.
It is therefore safe to run both from the locked development environment and
from a neutral directory with ``python -I`` in the clean runtime environment.
"""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import json
import re
import sys
import tarfile
import zipfile
from email.parser import BytesParser
from pathlib import Path, PurePosixPath
from typing import Final, Iterable, Mapping
from urllib.parse import unquote, urlsplit
from urllib.request import url2pathname


EXPECTED_DISTRIBUTION: Final = "customs-ai-agent"
EXPECTED_IMPORT_PACKAGE: Final = "customs_ai_agent"
EXPECTED_VERSION: Final = "0.3.0"
EXPECTED_REQUIRES_PYTHON: Final = frozenset({">=3.11", "<3.12"})
EXPECTED_REQUIRES_DIST: Final = frozenset(
    {
        "anthropic<1,>=0.117.0",
        "jsonschema[format]<5,>=4.23",
        "langgraph<2,>=0.2.0",
        "pillow<13,>=11.0",
        "pypdfium2<6,>=5.12",
        "pydantic<3,>=2.7",
        "python-dotenv<2,>=1.0",
        "rapidfuzz<4,>=3.9",
    }
)

EXPECTED_SCHEMA_RESOURCES: Final = frozenset(
    {
        "schemas/backend/v1/review-request.schema.json",
        "schemas/backend/v1/review-result.schema.json",
        "schemas/backend/v2/review-request.schema.json",
        "schemas/backend/v2/review-result.schema.json",
        "schemas/internal/v1/claude-crop-read-candidate.schema.json",
        "schemas/internal/v1/claude-text-candidate.schema.json",
        "schemas/internal/v1/claude-vision-candidate.schema.json",
        "schemas/internal/v1/ocr-layout-ledger.schema.json",
        "schemas/internal/v2/ocr-layout-ledger.schema.json",
        "schemas/local/v1/review-manifest.schema.json",
    }
)
EXPECTED_PROMPT_RESOURCES: Final = frozenset(
    {
        "document_analysis_v1.txt",
        "document_vision_analysis_v1.txt",
    }
)
EXPECTED_CONSOLE_ENTRY_POINT: Final = "customs-ai-agent = customs_ai_agent.cli:main"
_INSTALLER_OWNED_DIST_INFO_FILES: Final = frozenset(
    {
        "INSTALLER",
        "REQUESTED",
        "direct_url.json",
        "uv_cache.json",
    }
)
_ALLOWED_CONSOLE_RECORD_NAMES: Final = frozenset(
    {
        "../../../bin/customs-ai-agent",
        "../../Scripts/customs-ai-agent.exe",
    }
)

_WHEEL_REQUIRED_PACKAGE_FILES: Final = frozenset(
    {
        "customs_ai_agent/__init__.py",
        "customs_ai_agent/__main__.py",
        "customs_ai_agent/cli.py",
        "customs_ai_agent/ocr_freeze.py",
        "customs_ai_agent/ocr_worker.py",
        "customs_ai_agent/ocr_release.py",
        "customs_ai_agent/ocr_release_pin.py",
        "customs_ai_agent/pdf_worker.py",
        "customs_ai_agent/runtime.py",
        "customs_ai_agent/contracts/evidence.py",
        "customs_ai_agent/contracts/evidence/v1.lock.json",
        "customs_ai_agent/contracts/evidence/v1.schema.json",
        *{f"customs_ai_agent/prompts/{name}" for name in EXPECTED_PROMPT_RESOURCES},
        *{f"customs_ai_agent/contracts/{name}" for name in EXPECTED_SCHEMA_RESOURCES},
    }
)
_SDIST_REQUIRED_FILES: Final = frozenset(
    {
        "README.md",
        "SECURITY.md",
        "CONTRIBUTING.md",
        "MANIFEST.in",
        "pyproject.toml",
        "uv.lock",
        ".gitleaks.toml",
        ".github/workflows/ci.yml",
        "contracts/evidence/v1.lock.json",
        "contracts/evidence/v1/rag_retrieval_v1.schema.json",
        "contracts/agent/v1/review_request.schema.json",
        "contracts/agent/v1/review_result.schema.json",
        "contracts/agent/v2/review_request.schema.json",
        "contracts/agent/v2/review_result.schema.json",
        "evals/ocr_environments/v1/docling.requirements.in",
        "evals/ocr_environments/v1/docling.requirements.lock.txt",
        "evals/ocr_environments/v1/model-assets.lock.json",
        "evals/ocr_environments/v1/paddle.requirements.in",
        "evals/ocr_environments/v1/paddle.requirements.lock.txt",
        "evals/release_holdout_protocol.py",
        ("evals/release_holdout_protocol/v1/sealed-bundle-public-manifest.schema.json"),
        "evals/run_ocr_benchmark.py",
        "evals/run_release_holdout_validation.py",
        "fixtures/trade_documents_images/v1/dataset.lock.json",
        "docs/DATA_PROVIDER_HANDOFF.md",
        "docs/SECURITY_GATE.md",
        "docs/OCR_RELEASE_FREEZE.md",
        "security/toolchain.lock.toml",
        "tests/test_packaged_contracts.py",
        "tests/test_distribution_verifier.py",
        "tests/test_security_gate.py",
        "tests/test_ocr_freeze.py",
        "tools/run_security_gate.py",
        "tools/verify_ocr_release_freeze.py",
        "tools/verify_distribution.py",
        *{f"src/customs_ai_agent/prompts/{name}" for name in EXPECTED_PROMPT_RESOURCES},
        *{f"src/customs_ai_agent/contracts/{name}" for name in EXPECTED_SCHEMA_RESOURCES},
    }
)
_SDIST_FORBIDDEN_EXACT_FILES: Final = frozenset(
    {
        # This historical descriptor references a compromised pre-commit
        # holdout. Keep it in the source checkout for audit only; never ship it
        # as an apparently usable release input.
        "evals/ocr_benchmark_v1.json",
    }
)
_SDIST_ALLOWED_ROOT_FILES: Final = frozenset(
    {
        "README.md",
        "SECURITY.md",
        "CONTRIBUTING.md",
        "MANIFEST.in",
        "PKG-INFO",
        "pyproject.toml",
        "setup.cfg",
        "uv.lock",
        ".gitleaks.toml",
    }
)
_SDIST_ALLOWED_ROOT_DIRECTORIES: Final = frozenset(
    {
        "contracts",
        "docs",
        "evals",
        "fixtures",
        "prompts",
        "security",
        "src",
        "tests",
        "tools",
        ".github",
    }
)
_FORBIDDEN_MEDIA_SUFFIXES: Final = frozenset(
    {
        ".bmp",
        ".gif",
        ".jpeg",
        ".jpg",
        ".jks",
        ".key",
        ".p12",
        ".pdf",
        ".pem",
        ".pfx",
        ".png",
        ".tif",
        ".tiff",
        ".webp",
    }
)
_TEXT_SUFFIXES: Final = frozenset(
    {
        "",
        ".cfg",
        ".in",
        ".ini",
        ".json",
        ".lock",
        ".md",
        ".py",
        ".toml",
        ".txt",
        ".yaml",
        ".yml",
    }
)
_SECRET_PATTERNS: Final = (
    re.compile(rb"sk-ant-[A-Za-z0-9_-]{20,}"),
    re.compile(rb"AKIA[0-9A-Z]{16}"),
    re.compile(rb"gh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(rb"-----BEGIN (?:RSA |OPENSSH |EC |DSA )?PRIVATE KEY-----"),
)
_ABSOLUTE_PRIVATE_PATH_PATTERNS: Final = (
    re.compile(rb"(?i)[A-Z]:[\\/]+Users[\\/]+[^\\/\r\n]+"),
    # Build the POSIX prefix in fragments so this verifier does not match its
    # own source when it is intentionally shipped in the self-contained sdist.
    re.compile(rb"/" + b"home/" + rb"[^/\r\n]+/"),
)
_MAX_MEMBER_BYTES: Final = 5 * 1024 * 1024
_MAX_ARCHIVE_BYTES: Final = 24 * 1024 * 1024


class DistributionVerificationError(RuntimeError):
    """A content-free release verification failure."""


def _normalize_member_names(names: Iterable[str]) -> tuple[str, ...]:
    normalized: list[str] = []
    casefolded: set[str] = set()
    for raw_name in names:
        name = raw_name.replace("\\", "/")
        if not name or name.endswith("/"):
            continue
        path = PurePosixPath(name)
        if (
            path.is_absolute()
            or ".." in path.parts
            or "." in path.parts
            or re.match(r"^[A-Za-z]:", name)
        ):
            raise DistributionVerificationError(
                "distribution contains an unsafe archive member path"
            )
        canonical = path.as_posix()
        folded = canonical.casefold()
        if folded in casefolded:
            raise DistributionVerificationError(
                "distribution contains duplicate or case-colliding members"
            )
        casefolded.add(folded)
        normalized.append(canonical)
    return tuple(normalized)


def _validate_sdist_members(
    members: Iterable[tarfile.TarInfo],
) -> tuple[tuple[tarfile.TarInfo, str], ...]:
    """Validate every tar member before any member payload is read.

    An sdist is source material, not a filesystem image. Only ordinary files
    and directories with canonical POSIX-relative names are permitted.
    """

    validated: list[tuple[tarfile.TarInfo, str]] = []
    casefolded: set[str] = set()
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
            raise DistributionVerificationError(
                "sdist contains a non-regular or unsafe archive member"
            )
        name = raw_name.rstrip("/") if member.isdir() else raw_name
        raw_parts = name.split("/")
        path = PurePosixPath(name)
        if (
            not name
            or path.is_absolute()
            or any(part in {"", ".", ".."} for part in raw_parts)
            or re.match(r"^[A-Za-z]:", name)
            or (member.isdir() and member.size != 0)
            or (member.isfile() and raw_name.endswith("/"))
        ):
            raise DistributionVerificationError("sdist contains an unsafe archive member path")
        canonical = path.as_posix()
        folded = canonical.casefold()
        if folded in casefolded:
            raise DistributionVerificationError(
                "sdist contains duplicate or case-colliding archive members"
            )
        casefolded.add(folded)
        validated.append((member, canonical))
    if not validated:
        raise DistributionVerificationError("sdist archive is empty")
    return tuple(validated)


def _assert_safe_member_name(relative_name: str) -> None:
    path = PurePosixPath(relative_name)
    basename = path.name.casefold()
    suffix = path.suffix.casefold()
    if basename == ".env" or basename.startswith(".env."):
        raise DistributionVerificationError("distribution contains an environment file")
    if suffix in _FORBIDDEN_MEDIA_SUFFIXES:
        raise DistributionVerificationError(
            "distribution contains a raw document, image, or secret-key file"
        )
    if "__pycache__" in path.parts or suffix in {".pyc", ".pyo"}:
        raise DistributionVerificationError("distribution contains a Python cache artifact")
    forbidden_parts = {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        ".venv",
        "_artifacts",
        "build",
        "dist",
        "outputs",
    }
    if forbidden_parts.intersection(part.casefold() for part in path.parts):
        raise DistributionVerificationError(
            "distribution contains a local build, cache, output, or artifact path"
        )


def _assert_safe_member_content(relative_name: str, payload: bytes) -> None:
    if len(payload) > _MAX_MEMBER_BYTES:
        raise DistributionVerificationError("distribution member exceeds the release size limit")
    suffix = PurePosixPath(relative_name).suffix.casefold()
    if suffix not in _TEXT_SUFFIXES:
        return
    for pattern in _SECRET_PATTERNS:
        if pattern.search(payload):
            raise DistributionVerificationError(
                "distribution contains a high-confidence secret pattern"
            )
    for pattern in _ABSOLUTE_PRIVATE_PATH_PATTERNS:
        if pattern.search(payload):
            raise DistributionVerificationError(
                "distribution contains a private absolute filesystem path"
            )


def _assert_archive_size_limit(payloads: Mapping[str, bytes]) -> None:
    if sum(len(payload) for payload in payloads.values()) > _MAX_ARCHIVE_BYTES:
        raise DistributionVerificationError(
            "distribution uncompressed content exceeds the release size limit"
        )


def _metadata_from_bytes(payload: bytes) -> None:
    metadata = BytesParser().parsebytes(payload)
    if metadata.get("Name") != EXPECTED_DISTRIBUTION:
        raise DistributionVerificationError("distribution metadata name is invalid")
    if metadata.get("Version") != EXPECTED_VERSION:
        raise DistributionVerificationError("distribution metadata version is invalid")
    python_specifiers = frozenset(
        value.strip()
        for value in str(metadata.get("Requires-Python", "")).split(",")
        if value.strip()
    )
    if python_specifiers != EXPECTED_REQUIRES_PYTHON:
        raise DistributionVerificationError(
            "distribution metadata overstates the supported Python range"
        )
    requirements = metadata.get_all("Requires-Dist") or []
    for requirement in requirements:
        lowered = requirement.casefold()
        if (
            " @ " in lowered
            or "file:" in lowered
            or "customs-data-platform" in lowered
            or "customs-backend" in lowered
            or "operational_db" in lowered
        ):
            raise DistributionVerificationError(
                "distribution metadata contains a direct, local, or sibling dependency"
            )
    normalized_requirements = frozenset(
        requirement.replace(" ", "").casefold() for requirement in requirements
    )
    if normalized_requirements != EXPECTED_REQUIRES_DIST:
        raise DistributionVerificationError(
            "distribution metadata runtime dependency set is invalid"
        )


def _record_digest(payload: bytes) -> str:
    return base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()


def _parse_record(
    record_text: str,
    *,
    distribution_kind: str,
) -> dict[str, tuple[str, str]]:
    try:
        rows = list(csv.reader(record_text.splitlines()))
    except csv.Error:
        raise DistributionVerificationError(f"{distribution_kind} RECORD is invalid") from None
    recorded: dict[str, tuple[str, str]] = {}
    casefolded: set[str] = set()
    for row in rows:
        if len(row) != 3:
            raise DistributionVerificationError(f"{distribution_kind} RECORD row is invalid")
        name = row[0].replace("\\", "/")
        if (
            not name
            or "\x00" in name
            or "\r" in name
            or "\n" in name
            or name.startswith("/")
            or re.match(r"^[A-Za-z]:", name)
        ):
            raise DistributionVerificationError(f"{distribution_kind} RECORD path is invalid")
        folded = name.casefold()
        if name in recorded or folded in casefolded:
            raise DistributionVerificationError(f"{distribution_kind} RECORD contains duplicates")
        recorded[name] = (row[1], row[2])
        casefolded.add(folded)
    return recorded


def _verify_wheel_record(
    payloads: Mapping[str, bytes],
    *,
    dist_info_prefix: str,
) -> None:
    record_name = f"{dist_info_prefix}RECORD"
    try:
        record_text = payloads[record_name].decode("utf-8")
    except (KeyError, UnicodeDecodeError):
        raise DistributionVerificationError("wheel RECORD is missing or invalid") from None
    recorded = _parse_record(record_text, distribution_kind="wheel")
    if set(recorded) != set(payloads):
        raise DistributionVerificationError("wheel RECORD inventory does not match the archive")
    for name, payload in payloads.items():
        digest, size = recorded[name]
        if name == record_name:
            if digest or size:
                raise DistributionVerificationError(
                    "wheel RECORD must leave its own hash and size empty"
                )
            continue
        if digest != f"sha256={_record_digest(payload)}" or size != str(len(payload)):
            raise DistributionVerificationError(
                "wheel RECORD hash or size does not match archive content"
            )


def verify_wheel(path: Path) -> dict[str, str]:
    """Verify exact wheel inventory, metadata, entry point, and RECORD."""

    try:
        with zipfile.ZipFile(path) as archive:
            names = _normalize_member_names(archive.namelist())
            payloads = {name: archive.read(name) for name in names}
    except (OSError, zipfile.BadZipFile, KeyError):
        raise DistributionVerificationError("wheel archive is unreadable") from None
    _assert_archive_size_limit(payloads)
    for name, payload in payloads.items():
        _assert_safe_member_name(name)
        _assert_safe_member_content(name, payload)

    dist_info_candidates = {
        name.split("/", 1)[0] + "/" for name in names if name.endswith(".dist-info/METADATA")
    }
    if len(dist_info_candidates) != 1:
        raise DistributionVerificationError("wheel dist-info inventory is invalid")
    dist_info_prefix = next(iter(dist_info_candidates))
    if not re.fullmatch(
        rf"{EXPECTED_IMPORT_PACKAGE}-{re.escape(EXPECTED_VERSION)}\.dist-info/",
        dist_info_prefix,
    ):
        raise DistributionVerificationError("wheel dist-info identity is invalid")
    for name in names:
        if not (
            name.startswith(f"{EXPECTED_IMPORT_PACKAGE}/") or name.startswith(dist_info_prefix)
        ):
            raise DistributionVerificationError(
                "wheel contains a file outside the package and dist-info allowlist"
            )
    required = {
        *_WHEEL_REQUIRED_PACKAGE_FILES,
        f"{dist_info_prefix}METADATA",
        f"{dist_info_prefix}WHEEL",
        f"{dist_info_prefix}entry_points.txt",
        f"{dist_info_prefix}RECORD",
    }
    missing = required.difference(payloads)
    if missing:
        raise DistributionVerificationError(
            "wheel is missing a required runtime file or package resource"
        )

    _metadata_from_bytes(payloads[f"{dist_info_prefix}METADATA"])
    try:
        entry_points = payloads[f"{dist_info_prefix}entry_points.txt"].decode("utf-8")
    except UnicodeDecodeError:
        raise DistributionVerificationError("wheel entry point metadata is invalid") from None
    normalized_entry_points = {
        line.strip()
        for line in entry_points.splitlines()
        if line.strip() and not line.lstrip().startswith("[")
    }
    if normalized_entry_points != {EXPECTED_CONSOLE_ENTRY_POINT}:
        raise DistributionVerificationError("wheel console entry point is invalid")
    _verify_wheel_record(payloads, dist_info_prefix=dist_info_prefix)
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "member_count": str(len(payloads)),
    }


def verify_sdist(path: Path) -> dict[str, str]:
    """Verify the source distribution is self-contained and safely bounded."""

    try:
        with tarfile.open(path, mode="r:gz") as archive:
            validated_members = _validate_sdist_members(archive.getmembers())
            expected_root = f"{EXPECTED_IMPORT_PACKAGE}-{EXPECTED_VERSION}"
            for member, name in validated_members:
                if name == expected_root:
                    if not member.isdir():
                        raise DistributionVerificationError("sdist root member is not a directory")
                    continue
                if not name.startswith(expected_root + "/"):
                    raise DistributionVerificationError("sdist member is outside its root")
                relative_name = name.removeprefix(expected_root + "/")
                top_level = relative_name.split("/", 1)[0]
                if member.isdir():
                    if top_level not in _SDIST_ALLOWED_ROOT_DIRECTORIES:
                        raise DistributionVerificationError(
                            "sdist directory is outside the explicit source allowlist"
                        )
                elif (
                    relative_name not in _SDIST_ALLOWED_ROOT_FILES
                    and top_level not in _SDIST_ALLOWED_ROOT_DIRECTORIES
                ):
                    raise DistributionVerificationError(
                        "sdist contains a file outside the explicit source allowlist"
                    )

            payloads: dict[str, bytes] = {}
            for member, name in validated_members:
                if not member.isfile():
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise DistributionVerificationError("sdist regular member could not be read")
                payloads[name] = extracted.read(_MAX_MEMBER_BYTES + 1)
    except (
        OSError,
        tarfile.TarError,
        EOFError,
        DistributionVerificationError,
    ):
        raise DistributionVerificationError("sdist archive is unreadable or unsafe") from None
    _assert_archive_size_limit(payloads)

    roots = {name.split("/", 1)[0] for name in payloads}
    if roots != {expected_root}:
        raise DistributionVerificationError("sdist root directory is invalid")
    relative_payloads: dict[str, bytes] = {}
    for name, payload in payloads.items():
        relative_name = name.removeprefix(expected_root + "/")
        if not relative_name or relative_name == name:
            raise DistributionVerificationError("sdist member is outside its root")
        _assert_safe_member_name(relative_name)
        _assert_safe_member_content(relative_name, payload)
        top_level = relative_name.split("/", 1)[0]
        if (
            relative_name not in _SDIST_ALLOWED_ROOT_FILES
            and top_level not in _SDIST_ALLOWED_ROOT_DIRECTORIES
        ):
            raise DistributionVerificationError(
                "sdist contains a file outside the explicit source allowlist"
            )
        if relative_name in _SDIST_FORBIDDEN_EXACT_FILES:
            raise DistributionVerificationError(
                "sdist contains an invalidated historical release input"
            )
        relative_payloads[relative_name] = payload

    missing = _SDIST_REQUIRED_FILES.difference(relative_payloads)
    if missing:
        raise DistributionVerificationError(
            "sdist is missing required source, contract, evaluation, fixture, tool, or test files"
        )
    _metadata_from_bytes(relative_payloads["PKG-INFO"])
    return {
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "member_count": str(len(relative_payloads)),
    }


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _direct_url_wheel_path(
    direct_url: object,
    *,
    expected_wheel_sha256: str,
) -> Path:
    if not isinstance(direct_url, dict):
        raise DistributionVerificationError("installed Agent wheel-origin attestation is invalid")
    if direct_url.get("dir_info") is not None or direct_url.get("vcs_info") is not None:
        raise DistributionVerificationError(
            "installed Agent did not originate from a non-editable wheel"
        )
    archive_info = direct_url.get("archive_info")
    if not isinstance(archive_info, dict):
        raise DistributionVerificationError("installed Agent wheel-origin attestation is invalid")

    observed_hashes: set[str] = set()
    hashes = archive_info.get("hashes")
    if hashes is not None:
        if not isinstance(hashes, dict):
            raise DistributionVerificationError(
                "installed Agent wheel-origin attestation is invalid"
            )
        sha256_value = hashes.get("sha256")
        if sha256_value is not None:
            observed_hashes.add(str(sha256_value).casefold())
    raw_hash = archive_info.get("hash")
    if raw_hash is not None:
        if not isinstance(raw_hash, str) or not raw_hash.casefold().startswith("sha256="):
            raise DistributionVerificationError(
                "installed Agent wheel-origin attestation is invalid"
            )
        observed_hashes.add(raw_hash.split("=", 1)[1].casefold())
    if observed_hashes and observed_hashes != {expected_wheel_sha256}:
        raise DistributionVerificationError(
            "installed Agent wheel-origin hash does not match the attested wheel"
        )

    url = direct_url.get("url")
    if not isinstance(url, str):
        raise DistributionVerificationError("installed Agent wheel-origin attestation is invalid")
    parsed = urlsplit(url)
    if parsed.scheme.casefold() != "file" or parsed.netloc or parsed.query or parsed.fragment:
        raise DistributionVerificationError(
            "installed Agent wheel origin must be an exact local file URL"
        )
    try:
        wheel_path = Path(url2pathname(unquote(parsed.path))).resolve(strict=True)
    except (OSError, ValueError):
        raise DistributionVerificationError(
            "installed Agent wheel-origin file is unavailable"
        ) from None
    if wheel_path.suffix.casefold() != ".whl":
        raise DistributionVerificationError(
            "installed Agent did not originate from a wheel archive"
        )
    return wheel_path


def _verify_wheel_origin(
    direct_url: object,
    *,
    wheel_path: Path,
    expected_wheel_sha256: str,
) -> Path:
    try:
        resolved_wheel_path = wheel_path.resolve(strict=True)
    except OSError:
        raise DistributionVerificationError("attested wheel archive is unavailable") from None
    wheel_result = verify_wheel(resolved_wheel_path)
    if wheel_result["sha256"] != expected_wheel_sha256:
        raise DistributionVerificationError(
            "attested wheel archive hash does not match caller expectation"
        )
    direct_url_wheel_path = _direct_url_wheel_path(
        direct_url,
        expected_wheel_sha256=expected_wheel_sha256,
    )
    if direct_url_wheel_path != resolved_wheel_path:
        raise DistributionVerificationError(
            "installed Agent wheel origin does not reference the attested wheel"
        )
    return resolved_wheel_path


def _read_recorded_file(
    distribution: importlib.metadata.Distribution,
    *,
    name: str,
    record_entry: tuple[str, str],
) -> bytes:
    digest, size = record_entry
    if not re.fullmatch(r"sha256=[A-Za-z0-9_-]{43}", digest) or not re.fullmatch(
        r"(?:0|[1-9][0-9]*)",
        size,
    ):
        raise DistributionVerificationError("installed Agent RECORD hash or size is invalid")
    try:
        installed_path = Path(str(distribution.locate_file(name)))
        if installed_path.is_symlink() or not installed_path.is_file():
            raise OSError
        payload = installed_path.read_bytes()
    except (OSError, TypeError, ValueError):
        raise DistributionVerificationError(
            "installed Agent RECORD references a missing file"
        ) from None
    if digest != f"sha256={_record_digest(payload)}" or size != str(len(payload)):
        raise DistributionVerificationError("installed Agent file does not match its RECORD")
    return payload


def _verify_installed_record_against_wheel(
    distribution: importlib.metadata.Distribution,
    *,
    wheel_path: Path,
) -> dict[str, str]:
    try:
        with zipfile.ZipFile(wheel_path) as archive:
            wheel_names = _normalize_member_names(archive.namelist())
            wheel_payloads = {name: archive.read(name) for name in wheel_names}
    except (OSError, zipfile.BadZipFile, KeyError):
        raise DistributionVerificationError("attested wheel archive is unreadable") from None

    dist_info_prefix = f"{EXPECTED_IMPORT_PACKAGE}-{EXPECTED_VERSION}.dist-info/"
    wheel_record_name = f"{dist_info_prefix}RECORD"
    try:
        wheel_record_text = wheel_payloads[wheel_record_name].decode("utf-8")
    except (KeyError, UnicodeDecodeError):
        raise DistributionVerificationError("attested wheel RECORD is missing or invalid") from None
    wheel_record = _parse_record(
        wheel_record_text,
        distribution_kind="attested wheel",
    )
    installer_owned_names = {
        f"{dist_info_prefix}{name}" for name in _INSTALLER_OWNED_DIST_INFO_FILES
    }
    if installer_owned_names.intersection(wheel_record):
        raise DistributionVerificationError("wheel contains installer-owned origin metadata")

    installed_record_text = distribution.read_text("RECORD")
    if installed_record_text is None:
        raise DistributionVerificationError("installed Agent RECORD is missing")
    installed_record = _parse_record(
        installed_record_text,
        distribution_kind="installed Agent",
    )

    missing = set(wheel_record).difference(installed_record)
    if missing:
        raise DistributionVerificationError(
            "installed Agent RECORD is missing attested wheel files"
        )
    for name, expected_entry in wheel_record.items():
        installed_entry = installed_record[name]
        if name == wheel_record_name:
            if expected_entry != ("", "") or installed_entry != ("", ""):
                raise DistributionVerificationError("installed Agent RECORD self-entry is invalid")
            continue
        if installed_entry != expected_entry:
            raise DistributionVerificationError(
                "installed Agent RECORD differs from the attested wheel"
            )
        installed_payload = _read_recorded_file(
            distribution,
            name=name,
            record_entry=installed_entry,
        )
        if installed_payload != wheel_payloads[name]:
            raise DistributionVerificationError(
                "installed Agent payload differs from the attested wheel"
            )

    extra_names = set(installed_record).difference(wheel_record)
    console_names = extra_names.intersection(_ALLOWED_CONSOLE_RECORD_NAMES)
    unknown_names = extra_names.difference(installer_owned_names, console_names)
    if unknown_names:
        raise DistributionVerificationError("installed Agent RECORD contains an unapproved file")
    required_installer_names = {
        f"{dist_info_prefix}INSTALLER",
        f"{dist_info_prefix}direct_url.json",
    }
    if not required_installer_names.issubset(extra_names):
        raise DistributionVerificationError(
            "installed Agent lacks required installer provenance files"
        )
    if len(console_names) != 1:
        raise DistributionVerificationError("installed Agent console launcher inventory is invalid")

    installer_payloads: dict[str, bytes] = {}
    for name in sorted(extra_names):
        installer_payloads[name] = _read_recorded_file(
            distribution,
            name=name,
            record_entry=installed_record[name],
        )
    installer_name = f"{dist_info_prefix}INSTALLER"
    if installer_payloads[installer_name].strip().lower() != b"uv":
        raise DistributionVerificationError(
            "installed Agent was not installed by the pinned uv workflow"
        )
    return {
        "installed_record_count": str(len(installed_record)),
        "console_record": next(iter(console_names)),
    }


def verify_installed_distribution(
    *,
    expected_wheel_sha256: str,
    source_root: Path,
    wheel_path: Path,
) -> dict[str, str]:
    """Verify the installed package resolves only from the attested wheel."""

    if not sys.flags.isolated:
        raise DistributionVerificationError(
            "installed distribution verification requires Python isolated mode"
        )
    if not re.fullmatch(r"[a-f0-9]{64}", expected_wheel_sha256):
        raise DistributionVerificationError("expected wheel SHA-256 is invalid")
    try:
        distribution = importlib.metadata.distribution(EXPECTED_DISTRIBUTION)
        package = __import__(EXPECTED_IMPORT_PACKAGE)
        from customs_ai_agent import AgentRuntime, RuntimeConfig
        from customs_ai_agent.contracts import (
            available_schemas,
            verify_packaged_schemas,
        )
        from customs_ai_agent.prompts import (
            load_document_analysis_system_prompt,
            load_document_vision_analysis_system_prompt,
        )
    except (ImportError, importlib.metadata.PackageNotFoundError):
        raise DistributionVerificationError(
            "installed Agent distribution cannot be imported"
        ) from None

    if distribution.version != EXPECTED_VERSION:
        raise DistributionVerificationError("installed Agent version is invalid")
    if not isinstance(AgentRuntime, type) or not isinstance(RuntimeConfig, type):
        raise DistributionVerificationError("installed Agent public runtime facade is unavailable")
    module_path = Path(str(package.__file__)).resolve(strict=True)
    resolved_source_root = source_root.resolve(strict=True)
    if _is_relative_to(module_path, resolved_source_root):
        raise DistributionVerificationError(
            "installed Agent import resolves to the source checkout"
        )
    for search_path in sys.path:
        if not search_path:
            continue
        try:
            candidate = Path(search_path).resolve(strict=False)
        except OSError:
            continue
        if _is_relative_to(candidate, resolved_source_root):
            raise DistributionVerificationError(
                "installed Agent sys.path contains the source checkout"
            )

    direct_url_text = distribution.read_text("direct_url.json")
    if direct_url_text is None:
        raise DistributionVerificationError("installed Agent lacks wheel-origin attestation")
    try:
        direct_url = json.loads(direct_url_text)
    except json.JSONDecodeError:
        raise DistributionVerificationError(
            "installed Agent wheel-origin attestation is invalid"
        ) from None
    resolved_wheel_path = _verify_wheel_origin(
        direct_url,
        wheel_path=wheel_path,
        expected_wheel_sha256=expected_wheel_sha256,
    )
    installed_record_result = _verify_installed_record_against_wheel(
        distribution,
        wheel_path=resolved_wheel_path,
    )

    entry_points = {
        f"{entry.name} = {entry.value}"
        for entry in distribution.entry_points
        if entry.group == "console_scripts"
    }
    if entry_points != {EXPECTED_CONSOLE_ENTRY_POINT}:
        raise DistributionVerificationError("installed Agent console entry point is invalid")
    schemas = available_schemas()
    if len(schemas) != len(EXPECTED_SCHEMA_RESOURCES):
        raise DistributionVerificationError("installed Agent schema inventory is invalid")
    schema_hashes = verify_packaged_schemas()
    if len(schema_hashes) != len(EXPECTED_SCHEMA_RESOURCES):
        raise DistributionVerificationError(
            "installed Agent schema resources do not match their models"
        )
    prompts = {
        "document_analysis_v1.txt": load_document_analysis_system_prompt(),
        "document_vision_analysis_v1.txt": load_document_vision_analysis_system_prompt(),
    }
    if set(prompts) != EXPECTED_PROMPT_RESOURCES or any(
        not value.strip() for value in prompts.values()
    ):
        raise DistributionVerificationError("installed Agent prompt resources are incomplete")
    return {
        "version": distribution.version,
        "wheel_sha256": expected_wheel_sha256,
        "schema_count": str(len(schema_hashes)),
        "prompt_count": str(len(prompts)),
        **installed_record_result,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--sdist", type=Path)
    parser.add_argument("--installed", action="store_true")
    parser.add_argument("--expected-wheel-sha256")
    parser.add_argument("--source-root", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.wheel and not args.sdist and not args.installed:
        raise DistributionVerificationError(
            "at least one distribution verification target is required"
        )
    results: dict[str, Mapping[str, str]] = {}
    if args.wheel:
        results["wheel"] = verify_wheel(args.wheel.resolve(strict=True))
    if args.sdist:
        results["sdist"] = verify_sdist(args.sdist.resolve(strict=True))
    if args.installed:
        if args.expected_wheel_sha256 is None or args.source_root is None or args.wheel is None:
            raise DistributionVerificationError(
                "installed verification requires wheel archive, hash, and source root"
            )
        results["installed"] = verify_installed_distribution(
            expected_wheel_sha256=args.expected_wheel_sha256.casefold(),
            source_root=args.source_root,
            wheel_path=args.wheel,
        )
    print(json.dumps(results, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DistributionVerificationError as exc:
        print(f"distribution verification failed: {exc}", file=sys.stderr)
        raise SystemExit(1) from None
