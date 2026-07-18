"""Fail-closed runtime attestation for the canonical OCR release Gate."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal, cast

from customs_ai_agent.adapters.ocr import OcrLayoutProvider, OcrLayoutProviderError
from customs_ai_agent.ocr_identity import (
    OCR_MODEL_ASSETS_LOCK_SHA256,
    OCR_MODEL_SET_IDS,
)
from customs_ai_agent.ocr_release_pin import (
    CANONICAL_OCR_AUTHORIZATION_COMMIT_SHA,
    CANONICAL_OCR_DESCRIPTOR_SHA256,
)
from customs_ai_agent.schemas.models import ImagePageInput, OcrLayoutLedger

ProviderName = Literal["paddle", "docling"]
WheelAttestor = Callable[[Path], str]
EnvironmentAttestor = Callable[[Path, Path], str]
ModelAssetsAttestor = Callable[[ProviderName, Path, str], str]
_CANONICAL_RELEASE_PROOF = object()

OCR_BENCHMARK_ID = "rendered-trade-documents-ocr-v2"
OCR_RELEASE_REPORT_SCHEMA = "ocr-release-gate-report/v2"
OCR_RELEASE_PROVIDERS: tuple[ProviderName, ...] = ("paddle", "docling")
OCR_RELEASE_DOCUMENT_COUNT = 80
OCR_RELEASE_SAMPLE_DOCUMENT_COUNTS = {
    "locked": 60,
    "co_regression": 20,
}
OCR_RELEASE_REDACTION = {
    "source_text": "OMITTED",
    "image_payloads": "OMITTED",
    "filesystem_paths": "OMITTED",
    "secrets": "OMITTED",
    "backend_exception_messages": "OMITTED",
}
CANONICAL_OCR_MANIFEST_SHA256 = "e95a3a2f3d2539c8fb92c3988010c8aeb33736269f1bbfc1c5e637962ca25196"
CANONICAL_OCR_ENVIRONMENT_LOCK_SHA256: dict[ProviderName, str] = {
    "paddle": "607459fa093bd01ca1706dfa07222038f05fa190deccc58565ad1d846fc3bdf4",
    "docling": "1d9aa9b9f2ac28dc2d2bff3e61b33ebd37bbf4c052910f3412f130eb79af9e05",
}
CANONICAL_OCR_MODEL_ASSETS_LOCK_SHA256 = OCR_MODEL_ASSETS_LOCK_SHA256
CANONICAL_OCR_PROVIDER_ENGINE_IDS: dict[ProviderName, str] = {
    "paddle": "PP-StructureV3:customs-paddle-layout-table-v4",
    "docling": "DocumentConverter:default-image-pipeline-offline-v2",
}

SHA256 = re.compile(r"^[a-f0-9]{64}$")
SAFE_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
MAX_REPORT_BYTES = 4 * 1024 * 1024
MAX_DESCRIPTOR_BYTES = 512 * 1024
MAX_MANIFEST_BYTES = 4 * 1024 * 1024
MAX_ENVIRONMENT_LOCK_BYTES = 4 * 1024 * 1024
MAX_MODEL_ASSETS_LOCK_BYTES = 512 * 1024
MODEL_ASSETS_LOCK_SCHEMA = "ocr-model-assets-lock/v1"
MODEL_ASSETS_TREE_ALGORITHM = (
    "sha256(relative_posix_path + NUL + decimal_size + NUL + file_sha256 + LF), "
    "sorted by relative_posix_path"
)
MODEL_ASSETS_LOCK_KEYS = frozenset(
    {
        "schema_version",
        "tree_hash_algorithm",
        "providers",
    }
)
MODEL_ASSETS_PROVIDER_KEYS = frozenset(
    {
        "assets",
        "cache_layout",
        "engine_id",
    }
)
MODEL_ASSET_KEYS = frozenset(
    {
        "asset_id",
        "revision",
        "file_count",
        "total_bytes",
        "tree_sha256",
    }
)
MODEL_CACHE_LAYOUTS = {
    "paddle": "paddlex_official_models",
    "docling": "huggingface_hub_snapshot",
}
GIT_REVISION = re.compile(r"^[a-f0-9]{40}$")
MODEL_ASSET_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*$")
REPORT_KEYS = frozenset(
    {
        "schema_version",
        "benchmark_id",
        "status",
        "selection_status",
        "selected_provider",
        "smoke_included",
        "release_document_count",
        "descriptor_sha256",
        "dataset_manifest_sha256",
        "agent_wheel_sha256",
        "environment_lock_sha256",
        "model_assets_lock_sha256",
        "selected_engine",
        "input_reports",
        "provider_eligibility",
        "runtime_tiebreak",
        "failure_codes",
        "redaction",
    }
)
INPUT_REPORT_KEYS = frozenset({"locked", "co_regression"})
INPUT_REPORT_VALUE_KEYS = frozenset({"sha256", "document_count"})
ELIGIBILITY_KEYS = frozenset(
    {
        "provider",
        "locked_status",
        "co_regression_status",
        "eligible",
    }
)
ENGINE_KEYS = frozenset({"provider_id", "engine_id", "engine_version"})
RUNTIME_TIEBREAK_KEYS = frozenset(
    {
        "source_sample",
        "selection_order",
        "providers",
    }
)
RUNTIME_TIEBREAK_PROVIDER_KEYS = frozenset(
    {
        "provider",
        "latency_p95_ms",
        "peak_memory_bytes",
        "disk_footprint_bytes",
    }
)
PASSING_SELECTION_STATUSES = frozenset(
    {
        "SELECTED_SINGLE_COMMON_PROVIDER",
        "SELECTED_BY_LOCKED_QUALITY_METRICS",
        "SELECTED_BY_ATTESTED_RUNTIME_METRICS",
    }
)

_WHEEL_ATTESTATION_SCHEMA = "customs-ai-agent-wheel-attestation/v1"
_WHEEL_ATTESTATION_TIMEOUT_SECONDS = 15.0
_WHEEL_ATTESTATION_MAX_OUTPUT_BYTES = 2048
_WHEEL_ATTESTATION_ERROR_CODES = frozenset(
    {
        "agent_distribution_unavailable",
        "agent_direct_url_invalid",
        "agent_record_invalid",
        "agent_record_mismatch",
        "agent_source_install_rejected",
        "agent_untracked_executable",
        "agent_wheel_hash_unavailable",
    }
)
_ENVIRONMENT_ATTESTATION_SCHEMA = "customs-ocr-environment-attestation/v1"
_ENVIRONMENT_ATTESTATION_TIMEOUT_SECONDS = 30.0
_ENVIRONMENT_ATTESTATION_MAX_OUTPUT_BYTES = 2048
_ENVIRONMENT_ATTESTATION_ERROR_CODES = frozenset(
    {
        "environment_lock_invalid",
        "environment_lock_marker_unsupported",
        "environment_platform_mismatch",
        "environment_inventory_invalid",
        "environment_inventory_mismatch",
    }
)
_WHEEL_ATTESTATION_SCRIPT = r"""
import base64
import csv
import hashlib
import importlib.machinery
import importlib.metadata
import importlib.util
import io
import json
import marshal
import os
import re
import sys
import types
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit

SCHEMA = "customs-ai-agent-wheel-attestation/v1"
SHA256 = re.compile(r"^[a-fA-F0-9]{64}$")
MAX_RECORD_BYTES = 2 * 1024 * 1024
MAX_MEMBER_BYTES = 8 * 1024 * 1024
MAX_TOTAL_BYTES = 32 * 1024 * 1024
MAX_WHEEL_BYTES = 64 * 1024 * 1024


def emit_error(code):
    sys.stdout.write(
        json.dumps(
            {"schema_version": SCHEMA, "status": "error", "error_code": code},
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    raise SystemExit(0)


def record_digest(payload):
    return base64.urlsafe_b64encode(hashlib.sha256(payload).digest()).rstrip(b"=").decode()


def resolve_record_target(base, executable_dir, raw_name):
    if (
        not isinstance(raw_name, str)
        or not raw_name
        or "\\" in raw_name
        or any(ord(character) < 32 or ord(character) == 127 for character in raw_name)
        or re.match(r"^[A-Za-z]:", raw_name)
    ):
        emit_error("agent_record_invalid")
    path = PurePosixPath(raw_name)
    raw_parts = raw_name.split("/")
    if path.is_absolute() or any(part in {"", "."} for part in raw_parts):
        emit_error("agent_record_invalid")
    try:
        target = base.joinpath(*path.parts)
        if target.is_symlink():
            emit_error("agent_record_invalid")
        resolved = target.resolve(strict=True)
    except OSError:
        emit_error("agent_record_mismatch")
    try:
        resolved.relative_to(base)
    except ValueError:
        if (
            resolved.parent != executable_dir
            or not resolved.name.lower().startswith("customs-ai-agent")
        ):
            emit_error("agent_record_invalid")
    if not resolved.is_file() or resolved.is_symlink():
        emit_error("agent_record_mismatch")
    return path.as_posix(), resolved


def local_wheel_path(url):
    parsed = urlsplit(url)
    if parsed.scheme.lower() != "file":
        return None
    if (
        parsed.netloc.lower() not in {"", "localhost"}
        or parsed.query
        or parsed.fragment
    ):
        emit_error("agent_direct_url_invalid")
    raw_path = unquote(parsed.path)
    if (
        not raw_path
        or "\\" in raw_path
        or any(ord(character) < 32 or ord(character) == 127 for character in raw_path)
    ):
        emit_error("agent_direct_url_invalid")
    if os.name == "nt":
        if re.match(r"^/[A-Za-z]:/", raw_path) is None:
            emit_error("agent_direct_url_invalid")
        raw_path = raw_path[1:]
    elif not raw_path.startswith("/") or raw_path.startswith("//"):
        emit_error("agent_direct_url_invalid")
    candidate = Path(raw_path)
    if not candidate.is_absolute():
        emit_error("agent_direct_url_invalid")
    return candidate


def local_wheel_sha256(url):
    candidate = local_wheel_path(url)
    if candidate is None:
        return None
    try:
        if candidate.is_symlink():
            emit_error("agent_direct_url_invalid")
        resolved = candidate.resolve(strict=True)
        size = resolved.stat().st_size
    except OSError:
        emit_error("agent_wheel_hash_unavailable")
    if (
        not resolved.is_file()
        or resolved.is_symlink()
        or size < 1
        or size > MAX_WHEEL_BYTES
        or not resolved.name.lower().endswith(".whl")
    ):
        emit_error("agent_direct_url_invalid")
    digest = hashlib.sha256()
    try:
        with resolved.open("rb") as wheel:
            while chunk := wheel.read(1024 * 1024):
                digest.update(chunk)
    except OSError:
        emit_error("agent_wheel_hash_unavailable")
    return digest.hexdigest()


def verify_cached_bytecode(candidate, package_root, base, recorded):
    try:
        resolved = candidate.resolve(strict=True)
        relative = resolved.relative_to(package_root)
    except (OSError, ValueError):
        emit_error("agent_untracked_executable")
    if (
        len(relative.parts) < 2
        or relative.parts[-2] != "__pycache__"
        or not resolved.is_file()
        or resolved.is_symlink()
    ):
        emit_error("agent_untracked_executable")

    cache_tag = sys.implementation.cache_tag
    if not isinstance(cache_tag, str) or not cache_tag:
        emit_error("agent_untracked_executable")
    variants = (
        (f".{cache_tag}.pyc", 0),
        (f".{cache_tag}.opt-1.pyc", 1),
        (f".{cache_tag}.opt-2.pyc", 2),
    )
    matched = next(
        (
            (suffix, optimization)
            for suffix, optimization in variants
            if resolved.name.endswith(suffix)
        ),
        None,
    )
    if matched is None:
        emit_error("agent_untracked_executable")
    suffix, optimization = matched
    module_name = resolved.name[: -len(suffix)]
    if not module_name or not module_name.isidentifier():
        emit_error("agent_untracked_executable")

    source = resolved.parent.parent / f"{module_name}.py"
    try:
        if source.is_symlink():
            emit_error("agent_untracked_executable")
        source = source.resolve(strict=True)
        source.relative_to(package_root)
        canonical_source = source.relative_to(base).as_posix()
    except (OSError, ValueError):
        emit_error("agent_untracked_executable")
    if (
        not source.is_file()
        or canonical_source.casefold() not in recorded
    ):
        emit_error("agent_untracked_executable")

    try:
        payload = resolved.read_bytes()
        source_payload = source.read_bytes()
        source_stat = source.stat()
    except OSError:
        emit_error("agent_untracked_executable")
    if (
        len(payload) <= 16
        or len(payload) > MAX_MEMBER_BYTES
        or len(source_payload) > MAX_MEMBER_BYTES
        or payload[:4] != importlib.util.MAGIC_NUMBER
    ):
        emit_error("agent_untracked_executable")

    flags = int.from_bytes(payload[4:8], "little")
    if flags == 0:
        expected_header = (
            (int(source_stat.st_mtime) & 0xFFFFFFFF).to_bytes(4, "little")
            + (len(source_payload) & 0xFFFFFFFF).to_bytes(4, "little")
        )
    elif flags == 3:
        expected_header = importlib.util.source_hash(source_payload)
    else:
        emit_error("agent_untracked_executable")
    if payload[8:16] != expected_header:
        emit_error("agent_untracked_executable")

    try:
        marshalled = io.BytesIO(payload[16:])
        observed_code = marshal.load(marshalled)
        if marshalled.read(1):
            emit_error("agent_untracked_executable")
        if (
            not isinstance(observed_code, types.CodeType)
            or Path(observed_code.co_filename).resolve(strict=True) != source
        ):
            emit_error("agent_untracked_executable")
        expected_code = compile(
            source_payload,
            observed_code.co_filename,
            "exec",
            dont_inherit=True,
            optimize=optimization,
        )
    except (EOFError, OSError, TypeError, ValueError):
        emit_error("agent_untracked_executable")
    if observed_code != expected_code:
        emit_error("agent_untracked_executable")


try:
    distribution = importlib.metadata.distribution("customs-ai-agent")
    metadata_name = str(distribution.metadata.get("Name") or "")
    normalized_name = re.sub(r"[-_.]+", "-", metadata_name).lower()
    if normalized_name != "customs-ai-agent":
        emit_error("agent_distribution_unavailable")

    version = distribution.version
    if not isinstance(version, str) or not version:
        emit_error("agent_distribution_unavailable")
    base = Path(distribution.locate_file("")).resolve(strict=True)
    executable_dir = Path(sys.executable).resolve(strict=True).parent
    record_relative = f"customs_ai_agent-{version}.dist-info/RECORD"
    record_path = Path(distribution.locate_file(record_relative))
    if record_path.is_symlink():
        emit_error("agent_record_invalid")
    record_path = record_path.resolve(strict=True)
    record_path.relative_to(base)
    record_payload = record_path.read_bytes()
    if not record_payload or len(record_payload) > MAX_RECORD_BYTES:
        emit_error("agent_record_invalid")
    try:
        record_text = record_payload.decode("utf-8")
        rows = list(csv.reader(io.StringIO(record_text, newline="")))
    except (UnicodeDecodeError, csv.Error):
        emit_error("agent_record_invalid")
    if not rows:
        emit_error("agent_record_invalid")

    recorded = {}
    total_bytes = 0
    record_row_count = 0
    for row in rows:
        if len(row) != 3:
            emit_error("agent_record_invalid")
        raw_name, digest, size_text = row
        canonical, target = resolve_record_target(base, executable_dir, raw_name)
        folded = canonical.casefold()
        if folded in recorded:
            emit_error("agent_record_invalid")
        recorded[folded] = canonical
        if canonical == record_relative:
            record_row_count += 1
            if digest or size_text or target != record_path:
                emit_error("agent_record_invalid")
            continue
        if (
            not digest.startswith("sha256=")
            or not size_text.isascii()
            or not size_text.isdecimal()
        ):
            emit_error("agent_record_invalid")
        expected_size = int(size_text)
        try:
            actual_size = target.stat().st_size
        except OSError:
            emit_error("agent_record_mismatch")
        if (
            expected_size < 0
            or expected_size > MAX_MEMBER_BYTES
            or actual_size != expected_size
        ):
            emit_error("agent_record_mismatch")
        payload = target.read_bytes()
        total_bytes += len(payload)
        if (
            total_bytes > MAX_TOTAL_BYTES
            or len(payload) != expected_size
            or digest != f"sha256={record_digest(payload)}"
        ):
            emit_error("agent_record_mismatch")
    if record_row_count != 1:
        emit_error("agent_record_invalid")

    package_root = base / "customs_ai_agent"
    if package_root.is_symlink():
        emit_error("agent_untracked_executable")
    package_root = package_root.resolve(strict=True)
    package_root.relative_to(base)
    if not package_root.is_dir():
        emit_error("agent_distribution_unavailable")
    recorded_executable_suffixes = tuple(
        sorted(
            {
                *importlib.machinery.SOURCE_SUFFIXES,
                *importlib.machinery.EXTENSION_SUFFIXES,
            },
            key=len,
            reverse=True,
        )
    )
    bytecode_suffixes = tuple(importlib.machinery.BYTECODE_SUFFIXES)
    executable_count = 0
    for current_root, directory_names, file_names in os.walk(
        package_root,
        topdown=True,
        followlinks=False,
    ):
        current = Path(current_root)
        for directory_name in directory_names:
            if (current / directory_name).is_symlink():
                emit_error("agent_untracked_executable")
        for file_name in file_names:
            candidate = current / file_name
            if candidate.is_symlink():
                emit_error("agent_untracked_executable")
            if file_name.endswith(bytecode_suffixes):
                verify_cached_bytecode(candidate, package_root, base, recorded)
                continue
            if not file_name.endswith(recorded_executable_suffixes):
                continue
            executable_count += 1
            canonical = candidate.resolve(strict=True).relative_to(base).as_posix()
            if canonical.casefold() not in recorded:
                emit_error("agent_untracked_executable")
    for suffix in (*recorded_executable_suffixes, *bytecode_suffixes):
        candidate = base / f"customs_ai_agent{suffix}"
        if candidate.exists() and candidate.resolve(strict=True).relative_to(base).as_posix().casefold() not in recorded:
            emit_error("agent_untracked_executable")
    if executable_count < 1:
        emit_error("agent_distribution_unavailable")

    direct_url_relative = f"customs_ai_agent-{version}.dist-info/direct_url.json"
    if direct_url_relative.casefold() not in recorded:
        emit_error("agent_record_invalid")
    raw_direct_url = Path(distribution.locate_file(direct_url_relative)).read_text(
        encoding="utf-8"
    )
    if not raw_direct_url:
        emit_error("agent_direct_url_invalid")
    direct_url = json.loads(raw_direct_url)
    if not isinstance(direct_url, dict):
        emit_error("agent_direct_url_invalid")

    if direct_url.get("dir_info") is not None or direct_url.get("vcs_info") is not None:
        emit_error("agent_source_install_rejected")
    url = direct_url.get("url")
    if not isinstance(url, str):
        emit_error("agent_direct_url_invalid")
    parsed_url = urlsplit(url)
    if not unquote(parsed_url.path).lower().endswith(".whl"):
        emit_error("agent_source_install_rejected")
    local_wheel_path(url)

    archive_info = direct_url.get("archive_info")
    if not isinstance(archive_info, dict):
        emit_error("agent_wheel_hash_unavailable")
    candidates = []
    hashes = archive_info.get("hashes")
    if hashes is not None:
        if not isinstance(hashes, dict):
            emit_error("agent_direct_url_invalid")
        sha256_value = hashes.get("sha256")
        if sha256_value is not None:
            candidates.append(sha256_value)
    legacy_hash = archive_info.get("hash")
    if legacy_hash is not None:
        if not isinstance(legacy_hash, str) or not legacy_hash.lower().startswith("sha256="):
            emit_error("agent_direct_url_invalid")
        candidates.append(legacy_hash.split("=", 1)[1])
    if not candidates:
        local_digest = local_wheel_sha256(url)
        if local_digest is not None:
            candidates.append(local_digest)
    if not candidates:
        emit_error("agent_wheel_hash_unavailable")
    if not all(isinstance(value, str) and SHA256.fullmatch(value) for value in candidates):
        emit_error("agent_direct_url_invalid")
    normalized_hashes = {value.lower() for value in candidates}
    if len(normalized_hashes) != 1:
        emit_error("agent_direct_url_invalid")
    installed_sha256 = normalized_hashes.pop()
except SystemExit:
    raise
except Exception:
    emit_error("agent_direct_url_invalid")

sys.stdout.write(
    json.dumps(
        {
            "schema_version": SCHEMA,
            "status": "ok",
            "installed_agent_wheel_sha256": installed_sha256,
            "installed_agent_record_sha256": hashlib.sha256(record_payload).hexdigest(),
            "verified_executable_file_count": executable_count,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
)
"""

_ENVIRONMENT_ATTESTATION_SCRIPT = r"""
import hashlib
import importlib.metadata
import json
import platform
import re
import sys
from pathlib import Path

SCHEMA = "customs-ocr-environment-attestation/v1"
NAME = re.compile(r"[-_.]+")
REQUIREMENT = re.compile(
    r"^([A-Za-z0-9][A-Za-z0-9_.-]*)==([^ ;\\]+)(?:\s*;\s*(.*?))?"
    r"\s+--hash=sha256:([a-f0-9]{64})(?:\s+--hash=sha256:[a-f0-9]{64})*$"
)
MARKER_TERM = re.compile(
    r"^(sys_platform|platform_machine|python_version)\s*(==|!=)\s*'([^']+)'$"
)
MAX_LOCK_BYTES = 4 * 1024 * 1024


def emit_error(code):
    sys.stdout.write(
        json.dumps(
            {"schema_version": SCHEMA, "status": "error", "error_code": code},
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    raise SystemExit(0)


def normalize_name(value):
    return NAME.sub("-", value).lower()


def marker_applies(marker):
    if marker is None:
        return True
    values = {
        "sys_platform": sys.platform,
        "platform_machine": platform.machine(),
        "python_version": f"{sys.version_info.major}.{sys.version_info.minor}",
    }
    outcomes = []
    for term in marker.split(" or "):
        matched = MARKER_TERM.fullmatch(term.strip())
        if matched is None:
            emit_error("environment_lock_marker_unsupported")
        name, operator, expected = matched.groups()
        actual = values[name]
        outcomes.append(actual == expected if operator == "==" else actual != expected)
    return any(outcomes)


try:
    if len(sys.argv) != 2:
        emit_error("environment_lock_invalid")
    if sys.platform != "win32" or sys.version_info[:2] != (3, 11):
        emit_error("environment_platform_mismatch")

    lock_path = Path(sys.argv[1])
    payload = lock_path.read_bytes()
    if len(payload) < 2 or len(payload) > MAX_LOCK_BYTES:
        emit_error("environment_lock_invalid")
    lock_sha256 = hashlib.sha256(payload).hexdigest()
    text = payload.decode("utf-8")

    logical_lines = []
    pending = ""
    for raw_line in text.splitlines():
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            if pending:
                emit_error("environment_lock_invalid")
            continue
        pending += (" " if pending else "") + stripped.removesuffix("\\").strip()
        if stripped.endswith("\\"):
            continue
        logical_lines.append(pending)
        pending = ""
    if pending:
        emit_error("environment_lock_invalid")

    expected = {}
    for logical_line in logical_lines:
        matched = REQUIREMENT.fullmatch(logical_line)
        if matched is None:
            emit_error("environment_lock_invalid")
        name, version, marker, _first_hash = matched.groups()
        normalized = normalize_name(name)
        if normalized in expected:
            emit_error("environment_lock_invalid")
        if marker_applies(marker):
            expected[normalized] = version
    if not expected:
        emit_error("environment_lock_invalid")

    installed = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        if not isinstance(name, str) or not name or not isinstance(version, str) or not version:
            emit_error("environment_inventory_invalid")
        normalized = normalize_name(name)
        if normalized == "customs-ai-agent":
            continue
        if normalized in installed:
            emit_error("environment_inventory_invalid")
        installed[normalized] = version
    if installed != expected:
        emit_error("environment_inventory_mismatch")
except SystemExit:
    raise
except Exception:
    emit_error("environment_inventory_invalid")

sys.stdout.write(
    json.dumps(
        {
            "schema_version": SCHEMA,
            "status": "ok",
            "environment_lock_sha256": lock_sha256,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
)
"""


class OcrReleaseGateAttestationError(RuntimeError):
    """One content-free consumer preflight failure."""

    def __init__(self, code: str) -> None:
        if SAFE_CODE.fullmatch(code) is None:
            code = "ocr_release_gate_attestation_error"
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class OcrReleaseGateAttestation:
    """Identity transitively pinned by one combined release report hash."""

    report_sha256: str
    descriptor_sha256: str
    manifest_sha256: str
    agent_wheel_sha256: str
    environment_lock_sha256: str
    model_assets_lock_sha256: str
    provider: ProviderName
    provider_id: str
    engine_id: str
    engine_version: str
    _verification_proof: object | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )


class ReleaseLockedOcrProvider:
    """Reject runtime OCR identity drift before Claude receives the ledger."""

    def __init__(
        self,
        provider: OcrLayoutProvider,
        attestation: OcrReleaseGateAttestation,
    ) -> None:
        self._provider = provider
        self._attestation = attestation

    def assert_service_ready(self) -> None:
        """Fail closed unless this wrapper is bound to the canonical release."""

        if CANONICAL_OCR_DESCRIPTOR_SHA256 is None:
            raise OcrLayoutProviderError(
                "ocr_release_gate_not_frozen",
                "The canonical OCR release descriptor is not frozen.",
            )
        if CANONICAL_OCR_AUTHORIZATION_COMMIT_SHA is None:
            raise OcrLayoutProviderError(
                "ocr_release_gate_not_frozen",
                "The canonical OCR release authorization is not frozen.",
            )
        if (
            self._attestation._verification_proof is not _CANONICAL_RELEASE_PROOF
            or SHA256.fullmatch(CANONICAL_OCR_DESCRIPTOR_SHA256) is None
            or GIT_REVISION.fullmatch(CANONICAL_OCR_AUTHORIZATION_COMMIT_SHA) is None
            or self._attestation.descriptor_sha256 != CANONICAL_OCR_DESCRIPTOR_SHA256
            or self._attestation.manifest_sha256 != CANONICAL_OCR_MANIFEST_SHA256
            or self._attestation.model_assets_lock_sha256 != CANONICAL_OCR_MODEL_ASSETS_LOCK_SHA256
            or self._attestation.provider not in OCR_RELEASE_PROVIDERS
            or self._attestation.environment_lock_sha256
            != CANONICAL_OCR_ENVIRONMENT_LOCK_SHA256[self._attestation.provider]
            or self._attestation.provider_id != self._attestation.provider
            or self._attestation.engine_id
            != CANONICAL_OCR_PROVIDER_ENGINE_IDS[self._attestation.provider]
        ):
            raise OcrLayoutProviderError(
                "ocr_release_gate_identity_mismatch",
                "The OCR provider is not bound to the canonical release identity.",
            )

    def analyze(
        self,
        *,
        document_id: str,
        pages: Sequence[ImagePageInput],
    ) -> OcrLayoutLedger:
        ledger = self._provider.analyze(
            document_id=document_id,
            pages=pages,
        )
        if not isinstance(ledger, OcrLayoutLedger):
            raise OcrLayoutProviderError(
                "ocr_release_gate_runtime_ledger_invalid",
                "Runtime OCR output does not satisfy the locked ledger type.",
            )
        if (
            ledger.provider_id != self._attestation.provider_id
            or ledger.engine_id != self._attestation.engine_id
            or ledger.engine_version != self._attestation.engine_version
        ):
            raise OcrLayoutProviderError(
                "ocr_release_gate_runtime_engine_mismatch",
                "Runtime OCR identity differs from the combined release Gate.",
            )
        if (
            ledger.model_id != OCR_MODEL_SET_IDS[self._attestation.provider]
            or ledger.model_assets_lock_sha256 != OCR_MODEL_ASSETS_LOCK_SHA256
        ):
            raise OcrLayoutProviderError(
                "ocr_release_gate_runtime_model_mismatch",
                "Runtime OCR model identity differs from the canonical model-assets lock.",
            )
        return ledger

    def close(self) -> None:
        close_provider = getattr(self._provider, "close", None)
        if callable(close_provider):
            close_provider()


def _require_sha256(value: object, code: str) -> str:
    if not isinstance(value, str) or SHA256.fullmatch(value) is None:
        raise OcrReleaseGateAttestationError(code)
    return value


def _file_sha256(path: Path, *, unavailable_code: str) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise OcrReleaseGateAttestationError(unavailable_code) from exc
    return digest.hexdigest()


def _regular_file(
    path: Path,
    *,
    max_bytes: int,
    unavailable_code: str,
    invalid_code: str,
) -> Path:
    try:
        resolved = path.expanduser().resolve(strict=True)
        size = resolved.stat().st_size
    except OSError as exc:
        raise OcrReleaseGateAttestationError(unavailable_code) from exc
    if (
        not resolved.is_file()
        or resolved.is_symlink()
        or path.is_symlink()
        or size < 2
        or size > max_bytes
    ):
        raise OcrReleaseGateAttestationError(invalid_code)
    return resolved


def load_canonical_ocr_descriptor_identity(descriptor_path: Path) -> tuple[str, str]:
    """Validate the code-owned descriptor and its C0 authorization identity."""

    if CANONICAL_OCR_DESCRIPTOR_SHA256 is None:
        raise OcrReleaseGateAttestationError("ocr_release_gate_descriptor_not_frozen")
    if CANONICAL_OCR_AUTHORIZATION_COMMIT_SHA is None:
        raise OcrReleaseGateAttestationError("ocr_release_gate_authorization_commit_not_frozen")
    descriptor_pin = _require_sha256(
        CANONICAL_OCR_DESCRIPTOR_SHA256,
        "ocr_release_gate_descriptor_pin_invalid",
    )
    if GIT_REVISION.fullmatch(CANONICAL_OCR_AUTHORIZATION_COMMIT_SHA) is None:
        raise OcrReleaseGateAttestationError("ocr_release_gate_authorization_commit_invalid")
    descriptor = _regular_file(
        descriptor_path,
        max_bytes=MAX_DESCRIPTOR_BYTES,
        unavailable_code="ocr_release_gate_descriptor_unavailable",
        invalid_code="ocr_release_gate_descriptor_invalid",
    )
    descriptor_sha256 = _file_sha256(
        descriptor,
        unavailable_code="ocr_release_gate_descriptor_unavailable",
    )
    if descriptor_sha256 != descriptor_pin:
        raise OcrReleaseGateAttestationError("ocr_release_gate_descriptor_hash_mismatch")
    try:
        value = json.loads(descriptor.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OcrReleaseGateAttestationError("ocr_release_gate_descriptor_invalid") from exc
    release_context = value.get("release_context") if isinstance(value, dict) else None
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != "ocr-benchmark-descriptor/v3"
        or value.get("evaluation_version") != "evaluation-v3"
        or not isinstance(release_context, dict)
        or set(release_context)
        != {
            "authorization_commit_sha",
            "authorization_receipt_sha256",
            "authorization_receipt_registry_key",
        }
        or release_context.get("authorization_commit_sha") != CANONICAL_OCR_AUTHORIZATION_COMMIT_SHA
        or not isinstance(release_context.get("authorization_receipt_sha256"), str)
        or SHA256.fullmatch(release_context["authorization_receipt_sha256"]) is None
        or not isinstance(release_context.get("authorization_receipt_registry_key"), str)
        or SHA256.fullmatch(release_context["authorization_receipt_registry_key"]) is None
    ):
        raise OcrReleaseGateAttestationError("ocr_release_gate_descriptor_authorization_mismatch")
    return descriptor_sha256, CANONICAL_OCR_AUTHORIZATION_COMMIT_SHA


def _load_report(path: Path) -> tuple[str, dict[str, Any]]:
    resolved = _regular_file(
        path,
        max_bytes=MAX_REPORT_BYTES,
        unavailable_code="ocr_release_gate_report_unavailable",
        invalid_code="ocr_release_gate_report_invalid",
    )
    try:
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OcrReleaseGateAttestationError("ocr_release_gate_report_invalid") from exc
    if not isinstance(value, dict):
        raise OcrReleaseGateAttestationError("ocr_release_gate_report_invalid")
    return (
        _file_sha256(
            resolved,
            unavailable_code="ocr_release_gate_report_unavailable",
        ),
        cast(dict[str, Any], value),
    )


def _validate_input_reports(value: object) -> None:
    if not isinstance(value, dict) or set(value) != INPUT_REPORT_KEYS:
        raise OcrReleaseGateAttestationError("ocr_release_gate_report_invalid")
    report_hashes: list[str] = []
    for sample, expected_count in OCR_RELEASE_SAMPLE_DOCUMENT_COUNTS.items():
        sample_report = value.get(sample)
        if (
            not isinstance(sample_report, dict)
            or set(sample_report) != INPUT_REPORT_VALUE_KEYS
            or sample_report.get("document_count") != expected_count
        ):
            raise OcrReleaseGateAttestationError("ocr_release_gate_coverage_mismatch")
        report_hashes.append(
            _require_sha256(
                sample_report.get("sha256"),
                "ocr_release_gate_report_invalid",
            )
        )
    if len(set(report_hashes)) != len(report_hashes):
        raise OcrReleaseGateAttestationError("ocr_release_gate_report_invalid")


def _validate_provider_eligibility(
    value: object,
    *,
    selected_provider: ProviderName,
    selection_status: str,
) -> None:
    if (
        not isinstance(value, list)
        or len(value) != len(OCR_RELEASE_PROVIDERS)
        or [item.get("provider") if isinstance(item, dict) else None for item in value]
        != list(OCR_RELEASE_PROVIDERS)
    ):
        raise OcrReleaseGateAttestationError("ocr_release_gate_provider_set_invalid")
    eligible_providers: list[str] = []
    for item in value:
        if not isinstance(item, dict) or set(item) != ELIGIBILITY_KEYS:
            raise OcrReleaseGateAttestationError("ocr_release_gate_provider_set_invalid")
        locked_status = item.get("locked_status")
        co_status = item.get("co_regression_status")
        eligible = item.get("eligible")
        if (
            locked_status not in {"PASS", "FAIL"}
            or co_status not in {"PASS", "FAIL"}
            or not isinstance(eligible, bool)
            or eligible != (locked_status == "PASS" and co_status == "PASS")
        ):
            raise OcrReleaseGateAttestationError("ocr_release_gate_provider_status_invalid")
        if eligible:
            eligible_providers.append(cast(str, item["provider"]))
        if item["provider"] == selected_provider and (
            locked_status != "PASS" or co_status != "PASS" or not eligible
        ):
            raise OcrReleaseGateAttestationError("ocr_release_gate_selected_provider_not_passing")
    if selected_provider not in eligible_providers:
        raise OcrReleaseGateAttestationError("ocr_release_gate_selected_provider_not_passing")
    if (selection_status == "SELECTED_SINGLE_COMMON_PROVIDER" and len(eligible_providers) != 1) or (
        selection_status
        in {
            "SELECTED_BY_LOCKED_QUALITY_METRICS",
            "SELECTED_BY_ATTESTED_RUNTIME_METRICS",
        }
        and len(eligible_providers) != 2
    ):
        raise OcrReleaseGateAttestationError("ocr_release_gate_selection_inconsistent")


def _validate_runtime_tiebreak(
    value: object,
    *,
    selected_provider: ProviderName,
    selection_status: str,
) -> None:
    if selection_status != "SELECTED_BY_ATTESTED_RUNTIME_METRICS":
        if value is not None:
            raise OcrReleaseGateAttestationError("ocr_release_gate_runtime_tiebreak_inconsistent")
        return
    if (
        not isinstance(value, dict)
        or set(value) != RUNTIME_TIEBREAK_KEYS
        or value.get("source_sample") != "locked"
        or value.get("selection_order")
        != [
            "latency_p95_ms",
            "peak_memory_bytes",
            "disk_footprint_bytes",
        ]
    ):
        raise OcrReleaseGateAttestationError("ocr_release_gate_runtime_tiebreak_invalid")
    providers = value.get("providers")
    if (
        not isinstance(providers, list)
        or len(providers) != len(OCR_RELEASE_PROVIDERS)
        or [item.get("provider") if isinstance(item, dict) else None for item in providers]
        != list(OCR_RELEASE_PROVIDERS)
    ):
        raise OcrReleaseGateAttestationError("ocr_release_gate_runtime_tiebreak_invalid")
    comparable: dict[ProviderName, tuple[float, int, int]] = {}
    for item in providers:
        if not isinstance(item, dict) or set(item) != RUNTIME_TIEBREAK_PROVIDER_KEYS:
            raise OcrReleaseGateAttestationError("ocr_release_gate_runtime_tiebreak_invalid")
        provider = item.get("provider")
        latency = item.get("latency_p95_ms")
        peak_memory = item.get("peak_memory_bytes")
        disk_footprint = item.get("disk_footprint_bytes")
        if (
            provider not in OCR_RELEASE_PROVIDERS
            or not isinstance(latency, (int, float))
            or isinstance(latency, bool)
            or not math.isfinite(float(latency))
            or float(latency) <= 0.0
            or not isinstance(peak_memory, int)
            or isinstance(peak_memory, bool)
            or peak_memory < 1
            or not isinstance(disk_footprint, int)
            or isinstance(disk_footprint, bool)
            or disk_footprint < 1
        ):
            raise OcrReleaseGateAttestationError("ocr_release_gate_runtime_tiebreak_invalid")
        comparable[cast(ProviderName, provider)] = (
            float(latency),
            peak_memory,
            disk_footprint,
        )
    if (
        len(comparable) != len(OCR_RELEASE_PROVIDERS)
        or len(set(comparable.values())) == 1
        or min(comparable, key=lambda provider: comparable[provider]) != selected_provider
    ):
        raise OcrReleaseGateAttestationError("ocr_release_gate_runtime_tiebreak_inconsistent")


def load_ocr_release_gate_attestation(
    *,
    path: Path,
    expected_report_sha256: str,
    expected_provider: ProviderName,
    expected_descriptor_sha256: str,
    expected_manifest_sha256: str,
    expected_agent_wheel_sha256: str,
    expected_environment_lock_sha256: str,
    expected_model_assets_lock_sha256: str,
) -> OcrReleaseGateAttestation:
    """Load one exact PASS report and bind it to caller-owned identities."""

    expected_report_sha256 = _require_sha256(
        expected_report_sha256,
        "ocr_release_gate_report_hash_invalid",
    )
    expected_descriptor_sha256 = _require_sha256(
        expected_descriptor_sha256,
        "ocr_release_gate_descriptor_hash_invalid",
    )
    expected_manifest_sha256 = _require_sha256(
        expected_manifest_sha256,
        "ocr_release_gate_manifest_hash_invalid",
    )
    expected_agent_wheel_sha256 = _require_sha256(
        expected_agent_wheel_sha256,
        "ocr_release_gate_expected_wheel_hash_invalid",
    )
    expected_environment_lock_sha256 = _require_sha256(
        expected_environment_lock_sha256,
        "ocr_release_gate_expected_environment_lock_hash_invalid",
    )
    expected_model_assets_lock_sha256 = _require_sha256(
        expected_model_assets_lock_sha256,
        "ocr_release_gate_expected_model_assets_lock_hash_invalid",
    )
    if expected_provider not in OCR_RELEASE_PROVIDERS:
        raise OcrReleaseGateAttestationError("ocr_release_gate_requested_provider_invalid")

    actual_report_sha256, report = _load_report(path)
    if actual_report_sha256 != expected_report_sha256:
        raise OcrReleaseGateAttestationError("ocr_release_gate_report_hash_mismatch")
    if (
        set(report) != REPORT_KEYS
        or report.get("schema_version") != OCR_RELEASE_REPORT_SCHEMA
        or report.get("benchmark_id") != OCR_BENCHMARK_ID
        or report.get("smoke_included") is not False
        or report.get("release_document_count") != OCR_RELEASE_DOCUMENT_COUNT
        or report.get("redaction") != OCR_RELEASE_REDACTION
    ):
        raise OcrReleaseGateAttestationError("ocr_release_gate_report_invalid")
    if report.get("status") != "PASS":
        raise OcrReleaseGateAttestationError("ocr_release_gate_status_not_passing")
    if report.get("failure_codes") != []:
        raise OcrReleaseGateAttestationError("ocr_release_gate_report_invalid")

    selection_status = report.get("selection_status")
    selected_provider = report.get("selected_provider")
    if (
        not isinstance(selection_status, str)
        or selection_status not in PASSING_SELECTION_STATUSES
        or selected_provider not in OCR_RELEASE_PROVIDERS
    ):
        raise OcrReleaseGateAttestationError("ocr_release_gate_selection_not_passing")
    if selected_provider != expected_provider:
        raise OcrReleaseGateAttestationError("ocr_release_gate_provider_mismatch")

    descriptor_sha256 = _require_sha256(
        report.get("descriptor_sha256"),
        "ocr_release_gate_identity_invalid",
    )
    manifest_sha256 = _require_sha256(
        report.get("dataset_manifest_sha256"),
        "ocr_release_gate_identity_invalid",
    )
    if (
        descriptor_sha256 != expected_descriptor_sha256
        or manifest_sha256 != expected_manifest_sha256
    ):
        raise OcrReleaseGateAttestationError("ocr_release_gate_identity_mismatch")

    agent_wheel_sha256 = _require_sha256(
        report.get("agent_wheel_sha256"),
        "ocr_release_gate_wheel_hash_invalid",
    )
    if agent_wheel_sha256 != expected_agent_wheel_sha256:
        raise OcrReleaseGateAttestationError("ocr_release_gate_expected_wheel_mismatch")
    environment_lock_sha256 = _require_sha256(
        report.get("environment_lock_sha256"),
        "ocr_release_gate_environment_lock_hash_invalid",
    )
    model_assets_lock_sha256 = _require_sha256(
        report.get("model_assets_lock_sha256"),
        "ocr_release_gate_model_assets_lock_hash_invalid",
    )
    if environment_lock_sha256 != expected_environment_lock_sha256:
        raise OcrReleaseGateAttestationError("ocr_release_gate_environment_lock_hash_mismatch")
    if model_assets_lock_sha256 != expected_model_assets_lock_sha256:
        raise OcrReleaseGateAttestationError("ocr_release_gate_model_assets_lock_hash_mismatch")

    engine = report.get("selected_engine")
    if (
        not isinstance(engine, dict)
        or set(engine) != ENGINE_KEYS
        or not all(isinstance(engine.get(name), str) and bool(engine[name]) for name in ENGINE_KEYS)
        or engine.get("provider_id") != selected_provider
        or engine.get("provider_id") != expected_provider
    ):
        raise OcrReleaseGateAttestationError("ocr_release_gate_engine_mismatch")

    _validate_input_reports(report.get("input_reports"))
    _validate_provider_eligibility(
        report.get("provider_eligibility"),
        selected_provider=cast(ProviderName, selected_provider),
        selection_status=selection_status,
    )
    _validate_runtime_tiebreak(
        report.get("runtime_tiebreak"),
        selected_provider=cast(ProviderName, selected_provider),
        selection_status=selection_status,
    )
    return OcrReleaseGateAttestation(
        report_sha256=actual_report_sha256,
        descriptor_sha256=descriptor_sha256,
        manifest_sha256=manifest_sha256,
        agent_wheel_sha256=agent_wheel_sha256,
        environment_lock_sha256=environment_lock_sha256,
        model_assets_lock_sha256=model_assets_lock_sha256,
        provider=cast(ProviderName, selected_provider),
        provider_id=cast(str, engine["provider_id"]),
        engine_id=cast(str, engine["engine_id"]),
        engine_version=cast(str, engine["engine_version"]),
    )


def verify_installed_agent_wheel(
    attestation: OcrReleaseGateAttestation,
    installed_agent_wheel_sha256: object,
) -> None:
    """Complete the report/expected/installed three-way wheel check."""

    installed_sha256 = _require_sha256(
        installed_agent_wheel_sha256,
        "ocr_release_gate_installed_wheel_hash_invalid",
    )
    if installed_sha256 != attestation.agent_wheel_sha256:
        raise OcrReleaseGateAttestationError("ocr_release_gate_installed_wheel_mismatch")


def _attestation_environment() -> dict[str, str]:
    allowed = {
        "COMSPEC",
        "LANG",
        "LC_ALL",
        "PATH",
        "PATHEXT",
        "PROCESSOR_ARCHITECTURE",
        "PROCESSOR_ARCHITEW6432",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "WINDIR",
    }
    environment = {key: value for key, value in os.environ.items() if key.upper() in allowed}
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONNOUSERSITE"] = "1"
    return environment


def installed_agent_wheel_sha256(executable: Path) -> str:
    """Attest wheel origin, installed RECORD, and executable package inventory."""

    try:
        resolved_executable = executable.expanduser().resolve(strict=True)
    except OSError as exc:
        raise OcrReleaseGateAttestationError("ocr_release_gate_wheel_attestation_failed") from exc
    if not resolved_executable.is_file():
        raise OcrReleaseGateAttestationError("ocr_release_gate_wheel_attestation_failed")
    try:
        # The OCR interpreter is a resolved regular file and the attestation
        # argv is fixed; no shell is involved.
        completed = subprocess.run(  # noqa: S603
            [
                str(resolved_executable),
                "-I",
                "-c",
                _WHEEL_ATTESTATION_SCRIPT,
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=_attestation_environment(),
            timeout=_WHEEL_ATTESTATION_TIMEOUT_SECONDS,
            check=False,
            creationflags=(getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OcrReleaseGateAttestationError("ocr_release_gate_wheel_attestation_failed") from exc
    output = completed.stdout
    if completed.returncode != 0 or not output or len(output) > _WHEEL_ATTESTATION_MAX_OUTPUT_BYTES:
        raise OcrReleaseGateAttestationError("ocr_release_gate_wheel_attestation_failed")
    try:
        value = json.loads(output.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OcrReleaseGateAttestationError("ocr_release_gate_wheel_attestation_failed") from exc
    if not isinstance(value, dict):
        raise OcrReleaseGateAttestationError("ocr_release_gate_wheel_attestation_failed")
    if value.get("schema_version") != _WHEEL_ATTESTATION_SCHEMA or value.get("status") not in {
        "ok",
        "error",
    }:
        raise OcrReleaseGateAttestationError("ocr_release_gate_wheel_attestation_failed")
    if value["status"] == "error":
        if set(value) != {"schema_version", "status", "error_code"}:
            raise OcrReleaseGateAttestationError("ocr_release_gate_wheel_attestation_failed")
        error_code = value.get("error_code")
        if error_code not in _WHEEL_ATTESTATION_ERROR_CODES:
            raise OcrReleaseGateAttestationError("ocr_release_gate_wheel_attestation_failed")
        raise OcrReleaseGateAttestationError(f"ocr_release_gate_{error_code}")
    if set(value) != {
        "schema_version",
        "status",
        "installed_agent_wheel_sha256",
        "installed_agent_record_sha256",
        "verified_executable_file_count",
    }:
        raise OcrReleaseGateAttestationError("ocr_release_gate_wheel_attestation_failed")
    installed_sha256 = value.get("installed_agent_wheel_sha256")
    record_sha256 = value.get("installed_agent_record_sha256")
    executable_count = value.get("verified_executable_file_count")
    _require_sha256(
        record_sha256,
        "ocr_release_gate_wheel_attestation_failed",
    )
    if (
        not isinstance(executable_count, int)
        or isinstance(executable_count, bool)
        or executable_count < 1
    ):
        raise OcrReleaseGateAttestationError("ocr_release_gate_wheel_attestation_failed")
    return _require_sha256(
        installed_sha256,
        "ocr_release_gate_wheel_attestation_failed",
    )


def installed_ocr_environment_lock_sha256(
    executable: Path,
    lock_path: Path,
) -> str:
    """Verify the exact OCR distribution inventory against one hash-locked file."""

    try:
        resolved_executable = executable.expanduser().resolve(strict=True)
    except OSError as exc:
        raise OcrReleaseGateAttestationError(
            "ocr_release_gate_environment_attestation_failed"
        ) from exc
    if not resolved_executable.is_file():
        raise OcrReleaseGateAttestationError("ocr_release_gate_environment_attestation_failed")
    resolved_lock = _regular_file(
        lock_path,
        max_bytes=MAX_ENVIRONMENT_LOCK_BYTES,
        unavailable_code="ocr_release_gate_environment_lock_unavailable",
        invalid_code="ocr_release_gate_environment_lock_invalid",
    )
    try:
        # The OCR interpreter and lock are resolved regular files and argv is
        # fixed; no shell is involved.
        completed = subprocess.run(  # noqa: S603
            [
                str(resolved_executable),
                "-I",
                "-c",
                _ENVIRONMENT_ATTESTATION_SCRIPT,
                str(resolved_lock),
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=_attestation_environment(),
            timeout=_ENVIRONMENT_ATTESTATION_TIMEOUT_SECONDS,
            check=False,
            creationflags=(getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OcrReleaseGateAttestationError(
            "ocr_release_gate_environment_attestation_failed"
        ) from exc
    output = completed.stdout
    if (
        completed.returncode != 0
        or not output
        or len(output) > _ENVIRONMENT_ATTESTATION_MAX_OUTPUT_BYTES
    ):
        raise OcrReleaseGateAttestationError("ocr_release_gate_environment_attestation_failed")
    try:
        value = json.loads(output.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OcrReleaseGateAttestationError(
            "ocr_release_gate_environment_attestation_failed"
        ) from exc
    if not isinstance(value, dict):
        raise OcrReleaseGateAttestationError("ocr_release_gate_environment_attestation_failed")
    if value.get("schema_version") != _ENVIRONMENT_ATTESTATION_SCHEMA or value.get(
        "status"
    ) not in {"ok", "error"}:
        raise OcrReleaseGateAttestationError("ocr_release_gate_environment_attestation_failed")
    if value["status"] == "error":
        if set(value) != {"schema_version", "status", "error_code"}:
            raise OcrReleaseGateAttestationError("ocr_release_gate_environment_attestation_failed")
        error_code = value.get("error_code")
        if error_code not in _ENVIRONMENT_ATTESTATION_ERROR_CODES:
            raise OcrReleaseGateAttestationError("ocr_release_gate_environment_attestation_failed")
        raise OcrReleaseGateAttestationError(f"ocr_release_gate_{error_code}")
    if set(value) != {
        "schema_version",
        "status",
        "environment_lock_sha256",
    }:
        raise OcrReleaseGateAttestationError("ocr_release_gate_environment_attestation_failed")
    return _require_sha256(
        value.get("environment_lock_sha256"),
        "ocr_release_gate_environment_attestation_failed",
    )


def _resolved_cache_root(provider: ProviderName) -> Path:
    try:
        if provider == "paddle":
            from customs_ai_agent.adapters.ocr import _paddle_cache_root

            root = Path(_paddle_cache_root()).resolve(strict=True)
        else:
            hf_home = os.environ.get("HF_HOME")
            base = Path(hf_home).expanduser() if hf_home else Path.home() / ".cache" / "huggingface"
            root = (base / "hub").resolve(strict=True)
    except OSError as exc:
        raise OcrReleaseGateAttestationError("ocr_release_gate_model_cache_unavailable") from exc
    if not root.is_dir() or root.is_symlink():
        raise OcrReleaseGateAttestationError("ocr_release_gate_model_cache_invalid")
    return root


def _safe_child(root: Path, *parts: str, expect_directory: bool) -> Path:
    candidate = root.joinpath(*parts)
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise OcrReleaseGateAttestationError("ocr_release_gate_model_asset_unavailable") from exc
    if candidate.is_symlink() or (
        expect_directory and (not resolved.is_dir() or resolved.is_symlink())
    ):
        raise OcrReleaseGateAttestationError("ocr_release_gate_model_asset_invalid")
    return resolved


def _model_tree_identity(
    root: Path,
    *,
    allowed_root: Path,
    exclude_cache_metadata: bool,
) -> tuple[int, int, str]:
    records: list[tuple[str, int, str]] = []
    try:
        candidates = sorted(root.rglob("*"), key=lambda path: path.as_posix())
        for candidate in candidates:
            relative = candidate.relative_to(root)
            if exclude_cache_metadata and ".cache" in relative.parts:
                continue
            if candidate.is_dir():
                if candidate.is_symlink():
                    raise OcrReleaseGateAttestationError("ocr_release_gate_model_asset_invalid")
                continue
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(allowed_root)
            if not resolved.is_file():
                raise OcrReleaseGateAttestationError("ocr_release_gate_model_asset_invalid")
            size = resolved.stat().st_size
            if size < 0:
                raise OcrReleaseGateAttestationError("ocr_release_gate_model_asset_invalid")
            records.append(
                (
                    relative.as_posix(),
                    size,
                    _file_sha256(
                        resolved,
                        unavailable_code="ocr_release_gate_model_asset_unavailable",
                    ),
                )
            )
    except OcrReleaseGateAttestationError:
        raise
    except (OSError, ValueError) as exc:
        raise OcrReleaseGateAttestationError("ocr_release_gate_model_asset_invalid") from exc
    if not records:
        raise OcrReleaseGateAttestationError("ocr_release_gate_model_asset_invalid")
    tree = hashlib.sha256()
    total_bytes = 0
    for relative_path, size, file_sha256 in records:
        total_bytes += size
        tree.update(relative_path.encode("utf-8"))
        tree.update(b"\0")
        tree.update(str(size).encode("ascii"))
        tree.update(b"\0")
        tree.update(file_sha256.encode("ascii"))
        tree.update(b"\n")
    return len(records), total_bytes, tree.hexdigest()


def _read_revision(path: Path) -> str:
    resolved = _regular_file(
        path,
        max_bytes=256,
        unavailable_code="ocr_release_gate_model_revision_unavailable",
        invalid_code="ocr_release_gate_model_revision_invalid",
    )
    try:
        revision = resolved.read_text(encoding="ascii").strip()
    except (OSError, UnicodeDecodeError) as exc:
        raise OcrReleaseGateAttestationError("ocr_release_gate_model_revision_invalid") from exc
    if GIT_REVISION.fullmatch(revision) is None:
        raise OcrReleaseGateAttestationError("ocr_release_gate_model_revision_invalid")
    return revision


def _read_repository_revisions(path: Path) -> set[str]:
    refs_root = _safe_child(path, expect_directory=True)
    revisions: set[str] = set()
    try:
        entries = list(refs_root.rglob("*"))
    except OSError as exc:
        raise OcrReleaseGateAttestationError("ocr_release_gate_model_revision_unavailable") from exc
    for entry in entries:
        if entry.is_dir():
            if entry.is_symlink():
                raise OcrReleaseGateAttestationError("ocr_release_gate_model_revision_invalid")
            continue
        revisions.add(_read_revision(entry))
    if not revisions:
        raise OcrReleaseGateAttestationError("ocr_release_gate_model_revision_unavailable")
    return revisions


def _validate_model_asset_entry(value: object) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != MODEL_ASSET_KEYS:
        raise OcrReleaseGateAttestationError("ocr_release_gate_model_assets_lock_invalid")
    asset_id = value.get("asset_id")
    revision = value.get("revision")
    file_count = value.get("file_count")
    total_bytes = value.get("total_bytes")
    tree_sha256 = value.get("tree_sha256")
    if (
        not isinstance(asset_id, str)
        or MODEL_ASSET_ID.fullmatch(asset_id) is None
        or not isinstance(revision, str)
        or GIT_REVISION.fullmatch(revision) is None
        or not isinstance(file_count, int)
        or isinstance(file_count, bool)
        or file_count < 1
        or not isinstance(total_bytes, int)
        or isinstance(total_bytes, bool)
        or total_bytes < 1
        or not isinstance(tree_sha256, str)
        or SHA256.fullmatch(tree_sha256) is None
    ):
        raise OcrReleaseGateAttestationError("ocr_release_gate_model_assets_lock_invalid")
    return cast(dict[str, Any], value)


def _direct_model_directories(
    root: Path,
    *,
    prefixes: tuple[str, ...] | None,
) -> set[str]:
    """Return trusted direct model directories, rejecting unsafe entries."""

    observed: set[str] = set()
    try:
        entries = list(root.iterdir())
    except OSError as exc:
        raise OcrReleaseGateAttestationError(
            "ocr_release_gate_model_cache_inventory_unavailable"
        ) from exc
    for entry in entries:
        if prefixes is not None and not entry.name.startswith(prefixes):
            continue
        try:
            resolved = entry.resolve(strict=True)
        except OSError as exc:
            raise OcrReleaseGateAttestationError(
                "ocr_release_gate_model_cache_inventory_unavailable"
            ) from exc
        if entry.is_symlink() or not resolved.is_dir() or resolved.parent != root:
            raise OcrReleaseGateAttestationError("ocr_release_gate_model_cache_inventory_invalid")
        observed.add(entry.name)
    return observed


def _verify_provider_model_cache_allowlist(
    provider: ProviderName,
    cache_root: Path,
    assets: Sequence[Mapping[str, Any]],
) -> None:
    """Reject undeclared provider models before any release OCR is created."""

    parsed_assets = [cast(str, asset["asset_id"]).split("/", 1) for asset in assets]
    expected_repositories = {f"models--{namespace}--{name}" for namespace, name in parsed_assets}
    if provider == "paddle":
        official_models = _safe_child(
            cache_root,
            "official_models",
            expect_directory=True,
        )
        expected_models = {name for _namespace, name in parsed_assets}
        observed_models = _direct_model_directories(
            official_models,
            prefixes=None,
        )
        repository_root = _safe_child(
            cache_root,
            ".cache",
            "huggingface",
            "hub",
            expect_directory=True,
        )
        # The Paddle cache is service-owned and isolated, so every model
        # repository in it must be declared by the release lock.
        observed_repositories = _direct_model_directories(
            repository_root,
            prefixes=("models--",),
        )
        if observed_models != expected_models or observed_repositories != expected_repositories:
            raise OcrReleaseGateAttestationError("ocr_release_gate_model_cache_inventory_mismatch")
        return

    # A Hugging Face hub can be shared with unrelated products.  Reject every
    # undeclared repository in the namespaces actually owned by this provider
    # while leaving other namespaces out of the OCR attestation boundary.
    provider_prefixes = tuple(
        f"models--{namespace}--"
        for namespace in sorted({namespace for namespace, _name in parsed_assets})
    )
    observed_repositories = _direct_model_directories(
        cache_root,
        prefixes=provider_prefixes,
    )
    if observed_repositories != expected_repositories:
        raise OcrReleaseGateAttestationError("ocr_release_gate_model_cache_inventory_mismatch")


def verify_ocr_model_assets(
    provider: ProviderName,
    lock_path: Path,
    expected_engine_id: str,
) -> str:
    """Verify the selected provider cache against one path-free asset lock."""

    resolved_lock = _regular_file(
        lock_path,
        max_bytes=MAX_MODEL_ASSETS_LOCK_BYTES,
        unavailable_code="ocr_release_gate_model_assets_lock_unavailable",
        invalid_code="ocr_release_gate_model_assets_lock_invalid",
    )
    lock_sha256 = _file_sha256(
        resolved_lock,
        unavailable_code="ocr_release_gate_model_assets_lock_unavailable",
    )
    try:
        value = json.loads(resolved_lock.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OcrReleaseGateAttestationError("ocr_release_gate_model_assets_lock_invalid") from exc
    if (
        not isinstance(value, dict)
        or set(value) != MODEL_ASSETS_LOCK_KEYS
        or value.get("schema_version") != MODEL_ASSETS_LOCK_SCHEMA
        or value.get("tree_hash_algorithm") != MODEL_ASSETS_TREE_ALGORITHM
    ):
        raise OcrReleaseGateAttestationError("ocr_release_gate_model_assets_lock_invalid")
    providers = value.get("providers")
    if not isinstance(providers, dict) or set(providers) != set(OCR_RELEASE_PROVIDERS):
        raise OcrReleaseGateAttestationError("ocr_release_gate_model_assets_lock_invalid")
    selected = providers.get(provider)
    if (
        not isinstance(selected, dict)
        or set(selected) != MODEL_ASSETS_PROVIDER_KEYS
        or selected.get("cache_layout") != MODEL_CACHE_LAYOUTS[provider]
        or selected.get("engine_id") != expected_engine_id
        or not isinstance(selected.get("assets"), list)
        or not selected["assets"]
    ):
        raise OcrReleaseGateAttestationError("ocr_release_gate_model_assets_lock_invalid")
    assets = [_validate_model_asset_entry(asset) for asset in selected["assets"]]
    if len({asset["asset_id"] for asset in assets}) != len(assets):
        raise OcrReleaseGateAttestationError("ocr_release_gate_model_assets_lock_invalid")

    cache_root = _resolved_cache_root(provider)
    _verify_provider_model_cache_allowlist(provider, cache_root, assets)
    for asset in assets:
        namespace, name = cast(str, asset["asset_id"]).split("/", 1)
        revision = cast(str, asset["revision"])
        if provider == "paddle":
            model_root = _safe_child(
                cache_root,
                "official_models",
                name,
                expect_directory=True,
            )
            repository_cache = _safe_child(
                cache_root,
                ".cache",
                "huggingface",
                "hub",
                f"models--{namespace}--{name}",
                expect_directory=True,
            )
            observed_revision = _read_revision(repository_cache / "refs" / "main")
            allowed_root = cache_root
            exclude_cache_metadata = True
        else:
            repository_cache = _safe_child(
                cache_root,
                f"models--{namespace}--{name}",
                expect_directory=True,
            )
            observed_revisions = _read_repository_revisions(repository_cache / "refs")
            model_root = _safe_child(
                repository_cache,
                "snapshots",
                revision,
                expect_directory=True,
            )
            allowed_root = repository_cache
            exclude_cache_metadata = False
        if provider == "paddle":
            revision_matches = observed_revision == revision
        else:
            revision_matches = observed_revisions == {revision}
        if not revision_matches:
            raise OcrReleaseGateAttestationError("ocr_release_gate_model_revision_mismatch")
        observed = _model_tree_identity(
            model_root,
            allowed_root=allowed_root,
            exclude_cache_metadata=exclude_cache_metadata,
        )
        expected = (
            asset["file_count"],
            asset["total_bytes"],
            asset["tree_sha256"],
        )
        if observed != expected:
            raise OcrReleaseGateAttestationError("ocr_release_gate_model_asset_inventory_mismatch")
    return lock_sha256


def load_ocr_release_runtime_gate(
    *,
    report_path: Path,
    expected_report_sha256: str,
    expected_provider: ProviderName,
    expected_descriptor_sha256: str,
    expected_manifest_sha256: str,
    expected_agent_wheel_sha256: str,
    environment_lock_path: Path,
    expected_environment_lock_sha256: str,
    model_assets_lock_path: Path,
    expected_model_assets_lock_sha256: str,
    ocr_python: Path,
    wheel_attestor: WheelAttestor = installed_agent_wheel_sha256,
    environment_attestor: EnvironmentAttestor = installed_ocr_environment_lock_sha256,
    model_assets_attestor: ModelAssetsAttestor = verify_ocr_model_assets,
) -> OcrReleaseGateAttestation:
    """Bind a validated dataset identity to the exact selected OCR runtime."""

    environment_lock = _regular_file(
        environment_lock_path,
        max_bytes=MAX_ENVIRONMENT_LOCK_BYTES,
        unavailable_code="ocr_release_gate_environment_lock_unavailable",
        invalid_code="ocr_release_gate_environment_lock_invalid",
    )
    environment_lock_sha256 = _file_sha256(
        environment_lock,
        unavailable_code="ocr_release_gate_environment_lock_unavailable",
    )
    if environment_lock_sha256 != _require_sha256(
        expected_environment_lock_sha256,
        "ocr_release_gate_expected_environment_lock_hash_invalid",
    ):
        raise OcrReleaseGateAttestationError("ocr_release_gate_environment_lock_hash_mismatch")

    model_assets_lock = _regular_file(
        model_assets_lock_path,
        max_bytes=MAX_MODEL_ASSETS_LOCK_BYTES,
        unavailable_code="ocr_release_gate_model_assets_lock_unavailable",
        invalid_code="ocr_release_gate_model_assets_lock_invalid",
    )
    model_assets_lock_sha256 = _file_sha256(
        model_assets_lock,
        unavailable_code="ocr_release_gate_model_assets_lock_unavailable",
    )
    if model_assets_lock_sha256 != _require_sha256(
        expected_model_assets_lock_sha256,
        "ocr_release_gate_expected_model_assets_lock_hash_invalid",
    ):
        raise OcrReleaseGateAttestationError("ocr_release_gate_model_assets_lock_hash_mismatch")

    attestation = load_ocr_release_gate_attestation(
        path=report_path,
        expected_report_sha256=expected_report_sha256,
        expected_provider=expected_provider,
        expected_descriptor_sha256=expected_descriptor_sha256,
        expected_manifest_sha256=expected_manifest_sha256,
        expected_agent_wheel_sha256=expected_agent_wheel_sha256,
        expected_environment_lock_sha256=environment_lock_sha256,
        expected_model_assets_lock_sha256=model_assets_lock_sha256,
    )
    try:
        installed_wheel_sha256 = wheel_attestor(ocr_python)
    except OcrReleaseGateAttestationError:
        raise
    except Exception as exc:
        raise OcrReleaseGateAttestationError("ocr_release_gate_wheel_attestation_failed") from exc
    verify_installed_agent_wheel(attestation, installed_wheel_sha256)
    try:
        observed_environment_lock_sha256 = _require_sha256(
            environment_attestor(
                ocr_python,
                environment_lock,
            ),
            "ocr_release_gate_environment_attestation_failed",
        )
    except OcrReleaseGateAttestationError:
        raise
    except Exception as exc:
        raise OcrReleaseGateAttestationError(
            "ocr_release_gate_environment_attestation_failed"
        ) from exc
    if observed_environment_lock_sha256 != attestation.environment_lock_sha256:
        raise OcrReleaseGateAttestationError("ocr_release_gate_environment_lock_hash_mismatch")
    try:
        observed_model_assets_lock_sha256 = _require_sha256(
            model_assets_attestor(
                expected_provider,
                model_assets_lock,
                attestation.engine_id,
            ),
            "ocr_release_gate_model_assets_attestation_failed",
        )
    except OcrReleaseGateAttestationError:
        raise
    except Exception as exc:
        raise OcrReleaseGateAttestationError(
            "ocr_release_gate_model_assets_attestation_failed"
        ) from exc
    if observed_model_assets_lock_sha256 != attestation.model_assets_lock_sha256:
        raise OcrReleaseGateAttestationError("ocr_release_gate_model_assets_lock_hash_mismatch")
    return attestation


def load_canonical_ocr_release_gate(
    *,
    report_path: Path,
    expected_report_sha256: str,
    expected_provider: ProviderName,
    descriptor_path: Path,
    rendered_root: Path,
    expected_agent_wheel_sha256: str,
    environment_lock_path: Path,
    expected_environment_lock_sha256: str,
    model_assets_lock_path: Path,
    expected_model_assets_lock_sha256: str,
    ocr_python: Path,
    wheel_attestor: WheelAttestor = installed_agent_wheel_sha256,
    environment_attestor: EnvironmentAttestor = installed_ocr_environment_lock_sha256,
    model_assets_attestor: ModelAssetsAttestor = verify_ocr_model_assets,
) -> OcrReleaseGateAttestation:
    """Bind a real OCR runtime to the canonical 60+20 PASS report before use."""

    if expected_provider not in OCR_RELEASE_PROVIDERS:
        raise OcrReleaseGateAttestationError("ocr_release_gate_requested_provider_invalid")
    if expected_environment_lock_sha256 != CANONICAL_OCR_ENVIRONMENT_LOCK_SHA256[expected_provider]:
        raise OcrReleaseGateAttestationError("ocr_release_gate_environment_lock_not_canonical")
    if expected_model_assets_lock_sha256 != CANONICAL_OCR_MODEL_ASSETS_LOCK_SHA256:
        raise OcrReleaseGateAttestationError("ocr_release_gate_model_assets_lock_not_canonical")
    descriptor_sha256, _authorization_commit_sha = load_canonical_ocr_descriptor_identity(
        descriptor_path
    )

    try:
        resolved_root = rendered_root.expanduser().resolve(strict=True)
    except OSError as exc:
        raise OcrReleaseGateAttestationError("ocr_release_gate_rendered_root_unavailable") from exc
    if not resolved_root.is_dir() or resolved_root.is_symlink() or rendered_root.is_symlink():
        raise OcrReleaseGateAttestationError("ocr_release_gate_rendered_root_invalid")
    manifest = _regular_file(
        resolved_root / "manifest.json",
        max_bytes=MAX_MANIFEST_BYTES,
        unavailable_code="ocr_release_gate_manifest_unavailable",
        invalid_code="ocr_release_gate_manifest_invalid",
    )
    manifest_sha256 = _file_sha256(
        manifest,
        unavailable_code="ocr_release_gate_manifest_unavailable",
    )
    if manifest_sha256 != CANONICAL_OCR_MANIFEST_SHA256:
        raise OcrReleaseGateAttestationError("ocr_release_gate_manifest_hash_mismatch")

    attestation = load_ocr_release_runtime_gate(
        report_path=report_path,
        expected_report_sha256=expected_report_sha256,
        expected_provider=expected_provider,
        expected_descriptor_sha256=descriptor_sha256,
        expected_manifest_sha256=manifest_sha256,
        expected_agent_wheel_sha256=expected_agent_wheel_sha256,
        environment_lock_path=environment_lock_path,
        expected_environment_lock_sha256=expected_environment_lock_sha256,
        model_assets_lock_path=model_assets_lock_path,
        expected_model_assets_lock_sha256=expected_model_assets_lock_sha256,
        ocr_python=ocr_python,
        wheel_attestor=wheel_attestor,
        environment_attestor=environment_attestor,
        model_assets_attestor=model_assets_attestor,
    )
    if attestation.engine_id != CANONICAL_OCR_PROVIDER_ENGINE_IDS[expected_provider]:
        raise OcrReleaseGateAttestationError("ocr_release_gate_engine_not_canonical")
    sealed_attestation = replace(attestation)
    object.__setattr__(
        sealed_attestation,
        "_verification_proof",
        _CANONICAL_RELEASE_PROOF,
    )
    return sealed_attestation
