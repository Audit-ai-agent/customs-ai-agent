"""Compare isolated OCR/layout providers against one immutable rendered Gold.

The report deliberately contains hashes, counts, metrics, and safe error codes
only. It never contains source text, image payloads, filesystem paths, secrets,
or backend exception messages.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import math
import os
import re
import subprocess
import time
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Literal, cast

from customs_ai_agent.adapters import (
    OcrLayoutProvider,
    OcrLayoutProviderError,
    build_ocr_layout_provider,
    image_page_from_file,
)
from customs_ai_agent.ocr_release import (
    CANONICAL_OCR_ENVIRONMENT_LOCK_SHA256 as CANONICAL_ENVIRONMENT_LOCK_SHA256,
    CANONICAL_OCR_MODEL_ASSETS_LOCK_SHA256 as CANONICAL_MODEL_ASSETS_LOCK_SHA256,
    CANONICAL_OCR_PROVIDER_ENGINE_IDS as PROVIDER_ENGINE_IDS,
    OCR_RELEASE_PROVIDERS,
    OcrReleaseGateAttestationError,
    installed_agent_wheel_sha256 as _release_installed_agent_wheel_sha256,
    installed_ocr_environment_lock_sha256,
    verify_ocr_model_assets,
)
from customs_ai_agent.ocr_freeze import (
    OcrFreezeVerificationError,
    verify_ocr_release_freeze,
)
from customs_ai_agent.ocr_identity import OCR_MODEL_SET_IDS
from customs_ai_agent.schemas.models import OcrLayoutLedger, OcrLayoutSpan

REPORT_SCHEMA = "ocr-benchmark-report/v2"
DESCRIPTOR_SCHEMA = "ocr-benchmark-descriptor/v2"
BENCHMARK_ID = "rendered-trade-documents-ocr-v2"
PROVIDERS: tuple[Literal["paddle", "docling"], ...] = OCR_RELEASE_PROVIDERS
SAFE_CODE = re.compile(r"^[a-z][a-z0-9_]{0,63}$")
SHA256 = re.compile(r"^[a-f0-9]{64}$")

# Gate values are code-owned. A descriptor may repeat them for auditability,
# but it cannot weaken or override them.
LOCKED_THRESHOLDS: dict[str, int | float] = {
    "critical_identifier_exact_accuracy_min": 1.0,
    "character_error_rate_max": 0.01,
    "table_row_f1_min": 0.98,
    "table_cell_f1_min": 0.98,
    "bbox_f1_iou_0_5_min": 0.95,
    "crash_count_max": 0,
}
ALLOWED_SELECTORS = frozenset({"identity", "before_slash", "before_double_slash"})
LEAF_ROLES = frozenset({"text", "heading", "label", "value"})

ProviderName = Literal["paddle", "docling"]
SampleName = Literal["smoke", "locked", "co_regression"]
ProviderFactory = Callable[[ProviderName, Path], OcrLayoutProvider]
WheelAttestor = Callable[[Path], str]
EnvironmentAttestor = Callable[[Path, Path], str]
ModelAssetsAttestor = Callable[[ProviderName, Path, str], str]
RuntimeAttestor = Callable[
    [ProviderName, Path, OcrLayoutProvider, Path],
    tuple[int, int],
]
SourceFreezeVerifier = Callable[..., object]

_RUNTIME_ATTESTATION_SCHEMA = "customs-ocr-runtime-footprint/v1"
_RUNTIME_ATTESTATION_TIMEOUT_SECONDS = 30.0
_RUNTIME_ATTESTATION_MAX_OUTPUT_BYTES = 2048
_RUNTIME_ATTESTATION_SCRIPT = r"""
import importlib.metadata
import json
import os
import sys
from pathlib import Path

SCHEMA = "customs-ocr-runtime-footprint/v1"


def emit_error(code):
    sys.stdout.write(
        json.dumps(
            {"schema_version": SCHEMA, "status": "error", "error_code": code},
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    raise SystemExit(0)


try:
    prefix = Path(sys.prefix).resolve(strict=True)
    observed = {}
    total_bytes = 0
    distribution_count = 0
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        version = distribution.version
        files = distribution.files
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(version, str)
            or not version
            or files is None
        ):
            emit_error("runtime_inventory_invalid")
        identity = (name.casefold(), version)
        if identity in observed:
            emit_error("runtime_inventory_invalid")
        observed[identity] = True
        distribution_count += 1
        for relative in files:
            candidate = Path(distribution.locate_file(relative))
            if candidate.is_symlink():
                emit_error("runtime_inventory_invalid")
            resolved = candidate.resolve(strict=True)
            resolved.relative_to(prefix)
            if not resolved.is_file() or resolved.is_symlink():
                emit_error("runtime_inventory_invalid")
            key = os.path.normcase(str(resolved))
            if key in observed:
                continue
            observed[key] = True
            size = resolved.stat().st_size
            if size < 0:
                emit_error("runtime_inventory_invalid")
            total_bytes += size
    executable = Path(sys.executable)
    if executable.is_symlink():
        emit_error("runtime_inventory_invalid")
    resolved_executable = executable.resolve(strict=True)
    resolved_executable.relative_to(prefix)
    executable_key = os.path.normcase(str(resolved_executable))
    if executable_key not in observed:
        total_bytes += resolved_executable.stat().st_size
    if distribution_count < 1 or total_bytes < 1:
        emit_error("runtime_inventory_invalid")
except SystemExit:
    raise
except Exception:
    emit_error("runtime_inventory_invalid")

sys.stdout.write(
    json.dumps(
        {
            "schema_version": SCHEMA,
            "status": "ok",
            "distribution_count": distribution_count,
            "distribution_disk_bytes": total_bytes,
        },
        separators=(",", ":"),
        sort_keys=True,
    )
)
"""

LOCKED_DOCUMENT_TYPES = (
    ("commercial_invoice", "commercial-invoice"),
    ("packing_list", "packing-list"),
    ("bill_of_lading", "bill-of-lading"),
)
CANONICAL_SPLITS_SHA256 = "d7e6b86ef19bbae838a7f5da7b94428637481ab5c15d9599ee3af3e86bda7b64"
EVALUATION_VERSION = "evaluation-v2"
RELEASE_CONTEXT_KEYS = frozenset(
    {
        "authorization_commit_sha",
        "authorization_receipt_sha256",
        "authorization_receipt_registry_key",
    }
)
COMMIT_SHA = re.compile(r"^[a-f0-9]{40}$")
EXPECTED_COVERAGE_KEYS = {
    "smoke_split",
    "smoke_case_id",
    "smoke_document_count",
    "locked_split",
    "locked_case_ids",
    "locked_case_count",
    "locked_document_types",
    "locked_triplet_document_count",
    "co_document_count",
    "release_gate_document_count",
    "descriptor_document_count",
}


class OcrBenchmarkError(RuntimeError):
    """Safe benchmark preflight failure represented by a stable code only."""

    def __init__(self, code: str) -> None:
        if SAFE_CODE.fullmatch(code) is None:
            code = "benchmark_internal_error"
        super().__init__(code)
        self.code = code


@dataclass(frozen=True)
class BenchmarkDocument:
    benchmark_document_id: str
    image_sha256: str
    layout_sha256: str
    image_path: Path
    layout: dict[str, Any]


@dataclass
class MatchCounts:
    true_positive: int = 0
    false_positive: int = 0
    false_negative: int = 0

    def add(self, other: MatchCounts) -> None:
        self.true_positive += other.true_positive
        self.false_positive += other.false_positive
        self.false_negative += other.false_negative

    @property
    def f1(self) -> float:
        denominator = 2 * self.true_positive + self.false_positive + self.false_negative
        return 2 * self.true_positive / denominator if denominator else 1.0

    def report(self) -> dict[str, int | float]:
        return {
            "true_positive": self.true_positive,
            "false_positive": self.false_positive,
            "false_negative": self.false_negative,
            "f1": self.f1,
        }


@dataclass
class ProviderTotals:
    critical_correct: int = 0
    critical_total: int = 0
    edit_distance: int = 0
    gold_characters: int = 0
    table_rows: MatchCounts | None = None
    table_cells: MatchCounts | None = None
    bboxes: MatchCounts | None = None
    crash_count: int = 0
    successful_document_count: int = 0

    def __post_init__(self) -> None:
        self.table_rows = self.table_rows or MatchCounts()
        self.table_cells = self.table_cells or MatchCounts()
        self.bboxes = self.bboxes or MatchCounts()


def _p95_latency_ms(latencies_ns: Sequence[int]) -> float:
    if not latencies_ns or any(
        not isinstance(value, int) or isinstance(value, bool) or value <= 0
        for value in latencies_ns
    ):
        raise OcrBenchmarkError("benchmark_runtime_latency_invalid")
    ordered = sorted(latencies_ns)
    rank = math.ceil(0.95 * len(ordered))
    return ordered[rank - 1] / 1_000_000


def _peak_worker_memory_bytes(process_id: int) -> int:
    if not isinstance(process_id, int) or isinstance(process_id, bool) or process_id < 1:
        raise OcrBenchmarkError("benchmark_runtime_process_invalid")
    if os.name == "nt":

        class _ProcessMemoryCountersEx(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("PageFaultCount", ctypes.c_ulong),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        process_query_information = 0x0400
        process_vm_read = 0x0010
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        psapi = ctypes.WinDLL("psapi", use_last_error=True)
        kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
        kernel32.OpenProcess.restype = ctypes.c_void_p
        kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
        kernel32.CloseHandle.restype = ctypes.c_int
        psapi.GetProcessMemoryInfo.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(_ProcessMemoryCountersEx),
            ctypes.c_ulong,
        ]
        psapi.GetProcessMemoryInfo.restype = ctypes.c_int
        handle = kernel32.OpenProcess(
            process_query_information | process_vm_read,
            0,
            process_id,
        )
        if not handle:
            raise OcrBenchmarkError("benchmark_runtime_memory_unavailable")
        counters = _ProcessMemoryCountersEx()
        counters.cb = ctypes.sizeof(counters)
        try:
            if not psapi.GetProcessMemoryInfo(
                handle,
                ctypes.byref(counters),
                counters.cb,
            ):
                raise OcrBenchmarkError("benchmark_runtime_memory_unavailable")
        finally:
            kernel32.CloseHandle(handle)
        peak_bytes = int(counters.PeakWorkingSetSize)
    elif os.name == "posix":
        try:
            status = Path(f"/proc/{process_id}/status").read_text(encoding="ascii")
        except (OSError, UnicodeDecodeError) as exc:
            raise OcrBenchmarkError("benchmark_runtime_memory_unavailable") from exc
        matched = re.search(r"^VmHWM:\s+([1-9][0-9]*)\s+kB$", status, re.MULTILINE)
        if matched is None:
            raise OcrBenchmarkError("benchmark_runtime_memory_unavailable")
        peak_bytes = int(matched.group(1)) * 1024
    else:
        raise OcrBenchmarkError("benchmark_runtime_platform_unsupported")
    if peak_bytes < 1:
        raise OcrBenchmarkError("benchmark_runtime_memory_invalid")
    return peak_bytes


def _runtime_attestation_environment() -> dict[str, str]:
    allowed = {
        "COMSPEC",
        "LANG",
        "LC_ALL",
        "PATH",
        "PATHEXT",
        "SYSTEMROOT",
        "TEMP",
        "TMP",
        "TMPDIR",
        "WINDIR",
    }
    environment = {
        key: value
        for key, value in os.environ.items()
        if key.upper() in allowed and isinstance(value, str)
    }
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    return environment


def _installed_distribution_disk_bytes(executable: Path) -> int:
    try:
        resolved_executable = executable.expanduser().resolve(strict=True)
    except OSError as exc:
        raise OcrBenchmarkError("benchmark_runtime_disk_attestation_failed") from exc
    if not resolved_executable.is_file():
        raise OcrBenchmarkError("benchmark_runtime_disk_attestation_failed")
    try:
        # The interpreter is a resolved regular file and the attestation argv
        # is fixed; no shell is involved.
        completed = subprocess.run(  # noqa: S603
            [
                str(resolved_executable),
                "-I",
                "-c",
                _RUNTIME_ATTESTATION_SCRIPT,
            ],
            check=False,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=_RUNTIME_ATTESTATION_TIMEOUT_SECONDS,
            env=_runtime_attestation_environment(),
            creationflags=(getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise OcrBenchmarkError("benchmark_runtime_disk_attestation_failed") from exc
    output = completed.stdout
    if (
        completed.returncode != 0
        or not output
        or len(output) > _RUNTIME_ATTESTATION_MAX_OUTPUT_BYTES
    ):
        raise OcrBenchmarkError("benchmark_runtime_disk_attestation_failed")
    try:
        value = json.loads(output.decode("utf-8", errors="strict"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OcrBenchmarkError("benchmark_runtime_disk_attestation_failed") from exc
    if (
        not isinstance(value, dict)
        or value.get("schema_version") != _RUNTIME_ATTESTATION_SCHEMA
        or value.get("status") != "ok"
        or set(value)
        != {
            "schema_version",
            "status",
            "distribution_count",
            "distribution_disk_bytes",
        }
        or not isinstance(value.get("distribution_count"), int)
        or isinstance(value.get("distribution_count"), bool)
        or value["distribution_count"] < 1
        or not isinstance(value.get("distribution_disk_bytes"), int)
        or isinstance(value.get("distribution_disk_bytes"), bool)
        or value["distribution_disk_bytes"] < 1
    ):
        raise OcrBenchmarkError("benchmark_runtime_disk_attestation_failed")
    return cast(int, value["distribution_disk_bytes"])


def _model_asset_disk_bytes(provider: ProviderName, lock_path: Path) -> int:
    try:
        value = json.loads(lock_path.read_text(encoding="utf-8"))
        assets = value["providers"][provider]["assets"]
    except (OSError, UnicodeDecodeError, json.JSONDecodeError, KeyError, TypeError) as exc:
        raise OcrBenchmarkError("benchmark_runtime_model_footprint_invalid") from exc
    if (
        not isinstance(assets, list)
        or not assets
        or any(
            not isinstance(asset, dict)
            or not isinstance(asset.get("total_bytes"), int)
            or isinstance(asset.get("total_bytes"), bool)
            or asset["total_bytes"] < 1
            for asset in assets
        )
    ):
        raise OcrBenchmarkError("benchmark_runtime_model_footprint_invalid")
    return sum(cast(int, asset["total_bytes"]) for asset in assets)


def attest_runtime_footprint(
    provider: ProviderName,
    executable: Path,
    runtime_provider: OcrLayoutProvider,
    model_assets_lock_path: Path,
) -> tuple[int, int]:
    """Return path-free peak memory and exact-identity disk footprint metrics."""

    process_id_reader = getattr(runtime_provider, "runtime_process_id", None)
    if not callable(process_id_reader):
        raise OcrBenchmarkError("benchmark_runtime_process_unavailable")
    try:
        process_id = process_id_reader()
    except Exception as exc:
        raise OcrBenchmarkError("benchmark_runtime_process_unavailable") from exc
    peak_memory_bytes = _peak_worker_memory_bytes(process_id)
    distribution_bytes = _installed_distribution_disk_bytes(executable)
    model_asset_bytes = _model_asset_disk_bytes(provider, model_assets_lock_path)
    disk_footprint_bytes = distribution_bytes + model_asset_bytes
    if disk_footprint_bytes < 1:
        raise OcrBenchmarkError("benchmark_runtime_disk_footprint_invalid")
    return peak_memory_bytes, disk_footprint_bytes


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    except OSError as exc:
        raise OcrBenchmarkError("benchmark_file_unreadable") from exc
    return digest.hexdigest()


def _file_size(path: Path) -> int:
    try:
        return path.stat().st_size
    except OSError as exc:
        raise OcrBenchmarkError("benchmark_file_unreadable") from exc


def _load_json(path: Path, *, error_code: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise OcrBenchmarkError(error_code) from exc
    if not isinstance(value, dict):
        raise OcrBenchmarkError(error_code)
    return cast(dict[str, Any], value)


def _regular_relative_file(root: Path, relative: object) -> Path:
    if not isinstance(relative, str) or "\\" in relative:
        raise OcrBenchmarkError("benchmark_relative_path_invalid")
    pure = PurePosixPath(relative)
    if pure.is_absolute() or not pure.parts or any(part in {"", ".", ".."} for part in pure.parts):
        raise OcrBenchmarkError("benchmark_relative_path_invalid")
    try:
        root_resolved = root.resolve(strict=True)
        candidate = root_resolved.joinpath(*pure.parts).resolve(strict=True)
    except OSError as exc:
        raise OcrBenchmarkError("benchmark_file_unreadable") from exc
    if (
        not candidate.is_relative_to(root_resolved)
        or not candidate.is_file()
        or candidate.is_symlink()
    ):
        raise OcrBenchmarkError("benchmark_relative_path_invalid")
    return candidate


def _validate_descriptor(
    descriptor: dict[str, Any],
    *,
    sample: SampleName,
) -> tuple[list[str], list[dict[str, str]], list[dict[str, str]]]:
    if (
        descriptor.get("schema_version") != DESCRIPTOR_SCHEMA
        or descriptor.get("benchmark_id") != BENCHMARK_ID
        or descriptor.get("evaluation_version") != EVALUATION_VERSION
        or descriptor.get("input_variant") != "claude_input_png"
    ):
        raise OcrBenchmarkError("benchmark_descriptor_schema_invalid")
    if descriptor.get("thresholds") != LOCKED_THRESHOLDS:
        raise OcrBenchmarkError("benchmark_thresholds_changed")
    manifest_hash = descriptor.get("dataset_manifest_sha256")
    splits_hash = descriptor.get("dataset_splits_sha256")
    if (
        not isinstance(manifest_hash, str)
        or SHA256.fullmatch(manifest_hash) is None
        or splits_hash != CANONICAL_SPLITS_SHA256
    ):
        raise OcrBenchmarkError("benchmark_descriptor_invalid")
    release_context = descriptor.get("release_context")
    if (
        not isinstance(release_context, dict)
        or set(release_context) != RELEASE_CONTEXT_KEYS
        or not isinstance(release_context.get("authorization_commit_sha"), str)
        or COMMIT_SHA.fullmatch(release_context["authorization_commit_sha"]) is None
        or any(
            not isinstance(release_context.get(key), str)
            or SHA256.fullmatch(release_context[key]) is None
            for key in (
                "authorization_receipt_sha256",
                "authorization_receipt_registry_key",
            )
        )
    ):
        raise OcrBenchmarkError("benchmark_release_context_invalid")

    rules = descriptor.get("critical_identifier_rules")
    if not isinstance(rules, list) or not rules:
        raise OcrBenchmarkError("benchmark_descriptor_invalid")
    validated_rules: list[dict[str, str]] = []
    for rule in rules:
        if (
            not isinstance(rule, dict)
            or set(rule) != {"field_id_marker", "selector"}
            or not isinstance(rule.get("field_id_marker"), str)
            or not rule["field_id_marker"]
            or rule.get("selector") not in ALLOWED_SELECTORS
        ):
            raise OcrBenchmarkError("benchmark_descriptor_invalid")
        validated_rules.append(
            {
                "field_id_marker": rule["field_id_marker"],
                "selector": rule["selector"],
            }
        )

    samples = descriptor.get("samples")
    coverage = descriptor.get("coverage")
    if (
        not isinstance(samples, dict)
        or set(samples) != {"smoke", "locked", "co_regression"}
        or not isinstance(coverage, dict)
        or set(coverage) != EXPECTED_COVERAGE_KEYS
    ):
        raise OcrBenchmarkError("benchmark_coverage_invalid")
    locked_case_ids = coverage.get("locked_case_ids")
    expected_locked_types = [name for name, _ in LOCKED_DOCUMENT_TYPES]
    if (
        coverage.get("smoke_split") != "development"
        or coverage.get("smoke_case_id") != "trade-triplet-0001"
        or coverage.get("smoke_document_count") != 1
        or coverage.get("locked_split") != "locked_test"
        or not isinstance(locked_case_ids, list)
        or len(locked_case_ids) != 20
        or len(set(locked_case_ids)) != 20
        or not all(
            isinstance(case_id, str) and re.fullmatch(r"trade-triplet-\d{4}", case_id)
            for case_id in locked_case_ids
        )
        or locked_case_ids != sorted(locked_case_ids)
        or coverage.get("locked_case_count") != 20
        or coverage.get("locked_document_types") != expected_locked_types
        or coverage.get("locked_triplet_document_count") != 60
        or coverage.get("co_document_count") != 20
        or coverage.get("release_gate_document_count") != 80
        or coverage.get("descriptor_document_count") != 81
    ):
        raise OcrBenchmarkError("benchmark_coverage_invalid")

    expected_smoke_ids = ["trade-triplet-0001-commercial-invoice"]
    expected_locked_ids = [
        f"{case_id}-{suffix}" for case_id in locked_case_ids for _, suffix in LOCKED_DOCUMENT_TYPES
    ]
    expected_co_ids = [f"co-synthetic-{index:04d}-certificate-of-origin" for index in range(1, 21)]
    if (
        samples.get("smoke") != expected_smoke_ids
        or samples.get("locked") != expected_locked_ids
        or samples.get("co_regression") != expected_co_ids
    ):
        raise OcrBenchmarkError("benchmark_coverage_invalid")

    documents = descriptor.get("documents")
    if not isinstance(documents, list):
        raise OcrBenchmarkError("benchmark_descriptor_invalid")
    sample_ids = samples.get(sample)
    if (
        not isinstance(sample_ids, list)
        or not sample_ids
        or not all(isinstance(value, str) and value for value in sample_ids)
        or len(sample_ids) != len(set(sample_ids))
    ):
        raise OcrBenchmarkError("benchmark_descriptor_invalid")
    validated_documents: list[dict[str, str]] = []
    document_ids: set[str] = set()
    required_keys = {
        "benchmark_document_id",
        "image_relative_path",
        "image_sha256",
        "layout_relative_path",
        "layout_sha256",
    }
    for document in documents:
        if not isinstance(document, dict) or set(document) != required_keys:
            raise OcrBenchmarkError("benchmark_descriptor_invalid")
        if not all(isinstance(document[key], str) for key in required_keys):
            raise OcrBenchmarkError("benchmark_descriptor_invalid")
        document_id = document["benchmark_document_id"]
        if (
            not document_id
            or document_id in document_ids
            or SHA256.fullmatch(document["image_sha256"]) is None
            or SHA256.fullmatch(document["layout_sha256"]) is None
        ):
            raise OcrBenchmarkError("benchmark_descriptor_invalid")
        document_ids.add(document_id)
        validated_documents.append(cast(dict[str, str], document))
    expected_document_ids = [
        *expected_smoke_ids,
        *expected_locked_ids,
        *expected_co_ids,
    ]
    if [
        record["benchmark_document_id"] for record in validated_documents
    ] != expected_document_ids or document_ids != set(expected_document_ids):
        raise OcrBenchmarkError("benchmark_coverage_invalid")
    return cast(list[str], sample_ids), validated_documents, validated_rules


def _manifest_generated_files(
    manifest: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    generated_files = manifest.get("generated_files")
    counts = manifest.get("counts")
    if (
        not isinstance(generated_files, list)
        or not isinstance(counts, dict)
        or counts.get("source_case_count") != 100
        or counts.get("triplet_document_count") != 300
        or counts.get("certificate_of_origin_document_count") != 20
        or counts.get("document_count") != 320
        or counts.get("generated_document_file_count") != 1280
        or counts.get("claude_input_resized_document_count") != 320
        or len(generated_files) != 1280
    ):
        raise OcrBenchmarkError("benchmark_manifest_invalid")
    by_path: dict[str, dict[str, Any]] = {}
    for record in generated_files:
        if (
            not isinstance(record, dict)
            or set(record) != {"relative_path", "sha256", "byte_size"}
            or not isinstance(record.get("relative_path"), str)
            or not isinstance(record.get("sha256"), str)
            or SHA256.fullmatch(record["sha256"]) is None
            or not isinstance(record.get("byte_size"), int)
            or record["byte_size"] < 0
            or record["relative_path"] in by_path
        ):
            raise OcrBenchmarkError("benchmark_manifest_invalid")
        by_path[record["relative_path"]] = record
    return by_path


def _expected_layout_identity(
    benchmark_document_id: str,
) -> tuple[str, str, str]:
    triplet = re.fullmatch(
        r"(trade-triplet-\d{4})-(commercial-invoice|packing-list|bill-of-lading)",
        benchmark_document_id,
    )
    if triplet is not None:
        by_suffix = {suffix: document_type for document_type, suffix in LOCKED_DOCUMENT_TYPES}
        case_id = triplet.group(1)
        document_type = by_suffix[triplet.group(2)]
        return case_id, document_type, f"{case_id}:{document_type}"
    certificate = re.fullmatch(
        r"(co-synthetic-\d{4})-certificate-of-origin",
        benchmark_document_id,
    )
    if certificate is not None:
        case_id = certificate.group(1)
        document_type = "certificate_of_origin"
        return case_id, document_type, f"{case_id}:{document_type}"
    raise OcrBenchmarkError("benchmark_descriptor_invalid")


def load_benchmark(
    *,
    descriptor_path: Path,
    rendered_root: Path,
    sample: SampleName,
) -> tuple[str, str, list[BenchmarkDocument], list[dict[str, str]]]:
    if not descriptor_path.is_file() or descriptor_path.is_symlink() or not rendered_root.is_dir():
        raise OcrBenchmarkError("benchmark_input_unavailable")
    descriptor_hash = _sha256(descriptor_path)
    descriptor = _load_json(
        descriptor_path,
        error_code="benchmark_descriptor_invalid",
    )
    sample_ids, records, rules = _validate_descriptor(
        descriptor,
        sample=sample,
    )
    manifest = _regular_relative_file(rendered_root, "manifest.json")
    expected_manifest_hash = cast(str, descriptor["dataset_manifest_sha256"])
    if _sha256(manifest) != expected_manifest_hash:
        raise OcrBenchmarkError("benchmark_manifest_hash_mismatch")
    manifest_value = _load_json(
        manifest,
        error_code="benchmark_manifest_invalid",
    )
    if manifest_value.get("schema_version") != "trade-document-render-manifest/v2":
        raise OcrBenchmarkError("benchmark_manifest_invalid")
    manifest_files = _manifest_generated_files(manifest_value)
    for record in records:
        image_manifest = manifest_files.get(record["image_relative_path"])
        layout_manifest = manifest_files.get(record["layout_relative_path"])
        if (
            image_manifest is None
            or layout_manifest is None
            or image_manifest["sha256"] != record["image_sha256"]
            or layout_manifest["sha256"] != record["layout_sha256"]
        ):
            raise OcrBenchmarkError("benchmark_manifest_document_mismatch")

    by_id = {record["benchmark_document_id"]: record for record in records}
    loaded: list[BenchmarkDocument] = []
    for document_id in sample_ids:
        record = by_id[document_id]
        image_path = _regular_relative_file(
            rendered_root,
            record["image_relative_path"],
        )
        layout_path = _regular_relative_file(
            rendered_root,
            record["layout_relative_path"],
        )
        image_manifest = manifest_files[record["image_relative_path"]]
        layout_manifest = manifest_files[record["layout_relative_path"]]
        if (
            _sha256(image_path) != record["image_sha256"]
            or _sha256(layout_path) != record["layout_sha256"]
            or _file_size(image_path) != image_manifest["byte_size"]
            or _file_size(layout_path) != layout_manifest["byte_size"]
        ):
            raise OcrBenchmarkError("benchmark_document_hash_mismatch")
        layout = _load_json(
            layout_path,
            error_code="benchmark_layout_invalid",
        )
        files = layout.get("files")
        expected_case_id, expected_document_type, expected_document_id = _expected_layout_identity(
            document_id
        )
        if (
            layout.get("schema_version") != "trade-document-layout-gold/v2"
            or layout.get("document_id") != expected_document_id
            or layout.get("case_id") != expected_case_id
            or layout.get("document_type") != expected_document_type
            or not isinstance(files, dict)
            or files.get("claude_input_png_relative_path") != record["image_relative_path"]
            or files.get("claude_input_png_sha256") != record["image_sha256"]
        ):
            raise OcrBenchmarkError("benchmark_layout_invalid")
        try:
            page = image_page_from_file(image_path)
        except Exception as exc:
            raise OcrBenchmarkError("benchmark_image_invalid") from exc
        page_contract = layout.get("page")
        if (
            page.sha256 != record["image_sha256"]
            or not isinstance(page_contract, dict)
            or page.width_px != page_contract.get("claude_input_width_px")
            or page.height_px != page_contract.get("claude_input_height_px")
        ):
            raise OcrBenchmarkError("benchmark_image_invalid")
        # Validate all scored Gold before either provider is started.
        _gold_words(layout)
        _gold_lines(layout)
        _gold_table_units(layout)
        _critical_values(layout, rules)
        loaded.append(
            BenchmarkDocument(
                benchmark_document_id=document_id,
                image_sha256=record["image_sha256"],
                layout_sha256=record["layout_sha256"],
                image_path=image_path,
                layout=layout,
            )
        )
    return (
        descriptor_hash,
        expected_manifest_hash,
        loaded,
        rules,
    )


def _normalize_text(value: object) -> str:
    return " ".join(unicodedata.normalize("NFKC", str(value or "")).split())


def _bbox(value: object) -> tuple[float, float, float, float] | None:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or not all(
            isinstance(coordinate, (int, float))
            and not isinstance(coordinate, bool)
            and math.isfinite(float(coordinate))
            for coordinate in value
        )
    ):
        return None
    x1, y1, x2, y2 = (float(coordinate) for coordinate in value)
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


def _span_bbox(span: OcrLayoutSpan) -> tuple[float, float, float, float]:
    return (
        span.bbox.x1,
        span.bbox.y1,
        span.bbox.x2,
        span.bbox.y2,
    )


def _intersection_over_union(
    left: Sequence[float],
    right: Sequence[float],
) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    left_area = (left[2] - left[0]) * (left[3] - left[1])
    right_area = (right[2] - right[0]) * (right[3] - right[1])
    return intersection / (left_area + right_area - intersection)


def _coverage(
    inner: Sequence[float],
    outer: Sequence[float],
) -> float:
    x1 = max(inner[0], outer[0])
    y1 = max(inner[1], outer[1])
    x2 = min(inner[2], outer[2])
    y2 = min(inner[3], outer[3])
    if x2 <= x1 or y2 <= y1:
        return 0.0
    intersection = (x2 - x1) * (y2 - y1)
    area = (inner[2] - inner[0]) * (inner[3] - inner[1])
    return intersection / area


def _union(boxes: Sequence[Sequence[float]]) -> tuple[float, float, float, float]:
    return (
        min(box[0] for box in boxes),
        min(box[1] for box in boxes),
        max(box[2] for box in boxes),
        max(box[3] for box in boxes),
    )


def _gold_words(layout: Mapping[str, Any]) -> list[dict[str, Any]]:
    words = layout.get("words")
    if not isinstance(words, list) or not words:
        raise OcrBenchmarkError("benchmark_layout_invalid")
    result: list[dict[str, Any]] = []
    for index, word in enumerate(words):
        if not isinstance(word, dict):
            raise OcrBenchmarkError("benchmark_layout_invalid")
        text = _normalize_text(word.get("text"))
        box = _bbox(word.get("bbox_claude_input_px"))
        role = word.get("role")
        if not text or box is None or not isinstance(role, str):
            raise OcrBenchmarkError("benchmark_layout_invalid")
        result.append(
            {
                "source_index": index,
                "text": text,
                "bbox": box,
                "role": role,
            }
        )
    return sorted(
        result,
        key=lambda item: (
            round(item["bbox"][1], 3),
            round(item["bbox"][0], 3),
            item["source_index"],
        ),
    )


def _gold_lines(layout: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Group exact rendered words into spatial lines comparable to OCR boxes."""

    words = _gold_words(layout)
    rows: list[list[dict[str, Any]]] = []
    for word in words:
        box = word["bbox"]
        center = (box[1] + box[3]) / 2
        height = box[3] - box[1]
        best_row: list[dict[str, Any]] | None = None
        best_delta = math.inf
        for row in rows:
            row_center = sum((item["bbox"][1] + item["bbox"][3]) / 2 for item in row) / len(row)
            row_height = sum(item["bbox"][3] - item["bbox"][1] for item in row) / len(row)
            delta = abs(center - row_center)
            if delta <= max(height, row_height) * 0.35 and delta < best_delta:
                best_row = row
                best_delta = delta
        if best_row is None:
            rows.append([word])
        else:
            best_row.append(word)

    lines: list[dict[str, Any]] = []
    for row in rows:
        current: list[dict[str, Any]] = []
        for word in sorted(row, key=lambda item: item["bbox"][0]):
            if current:
                previous = current[-1]
                previous_box = previous["bbox"]
                word_height = word["bbox"][3] - word["bbox"][1]
                previous_height = previous_box[3] - previous_box[1]
                gap = word["bbox"][0] - previous_box[2]
                if word["role"] != previous["role"] or gap > max(word_height, previous_height) * 2:
                    lines.append(_line_from_words(current))
                    current = []
            current.append(word)
        if current:
            lines.append(_line_from_words(current))
    return sorted(
        lines,
        key=lambda item: (
            round(item["bbox"][1], 3),
            round(item["bbox"][0], 3),
        ),
    )


def _line_from_words(words: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "text": " ".join(cast(str, word["text"]) for word in words),
        "bbox": _union([cast(Sequence[float], word["bbox"]) for word in words]),
    }


def _prediction_leaves(ledger: OcrLayoutLedger) -> list[dict[str, Any]]:
    base = [
        {
            "text": _normalize_text(span.text),
            "bbox": _span_bbox(span),
            "reading_order": span.reading_order,
        }
        for page in ledger.pages
        for span in page.spans
        if span.role in LEAF_ROLES
    ]
    cells = [span for page in ledger.pages for span in page.spans if span.role == "table_cell"]
    # Some engines emit table text both as ordinary OCR boxes and structured
    # cells. Add a cell only when no ordinary OCR box is spatially inside it.
    for cell in cells:
        cell_box = _span_bbox(cell)
        if any(
            max(
                _coverage(cast(Sequence[float], item["bbox"]), cell_box),
                _coverage(cell_box, cast(Sequence[float], item["bbox"])),
            )
            >= 0.5
            for item in base
        ):
            continue
        base.append(
            {
                "text": _normalize_text(cell.text),
                "bbox": cell_box,
                "reading_order": cell.reading_order,
            }
        )
    return sorted(
        (item for item in base if item["text"]),
        key=lambda item: (
            round(item["bbox"][1], 3),
            round(item["bbox"][0], 3),
            item["reading_order"],
        ),
    )


def _gold_text(layout: Mapping[str, Any]) -> str:
    return _normalize_text(" ".join(word["text"] for word in _gold_words(layout)))


def _predicted_text(ledger: OcrLayoutLedger) -> str:
    return _normalize_text(" ".join(item["text"] for item in _prediction_leaves(ledger)))


def _levenshtein(left: str, right: str) -> int:
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, start=1):
        current = [left_index]
        for right_index, right_character in enumerate(right, start=1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + int(left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def _critical_values(
    layout: Mapping[str, Any],
    rules: Sequence[Mapping[str, str]],
) -> list[str]:
    fields = layout.get("fields")
    if not isinstance(fields, list):
        raise OcrBenchmarkError("benchmark_layout_invalid")
    values: list[str] = []
    for field in fields:
        if not isinstance(field, dict):
            raise OcrBenchmarkError("benchmark_layout_invalid")
        field_id = field.get("field_id")
        raw_value = field.get("raw_value")
        if not isinstance(field_id, str) or not isinstance(raw_value, str):
            raise OcrBenchmarkError("benchmark_layout_invalid")
        for rule in rules:
            if rule["field_id_marker"] not in field_id:
                continue
            value = raw_value
            if rule["selector"] == "before_slash":
                value, separator, _ = value.partition(" / ")
                if not separator:
                    raise OcrBenchmarkError("benchmark_critical_gold_invalid")
            elif rule["selector"] == "before_double_slash":
                value, separator, _ = value.partition(" // ")
                if not separator:
                    raise OcrBenchmarkError("benchmark_critical_gold_invalid")
            normalized = _normalize_text(value)
            if not normalized:
                raise OcrBenchmarkError("benchmark_critical_gold_invalid")
            values.append(normalized)
    return values


def _contains_exact_identifier(text: str, identifier: str) -> bool:
    pattern = re.compile(rf"(?<![0-9A-Za-z]){re.escape(identifier)}(?![0-9A-Za-z])")
    return pattern.search(text) is not None


def _gold_table_units(
    layout: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    fields = layout.get("fields")
    tables = layout.get("tables")
    if not isinstance(fields, list) or not isinstance(tables, list):
        raise OcrBenchmarkError("benchmark_layout_invalid")
    field_values = {
        field["field_id"]: _normalize_text(field["raw_value"])
        for field in fields
        if isinstance(field, dict)
        and isinstance(field.get("field_id"), str)
        and isinstance(field.get("raw_value"), str)
    }
    rows_result: list[dict[str, Any]] = []
    cells_result: list[dict[str, Any]] = []
    for table in tables:
        if not isinstance(table, dict) or not isinstance(table.get("rows"), list):
            raise OcrBenchmarkError("benchmark_layout_invalid")
        for row in table["rows"]:
            if not isinstance(row, dict) or not isinstance(row.get("cells"), list):
                raise OcrBenchmarkError("benchmark_layout_invalid")
            row_box = _bbox(row.get("bbox_claude_input_px"))
            if row_box is None:
                raise OcrBenchmarkError("benchmark_layout_invalid")
            row_texts: list[str] = []
            for cell in row["cells"]:
                if not isinstance(cell, dict):
                    raise OcrBenchmarkError("benchmark_layout_invalid")
                cell_box = _bbox(cell.get("bbox_claude_input_px"))
                field_id = cell.get("field_id")
                if field_id is None:
                    text = _normalize_text(cell.get("column_label"))
                elif isinstance(field_id, str) and field_id in field_values:
                    text = field_values[field_id]
                else:
                    raise OcrBenchmarkError("benchmark_layout_invalid")
                if cell_box is None or not text:
                    raise OcrBenchmarkError("benchmark_layout_invalid")
                row_texts.append(text)
                cells_result.append({"text": text, "bbox": cell_box})
            rows_result.append(
                {
                    "text": _normalize_text(" | ".join(row_texts)),
                    "bbox": row_box,
                }
            )
    return rows_result, cells_result


def _predicted_table_units(
    ledger: OcrLayoutLedger,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    cells: list[dict[str, Any]] = []
    for page in ledger.pages:
        for span in page.spans:
            unit = {
                "text": _normalize_text(span.text),
                "bbox": _span_bbox(span),
            }
            if span.role == "table_row":
                rows.append(unit)
            elif span.role == "table_cell":
                cells.append(unit)
    return rows, cells


def _match_units(
    gold: Sequence[Mapping[str, Any]],
    predicted: Sequence[Mapping[str, Any]],
    *,
    require_text: bool,
) -> MatchCounts:
    candidates: list[tuple[float, int, int]] = []
    for gold_index, gold_unit in enumerate(gold):
        for predicted_index, predicted_unit in enumerate(predicted):
            if require_text and gold_unit["text"] != predicted_unit["text"]:
                continue
            overlap = _intersection_over_union(
                cast(Sequence[float], gold_unit["bbox"]),
                cast(Sequence[float], predicted_unit["bbox"]),
            )
            if overlap >= 0.5:
                candidates.append((-overlap, gold_index, predicted_index))
    matched_gold: set[int] = set()
    matched_predicted: set[int] = set()
    for _, gold_index, predicted_index in sorted(candidates):
        if gold_index in matched_gold or predicted_index in matched_predicted:
            continue
        matched_gold.add(gold_index)
        matched_predicted.add(predicted_index)
    true_positive = len(matched_gold)
    return MatchCounts(
        true_positive=true_positive,
        false_positive=len(predicted) - true_positive,
        false_negative=len(gold) - true_positive,
    )


def score_document(
    *,
    layout: Mapping[str, Any],
    ledger: OcrLayoutLedger,
    critical_rules: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    gold_text = _gold_text(layout)
    predicted_text = _predicted_text(ledger)
    critical_values = _critical_values(layout, critical_rules)
    critical_correct = sum(
        _contains_exact_identifier(predicted_text, value) for value in critical_values
    )
    gold_rows, gold_cells = _gold_table_units(layout)
    predicted_rows, predicted_cells = _predicted_table_units(ledger)
    gold_lines = _gold_lines(layout)
    predicted_leaves = _prediction_leaves(ledger)
    edit_distance = _levenshtein(gold_text, predicted_text)
    return {
        "critical_identifier_correct": critical_correct,
        "critical_identifier_total": len(critical_values),
        "character_edit_distance": edit_distance,
        "gold_character_count": len(gold_text),
        "table_rows": _match_units(
            gold_rows,
            predicted_rows,
            require_text=True,
        ),
        "table_cells": _match_units(
            gold_cells,
            predicted_cells,
            require_text=True,
        ),
        "bboxes": _match_units(
            gold_lines,
            predicted_leaves,
            require_text=False,
        ),
    }


def gold_document_metric_denominators(
    *,
    layout: Mapping[str, Any],
    critical_rules: Sequence[Mapping[str, str]],
) -> dict[str, int]:
    """Return non-sensitive Gold counts used to attest per-document metrics."""

    gold_text = _gold_text(layout)
    critical_values = _critical_values(layout, critical_rules)
    gold_rows, gold_cells = _gold_table_units(layout)
    gold_lines = _gold_lines(layout)
    return {
        "critical_identifier_total": len(critical_values),
        "gold_character_count": len(gold_text),
        "table_row_gold_count": len(gold_rows),
        "table_cell_gold_count": len(gold_cells),
        "bbox_gold_count": len(gold_lines),
    }


def _add_missing_document(
    totals: ProviderTotals,
    *,
    layout: Mapping[str, Any],
    critical_rules: Sequence[Mapping[str, str]],
) -> dict[str, Any]:
    """Score a failed document as an empty prediction, never as skipped."""

    gold_text = _gold_text(layout)
    critical_values = _critical_values(layout, critical_rules)
    gold_rows, gold_cells = _gold_table_units(layout)
    gold_lines = _gold_lines(layout)
    document_score = {
        "critical_identifier_correct": 0,
        "critical_identifier_total": len(critical_values),
        "character_edit_distance": len(gold_text),
        "gold_character_count": len(gold_text),
        "table_rows": MatchCounts(false_negative=len(gold_rows)),
        "table_cells": MatchCounts(false_negative=len(gold_cells)),
        "bboxes": MatchCounts(false_negative=len(gold_lines)),
    }
    totals.critical_total += len(critical_values)
    totals.edit_distance += len(gold_text)
    totals.gold_characters += len(gold_text)
    cast(MatchCounts, totals.table_rows).false_negative += len(gold_rows)
    cast(MatchCounts, totals.table_cells).false_negative += len(gold_cells)
    cast(MatchCounts, totals.bboxes).false_negative += len(gold_lines)
    return document_score


def _document_score_report(score: Mapping[str, Any]) -> dict[str, Any]:
    """Serialize additive counts so a release gate can recompute every aggregate."""

    return {
        "critical_identifier_correct": score["critical_identifier_correct"],
        "critical_identifier_total": score["critical_identifier_total"],
        "character_edit_distance": score["character_edit_distance"],
        "gold_character_count": score["gold_character_count"],
        "table_rows": cast(MatchCounts, score["table_rows"]).report(),
        "table_cells": cast(MatchCounts, score["table_cells"]).report(),
        "bboxes_iou_0_5": cast(MatchCounts, score["bboxes"]).report(),
    }


def installed_agent_wheel_sha256(executable: Path) -> str:
    """Map the shared RECORD-bound runtime attestation to benchmark-safe codes."""

    try:
        return _release_installed_agent_wheel_sha256(executable)
    except OcrReleaseGateAttestationError as exc:
        release_prefix = "ocr_release_gate_"
        detail = exc.code.removeprefix(release_prefix)
        preserved = {
            "agent_distribution_unavailable",
            "agent_direct_url_invalid",
            "agent_record_invalid",
            "agent_record_mismatch",
            "agent_source_install_rejected",
            "agent_untracked_executable",
            "agent_wheel_hash_unavailable",
        }
        code = (
            f"benchmark_{detail}" if detail in preserved else "benchmark_wheel_attestation_failed"
        )
        raise OcrBenchmarkError(code) from exc


# Backward-compatible private alias for callers that imported the original
# helper while the benchmark contract was still being finalized.
_installed_agent_wheel_sha256 = installed_agent_wheel_sha256


def _default_provider_factory(
    provider: ProviderName,
    executable: Path,
) -> OcrLayoutProvider:
    return build_ocr_layout_provider(
        provider,
        python_executable=executable,
    )


def _safe_provider_code(error: OcrLayoutProviderError) -> str:
    return error.code if SAFE_CODE.fullmatch(error.code) is not None else "ocr_provider_error"


def _provider_report(
    *,
    provider_name: ProviderName,
    executable: Path,
    documents: Sequence[BenchmarkDocument],
    critical_rules: Sequence[Mapping[str, str]],
    provider_factory: ProviderFactory,
    runtime_attestor: RuntimeAttestor,
    model_assets_lock_path: Path,
) -> dict[str, Any]:
    totals = ProviderTotals()
    document_reports: list[dict[str, Any]] = []
    latencies_ns: list[int] = []
    engine: tuple[str, str, str] | None = None
    provider: OcrLayoutProvider | None = None
    provider_error_code: str | None = None
    cleanup_error_code: str | None = None
    runtime_error_code: str | None = None
    runtime_metrics: dict[str, int | float] | None = None
    try:
        provider = provider_factory(provider_name, executable)
    except OcrLayoutProviderError as exc:
        provider_error_code = _safe_provider_code(exc)
    except Exception:
        provider_error_code = "ocr_provider_initialization_failed"

    if provider is None:
        totals.crash_count = len(documents)
        for document in documents:
            document_score = _add_missing_document(
                totals,
                layout=document.layout,
                critical_rules=critical_rules,
            )
            document_reports.append(
                {
                    "benchmark_document_id": document.benchmark_document_id,
                    "input_sha256": document.image_sha256,
                    "layout_sha256": document.layout_sha256,
                    "status": "FAIL",
                    "error_code": provider_error_code,
                    "latency_ns": 0,
                    "metrics": _document_score_report(document_score),
                }
            )
    else:
        try:
            for document in documents:
                started_ns = time.perf_counter_ns()
                try:
                    page = image_page_from_file(document.image_path)
                    if page.sha256 != document.image_sha256:
                        raise OcrBenchmarkError("benchmark_document_hash_mismatch")
                    ledger = provider.analyze(
                        document_id=cast(str, document.layout["document_id"]),
                        pages=[page],
                    )
                    if (
                        ledger.model_id != OCR_MODEL_SET_IDS[provider_name]
                        or ledger.model_assets_lock_sha256 != CANONICAL_MODEL_ASSETS_LOCK_SHA256
                    ):
                        raise OcrLayoutProviderError(
                            "ocr_model_identity_mismatch",
                            "OCR model identity differs from the canonical release lock.",
                        )
                    current_engine = (
                        ledger.provider_id,
                        ledger.engine_id,
                        ledger.engine_version,
                    )
                    if engine is None:
                        engine = current_engine
                    elif engine != current_engine:
                        raise OcrLayoutProviderError(
                            "ocr_engine_drift",
                            "OCR engine identity changed within one benchmark.",
                        )
                    document_score = score_document(
                        layout=document.layout,
                        ledger=ledger,
                        critical_rules=critical_rules,
                    )
                except OcrLayoutProviderError as exc:
                    elapsed_ns = max(1, time.perf_counter_ns() - started_ns)
                    latencies_ns.append(elapsed_ns)
                    totals.crash_count += 1
                    document_score = _add_missing_document(
                        totals,
                        layout=document.layout,
                        critical_rules=critical_rules,
                    )
                    document_reports.append(
                        {
                            "benchmark_document_id": document.benchmark_document_id,
                            "input_sha256": document.image_sha256,
                            "layout_sha256": document.layout_sha256,
                            "status": "FAIL",
                            "error_code": _safe_provider_code(exc),
                            "latency_ns": elapsed_ns,
                            "metrics": _document_score_report(document_score),
                        }
                    )
                    continue
                except Exception:
                    elapsed_ns = max(1, time.perf_counter_ns() - started_ns)
                    latencies_ns.append(elapsed_ns)
                    totals.crash_count += 1
                    document_score = _add_missing_document(
                        totals,
                        layout=document.layout,
                        critical_rules=critical_rules,
                    )
                    document_reports.append(
                        {
                            "benchmark_document_id": document.benchmark_document_id,
                            "input_sha256": document.image_sha256,
                            "layout_sha256": document.layout_sha256,
                            "status": "FAIL",
                            "error_code": "ocr_benchmark_scoring_failed",
                            "latency_ns": elapsed_ns,
                            "metrics": _document_score_report(document_score),
                        }
                    )
                    continue
                elapsed_ns = max(1, time.perf_counter_ns() - started_ns)
                latencies_ns.append(elapsed_ns)
                totals.successful_document_count += 1
                totals.critical_correct += document_score["critical_identifier_correct"]
                totals.critical_total += document_score["critical_identifier_total"]
                totals.edit_distance += document_score["character_edit_distance"]
                totals.gold_characters += document_score["gold_character_count"]
                cast(MatchCounts, totals.table_rows).add(document_score["table_rows"])
                cast(MatchCounts, totals.table_cells).add(document_score["table_cells"])
                cast(MatchCounts, totals.bboxes).add(document_score["bboxes"])
                document_reports.append(
                    {
                        "benchmark_document_id": document.benchmark_document_id,
                        "input_sha256": document.image_sha256,
                        "layout_sha256": document.layout_sha256,
                        "status": "SCORED",
                        "latency_ns": elapsed_ns,
                        "metrics": _document_score_report(document_score),
                    }
                )
            if totals.successful_document_count == len(documents) and engine is not None:
                try:
                    peak_memory_bytes, disk_footprint_bytes = runtime_attestor(
                        provider_name,
                        executable,
                        provider,
                        model_assets_lock_path,
                    )
                    if (
                        not isinstance(peak_memory_bytes, int)
                        or isinstance(peak_memory_bytes, bool)
                        or peak_memory_bytes < 1
                        or not isinstance(disk_footprint_bytes, int)
                        or isinstance(disk_footprint_bytes, bool)
                        or disk_footprint_bytes < 1
                    ):
                        raise OcrBenchmarkError("benchmark_runtime_metrics_invalid")
                    runtime_metrics = {
                        "latency_sample_count": len(latencies_ns),
                        "latency_p95_ms": _p95_latency_ms(latencies_ns),
                        "peak_memory_bytes": peak_memory_bytes,
                        "disk_footprint_bytes": disk_footprint_bytes,
                    }
                except OcrBenchmarkError as exc:
                    runtime_error_code = exc.code
                except Exception:
                    runtime_error_code = "benchmark_runtime_attestation_failed"
        finally:
            close = getattr(provider, "close", None)
            if callable(close):
                try:
                    close()
                except Exception:
                    cleanup_error_code = "ocr_provider_cleanup_failed"
                    totals.crash_count += 1

    critical_accuracy = (
        totals.critical_correct / totals.critical_total if totals.critical_total else 0.0
    )
    character_error_rate = (
        totals.edit_distance / totals.gold_characters if totals.gold_characters else 1.0
    )
    table_rows = cast(MatchCounts, totals.table_rows)
    table_cells = cast(MatchCounts, totals.table_cells)
    bboxes = cast(MatchCounts, totals.bboxes)
    metrics = {
        "critical_identifier_exact_accuracy": critical_accuracy,
        "critical_identifier_correct": totals.critical_correct,
        "critical_identifier_total": totals.critical_total,
        "character_error_rate": character_error_rate,
        "character_edit_distance": totals.edit_distance,
        "gold_character_count": totals.gold_characters,
        "table_rows": table_rows.report(),
        "table_cells": table_cells.report(),
        "bboxes_iou_0_5": bboxes.report(),
        "crash_count": totals.crash_count,
        "successful_document_count": totals.successful_document_count,
        "document_count": len(documents),
    }
    checks = {
        "critical_identifier_exact": (
            critical_accuracy >= LOCKED_THRESHOLDS["critical_identifier_exact_accuracy_min"]
        ),
        "character_error_rate": (
            character_error_rate <= LOCKED_THRESHOLDS["character_error_rate_max"]
        ),
        "table_row_f1": (table_rows.f1 >= LOCKED_THRESHOLDS["table_row_f1_min"]),
        "table_cell_f1": (table_cells.f1 >= LOCKED_THRESHOLDS["table_cell_f1_min"]),
        "bbox_f1_iou_0_5": (bboxes.f1 >= LOCKED_THRESHOLDS["bbox_f1_iou_0_5_min"]),
        "crash_count": (totals.crash_count <= LOCKED_THRESHOLDS["crash_count_max"]),
        "all_documents_scored": (totals.successful_document_count == len(documents)),
    }
    result: dict[str, Any] = {
        "provider": provider_name,
        "status": "PASS" if all(checks.values()) else "FAIL",
        "engine": (
            {
                "provider_id": engine[0],
                "engine_id": engine[1],
                "engine_version": engine[2],
            }
            if engine is not None
            else None
        ),
        "metrics": metrics,
        "runtime_metrics": runtime_metrics,
        "checks": checks,
        "failure_codes": sorted(check for check, passed in checks.items() if not passed),
        "documents": document_reports,
        "cleanup_error_code": cleanup_error_code,
        "runtime_error_code": runtime_error_code,
    }
    return result


def _selection(
    provider_reports: Sequence[Mapping[str, Any]],
) -> tuple[str, str | None]:
    passing = [report for report in provider_reports if report["status"] == "PASS"]
    if not passing:
        return "NO_PROVIDER_PASSED", None
    if len(passing) == 1:
        return "SELECTED_ONLY_PASSING_PROVIDER", cast(str, passing[0]["provider"])

    def quality(report: Mapping[str, Any]) -> tuple[float, float, float, float]:
        metrics = cast(Mapping[str, Any], report["metrics"])
        rows = cast(Mapping[str, Any], metrics["table_rows"])
        cells = cast(Mapping[str, Any], metrics["table_cells"])
        bboxes = cast(Mapping[str, Any], metrics["bboxes_iou_0_5"])
        return (
            float(metrics["critical_identifier_exact_accuracy"]),
            -float(metrics["character_error_rate"]),
            min(float(rows["f1"]), float(cells["f1"])),
            float(bboxes["f1"]),
        )

    first, second = passing
    first_quality = quality(first)
    second_quality = quality(second)
    if first_quality == second_quality:
        return "QUALITY_TIE_NO_AUTOMATIC_SELECTION", None
    selected = first if first_quality > second_quality else second
    return "SELECTED_BY_LOCKED_QUALITY_METRICS", cast(str, selected["provider"])


def evaluate(
    *,
    descriptor_path: Path,
    rendered_root: Path,
    sample: SampleName,
    provider_executables: Mapping[ProviderName, Path],
    wheel_sha256: str,
    provider_environment_locks: Mapping[ProviderName, Path],
    provider_environment_lock_sha256: Mapping[ProviderName, str],
    model_assets_lock_path: Path,
    model_assets_lock_sha256: str,
    repo_root: Path,
    authorization_receipt_path: Path,
    provider_factory: ProviderFactory = _default_provider_factory,
    wheel_attestor: WheelAttestor = installed_agent_wheel_sha256,
    environment_attestor: EnvironmentAttestor = installed_ocr_environment_lock_sha256,
    model_assets_attestor: ModelAssetsAttestor = verify_ocr_model_assets,
    runtime_attestor: RuntimeAttestor = attest_runtime_footprint,
    source_freeze_verifier: SourceFreezeVerifier = verify_ocr_release_freeze,
) -> tuple[int, dict[str, Any]]:
    try:
        source_freeze_verifier(
            repo_root=repo_root,
            descriptor_path=descriptor_path,
            authorization_receipt_path=authorization_receipt_path,
        )
    except OcrFreezeVerificationError as exc:
        detail = exc.code.removeprefix("ocr_release_freeze_")
        raise OcrBenchmarkError(f"benchmark_release_freeze_{detail}") from exc
    except Exception as exc:
        raise OcrBenchmarkError("benchmark_release_freeze_verification_failed") from exc
    if SHA256.fullmatch(wheel_sha256) is None:
        raise OcrBenchmarkError("benchmark_wheel_hash_invalid")
    if (
        set(provider_executables) != set(PROVIDERS)
        or set(provider_environment_locks) != set(PROVIDERS)
        or set(provider_environment_lock_sha256) != set(PROVIDERS)
    ):
        raise OcrBenchmarkError("benchmark_provider_set_invalid")
    if dict(provider_environment_lock_sha256) != CANONICAL_ENVIRONMENT_LOCK_SHA256:
        raise OcrBenchmarkError("benchmark_environment_lock_hash_invalid")
    if model_assets_lock_sha256 != CANONICAL_MODEL_ASSETS_LOCK_SHA256:
        raise OcrBenchmarkError("benchmark_model_assets_lock_hash_invalid")

    installed_wheel_hashes: dict[ProviderName, str] = {}
    installed_environment_lock_hashes: dict[ProviderName, str] = {}
    installed_model_assets_lock_hashes: dict[ProviderName, str] = {}
    for provider in PROVIDERS:
        try:
            installed_sha256 = wheel_attestor(provider_executables[provider])
        except OcrBenchmarkError:
            raise
        except Exception as exc:
            raise OcrBenchmarkError("benchmark_wheel_attestation_failed") from exc
        if (
            not isinstance(installed_sha256, str)
            or SHA256.fullmatch(installed_sha256) is None
            or installed_sha256 != wheel_sha256
        ):
            raise OcrBenchmarkError("benchmark_wheel_hash_mismatch")
        installed_wheel_hashes[provider] = installed_sha256
        try:
            environment_sha256 = environment_attestor(
                provider_executables[provider],
                provider_environment_locks[provider],
            )
        except OcrReleaseGateAttestationError as exc:
            raise OcrBenchmarkError("benchmark_environment_attestation_failed") from exc
        except Exception as exc:
            raise OcrBenchmarkError("benchmark_environment_attestation_failed") from exc
        if (
            not isinstance(environment_sha256, str)
            or SHA256.fullmatch(environment_sha256) is None
            or environment_sha256 != provider_environment_lock_sha256[provider]
        ):
            raise OcrBenchmarkError("benchmark_environment_lock_hash_mismatch")
        installed_environment_lock_hashes[provider] = environment_sha256
        try:
            observed_model_assets_sha256 = model_assets_attestor(
                provider,
                model_assets_lock_path,
                PROVIDER_ENGINE_IDS[provider],
            )
        except OcrReleaseGateAttestationError as exc:
            raise OcrBenchmarkError("benchmark_model_assets_attestation_failed") from exc
        except Exception as exc:
            raise OcrBenchmarkError("benchmark_model_assets_attestation_failed") from exc
        if (
            not isinstance(observed_model_assets_sha256, str)
            or SHA256.fullmatch(observed_model_assets_sha256) is None
            or observed_model_assets_sha256 != model_assets_lock_sha256
        ):
            raise OcrBenchmarkError("benchmark_model_assets_lock_hash_mismatch")
        installed_model_assets_lock_hashes[provider] = observed_model_assets_sha256

    descriptor_hash, manifest_hash, documents, critical_rules = load_benchmark(
        descriptor_path=descriptor_path,
        rendered_root=rendered_root,
        sample=sample,
    )
    provider_reports: list[dict[str, Any]] = []
    for provider in PROVIDERS:
        provider_report = _provider_report(
            provider_name=provider,
            executable=provider_executables[provider],
            documents=documents,
            critical_rules=critical_rules,
            provider_factory=provider_factory,
            runtime_attestor=runtime_attestor,
            model_assets_lock_path=model_assets_lock_path,
        )
        provider_report["installed_agent_wheel_sha256"] = installed_wheel_hashes[provider]
        provider_report["environment_lock_sha256"] = installed_environment_lock_hashes[provider]
        provider_report["model_assets_lock_sha256"] = installed_model_assets_lock_hashes[provider]
        engine = provider_report.get("engine")
        if engine is not None and (
            not isinstance(engine, dict)
            or engine.get("provider_id") != provider
            or engine.get("engine_id") != PROVIDER_ENGINE_IDS[provider]
        ):
            raise OcrBenchmarkError("benchmark_engine_identity_mismatch")
        provider_reports.append(provider_report)
    selection_status, selected_provider = _selection(provider_reports)
    report = {
        "schema_version": REPORT_SCHEMA,
        "benchmark_id": BENCHMARK_ID,
        "sample": sample,
        "status": "PASS" if selected_provider is not None else "FAIL",
        "selection_status": selection_status,
        "selected_provider": selected_provider,
        "descriptor_sha256": descriptor_hash,
        "dataset_manifest_sha256": manifest_hash,
        "thresholds": LOCKED_THRESHOLDS,
        "document_count": len(documents),
        "coverage": {
            "selected_sample": sample,
            "selected_document_count": len(documents),
            "locked_case_count": 20,
            "locked_triplet_document_count": 60,
            "co_document_count": 20,
            "release_gate_document_count": 80,
        },
        "providers": provider_reports,
        "redaction": {
            "source_text": "OMITTED",
            "image_payloads": "OMITTED",
            "filesystem_paths": "OMITTED",
            "secrets": "OMITTED",
            "backend_exception_messages": "OMITTED",
        },
    }
    return (0 if selected_provider is not None else 1), report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark PaddleOCR and Docling against locked rendered Gold."
    )
    parser.add_argument("--descriptor", required=True, type=Path)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--authorization-receipt", required=True, type=Path)
    parser.add_argument("--rendered-root", required=True, type=Path)
    parser.add_argument(
        "--sample",
        choices=("smoke", "locked", "co_regression"),
        required=True,
    )
    parser.add_argument("--paddle-python", required=True, type=Path)
    parser.add_argument("--docling-python", required=True, type=Path)
    parser.add_argument("--wheel-sha256", required=True)
    parser.add_argument("--paddle-environment-lock", required=True, type=Path)
    parser.add_argument("--paddle-environment-lock-sha256", required=True)
    parser.add_argument("--docling-environment-lock", required=True, type=Path)
    parser.add_argument("--docling-environment-lock-sha256", required=True)
    parser.add_argument("--model-assets-lock", required=True, type=Path)
    parser.add_argument("--model-assets-lock-sha256", required=True)
    parser.add_argument("--out", type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        exit_code, report = evaluate(
            descriptor_path=args.descriptor,
            repo_root=args.repo_root,
            authorization_receipt_path=args.authorization_receipt,
            rendered_root=args.rendered_root,
            sample=cast(SampleName, args.sample),
            provider_executables={
                "paddle": args.paddle_python,
                "docling": args.docling_python,
            },
            wheel_sha256=args.wheel_sha256.lower(),
            provider_environment_locks={
                "paddle": args.paddle_environment_lock,
                "docling": args.docling_environment_lock,
            },
            provider_environment_lock_sha256={
                "paddle": args.paddle_environment_lock_sha256.lower(),
                "docling": args.docling_environment_lock_sha256.lower(),
            },
            model_assets_lock_path=args.model_assets_lock,
            model_assets_lock_sha256=args.model_assets_lock_sha256.lower(),
        )
    except OcrBenchmarkError as exc:
        exit_code = 2
        report = {
            "schema_version": REPORT_SCHEMA,
            "status": "FAIL",
            "selection_status": "PREFLIGHT_FAILED",
            "selected_provider": None,
            "error_code": exc.code,
            "redaction": {
                "source_text": "OMITTED",
                "image_payloads": "OMITTED",
                "filesystem_paths": "OMITTED",
                "secrets": "OMITTED",
                "backend_exception_messages": "OMITTED",
            },
        }
    payload = (
        json.dumps(
            report,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    if args.out is not None:
        try:
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(payload, encoding="utf-8")
        except OSError:
            return 2
    print(payload, end="")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
