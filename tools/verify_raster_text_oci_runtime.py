"""Verify one minimal Linux Raster Text OCR deployment context.

The verifier intentionally uses only the Python standard library so it can run
both in a trusted exporter and in the final OCI build before application code
is copied.  It never downloads, repairs, or selects runtime material.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
import sys
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_PROFILE_REVISION = re.compile(r"^synthetic-raster-text-v1:[a-f0-9]{24}$")
_ENGINE_KIND = (
    "direct_paddle_mobile_det_public_finetuned_rec_harmonic_recall_native960_"
    "lowbox_strip2_semantic_repair_v2_onednn"
)
_METRIC_REVISION = "source-attested-semantic-text-v3"
_EVIDENCE_REVISION = "source-attested-public-finetuned-staged-train-holdout-v2"
_LATENCY_GATE_REVISION = "warm-p95-10s-v1"
_QUALITY_GATE_REVISION = "preview-draft-critical97-cer2-v1"
_EVIDENCE_GENERATION_REVISION = "current-runner-native-full-gate-v1"
_RUNTIME_VALIDATION_REVISION = "activation-full-runtime-compact-v1"
_POINTER_V12_FIELDS = frozenset(
    {
        "activation_kind",
        "current_runner_implementation_sha256",
        "dataset_id",
        "engine_kind",
        "evidence_generation_revision",
        "evidence_revision",
        "latency_gate_revision",
        "metric_revision",
        "quality_gate_revision",
        "raster_text_profile_revision",
        "runtime_configuration_path",
        "runtime_configuration_sha256",
        "runtime_validation_revision",
        "schema_version",
        "sealed_evaluation_path",
        "sealed_evaluation_sha256",
        "source_recognizer_parent_selection_receipt_sha256",
        "source_recognizer_receipt_sha256",
        "source_recognizer_release_sha256",
        "source_recognizer_selection_receipt_path",
        "source_recognizer_selection_receipt_sha256",
        "source_recognizer_training_intent_sha256",
        "status",
    }
)
_SEALED_V9_FIELDS = frozenset(
    {
        "artifact_kind",
        "asset_count",
        "calibration_receipt_path",
        "calibration_receipt_sha256",
        "candidate_execution_path",
        "candidate_execution_sha256",
        "current_runner_implementation_sha256",
        "dataset_id",
        "evidence_generation_revision",
        "evidence_revision",
        "extractor_revision",
        "family_count",
        "hard_timeout_seconds",
        "latency_gate_revision",
        "latency_p95_seconds",
        "metric_revision",
        "minimum_page_confidence",
        "projection_kind",
        "projection_receipt_path",
        "projection_receipt_sha256",
        "quality_gate_revision",
        "raster_text_profile_revision",
        "runtime_configuration_sha256",
        "schema_version",
        "scores",
        "selection_receipt_path",
        "selection_receipt_sha256",
        "service_warm_latency_p95_seconds",
        "source_gold_receipt_path",
        "source_gold_receipt_sha256",
        "source_recognizer_parent_selection_receipt_sha256",
        "source_recognizer_receipt_sha256",
        "source_recognizer_release_sha256",
        "source_recognizer_training_intent_sha256",
        "source_truth_kind",
        "split",
        "stage1_decision",
        "stage1_decision_receipt_path",
        "stage1_decision_receipt_sha256",
        "warm_service_smoke_receipt_path",
        "warm_service_smoke_receipt_sha256",
    }
)
_MANIFEST_RELATIVE_PATH = PurePosixPath("_deployment/raster-text-runtime-oci-v1.json")
_POINTER_RELATIVE_PATH = PurePosixPath("approvals/raster-text-active.json")
_ALLOWED_ROLES = frozenset(
    {
        "activation-evidence",
        "active-pointer",
        "inventory",
        "licenses",
        "model",
        "runtime-profile",
        "sbom",
        "sidecar-runtime",
    }
)
_REQUIRED_ROLE_COUNTS = frozenset(
    {
        "activation-evidence",
        "active-pointer",
        "inventory",
        "licenses",
        "model",
        "runtime-profile",
        "sbom",
        "sidecar-runtime",
    }
)
_FORBIDDEN_PATH_COMPONENTS = frozenset(
    {
        "candidate-outputs",
        "gold-review",
        "ocr-gold",
        "source-layout-truths",
        "synthetic-raster-build",
        "synthetic-raster-build-v2",
        "synthetic-raster-v1",
        "trade_documents_images",
    }
)
_ALWAYS_FORBIDDEN_DOCUMENT_SUFFIXES = frozenset(
    {".bmp", ".jpeg", ".jpg", ".jsonl", ".parquet", ".pdf", ".tif", ".tiff"}
)
_EVIDENCE_RESOURCE_SUFFIXES = frozenset({".csv", ".png"})
_RECEIPT_NAME_MARKERS = frozenset(
    {"calibration", "candidate", "evidence", "gold", "projection", "receipt", "sealed", "smoke"}
)
_MAX_MANIFEST_BYTES = 24 * 1024 * 1024
# A direct Paddle Linux bundle can be close to 20,000 files.  The 25% bounded
# headroom avoids rejecting a measured CPython/Paddle patch-level closure while
# retaining a fixed, reviewable ceiling.  Export output reports the exact count
# so this constant can be tightened after the release closure is measured.
_MAX_FILE_COUNT = 25_000
_MAX_DECLARED_FILE_COUNT = _MAX_FILE_COUNT - 1
_MAX_TREE_ENTRIES = 50_000
_MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
_MAX_EVIDENCE_FILES = 128
_MAX_EVIDENCE_BYTES = 64 * 1024 * 1024


class RuntimeContextError(RuntimeError):
    """A content-free deployment contract failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _canonical_pointer_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
                size += len(chunk)
    except OSError:
        raise RuntimeContextError("runtime_file_unreadable") from None
    return digest.hexdigest(), size


def _relative_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value or len(value) > 512 or "\\" in value:
        raise RuntimeContextError("runtime_manifest_path_invalid")
    relative = PurePosixPath(value)
    if relative.is_absolute() or relative == PurePosixPath(".") or ".." in relative.parts:
        raise RuntimeContextError("runtime_manifest_path_invalid")
    if any(part in {"", "."} for part in relative.parts):
        raise RuntimeContextError("runtime_manifest_path_invalid")
    return relative


def _is_beneath(relative: PurePosixPath, parent: PurePosixPath) -> bool:
    try:
        relative.relative_to(parent)
    except ValueError:
        return False
    return True


def _regular_child(root: Path, relative: PurePosixPath) -> Path:
    candidate = root.joinpath(*relative.parts)
    current = root
    try:
        if root.is_symlink() or not root.is_dir():
            raise OSError
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise OSError
        if not candidate.is_file():
            raise OSError
        resolved_root = root.resolve(strict=True)
        resolved_candidate = candidate.resolve(strict=True)
        if not resolved_candidate.is_relative_to(resolved_root):
            raise OSError
    except OSError:
        raise RuntimeContextError("runtime_file_invalid") from None
    return candidate


def _read_json(path: Path, *, maximum: int, code: str) -> tuple[dict[str, Any], bytes]:
    try:
        if path.stat().st_size > maximum:
            raise ValueError
        raw = path.read_bytes()
        value = json.loads(raw.decode("utf-8"), parse_constant=_reject_nonfinite_json)
        if not isinstance(value, dict):
            raise ValueError
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        raise RuntimeContextError(code) from None
    return value, raw


def _normalized_package_name(value: str) -> str:
    return re.sub(r"[-_.]+", "-", value).casefold()


def _validate_license_manifest(path: Path) -> tuple[set[tuple[str, str]], set[str]]:
    value, _raw = _read_json(path, maximum=4 * 1024 * 1024, code="license_manifest_invalid")
    if set(value) != {"components", "schema_version"} or value.get("schema_version") != "1.0":
        raise RuntimeContextError("license_manifest_invalid")
    components = value.get("components")
    if not isinstance(components, list) or not components:
        raise RuntimeContextError("license_manifest_invalid")
    observed: set[tuple[str, str]] = set()
    scopes: set[str] = set()
    for component in components:
        if not isinstance(component, dict) or set(component) != {
            "license_id",
            "name",
            "scope",
            "source_uri",
            "version",
        }:
            raise RuntimeContextError("license_manifest_invalid")
        fields = tuple(
            component.get(name) for name in ("name", "version", "license_id", "source_uri")
        )
        if not all(isinstance(item, str) and item.strip() for item in fields):
            raise RuntimeContextError("license_manifest_invalid")
        if component.get("scope") not in {"model", "python-package", "runtime"}:
            raise RuntimeContextError("license_manifest_invalid")
        scope = str(component["scope"])
        identity = (_normalized_package_name(str(component["name"])), str(component["version"]))
        if identity in observed:
            raise RuntimeContextError("license_manifest_invalid")
        observed.add(identity)
        scopes.add(scope)
    if scopes != {"model", "python-package", "runtime"}:
        raise RuntimeContextError("license_manifest_invalid")
    return observed, scopes


def _inventory_packages(path: Path) -> set[tuple[str, str]]:
    try:
        if path.suffix.casefold() != ".json" or path.stat().st_size > 4 * 1024 * 1024:
            return set()
        value = json.loads(path.read_text(encoding="utf-8"))
        packages = value.get("packages") if isinstance(value, dict) else None
        if packages is None:
            return set()
        if not isinstance(packages, list):
            raise ValueError
        result: set[tuple[str, str]] = set()
        for package in packages:
            if (
                not isinstance(package, dict)
                or not isinstance(package.get("name"), str)
                or not isinstance(package.get("version"), str)
                or not package["name"]
                or not package["version"]
            ):
                raise ValueError
            result.add((_normalized_package_name(package["name"]), package["version"]))
        return result
    except (OSError, UnicodeDecodeError, ValueError, json.JSONDecodeError):
        raise RuntimeContextError("runtime_package_inventory_invalid") from None


def _validate_spdx(path: Path) -> None:
    value, _raw = _read_json(path, maximum=16 * 1024 * 1024, code="runtime_sbom_invalid")
    if (
        not isinstance(value.get("spdxVersion"), str)
        or not value["spdxVersion"].startswith("SPDX-2.")
        or value.get("dataLicense") != "CC0-1.0"
        or not isinstance(value.get("packages"), list)
        or not value["packages"]
    ):
        raise RuntimeContextError("runtime_sbom_invalid")


def _validate_pointer(path: Path, expected_sha256: str, expected_revision: str) -> dict[str, Any]:
    value, raw = _read_json(path, maximum=2 * 1024 * 1024, code="active_pointer_invalid")
    sha_fields = (
        "current_runner_implementation_sha256",
        "runtime_configuration_sha256",
        "sealed_evaluation_sha256",
        "source_recognizer_release_sha256",
        "source_recognizer_receipt_sha256",
        "source_recognizer_training_intent_sha256",
        "source_recognizer_selection_receipt_sha256",
        "source_recognizer_parent_selection_receipt_sha256",
    )
    if (
        hashlib.sha256(raw).hexdigest() != expected_sha256
        or raw != _canonical_pointer_bytes(value)
        or set(value) != _POINTER_V12_FIELDS
        or value.get("schema_version") != "12.0"
        or value.get("status") != "activated"
        or value.get("activation_kind") != "source_attested_synthetic_raster_text"
        or value.get("dataset_id") != "synthetic-raster-text-v1"
        or value.get("engine_kind") != _ENGINE_KIND
        or value.get("metric_revision") != _METRIC_REVISION
        or value.get("evidence_revision") != _EVIDENCE_REVISION
        or value.get("latency_gate_revision") != _LATENCY_GATE_REVISION
        or value.get("quality_gate_revision") != _QUALITY_GATE_REVISION
        or value.get("evidence_generation_revision") != _EVIDENCE_GENERATION_REVISION
        or value.get("runtime_validation_revision") != _RUNTIME_VALIDATION_REVISION
        or value.get("raster_text_profile_revision") != expected_revision
        or any(not _SHA256.fullmatch(str(value.get(field, ""))) for field in sha_fields)
    ):
        raise RuntimeContextError("active_pointer_invalid")
    return value


def _validate_runtime_profile_v19(
    path: Path, expected_sha256: str, pointer: dict[str, Any]
) -> dict[str, Any]:
    value, raw = _read_json(path, maximum=2 * 1024 * 1024, code="runtime_profile_invalid")
    if (
        hashlib.sha256(raw).hexdigest() != expected_sha256
        or raw != _canonical_json_bytes(value)
        or value.get("schema_version") != "19.0"
        or value.get("engine_kind") != _ENGINE_KIND
        or value.get("raster_text_profile_revision") != pointer.get("raster_text_profile_revision")
        or not _SHA256.fullmatch(str(value.get("bundle_manifest_sha256", "")))
        or not _SHA256.fullmatch(str(value.get("mobile_model_bundle_manifest_sha256", "")))
        or not _SHA256.fullmatch(str(value.get("sidecar_interpreter_sha256", "")))
        or value.get("source_recognizer_release_sha256")
        != pointer.get("source_recognizer_release_sha256")
        or value.get("source_recognizer_receipt_sha256")
        != pointer.get("source_recognizer_receipt_sha256")
        or value.get("source_recognizer_training_intent_sha256")
        != pointer.get("source_recognizer_training_intent_sha256")
        or value.get("source_recognizer_selection_receipt_relative_path")
        != pointer.get("source_recognizer_selection_receipt_path")
        or value.get("source_recognizer_selection_receipt_sha256")
        != pointer.get("source_recognizer_selection_receipt_sha256")
        or value.get("source_recognizer_parent_selection_receipt_sha256")
        != pointer.get("source_recognizer_parent_selection_receipt_sha256")
    ):
        raise RuntimeContextError("runtime_profile_invalid")
    for field in (
        "bundle_root_relative_path",
        "mobile_model_bundle_root_relative_path",
        "sidecar_interpreter_path",
    ):
        _relative_path(value.get(field))
    return value


def _validate_sealed_evaluation_v9(
    path: Path, expected_sha256: str, pointer: dict[str, Any]
) -> dict[str, Any]:
    value, raw = _read_json(path, maximum=2 * 1024 * 1024, code="sealed_evaluation_invalid")
    mirrored_fields = (
        "metric_revision",
        "evidence_revision",
        "latency_gate_revision",
        "quality_gate_revision",
        "evidence_generation_revision",
        "current_runner_implementation_sha256",
        "raster_text_profile_revision",
        "runtime_configuration_sha256",
        "source_recognizer_release_sha256",
        "source_recognizer_receipt_sha256",
        "source_recognizer_training_intent_sha256",
        "source_recognizer_parent_selection_receipt_sha256",
    )
    expected_stage = {
        "sealed-stage1-50-family-v1": ("early_pass", 50, 1_200),
        "sealed-stage2-100-family-v1": ("continue", 100, 2_400),
    }
    projection = value.get("projection_kind")
    latency = value.get("latency_p95_seconds")
    warm_latency = value.get("service_warm_latency_p95_seconds")
    minimum_confidence = value.get("minimum_page_confidence")
    if (
        hashlib.sha256(raw).hexdigest() != expected_sha256
        or raw != _canonical_json_bytes(value)
        or set(value) != _SEALED_V9_FIELDS
        or value.get("schema_version") != "9.0"
        or value.get("artifact_kind") != "synthetic_raster_semantic_text_fresh_staged_evaluation"
        or value.get("dataset_id") != "synthetic-raster-text-v1"
        or value.get("split") != "fresh-sealed"
        or value.get("source_truth_kind") != "pdf_content_stream_text_v1"
        or any(value.get(field) != pointer.get(field) for field in mirrored_fields)
        or projection not in expected_stage
        or (
            value.get("stage1_decision"),
            value.get("family_count"),
            value.get("asset_count"),
        )
        != expected_stage.get(str(projection))
        or value.get("selection_receipt_path")
        != pointer.get("source_recognizer_selection_receipt_path")
        or value.get("selection_receipt_sha256")
        != pointer.get("source_recognizer_selection_receipt_sha256")
        or not isinstance(value.get("extractor_revision"), str)
        or not value["extractor_revision"]
        or value.get("hard_timeout_seconds") != 30
        or isinstance(latency, bool)
        or not isinstance(latency, (int, float))
        or not 0 < latency <= 10
        or isinstance(warm_latency, bool)
        or not isinstance(warm_latency, (int, float))
        or not 0 < warm_latency <= 10
        or isinstance(minimum_confidence, bool)
        or not isinstance(minimum_confidence, (int, float))
        or not 0 <= minimum_confidence <= 1
    ):
        raise RuntimeContextError("sealed_evaluation_invalid")

    expected_scores = {
        (document_type, profile_id)
        for document_type in ("bill_of_lading", "commercial_invoice", "packing_list")
        for profile_id in ("png-216", "jpeg-216-q75", "jpeg-150-q75", "scan-pdf-216-q75")
    }
    scores = value.get("scores")
    if not isinstance(scores, list) or len(scores) != 12:
        raise RuntimeContextError("sealed_evaluation_invalid")
    observed_scores: set[tuple[object, object]] = set()
    for score in scores:
        if not isinstance(score, dict) or set(score) != {
            "character_error_rate",
            "critical_value_exact_match",
            "document_type",
            "profile_id",
        }:
            raise RuntimeContextError("sealed_evaluation_invalid")
        cer = score.get("character_error_rate")
        exact = score.get("critical_value_exact_match")
        if (
            isinstance(cer, bool)
            or not isinstance(cer, (int, float))
            or not 0 <= cer <= 0.02
            or isinstance(exact, bool)
            or not isinstance(exact, (int, float))
            or not 0.97 <= exact <= 1
        ):
            raise RuntimeContextError("sealed_evaluation_invalid")
        document_type = score.get("document_type")
        profile_id = score.get("profile_id")
        if (
            not isinstance(document_type, str)
            or not isinstance(profile_id, str)
            or (document_type, profile_id) not in expected_scores
        ):
            raise RuntimeContextError("sealed_evaluation_invalid")
        observed_scores.add((document_type, profile_id))
    if observed_scores != expected_scores:
        raise RuntimeContextError("sealed_evaluation_invalid")
    return value


def _declared_path_map(manifest: dict[str, Any]) -> dict[PurePosixPath, dict[str, Any]]:
    files = manifest.get("files")
    # The trust-root deployment manifest is intentionally not self-declared,
    # but it is still one file in the transported closure.
    if not isinstance(files, list) or not 1 <= len(files) <= _MAX_DECLARED_FILE_COUNT:
        raise RuntimeContextError("runtime_manifest_files_invalid")
    declared: dict[PurePosixPath, dict[str, Any]] = {}
    for record in files:
        if not isinstance(record, dict) or set(record) != {
            "byte_size",
            "relative_path",
            "role",
            "sha256",
        }:
            raise RuntimeContextError("runtime_manifest_files_invalid")
        relative = _relative_path(record.get("relative_path"))
        digest = record.get("sha256")
        size = record.get("byte_size")
        role = record.get("role")
        if (
            relative == _MANIFEST_RELATIVE_PATH
            or relative in declared
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
            or role not in _ALLOWED_ROLES
        ):
            raise RuntimeContextError("runtime_manifest_files_invalid")
        declared[relative] = record
    if list(declared) != sorted(declared, key=lambda item: item.as_posix()):
        raise RuntimeContextError("runtime_manifest_files_invalid")
    return declared


def _release_manifest_records(
    *,
    root: Path,
    release_root: PurePosixPath,
    expected_sha256: object,
    declared: dict[PurePosixPath, dict[str, Any]],
    payload_role: str,
    code: str,
) -> dict[PurePosixPath, dict[str, Any]]:
    """Bind one copied release root to its own exhaustive byte manifest."""

    manifest_relative = release_root / "manifest.json"
    manifest_record = declared.get(manifest_relative)
    if (
        not isinstance(expected_sha256, str)
        or _SHA256.fullmatch(expected_sha256) is None
        or manifest_record is None
        or manifest_record["role"] != "inventory"
        or manifest_record["sha256"] != expected_sha256
    ):
        raise RuntimeContextError(code)
    value, raw = _read_json(
        _regular_child(root, manifest_relative),
        maximum=_MAX_MANIFEST_BYTES,
        code=code,
    )
    files = value.get("files")
    if (
        raw != _canonical_json_bytes(value)
        or hashlib.sha256(raw).hexdigest() != expected_sha256
        or not isinstance(files, list)
        or not 1 <= len(files) <= _MAX_FILE_COUNT
    ):
        raise RuntimeContextError(code)

    release_records: dict[PurePosixPath, dict[str, Any]] = {}
    relative_order: list[str] = []
    for item in files:
        if not isinstance(item, dict) or set(item) != {"byte_size", "path", "sha256"}:
            raise RuntimeContextError(code)
        suffix = _relative_path(item.get("path"))
        digest = item.get("sha256")
        size = item.get("byte_size")
        relative = release_root / suffix
        if (
            suffix == PurePosixPath("manifest.json")
            or relative in release_records
            or not isinstance(digest, str)
            or _SHA256.fullmatch(digest) is None
            or isinstance(size, bool)
            or not isinstance(size, int)
            or size < 0
        ):
            raise RuntimeContextError(code)
        release_records[relative] = item
        relative_order.append(suffix.as_posix())
    if relative_order != sorted(relative_order):
        raise RuntimeContextError(code)

    expected_paths = {manifest_relative, *release_records}
    actual_paths = {relative for relative in declared if _is_beneath(relative, release_root)}
    if actual_paths != expected_paths:
        raise RuntimeContextError(code)
    for relative in expected_paths:
        outer = declared[relative]
        suffix = relative.relative_to(release_root)
        expected_role = (
            "inventory"
            if suffix.name == "manifest.json"
            or "inventories" in suffix.parts
            or payload_role == "model"
            and "provenance" in suffix.parts
            else payload_role
        )
        if outer["role"] != expected_role:
            raise RuntimeContextError("runtime_role_path_invalid")
        if relative == manifest_relative:
            continue
        inner = release_records[relative]
        if outer["sha256"] != inner["sha256"] or outer["byte_size"] != inner["byte_size"]:
            raise RuntimeContextError(code)
    return release_records


def _tree_paths(root: Path) -> tuple[set[PurePosixPath], list[Path]]:
    """Walk without following links and stop at the fixed entry/file bounds."""

    files: set[PurePosixPath] = set()
    directories: list[Path] = [root]
    pending: list[Path] = [root]
    entries_seen = 0
    try:
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    entries_seen += 1
                    if entries_seen > _MAX_TREE_ENTRIES:
                        raise RuntimeContextError("runtime_tree_entry_limit_exceeded")
                    candidate = Path(entry.path)
                    if entry.is_symlink():
                        raise RuntimeContextError("runtime_symlink_forbidden")
                    if entry.is_dir(follow_symlinks=False):
                        directories.append(candidate)
                        pending.append(candidate)
                    elif entry.is_file(follow_symlinks=False):
                        relative = _relative_path(candidate.relative_to(root).as_posix())
                        if relative != _MANIFEST_RELATIVE_PATH:
                            files.add(relative)
                        if len(files) > _MAX_DECLARED_FILE_COUNT:
                            raise RuntimeContextError("runtime_tree_entry_limit_exceeded")
                    else:
                        raise RuntimeContextError("runtime_special_file_forbidden")
    except RuntimeContextError:
        raise
    except (OSError, ValueError):
        raise RuntimeContextError("runtime_tree_invalid") from None
    return files, directories


def _validate_no_forbidden_material(
    declared: dict[PurePosixPath, dict[str, Any]], role_counts: Counter[str]
) -> None:
    evidence_bytes = 0
    for relative, record in declared.items():
        if any(part.casefold() in _FORBIDDEN_PATH_COMPONENTS for part in relative.parts):
            raise RuntimeContextError("evaluation_corpus_forbidden")
        if any(
            part.casefold() in {".git", ".venv", "__pycache__", "cache", "secrets"}
            for part in relative.parts
        ):
            raise RuntimeContextError("runtime_cache_or_secret_forbidden")
        role = str(record["role"])
        role_counts[role] += 1
        suffix = relative.suffix.casefold()
        if suffix in _ALWAYS_FORBIDDEN_DOCUMENT_SUFFIXES:
            raise RuntimeContextError("evaluation_corpus_forbidden")
        if suffix in _EVIDENCE_RESOURCE_SUFFIXES and role != "sidecar-runtime":
            raise RuntimeContextError("evaluation_corpus_forbidden")
        if role == "activation-evidence":
            evidence_bytes += int(record["byte_size"])
    if (
        role_counts["activation-evidence"] > _MAX_EVIDENCE_FILES
        or evidence_bytes > _MAX_EVIDENCE_BYTES
    ):
        raise RuntimeContextError("evaluation_corpus_forbidden")
    if any(role_counts[role] == 0 for role in _REQUIRED_ROLE_COUNTS):
        raise RuntimeContextError("runtime_required_role_missing")
    if role_counts["active-pointer"] != 1 or role_counts["runtime-profile"] != 1:
        raise RuntimeContextError("runtime_required_role_invalid")
    if role_counts["sbom"] != 1 or role_counts["licenses"] != 1:
        raise RuntimeContextError("runtime_required_role_invalid")


def _verify_runtime_context(root: Path) -> dict[str, object]:
    """Verify all declared bytes and return a safe, compact result."""

    try:
        resolved_root = root.resolve(strict=True)
    except OSError:
        raise RuntimeContextError("runtime_root_invalid") from None
    if root.is_symlink() or not resolved_root.is_dir():
        raise RuntimeContextError("runtime_root_invalid")
    manifest_path = _regular_child(resolved_root, _MANIFEST_RELATIVE_PATH)
    manifest, raw = _read_json(
        manifest_path,
        maximum=_MAX_MANIFEST_BYTES,
        code="runtime_manifest_invalid",
    )
    if os.name == "posix":
        try:
            manifest_mode = manifest_path.stat().st_mode
        except OSError:
            raise RuntimeContextError("runtime_manifest_invalid") from None
        if manifest_mode & (stat.S_IWGRP | stat.S_IWOTH) or not manifest_mode & stat.S_IROTH:
            raise RuntimeContextError("runtime_file_permissions_invalid")
    if raw != _canonical_json_bytes(manifest):
        raise RuntimeContextError("runtime_manifest_not_canonical")
    required = {
        "active_pointer",
        "artifact_kind",
        "files",
        "license_manifest",
        "platform",
        "runtime_binding_sha256",
        "sbom",
        "schema_version",
    }
    if (
        set(manifest) != required
        or manifest.get("schema_version") != "1.0"
        or manifest.get("artifact_kind") != "customs-raster-text-runtime-oci"
        or not _SHA256.fullmatch(str(manifest.get("runtime_binding_sha256", "")))
        or manifest.get("platform")
        != {"architecture": "amd64", "os": "linux", "python_abi": "cp311"}
    ):
        raise RuntimeContextError("runtime_manifest_invalid")
    active = manifest.get("active_pointer")
    sbom = manifest.get("sbom")
    licenses = manifest.get("license_manifest")
    if (
        not isinstance(active, dict)
        or set(active) != {"raster_text_profile_revision", "relative_path", "sha256"}
        or active.get("relative_path") != _POINTER_RELATIVE_PATH.as_posix()
        or not _SHA256.fullmatch(str(active.get("sha256", "")))
        or not _PROFILE_REVISION.fullmatch(str(active.get("raster_text_profile_revision", "")))
        or not isinstance(sbom, dict)
        or set(sbom) != {"relative_path", "sha256"}
        or not _SHA256.fullmatch(str(sbom.get("sha256", "")))
        or not isinstance(licenses, dict)
        or set(licenses) != {"relative_path", "sha256"}
        or not _SHA256.fullmatch(str(licenses.get("sha256", "")))
    ):
        raise RuntimeContextError("runtime_manifest_invalid")
    assert isinstance(sbom, dict)
    assert isinstance(licenses, dict)

    declared = _declared_path_map(manifest)
    roles: Counter[str] = Counter()
    _validate_no_forbidden_material(declared, roles)

    total_bytes = len(raw)
    if total_bytes > _MAX_TOTAL_BYTES:
        raise RuntimeContextError("runtime_size_limit_exceeded")
    if os.name == "posix":
        try:
            root_mode = resolved_root.stat().st_mode
        except OSError:
            raise RuntimeContextError("runtime_tree_invalid") from None
        if root_mode & (stat.S_IROTH | stat.S_IXOTH) != (
            stat.S_IROTH | stat.S_IXOTH
        ) or root_mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise RuntimeContextError("runtime_directory_permissions_invalid")
    actual, directories = _tree_paths(resolved_root)
    for candidate in directories:
        if os.name == "posix":
            try:
                directory_mode = candidate.stat().st_mode
            except OSError:
                raise RuntimeContextError("runtime_tree_invalid") from None
            if directory_mode & (stat.S_IROTH | stat.S_IXOTH) != (
                stat.S_IROTH | stat.S_IXOTH
            ) or directory_mode & (stat.S_IWGRP | stat.S_IWOTH):
                raise RuntimeContextError("runtime_directory_permissions_invalid")
    if actual != set(declared):
        raise RuntimeContextError("runtime_manifest_file_set_mismatch")

    for relative, record in declared.items():
        candidate = _regular_child(resolved_root, relative)
        digest, size = _sha256(candidate)
        if digest != record["sha256"] or size != record["byte_size"]:
            raise RuntimeContextError("runtime_file_digest_mismatch")
        try:
            mode = candidate.stat().st_mode
        except OSError:
            raise RuntimeContextError("runtime_file_invalid") from None
        if os.name == "posix" and (mode & (stat.S_IWGRP | stat.S_IWOTH) or not mode & stat.S_IROTH):
            raise RuntimeContextError("runtime_file_permissions_invalid")
        receipt_like = candidate.suffix.casefold() == ".json" and any(
            marker in candidate.name.casefold() for marker in _RECEIPT_NAME_MARKERS
        )
        if record["role"] == "activation-evidence" or receipt_like:
            evidence_value, _evidence_raw = _read_json(
                candidate,
                maximum=_MAX_EVIDENCE_BYTES,
                code="activation_evidence_invalid",
            )
            if not evidence_value:
                raise RuntimeContextError("activation_evidence_invalid")
        total_bytes += size
        if total_bytes > _MAX_TOTAL_BYTES:
            raise RuntimeContextError("runtime_size_limit_exceeded")

    pointer_record = declared.get(_POINTER_RELATIVE_PATH)
    if pointer_record is None or pointer_record["role"] != "active-pointer":
        raise RuntimeContextError("active_pointer_invalid")
    pointer = _validate_pointer(
        _regular_child(resolved_root, _POINTER_RELATIVE_PATH),
        str(active["sha256"]),
        str(active["raster_text_profile_revision"]),
    )
    if pointer_record["sha256"] != active["sha256"]:
        raise RuntimeContextError("active_pointer_invalid")
    if pointer["current_runner_implementation_sha256"] != manifest["runtime_binding_sha256"]:
        raise RuntimeContextError("runtime_binding_invalid")

    def require_binding(
        *, relative_path: object, expected_sha256: object, expected_role: str
    ) -> tuple[PurePosixPath, Path]:
        relative = _relative_path(relative_path)
        record = declared.get(relative)
        if record is None or record["role"] != expected_role or record["sha256"] != expected_sha256:
            raise RuntimeContextError("pointer_material_binding_invalid")
        return relative, _regular_child(resolved_root, relative)

    _profile_relative, profile_path = require_binding(
        relative_path=pointer.get("runtime_configuration_path"),
        expected_sha256=pointer.get("runtime_configuration_sha256"),
        expected_role="runtime-profile",
    )
    profile = _validate_runtime_profile_v19(
        profile_path,
        str(pointer["runtime_configuration_sha256"]),
        pointer,
    )
    interpreter_relative = _relative_path(profile.get("sidecar_interpreter_path"))
    interpreter_record = declared.get(interpreter_relative)
    if (
        interpreter_record is None
        or interpreter_record["role"] != "sidecar-runtime"
        or interpreter_record["sha256"] != profile.get("sidecar_interpreter_sha256")
    ):
        raise RuntimeContextError("runtime_profile_invalid")
    interpreter_path = _regular_child(resolved_root, interpreter_relative)
    if os.name == "posix" and not interpreter_path.stat().st_mode & stat.S_IXOTH:
        raise RuntimeContextError("runtime_file_permissions_invalid")
    try:
        with interpreter_path.open("rb") as stream:
            elf_header = stream.read(20)
    except OSError:
        raise RuntimeContextError("runtime_interpreter_invalid") from None
    # ELF64, little-endian, current ELF version, ET_EXEC/ET_DYN, x86-64.
    if (
        len(elf_header) != 20
        or elf_header[:7] != b"\x7fELF\x02\x01\x01"
        or elf_header[16:18] not in {b"\x02\x00", b"\x03\x00"}
        or elf_header[18:20] != b"\x3e\x00"
    ):
        raise RuntimeContextError("runtime_interpreter_invalid")
    sidecar_root = _relative_path(profile.get("bundle_root_relative_path"))
    model_root = _relative_path(profile.get("mobile_model_bundle_root_relative_path"))
    if (
        sidecar_root == model_root
        or _is_beneath(sidecar_root, model_root)
        or _is_beneath(model_root, sidecar_root)
    ):
        raise RuntimeContextError("runtime_profile_invalid")
    for relative, runtime_record in declared.items():
        role = runtime_record["role"]
        if role == "sidecar-runtime" and not _is_beneath(relative, sidecar_root):
            raise RuntimeContextError("runtime_role_path_invalid")
        if role == "model" and not _is_beneath(relative, model_root):
            raise RuntimeContextError("runtime_role_path_invalid")
        if role == "inventory" and not (
            _is_beneath(relative, sidecar_root) or _is_beneath(relative, model_root)
        ):
            raise RuntimeContextError("runtime_role_path_invalid")
    sidecar_release_records = _release_manifest_records(
        root=resolved_root,
        release_root=sidecar_root,
        expected_sha256=profile.get("bundle_manifest_sha256"),
        declared=declared,
        payload_role="sidecar-runtime",
        code="runtime_platform_manifest_invalid",
    )
    _release_manifest_records(
        root=resolved_root,
        release_root=model_root,
        expected_sha256=profile.get("mobile_model_bundle_manifest_sha256"),
        declared=declared,
        payload_role="model",
        code="runtime_model_manifest_invalid",
    )
    # CSV/PNG package resources are allowed only when the content-addressed
    # sidecar release manifest itself declares their exact path, size and SHA.
    # Directory-name heuristics are deliberately not an authorization source.
    for relative, runtime_record in declared.items():
        if relative.suffix.casefold() in _EVIDENCE_RESOURCE_SUFFIXES:
            release_record = sidecar_release_records.get(relative)
            if (
                release_record is None
                or runtime_record["role"] != "sidecar-runtime"
                or runtime_record["sha256"] != release_record["sha256"]
                or runtime_record["byte_size"] != release_record["byte_size"]
            ):
                raise RuntimeContextError("evaluation_corpus_forbidden")
    try:
        interpreter_in_bundle = interpreter_relative.relative_to(sidecar_root).as_posix()
    except ValueError:
        raise RuntimeContextError("runtime_profile_invalid") from None
    sidecar_manifest_relative = sidecar_root / "manifest.json"
    sidecar_manifest_record = declared.get(sidecar_manifest_relative)
    if (
        sidecar_manifest_record is None
        or sidecar_manifest_record["role"] != "inventory"
        or sidecar_manifest_record["sha256"] != profile.get("bundle_manifest_sha256")
    ):
        raise RuntimeContextError("runtime_platform_manifest_invalid")
    sidecar_manifest, sidecar_manifest_raw = _read_json(
        _regular_child(resolved_root, sidecar_manifest_relative),
        maximum=_MAX_MANIFEST_BYTES,
        code="runtime_platform_manifest_invalid",
    )
    if (
        sidecar_manifest_raw != _canonical_json_bytes(sidecar_manifest)
        or hashlib.sha256(sidecar_manifest_raw).hexdigest() != profile.get("bundle_manifest_sha256")
        or sidecar_manifest.get("bundle_id") != "raster-text-direct-paddle-bundle-v1"
        or sidecar_manifest.get("schema_version") != "2.0"
        or (
            sidecar_manifest.get("runtime_operating_system"),
            sidecar_manifest.get("runtime_architecture"),
            sidecar_manifest.get("runtime_python_abi"),
        )
        != ("linux", "amd64", "cp311")
        or sidecar_manifest.get("sidecar_interpreter_relative_path") != interpreter_in_bundle
    ):
        raise RuntimeContextError("runtime_platform_manifest_invalid")

    _sealed_relative, sealed_path = require_binding(
        relative_path=pointer.get("sealed_evaluation_path"),
        expected_sha256=pointer.get("sealed_evaluation_sha256"),
        expected_role="activation-evidence",
    )
    sealed = _validate_sealed_evaluation_v9(
        sealed_path,
        str(pointer["sealed_evaluation_sha256"]),
        pointer,
    )
    # Bind the seven small, direct V9 attestation receipts.  Exact receipt
    # semantics and calibration's nested closure remain owned by the Agent
    # parser/verifier, not this data-only packaging verifier.
    evidence_bindings = (
        ("selection_receipt_path", "selection_receipt_sha256"),
        ("projection_receipt_path", "projection_receipt_sha256"),
        ("stage1_decision_receipt_path", "stage1_decision_receipt_sha256"),
        ("source_gold_receipt_path", "source_gold_receipt_sha256"),
        ("calibration_receipt_path", "calibration_receipt_sha256"),
        ("candidate_execution_path", "candidate_execution_sha256"),
        ("warm_service_smoke_receipt_path", "warm_service_smoke_receipt_sha256"),
    )
    for path_key, sha_key in evidence_bindings:
        require_binding(
            relative_path=sealed.get(path_key),
            expected_sha256=sealed.get(sha_key),
            expected_role="activation-evidence",
        )

    sbom_relative = _relative_path(sbom.get("relative_path"))
    license_relative = _relative_path(licenses.get("relative_path"))
    for relative, binding, role in (
        (sbom_relative, sbom, "sbom"),
        (license_relative, licenses, "licenses"),
    ):
        metadata_record = declared.get(relative)
        if (
            metadata_record is None
            or metadata_record["role"] != role
            or metadata_record["sha256"] != binding["sha256"]
        ):
            raise RuntimeContextError("deployment_metadata_binding_invalid")
    _validate_spdx(_regular_child(resolved_root, sbom_relative))
    licensed, _scopes = _validate_license_manifest(_regular_child(resolved_root, license_relative))
    inventory_packages: set[tuple[str, str]] = set()
    for relative, record in declared.items():
        if record["role"] == "inventory":
            inventory_packages.update(_inventory_packages(_regular_child(resolved_root, relative)))
    if not inventory_packages or not inventory_packages.issubset(licensed):
        raise RuntimeContextError("license_manifest_incomplete")

    return {
        "file_count": len(declared) + 1,
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "raster_text_profile_revision": active["raster_text_profile_revision"],
        "total_bytes": total_bytes,
    }


def verify_runtime_context(root: Path) -> dict[str, object]:
    """Fail closed with one content-free error for every malformed input."""

    try:
        return _verify_runtime_context(root)
    except RuntimeContextError:
        raise
    except Exception:
        # Do not leak parser, filesystem, or native-path details from a
        # publisher/final-image trust boundary.  KeyboardInterrupt and
        # SystemExit intentionally remain operator controls.
        raise RuntimeContextError("runtime_context_invalid") from None


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-preview-root", required=True, type=Path)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = verify_runtime_context(args.document_preview_root)
    except RuntimeContextError as exc:
        print(exc.code, file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
