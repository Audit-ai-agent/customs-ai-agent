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
_FORBIDDEN_TOP_LEVEL = frozenset(
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
_RAW_DOCUMENT_SUFFIXES = frozenset(
    {".bmp", ".csv", ".jpeg", ".jpg", ".jsonl", ".parquet", ".pdf", ".png", ".tif", ".tiff"}
)
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024
_MAX_FILE_COUNT = 20_000
_MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
_MAX_EVIDENCE_FILES = 128
_MAX_EVIDENCE_BYTES = 64 * 1024 * 1024


class RuntimeContextError(RuntimeError):
    """A content-free deployment contract failure."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


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
    if not isinstance(value, str) or not value or "\\" in value:
        raise RuntimeContextError("runtime_manifest_path_invalid")
    relative = PurePosixPath(value)
    if relative.is_absolute() or relative == PurePosixPath(".") or ".." in relative.parts:
        raise RuntimeContextError("runtime_manifest_path_invalid")
    if any(part in {"", "."} for part in relative.parts):
        raise RuntimeContextError("runtime_manifest_path_invalid")
    return relative


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
        value = json.loads(raw.decode("utf-8"))
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
    if not {"model", "python-package"}.issubset(scopes):
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
    if (
        hashlib.sha256(raw).hexdigest() != expected_sha256
        or value.get("schema_version") != "11.0"
        or value.get("status") != "activated"
        or value.get("activation_kind") != "source_attested_synthetic_raster_text"
        or value.get("runtime_validation_revision") != "activation-full-runtime-compact-v1"
        or value.get("raster_text_profile_revision") != expected_revision
        or not _SHA256.fullmatch(str(value.get("current_runner_implementation_sha256", "")))
    ):
        raise RuntimeContextError("active_pointer_invalid")
    return value


def _declared_path_map(manifest: dict[str, Any]) -> dict[PurePosixPath, dict[str, Any]]:
    files = manifest.get("files")
    if not isinstance(files, list) or not 1 <= len(files) <= _MAX_FILE_COUNT:
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


def _validate_no_forbidden_material(
    declared: dict[PurePosixPath, dict[str, Any]], role_counts: Counter[str]
) -> None:
    evidence_bytes = 0
    for relative, record in declared.items():
        if relative.parts[0].casefold() in _FORBIDDEN_TOP_LEVEL:
            raise RuntimeContextError("evaluation_corpus_forbidden")
        if any(
            part.casefold() in {".git", ".venv", "__pycache__", "cache", "secrets"}
            for part in relative.parts
        ):
            raise RuntimeContextError("runtime_cache_or_secret_forbidden")
        role = str(record["role"])
        role_counts[role] += 1
        if role == "activation-evidence":
            evidence_bytes += int(record["byte_size"])
            if relative.suffix.casefold() in _RAW_DOCUMENT_SUFFIXES:
                raise RuntimeContextError("evaluation_corpus_forbidden")
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


def verify_runtime_context(root: Path) -> dict[str, object]:
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

    declared = _declared_path_map(manifest)
    roles: Counter[str] = Counter()
    _validate_no_forbidden_material(declared, roles)

    actual: set[PurePosixPath] = set()
    total_bytes = 0
    for candidate in resolved_root.rglob("*"):
        if candidate.is_symlink():
            raise RuntimeContextError("runtime_symlink_forbidden")
        if candidate.is_dir():
            continue
        if not candidate.is_file():
            raise RuntimeContextError("runtime_special_file_forbidden")
        relative = PurePosixPath(candidate.relative_to(resolved_root).as_posix())
        if relative != _MANIFEST_RELATIVE_PATH:
            actual.add(relative)
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
        if os.name == "posix" and mode & (stat.S_IWGRP | stat.S_IWOTH):
            raise RuntimeContextError("runtime_file_permissions_invalid")
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

    for key, expected_role in (
        ("runtime_configuration_path", "runtime-profile"),
        ("sealed_evaluation_path", "activation-evidence"),
        ("revalidation_candidate_execution_path", "activation-evidence"),
        ("revalidation_smoke_receipt_path", "activation-evidence"),
        ("source_recognizer_selection_receipt_path", "activation-evidence"),
    ):
        relative = _relative_path(pointer.get(key))
        record = declared.get(relative)
        expected_digest = pointer.get(key.removesuffix("_path") + "_sha256")
        if record is None or record["role"] != expected_role or record["sha256"] != expected_digest:
            raise RuntimeContextError("pointer_material_binding_invalid")

    sbom_relative = _relative_path(sbom.get("relative_path"))
    license_relative = _relative_path(licenses.get("relative_path"))
    for relative, binding, role in (
        (sbom_relative, sbom, "sbom"),
        (license_relative, licenses, "licenses"),
    ):
        record = declared.get(relative)
        if record is None or record["role"] != role or record["sha256"] != binding["sha256"]:
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
        "file_count": len(declared),
        "manifest_sha256": hashlib.sha256(raw).hexdigest(),
        "raster_text_profile_revision": active["raster_text_profile_revision"],
        "total_bytes": total_bytes,
    }


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
