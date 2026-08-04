"""Verify the workspace-local synthetic dataset relocation contract.

The product package never reads local datasets.  These checks protect the
evaluation and fixture tooling that runs outside the wheel: a dataset is
usable only after its Agent-owned relocation binding, workspace manifest,
canonical provenance, and retired legacy paths all agree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Any

from jsonschema import Draft202012Validator


PROVENANCE_SCHEMA_VERSION = "customs-dataset-provenance/v2"
BINDING_SCHEMA_VERSION = "customs-dataset-relocation-binding/v1"
MANIFEST_SCHEMA_VERSION = "customs-dataset-relocation-manifest/v1"
MIGRATION_ID = "TASK-0028-dataset-canonical-relocation"
DEFAULT_BINDING_RELATIVE_PATH = Path("fixtures/dataset_provenance/v2/relocation-binding.json")
DEFAULT_MANIFEST_WORKSPACE_RELATIVE_PATH = PurePosixPath(
    "_local/data/dataset-relocation-manifest.v1.json"
)
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_CHECKSUM_LINE_PATTERN = re.compile(r"^([0-9a-f]{64})  (.+)$")


class DatasetProvenanceError(ValueError):
    """Raised when local dataset authority cannot be established safely."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


@dataclass(frozen=True, slots=True)
class VerifiedDatasetProvenance:
    dataset_id: str
    dataset_version: str
    workspace_root: Path
    dataset_root: Path
    raw_root: Path
    provenance_path: Path
    relocation_manifest_path: Path
    binding_sha256: str
    provenance_sha256: str


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise DatasetProvenanceError(
            "dataset_provenance_file_unreadable",
            "dataset provenance input cannot be read",
        ) from exc
    return digest.hexdigest()


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    value: dict[str, Any] = {}
    for key, item in pairs:
        if key in value:
            raise DatasetProvenanceError(
                "dataset_provenance_duplicate_json_key",
                "dataset provenance JSON contains a duplicate object key",
            )
        value[key] = item
    return value


def _read_json_object(
    path: Path, *, maximum_bytes: int = 1024 * 1024
) -> tuple[dict[str, Any], bytes]:
    if not path.is_file() or path.is_symlink():
        raise DatasetProvenanceError(
            "dataset_provenance_regular_file_required",
            "dataset provenance input must be a regular non-symlink file",
        )
    try:
        if path.stat().st_size > maximum_bytes:
            raise DatasetProvenanceError(
                "dataset_provenance_file_too_large",
                "dataset provenance input exceeds its size limit",
            )
        content = path.read_bytes()
        value = json.loads(
            content.decode("utf-8"),
            object_pairs_hook=_reject_duplicate_keys,
        )
    except DatasetProvenanceError:
        raise
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise DatasetProvenanceError(
            "dataset_provenance_invalid_json",
            "dataset provenance input must be valid UTF-8 JSON",
        ) from exc
    if not isinstance(value, dict):
        raise DatasetProvenanceError(
            "dataset_provenance_object_required",
            "dataset provenance JSON root must be an object",
        )
    return value, content


def _validate_schema(value: dict[str, Any], schema_path: Path, *, code: str) -> None:
    schema, _content = _read_json_object(schema_path)
    errors = sorted(
        Draft202012Validator(schema).iter_errors(value), key=lambda item: list(item.path)
    )
    if errors:
        raise DatasetProvenanceError(code, "dataset provenance JSON does not match its schema")


def _safe_workspace_path(workspace_root: Path, relative_value: object) -> Path:
    if not isinstance(relative_value, str) or not relative_value:
        raise DatasetProvenanceError(
            "dataset_provenance_path_invalid",
            "dataset provenance path must be a non-empty workspace-relative string",
        )
    if "\\" in relative_value:
        raise DatasetProvenanceError(
            "dataset_provenance_path_invalid",
            "dataset provenance paths must use portable POSIX separators",
        )
    relative = PurePosixPath(relative_value)
    if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
        raise DatasetProvenanceError(
            "dataset_provenance_path_invalid",
            "dataset provenance path must stay inside the workspace",
        )
    candidate = workspace_root.joinpath(*relative.parts)
    resolved_workspace = workspace_root.resolve(strict=True)
    resolved_candidate = candidate.resolve(strict=False)
    try:
        resolved_candidate.relative_to(resolved_workspace)
    except ValueError as exc:
        raise DatasetProvenanceError(
            "dataset_provenance_path_escape",
            "dataset provenance path escapes the workspace",
        ) from exc
    return candidate


def _assert_no_symlink_components(workspace_root: Path, path: Path) -> None:
    relative = path.relative_to(workspace_root)
    current = workspace_root
    for part in relative.parts:
        current = current / part
        if current.exists() and current.is_symlink():
            raise DatasetProvenanceError(
                "dataset_provenance_symlink_forbidden",
                "dataset provenance paths must not traverse symbolic links",
            )


def _discover_repository_root(lock_path: Path) -> Path:
    resolved_lock = lock_path.resolve(strict=True)
    for ancestor in (resolved_lock.parent, *resolved_lock.parents):
        if (ancestor / "pyproject.toml").is_file() and (ancestor / "fixtures").is_dir():
            return ancestor
    raise DatasetProvenanceError(
        "dataset_repository_root_not_found",
        "cannot locate the Agent repository that owns the dataset binding",
    )


def _discover_workspace_root(repository_root: Path) -> Path:
    for ancestor in (repository_root.parent, *repository_root.parents):
        if (ancestor / "AGENTS.md").is_file():
            return ancestor.resolve(strict=True)
    raise DatasetProvenanceError(
        "dataset_workspace_root_not_found",
        "cannot locate the workspace that owns _local/data",
    )


def _binding_entry(
    binding: dict[str, Any], dataset_id: str, dataset_version: str
) -> dict[str, Any]:
    matches = [
        item
        for item in binding.get("datasets", [])
        if isinstance(item, dict)
        and item.get("dataset_id") == dataset_id
        and item.get("dataset_version") == dataset_version
    ]
    if len(matches) != 1:
        raise DatasetProvenanceError(
            "dataset_relocation_binding_identity_mismatch",
            "relocation binding must contain exactly one matching dataset",
        )
    return matches[0]


def _validate_manifest_timestamp(value: object) -> None:
    if not isinstance(value, str):
        raise DatasetProvenanceError(
            "dataset_relocation_timestamp_invalid",
            "relocation verification time is missing",
        )
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError as exc:
        raise DatasetProvenanceError(
            "dataset_relocation_timestamp_invalid",
            "relocation verification time is not ISO-8601",
        ) from exc
    if parsed.tzinfo is None:
        raise DatasetProvenanceError(
            "dataset_relocation_timestamp_invalid",
            "relocation verification time must include a timezone",
        )


def _verify_raw_checksum_manifest(
    dataset_root: Path,
    raw_root: Path,
    checksums_path: Path,
    *,
    supplemental_readme_sha256: object,
) -> tuple[int, int]:
    """Verify every canonical raw byte against the pinned checksum manifest."""

    try:
        lines = checksums_path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise DatasetProvenanceError(
            "dataset_canonical_checksum_manifest_invalid",
            "canonical raw checksum manifest is not readable UTF-8 text",
        ) from exc
    if not lines:
        raise DatasetProvenanceError(
            "dataset_canonical_checksum_manifest_invalid",
            "canonical raw checksum manifest must not be empty",
        )

    resolved_raw = raw_root.resolve(strict=True)
    verified_targets: set[Path] = set()
    for line in lines:
        match = _CHECKSUM_LINE_PATTERN.fullmatch(line)
        if match is None:
            raise DatasetProvenanceError(
                "dataset_canonical_checksum_manifest_invalid",
                "canonical raw checksum manifest has an invalid line",
            )
        expected_sha256, relative_value = match.groups()
        if "\\" in relative_value:
            raise DatasetProvenanceError(
                "dataset_canonical_checksum_manifest_invalid",
                "canonical raw checksum paths must use POSIX separators",
            )
        relative = PurePosixPath(relative_value)
        if relative.is_absolute() or any(part in {"", ".", ".."} for part in relative.parts):
            raise DatasetProvenanceError(
                "dataset_canonical_checksum_manifest_invalid",
                "canonical raw checksum paths must be safe relative paths",
            )

        candidates: set[Path] = set()
        for base in (dataset_root, raw_root):
            candidate = base.joinpath(*relative.parts)
            resolved = candidate.resolve(strict=False)
            try:
                resolved.relative_to(resolved_raw)
            except ValueError:
                continue
            if candidate.is_file() and not candidate.is_symlink():
                _assert_no_symlink_components(dataset_root, candidate)
                candidates.add(resolved)
        if len(candidates) != 1:
            raise DatasetProvenanceError(
                "dataset_canonical_checksum_target_invalid",
                "each checksum entry must resolve to exactly one canonical raw file",
            )
        target = candidates.pop()
        if target in verified_targets:
            raise DatasetProvenanceError(
                "dataset_canonical_checksum_target_duplicate",
                "canonical raw checksum manifest contains a duplicate target",
            )
        if _sha256_file(target) != expected_sha256:
            raise DatasetProvenanceError(
                "dataset_canonical_raw_checksum_mismatch",
                "canonical raw file content differs from its pinned checksum",
            )
        verified_targets.add(target)

    if supplemental_readme_sha256 is not None:
        if not isinstance(supplemental_readme_sha256, str) or not _SHA256_PATTERN.fullmatch(
            supplemental_readme_sha256
        ):
            raise DatasetProvenanceError(
                "dataset_canonical_supplemental_file_mismatch",
                "canonical supplemental source hash is invalid",
            )
        readme = raw_root / "README.txt"
        if readme.is_symlink() or _sha256_file(readme) != supplemental_readme_sha256:
            raise DatasetProvenanceError(
                "dataset_canonical_supplemental_file_mismatch",
                "canonical supplemental source evidence differs from provenance",
            )
        _assert_no_symlink_components(dataset_root, readme)
        verified_targets.add(readme.resolve(strict=True))

    raw_files = [path for path in raw_root.rglob("*") if path.is_file()]
    if any(path.is_symlink() for path in raw_files):
        raise DatasetProvenanceError(
            "dataset_provenance_symlink_forbidden",
            "canonical raw files must not be symbolic links",
        )
    resolved_files = {path.resolve(strict=True) for path in raw_files}
    if resolved_files != verified_targets:
        raise DatasetProvenanceError(
            "dataset_canonical_raw_checksum_coverage_mismatch",
            "every canonical raw file must be covered by a pinned content hash",
        )
    return len(raw_files), sum(path.stat().st_size for path in raw_files)


def verify_dataset_provenance(
    lock_path: Path,
    *,
    workspace_root: Path | None = None,
    expected_dataset_root: Path | None = None,
    expected_raw_root: Path | None = None,
) -> VerifiedDatasetProvenance:
    """Fail closed unless a canonical `_local/data` dataset is fully bound."""

    repository_root = _discover_repository_root(lock_path)
    workspace = (
        workspace_root.resolve(strict=True)
        if workspace_root is not None
        else _discover_workspace_root(repository_root)
    )
    if not workspace.is_dir() or workspace.is_symlink():
        raise DatasetProvenanceError(
            "dataset_workspace_root_invalid",
            "workspace root must be a real directory",
        )

    lock, _lock_content = _read_json_object(lock_path)
    contract = lock.get("dataset_provenance_v2")
    if not isinstance(contract, dict):
        raise DatasetProvenanceError(
            "dataset_provenance_v2_contract_missing",
            "dataset lock does not contain the V2 provenance binding",
        )
    dataset_id = lock.get("dataset_id")
    dataset_version = lock.get("dataset_version")
    if not isinstance(dataset_id, str) or not isinstance(dataset_version, str):
        raise DatasetProvenanceError(
            "dataset_provenance_identity_invalid",
            "dataset lock identity is invalid",
        )
    if (
        contract.get("schema_version") != "customs-dataset-provenance-binding/v2"
        or contract.get("migration_id") != MIGRATION_ID
        or contract.get("root_reference_preserved") is not False
        or contract.get("root_reference_status") != "retired"
        or contract.get("fallback_to_legacy_paths_allowed") is not False
    ):
        raise DatasetProvenanceError(
            "dataset_provenance_v2_contract_invalid",
            "dataset lock does not enforce the retired-root V2 contract",
        )

    binding_relative = contract.get("binding_repository_relative_path")
    if binding_relative != DEFAULT_BINDING_RELATIVE_PATH.as_posix():
        raise DatasetProvenanceError(
            "dataset_relocation_binding_path_mismatch",
            "dataset lock points at an unexpected relocation binding",
        )
    binding_path = repository_root / DEFAULT_BINDING_RELATIVE_PATH
    binding, binding_content = _read_json_object(binding_path)
    binding_sha256 = _sha256_bytes(binding_content)
    if (
        binding.get("schema_version") != BINDING_SCHEMA_VERSION
        or binding.get("migration_id") != MIGRATION_ID
        or contract.get("binding_sha256") != binding_sha256
        or not _SHA256_PATTERN.fullmatch(binding_sha256)
    ):
        raise DatasetProvenanceError(
            "dataset_relocation_binding_hash_mismatch",
            "Agent-owned relocation binding does not match the dataset lock",
        )
    schema_root = binding_path.parent / "schemas"
    _validate_schema(
        binding,
        schema_root / "relocation-binding.schema.json",
        code="dataset_relocation_binding_schema_invalid",
    )
    entry = _binding_entry(binding, dataset_id, dataset_version)

    compared_contract = {
        key: contract.get(key)
        for key in (
            "canonical_dataset_workspace_relative_path",
            "canonical_raw_workspace_relative_path",
            "provenance_workspace_relative_path",
            "legacy_source_workspace_relative_path",
            "legacy_canonical_workspace_relative_path",
        )
    }
    if compared_contract != {
        "canonical_dataset_workspace_relative_path": entry[
            "canonical_dataset_workspace_relative_path"
        ],
        "canonical_raw_workspace_relative_path": entry["canonical_raw_workspace_relative_path"],
        "provenance_workspace_relative_path": entry["provenance_workspace_relative_path"],
        "legacy_source_workspace_relative_path": entry["legacy_source_workspace_relative_path"],
        "legacy_canonical_workspace_relative_path": entry[
            "legacy_canonical_workspace_relative_path"
        ],
    }:
        raise DatasetProvenanceError(
            "dataset_relocation_lock_binding_mismatch",
            "dataset lock paths do not match the Agent-owned relocation binding",
        )

    manifest_relative = binding.get("manifest_workspace_relative_path")
    if (
        manifest_relative != DEFAULT_MANIFEST_WORKSPACE_RELATIVE_PATH.as_posix()
        or contract.get("relocation_manifest_workspace_relative_path") != manifest_relative
    ):
        raise DatasetProvenanceError(
            "dataset_relocation_manifest_path_mismatch",
            "relocation manifest path is not pinned",
        )
    manifest_path = _safe_workspace_path(workspace, manifest_relative)
    _assert_no_symlink_components(workspace, manifest_path)
    manifest, _manifest_content = _read_json_object(manifest_path)
    _validate_schema(
        manifest,
        schema_root / "relocation-manifest.schema.json",
        code="dataset_relocation_manifest_schema_invalid",
    )
    _validate_manifest_timestamp(manifest.get("verified_at"))
    if (
        manifest.get("schema_version") != MANIFEST_SCHEMA_VERSION
        or manifest.get("migration_id") != MIGRATION_ID
        or manifest.get("relocation_status") != "completed_and_verified"
        or manifest.get("binding_sha256") != binding_sha256
        or manifest.get("binding_repository_relative_path")
        != DEFAULT_BINDING_RELATIVE_PATH.as_posix()
    ):
        raise DatasetProvenanceError(
            "dataset_relocation_manifest_binding_mismatch",
            "workspace relocation manifest does not match the Agent-owned binding",
        )
    manifest_entry = _binding_entry(manifest, dataset_id, dataset_version)
    expected_manifest_entry = {
        "dataset_id": dataset_id,
        "dataset_version": dataset_version,
        "canonical_dataset_workspace_relative_path": entry[
            "canonical_dataset_workspace_relative_path"
        ],
        "canonical_raw_workspace_relative_path": entry["canonical_raw_workspace_relative_path"],
        "provenance_sha256": entry["provenance_sha256"],
        "raw_file_count": entry["raw_integrity"]["raw_file_count"],
        "raw_bytes": entry["raw_integrity"]["raw_bytes"],
        "raw_checksums_sha256": entry["raw_integrity"]["raw_checksums_sha256"],
        "inventory_sha256": entry["raw_integrity"]["inventory_sha256"],
        "legacy_source_reference_removed": True,
        "legacy_canonical_registration_removed": True,
        "canonical_hash_verification_passed": True,
    }
    if manifest_entry != expected_manifest_entry:
        raise DatasetProvenanceError(
            "dataset_relocation_manifest_entry_mismatch",
            "workspace relocation evidence does not match the pinned dataset entry",
        )

    dataset_root = _safe_workspace_path(
        workspace,
        entry["canonical_dataset_workspace_relative_path"],
    )
    raw_root = _safe_workspace_path(workspace, entry["canonical_raw_workspace_relative_path"])
    provenance_path = _safe_workspace_path(workspace, entry["provenance_workspace_relative_path"])
    for path in (dataset_root, raw_root, provenance_path):
        _assert_no_symlink_components(workspace, path)
    if (
        not dataset_root.is_dir()
        or dataset_root.is_symlink()
        or not raw_root.is_dir()
        or raw_root.is_symlink()
        or not provenance_path.is_file()
        or provenance_path.is_symlink()
    ):
        raise DatasetProvenanceError(
            "dataset_canonical_storage_missing",
            "canonical dataset, raw tree, or provenance is missing",
        )
    if (
        expected_dataset_root is not None
        and expected_dataset_root.resolve(strict=False) != dataset_root.resolve()
    ):
        raise DatasetProvenanceError(
            "dataset_runner_root_mismatch",
            "runner dataset root is not the canonical V2 dataset root",
        )
    if (
        expected_raw_root is not None
        and expected_raw_root.resolve(strict=False) != raw_root.resolve()
    ):
        raise DatasetProvenanceError(
            "dataset_runner_raw_root_mismatch",
            "runner raw root is not the canonical V2 raw tree",
        )

    provenance, provenance_content = _read_json_object(provenance_path)
    _validate_schema(
        provenance,
        schema_root / "dataset-provenance.schema.json",
        code="dataset_provenance_v2_schema_invalid",
    )
    provenance_sha256 = _sha256_bytes(provenance_content)
    if (
        provenance_sha256 != entry["provenance_sha256"]
        or provenance.get("schema_version") != PROVENANCE_SCHEMA_VERSION
        or provenance.get("dataset") != {"id": dataset_id, "version": dataset_version}
        or provenance.get("canonical_storage", {}).get("workspace_relative_path")
        != entry["canonical_dataset_workspace_relative_path"]
        or provenance.get("canonical_storage", {}).get("raw_workspace_relative_path")
        != entry["canonical_raw_workspace_relative_path"]
        or provenance.get("migration", {}).get("migration_id") != MIGRATION_ID
    ):
        raise DatasetProvenanceError(
            "dataset_provenance_v2_hash_or_identity_mismatch",
            "canonical DatasetProvenanceV2 does not match the relocation binding",
        )

    integrity = entry["raw_integrity"]
    provenance_integrity = provenance.get("integrity")
    if not isinstance(provenance_integrity, dict) or any(
        provenance_integrity.get(key) != value for key, value in integrity.items()
    ):
        raise DatasetProvenanceError(
            "dataset_provenance_integrity_binding_mismatch",
            "canonical provenance integrity does not match the relocation binding",
        )
    checksums_path = _safe_workspace_path(
        workspace,
        integrity["raw_checksums_workspace_relative_path"],
    )
    inventory_path = _safe_workspace_path(
        workspace,
        integrity["inventory_workspace_relative_path"],
    )
    for path in (checksums_path, inventory_path):
        _assert_no_symlink_components(workspace, path)
    if (
        _sha256_file(checksums_path) != integrity["raw_checksums_sha256"]
        or _sha256_file(inventory_path) != integrity["inventory_sha256"]
    ):
        raise DatasetProvenanceError(
            "dataset_canonical_integrity_file_mismatch",
            "canonical checksum or inventory evidence differs from the binding",
        )
    raw_file_count, raw_bytes = _verify_raw_checksum_manifest(
        dataset_root,
        raw_root,
        checksums_path,
        supplemental_readme_sha256=provenance_integrity.get("supplemental_readme_sha256"),
    )
    if raw_file_count != integrity["raw_file_count"] or raw_bytes != integrity["raw_bytes"]:
        raise DatasetProvenanceError(
            "dataset_canonical_raw_inventory_mismatch",
            "canonical raw file count or byte total differs from the binding",
        )

    for legacy_key in (
        "legacy_source_workspace_relative_path",
        "legacy_canonical_workspace_relative_path",
    ):
        legacy_path = _safe_workspace_path(workspace, entry[legacy_key])
        _assert_no_symlink_components(workspace, legacy_path)
        if legacy_path.exists():
            raise DatasetProvenanceError(
                "dataset_legacy_path_not_retired",
                "a retired legacy dataset path still exists",
            )

    return VerifiedDatasetProvenance(
        dataset_id=dataset_id,
        dataset_version=dataset_version,
        workspace_root=workspace,
        dataset_root=dataset_root,
        raw_root=raw_root,
        provenance_path=provenance_path,
        relocation_manifest_path=manifest_path,
        binding_sha256=binding_sha256,
        provenance_sha256=provenance_sha256,
    )


def build_relocation_manifest(
    repository_root: Path,
    workspace_root: Path,
    *,
    verified_at: str,
) -> dict[str, Any]:
    """Build final relocation evidence only after canonical data is complete."""

    repository = repository_root.resolve(strict=True)
    workspace = workspace_root.resolve(strict=True)
    _validate_manifest_timestamp(verified_at)
    binding_path = repository / DEFAULT_BINDING_RELATIVE_PATH
    binding, binding_content = _read_json_object(binding_path)
    _validate_schema(
        binding,
        binding_path.parent / "schemas" / "relocation-binding.schema.json",
        code="dataset_relocation_binding_schema_invalid",
    )
    binding_sha256 = _sha256_bytes(binding_content)
    manifest_entries: list[dict[str, Any]] = []
    for entry in binding["datasets"]:
        dataset_root = _safe_workspace_path(
            workspace,
            entry["canonical_dataset_workspace_relative_path"],
        )
        raw_root = _safe_workspace_path(
            workspace,
            entry["canonical_raw_workspace_relative_path"],
        )
        provenance_path = _safe_workspace_path(
            workspace,
            entry["provenance_workspace_relative_path"],
        )
        checksums_path = _safe_workspace_path(
            workspace,
            entry["raw_integrity"]["raw_checksums_workspace_relative_path"],
        )
        inventory_path = _safe_workspace_path(
            workspace,
            entry["raw_integrity"]["inventory_workspace_relative_path"],
        )
        for path in (dataset_root, raw_root, provenance_path, checksums_path, inventory_path):
            _assert_no_symlink_components(workspace, path)
        if not dataset_root.is_dir() or dataset_root.is_symlink() or not raw_root.is_dir():
            raise DatasetProvenanceError(
                "dataset_canonical_storage_missing",
                "canonical dataset is incomplete",
            )
        provenance, provenance_content = _read_json_object(provenance_path)
        _validate_schema(
            provenance,
            binding_path.parent / "schemas" / "dataset-provenance.schema.json",
            code="dataset_provenance_v2_schema_invalid",
        )
        integrity = entry["raw_integrity"]
        provenance_integrity = provenance.get("integrity", {})
        if (
            _sha256_bytes(provenance_content) != entry["provenance_sha256"]
            or provenance.get("dataset")
            != {"id": entry["dataset_id"], "version": entry["dataset_version"]}
            or any(provenance_integrity.get(key) != value for key, value in integrity.items())
            or _sha256_file(checksums_path) != integrity["raw_checksums_sha256"]
            or _sha256_file(inventory_path) != integrity["inventory_sha256"]
        ):
            raise DatasetProvenanceError(
                "dataset_canonical_integrity_binding_mismatch",
                "canonical data differs from the relocation binding",
            )
        raw_file_count, raw_bytes = _verify_raw_checksum_manifest(
            dataset_root,
            raw_root,
            checksums_path,
            supplemental_readme_sha256=provenance_integrity.get("supplemental_readme_sha256"),
        )
        if raw_file_count != integrity["raw_file_count"] or raw_bytes != integrity["raw_bytes"]:
            raise DatasetProvenanceError(
                "dataset_canonical_integrity_binding_mismatch",
                "canonical data differs from the relocation binding",
            )
        for legacy_key in (
            "legacy_source_workspace_relative_path",
            "legacy_canonical_workspace_relative_path",
        ):
            if _safe_workspace_path(workspace, entry[legacy_key]).exists():
                raise DatasetProvenanceError(
                    "dataset_legacy_path_not_retired",
                    "relocation manifest cannot be built while a legacy path exists",
                )
        manifest_entries.append(
            {
                "dataset_id": entry["dataset_id"],
                "dataset_version": entry["dataset_version"],
                "canonical_dataset_workspace_relative_path": entry[
                    "canonical_dataset_workspace_relative_path"
                ],
                "canonical_raw_workspace_relative_path": entry[
                    "canonical_raw_workspace_relative_path"
                ],
                "provenance_sha256": entry["provenance_sha256"],
                "raw_file_count": integrity["raw_file_count"],
                "raw_bytes": integrity["raw_bytes"],
                "raw_checksums_sha256": integrity["raw_checksums_sha256"],
                "inventory_sha256": integrity["inventory_sha256"],
                "legacy_source_reference_removed": True,
                "legacy_canonical_registration_removed": True,
                "canonical_hash_verification_passed": True,
            }
        )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "migration_id": MIGRATION_ID,
        "relocation_status": "completed_and_verified",
        "verified_at": verified_at,
        "binding_repository_relative_path": DEFAULT_BINDING_RELATIVE_PATH.as_posix(),
        "binding_sha256": binding_sha256,
        "datasets": manifest_entries,
    }
    _validate_schema(
        manifest,
        binding_path.parent / "schemas" / "relocation-manifest.schema.json",
        code="dataset_relocation_manifest_schema_invalid",
    )
    return manifest


def write_relocation_manifest(
    repository_root: Path,
    workspace_root: Path,
    *,
    verified_at: str,
) -> Path:
    manifest = build_relocation_manifest(
        repository_root,
        workspace_root,
        verified_at=verified_at,
    )
    workspace = workspace_root.resolve(strict=True)
    output = _safe_workspace_path(
        workspace,
        DEFAULT_MANIFEST_WORKSPACE_RELATIVE_PATH.as_posix(),
    )
    content = (json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode(
        "utf-8"
    )
    if output.exists():
        if output.is_symlink() or not output.is_file() or output.read_bytes() != content:
            raise DatasetProvenanceError(
                "dataset_relocation_manifest_overwrite_forbidden",
                "existing relocation manifest differs and cannot be overwritten",
            )
        return output
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=output.parent,
            prefix=f".{output.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(content)
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return output


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Manage workspace DatasetProvenanceV2.")
    subparsers = parser.add_subparsers(dest="command", required=True)
    verify = subparsers.add_parser("verify")
    verify.add_argument("--dataset-lock", required=True, type=Path)
    verify.add_argument("--workspace-root", type=Path)
    build = subparsers.add_parser("build-manifest")
    build.add_argument("--repository-root", required=True, type=Path)
    build.add_argument("--workspace-root", required=True, type=Path)
    build.add_argument("--verified-at", required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.command == "build-manifest":
            path = write_relocation_manifest(
                args.repository_root,
                args.workspace_root,
                verified_at=args.verified_at,
            )
            print(f"dataset relocation manifest built: {path.name}")
            return 0
        result = verify_dataset_provenance(args.dataset_lock, workspace_root=args.workspace_root)
    except DatasetProvenanceError as exc:
        print(f"dataset provenance verification failed: {exc.code}")
        return 1
    print(
        "dataset provenance verification passed: "
        f"{result.dataset_id}@{result.dataset_version} {result.binding_sha256}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
