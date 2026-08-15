"""Build one local-only PDF-preview runtime inventory from staged assets.

This utility deliberately has a narrow job: it turns an already-downloaded,
already-local model tree plus operator-supplied provenance terms into the
per-file inventory consumed by ``provision_document_preview_runtime.py``.
It never downloads packages or models, imports an OCR engine, reads a PDF, or
prints asset content.  The network guard is also active while it runs so an
unexpected dependency cannot silently contact a registry.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import stat
from pathlib import Path
from typing import Callable, Mapping, Sequence, cast

from customs_ai_agent.service import document_preview_runtime as runtime_module
from customs_ai_agent.service.document_preview_runtime import (
    DocumentPreviewRuntimeError,
    RuntimeConfigurationV1,
    local_files_only,
    parse_model_inventory,
    parse_runtime_configuration,
    validate_document_preview_root,
    verify_configured_model_directory_artifacts,
    verify_inventory_packages,
)


_MAX_LOCAL_JSON_BYTES = 1024 * 1024
_MAX_ARTIFACT_BYTES = 8 * 1024**3
_STAGING_DIRECTORY = "staging"
_PROVENANCE_SCHEMA_VERSION = "1.0"


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _normalized_distribution_name(value: str) -> str:
    return value.casefold().replace("_", "-")


def _is_link_or_reparse_point(metadata: os.stat_result) -> bool:
    reparse_point = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    attributes = getattr(metadata, "st_file_attributes", 0)
    return stat.S_ISLNK(metadata.st_mode) or bool(reparse_point and attributes & reparse_point)


def _absolute_path(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _path_under_root(
    *,
    root: Path,
    path: Path,
    must_exist: bool,
    code: str,
) -> Path:
    """Return one non-link path under ``root`` without following a redirect.

    ``Path.resolve`` alone is insufficient on Windows because a junction can
    move a local model directory outside the inventory boundary.  Walk every
    existing component with ``lstat`` first, then resolve only to confirm the
    resulting location remains under the approved root.
    """

    candidate = _absolute_path(path)
    try:
        relative = candidate.relative_to(root)
    except ValueError as error:
        raise DocumentPreviewRuntimeError(code) from error
    current = root
    for part in relative.parts:
        current /= part
        try:
            metadata = current.lstat()
        except FileNotFoundError:
            if must_exist:
                raise DocumentPreviewRuntimeError(code) from None
            break
        except OSError as error:
            raise DocumentPreviewRuntimeError(code) from error
        if _is_link_or_reparse_point(metadata):
            raise DocumentPreviewRuntimeError(code)
    try:
        resolved = candidate.resolve(strict=must_exist)
        resolved.relative_to(root)
    except (OSError, ValueError) as error:
        raise DocumentPreviewRuntimeError(code) from error
    return candidate


def _read_local_json_object(*, root: Path, path: Path, code: str) -> dict[str, object]:
    candidate = _path_under_root(root=root, path=path, must_exist=True, code=code)
    try:
        metadata = candidate.lstat()
        if (
            _is_link_or_reparse_point(metadata)
            or not stat.S_ISREG(metadata.st_mode)
            or metadata.st_size < 1
            or metadata.st_size > _MAX_LOCAL_JSON_BYTES
        ):
            raise DocumentPreviewRuntimeError(code)
        raw = candidate.read_bytes()
    except DocumentPreviewRuntimeError:
        raise
    except OSError as error:
        raise DocumentPreviewRuntimeError(code) from error
    if len(raw) != metadata.st_size:
        raise DocumentPreviewRuntimeError(code)
    try:
        value = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DocumentPreviewRuntimeError(code) from error
    if not isinstance(value, dict):
        raise DocumentPreviewRuntimeError(code)
    return cast(dict[str, object], value)


def _strict_mapping(value: object, *, expected: set[str], code: str) -> dict[str, object]:
    if not isinstance(value, dict) or set(value) != expected:
        raise DocumentPreviewRuntimeError(code)
    return cast(dict[str, object], value)


def _configured_model_directories(configuration: RuntimeConfigurationV1) -> tuple[str, ...]:
    directories = [
        value
        for key, value in configuration.options.items()
        if key.endswith("_directory") and isinstance(value, str)
    ]
    return tuple(sorted(set(directories)))


def _is_nested_directory(parent: str, child: str) -> bool:
    parent_parts = Path(parent).parts
    child_parts = Path(child).parts
    return len(child_parts) > len(parent_parts) and child_parts[: len(parent_parts)] == parent_parts


def _parse_provenance(
    *,
    value: Mapping[str, object],
    configuration: RuntimeConfigurationV1,
) -> tuple[dict[str, dict[str, str]], dict[str, dict[str, str]]]:
    code = "document_preview_inventory_provenance_invalid"
    provenance = _strict_mapping(
        value,
        expected={"schema_version", "engine_kind", "packages", "artifact_directories"},
        code=code,
    )
    if (
        provenance["schema_version"] != _PROVENANCE_SCHEMA_VERSION
        or provenance["engine_kind"] != configuration.engine_kind
        or not isinstance(provenance["packages"], list)
        or not isinstance(provenance["artifact_directories"], list)
    ):
        raise DocumentPreviewRuntimeError(code)

    required = {
        _normalized_distribution_name(name): name
        for name in runtime_module._required_packages(configuration.engine_kind)
    }
    packages: dict[str, dict[str, str]] = {}
    for raw_item in provenance["packages"]:
        item = _strict_mapping(
            raw_item,
            expected={"name", "license_id", "source_uri"},
            code=code,
        )
        name = item["name"]
        license_id = item["license_id"]
        source_uri = item["source_uri"]
        if not all(isinstance(part, str) for part in (name, license_id, source_uri)):
            raise DocumentPreviewRuntimeError(code)
        normalized = _normalized_distribution_name(name)
        if normalized not in required or normalized in packages:
            raise DocumentPreviewRuntimeError(code)
        packages[normalized] = {"license_id": license_id, "source_uri": source_uri}
    if set(packages) != set(required):
        raise DocumentPreviewRuntimeError(code)

    configured_directories = _configured_model_directories(configuration)
    if any(
        _is_nested_directory(parent, child)
        for parent in configured_directories
        for child in configured_directories
        if parent != child
    ):
        raise DocumentPreviewRuntimeError(code)
    artifact_directories: dict[str, dict[str, str]] = {}
    for raw_item in provenance["artifact_directories"]:
        item = _strict_mapping(
            raw_item,
            expected={"relative_directory", "license_id", "source_uri"},
            code=code,
        )
        relative_directory = item["relative_directory"]
        license_id = item["license_id"]
        source_uri = item["source_uri"]
        if not all(isinstance(part, str) for part in (relative_directory, license_id, source_uri)):
            raise DocumentPreviewRuntimeError(code)
        if (
            relative_directory not in configured_directories
            or relative_directory in artifact_directories
        ):
            raise DocumentPreviewRuntimeError(code)
        artifact_directories[relative_directory] = {
            "license_id": license_id,
            "source_uri": source_uri,
        }
    if set(artifact_directories) != set(configured_directories):
        raise DocumentPreviewRuntimeError(code)
    return packages, artifact_directories


def _configured_model_directory(*, root: Path, relative_directory: str) -> Path:
    path = root.joinpath(*Path(relative_directory).parts)
    candidate = _path_under_root(
        root=root,
        path=path,
        must_exist=True,
        code="document_preview_model_artifact_invalid",
    )
    try:
        metadata = candidate.lstat()
    except OSError as error:
        raise DocumentPreviewRuntimeError("document_preview_model_artifact_invalid") from error
    if _is_link_or_reparse_point(metadata) or not stat.S_ISDIR(metadata.st_mode):
        raise DocumentPreviewRuntimeError("document_preview_model_artifact_invalid")
    return candidate


def _walk_regular_files(*, root: Path, directory: Path) -> tuple[Path, ...]:
    """Enumerate every regular file below one approved model directory."""

    discovered: list[Path] = []
    pending = [directory]
    while pending:
        current = pending.pop()
        try:
            children = tuple(current.iterdir())
        except OSError as error:
            raise DocumentPreviewRuntimeError("document_preview_model_artifact_invalid") from error
        for child in children:
            try:
                metadata = child.lstat()
            except OSError as error:
                raise DocumentPreviewRuntimeError(
                    "document_preview_model_artifact_invalid"
                ) from error
            if _is_link_or_reparse_point(metadata):
                raise DocumentPreviewRuntimeError("document_preview_model_artifact_invalid")
            if stat.S_ISDIR(metadata.st_mode):
                pending.append(child)
                continue
            if not stat.S_ISREG(metadata.st_mode):
                raise DocumentPreviewRuntimeError("document_preview_model_artifact_invalid")
            if not 1 <= metadata.st_size <= _MAX_ARTIFACT_BYTES:
                raise DocumentPreviewRuntimeError("document_preview_model_artifact_invalid")
            _path_under_root(
                root=root,
                path=child,
                must_exist=True,
                code="document_preview_model_artifact_invalid",
            )
            discovered.append(child)
    return tuple(sorted(discovered, key=lambda item: item.relative_to(root).as_posix()))


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise DocumentPreviewRuntimeError("document_preview_model_artifact_invalid") from error
    return digest.hexdigest()


def _build_artifacts(
    *,
    root: Path,
    artifact_directory_provenance: Mapping[str, Mapping[str, str]],
) -> list[dict[str, object]]:
    artifacts: list[dict[str, object]] = []
    seen_paths: set[str] = set()
    for relative_directory in sorted(artifact_directory_provenance):
        directory = _configured_model_directory(root=root, relative_directory=relative_directory)
        provenance = artifact_directory_provenance[relative_directory]
        for path in _walk_regular_files(root=root, directory=directory):
            try:
                relative_path = path.relative_to(root).as_posix()
                metadata = path.stat()
            except OSError as error:
                raise DocumentPreviewRuntimeError(
                    "document_preview_model_artifact_invalid"
                ) from error
            if relative_path in seen_paths or not stat.S_ISREG(metadata.st_mode):
                raise DocumentPreviewRuntimeError("document_preview_model_artifact_invalid")
            seen_paths.add(relative_path)
            artifacts.append(
                {
                    "artifact_id": f"artifact-{_sha256_bytes(relative_path.encode('utf-8'))}",
                    "relative_path": relative_path,
                    "sha256": _hash_file(path),
                    "byte_size": metadata.st_size,
                    "license_id": provenance["license_id"],
                    "source_uri": provenance["source_uri"],
                }
            )
    return sorted(artifacts, key=lambda item: cast(str, item["relative_path"]))


def _package_version(
    name: str,
    *,
    package_version_reader: Callable[[str], str],
) -> str:
    try:
        version = package_version_reader(name)
    except Exception as error:
        raise DocumentPreviewRuntimeError(
            "document_preview_runtime_dependency_unavailable"
        ) from error
    if not isinstance(version, str):
        raise DocumentPreviewRuntimeError("document_preview_runtime_dependency_unavailable")
    return version


def build_runtime_inventory(
    *,
    document_preview_root: Path,
    configuration_path: Path,
    provenance_path: Path,
    package_version_reader: Callable[[str], str] = importlib.metadata.version,
) -> tuple[dict[str, object], str]:
    """Return a validated inventory and the canonical configuration SHA-256.

    The function only reads files already below ``document_preview_root``.  It
    deliberately returns data rather than writing sidecars; the caller chooses
    the mutable staging file, and the existing provision command remains the
    sole writer for immutable runtime sidecars.
    """

    root = validate_document_preview_root(document_preview_root, create=False)
    with local_files_only():
        configuration_value = _read_local_json_object(
            root=root,
            path=configuration_path,
            code="document_preview_inventory_configuration_invalid",
        )
        configuration_raw = _canonical_json_bytes(configuration_value)
        try:
            configuration = parse_runtime_configuration(configuration_raw)
        except DocumentPreviewRuntimeError as error:
            raise DocumentPreviewRuntimeError(
                "document_preview_inventory_configuration_invalid"
            ) from error
        provenance_value = _read_local_json_object(
            root=root,
            path=provenance_path,
            code="document_preview_inventory_provenance_invalid",
        )
        package_provenance, artifact_directory_provenance = _parse_provenance(
            value=provenance_value,
            configuration=configuration,
        )
        required = runtime_module._required_packages(configuration.engine_kind)
        packages = [
            {
                "name": name,
                "version": _package_version(
                    name,
                    package_version_reader=package_version_reader,
                ),
                "license_id": package_provenance[_normalized_distribution_name(name)]["license_id"],
                "source_uri": package_provenance[_normalized_distribution_name(name)]["source_uri"],
            }
            for name in sorted(required)
        ]
        inventory_value: dict[str, object] = {
            "schema_version": "1.0",
            "engine_kind": configuration.engine_kind,
            "packages": packages,
            "artifacts": _build_artifacts(
                root=root,
                artifact_directory_provenance=artifact_directory_provenance,
            ),
        }
        inventory_raw = _canonical_json_bytes(inventory_value)
        try:
            inventory = parse_model_inventory(inventory_raw)
            if inventory.engine_kind != configuration.engine_kind:
                raise DocumentPreviewRuntimeError("document_preview_provisioning_kind_mismatch")
            verify_inventory_packages(inventory, package_version_reader=package_version_reader)
            verify_configured_model_directory_artifacts(root, configuration, inventory)
        except DocumentPreviewRuntimeError:
            raise
        return inventory_value, _sha256_bytes(configuration_raw)


def _staging_inventory_path(*, root: Path, output_path: Path) -> Path:
    staging = root / _STAGING_DIRECTORY
    try:
        staging.mkdir(exist_ok=True)
    except OSError as error:
        raise DocumentPreviewRuntimeError("document_preview_inventory_write_failed") from error
    _path_under_root(
        root=root,
        path=staging,
        must_exist=True,
        code="document_preview_inventory_write_failed",
    )
    candidate = _path_under_root(
        root=root,
        path=output_path,
        must_exist=False,
        code="document_preview_inventory_write_failed",
    )
    if (
        candidate.parent != staging
        or candidate.suffix != ".json"
        or not candidate.stem.startswith("inventory")
    ):
        raise DocumentPreviewRuntimeError("document_preview_inventory_write_failed")
    try:
        metadata = candidate.lstat()
    except FileNotFoundError:
        return candidate
    except OSError as error:
        raise DocumentPreviewRuntimeError("document_preview_inventory_write_failed") from error
    if _is_link_or_reparse_point(metadata) or not stat.S_ISREG(metadata.st_mode):
        raise DocumentPreviewRuntimeError("document_preview_inventory_write_failed")
    return candidate


def write_staging_inventory(
    *,
    root: Path,
    output_path: Path,
    inventory_value: Mapping[str, object],
    protected_paths: Sequence[Path] = (),
) -> str:
    """Atomically replace one mutable staging inventory without touching config."""

    destination = _staging_inventory_path(root=root, output_path=output_path)
    if any(_absolute_path(path) == destination for path in protected_paths):
        raise DocumentPreviewRuntimeError("document_preview_inventory_write_failed")
    payload = _canonical_json_bytes(inventory_value)
    temporary = destination.parent / f".{destination.name}.{os.getpid()}.tmp"
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
    except OSError as error:
        raise DocumentPreviewRuntimeError("document_preview_inventory_write_failed") from error
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass
    return _sha256_bytes(payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build a local PDF-preview inventory from pre-provisioned files; "
            "this command never downloads or contacts a network service."
        )
    )
    parser.add_argument("--document-preview-root", type=Path, required=True)
    parser.add_argument(
        "--configuration",
        type=Path,
        required=True,
        help="Strict configuration JSON already below document-preview root; it is never rewritten.",
    )
    parser.add_argument(
        "--provenance",
        type=Path,
        required=True,
        help="Explicit package and configured-model-directory provenance JSON under root.",
    )
    parser.add_argument(
        "--inventory-output",
        type=Path,
        help="Optional direct staging/*.json output; defaults to staging/inventory.json.",
    )
    parser.add_argument(
        "--confirm-local-build",
        action="store_true",
        help="Required acknowledgement that packages/models were explicitly provisioned locally.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if not arguments.confirm_local_build:
        print(
            json.dumps({"status": "failed", "code": "confirmation_required"}, separators=(",", ":"))
        )
        return 2
    try:
        root = validate_document_preview_root(arguments.document_preview_root, create=False)
        inventory_value, configuration_sha256 = build_runtime_inventory(
            document_preview_root=root,
            configuration_path=arguments.configuration,
            provenance_path=arguments.provenance,
        )
        output_path = arguments.inventory_output or root / _STAGING_DIRECTORY / "inventory.json"
        inventory_sha256 = write_staging_inventory(
            root=root,
            output_path=output_path,
            inventory_value=inventory_value,
            protected_paths=(arguments.configuration, arguments.provenance),
        )
    except DocumentPreviewRuntimeError as error:
        print(json.dumps({"status": "failed", "code": error.code}, separators=(",", ":")))
        return 2
    print(
        json.dumps(
            {
                "status": "built",
                "configuration_sha256": configuration_sha256,
                "model_inventory_sha256": inventory_sha256,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
