"""Export the minimal verified Linux Raster Text OCR OCI build context.

The command does not download, install, train, activate, or repair anything.
It accepts only an already activated schema-12 Linux/amd64 profile, derives
SPDX 2.3 and license metadata from its hash-bound package/model/CPython
evidence, and publishes a new immutable directory only after the complete
exported closure passes runtime validation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import stat
import sys
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

from customs_ai_agent.service.raster_text_preview_runtime import (
    RasterTextActivationPointerV12,
    RasterTextOfflineProfileV19,
    _load_binding_from_pointer,
    parse_raster_text_activation_pointer,
    parse_raster_text_offline_profile,
    validate_raster_text_approval_pointer_path,
)
from raster_text_oci_evidence import verify_v9_direct_evidence_closure
from raster_text_oci_metadata import (
    _POLICY_RELATIVE_PATH,
    RuntimeMetadataError,
    build_runtime_metadata,
)
from verify_raster_text_oci_runtime import (
    RuntimeContextError,
    verify_runtime_context,
)

_POINTER_RELATIVE_PATH = PurePosixPath("approvals/raster-text-active.json")
_MANIFEST_RELATIVE_PATH = PurePosixPath("_deployment/raster-text-runtime-oci-v1.json")
_SBOM_RELATIVE_PATH = PurePosixPath("_deployment/runtime.spdx.json")
_LICENSE_RELATIVE_PATH = PurePosixPath("_deployment/third-party-licenses.json")
_POLICY_SOURCE_PATH = (
    Path(__file__).resolve().parents[1] / "deploy" / "ocr-oci" / "runtime-license-policy.json"
)
_MAX_FILE_COUNT = 25_000
_MAX_TREE_ENTRIES = 50_000


class RuntimeExportError(RuntimeError):
    """A content-free export failure."""

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
        raise RuntimeExportError("runtime_source_file_unreadable") from None
    return digest.hexdigest(), size


def _safe_relative(value: str) -> PurePosixPath:
    if not value or "\\" in value:
        raise RuntimeExportError("runtime_source_path_invalid")
    result = PurePosixPath(value)
    if result.is_absolute() or result == PurePosixPath(".") or ".." in result.parts:
        raise RuntimeExportError("runtime_source_path_invalid")
    return result


def _safe_source_path(root: Path, relative: PurePosixPath) -> Path:
    current = root
    try:
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise OSError
        resolved = current.resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise OSError
    except OSError:
        raise RuntimeExportError("runtime_source_file_invalid") from None
    return resolved


def _directory_files(root: Path, relative: PurePosixPath) -> set[PurePosixPath]:
    directory = root.joinpath(*relative.parts)
    try:
        if directory.is_symlink() or not directory.is_dir():
            raise OSError
        resolved = directory.resolve(strict=True)
        if not resolved.is_relative_to(root):
            raise OSError
    except OSError:
        raise RuntimeExportError("runtime_source_directory_invalid") from None
    files: set[PurePosixPath] = set()
    pending = [resolved]
    entries_seen = 0
    try:
        while pending:
            directory = pending.pop()
            with os.scandir(directory) as entries:
                for entry in entries:
                    entries_seen += 1
                    if entries_seen > _MAX_TREE_ENTRIES:
                        raise RuntimeExportError("runtime_source_tree_limit_exceeded")
                    candidate = Path(entry.path)
                    if entry.is_symlink():
                        raise RuntimeExportError("runtime_source_symlink_forbidden")
                    if entry.is_dir(follow_symlinks=False):
                        pending.append(candidate)
                    elif entry.is_file(follow_symlinks=False):
                        files.add(PurePosixPath(candidate.relative_to(root).as_posix()))
                        if len(files) > _MAX_FILE_COUNT:
                            raise RuntimeExportError("runtime_source_tree_limit_exceeded")
                    else:
                        raise RuntimeExportError("runtime_source_special_file_forbidden")
    except RuntimeExportError:
        raise
    except (OSError, ValueError):
        raise RuntimeExportError("runtime_source_directory_invalid") from None
    if not files:
        raise RuntimeExportError("runtime_source_directory_empty")
    return files


def _assert_linux_amd64_profile(root: Path, profile: RasterTextOfflineProfileV19) -> None:
    machine = platform.machine().casefold()
    if platform.system() != "Linux" or machine not in {"amd64", "x86_64"}:
        raise RuntimeExportError("linux_amd64_builder_required")
    if sys.implementation.name != "cpython" or sys.version_info[:3] != (3, 11, 15):
        # Parent-only packages are observed from this interpreter environment.
        # A different ABI could attest metadata for packages that the cp311
        # final image never transports.
        raise RuntimeExportError("linux_cp311_builder_required")
    interpreter_relative = _safe_relative(profile.sidecar_interpreter_path)
    if interpreter_relative.as_posix().split("/")[-3:] != [
        "interpreter",
        "bin",
        "python3.11",
    ]:
        raise RuntimeExportError("linux_cp311_sidecar_required")
    interpreter = _safe_source_path(root, interpreter_relative)
    try:
        header = interpreter.read_bytes()[:20]
    except OSError:
        raise RuntimeExportError("linux_cp311_sidecar_required") from None
    # ELF64, little-endian, x86-64 (e_machine 62).
    if len(header) < 20 or header[:6] != b"\x7fELF\x02\x01" or header[18:20] != b"\x3e\x00":
        raise RuntimeExportError("linux_cp311_sidecar_required")
    if not os.access(interpreter, os.X_OK):
        raise RuntimeExportError("linux_cp311_sidecar_required")


def _load_binding(
    *,
    root: Path,
    approval_path: Path,
    approval_raw: bytes,
    approval_sha256: str,
    verification_mode: str,
) -> Any:
    binding = None
    try:
        binding = _load_binding_from_pointer(
            root=root,
            approval_path=approval_path,
            approval_raw=approval_raw,
            approval_sha256=approval_sha256,
            start_service_session=False,
            verification_mode=verification_mode,
        )
        verify_v9_direct_evidence_closure(root=root, pointer=binding.pointer)
        return binding
    except Exception:
        raise RuntimeExportError(
            "activation_verification_failed"
            if verification_mode == "activation"
            else "compact_runtime_verification_failed"
        ) from None
    finally:
        if binding is not None:
            binding.close()


def _record_compact_runtime_reads(
    *, root: Path, approval_path: Path, approval_raw: bytes, approval_sha256: str
) -> set[PurePosixPath]:
    opened: set[PurePosixPath] = set()

    def audit(event: str, args: tuple[object, ...]) -> None:
        if event != "open" or not args or not isinstance(args[0], (str, bytes, os.PathLike)):
            return
        try:
            candidate = Path(os.fsdecode(args[0])).resolve(strict=False)
            if candidate.is_relative_to(root) and candidate.is_file():
                opened.add(PurePosixPath(candidate.relative_to(root).as_posix()))
        except (OSError, TypeError, ValueError):
            return

    sys.addaudithook(audit)
    _load_binding(
        root=root,
        approval_path=approval_path,
        approval_raw=approval_raw,
        approval_sha256=approval_sha256,
        verification_mode="runtime",
    )
    return opened


def _is_beneath(relative: PurePosixPath, parent: PurePosixPath) -> bool:
    try:
        relative.relative_to(parent)
        return True
    except ValueError:
        return False


def _role_for(
    relative: PurePosixPath,
    *,
    runtime_profile: PurePosixPath,
    sidecar_root: PurePosixPath,
    model_bundle_root: PurePosixPath,
    component_evidence: frozenset[PurePosixPath],
) -> str:
    if relative == _POINTER_RELATIVE_PATH:
        return "active-pointer"
    if relative == runtime_profile:
        return "runtime-profile"
    if relative in component_evidence:
        return "component-evidence"
    if _is_beneath(relative, sidecar_root):
        suffix = relative.relative_to(sidecar_root)
        return (
            "inventory"
            if "inventories" in suffix.parts or suffix.name == "manifest.json"
            else "sidecar-runtime"
        )
    if _is_beneath(relative, model_bundle_root):
        suffix = relative.relative_to(model_bundle_root)
        if (
            "inventories" in suffix.parts
            or "provenance" in suffix.parts
            or suffix.name == "manifest.json"
        ):
            return "inventory"
        return "model"
    return "activation-evidence"


def _copy_one(source_root: Path, output_root: Path, relative: PurePosixPath) -> None:
    source = _safe_source_path(source_root, relative)
    destination = output_root.joinpath(*relative.parts)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        source_mode = source.stat().st_mode
        shutil.copy2(source, destination, follow_symlinks=False)
        executable = source_mode & (stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        destination.chmod(0o555 if executable else 0o444)
    except OSError:
        raise RuntimeExportError("runtime_copy_failed") from None


def _write_metadata(payload: bytes, output_root: Path, relative: PurePosixPath) -> None:
    try:
        destination = output_root.joinpath(*relative.parts)
        destination.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(str(destination), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        destination.chmod(0o444)
    except OSError:
        raise RuntimeExportError("deployment_metadata_write_failed") from None


def _metadata_payload(source: Path | None, generated: bytes) -> bytes:
    if source is None:
        return generated
    try:
        if source.is_symlink() or not source.is_file() or source.stat().st_size != len(generated):
            raise OSError
        raw = source.read_bytes()
    except OSError:
        raise RuntimeExportError("deployment_metadata_input_invalid") from None
    if raw != generated:
        raise RuntimeExportError("deployment_metadata_input_mismatch")
    return raw


def _make_runtime_tree_nonroot_readable(root: Path) -> None:
    try:
        directories = [root, *(path for path in root.rglob("*") if path.is_dir())]
        for directory in directories:
            if directory.is_symlink():
                raise OSError
            directory.chmod(0o555)
    except OSError:
        raise RuntimeExportError("runtime_permissions_normalization_failed") from None


def _file_record(path: Path, relative: PurePosixPath, role: str) -> dict[str, object]:
    digest, size = _sha256(path)
    return {
        "byte_size": size,
        "relative_path": relative.as_posix(),
        "role": role,
        "sha256": digest,
    }


def _manifest(
    *,
    output_root: Path,
    source_relatives: Iterable[PurePosixPath],
    runtime_profile: PurePosixPath,
    sidecar_root: PurePosixPath,
    model_bundle_root: PurePosixPath,
    pointer: RasterTextActivationPointerV12,
    component_evidence: frozenset[PurePosixPath],
    generated_evidence: frozenset[PurePosixPath],
) -> dict[str, object]:
    records = [
        _file_record(
            output_root.joinpath(*relative.parts),
            relative,
            _role_for(
                relative,
                runtime_profile=runtime_profile,
                sidecar_root=sidecar_root,
                model_bundle_root=model_bundle_root,
                component_evidence=component_evidence,
            ),
        )
        for relative in source_relatives
    ]
    records.extend(
        (
            _file_record(
                output_root.joinpath(*_SBOM_RELATIVE_PATH.parts),
                _SBOM_RELATIVE_PATH,
                "sbom",
            ),
            _file_record(
                output_root.joinpath(*_LICENSE_RELATIVE_PATH.parts),
                _LICENSE_RELATIVE_PATH,
                "licenses",
            ),
            _file_record(
                output_root.joinpath(*_POLICY_RELATIVE_PATH.parts),
                _POLICY_RELATIVE_PATH,
                "component-evidence",
            ),
        )
    )
    records.extend(
        _file_record(
            output_root.joinpath(*relative.parts),
            relative,
            "component-evidence",
        )
        for relative in generated_evidence
    )
    records.sort(key=lambda record: str(record["relative_path"]))
    by_path = {str(record["relative_path"]): record for record in records}
    pointer_record = by_path[_POINTER_RELATIVE_PATH.as_posix()]
    sbom_record = by_path[_SBOM_RELATIVE_PATH.as_posix()]
    license_record = by_path[_LICENSE_RELATIVE_PATH.as_posix()]
    return {
        "active_pointer": {
            "raster_text_profile_revision": pointer.raster_text_profile_revision,
            "relative_path": _POINTER_RELATIVE_PATH.as_posix(),
            "sha256": pointer_record["sha256"],
        },
        "artifact_kind": "customs-raster-text-runtime-oci",
        "files": records,
        "license_manifest": {
            "relative_path": _LICENSE_RELATIVE_PATH.as_posix(),
            "sha256": license_record["sha256"],
        },
        "platform": {"architecture": "amd64", "os": "linux", "python_abi": "cp311"},
        "runtime_binding_sha256": pointer.current_runner_implementation_sha256,
        "sbom": {
            "relative_path": _SBOM_RELATIVE_PATH.as_posix(),
            "sha256": sbom_record["sha256"],
        },
        "schema_version": "1.0",
    }


def export_runtime(
    *,
    source_root: Path,
    destination: Path,
    runtime_sbom: Path | None = None,
    runtime_license_manifest: Path | None = None,
) -> dict[str, object]:
    if not source_root.is_absolute() or not destination.is_absolute():
        raise RuntimeExportError("absolute_paths_required")
    try:
        root = source_root.resolve(strict=True)
        destination_parent = destination.parent.resolve(strict=True)
        destination_resolved = destination.resolve(strict=False)
    except OSError:
        raise RuntimeExportError("export_path_invalid") from None
    if (
        root.is_symlink()
        or destination_parent.is_symlink()
        or destination.exists()
        or destination_resolved.is_relative_to(root)
        or root.is_relative_to(destination_resolved)
    ):
        raise RuntimeExportError("export_path_invalid")

    approval_path = root.joinpath(*_POINTER_RELATIVE_PATH.parts)
    try:
        validated_root, validated_pointer = validate_raster_text_approval_pointer_path(
            approval_path
        )
        if validated_root != root or validated_pointer != approval_path:
            raise ValueError
        approval_raw = approval_path.read_bytes()
        approval_sha256 = hashlib.sha256(approval_raw).hexdigest()
        pointer = parse_raster_text_activation_pointer(approval_raw)
        if type(pointer) is not RasterTextActivationPointerV12:
            raise ValueError
        runtime_profile = _safe_relative(pointer.runtime_configuration_path)
        profile_raw = _safe_source_path(root, runtime_profile).read_bytes()
        profile = parse_raster_text_offline_profile(profile_raw)
        if type(profile) is not RasterTextOfflineProfileV19:
            raise ValueError
    except Exception:
        raise RuntimeExportError("schema12_active_pointer_required") from None
    _assert_linux_amd64_profile(root, profile)
    if (runtime_sbom is None) != (runtime_license_manifest is None):
        raise RuntimeExportError("deployment_metadata_inputs_incomplete")

    # The source must first pass the expensive, complete activation boundary.
    # Only the later compact-runtime observation determines exported evidence.
    _load_binding(
        root=root,
        approval_path=approval_path,
        approval_raw=approval_raw,
        approval_sha256=approval_sha256,
        verification_mode="activation",
    )
    source_relatives = _record_compact_runtime_reads(
        root=root,
        approval_path=approval_path,
        approval_raw=approval_raw,
        approval_sha256=approval_sha256,
    )
    source_relatives.add(_POINTER_RELATIVE_PATH)
    source_relatives.add(runtime_profile)
    sidecar_root = _safe_relative(profile.bundle_root_relative_path)
    model_bundle_root = _safe_relative(profile.mobile_model_bundle_root_relative_path)
    source_relatives.update(_directory_files(root, sidecar_root))
    source_relatives.update(_directory_files(root, model_bundle_root))
    try:
        generated_metadata = build_runtime_metadata(
            root,
            profile.model_dump(mode="json", warnings=False),
            policy_path=_POLICY_SOURCE_PATH,
        )
    except RuntimeMetadataError as exc:
        raise RuntimeExportError(exc.code) from None
    component_evidence = frozenset(generated_metadata.source_evidence_paths)
    generated_evidence = frozenset(
        relative for relative, _payload in generated_metadata.generated_evidence
    )
    source_relatives.update(component_evidence)
    # Account for SPDX, licenses, policy, and the trust-root deployment manifest.
    if len(source_relatives) > _MAX_FILE_COUNT - 4 - len(generated_evidence):
        raise RuntimeExportError("runtime_source_tree_limit_exceeded")
    sbom_payload = _metadata_payload(runtime_sbom, generated_metadata.sbom)
    license_payload = _metadata_payload(
        runtime_license_manifest, generated_metadata.license_manifest
    )

    stage = destination_parent / f".{destination.name}.staging-{uuid.uuid4().hex}"
    output_root = stage / "document-preview"
    try:
        stage.mkdir(mode=0o700)
        output_root.mkdir(mode=0o755)
        for relative in sorted(source_relatives, key=lambda item: item.as_posix()):
            _copy_one(root, output_root, relative)
        _write_metadata(sbom_payload, output_root, _SBOM_RELATIVE_PATH)
        _write_metadata(license_payload, output_root, _LICENSE_RELATIVE_PATH)
        _write_metadata(generated_metadata.policy, output_root, _POLICY_RELATIVE_PATH)
        for relative, payload in generated_metadata.generated_evidence:
            _write_metadata(payload, output_root, relative)

        # Re-run the real compact loader against only the staged closure before
        # adding the OCI manifest.  Missing evidence cannot be masked by the
        # exporter inventory.
        staged_pointer = output_root.joinpath(*_POINTER_RELATIVE_PATH.parts)
        staged_raw = staged_pointer.read_bytes()
        _load_binding(
            root=output_root,
            approval_path=staged_pointer,
            approval_raw=staged_raw,
            approval_sha256=hashlib.sha256(staged_raw).hexdigest(),
            verification_mode="runtime",
        )
        manifest = _manifest(
            output_root=output_root,
            source_relatives=sorted(source_relatives, key=lambda item: item.as_posix()),
            runtime_profile=runtime_profile,
            sidecar_root=sidecar_root,
            model_bundle_root=model_bundle_root,
            pointer=pointer,
            component_evidence=component_evidence,
            generated_evidence=generated_evidence,
        )
        manifest_path = output_root.joinpath(*_MANIFEST_RELATIVE_PATH.parts)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_bytes(_canonical_json_bytes(manifest))
        manifest_path.chmod(0o444)
        _make_runtime_tree_nonroot_readable(output_root)
        result = verify_runtime_context(output_root, expected_license_policy=_POLICY_SOURCE_PATH)
        os.replace(stage, destination)
        return result
    except RuntimeContextError as exc:
        raise RuntimeExportError(exc.code) from None
    except RuntimeExportError:
        raise
    except OSError:
        raise RuntimeExportError("runtime_export_publish_failed") from None
    finally:
        if stage.exists():
            shutil.rmtree(stage, ignore_errors=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-preview-root", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    parser.add_argument(
        "--runtime-sbom",
        type=Path,
        help="Optional compatibility input; it must byte-match the automatic SPDX output.",
    )
    parser.add_argument(
        "--runtime-license-manifest",
        type=Path,
        help="Optional compatibility input; it must byte-match the automatic license output.",
    )
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        result = export_runtime(
            source_root=args.document_preview_root,
            destination=args.destination,
            runtime_sbom=args.runtime_sbom,
            runtime_license_manifest=args.runtime_license_manifest,
        )
    except RuntimeExportError as exc:
        print(exc.code, file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
