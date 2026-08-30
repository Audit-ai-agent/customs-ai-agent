"""Warm the exported OCR child through the exact authoritative Agent loader."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path, PurePosixPath

_REPOSITORY_TOOLS = Path(__file__).resolve().parents[2] / "tools"
if _REPOSITORY_TOOLS.is_dir():
    sys.path.insert(0, str(_REPOSITORY_TOOLS))

from customs_ai_agent.service.raster_text_preview_runtime import (  # noqa: E402
    RasterTextActivationPointerV12,
    RasterTextOfflineProfileV19,
    _load_binding_from_pointer,
    parse_raster_text_activation_pointer,
)
from raster_text_oci_evidence import verify_v9_direct_evidence_closure  # noqa: E402

_MAX_MANIFEST_BYTES = 24 * 1024 * 1024
_MAX_DECLARED_FILES = 24_999


def _relative_path(value: object) -> PurePosixPath:
    if not isinstance(value, str) or not value or len(value) > 512 or "\\" in value:
        raise ValueError
    relative = PurePosixPath(value)
    if (
        relative.is_absolute()
        or ".." in relative.parts
        or any(part in {"", "."} for part in relative.parts)
    ):
        raise ValueError
    return relative


def _declared_activation_evidence(root: Path) -> set[PurePosixPath]:
    manifest_path = root / "_deployment/raster-text-runtime-oci-v1.json"
    if manifest_path.is_symlink() or not manifest_path.is_file():
        raise ValueError
    raw = manifest_path.read_bytes()
    if not raw or len(raw) > _MAX_MANIFEST_BYTES:
        raise ValueError
    value = json.loads(raw.decode("utf-8"))
    files = value.get("files") if isinstance(value, dict) else None
    if not isinstance(files, list) or not 1 <= len(files) <= _MAX_DECLARED_FILES:
        raise ValueError
    evidence: set[PurePosixPath] = set()
    for record in files:
        if not isinstance(record, dict) or set(record) != {
            "byte_size",
            "relative_path",
            "role",
            "sha256",
        }:
            raise ValueError
        if record.get("role") == "activation-evidence":
            relative = _relative_path(record.get("relative_path"))
            if relative in evidence:
                raise ValueError
            evidence.add(relative)
    if not evidence:
        raise ValueError
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-preview-root", required=True, type=Path)
    args = parser.parse_args()
    binding = None
    try:
        root = args.document_preview_root.resolve(strict=True)
        if args.document_preview_root.is_symlink() or not root.is_dir():
            raise ValueError
        declared_evidence = _declared_activation_evidence(root)
        opened: set[PurePosixPath] = set()

        def audit(event: str, event_args: tuple[object, ...]) -> None:
            if (
                event != "open"
                or not event_args
                or not isinstance(event_args[0], (str, bytes, os.PathLike))
            ):
                return
            try:
                candidate = Path(os.fsdecode(event_args[0])).resolve(strict=False)
                if candidate.is_relative_to(root):
                    opened.add(PurePosixPath(candidate.relative_to(root).as_posix()))
            except (OSError, TypeError, ValueError):
                return

        sys.addaudithook(audit)
        pointer = root / "approvals/raster-text-active.json"
        if pointer.is_symlink() or not pointer.is_file():
            raise ValueError
        raw = pointer.read_bytes()
        pointer_value = parse_raster_text_activation_pointer(raw)
        if type(pointer_value) is not RasterTextActivationPointerV12:
            raise ValueError
        expected_revision = pointer_value.raster_text_profile_revision
        binding = _load_binding_from_pointer(
            root=root,
            approval_path=pointer,
            approval_sha256=hashlib.sha256(raw).hexdigest(),
            approval_raw=raw,
            start_service_session=True,
            verification_mode="runtime",
        )
        verify_v9_direct_evidence_closure(root=root, pointer=binding.pointer)
        if (
            type(binding.pointer) is not RasterTextActivationPointerV12
            or type(binding.profile) is not RasterTextOfflineProfileV19
            or binding.raster_text_profile_revision != expected_revision
        ):
            raise ValueError
        # start_service_session=True has already waited for the child ready
        # frame after package/model checks and native cv2/Paddle imports.
        binding.verify_runtime_state()
        binding.verify_runtime_state_nonblocking()
        if not declared_evidence.issubset(opened):
            raise ValueError
        binding.close()
        binding = None
    except Exception:
        print("agent_runtime_binding_invalid", file=sys.stderr)
        return 1
    finally:
        if binding is not None:
            try:
                binding.close()
            except Exception:
                pass
    print(
        json.dumps(
            {
                "raster_text_profile_revision": expected_revision,
                "status": "warmed_available",
            },
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
