"""Register already-local OCR runtime files as immutable sidecars.

This command never invokes a package manager, downloads a model, or contacts a
network service.  An operator must explicitly provision pinned dependencies
and model files first, then use ``--confirm-provision`` to record their exact
local inventory under ``_local/data/document-preview``.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Mapping, Sequence, cast

from customs_ai_agent.service.document_preview_runtime import (
    DocumentPreviewRuntimeError,
    parse_runtime_configuration,
    preflight_provisioned_docling_native_runtime,
    validate_document_preview_root,
    write_runtime_sidecars,
)


_DEFAULT_MINIMUM_FREE_GIB = 12.0


def _json_object(path: Path, *, root: Path) -> Mapping[str, object]:
    try:
        resolved = path.resolve(strict=True)
        resolved.relative_to(root)
        value = json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise DocumentPreviewRuntimeError("document_preview_provisioning_input_invalid") from error
    if not isinstance(value, dict):
        raise DocumentPreviewRuntimeError("document_preview_provisioning_input_invalid")
    return cast(Mapping[str, object], value)


def _check_free_space(path: Path, minimum_free_gib: float) -> None:
    if minimum_free_gib <= 0 or minimum_free_gib > 1_024:
        raise DocumentPreviewRuntimeError("document_preview_provisioning_space_budget_invalid")
    try:
        free_bytes = shutil.disk_usage(path).free
    except OSError as error:
        raise DocumentPreviewRuntimeError(
            "document_preview_provisioning_space_check_failed"
        ) from error
    if free_bytes < int(minimum_free_gib * 1024**3):
        raise DocumentPreviewRuntimeError("document_preview_provisioning_space_insufficient")


def provision_runtime(
    *,
    document_preview_root: Path,
    configuration_path: Path,
    inventory_path: Path,
    cache_directory: Path | None,
    minimum_free_gib: float,
) -> tuple[str, str]:
    root = validate_document_preview_root(document_preview_root, create=True)
    _check_free_space(root, minimum_free_gib)
    if cache_directory is not None:
        try:
            cache = cache_directory.resolve(strict=True)
        except OSError as error:
            raise DocumentPreviewRuntimeError(
                "document_preview_provisioning_space_check_failed"
            ) from error
        _check_free_space(cache, minimum_free_gib)
    configuration = _json_object(configuration_path, root=root)
    inventory = _json_object(inventory_path, root=root)
    configuration_sha256, model_inventory_sha256 = write_runtime_sidecars(
        document_preview_root=root,
        configuration_value=configuration,
        inventory_value=inventory,
    )
    configuration_raw = (
        json.dumps(
            configuration,
            ensure_ascii=False,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        + b"\n"
    )
    parsed_configuration = parse_runtime_configuration(configuration_raw)
    if parsed_configuration.engine_kind == "docling":
        preflight_provisioned_docling_native_runtime(
            document_preview_root=root,
            configuration=parsed_configuration,
        )
    return configuration_sha256, model_inventory_sha256


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Register explicitly provisioned local PDF OCR runtime assets without downloads."
    )
    parser.add_argument("--document-preview-root", type=Path, required=True)
    parser.add_argument("--configuration", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument(
        "--cache-directory",
        type=Path,
        help="Optional package/model cache volume to check before an operator provisions assets.",
    )
    parser.add_argument(
        "--minimum-free-gib",
        type=float,
        default=_DEFAULT_MINIMUM_FREE_GIB,
        help="Conservative required free space on each checked volume (default: 12 GiB).",
    )
    parser.add_argument(
        "--confirm-provision",
        action="store_true",
        help="Required acknowledgement that packages and models were explicitly provisioned locally.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if not arguments.confirm_provision:
        print(
            json.dumps({"status": "failed", "code": "confirmation_required"}, separators=(",", ":"))
        )
        return 2
    try:
        configuration_sha256, model_inventory_sha256 = provision_runtime(
            document_preview_root=arguments.document_preview_root,
            configuration_path=arguments.configuration,
            inventory_path=arguments.inventory,
            cache_directory=arguments.cache_directory,
            minimum_free_gib=arguments.minimum_free_gib,
        )
    except DocumentPreviewRuntimeError as error:
        print(json.dumps({"status": "failed", "code": error.code}, separators=(",", ":")))
        return 2
    print(
        json.dumps(
            {
                "status": "provisioned",
                "configuration_sha256": configuration_sha256,
                "model_inventory_sha256": model_inventory_sha256,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
