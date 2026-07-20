"""Validate and inventory the synthetic trade-document image dataset.

The source dataset is never modified.  The only writes are the two output paths
that the caller must provide explicitly.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError


SCHEMA_VERSION = "trade-images-inventory/v1"
MANIFEST_NAME = "manifest.csv"
DOCUMENT_COLUMNS = {
    "commercial_invoice": "ci_file",
    "packing_list": "pl_file",
    "bill_of_lading": "bl_file",
}
REQUIRED_COLUMNS = frozenset({"shipment_id", *DOCUMENT_COLUMNS.values()})
SUPPORTED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".webp"})
MEDIA_TYPES = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


class DatasetValidationError(ValueError):
    """Raised when the dataset cannot be safely inventoried."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _normalise_relative_path(
    dataset_root: Path,
    raw_path: str,
    *,
    document_type: str,
    row_number: int,
) -> tuple[Path, str]:
    if not raw_path or raw_path.isspace():
        raise DatasetValidationError(f"row {row_number}: document path is empty")
    if any(character in raw_path for character in ("\x00", "\r", "\n")):
        raise DatasetValidationError(f"row {row_number}: document path contains control characters")

    relative_path = Path(raw_path)
    if relative_path.is_absolute():
        raise DatasetValidationError(f"row {row_number}: absolute document paths are not allowed")
    # The supplied source manifest stores basenames while document type
    # determines the containing directory. Also accept an explicit portable
    # relative path so converted manifests remain self-describing.
    if relative_path.parent == Path("."):
        relative_path = Path(document_type) / relative_path

    resolved_path = (dataset_root / relative_path).resolve()
    try:
        canonical_relative = resolved_path.relative_to(dataset_root)
    except ValueError as exc:
        raise DatasetValidationError(
            f"row {row_number}: document path escapes the dataset root"
        ) from exc

    portable_relative = canonical_relative.as_posix()
    if resolved_path.suffix.lower() not in SUPPORTED_IMAGE_SUFFIXES:
        raise DatasetValidationError(
            f"row {row_number}: unsupported document image suffix: {resolved_path.suffix}"
        )
    return resolved_path, portable_relative


def _decode_image(path: Path, *, relative_path: str) -> tuple[int, int, str]:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(path) as image:
                image_format = image.format
                width, height = image.size
                image.verify()
            with Image.open(path) as image:
                image.load()
    except (OSError, UnidentifiedImageError, Image.DecompressionBombWarning) as exc:
        raise DatasetValidationError(f"cannot decode image: {relative_path}") from exc

    if image_format not in MEDIA_TYPES:
        raise DatasetValidationError(
            f"unsupported decoded image format for {relative_path}: {image_format}"
        )
    if width <= 0 or height <= 0:
        raise DatasetValidationError(f"invalid image dimensions for {relative_path}")
    return width, height, MEDIA_TYPES[image_format]


def _read_manifest(dataset_root: Path) -> tuple[list[dict[str, str]], Path]:
    manifest_path = dataset_root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise DatasetValidationError(f"required manifest is missing: {MANIFEST_NAME}")

    try:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as file_handle:
            reader = csv.DictReader(file_handle)
            if reader.fieldnames is None:
                raise DatasetValidationError("manifest has no header")
            missing_columns = sorted(REQUIRED_COLUMNS.difference(reader.fieldnames))
            if missing_columns:
                raise DatasetValidationError(
                    "manifest is missing required columns: " + ", ".join(missing_columns)
                )
            rows = list(reader)
    except UnicodeDecodeError as exc:
        raise DatasetValidationError("manifest must be UTF-8 encoded") from exc

    if not rows:
        raise DatasetValidationError("manifest contains no shipment rows")
    return rows, manifest_path


def _validate_output_paths(
    dataset_root: Path,
    inventory_output: Path,
    checksums_output: Path,
    *,
    overwrite: bool,
) -> tuple[Path, Path]:
    resolved_inventory = inventory_output.resolve()
    resolved_checksums = checksums_output.resolve()
    if resolved_inventory == resolved_checksums:
        raise DatasetValidationError("inventory and checksums outputs must be different files")

    source_manifest = (dataset_root / MANIFEST_NAME).resolve()
    if source_manifest in {resolved_inventory, resolved_checksums}:
        raise DatasetValidationError("an output path cannot overwrite the source manifest")

    if not overwrite:
        existing = [str(path) for path in (resolved_inventory, resolved_checksums) if path.exists()]
        if existing:
            raise DatasetValidationError("output already exists: " + ", ".join(existing))
    return resolved_inventory, resolved_checksums


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file_handle:
            file_handle.write(content)
            temporary_name = file_handle.name
        os.replace(temporary_name, path)
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def build_inventory(
    dataset_root: Path,
    inventory_output: Path,
    checksums_output: Path,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate ``dataset_root`` and write deterministic inventory evidence."""

    root = dataset_root.resolve()
    if not root.is_dir():
        raise DatasetValidationError(f"dataset root is not a directory: {root}")
    resolved_inventory, resolved_checksums = _validate_output_paths(
        root,
        inventory_output,
        checksums_output,
        overwrite=overwrite,
    )
    rows, manifest_path = _read_manifest(root)

    shipment_ids: set[str] = set()
    referenced_paths: dict[str, str] = {}
    image_records: list[dict[str, Any]] = []
    hashes: dict[str, str] = {}
    type_counts: Counter[str] = Counter()
    media_type_counts: Counter[str] = Counter()
    dimension_counts: Counter[tuple[int, int]] = Counter()
    total_bytes = 0

    for row_number, row in enumerate(rows, start=2):
        shipment_id = (row.get("shipment_id") or "").strip()
        if not shipment_id:
            raise DatasetValidationError(f"row {row_number}: shipment_id is empty")
        if shipment_id in shipment_ids:
            raise DatasetValidationError(f"row {row_number}: duplicate shipment_id: {shipment_id}")
        shipment_ids.add(shipment_id)

        for document_type, column in DOCUMENT_COLUMNS.items():
            resolved_path, relative_path = _normalise_relative_path(
                root,
                row.get(column) or "",
                document_type=document_type,
                row_number=row_number,
            )
            path_key = str(resolved_path).casefold()
            if path_key in referenced_paths:
                raise DatasetValidationError(
                    f"row {row_number}: image path is assigned more than once: {relative_path}"
                )
            referenced_paths[path_key] = relative_path
            if not resolved_path.is_file():
                raise DatasetValidationError(f"referenced image is missing: {relative_path}")

            width, height, media_type = _decode_image(
                resolved_path,
                relative_path=relative_path,
            )
            content_sha256 = _sha256(resolved_path)
            if content_sha256 in hashes:
                raise DatasetValidationError(
                    f"duplicate image content: {hashes[content_sha256]} and {relative_path}"
                )
            hashes[content_sha256] = relative_path

            byte_size = resolved_path.stat().st_size
            image_records.append(
                {
                    "document_type": document_type,
                    "relative_path": relative_path,
                    "sha256": content_sha256,
                    "byte_size": byte_size,
                    "width": width,
                    "height": height,
                    "media_type": media_type,
                }
            )
            type_counts[document_type] += 1
            media_type_counts[media_type] += 1
            dimension_counts[(width, height)] += 1
            total_bytes += byte_size

    discovered_paths = {
        str(path.resolve()).casefold(): path.resolve().relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_SUFFIXES
    }
    referenced_keys = set(referenced_paths)
    unreferenced = sorted(
        discovered_paths[key] for key in discovered_paths.keys() - referenced_keys
    )
    if unreferenced:
        sample = ", ".join(unreferenced[:3])
        raise DatasetValidationError(
            f"dataset contains {len(unreferenced)} unreferenced image(s): {sample}"
        )

    missing_from_tree = sorted(
        referenced_paths[key] for key in referenced_keys - discovered_paths.keys()
    )
    if missing_from_tree:
        raise DatasetValidationError(
            "referenced image was not discovered during final inventory: "
            + ", ".join(missing_from_tree[:3])
        )

    image_records.sort(key=lambda record: (record["document_type"], record["relative_path"]))
    manifest_sha256 = _sha256(manifest_path)
    inventory: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "source_manifest": {
            "relative_path": MANIFEST_NAME,
            "sha256": manifest_sha256,
            "row_count": len(rows),
            "required_columns": sorted(REQUIRED_COLUMNS),
        },
        "summary": {
            "shipment_count": len(shipment_ids),
            "image_count": len(image_records),
            "document_type_counts": dict(sorted(type_counts.items())),
            "media_type_counts": dict(sorted(media_type_counts.items())),
            "dimension_counts": [
                {"width": width, "height": height, "count": count}
                for (width, height), count in sorted(dimension_counts.items())
            ],
            "total_bytes": total_bytes,
            "duplicate_sha256_groups": 0,
            "unreferenced_image_count": 0,
            "contains_image_content": False,
            "contains_manifest_business_values": False,
        },
        "images": image_records,
    }

    inventory_text = (
        json.dumps(
            inventory,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    checksum_lines = [f"{manifest_sha256}  {MANIFEST_NAME}"]
    checksum_lines.extend(
        f"{record['sha256']}  {record['relative_path']}" for record in image_records
    )
    checksums_text = "\n".join(checksum_lines) + "\n"

    _atomic_write(resolved_inventory, inventory_text)
    _atomic_write(resolved_checksums, checksums_text)
    return inventory


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and inventory a trade-document image dataset.",
    )
    parser.add_argument("--dataset-root", required=True, type=Path)
    parser.add_argument("--inventory-json", required=True, type=Path)
    parser.add_argument("--checksums-sha256", required=True, type=Path)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace explicitly named output files if they already exist.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        inventory = build_inventory(
            args.dataset_root,
            args.inventory_json,
            args.checksums_sha256,
            overwrite=args.overwrite,
        )
    except DatasetValidationError as exc:
        print(f"dataset inventory failed: {exc}")
        return 1

    print(
        "dataset inventory passed: "
        f"{inventory['summary']['shipment_count']} shipments, "
        f"{inventory['summary']['image_count']} images"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
