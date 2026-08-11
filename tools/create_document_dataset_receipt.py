"""Create an immutable, local-only metadata receipt for a document dataset.

The command hashes source files in place and writes no source byte, document
text, OCR output, or absolute path.  It is intentionally separate from Gold
packet preparation and never marks a dataset as human approved.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
from typing import Sequence

from customs_ai_agent.evaluation.document_gold import (
    DatasetAssetReceiptV1,
    DatasetReceiptV1,
)


_MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".jpeg": "image/jpeg",
    ".jpg": "image/jpeg",
    ".png": "image/png",
    ".json": "application/json",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _local_data_directory(path: Path) -> Path:
    resolved = path.resolve()
    parts = [part.casefold() for part in resolved.parts]
    if not any(parts[index : index + 2] == ["_local", "data"] for index in range(len(parts) - 1)):
        raise ValueError("output directory must be below the local-only _local/data boundary")
    return resolved


def build_receipt(
    *,
    source_directory: Path,
    dataset_id: str,
    dataset_revision: str,
    received_as_of: date,
) -> DatasetReceiptV1:
    root = source_directory.resolve()
    if not root.is_dir():
        raise ValueError("source directory is unavailable")
    assets: list[DatasetAssetReceiptV1] = []
    for path in sorted((candidate for candidate in root.rglob("*") if candidate.is_file()), key=str):
        media_type = _MEDIA_TYPES.get(path.suffix.casefold())
        if media_type is None:
            continue
        relative_path = path.relative_to(root).as_posix()
        digest = _sha256(path)
        path_digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:12]
        assets.append(
            DatasetAssetReceiptV1(
                asset_id=f"asset-{digest[:24]}-{path_digest}",
                relative_path=relative_path,
                media_type=media_type,
                byte_size=path.stat().st_size,
                sha256=digest,
                source_role=(
                    "structured_reference" if media_type == "application/json" else "preview_candidate"
                ),
            )
        )
    if not assets:
        raise ValueError("source directory has no permitted PDF/JPEG/PNG/JSON assets")
    return DatasetReceiptV1(
        dataset_id=dataset_id,
        dataset_revision=dataset_revision,
        received_as_of=received_as_of,
        assets=assets,
    )


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    ) + "\n"


def write_immutable_receipt(*, receipt: DatasetReceiptV1, output_directory: Path) -> dict[str, object]:
    output = _local_data_directory(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    destination = output / "dataset-receipt.json"
    text = _canonical_json(receipt.model_dump(mode="json", warnings=False))
    if destination.exists() and destination.read_text(encoding="utf-8") != text:
        raise ValueError("an immutable receipt already exists with different content")
    destination.write_text(text, encoding="utf-8")
    return {
        "status": "receipt_created",
        "dataset_id": receipt.dataset_id,
        "dataset_revision": receipt.dataset_revision,
        "asset_count": len(receipt.assets),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Hash a local document dataset into an immutable receipt.")
    parser.add_argument("--source-directory", type=Path, required=True)
    parser.add_argument("--dataset-id", required=True)
    parser.add_argument("--dataset-revision", required=True)
    parser.add_argument("--received-as-of", required=True, help="YYYY-MM-DD")
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        received_as_of = date.fromisoformat(arguments.received_as_of)
        if received_as_of.isoformat() != arguments.received_as_of:
            raise ValueError("--received-as-of must be an ISO calendar date")
        receipt = build_receipt(
            source_directory=arguments.source_directory,
            dataset_id=arguments.dataset_id,
            dataset_revision=arguments.dataset_revision,
            received_as_of=received_as_of,
        )
        summary = write_immutable_receipt(receipt=receipt, output_directory=arguments.output_directory)
    except (OSError, ValueError) as error:
        # Do not expose absolute local paths or source content in command output.
        print(json.dumps({"status": "failed", "code": type(error).__name__}, separators=(",", ":")))
        return 2
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
