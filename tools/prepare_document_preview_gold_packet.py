"""Create local-only, draft 60+20 OCR Gold packets from an immutable receipt.

This command does not read source document bytes, run OCR, or approve Gold. It
only selects receipt entries deterministically and writes metadata below
``_local/data`` for a later human adjudication process.
"""

from __future__ import annotations

import argparse
import json
from datetime import date
from pathlib import Path
from typing import Sequence

from customs_ai_agent.evaluation.document_gold import DatasetReceiptV1, build_draft_gold_packets


def _local_data_directory(path: Path) -> Path:
    resolved = path.resolve()
    parts = [part.casefold() for part in resolved.parts]
    if not any(parts[index : index + 2] == ["_local", "data"] for index in range(len(parts) - 1)):
        raise ValueError("output directory must be below the local-only _local/data boundary")
    return resolved


def _write_json(path: Path, payload: object) -> None:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    path.write_text(f"{encoded}\n", encoding="utf-8")


def prepare_packets(
    *, receipt_path: Path, output_directory: Path, prepared_as_of: date
) -> dict[str, object]:
    receipt_raw = json.loads(receipt_path.read_text(encoding="utf-8"))
    receipt = DatasetReceiptV1.model_validate(receipt_raw)
    development, sealed = build_draft_gold_packets(receipt, prepared_as_of=prepared_as_of)
    output = _local_data_directory(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    _write_json(
        output / "document-preview-development-draft.json",
        development.model_dump(mode="json", warnings=False),
    )
    _write_json(
        output / "document-preview-sealed-draft.json",
        sealed.model_dump(mode="json", warnings=False),
    )
    return {
        "status": "awaiting_human_adjudication",
        "dataset_id": receipt.dataset_id,
        "dataset_revision": receipt.dataset_revision,
        "development_count": len(development.items),
        "sealed_count": len(sealed.items),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare local-only draft document-preview Gold packets."
    )
    parser.add_argument(
        "--receipt", type=Path, required=True, help="Local immutable dataset receipt JSON."
    )
    parser.add_argument(
        "--output-directory",
        type=Path,
        required=True,
        help="Existing or new directory below _local/data for the draft packets.",
    )
    parser.add_argument(
        "--as-of", required=True, help="Packet preparation date in YYYY-MM-DD format."
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        prepared_as_of = date.fromisoformat(arguments.as_of)
        if prepared_as_of.isoformat() != arguments.as_of:
            raise ValueError("--as-of must be an ISO calendar date")
        summary = prepare_packets(
            receipt_path=arguments.receipt,
            output_directory=arguments.output_directory,
            prepared_as_of=prepared_as_of,
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        # Do not print source data, absolute paths, or parser/provider details.
        print(json.dumps({"status": "failed", "code": type(error).__name__}, separators=(",", ":")))
        return 2
    print(json.dumps(summary, ensure_ascii=False, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
