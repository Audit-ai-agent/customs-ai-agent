"""Revalidate and activate the best retained V19 preview-draft OCR lane.

The wrapper accepts only the fixed local document-preview root.  It neither
accepts nor selects a recognizer and exposes no model, hash, score, OCR JSON,
fallback, or human-approval input.  The activation module replays the fixed
2,400-asset source/Gold/checkpoint chain, requires CER <= 2% and critical
exact >= 97% in every 3x4 cell, and runs a fresh current-runtime 24-asset
smoke under the unchanged 10-second p95 and 30-second request ceilings.  It
writes a schema-11 pointer only after every evidence boundary passes.  Any
failure leaves the active pointer intact.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from customs_ai_agent.evaluation.synthetic_raster_text_activation import (
    SyntheticRasterTextActivationError,
    SyntheticRasterTextAutoActivationResultV1,
    auto_activate_synthetic_raster_text_preview97_v1,
)


_DOCUMENT_PREVIEW_ROOT_SUFFIX = ("_local", "data", "document-preview")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-preview-root", type=Path, required=True)
    return parser


def _root(path: Path) -> Path:
    try:
        if path.is_symlink():
            raise OSError
        root = path.resolve(strict=True)
        if (
            root.is_symlink()
            or not root.is_dir()
            or tuple(part.casefold() for part in root.parts[-3:]) != _DOCUMENT_PREVIEW_ROOT_SUFFIX
        ):
            raise OSError
        return root
    except OSError:
        raise SyntheticRasterTextActivationError("raster_text_activation_root_invalid") from None


def _run(document_preview_root: Path) -> SyntheticRasterTextAutoActivationResultV1:
    """Enter the fixed V19 preview-97 revalidation transaction."""

    return auto_activate_synthetic_raster_text_preview97_v1(
        document_preview_root=_root(document_preview_root)
    )


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = _run(arguments.document_preview_root)
    except SyntheticRasterTextActivationError as error:
        print(json.dumps({"status": "failed", "code": error.code}, separators=(",", ":")))
        return 2
    print(
        json.dumps(
            {
                "status": "activated",
                "dataset_id": "synthetic-raster-text-v1",
                "raster_text_profile_revision": result.raster_text_profile_revision,
                "activation_pointer_sha256": result.activation_pointer_sha256,
                "runtime_network": "disabled",
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
