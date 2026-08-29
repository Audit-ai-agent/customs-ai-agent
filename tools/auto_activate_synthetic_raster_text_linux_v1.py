"""Build current Linux OCR evidence and atomically activate schema V12.

The command accepts only the fixed document-preview root.  It exposes no
model, hash, score, OCR payload, fallback or approval option.  A native
Linux/AMD64/CPython-3.11 bundle and the fixed public V19 model materials must
already exist locally; network access is blocked by the activation module.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from customs_ai_agent.evaluation.synthetic_raster_text_activation import (
    SyntheticRasterTextActivationError,
    SyntheticRasterTextAutoActivationResultV1,
    auto_activate_synthetic_raster_text_linux_preview97_v1,
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
    return auto_activate_synthetic_raster_text_linux_preview97_v1(
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
                "runtime_platform": "linux/amd64/cp311",
                "runtime_network": "disabled",
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
