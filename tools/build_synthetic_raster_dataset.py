"""Build the fixed 6,000-PDF / 24,000-raster synthetic-raster-v1 dataset.

The source tree is enumerated solely from its deterministic clean/error PDF
paths.  No manifest, error label, existing JSON, OCR result, or VLM output is
read.  All outputs are create-only beneath the local document-preview root.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from customs_ai_agent.evaluation.synthetic_raster import (
    SyntheticRasterCompilerUnavailable,
    compile_synthetic_raster_dataset,
    ensure_fixed_synthetic_raster_build_inputs,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-preview-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        build_inputs = ensure_fixed_synthetic_raster_build_inputs(
            document_preview_root=arguments.document_preview_root
        )
        receipt = compile_synthetic_raster_dataset(
            document_preview_root=arguments.document_preview_root,
            build_inputs_relative_path=build_inputs.build_inputs_relative_path,
        )
    except (OSError, ValueError, SyntheticRasterCompilerUnavailable):
        print(json.dumps({"status": "failed", "code": "synthetic_raster_dataset_failed"}, separators=(",", ":")))
        return 2
    print(
        json.dumps(
            {
                "status": "synthetic_raster_dataset_written",
                "asset_profile_count": len(receipt.assets),
                "dataset_id": receipt.dataset_id,
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
