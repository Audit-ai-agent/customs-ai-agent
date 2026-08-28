"""Build, evaluate, Gate, and activate public-mobile V15 raster text OCR.

This compatibility entrypoint intentionally invokes the same single V15,
staged semantic-metric-V3 text activation transaction as the text-specific wrapper.
It exposes no human approval, model/hash, score, threshold, OCR JSON, training,
package-install, network, or fallback option.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from customs_ai_agent.evaluation.synthetic_raster_text_activation import (
    SyntheticRasterTextActivationError,
    auto_activate_synthetic_raster_text_v1,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--document-preview-root",
        type=Path,
        required=True,
        help="fixed local _local/data/document-preview directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        result = auto_activate_synthetic_raster_text_v1(
            document_preview_root=arguments.document_preview_root
        )
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
