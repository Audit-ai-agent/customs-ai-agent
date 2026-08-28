"""Create one leakage-safe, source-attested public OCR head release.

The command accepts no model, checkpoint, label, score or hash.  It always
uses the pinned public mobile recognizer, selection V2's remaining 500 train
families and exactly two source-Gold field crops per family.  Development uses
100 separate families with two crops each.  Publication requires a strict
development improvement plus static/native roundtrip verification.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from customs_ai_agent.evaluation.source_attested_public_raster_text_recognizer_finetuning import (
    SourceAttestedPublicRecognizerFineTuneError,
)
from customs_ai_agent.evaluation.source_attested_public_raster_text_recognizer_finetuning_runtime import (
    ensure_source_attested_public_recognizer_leakage_safe_v2,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-preview-root", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("leakage-safe-v2",), default="leakage-safe-v2"
    )
    return parser


def _run(document_preview_root: Path) -> dict[str, object]:
    result = ensure_source_attested_public_recognizer_leakage_safe_v2(
        document_preview_root, tuning_scope="head"
    )
    root = document_preview_root.resolve(strict=True)
    return {
        "release_relative_path": (
            result.release.release_path.resolve(strict=True).relative_to(root).as_posix()
        ),
        "release_sha256": result.release.release_sha256,
        "stage": "leakage-safe-v2",
        "status": "reused" if result.reused else "published",
        "tuning_scope": "head",
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        summary = _run(arguments.document_preview_root)
    except SourceAttestedPublicRecognizerFineTuneError as error:
        summary = {"code": error.code, "status": "failed"}
        print(json.dumps(summary, separators=(",", ":"), sort_keys=True))
        return 2
    except (OSError, ValueError):
        summary = {"code": "public_finetune_failed", "status": "failed"}
        print(json.dumps(summary, separators=(",", ":"), sort_keys=True))
        return 2
    print(json.dumps(summary, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
