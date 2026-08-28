"""Train one source-attested, random-init raster text recognizer release.

Only the bounded ``overfit`` stage is exposed.  The command discovers and
fully verifies the local direct-Paddle bundle; callers cannot select a model,
checkpoint, package tree, score, or release hash.  Output is content-free.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from customs_ai_agent.evaluation.source_attested_raster_text_recognizer_training import (
    SourceAttestedRecognizerTrainingError,
)
from customs_ai_agent.evaluation.source_attested_raster_text_recognizer_training_runtime import (
    ensure_source_attested_raster_text_recognizer_overfit,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-preview-root", type=Path, required=True)
    parser.add_argument("--stage", choices=("overfit",), default="overfit")
    return parser


def _run(document_preview_root: Path) -> dict[str, object]:
    result = ensure_source_attested_raster_text_recognizer_overfit(document_preview_root)
    root = document_preview_root.resolve(strict=True)
    relative = result.release.release_path.relative_to(root).as_posix()
    return {
        "status": "reused" if result.reused else "published",
        "stage": "overfit",
        "release_relative_path": relative,
        "receipt_sha256": result.release.release_sha256,
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        summary = _run(arguments.document_preview_root)
    except SourceAttestedRecognizerTrainingError as error:
        summary = {"status": "failed", "code": error.code}
        print(json.dumps(summary, separators=(",", ":"), sort_keys=True))
        return 2
    except (OSError, ValueError):
        summary = {"status": "failed", "code": "training_failed"}
        print(json.dumps(summary, separators=(",", ":"), sort_keys=True))
        return 2
    print(json.dumps(summary, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
