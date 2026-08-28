"""Create or reuse the fixed, leakage-safe Voyage replay OCR release."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from customs_ai_agent.evaluation.source_attested_public_raster_text_recognizer_voyage_replay import (
    VoyageReplayError,
)
from customs_ai_agent.evaluation.source_attested_public_raster_text_recognizer_voyage_replay_runtime import (
    ensure_source_attested_public_recognizer_voyage_replay_v1,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-preview-root", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("voyage-replay-v3",), default="voyage-replay-v3"
    )
    return parser


def _run(document_preview_root: Path) -> dict[str, object]:
    result = ensure_source_attested_public_recognizer_voyage_replay_v1(
        document_preview_root
    )
    root = document_preview_root.resolve(strict=True)
    return {
        "release_relative_path": (
            result.release.release_path.resolve(strict=True).relative_to(root).as_posix()
        ),
        "release_sha256": result.release.release_sha256,
        "stage": "voyage-replay-v3",
        "status": "reused" if result.reused else "published",
        "tuning_scope": "ctc_head_only",
    }


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    try:
        summary = _run(arguments.document_preview_root)
    except VoyageReplayError as error:
        summary = {"code": error.code, "status": "failed"}
        print(json.dumps(summary, separators=(",", ":"), sort_keys=True))
        return 2
    except (OSError, ValueError):
        summary = {"code": "voyage_replay_failed", "status": "failed"}
        print(json.dumps(summary, separators=(",", ":"), sort_keys=True))
        return 2
    print(json.dumps(summary, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
