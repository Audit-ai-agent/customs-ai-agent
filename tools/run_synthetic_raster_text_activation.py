"""Start or inspect the detached synthetic Raster text OCR activation.

Examples:
  python tools/run_synthetic_raster_text_activation.py start \
    --document-preview-root <fixed-document-preview-root>
  python tools/run_synthetic_raster_text_activation.py status \
    --document-preview-root <fixed-document-preview-root>

The internal worker subcommand is created only by ``start``.  It invokes the
existing one-command activation with fixed arguments.  It discards stderr
and retains only a bounded, schema-checked stdout status/code.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from customs_ai_agent.evaluation.raster_text_background_activation import (
    RasterTextBackgroundActivationError,
    background_activation_status,
    run_background_worker,
    start_background_activation,
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("start", "status"):
        command = commands.add_parser(name)
        command.add_argument("--document-preview-root", type=Path, required=True)
    worker = commands.add_parser("_worker", help=argparse.SUPPRESS)
    worker.add_argument("--document-preview-root", type=Path, required=True)
    worker.add_argument("--run-id", required=True)
    worker.add_argument("--command-fingerprint", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    if arguments.command == "_worker":
        return run_background_worker(
            arguments.document_preview_root,
            run_id=arguments.run_id,
            command_fingerprint=arguments.command_fingerprint,
        )
    try:
        if arguments.command == "start":
            result = start_background_activation(arguments.document_preview_root)
        else:
            result = background_activation_status(arguments.document_preview_root)
    except RasterTextBackgroundActivationError as error:
        print(json.dumps({"code": error.code, "status": "failed"}, separators=(",", ":")))
        return 2
    print(json.dumps(result, ensure_ascii=True, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
