"""Verify the exported pointer with the exact Agent code in the final image."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

from customs_ai_agent.service.raster_text_preview_runtime import (
    resolve_raster_text_preview_engine,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--document-preview-root", required=True, type=Path)
    args = parser.parse_args()
    pointer = args.document_preview_root / "approvals/raster-text-active.json"
    try:
        if pointer.is_symlink() or not pointer.is_file():
            raise ValueError
        raw = pointer.read_bytes()
        value = json.loads(raw.decode("utf-8"))
        if (
            not isinstance(value, dict)
            or value.get("schema_version") != "11.0"
            or value.get("status") != "activated"
            or not isinstance(value.get("raster_text_profile_revision"), str)
        ):
            raise ValueError
        expected_revision = value["raster_text_profile_revision"]
        resolution = resolve_raster_text_preview_engine(
            approval_path=pointer,
            approval_sha256=hashlib.sha256(raw).hexdigest(),
        )
        if (
            resolution.status != "available"
            or resolution.raster_text_profile_revision != expected_revision
        ):
            raise ValueError
    except Exception:
        print("agent_runtime_binding_invalid", file=sys.stderr)
        return 1
    print(
        json.dumps(
            {"raster_text_profile_revision": expected_revision, "status": "available"},
            ensure_ascii=True,
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
