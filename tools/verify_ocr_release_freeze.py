"""CLI for the non-circular C0 authorization / C1 pin freeze."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from customs_ai_agent.ocr_freeze import (
    OcrFreezeVerificationError,
    verify_ocr_release_freeze,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the exact clean C1 OCR release pin commit."
    )
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--descriptor", required=True, type=Path)
    parser.add_argument("--authorization-receipt", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        attestation = verify_ocr_release_freeze(
            repo_root=args.repo_root,
            descriptor_path=args.descriptor,
            authorization_receipt_path=args.authorization_receipt,
        )
    except OcrFreezeVerificationError as exc:
        print(
            json.dumps(
                {
                    "schema_version": "ocr-release-source-freeze-attestation/v1",
                    "status": "NOT_FROZEN",
                    "error_code": exc.code,
                },
                sort_keys=True,
            )
        )
        return 1
    print(
        json.dumps(
            {
                "schema_version": "ocr-release-source-freeze-attestation/v1",
                "status": "FROZEN",
                "pin_commit_sha": attestation.pin_commit_sha,
                "authorization_commit_sha": attestation.authorization_commit_sha,
                "descriptor_sha256": attestation.descriptor_sha256,
                "authorization_receipt_sha256": (attestation.authorization_receipt_sha256),
                "authorization_receipt_registry_key": (
                    attestation.authorization_receipt_registry_key
                ),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
