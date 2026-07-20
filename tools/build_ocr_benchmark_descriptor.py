"""Fail-closed OCR descriptor entry point for invalidated evaluation-v2.

The former builder parsed a repository-visible split and materialized its
locked membership.  Since that membership is publicly derivable, no descriptor
built from evaluation-v2 can be release evidence.  A replacement builder may
only be introduced with the evaluation-v3 sealed-bundle protocol.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path
from typing import Any

from evals.release_holdout_protocol import (
    INVALIDATED_EVALUATION_VERSION,
    RELEASE_INVALIDATION_CODE,
)


ReleaseAccessValidator = Callable[[Path, Path, Path], dict[str, str]]


class DescriptorBuildError(ValueError):
    """Raised when an OCR release descriptor cannot be built safely."""


def build_descriptor(
    *,
    splits_path: Path,
    rendered_root: Path,
    manifest_sha256: str,
    repo_root: Path,
    release_receipt_path: Path,
    release_access_validator: ReleaseAccessValidator | None = None,
    expected_manifest_sha256: str | None = None,
    evaluation_version: str = INVALIDATED_EVALUATION_VERSION,
) -> dict[str, Any]:
    """Reject every repository-visible evaluation-v2 descriptor build.

    Arguments are deliberately not inspected.  This ensures a failed request
    cannot read split membership, consume authorization, or enumerate rendered
    documents.
    """

    del (
        splits_path,
        rendered_root,
        manifest_sha256,
        repo_root,
        release_receipt_path,
        release_access_validator,
        expected_manifest_sha256,
    )
    if evaluation_version == INVALIDATED_EVALUATION_VERSION:
        raise DescriptorBuildError(RELEASE_INVALIDATION_CODE)
    raise DescriptorBuildError("ocr_descriptor_requires_sealed_evaluation_v3_runner")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fail-closed OCR release descriptor preflight.")
    parser.add_argument("--splits", required=True, type=Path)
    parser.add_argument("--rendered-root", required=True, type=Path)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--repo-root", required=True, type=Path)
    parser.add_argument("--release-receipt", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument(
        "--evaluation-version",
        default=INVALIDATED_EVALUATION_VERSION,
    )
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        build_descriptor(
            splits_path=args.splits,
            rendered_root=args.rendered_root,
            manifest_sha256=args.manifest_sha256.lower(),
            repo_root=args.repo_root,
            release_receipt_path=args.release_receipt,
            evaluation_version=args.evaluation_version,
        )
    except DescriptorBuildError as exc:
        print(f"OCR descriptor build blocked: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
