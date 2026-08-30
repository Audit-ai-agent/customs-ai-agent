"""Verify a digest-pinned GHCR image's authenticated Linux config binding."""

from __future__ import annotations

import argparse
import json
import re
import sys

_DIGEST = re.compile(r"sha256:[a-f0-9]{64}")
_REPOSITORY = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})/[A-Za-z0-9._-]{1,100}")
_REVISION = re.compile(r"[a-f0-9]{40}")
_MANIFEST_SHA = re.compile(r"[a-f0-9]{64}")
_PROFILE_REVISION = re.compile(r"synthetic-raster-text-v1:[a-f0-9]{24}")
_MAX_CONFIG_BYTES = 64 * 1024


class GhcrSourceAttestationError(RuntimeError):
    pass


def _reject_nonfinite(_value: str) -> None:
    raise ValueError


def parse_image_config(raw: bytes) -> object:
    if not raw or len(raw) > _MAX_CONFIG_BYTES:
        raise GhcrSourceAttestationError
    try:
        return json.loads(raw, parse_constant=_reject_nonfinite)
    except (UnicodeDecodeError, ValueError):
        raise GhcrSourceAttestationError from None


def _expected_image(repository: str, kind: str) -> str:
    suffix = "raster-runtime" if kind == "runtime" else "raster-ocr"
    return f"ghcr.io/{repository.casefold()}-{suffix}"


def _matches_digest_reference(reference: str, expected_repository: str) -> bool:
    parts = reference.split("@")
    return (
        len(parts) == 2
        and parts[0] == expected_repository
        and _DIGEST.fullmatch(parts[1]) is not None
    )


def verify_source_attestation(
    config: object,
    *,
    image: str,
    kind: str,
    repository: str,
    runtime_image: str | None = None,
    source_revision: str | None = None,
) -> None:
    if _REPOSITORY.fullmatch(repository) is None or kind not in {"runtime", "agent"}:
        raise GhcrSourceAttestationError
    if not _matches_digest_reference(image, _expected_image(repository, kind)):
        raise GhcrSourceAttestationError
    if (
        not isinstance(config, dict)
        or config.get("architecture") != "amd64"
        or config.get("os") != "linux"
    ):
        raise GhcrSourceAttestationError
    image_config = config.get("config")
    labels = image_config.get("Labels") if isinstance(image_config, dict) else None
    if not isinstance(labels, dict) or any(
        not isinstance(key, str) or not isinstance(value, str) for key, value in labels.items()
    ):
        raise GhcrSourceAttestationError
    expected_source = f"https://github.com/{repository.casefold()}"
    if labels.get("org.opencontainers.image.source") != expected_source:
        raise GhcrSourceAttestationError

    if kind == "runtime":
        if runtime_image is not None or source_revision is not None:
            raise GhcrSourceAttestationError
        if (
            labels.get("io.customs.artifact.kind") != "raster-text-runtime-context-v1"
            or _MANIFEST_SHA.fullmatch(labels.get("io.customs.runtime.manifest-sha256", "")) is None
            or _PROFILE_REVISION.fullmatch(
                labels.get("io.customs.raster-text.profile-revision", "")
            )
            is None
        ):
            raise GhcrSourceAttestationError
        return

    if (
        source_revision is None
        or _REVISION.fullmatch(source_revision) is None
        or runtime_image is None
        or not _matches_digest_reference(runtime_image, _expected_image(repository, "runtime"))
        or labels.get("io.customs.artifact.kind") != "customs-ai-agent-raster-text-ocr-v1"
        or labels.get("org.opencontainers.image.revision") != source_revision
        or labels.get("io.customs.raster-runtime.image") != runtime_image
        or labels.get("io.customs.raster-runtime.contract") != "raster-text-runtime-oci-v1"
    ):
        raise GhcrSourceAttestationError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--kind", choices=("runtime", "agent"), required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--runtime-image")
    parser.add_argument("--source-revision")
    args = parser.parse_args()
    try:
        raw = sys.stdin.buffer.read(_MAX_CONFIG_BYTES + 1)
        verify_source_attestation(
            parse_image_config(raw),
            image=args.image,
            kind=args.kind,
            repository=args.repository,
            runtime_image=args.runtime_image,
            source_revision=args.source_revision,
        )
    except Exception:
        print("ghcr_image_source_attestation_failed", file=sys.stderr)
        return 1
    print("ghcr_image_source_attestation_verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
