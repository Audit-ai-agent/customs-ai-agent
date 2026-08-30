"""Fail closed unless an existing GHCR container package is private."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request

_OWNER = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,38})")
_PACKAGE_COMPONENT = re.compile(r"[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?")
_DIGEST = re.compile(r"sha256:[a-f0-9]{64}")


class GhcrPrivacyError(RuntimeError):
    pass


def _package_identity(image: str) -> tuple[str, str]:
    if not image.startswith("ghcr.io/") or len(image) > 512:
        raise GhcrPrivacyError
    reference = image.removeprefix("ghcr.io/")
    if "@" in reference:
        name, digest = reference.rsplit("@", 1)
        if _DIGEST.fullmatch(digest) is None:
            raise GhcrPrivacyError
    else:
        name = reference
    if ":" in name:
        # Mutable tags are never needed by a privacy decision in this pipeline.
        raise GhcrPrivacyError
    try:
        owner, package = name.split("/", 1)
    except ValueError:
        raise GhcrPrivacyError from None
    package_parts = package.split("/")
    if (
        _OWNER.fullmatch(owner) is None
        or not package_parts
        or any(_PACKAGE_COMPONENT.fullmatch(part) is None for part in package_parts)
    ):
        raise GhcrPrivacyError
    return owner, package


def _github_json(path: str, token: str) -> object:
    request = urllib.request.Request(
        "https://api.github.com" + path,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "customs-raster-ocr-private-package-verifier",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310
        return json.load(response)


def verify_private_package(image: str, token: str) -> None:
    owner, package = _package_identity(image)
    if not token or len(token) > 4096 or any(character.isspace() for character in token):
        raise GhcrPrivacyError
    quoted_owner = urllib.parse.quote(owner, safe="")
    identity = _github_json(f"/users/{quoted_owner}", token)
    if not isinstance(identity, dict) or identity.get("type") not in {"Organization", "User"}:
        raise GhcrPrivacyError
    scope = "orgs" if identity["type"] == "Organization" else "users"
    metadata = _github_json(
        f"/{scope}/{quoted_owner}/packages/container/{urllib.parse.quote(package, safe='')}",
        token,
    )
    if not isinstance(metadata, dict) or metadata.get("visibility") != "private":
        raise GhcrPrivacyError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    args = parser.parse_args()
    try:
        verify_private_package(args.image, os.environ.get("GH_TOKEN", ""))
    except Exception:
        # Never echo a token, response body, owner, or package metadata.
        print("ghcr_package_privacy_check_failed", file=sys.stderr)
        return 1
    print("private_ghcr_package_verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
