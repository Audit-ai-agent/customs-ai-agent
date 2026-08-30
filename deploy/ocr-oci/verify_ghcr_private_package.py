"""Fail closed unless an existing GHCR container package is private."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

_OWNER = re.compile(r"[a-z0-9](?:[a-z0-9-]{0,38})")
_PACKAGE_COMPONENT = re.compile(r"[a-z0-9](?:[a-z0-9._-]*[a-z0-9])?")
_DIGEST = re.compile(r"sha256:[a-f0-9]{64}")
_REPOSITORY = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})/[A-Za-z0-9._-]{1,100}")
_MAX_ATTEMPTS = 6
_MAX_RETRY_DELAY_SECONDS = 10


class GhcrPrivacyError(RuntimeError):
    pass


class _GhcrMetadataPending(GhcrPrivacyError):
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


def _verify_package_metadata(
    metadata: object,
    *,
    allow_missing_repository: bool,
    identity_type: str,
    owner: str,
    package: str,
    repository: str,
) -> None:
    if not isinstance(metadata, dict):
        raise GhcrPrivacyError

    for key in ("name", "package_type", "visibility", "owner", "version_count"):
        if key not in metadata:
            raise _GhcrMetadataPending
    if (
        metadata["name"] != package
        or metadata["package_type"] != "container"
        or metadata["visibility"] != "private"
    ):
        raise GhcrPrivacyError

    package_owner = metadata["owner"]
    if not isinstance(package_owner, dict):
        raise GhcrPrivacyError
    owner_login = package_owner.get("login")
    owner_type = package_owner.get("type")
    if (
        not isinstance(owner_login, str)
        or owner_login.casefold() != owner.casefold()
        or owner_type != identity_type
    ):
        raise GhcrPrivacyError

    version_count = metadata["version_count"]
    if isinstance(version_count, bool) or not isinstance(version_count, int) or version_count < 0:
        raise GhcrPrivacyError
    if version_count == 0:
        raise _GhcrMetadataPending

    # GitHub's documented Get-package response does not guarantee a repository
    # member. The exact source label and same-repository GITHUB_TOKEN used for
    # the first push establish that link. If the API does return the optional
    # member, however, it must agree with the expected repository.
    if "repository" not in metadata:
        if not allow_missing_repository:
            raise _GhcrMetadataPending
    else:
        linked_repository = metadata["repository"]
        if linked_repository is None:
            raise _GhcrMetadataPending
        if (
            not isinstance(linked_repository, dict)
            or str(linked_repository.get("full_name", "")).casefold() != repository.casefold()
        ):
            raise GhcrPrivacyError


def _package_json(path: str, token: str, *, retry_enabled: bool) -> object:
    try:
        return _github_json(path, token)
    except urllib.error.HTTPError as exc:
        if retry_enabled and (exc.code in {403, 404, 429} or 500 <= exc.code <= 599):
            raise _GhcrMetadataPending from None
        raise GhcrPrivacyError from None
    except (TimeoutError, urllib.error.URLError):
        if retry_enabled:
            raise _GhcrMetadataPending from None
        raise GhcrPrivacyError from None


def verify_private_package(
    image: str,
    repository: str,
    token: str,
    *,
    allow_missing_repository: bool = False,
    max_attempts: int = 1,
    retry_delay_seconds: int = 2,
) -> None:
    owner, package = _package_identity(image)
    if (
        _REPOSITORY.fullmatch(repository) is None
        or repository.split("/", 1)[0].casefold() != owner.casefold()
        or not token
        or len(token) > 4096
        or any(character.isspace() for character in token)
        or not isinstance(allow_missing_repository, bool)
        or isinstance(max_attempts, bool)
        or not isinstance(max_attempts, int)
        or not 1 <= max_attempts <= _MAX_ATTEMPTS
        or isinstance(retry_delay_seconds, bool)
        or not isinstance(retry_delay_seconds, int)
        or not 1 <= retry_delay_seconds <= _MAX_RETRY_DELAY_SECONDS
    ):
        raise GhcrPrivacyError
    quoted_owner = urllib.parse.quote(owner, safe="")
    identity = _github_json(f"/users/{quoted_owner}", token)
    if not isinstance(identity, dict) or identity.get("type") not in {"Organization", "User"}:
        raise GhcrPrivacyError
    scope = "orgs" if identity["type"] == "Organization" else "users"
    package_path = (
        f"/{scope}/{quoted_owner}/packages/container/{urllib.parse.quote(package, safe='')}"
    )
    retry_enabled = max_attempts > 1
    for attempt in range(max_attempts):
        try:
            metadata = _package_json(package_path, token, retry_enabled=retry_enabled)
            _verify_package_metadata(
                metadata,
                allow_missing_repository=allow_missing_repository,
                identity_type=identity["type"],
                owner=owner,
                package=package,
                repository=repository,
            )
            return
        except _GhcrMetadataPending:
            if attempt + 1 >= max_attempts:
                raise GhcrPrivacyError from None
            time.sleep(retry_delay_seconds)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--repository", required=True)
    parser.add_argument(
        "--allow-missing-repository-for-source-attested-workflow",
        action="store_true",
    )
    parser.add_argument("--max-attempts", default=1, type=int)
    parser.add_argument("--retry-delay-seconds", default=2, type=int)
    args = parser.parse_args()
    try:
        verify_private_package(
            args.image,
            args.repository,
            os.environ.get("GH_TOKEN", ""),
            allow_missing_repository=(args.allow_missing_repository_for_source_attested_workflow),
            max_attempts=args.max_attempts,
            retry_delay_seconds=args.retry_delay_seconds,
        )
    except Exception:
        # Never echo a token, response body, owner, or package metadata.
        print("ghcr_package_privacy_check_failed", file=sys.stderr)
        return 1
    print("private_ghcr_package_verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
