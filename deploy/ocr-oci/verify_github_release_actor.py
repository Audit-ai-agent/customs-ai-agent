"""Fail closed unless a private main release was dispatched by a repository admin."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request

_REPOSITORY = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})/[A-Za-z0-9._-]{1,100}")
_ACTOR = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,37}[A-Za-z0-9])?")


class GithubReleaseActorError(RuntimeError):
    pass


def _github_json(path: str, token: str) -> object:
    request = urllib.request.Request(
        "https://api.github.com" + path,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "customs-raster-ocr-release-actor-verifier",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310
        return json.load(response)


def verify_release_actor(repository: str, actor: str, token: str) -> None:
    if (
        _REPOSITORY.fullmatch(repository) is None
        or _ACTOR.fullmatch(actor) is None
        or not token
        or len(token) > 4096
        or any(character.isspace() for character in token)
    ):
        raise GithubReleaseActorError

    owner, name = repository.split("/", 1)
    repository_path = (
        f"/repos/{urllib.parse.quote(owner, safe='')}/{urllib.parse.quote(name, safe='')}"
    )
    metadata = _github_json(repository_path, token)
    if (
        not isinstance(metadata, dict)
        or str(metadata.get("full_name", "")).casefold() != repository.casefold()
        or metadata.get("private") is not True
        or metadata.get("visibility") != "private"
        or metadata.get("default_branch") != "main"
        or metadata.get("archived") is not False
        or metadata.get("fork") is not False
    ):
        raise GithubReleaseActorError

    permission = _github_json(
        repository_path + "/collaborators/" + urllib.parse.quote(actor, safe="") + "/permission",
        token,
    )
    user = permission.get("user") if isinstance(permission, dict) else None
    if (
        not isinstance(permission, dict)
        or permission.get("permission") != "admin"
        or not isinstance(user, dict)
        or str(user.get("login", "")).casefold() != actor.casefold()
        or user.get("type") != "User"
    ):
        raise GithubReleaseActorError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--actor", required=True)
    args = parser.parse_args()
    try:
        verify_release_actor(
            args.repository,
            args.actor,
            os.environ.get("GH_TOKEN", ""),
        )
    except Exception:
        print("github_release_actor_invalid", file=sys.stderr)
        return 1
    print("github_release_actor_verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
