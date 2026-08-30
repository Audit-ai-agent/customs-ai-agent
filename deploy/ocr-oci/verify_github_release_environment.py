"""Fail closed unless the OCR release environment has an independent main-only gate."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.parse
import urllib.request

_REPOSITORY = re.compile(r"[A-Za-z0-9](?:[A-Za-z0-9-]{0,38})/[A-Za-z0-9._-]{1,100}")
_ENVIRONMENT = "raster-ocr-production"


class GithubReleaseEnvironmentError(RuntimeError):
    pass


def _github_json(path: str, token: str) -> object:
    request = urllib.request.Request(
        "https://api.github.com" + path,
        headers={
            "Accept": "application/vnd.github+json",
            "Authorization": f"Bearer {token}",
            "User-Agent": "customs-raster-ocr-release-environment-verifier",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )
    with urllib.request.urlopen(request, timeout=15) as response:  # noqa: S310
        return json.load(response)


def verify_release_environment(repository: str, environment: str, token: str) -> None:
    if (
        _REPOSITORY.fullmatch(repository) is None
        or environment != _ENVIRONMENT
        or not token
        or len(token) > 4096
        or any(character.isspace() for character in token)
    ):
        raise GithubReleaseEnvironmentError
    owner, name = repository.split("/", 1)
    base = (
        f"/repos/{urllib.parse.quote(owner, safe='')}/"
        f"{urllib.parse.quote(name, safe='')}/environments/"
        f"{urllib.parse.quote(environment, safe='')}"
    )
    metadata = _github_json(base, token)
    if not isinstance(metadata, dict) or metadata.get("name") != environment:
        raise GithubReleaseEnvironmentError
    rules = metadata.get("protection_rules")
    if not isinstance(rules, list):
        raise GithubReleaseEnvironmentError
    reviewer_rules = [
        rule
        for rule in rules
        if isinstance(rule, dict) and rule.get("type") == "required_reviewers"
    ]
    if len(reviewer_rules) != 1:
        raise GithubReleaseEnvironmentError
    reviewer_rule = reviewer_rules[0]
    reviewers = reviewer_rule.get("reviewers")
    if (
        reviewer_rule.get("prevent_self_review") is not True
        or not isinstance(reviewers, list)
        or not reviewers
        or any(
            not isinstance(reviewer, dict) or reviewer.get("type") not in {"User", "Team"}
            for reviewer in reviewers
        )
    ):
        raise GithubReleaseEnvironmentError
    branch_policy = metadata.get("deployment_branch_policy")
    if branch_policy != {"custom_branch_policies": True, "protected_branches": False}:
        raise GithubReleaseEnvironmentError
    policies = _github_json(base + "/deployment-branch-policies?per_page=100", token)
    declared = policies.get("branch_policies") if isinstance(policies, dict) else None
    total_count = policies.get("total_count") if isinstance(policies, dict) else None
    if (
        not isinstance(declared, list)
        or len(declared) != 1
        or not isinstance(declared[0], dict)
        or declared[0].get("name") != "main"
        or declared[0].get("type", "branch") != "branch"
        or total_count != 1
    ):
        raise GithubReleaseEnvironmentError


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", required=True)
    parser.add_argument("--environment", required=True)
    args = parser.parse_args()
    try:
        verify_release_environment(
            args.repository,
            args.environment,
            os.environ.get("GH_TOKEN", ""),
        )
    except Exception:
        print("github_release_environment_invalid", file=sys.stderr)
        return 1
    print("github_release_environment_verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
