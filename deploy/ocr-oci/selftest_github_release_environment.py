"""Dependency-free checks for the protected OCR release environment verifier."""

from __future__ import annotations

import verify_github_release_environment as target


def _environment(*, prevent_self_review: bool = True) -> dict[str, object]:
    return {
        "deployment_branch_policy": {
            "custom_branch_policies": True,
            "protected_branches": False,
        },
        "name": "raster-ocr-production",
        "protection_rules": [
            {
                "prevent_self_review": prevent_self_review,
                "reviewers": [{"reviewer": {"login": "release-reviewer"}, "type": "User"}],
                "type": "required_reviewers",
            }
        ],
    }


def _policies(name: str = "main") -> dict[str, object]:
    return {"branch_policies": [{"name": name, "type": "branch"}], "total_count": 1}


def _verify_with(environment: object, policies: object) -> None:
    calls = iter((environment, policies))
    original = target._github_json

    def fake_github_json(path: str, token: str) -> object:
        del path, token
        return next(calls)

    target._github_json = fake_github_json
    try:
        target.verify_release_environment(
            "Audit-ai-agent/customs-ai-agent",
            "raster-ocr-production",
            "test-token",
        )
    finally:
        target._github_json = original


def _expect_failure(environment: object, policies: object) -> None:
    try:
        _verify_with(environment, policies)
    except (target.GithubReleaseEnvironmentError, StopIteration):
        return
    raise AssertionError("unsafe release environment accepted")


def main() -> int:
    _verify_with(_environment(), _policies())
    _expect_failure(_environment(prevent_self_review=False), _policies())
    _expect_failure(_environment(), _policies("release/*"))
    _expect_failure({}, _policies())
    print("github_release_environment_selftest_passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
