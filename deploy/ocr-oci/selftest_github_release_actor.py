"""Dependency-free checks for the private OCR release actor verifier."""

from __future__ import annotations

import verify_github_release_actor as target


def _repository(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "archived": False,
        "default_branch": "main",
        "fork": False,
        "full_name": "Audit-ai-agent/customs-ai-agent",
        "private": True,
        "visibility": "private",
    }
    value.update(overrides)
    return value


def _permission(**overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "permission": "admin",
        "role_name": "admin",
        "user": {"login": "release-admin", "type": "User"},
    }
    value.update(overrides)
    return value


def _verify_with(repository: object, permission: object) -> None:
    calls = iter(
        (
            ("/repos/Audit-ai-agent/customs-ai-agent", repository),
            (
                "/repos/Audit-ai-agent/customs-ai-agent/collaborators/release-admin/permission",
                permission,
            ),
        )
    )
    original = target._github_json

    def fake_github_json(path: str, token: str) -> object:
        expected_path, response = next(calls)
        if path != expected_path or token != "test-token":
            raise AssertionError("release actor API request mismatch")
        return response

    target._github_json = fake_github_json
    try:
        target.verify_release_actor(
            "Audit-ai-agent/customs-ai-agent",
            "release-admin",
            "test-token",
        )
    finally:
        target._github_json = original


def _expect_failure(repository: object, permission: object) -> None:
    try:
        _verify_with(repository, permission)
    except (target.GithubReleaseActorError, StopIteration):
        return
    raise AssertionError("unsafe release actor accepted")


def main() -> int:
    _verify_with(_repository(), _permission())
    _expect_failure(_repository(private=False, visibility="public"), _permission())
    _expect_failure(_repository(default_branch="develop"), _permission())
    _expect_failure(_repository(), _permission(permission="write", role_name="write"))
    _expect_failure(
        _repository(),
        _permission(user={"login": "different-admin", "type": "User"}),
    )
    _expect_failure(
        _repository(),
        _permission(user={"login": "release-admin", "type": "Bot"}),
    )
    print("github_release_actor_selftest_passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
