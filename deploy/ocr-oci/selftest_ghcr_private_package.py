"""Offline contract checks for the GHCR privacy verifier."""

from __future__ import annotations

import contextlib
import io
import json
import urllib.error
from unittest.mock import patch

from verify_ghcr_private_package import (
    GhcrPrivacyError,
    _package_identity,
    main,
    verify_private_package,
)


class _Response(io.BytesIO):
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _payload(value: object) -> _Response:
    return _Response(json.dumps(value).encode("utf-8"))


def _organization() -> dict[str, object]:
    return {"type": "Organization"}


def _private_package(**updates: object) -> dict[str, object]:
    value: dict[str, object] = {
        "name": "runtime/sub",
        "owner": {"login": "example", "type": "Organization"},
        "package_type": "container",
        "version_count": 1,
        "visibility": "private",
    }
    value.update(updates)
    return value


def _assert_rejected(
    package_payload: object,
    *,
    allow_missing_repository: bool = False,
    max_attempts: int = 1,
) -> None:
    def opened(request: object, *, timeout: int) -> _Response:
        url = request.full_url  # type: ignore[attr-defined]
        return _payload(_organization() if url.endswith("/users/example") else package_payload)

    with patch("urllib.request.urlopen", opened):
        try:
            verify_private_package(
                "ghcr.io/example/runtime/sub",
                "example/service",
                "token",
                allow_missing_repository=allow_missing_repository,
                max_attempts=max_attempts,
            )
        except GhcrPrivacyError:
            return
    raise AssertionError("invalid GHCR package metadata accepted")


def main_selftest() -> int:
    digest = "a" * 64
    if _package_identity(f"ghcr.io/example/runtime/sub@sha256:{digest}") != (
        "example",
        "runtime/sub",
    ):
        raise AssertionError("digest reference parsing failed")
    for invalid in (
        "docker.io/example/runtime",
        "ghcr.io/example/runtime:latest",
        "ghcr.io/Example/runtime",
        "ghcr.io/example//runtime",
    ):
        try:
            _package_identity(invalid)
        except GhcrPrivacyError:
            continue
        raise AssertionError("invalid GHCR reference accepted")

    requested_urls: list[str] = []

    def private_open(request: object, *, timeout: int) -> _Response:
        if timeout != 15:
            raise AssertionError("timeout not bounded")
        url = request.full_url  # type: ignore[attr-defined]
        requested_urls.append(url)
        return _payload(_organization() if url.endswith("/users/example") else _private_package())

    # The documented package response does not guarantee a repository member.
    with patch("urllib.request.urlopen", private_open):
        verify_private_package(
            "ghcr.io/example/runtime/sub",
            "example/service",
            "token",
            allow_missing_repository=True,
        )
    if not requested_urls[-1].endswith("/orgs/example/packages/container/runtime%2Fsub"):
        raise AssertionError("package API path is not encoded")

    linked_payload = _private_package(repository={"full_name": "example/service"})
    with patch(
        "urllib.request.urlopen",
        lambda request, *, timeout: _payload(
            _organization()
            if request.full_url.endswith("/users/example")  # type: ignore[attr-defined]
            else linked_payload
        ),
    ):
        verify_private_package("ghcr.io/example/runtime/sub", "example/service", "token")

    for invalid_payload in (
        _private_package(visibility="public"),
        _private_package(visibility="internal"),
        _private_package(name="other"),
        _private_package(package_type="npm"),
        _private_package(owner={"login": "other", "type": "Organization"}),
        _private_package(owner={"login": "example", "type": "User"}),
        _private_package(repository={"full_name": "example/other"}),
        _private_package(version_count=-1),
        _private_package(version_count=True),
        [],
    ):
        _assert_rejected(invalid_payload)
    for pending_payload in (
        _private_package(),
        {key: value for key, value in _private_package().items() if key != "version_count"},
        _private_package(version_count=0),
        _private_package(repository=None),
    ):
        _assert_rejected(pending_payload)
    for invalid_source_attested_payload in (
        _private_package(repository=None),
        _private_package(repository={"full_name": "example/other"}),
    ):
        _assert_rejected(
            invalid_source_attested_payload,
            allow_missing_repository=True,
        )

    package_calls = 0

    def eventually_private(request: object, *, timeout: int) -> _Response:
        nonlocal package_calls
        url = request.full_url  # type: ignore[attr-defined]
        if url.endswith("/users/example"):
            return _payload(_organization())
        package_calls += 1
        if package_calls == 1:
            raise urllib.error.HTTPError(url, 403, "forbidden", None, None)
        if package_calls == 2:
            raise urllib.error.HTTPError(url, 404, "not found", None, None)
        if package_calls == 3:
            return _payload(
                _private_package(
                    repository={"full_name": "example/service"},
                    version_count=0,
                )
            )
        if package_calls == 4:
            return _payload(_private_package(repository=None))
        return _payload(_private_package(repository={"full_name": "example/service"}))

    with (
        patch("urllib.request.urlopen", eventually_private),
        patch("time.sleep") as sleep,
    ):
        verify_private_package(
            "ghcr.io/example/runtime/sub",
            "example/service",
            "token",
            max_attempts=5,
            retry_delay_seconds=1,
        )
    if package_calls != 5 or sleep.call_count != 4:
        raise AssertionError("bounded package propagation retry was not exact")

    package_calls = 0

    def always_pending(request: object, *, timeout: int) -> _Response:
        nonlocal package_calls
        url = request.full_url  # type: ignore[attr-defined]
        if url.endswith("/users/example"):
            return _payload(_organization())
        package_calls += 1
        return _payload(_private_package(repository=None))

    with (
        patch("urllib.request.urlopen", always_pending),
        patch("time.sleep") as sleep,
    ):
        try:
            verify_private_package(
                "ghcr.io/example/runtime/sub",
                "example/service",
                "token",
                max_attempts=3,
                retry_delay_seconds=1,
            )
        except GhcrPrivacyError:
            pass
        else:
            raise AssertionError("exhausted propagation retry accepted")
    if package_calls != 3 or sleep.call_count != 2:
        raise AssertionError("exhausted retry count was not bounded")

    for fatal_payload in (
        _private_package(visibility="public"),
        _private_package(repository={"full_name": "example/other"}),
    ):
        package_calls = 0

        def immediately_invalid(request: object, *, timeout: int) -> _Response:
            nonlocal package_calls
            url = request.full_url  # type: ignore[attr-defined]
            if url.endswith("/users/example"):
                return _payload(_organization())
            package_calls += 1
            return _payload(fatal_payload)

        with (
            patch("urllib.request.urlopen", immediately_invalid),
            patch("time.sleep") as sleep,
        ):
            try:
                verify_private_package(
                    "ghcr.io/example/runtime/sub",
                    "example/service",
                    "token",
                    max_attempts=6,
                )
            except GhcrPrivacyError:
                pass
            else:
                raise AssertionError("fatal package state accepted during retry mode")
        if package_calls != 1 or sleep.call_count != 0:
            raise AssertionError("fatal package state was retried")

    for invalid_attempts, invalid_delay in ((0, 2), (7, 2), (1, 0), (1, 11), (True, 2)):
        try:
            verify_private_package(
                "ghcr.io/example/runtime/sub",
                "example/service",
                "token",
                max_attempts=invalid_attempts,
                retry_delay_seconds=invalid_delay,
            )
        except GhcrPrivacyError:
            continue
        raise AssertionError("invalid retry bounds accepted")

    stderr = io.StringIO()
    with (
        patch("sys.argv", ["verify", "--image", "invalid", "--repository", "example/service"]),
        patch.dict("os.environ", {"GH_TOKEN": "secret-token"}, clear=True),
        contextlib.redirect_stderr(stderr),
    ):
        if main() != 1:
            raise AssertionError("CLI failure was accepted")
    if stderr.getvalue() != "ghcr_package_privacy_check_failed\n":
        raise AssertionError("CLI failure leaked details")

    print("ghcr_private_package_selftest_passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main_selftest())
