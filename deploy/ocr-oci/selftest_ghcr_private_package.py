"""Offline contract checks for the GHCR privacy verifier."""

from __future__ import annotations

import io
import json
from unittest.mock import patch

from verify_ghcr_private_package import GhcrPrivacyError, _package_identity, verify_private_package


class _Response(io.BytesIO):
    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        self.close()


def _payload(value: object) -> _Response:
    return _Response(json.dumps(value).encode("utf-8"))


def main() -> int:
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
        return _payload(
            {"type": "Organization"}
            if url.endswith("/users/example")
            else {"visibility": "private"}
        )

    with patch("urllib.request.urlopen", private_open):
        verify_private_package("ghcr.io/example/runtime/sub", "token")
    if not requested_urls[-1].endswith("/orgs/example/packages/container/runtime%2Fsub"):
        raise AssertionError("package API path is not encoded")

    def public_open(request: object, *, timeout: int) -> _Response:
        url = request.full_url  # type: ignore[attr-defined]
        return _payload(
            {"type": "User"} if url.endswith("/users/example") else {"visibility": "public"}
        )

    with patch("urllib.request.urlopen", public_open):
        try:
            verify_private_package("ghcr.io/example/runtime", "token")
        except GhcrPrivacyError:
            pass
        else:
            raise AssertionError("public package accepted")

    print("ghcr_private_package_selftest_passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
