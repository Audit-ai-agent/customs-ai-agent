"""Fail unless the in-container Raster Text OCR binding is actually ready."""

from __future__ import annotations

import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

_DEFAULT_POINTER = Path(
    "/opt/customs/_local/data/document-preview/approvals/raster-text-active.json"
)
_MAX_RESPONSE_BYTES = 4096


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, file_pointer, code, message, headers, new_url):
        return None


def _fail(code: str) -> int:
    print(code, file=sys.stderr)
    return 1


def main() -> int:
    token = os.environ.get("CUSTOMS_AGENT_SERVICE_BEARER_TOKEN", "")
    mode = os.environ.get("CUSTOMS_AGENT_SERVICE_MODE", "local-development")
    pointer_value = os.environ.get(
        "CUSTOMS_AGENT_RASTER_TEXT_PREVIEW_APPROVAL_PATH",
        str(_DEFAULT_POINTER),
    )
    if not token or mode not in {"local-development", "functional-uat"}:
        return _fail("ocr_readiness_configuration_invalid")
    try:
        port = int(os.environ.get("CUSTOMS_AGENT_PORT", "18090"))
        if not 1 <= port <= 65535:
            raise ValueError
        pointer_path = Path(pointer_value)
        if pointer_path != _DEFAULT_POINTER or pointer_path.is_symlink():
            raise ValueError
        pointer_raw = pointer_path.read_bytes()
        if len(pointer_raw) > 2 * 1024 * 1024:
            raise ValueError
        pointer = json.loads(pointer_raw.decode("utf-8"))
        if (
            not isinstance(pointer, dict)
            or pointer.get("schema_version") != "11.0"
            or pointer.get("status") != "activated"
            or not isinstance(pointer.get("raster_text_profile_revision"), str)
        ):
            raise ValueError
        expected_revision = pointer["raster_text_profile_revision"]
    except (OSError, TypeError, UnicodeDecodeError, ValueError, KeyError, json.JSONDecodeError):
        return _fail("ocr_readiness_pointer_invalid")

    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/internal/agent/v3/raster-text-preview-status",
        headers={
            "Authorization": f"Bearer {token}",
            "X-Customs-Service-Mode": mode,
        },
        method="GET",
    )
    try:
        opener = urllib.request.build_opener(_NoRedirect())
        with opener.open(request, timeout=4.0) as response:
            if response.status != 200:
                return _fail("ocr_readiness_http_unavailable")
            raw = response.read(_MAX_RESPONSE_BYTES + 1)
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise ValueError
        status = json.loads(raw.decode("utf-8"))
    except (
        OSError,
        TimeoutError,
        UnicodeDecodeError,
        ValueError,
        json.JSONDecodeError,
        urllib.error.URLError,
    ):
        return _fail("ocr_readiness_http_unavailable")
    if (
        not isinstance(status, dict)
        or set(status) != {"availability", "raster_text_profile_revision", "schema_version"}
        or status.get("schema_version") != "1.0"
        or status.get("availability") != "available"
        or status.get("raster_text_profile_revision") != expected_revision
    ):
        return _fail("ocr_binding_unavailable")
    print("raster_text_ready")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
