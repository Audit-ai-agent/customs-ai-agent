from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import fastapi
from pydantic import SecretStr

from customs_ai_agent.service.app import create_app
from customs_ai_agent.service.settings import ServiceSettings


ROOT = Path(__file__).parents[1]
LOCK_PATH = ROOT / "contracts" / "agent" / "service" / "v1" / "openapi.lock.json"
PACKAGED_LOCK_PATH = (
    ROOT / "src" / "customs_ai_agent" / "contracts" / "service" / "v1" / "openapi.lock.json"
)
_LOCK_TOKEN = "Abcdefghijklmnopqrstuvwxyz0123456789-._~+/=XYZ"  # noqa: S105


class OpenApiDriftError(RuntimeError):
    pass


def canonical_openapi() -> tuple[dict[str, Any], bytes]:
    settings = ServiceSettings(
        bearer_token=SecretStr(_LOCK_TOKEN),
        evidence_base_url="http://127.0.0.1:8010",
        evidence_allowed_origins=frozenset({"http://127.0.0.1:8010"}),
        environment="test",
    )
    app = create_app(
        settings,
        runtime_factory=lambda _settings: None,  # type: ignore[return-value]
        readiness_probe_factory=lambda _settings: lambda: object(),
    )
    schema = app.openapi()
    encoded = json.dumps(
        schema,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return schema, encoded


def _expected_openapi_lock() -> dict[str, Any]:
    schema, encoded = canonical_openapi()
    observed_paths = {
        path: sorted(methods)
        for path, methods in schema.get("paths", {}).items()
        if isinstance(methods, dict)
    }
    return {
        "schema_version": "customs-agent-service-openapi-lock/v1",
        "canonical_json_bytes": len(encoded),
        "canonical_sha256": hashlib.sha256(encoded).hexdigest(),
        "paths": observed_paths,
        "generator": {
            "fastapi": fastapi.__version__,
            "openapi": schema.get("openapi"),
            "service_api_version": schema.get("info", {}).get("version"),
        },
    }


def verify_openapi_lock() -> str:
    try:
        lock_bytes = LOCK_PATH.read_bytes()
        packaged_lock_bytes = PACKAGED_LOCK_PATH.read_bytes()
        lock = json.loads(lock_bytes.decode("utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise OpenApiDriftError("service_openapi_lock_invalid") from None
    if packaged_lock_bytes != lock_bytes:
        raise OpenApiDriftError("service_openapi_packaged_lock_drift")
    expected = _expected_openapi_lock()
    if lock != expected:
        raise OpenApiDriftError("service_openapi_drift")
    return expected["canonical_sha256"]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--write",
        action="store_true",
        help="Regenerate both reviewed provider and packaged OpenAPI lock files.",
    )
    args = parser.parse_args()
    if args.write:
        encoded = (
            json.dumps(
                _expected_openapi_lock(),
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")
        for path in (LOCK_PATH, PACKAGED_LOCK_PATH):
            path.write_bytes(encoded)
    try:
        digest = verify_openapi_lock()
    except OpenApiDriftError as exc:
        print(exc)
        return 1
    print(f"service_openapi_sha256={digest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
