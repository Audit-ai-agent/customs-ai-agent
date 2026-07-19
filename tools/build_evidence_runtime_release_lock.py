"""Build the production Agent lock from an attested Data readiness document.

This tool never creates generation or evaluation identities. It only verifies
and copies an identity already emitted by a pinned Data provider build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from importlib.resources import files
from pathlib import Path
from typing import Any

from customs_ai_agent.contracts.evidence_runtime_release import (
    EvidenceRuntimeReleaseIdentity,
    canonical_json_bytes,
    parse_evidence_runtime_readiness,
)


_GIT_REVISION = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_EXPECTED_DATA_LOCK_KEYS = {
    "contract",
    "schema_version",
    "provider_release",
    "provider_revision",
    "provider_build_sha256",
    "schema_sha256",
    "source_manifest_sha256",
}


class ReleaseLockBuildError(RuntimeError):
    pass


def _strict_json(path: Path) -> dict[str, Any]:
    def reject_constant(_value: str) -> None:
        raise ValueError

    def reject_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise ValueError
            value[key] = item
        return value

    try:
        payload = json.loads(
            path.read_bytes(),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicates,
        )
    except (OSError, UnicodeError, ValueError, json.JSONDecodeError):
        raise ReleaseLockBuildError("release_lock_input_invalid") from None
    if not isinstance(payload, dict):
        raise ReleaseLockBuildError("release_lock_input_invalid")
    return payload


def _verified_data_lock(payload: dict[str, Any]) -> dict[str, str]:
    if (
        set(payload) != _EXPECTED_DATA_LOCK_KEYS
        or payload.get("contract") != "evidence-retrieval"
        or payload.get("schema_version") != "1.0"
        or payload.get("provider_release") != "evidence-api-v1.2.0"
    ):
        raise ReleaseLockBuildError("expected_data_lock_invalid")
    revision = payload.get("provider_revision")
    if not isinstance(revision, str) or _GIT_REVISION.fullmatch(revision) is None:
        raise ReleaseLockBuildError("expected_data_lock_invalid")
    for field_name in (
        "provider_build_sha256",
        "schema_sha256",
        "source_manifest_sha256",
    ):
        value = payload.get(field_name)
        if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
            raise ReleaseLockBuildError("expected_data_lock_invalid")
    digest_input = {
        field_name: payload[field_name]
        for field_name in (
            "contract",
            "schema_version",
            "provider_release",
            "provider_revision",
            "schema_sha256",
            "source_manifest_sha256",
        )
    }
    if (
        hashlib.sha256(canonical_json_bytes(digest_input)).hexdigest()
        != payload["provider_build_sha256"]
    ):
        raise ReleaseLockBuildError("expected_data_lock_invalid")
    return payload  # type: ignore[return-value]


def build_released_lock(
    *,
    readiness: dict[str, Any],
    expected_data_lock: dict[str, Any],
    readiness_schema_bytes: bytes,
    retrieval_schema_bytes: bytes,
) -> dict[str, Any]:
    packaged = files("customs_ai_agent.contracts").joinpath("evidence")
    try:
        packaged_readiness_schema = packaged.joinpath("runtime-release-v2.schema.json").read_bytes()
        packaged_retrieval_schema = packaged.joinpath("v1.schema.json").read_bytes()
    except (FileNotFoundError, OSError):
        raise ReleaseLockBuildError("packaged_evidence_schema_missing") from None
    if (
        readiness_schema_bytes != packaged_readiness_schema
        or retrieval_schema_bytes != packaged_retrieval_schema
    ):
        raise ReleaseLockBuildError("data_schema_bytes_do_not_match_agent_package")
    expected = _verified_data_lock(expected_data_lock)
    try:
        identity = parse_evidence_runtime_readiness(readiness)
    except ValueError:
        raise ReleaseLockBuildError("data_readiness_invalid") from None
    comparisons = {
        "provider_release": identity.provider_release,
        "provider_revision": identity.provider_revision,
        "provider_build_sha256": identity.provider_build_sha256,
        "schema_sha256": identity.retrieval_schema_sha256,
        "source_manifest_sha256": identity.provider_source_manifest_sha256,
    }
    if any(expected[field] != observed for field, observed in comparisons.items()):
        raise ReleaseLockBuildError("data_readiness_does_not_match_expected_lock")
    readiness_schema_sha256 = hashlib.sha256(readiness_schema_bytes).hexdigest()
    retrieval_schema_sha256 = hashlib.sha256(retrieval_schema_bytes).hexdigest()
    if (
        expected["schema_sha256"] != retrieval_schema_sha256
        or identity.retrieval_schema_sha256 != retrieval_schema_sha256
    ):
        raise ReleaseLockBuildError("data_schema_does_not_match_expected_lock")
    return {
        "contract": "evidence-runtime-release-consumer-lock",
        "schema_version": "1.0",
        "canonical_owner": "customs-data-platform",
        "consumer": "customs-ai-agent",
        "provider_release": identity.provider_release,
        "retrieval_schema_version": identity.retrieval_schema_version,
        "retrieval_schema_sha256": identity.retrieval_schema_sha256,
        "readiness_schema_sha256": readiness_schema_sha256,
        "release_state": "released",
        "identity": identity.to_wire(),
    }


def _atomic_write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            allow_nan=False,
            indent=2,
            sort_keys=False,
        ).encode("utf-8")
        + b"\n"
    )
    descriptor, temporary_name = tempfile.mkstemp(
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--readiness", type=Path, required=True)
    parser.add_argument("--expected-data-lock", type=Path, required=True)
    parser.add_argument("--readiness-schema", type=Path, required=True)
    parser.add_argument("--retrieval-schema", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        lock = build_released_lock(
            readiness=_strict_json(args.readiness),
            expected_data_lock=_strict_json(args.expected_data_lock),
            readiness_schema_bytes=args.readiness_schema.read_bytes(),
            retrieval_schema_bytes=args.retrieval_schema.read_bytes(),
        )
        _atomic_write(args.output, lock)
    except (OSError, ReleaseLockBuildError) as exc:
        code = str(exc) if isinstance(exc, ReleaseLockBuildError) else "release_lock_io_failed"
        print(code)
        return 1
    identity = EvidenceRuntimeReleaseIdentity.from_mapping(lock["identity"])
    print(f"runtime_release_sha256={identity.runtime_release_sha256}")
    print(f"lock_sha256={hashlib.sha256(canonical_json_bytes(lock)).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
