"""Vendor exact provider-owned Evidence v3 schemas into the Agent package.

This is an explicit release-engineering command, never a runtime sibling-repository
dependency. The caller supplies one immutable provider build context. The command
copies only the three allowlisted JSON schemas and the canonical provider runtime
lock after validating the exact final Data revision and every known digest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


EXPECTED = {
    "rag_retrieval_v3.schema.json": "d20de13f4cc8db14a2cd3213e063916fd046922ae32687ac939dbb4019e485d6",
    "readiness_v3.schema.json": "3488e98b9899ce6cbb375fad4120360a2d4a77d82117ef56f098514547a7d882",
    "requirement_policy_v1.schema.json": (
        "b1e9e5918d03c247d8503af861ee647451b0c2f378c1a9d2cc380f23ca26c226"
    ),
}
EXPECTED_PROVIDER_RUNTIME_LOCK_SHA256 = (
    "fe36dac90a034cb3b506e5a9071ded53865f8d3594dd557c3d9f42c58175ad86"
)
EXPECTED_PROVIDER_IDENTITY = {
    "contract": "evidence-runtime-release-provider-lock",
    "schema_version": "1.0",
    "provider_release": "evidence-api-v1.2.0",
    "provider_revision": "0f6fbe031ddf531b44bc8f338bd2ec1d39e47039",
    "provider_build_sha256": ("a2e17f85f189a964f48a3650e0fd76a5203e718e05427da539eba3beeacf2d1c"),
    "provider_source_manifest_sha256": (
        "dada3f2e5406e95c1c092250e0229af5967428a39856b6535808984f7154037f"
    ),
    "retrieval_schema_sha256": ("531b59ba4e86f78cef20a4fca052996efd8b88c48853ac12113ca76e69cdd69a"),
    "readiness_schema_sha256": ("0a0b50fef627ecbbad954af5878ccb3630ecc51f618132f8cfc674c324467a69"),
}


def sync(source: Path, destination: Path) -> None:
    source = source.resolve()
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for name, expected_sha256 in EXPECTED.items():
        raw = (source / name).read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_sha256:
            raise ValueError(f"provider contract digest mismatch: {name}")
        payload = json.loads(raw.decode("utf-8"))
        if not isinstance(payload, dict) or payload.get("$schema") != (
            "https://json-schema.org/draft/2020-12/schema"
        ):
            raise ValueError(f"provider contract is not Draft 2020-12 JSON Schema: {name}")
        shutil.copyfile(source / name, destination / name)


def sync_provider_runtime_lock(source: Path, destinations: tuple[Path, ...]) -> None:
    source = source.resolve()
    raw = source.read_bytes()
    if hashlib.sha256(raw).hexdigest() != EXPECTED_PROVIDER_RUNTIME_LOCK_SHA256:
        raise ValueError("provider runtime lock digest mismatch")
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, dict) or payload != EXPECTED_PROVIDER_IDENTITY:
        raise ValueError("provider runtime lock identity mismatch")
    for destination in destinations:
        resolved = destination.resolve()
        resolved.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, resolved)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--provider-contract-dir", type=Path, required=True)
    parser.add_argument("--provider-runtime-lock", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--checkout-provider-lock", type=Path, required=True)
    args = parser.parse_args()
    sync(args.provider_contract_dir, args.destination)
    sync_provider_runtime_lock(
        args.provider_runtime_lock,
        (
            args.destination.parent / "provider-runtime-lock.json",
            args.checkout_provider_lock,
        ),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
