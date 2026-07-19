from __future__ import annotations

import json
import re
from dataclasses import dataclass
from importlib import resources


BUILD_MANIFEST_CONTRACT = "customs-ai-agent-build-manifest"
BUILD_MANIFEST_SCHEMA_VERSION = "1.0"
SOURCE_SNAPSHOT_ALGORITHM = "git-archive-files-sha256-v1"
_GIT_SHA = re.compile(r"^[a-f0-9]{40}$")
_SHA256 = re.compile(r"^[a-f0-9]{64}$")


class BuildManifestError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class BuildManifest:
    git_sha: str
    source_snapshot_sha256: str


def load_packaged_build_manifest(*, required: bool) -> BuildManifest | None:
    try:
        payload = resources.files("customs_ai_agent").joinpath("build-manifest.json").read_bytes()
    except (FileNotFoundError, ModuleNotFoundError, OSError):
        if required:
            raise BuildManifestError("build_manifest_missing") from None
        return None
    try:
        value = json.loads(payload)
    except (UnicodeError, ValueError, json.JSONDecodeError):
        raise BuildManifestError("build_manifest_invalid") from None
    expected_keys = {
        "contract",
        "distribution",
        "git_sha",
        "package_version",
        "schema_version",
        "source_snapshot_algorithm",
        "source_snapshot_sha256",
    }
    if (
        not isinstance(value, dict)
        or set(value) != expected_keys
        or value.get("contract") != BUILD_MANIFEST_CONTRACT
        or value.get("distribution") != "customs-ai-agent"
        or value.get("package_version") != "0.3.0"
        or value.get("schema_version") != BUILD_MANIFEST_SCHEMA_VERSION
        or value.get("source_snapshot_algorithm") != SOURCE_SNAPSHOT_ALGORITHM
        or not isinstance(value.get("git_sha"), str)
        or _GIT_SHA.fullmatch(value["git_sha"]) is None
        or not isinstance(value.get("source_snapshot_sha256"), str)
        or _SHA256.fullmatch(value["source_snapshot_sha256"]) is None
    ):
        raise BuildManifestError("build_manifest_invalid")
    canonical = (
        json.dumps(
            value,
            ensure_ascii=True,
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if payload != canonical:
        raise BuildManifestError("build_manifest_invalid")
    return BuildManifest(
        git_sha=value["git_sha"],
        source_snapshot_sha256=value["source_snapshot_sha256"],
    )


__all__ = [
    "BuildManifest",
    "BuildManifestError",
    "load_packaged_build_manifest",
]
