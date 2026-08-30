"""Offline contract checks for digest-bound GHCR source attestation."""

from __future__ import annotations

import json

from verify_ghcr_source_attestation import (
    GhcrSourceAttestationError,
    parse_image_config,
    verify_source_attestation,
)

_REPOSITORY = "Audit-ai-agent/customs-ai-agent"
_RUNTIME = "ghcr.io/audit-ai-agent/customs-ai-agent-raster-runtime@sha256:" + "a" * 64
_AGENT = "ghcr.io/audit-ai-agent/customs-ai-agent-raster-ocr@sha256:" + "b" * 64
_REVISION = "c" * 40


def _runtime_config() -> dict[str, object]:
    return {
        "architecture": "amd64",
        "os": "linux",
        "config": {
            "Labels": {
                "io.customs.artifact.kind": "raster-text-runtime-context-v1",
                "io.customs.raster-text.profile-revision": ("synthetic-raster-text-v1:" + "d" * 24),
                "io.customs.runtime.manifest-sha256": "e" * 64,
                "org.opencontainers.image.source": (
                    "https://github.com/audit-ai-agent/customs-ai-agent"
                ),
            }
        },
    }


def _agent_config() -> dict[str, object]:
    return {
        "architecture": "amd64",
        "os": "linux",
        "config": {
            "Labels": {
                "io.customs.artifact.kind": "customs-ai-agent-raster-text-ocr-v1",
                "io.customs.raster-runtime.contract": "raster-text-runtime-oci-v1",
                "io.customs.raster-runtime.image": _RUNTIME,
                "org.opencontainers.image.revision": _REVISION,
                "org.opencontainers.image.source": (
                    "https://github.com/audit-ai-agent/customs-ai-agent"
                ),
            }
        },
    }


def _verify_runtime(config: object, *, image: str = _RUNTIME) -> None:
    verify_source_attestation(
        config,
        image=image,
        kind="runtime",
        repository=_REPOSITORY,
    )


def _assert_runtime_rejected(config: object, *, image: str = _RUNTIME) -> None:
    try:
        _verify_runtime(config, image=image)
    except GhcrSourceAttestationError:
        return
    raise AssertionError("invalid runtime source attestation accepted")


def _assert_agent_rejected(
    config: object,
    *,
    runtime_image: str = _RUNTIME,
    image: str = _AGENT,
    source_revision: str = _REVISION,
) -> None:
    try:
        verify_source_attestation(
            config,
            image=image,
            kind="agent",
            repository=_REPOSITORY,
            runtime_image=runtime_image,
            source_revision=source_revision,
        )
    except GhcrSourceAttestationError:
        return
    raise AssertionError("invalid Agent source attestation accepted")


def main() -> int:
    runtime = _runtime_config()
    _verify_runtime(runtime)
    verify_source_attestation(
        _agent_config(),
        image=_AGENT,
        kind="agent",
        repository=_REPOSITORY,
        runtime_image=_RUNTIME,
        source_revision=_REVISION,
    )

    _assert_runtime_rejected(runtime, image=_RUNTIME.replace("@sha256:", ":latest@sha256:"))
    for path, replacement in (
        (("architecture",), "arm64"),
        (("os",), "windows"),
        (("config", "Labels", "org.opencontainers.image.source"), "https://github.com/other/repo"),
        (("config", "Labels", "org.opencontainers.image.source"), None),
        (("config", "Labels", "io.customs.artifact.kind"), "other"),
        (("config", "Labels", "io.customs.runtime.manifest-sha256"), "bad"),
        (("config", "Labels", "io.customs.raster-text.profile-revision"), "bad"),
    ):
        changed = json.loads(json.dumps(runtime))
        target = changed
        for part in path[:-1]:
            target = target[part]  # type: ignore[index,assignment]
        target[path[-1]] = replacement  # type: ignore[index]
        _assert_runtime_rejected(changed)

    for raw in (b"", b"not-json", b'{"value":NaN}', b" " * (64 * 1024 + 1)):
        try:
            parse_image_config(raw)
        except GhcrSourceAttestationError:
            continue
        raise AssertionError("invalid source attestation JSON accepted")

    wrong_agent = _agent_config()
    labels = wrong_agent["config"]["Labels"]  # type: ignore[index]
    labels["io.customs.raster-runtime.image"] = _RUNTIME[:-1] + "f"  # type: ignore[index]
    _assert_agent_rejected(wrong_agent)

    double_at_runtime = _RUNTIME + "@sha256:" + "f" * 64
    double_at_agent = _agent_config()
    double_at_labels = double_at_agent["config"]["Labels"]  # type: ignore[index]
    double_at_labels["io.customs.raster-runtime.image"] = double_at_runtime  # type: ignore[index]
    _assert_agent_rejected(double_at_agent, runtime_image=double_at_runtime)

    _assert_agent_rejected(
        _agent_config(),
        image="ghcr.io/audit-ai-agent/other-raster-ocr@sha256:" + "b" * 64,
    )
    _assert_agent_rejected(_agent_config(), source_revision="f" * 40)
    wrong_runtime_repository = "ghcr.io/audit-ai-agent/other-raster-runtime@sha256:" + "a" * 64
    wrong_runtime_agent = _agent_config()
    wrong_runtime_labels = wrong_runtime_agent["config"]["Labels"]  # type: ignore[index]
    wrong_runtime_labels["io.customs.raster-runtime.image"] = wrong_runtime_repository  # type: ignore[index]
    _assert_agent_rejected(
        wrong_runtime_agent,
        runtime_image=wrong_runtime_repository,
    )
    for label, replacement in (
        ("org.opencontainers.image.source", "https://github.com/other/repo"),
        ("io.customs.artifact.kind", "other"),
        ("io.customs.raster-runtime.contract", "other"),
    ):
        changed_agent = _agent_config()
        changed_labels = changed_agent["config"]["Labels"]  # type: ignore[index]
        changed_labels[label] = replacement  # type: ignore[index]
        _assert_agent_rejected(changed_agent)

    print("ghcr_source_attestation_selftest_passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
