"""Static fail-closed checks for the asset-free OCR OCI deployment sources."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


class DeploymentSourceError(RuntimeError):
    pass


def _require(text: str, pattern: str, code: str) -> None:
    if re.search(pattern, text, flags=re.MULTILINE) is None:
        raise DeploymentSourceError(code)


def verify(repository_root: Path) -> None:
    root = repository_root.resolve(strict=True)
    dockerfile = (root / "deploy/ocr-oci/Dockerfile").read_text(encoding="utf-8")
    runtime_dockerfile = (root / "deploy/ocr-oci/runtime-context.Dockerfile").read_text(
        encoding="utf-8"
    )
    workflow = (root / ".github/workflows/ocr-oci-publish.yml").read_text(encoding="utf-8")
    dockerignore = (root / ".dockerignore").read_text(encoding="utf-8")
    schema_path = root / "deploy/ocr-oci/runtime-bundle-manifest.schema.json"
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    if schema.get("$schema") != "https://json-schema.org/draft/2020-12/schema":
        raise DeploymentSourceError("runtime_manifest_schema_invalid")

    first_rule = next(
        (
            line.strip()
            for line in dockerignore.splitlines()
            if line.strip() and not line.startswith("#")
        ),
        None,
    )
    if first_rule != "**":
        raise DeploymentSourceError("docker_context_not_allowlisted")
    for required in (
        "!pyproject.toml",
        "!uv.lock",
        "!src/**",
        "!deploy/ocr-oci/**",
        "!tools/verify_raster_text_oci_runtime.py",
    ):
        if required not in dockerignore.splitlines():
            raise DeploymentSourceError("docker_context_allowlist_incomplete")

    _require(dockerfile, r"^# syntax=docker/dockerfile:1\.10$", "docker_syntax_unpinned")
    _require(
        dockerfile,
        r"^ARG PYTHON_IMAGE=python:3\.11\.[0-9]+-slim-bookworm@sha256:[a-f0-9]{64}$",
        "python_base_not_digest_pinned",
    )
    _require(
        dockerfile,
        r"^ARG UV_IMAGE=ghcr\.io/astral-sh/uv:[^@\s]+@sha256:[a-f0-9]{64}$",
        "uv_base_not_digest_pinned",
    )
    _require(
        dockerfile,
        r"^ARG RASTER_RUNTIME_IMAGE=scratch$",
        "runtime_image_fail_closed_default_missing",
    )
    for literal in (
        "FROM ${RASTER_RUNTIME_IMAGE} AS raster-runtime",
        "COPY --from=raster-runtime --link --chown=0:0",
        "tools/verify_raster_text_oci_runtime.py",
        "verify_agent_runtime_binding.py",
        "CUSTOMS_AGENT_RASTER_TEXT_PREVIEW_AUTO_ACTIVATION=source-attested-synthetic-raster-text-v1",
        "CUSTOMS_AGENT_RASTER_TEXT_PREVIEW_APPROVAL_PATH=/opt/customs/_local/data/document-preview/approvals/raster-text-active.json",
        "USER 10001:10001",
        "HEALTHCHECK",
        'ENTRYPOINT ["/opt/customs/app/.venv/bin/customs-ai-agent-serve"]',
    ):
        if literal not in dockerfile:
            raise DeploymentSourceError("release_dockerfile_boundary_incomplete")
    if re.search(r"^\s*(?:ADD\s+https?://|COPY\s+\.\s)", dockerfile, flags=re.MULTILINE):
        raise DeploymentSourceError("unbounded_docker_copy_or_remote_add")
    if "CUSTOMS_AGENT_RASTER_TEXT_PREVIEW_APPROVAL_SHA256" in dockerfile:
        raise DeploymentSourceError("auto_activation_sha_environment_forbidden")

    if (
        "FROM scratch" not in runtime_dockerfile
        or "COPY --chown=0:0 document-preview/" not in runtime_dockerfile
    ):
        raise DeploymentSourceError("runtime_context_dockerfile_invalid")
    if re.search(r"^\s*(?:RUN|ADD)\s", runtime_dockerfile, flags=re.MULTILINE):
        raise DeploymentSourceError("runtime_context_dockerfile_not_data_only")

    for literal in (
        "workflow_dispatch:",
        "tags:",
        '"ocr-oci-v*"',
        "packages: write",
        "attestations: write",
        "provenance: mode=max",
        "sbom: true",
        "RASTER_RUNTIME_IMAGE=${{ steps.release.outputs.runtime_image }}",
        "SOURCE_REVISION=${{ github.sha }}",
        "@sha256:[a-f0-9]{64}",
    ):
        if literal not in workflow:
            raise DeploymentSourceError("publish_workflow_boundary_incomplete")
    if re.search(r"^\s+pull_request:\s*$", workflow, flags=re.MULTILINE):
        raise DeploymentSourceError("publish_on_pull_request_forbidden")
    if re.search(r"^\s+branches:\s*", workflow, flags=re.MULTILINE):
        raise DeploymentSourceError("publish_on_branch_push_forbidden")

    for relative in (
        "deploy/ocr-oci/create_app_license_manifest.py",
        "deploy/ocr-oci/ocr_readiness.py",
        "deploy/ocr-oci/selftest_ocr_readiness.py",
        "deploy/ocr-oci/selftest_runtime_contract.py",
        "deploy/ocr-oci/verify_agent_runtime_binding.py",
        "deploy/ocr-oci/verify_deployment_sources.py",
        "tools/export_raster_text_oci_runtime.py",
        "tools/verify_raster_text_oci_runtime.py",
    ):
        source = (root / relative).read_text(encoding="utf-8")
        compile(source, relative, "exec")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository-root", default=Path.cwd(), type=Path)
    args = parser.parse_args()
    try:
        verify(args.repository_root)
    except (DeploymentSourceError, OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        code = str(exc) if isinstance(exc, DeploymentSourceError) else "deployment_source_invalid"
        print(code, file=sys.stderr)
        return 1
    print("ocr_oci_deployment_sources_valid")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
